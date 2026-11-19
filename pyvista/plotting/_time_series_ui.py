"""Optional notebook controls for discrete time series."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
import uuid

import numpy as np
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3
from trame_pyvista.jupyter import launch_server
from trame_pyvista.ui.vuetify3 import Viewer
from trame_server import Server

import pyvista as pv

if TYPE_CHECKING:
    import asyncio

    from pyvista.plotting.time_series import TimeSeriesPlayer


class _TimeSeriesView:
    """Own notebook controls and their server without replacing the player's actor.

    Parameters
    ----------
    player : TimeSeriesPlayer
        Player supplying frames and the plotter.
    cycle_seconds : float
        Duration of a playback cycle in seconds.
    title : str
        Title displayed above the view.

    """

    def __init__(self, player: TimeSeriesPlayer, *, cycle_seconds: float, title: str) -> None:
        """Build one isolated set of controls around the current plotter."""
        cycle_seconds = float(cycle_seconds)
        if not np.isfinite(cycle_seconds) or cycle_seconds <= 0:
            msg = 'Cycle duration must be finite and positive.'
            raise ValueError(msg)
        self.player = player
        self.pl = player._active()
        from pyvista.jupyter import _resolve_backend  # noqa: PLC0415

        backend = self.pl._theme.jupyter_backend
        if backend is None:
            backend = _resolve_backend()
        if backend not in ('trame', 'server'):
            msg = (
                f'Time series playback requires the "trame" or "server" Jupyter backend; '
                f'got {backend!r}.'
            )
            raise ValueError(msg)
        self.cycle_seconds = cycle_seconds
        self.server = Server(f'time-series-{uuid.uuid4().hex}', client_type='vue3')
        self.state = self.server.state
        self.state.frame = player.index
        self.state.frame_count = len(player.times)
        self.state.time = player.time
        self.state.playing = False
        self.state.opacity = 1.0
        self._updating = False
        self._closed = False
        self._detached = False
        self._on_frame = player._on_frame
        self._render = player._render
        self.viewer = Viewer(
            self.pl, server=self.server, suppress_rendering=self.pl.suppress_rendering
        )
        self._alpha = (
            player.actor.mapper.lookup_table.values[:, 3].copy()
            if isinstance(player.actor, pv.Volume)
            else None
        )

        self.state.change('frame')(self._seek)
        self.state.change('playing')(self._set_playing)
        self.state.change('opacity')(self._set_opacity)
        trame_config = self.pl._theme.trame
        layout_options: dict[str, Any] = {}
        if trame_config.jupyter_extension_enabled:
            layout_options['iframe_builder'] = 'jupyter-extension'
        elif trame_config.server_proxy_enabled:
            layout_options['iframe_builder'] = 'serverproxy'
            layout_options['base_url'] = trame_config.server_proxy_prefix.rstrip('/')
        with SinglePageLayout(self.server, **layout_options) as self.layout:
            self.layout.title.set_text(title)
            with self.layout.toolbar:
                v3.VBtn(
                    "{{ playing ? 'Pause' : 'Play' }}",
                    prepend_icon=("playing ? 'mdi-pause' : 'mdi-play'",),
                    click='playing = !playing',
                    variant='tonal',
                    classes='mr-4',
                )
                v3.VSlider(
                    v_model=('frame', player.index),
                    min=0,
                    max=len(player.times) - 1,
                    step=1,
                    label='Frame',
                    thumb_label=True,
                    hide_details=True,
                    style='max-width: 280px',
                )
                v3.VChip('{{ frame + 1 }} / {{ frame_count }}', classes='mx-4')
                v3.VChip('Time: {{ time }}', classes='mr-4')
                if self._alpha is not None:
                    v3.VSlider(
                        v_model=('opacity', 1.0),
                        min=0.1,
                        max=3.0,
                        step=0.1,
                        label='Opacity',
                        thumb_label=True,
                        hide_details=True,
                        style='max-width: 200px',
                    )
            with self.layout.content:
                self.viewer.ui(
                    mode='server', interactive_ratio=1, interactive_quality=80, still_quality=95
                )
            self.layout.footer.hide()
        player._render = False
        player._on_frame = self._publish_frame

    def _active(self) -> bool:
        """Ignore controls after either owner has closed."""
        return not self._detached and self.player._pl is not None

    def _sync(self) -> None:
        """Publish timeline state without feeding it back into the player."""
        if not self._active():
            return
        self._updating = True
        try:
            with self.state:
                self.state.frame = self.player.index
                self.state.time = self.player.time
                self.state.playing = self.player.playing
        finally:
            self._updating = False

    def _publish_frame(self, player: TimeSeriesPlayer) -> None:
        """Render and publish a committed frame, preserving the caller's callback."""
        self._sync()
        self.pl.reset_camera_clipping_range()
        self.pl.render()
        self.viewer.update()
        if self._on_frame is not None:
            self._on_frame(player)

    def _seek(self, *, frame: int, **_kwargs: Any) -> None:
        """Seek from the slider without resetting the camera."""
        if self._active() and not self._updating:
            try:
                self.player.seek_index(int(frame))
            finally:
                self._sync()

    def _set_playing(self, *, playing: bool, **_kwargs: Any) -> None:
        """Schedule or pause playback using the application's event loop."""
        if self._active() and not self._updating:
            try:
                if playing:
                    task = self.player.start(self.cycle_seconds)
                    task.add_done_callback(self._animation_done)
                else:
                    self.player.pause()
            finally:
                self._sync()

    def _animation_done(self, task: asyncio.Task[None]) -> None:
        """Reset the controls after playback stops and report background failures."""
        self._sync()
        if not task.cancelled() and (error := task.exception()) is not None:
            task.get_loop().call_exception_handler(
                {'message': 'Time series playback failed.', 'exception': error, 'task': task}
            )

    def _set_opacity(self, *, opacity: float, **_kwargs: Any) -> None:
        """Scale the original volume opacity curve without accumulating changes."""
        if self._active() and self._alpha is not None:
            alpha = np.clip(self._alpha.astype(float) * opacity, 0, 255).astype(np.uint8)
            self.player.actor.mapper.lookup_table.apply_opacity(alpha)
            self.pl.render()
            self.viewer.update()

    async def start(self, **kwargs: Any) -> None:
        """Launch the local notebook server and wait for its layout.

        Parameters
        ----------
        **kwargs : dict
            Options passed to the Trame notebook server launcher.

        """
        config = self.pl._theme.trame
        kwargs.setdefault('port', config.jupyter_server_port)
        kwargs.setdefault(
            'wslink_backend', 'jupyter' if config.jupyter_extension_enabled else 'aiohttp'
        )
        launch_server(self.server, **kwargs)
        await self.server.ready

    async def stop(self) -> None:
        """Detach the view, restore caller callbacks, and stop the server."""
        if self._closed:
            return
        self._detached = True
        try:
            if self.player._pl is not None:
                self.player.pause()
        finally:
            if self.player._pl is not None:
                self.player._on_frame = self._on_frame
                self.player._render = self._render
            await self.server.stop()
            self._closed = True
