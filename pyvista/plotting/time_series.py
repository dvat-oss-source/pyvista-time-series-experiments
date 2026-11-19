"""Display discrete datasets on an externally driven timeline."""

from __future__ import annotations

import asyncio
import bisect
import math
import operator
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

import pyvista as pv

__all__ = ['TimeReaderSource', 'TimeSeriesPlayer']

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    import os

    from trame_client.ui.core import AbstractLayout

    from pyvista.core.dataset import DataSet
    from pyvista.core.utilities.reader import TimeReader
    from pyvista.plotting.actor import Actor
    from pyvista.plotting.plotter import Plotter
    from pyvista.plotting.volume import Volume

    from ._time_series_ui import _TimeSeriesView


class TimeSeriesPlayer:
    """Display independently loaded samples through one actor.

    For a sequence of files, :meth:`from_files` provides a convenient starting
    point. This player displays discrete frames through one actor; it does not
    interpolate between samples. The loader returns one dataset by index and
    must keep its buffers valid and unchanged while that sample is displayed.
    Only the current sample is retained by the player. The caller owns the
    plotter. In a notebook, await
    :meth:`show` for playback controls. In another asynchronous application, use
    :meth:`start` on its event loop. For an external timer, call :meth:`play`
    and :meth:`tick`. Use the plotter's thread for all operations.

    Parameters
    ----------
    pl : pyvista.Plotter
        Plotter in which to create one actor. The player does not close it.
    times : sequence of float
        Nonempty, finite, strictly increasing source timestamps.
    load_frame : callable
        Function accepting an integer index and returning one PyVista dataset.
    cycle_end : float
        End of the playback cycle in source time, later than the last sample.
        The gap after the last sample is its display dwell.
    scalars : str
        Name of a one-component real numeric array on every frame.
    clim : sequence of float
        Fixed, increasing color limits for every frame.
    preference : str, default: 'point'
        Association of ``scalars``: ``'point'`` or ``'cell'``.
    clock : callable, optional
        Monotonic clock returning seconds. Defaults to :func:`time.monotonic`.
    actor : pyvista.Actor or pyvista.Volume, optional
        Existing actor to drive instead of creating a mesh actor. Rendering
        properties are preserved. The player replaces its input and removes it
        on close. Closing also disconnects a volume's mapper to release its
        internal rendering inputs. Configure volume appearance with
        ``pl.add_volume`` first.
    render : bool, default: True
        Render after each frame change. Disable when an external view performs
        rendering and publishes the changed frame itself.
    on_frame : callable, optional
        Called with this player after each changed frame has been committed and
        rendered. It is not called for the initial frame or a same-index seek.
        Callback errors leave the changed frame and playback position committed.
    volume : bool, default: False
        Create a volume actor instead of a surface mesh actor.
    **kwargs : dict, optional
        Appearance options passed to :meth:`pyvista.Plotter.add_mesh` or
        :meth:`pyvista.Plotter.add_volume` when creating an actor.

    Notes
    -----
    Seeking loads only the requested frame. The player makes a shallow copy to
    isolate active-scalar metadata, sharing the loader's geometry and array
    buffers. During replacement, the old and new frames can coexist. VTK, a
    reader, renderer, or caller may retain additional buffers independently.
    A failed load or validation leaves the displayed frame and playback position
    unchanged. Rendering failures after mapper replacement cannot be rolled back.
    Callback errors leave the displayed frame and playback position committed.

    Examples
    --------
    Display two samples and select the second one.

    >>> import pyvista as pv
    >>> pl = pv.Plotter()
    >>> frames = [pv.Sphere(radius=radius) for radius in (1, 2)]
    >>> for i, frame in enumerate(frames):
    ...     frame['value'] = [i] * frame.n_points
    >>> player = pv.TimeSeriesPlayer(
    ...     pl,
    ...     [0, 1],
    ...     frames.__getitem__,
    ...     cycle_end=2,
    ...     scalars='value',
    ...     clim=(0, 1),
    ... )
    >>> player.seek_index(1)
    >>> player.close()
    >>> pl.close()

    .. versionadded:: 0.50

    """

    def __init__(
        self,
        pl: Plotter,
        times: Sequence[float],
        load_frame: Callable[[int], DataSet],
        *,
        cycle_end: float,
        scalars: str,
        clim: Sequence[float],
        preference: Literal['point', 'cell'] = 'point',
        clock: Callable[[], float] | None = None,
        actor: Actor | Volume | None = None,
        render: bool = True,
        on_frame: Callable[[TimeSeriesPlayer], None] | None = None,
        volume: bool = False,
        **kwargs: Any,
    ) -> None:
        """Validate the sequence and display its first sample."""
        if actor is not None and (volume or kwargs):
            msg = 'Configure appearance on the existing actor before attaching it.'
            raise ValueError(msg)
        self.times = tuple(float(value) for value in times)
        if not self.times or not all(math.isfinite(value) for value in self.times):
            msg = 'Times must be nonempty and finite.'
            raise ValueError(msg)
        if any(right <= left for left, right in zip(self.times, self.times[1:], strict=False)):
            msg = 'Times must be strictly increasing.'
            raise ValueError(msg)
        self.cycle_end = float(cycle_end)
        if (
            not math.isfinite(self.cycle_end)
            or self.cycle_end <= self.times[-1]
            or not math.isfinite(self.cycle_end - self.times[0])
        ):
            msg = 'Cycle end must be finite and later than the final sample.'
            raise ValueError(msg)
        if preference not in ('point', 'cell'):
            msg = "Preference must be 'point' or 'cell'."
            raise ValueError(msg)
        self.clim = tuple(float(value) for value in clim)
        if (
            len(self.clim) != 2
            or not all(math.isfinite(value) for value in self.clim)
            or self.clim[0] >= self.clim[1]
        ):
            msg = 'Color limits must be two finite increasing values.'
            raise ValueError(msg)
        self._pl: Plotter | None = pl
        self._load_frame: Callable[[int], DataSet] | None = load_frame
        self._scalars = scalars
        self._preference = preference
        self._clock: Callable[[], float] | None = time.monotonic if clock is None else clock
        self._index = 0
        self._position = self.times[0]
        self._started_at: float | None = None
        self._cycle_seconds: float | None = None
        self._loop = True
        self._playing = False
        self._render = render
        self._on_frame = on_frame
        self._async_task: asyncio.Task[None] | None = None
        self._closing_tasks: set[asyncio.Task[None]] = set()
        self._view: _TimeSeriesView | None = None
        self._view_starting = False
        self._view_started: asyncio.Event | None = None
        self._view_closing = False
        frame = self._load(0)
        self._frame: DataSet | None = frame
        if actor is None:
            options = (
                {}
                if volume
                else {
                    'static': False,
                    'rgb': False,
                    'smooth_shading': False,
                    'silhouette': False,
                    'show_vertices': False,
                }
            )
            options.update(show_scalar_bar=False)
            options.update(kwargs)
            add_actor = pl.add_volume if volume else pl.add_mesh
            actor = add_actor(
                frame,
                scalars=scalars,
                preference=preference,
                clim=self.clim,
                **options,
            )
        else:
            if actor.mapper is None:
                msg = 'The playback actor has no mapper.'
                raise ValueError(msg)
            previous = actor.mapper.dataset
            actor.mapper.dataset = frame
            if pl.mesh is previous:
                pl.mesh = frame
            actor.mapper.scalar_range = self.clim
            if isinstance(actor.mapper, pv.DataSetMapper):
                actor.mapper.set_active_scalars(scalars, preference=preference)
            elif isinstance(actor, pv.Volume):
                actor.mapper.scalar_map_mode = preference
        self._actor: Actor | Volume | None = actor

    @classmethod
    def from_files(
        cls,
        pl: Plotter,
        files: Sequence[str | os.PathLike[str]],
        *,
        scalars: str,
        clim: Sequence[float],
        volume: bool = False,
        **kwargs: Any,
    ) -> TimeSeriesPlayer:
        """Display equally spaced frame files, reading only the selected frame.

        Parameters
        ----------
        pl : pyvista.Plotter
            Plotter in which to display the sequence.
        files : sequence of str or path-like
            Frame files in playback order. Use ``.pv`` for native caches.
        scalars : str
            Scalar array present on every frame.
        clim : sequence of float
            Fixed color limits shared by every frame.
        volume : bool, default: False
            Display volumes instead of surface meshes.
        **kwargs : dict, optional
            Player options such as ``preference``, ``render``, and ``on_frame``.
            Remaining options configure the actor through
            :meth:`pyvista.Plotter.add_volume` or :meth:`pyvista.Plotter.add_mesh`.

        Returns
        -------
        pyvista.TimeSeriesPlayer
            Player with timestamps ``0`` through ``len(files) - 1`` and a
            cycle ending at ``len(files)``. For physical timestamps, use the
            main constructor.

        Examples
        --------
        Save two sample files and select the second frame.

        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> import pyvista as pv
        >>> with TemporaryDirectory() as directory:
        ...     paths = []
        ...     for i in range(2):
        ...         mesh = pv.Sphere(radius=i + 1)
        ...         mesh['value'] = [i] * mesh.n_points
        ...         path = Path(directory) / f'frame{i}.vtp'
        ...         mesh.save(path)
        ...         paths.append(path)
        ...     pl = pv.Plotter()
        ...     player = pv.TimeSeriesPlayer.from_files(
        ...         pl, paths, scalars='value', clim=(0, 1)
        ...     )
        ...     player.seek_index(1)
        ...     player.close()
        ...     pl.close()

        """
        paths = tuple(files)
        if not paths:
            msg = 'Frame files must be nonempty.'
            raise ValueError(msg)
        return cls(
            pl,
            range(len(paths)),
            lambda index: pv.read(paths[index]),
            cycle_end=len(paths),
            scalars=scalars,
            clim=clim,
            volume=volume,
            **kwargs,
        )

    async def show(
        self,
        cycle_seconds: float = 10.0,
        *,
        title: str = 'Time series',
        **kwargs: Any,
    ) -> AbstractLayout:
        """Show a notebook view with playback, frame, and volume opacity controls.

        Await this method in a notebook; use :meth:`start` for playback in other
        asynchronous applications. Requires the ``jupyter`` extra. Call :meth:`aclose`
        before replacing the view or restarting its notebook cell. Repeated
        calls return the existing view. The configured notebook backend must be
        ``'trame'`` or ``'server'``. Rendering stays on the application's event
        loop.

        Parameters
        ----------
        cycle_seconds : float, default: 10.0
            Wall-clock duration of one looping playback cycle, in seconds.
        title : str, default: 'Time series'
            Title displayed above the view.
        **kwargs : dict, optional
            Server launch options ``host``, ``port``, or ``wslink_backend``,
            passed to ``trame_pyvista.jupyter.launch_server``.

        Returns
        -------
        trame_client.ui.core.AbstractLayout
            Ready-to-display notebook layout. Playback starts with its Play button.

        """
        from ._time_series_ui import _TimeSeriesView  # noqa: PLC0415

        self._active()
        if self._view_starting:
            msg = 'The notebook view is still starting.'
            raise RuntimeError(msg)
        if self._view is None:
            view = _TimeSeriesView(self, cycle_seconds=cycle_seconds, title=title)
            self._view = view
            self._view_starting = True
            started = self._view_started = asyncio.Event()
            try:
                await view.start(**kwargs)
            except BaseException as launch_error:
                try:
                    await view.stop()
                except BaseException as cleanup_error:
                    raise launch_error from cleanup_error
                if self._view is view:
                    self._view = None
                raise
            finally:
                self._view_starting = False
                started.set()
            if self._view is not view or self._pl is None:
                if not self._view_closing:
                    await view.stop()
                msg = 'The time series player was closed while the notebook view was starting.'
                raise RuntimeError(msg)
        return self._view.layout

    async def aclose(self) -> None:
        """Stop playback and the notebook server without closing the plotter.

        This method is idempotent. Unlike :meth:`close`, it waits for the
        animation task and notebook server to stop. The caller owns the plotter.

        """
        view = self._view
        self.close()
        self._view_closing = True
        try:
            tasks = self._closing_tasks - {asyncio.current_task()}
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if self._view_starting and self._view_started is not None:
                await self._view_started.wait()
            if view is not None:
                await view.stop()
            if self._view is view:
                self._view = None
        finally:
            self._view_closing = False

    @property
    def actor(self) -> Actor | Volume:
        """Return the actor displaying the current sample."""
        self._active()
        return cast('Actor | Volume', self._actor)

    @property
    def index(self) -> int:
        """Return the index of the displayed sample."""
        self._active()
        return self._index

    @property
    def time(self) -> float:
        """Return the source timestamp of the displayed sample."""
        return self.times[self.index]

    @property
    def playing(self) -> bool:
        """Return whether playback is running."""
        self._active()
        return self._playing

    def _active(self) -> Plotter:
        """Return the open plotter or reject use after either owner closes."""
        pl = self._pl
        if pl is None:
            msg = 'The time series player is closed.'
            raise RuntimeError(msg)
        if getattr(pl, '_closed', False):
            self.close()
            msg = 'The player plotter is closed.'
            raise RuntimeError(msg)
        return pl

    def _load(self, index: int) -> DataSet:
        """Validate a replacement before touching the mapper."""
        loaded: object = cast('Callable[[int], DataSet]', self._load_frame)(index)
        if not isinstance(loaded, pv.DataSet) or isinstance(loaded, pv.PointSet):
            msg = 'The frame loader must return one plottable PyVista dataset.'
            raise TypeError(msg)
        data = loaded.point_data if self._preference == 'point' else loaded.cell_data
        if self._scalars not in data:
            msg = f'Frame {index} has no {self._preference} array {self._scalars!r}.'
            raise ValueError(msg)
        values = data[self._scalars]
        if loaded.n_points == 0 or values.size == 0:
            msg = f'Frame {index} is empty.'
            raise ValueError(msg)
        if values.ndim != 1 or values.dtype.kind not in 'iuf':
            msg = f'Frame {index} requires a one-component real numeric scalar array.'
            raise ValueError(msg)
        frame = loaded.copy(deep=False)
        frame.set_active_scalars(self._scalars, preference=self._preference)
        return frame

    def _now(self) -> float:
        """Read a finite value from the supplied monotonic clock."""
        now = float(cast('Callable[[], float]', self._clock)())
        if not math.isfinite(now):
            msg = 'The playback clock must return a finite time.'
            raise ValueError(msg)
        return now

    def _select(self, index: int) -> bool:
        """Commit one validated sample and render it, returning whether it changed."""
        pl = self._active()
        if index == self._index:
            return False
        frame = self._load(index)
        old_frame = self._frame
        actor = self.actor
        mapper = actor.mapper
        if mapper is None:
            msg = 'The playback actor has no mapper.'
            raise RuntimeError(msg)
        mapper.dataset = frame
        mapper.scalar_range = self.clim
        if pl.mesh is old_frame:
            pl.mesh = frame
        self._frame = frame
        self._index = index
        if self._render:
            pl.render()
        return True

    def _notify_frame(self) -> None:
        """Notify the caller after the frame and timeline state are committed."""
        if self._on_frame is not None:
            self._on_frame(self)

    def seek_index(self, index: int) -> None:
        """Display an exact sample, rejecting invalid or out-of-range indices.

        Parameters
        ----------
        index : int
            Index of the sample to display.

        """
        self._active()
        if isinstance(index, bool):
            msg = 'Frame index must be an integer, not a boolean.'
            raise TypeError(msg)
        index = operator.index(index)
        if not 0 <= index < len(self.times):
            msg = 'Frame index is outside the available samples.'
            raise IndexError(msg)
        if index != self._index:
            now = self._now() if self._playing else None
            changed = self._select(index)
            self._position = self.times[index]
            if self._playing:
                self._started_at = now
            if changed:
                self._notify_frame()

    def seek_time(self, value: float) -> None:
        """Display the nearest sample, clamping endpoints and breaking ties earlier.

        Parameters
        ----------
        value : float
            Source timestamp to seek.

        """
        value = float(value)
        if not math.isfinite(value):
            msg = 'Selected time must be finite.'
            raise ValueError(msg)
        right = bisect.bisect_left(self.times, value)
        if right == 0:
            index = 0
        elif right == len(self.times) or value <= (
            self.times[right - 1] + (self.times[right] - self.times[right - 1]) / 2
        ):
            index = right - 1
        else:
            index = right
        self.seek_index(index)

    def step(self, count: int = 1) -> None:
        """Move by sample count, wrapping within the available samples.

        Parameters
        ----------
        count : int, default: 1
            Number of samples to move. Negative values step backward.

        """
        if isinstance(count, bool):
            msg = 'Step count must be an integer, not a boolean.'
            raise TypeError(msg)
        count = operator.index(count)
        self.seek_index((self.index + count) % len(self.times))

    def play(self, cycle_seconds: float, *, loop: bool | None = None) -> None:
        """Start or resume playback; the caller remains responsible for ticks.

        Parameters
        ----------
        cycle_seconds : float
            Wall-clock duration of one source cycle, in seconds.
        loop : bool, optional
            Restart from the first sample after the final dwell. If omitted,
            retain the previous mode; the initial mode is looping.

        """
        self._active()
        if self._playing:
            return
        cycle_seconds = float(cycle_seconds)
        if not math.isfinite(cycle_seconds) or cycle_seconds <= 0:
            msg = 'Cycle duration must be finite and positive.'
            raise ValueError(msg)
        now = self._now()
        changed = False
        if self._position == self.cycle_end:
            changed = self._select(0)
            self._position = self.times[0]
        self._cycle_seconds = cycle_seconds
        if loop is not None:
            self._loop = loop
        self._started_at = now
        self._playing = True
        if changed:
            self._notify_frame()

    def pause(self) -> None:
        """Display the sample due now, then preserve fractional cycle position."""
        self._active()
        task = self._async_task
        try:
            current_task = asyncio.current_task(loop=asyncio.get_running_loop())
        except RuntimeError:
            current_task = None
        if task is not None and task is not current_task:
            self._async_task = None
            if not task.done():
                self._track_closing_task(task)
                task.cancel()
        if self._playing:
            try:
                self.tick()
            finally:
                self._playing = False
                self._started_at = None

    def tick(self) -> None:
        """Advance to the sample due at the injected clock's current time."""
        self._active()
        if not self._playing:
            return
        now = self._now()
        started_at = cast('float', self._started_at)
        elapsed = max(0.0, now - started_at)
        if not math.isfinite(elapsed):
            msg = 'The playback clock interval must be finite.'
            raise ValueError(msg)
        span = self.cycle_end - self.times[0]
        offset = (self._position - self.times[0]) / span
        cycle_seconds = cast('float', self._cycle_seconds)
        if self._loop:
            fraction = (offset + (elapsed % cycle_seconds) / cycle_seconds) % 1
            position = self.times[0] + fraction * span
        else:
            position = (
                self.cycle_end
                if elapsed >= (1 - offset) * cycle_seconds
                else self.times[0] + (offset + elapsed / cycle_seconds) * span
            )
        index = bisect.bisect_right(self.times, position) - 1
        changed = self._select(index)
        self._position = position
        self._started_at = max(now, started_at)
        if not self._loop and position >= self.cycle_end:
            self._playing = False
            self._started_at = None
        if changed:
            self._notify_frame()

    async def animate(
        self,
        cycle_seconds: float,
        *,
        interval: float = 1 / 30,
        loop: bool | None = None,
    ) -> None:
        """Advance playback on the application's event loop.

        Await this method from the application that owns the plotter. It
        creates no background task and must run on the plotter's thread. Only
        one animation task may run for a player at a time.

        Parameters
        ----------
        cycle_seconds : float
            Wall-clock duration of one source cycle, in seconds.
        interval : float, default: 1 / 30
            Delay between playback ticks, in seconds.
        loop : bool, optional
            Restart from the first sample after the final dwell. If omitted,
            retain the previous playback mode.

        Notes
        -----
        Cancellation pauses playback and propagates to the caller. A
        non-looping animation returns after its final dwell. Closing the player
        cancels its animation task.

        """
        interval = self._validate_interval(interval)
        self._active()
        current_task = asyncio.current_task()
        if current_task is None:
            msg = 'Animation must run inside an asyncio task.'
            raise RuntimeError(msg)
        if self._async_task is not None and self._async_task is not current_task:
            msg = 'An animation coroutine is already running for this player.'
            raise RuntimeError(msg)
        self._async_task = current_task
        try:
            self.play(cycle_seconds, loop=loop)
        except BaseException:
            self._finish_animation(current_task)
            raise
        await self._drive_animation(interval)

    async def _drive_animation(self, interval: float) -> None:
        """Tick until playback stops, then release this task's ownership."""
        current_task = asyncio.current_task()
        if current_task is None:
            msg = 'Animation must run inside an asyncio task.'
            raise RuntimeError(msg)
        try:
            while self._pl is not None and self._playing:
                await asyncio.sleep(interval)
                if self._pl is not None and self._playing:
                    self.tick()
        finally:
            self._finish_animation(current_task)

    def _finish_animation(self, task: asyncio.Task[None]) -> None:
        """Stop playback and release the task only while it still owns the player."""
        if self._async_task is task:
            self._async_task = None
            if self._pl is not None:
                self._playing = False
                self._started_at = None

    @staticmethod
    def _validate_interval(interval: float) -> float:
        """Return a finite positive animation polling interval."""
        interval = float(interval)
        if not math.isfinite(interval) or interval <= 0:
            msg = 'Animation interval must be finite and positive.'
            raise ValueError(msg)
        return interval

    def start(
        self,
        cycle_seconds: float,
        *,
        interval: float = 1 / 30,
        loop: bool | None = None,
    ) -> asyncio.Task[None]:
        """Start playback as one task on the application's running event loop.

        Repeated calls while an animation task is live return that task without
        restarting playback. Await the returned task to observe playback errors.
        Calling :meth:`pause` or :meth:`close` cancels the task. If a frame
        callback pauses or closes the player during startup, the returned task
        is canceled before its first tick.

        Parameters
        ----------
        cycle_seconds : float
            Wall-clock duration of one source cycle, in seconds.
        interval : float, default: 1 / 30
            Delay between playback ticks, in seconds.
        loop : bool, optional
            Restart from the first sample after the final dwell. If omitted,
            retain the previous playback mode.

        Returns
        -------
        asyncio.Task
            The application's animation task.

        """
        interval = self._validate_interval(interval)
        self._active()
        event_loop = asyncio.get_running_loop()
        task = self._async_task
        if task is not None and not task.done():
            return task
        self._async_task = None
        coroutine = self._drive_animation(interval)
        try:
            task = event_loop.create_task(coroutine)
        except BaseException:
            coroutine.close()
            raise
        self._async_task = task

        def release_unstarted_task(done_task: asyncio.Task[None]) -> None:
            # A task cancelled before its first step cannot run animate's finally block.
            self._finish_animation(done_task)

        task.add_done_callback(release_unstarted_task)
        try:
            self.play(cycle_seconds, loop=loop)
        except BaseException:
            self._finish_animation(task)
            task.cancel()
            raise
        if self._async_task is not task or not self._playing:
            task.cancel()
        return task

    def _track_closing_task(self, task: asyncio.Task[None]) -> None:
        """Retain a canceled task until its cleanup has finished."""
        self._closing_tasks.add(task)
        task.add_done_callback(self._closing_tasks.discard)

    def close(self) -> None:
        """Remove this actor and release player references without closing the plotter."""
        pl = self._pl
        if pl is None:
            return
        task = self._async_task
        self._async_task = None
        if task is not None and not task.done():
            self._track_closing_task(task)
        try:
            current_task = asyncio.current_task(loop=asyncio.get_running_loop())
        except RuntimeError:
            current_task = None
        if task is not None and task is not current_task and not task.done():
            task.cancel()
        actor = self._actor
        if isinstance(actor, pv.Volume) and actor.mapper is not None:
            actor.mapper.dataset = None
        if actor is not None and not getattr(pl, '_closed', False):
            pl.remove_actor(actor, render=False)
            if getattr(pl, 'mapper', None) is actor.mapper:
                pl.mapper = None
        if isinstance(actor, pv.Volume):
            # Smart volume mappers retain internal inputs after SetInputData(None).
            actor.mapper = None
        if getattr(pl, 'volume', None) is actor:
            pl.volume = None
        if pl.mesh is self._frame:
            pl.mesh = None
        self._playing = False
        self._started_at = None
        self._frame = None
        self._actor = None
        self._load_frame = None
        self._clock = None
        self._on_frame = None
        self._pl = None


class TimeReaderSource:
    """Adapt exact samples from a dedicated temporal reader.

    The source exposes ``times`` and ``frame(index)`` for
    :class:`TimeSeriesPlayer`. A selected MultiBlock part must be named
    explicitly. One deep copy detaches each returned frame before restoring
    the reader's previous active time. The caller owns the reader and must not
    change its time concurrently.

    .. versionadded:: 0.50

    Parameters
    ----------
    reader : pyvista.TimeReader
        Dedicated temporal reader with a ``read`` method.
    scalars : str
        Array required on the selected spatial part.
    block : int or str, optional
        Top-level spatial part of MultiBlock output.
    preference : str, default: 'point'
        Association of ``scalars``: ``'point'`` or ``'cell'``.

    """

    def __init__(
        self,
        reader: TimeReader,
        *,
        scalars: str,
        block: int | str | None = None,
        preference: Literal['point', 'cell'] = 'point',
    ) -> None:
        """Validate reader metadata without reading a result frame."""
        if preference not in ('point', 'cell'):
            msg = "Preference must be 'point' or 'cell'."
            raise ValueError(msg)
        self.times = tuple(float(value) for value in reader.time_values)
        if (
            not self.times
            or not all(math.isfinite(value) for value in self.times)
            or any(right <= left for left, right in zip(self.times, self.times[1:], strict=False))
        ):
            msg = 'Reader times must be finite, nonempty, and strictly increasing.'
            raise ValueError(msg)
        self.reader = reader
        self.scalars = scalars
        self.block = block
        self.preference = preference

    def frame(self, index: int) -> DataSet:
        """Read an exact sample without leaving the reader at another time.

        Parameters
        ----------
        index : int
            Index of the temporal sample to read.

        Returns
        -------
        pyvista.DataSet
            Independently owned selected spatial part.

        """
        if isinstance(index, bool):
            msg = 'Frame index must be an integer, not a boolean.'
            raise TypeError(msg)
        index = operator.index(index)
        if not 0 <= index < len(self.times):
            msg = 'Frame index is outside the available samples.'
            raise IndexError(msg)
        previous = self.reader.active_time_value
        try:
            self.reader.set_active_time_point(index)
            if self.reader.active_time_value != self.times[index]:
                msg = 'Reader did not select the requested simulation timestamp.'
                raise ValueError(msg)
            mesh: object = cast('Any', self.reader).read()
            if isinstance(mesh, pv.MultiBlock):
                if self.block is None:
                    msg = 'Select a spatial block explicitly for MultiBlock output.'
                    raise ValueError(msg)
                mesh = mesh[self.block]
            elif self.block is not None:
                msg = 'A block selection requires MultiBlock reader output.'
                raise ValueError(msg)
            if not isinstance(mesh, pv.DataSet):
                msg = 'The selected spatial part must be a dataset, not a null or composite block.'
                raise TypeError(msg)
            data = mesh.point_data if self.preference == 'point' else mesh.cell_data
            if self.scalars not in data:
                msg = f'Frame {index} has no {self.preference} array {self.scalars!r}.'
                raise ValueError(msg)
            return mesh.copy(deep=True)
        finally:
            self.reader.set_active_time_value(previous)
