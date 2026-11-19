"""Tests for notebook time-series display without a native render window."""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import pyvista as pv
from pyvista.plotting import _time_series_ui


class _Viewer:
    """Keep the real Trame layout while avoiding native rendering."""

    def __init__(self, pl, *, server, suppress_rendering=False):
        """Mirror the Trame viewer's assignment to the plotter."""
        self.server = server
        pl.suppress_rendering = suppress_rendering

    def ui(self, **kwargs):
        """Leave the rendering area empty."""


def _player(backend):
    """Make the minimal player needed to construct the controls."""
    theme = copy.deepcopy(pv.global_theme)
    theme.jupyter_backend = backend
    pl = SimpleNamespace(_theme=theme, suppress_rendering=True)
    return SimpleNamespace(
        _pl=pl,
        _active=lambda: pl,
        _on_frame=None,
        _render=True,
        actor=object(),
        index=0,
        time=0.0,
        times=(0, 1),
        pause=lambda: None,
    )


@pytest.mark.parametrize('backend', ['static', 'none', 'client', 'html'])
def test_notebook_view_rejects_unsupported_backend(backend):
    """Playback requires a live server-rendered notebook backend."""
    player = _player(backend)
    with pytest.raises(ValueError, match='requires the "trame" or "server"'):
        _time_series_ui._TimeSeriesView(player, cycle_seconds=1, title='Frames')
    assert player._render
    assert player._on_frame is None


def test_notebook_launch_uses_proxy_and_owned_server(monkeypatch):
    """Launch and embed the owned server using the plotter's notebook settings."""
    monkeypatch.setattr(_time_series_ui, 'Viewer', _Viewer)
    player = _player('server')
    player._pl._theme.trame.server_proxy_enabled = True
    player._pl._theme.trame.server_proxy_prefix = '/user/example/proxy/'
    player._pl._theme.trame.jupyter_server_port = 1234
    calls = []

    def launch(server, **kwargs):
        """Resolve readiness without binding a socket."""
        calls.append((server, kwargs))
        server._resolve_ready_future()

    monkeypatch.setattr(_time_series_ui, 'launch_server', launch)

    async def run():
        """Exercise startup and cleanup on the notebook event loop."""
        view = _time_series_ui._TimeSeriesView(player, cycle_seconds=1, title='Frames')
        assert player._pl.suppress_rendering
        view.server.stop = AsyncMock()
        await view.start(host='localhost')
        assert calls == [
            (view.server, {'host': 'localhost', 'port': 1234, 'wslink_backend': 'aiohttp'})
        ]
        assert view.layout.iframe_base_url == '/user/example/proxy'
        assert view.layout._iframe_builder.__name__ == 'iframe_url_builder_serverproxy'
        await view.stop()
        view.server.stop.assert_awaited_once()
        assert player._render
        assert player._on_frame is None
        assert player._pl.suppress_rendering

    asyncio.run(run())


def test_notebook_stop_restores_callbacks_when_pause_fails(monkeypatch):
    """A failing pause does not prevent detachment or server shutdown."""
    monkeypatch.setattr(_time_series_ui, 'Viewer', _Viewer)
    player = _player('server')
    callback = lambda _player: None
    player._on_frame = callback

    def fail_pause():
        """Simulate a frame callback or loader failing during pause."""
        msg = 'Frame failed.'
        raise ValueError(msg)

    player.pause = fail_pause

    async def run():
        """Stop the view on its event loop."""
        view = _time_series_ui._TimeSeriesView(player, cycle_seconds=1, title='Frames')
        view.server.stop = AsyncMock()
        with pytest.raises(ValueError, match='Frame failed'):
            await view.stop()
        assert player._on_frame is callback
        assert player._render
        view.server.stop.assert_awaited_once()
        await view.stop()
        view.server.stop.assert_awaited_once()

    asyncio.run(run())


def test_notebook_stop_can_retry_after_server_failure(monkeypatch):
    """A failed shutdown leaves the server available for another stop."""
    monkeypatch.setattr(_time_series_ui, 'Viewer', _Viewer)
    player = _player('server')

    async def run():
        """Retry cleanup after a server error."""
        view = _time_series_ui._TimeSeriesView(player, cycle_seconds=1, title='Frames')
        view.server.stop = AsyncMock(side_effect=[RuntimeError('Stop failed.'), None])
        with pytest.raises(RuntimeError, match='Stop failed'):
            await view.stop()
        assert player._render
        assert player._on_frame is None
        await view.stop()
        assert view.server.stop.await_count == 2

    asyncio.run(run())


def test_notebook_failed_launch_restores_player(monkeypatch):
    """An unsuccessful launch can be stopped without retaining callbacks."""
    monkeypatch.setattr(_time_series_ui, 'Viewer', _Viewer)
    player = _player('server')

    def fail_launch(_server, **_kwargs):
        """Fail after view construction."""
        msg = 'Cannot bind server.'
        raise OSError(msg)

    monkeypatch.setattr(_time_series_ui, 'launch_server', fail_launch)

    async def run():
        """Clean up after a launch error."""
        view = _time_series_ui._TimeSeriesView(player, cycle_seconds=1, title='Frames')
        view.server.stop = AsyncMock()
        with pytest.raises(OSError, match='Cannot bind'):
            await view.start()
        await view.stop()
        view.server.stop.assert_awaited_once()
        assert player._render
        assert player._on_frame is None

    asyncio.run(run())
