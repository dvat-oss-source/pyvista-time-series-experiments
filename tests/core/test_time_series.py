"""Check discrete playback, ownership, and externally driven scheduling."""

from __future__ import annotations

import asyncio
import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

import pyvista as pv


class Scene:
    """Exercise one real mapper without opening a render window."""

    def add_mesh(self, mesh, **kwargs):
        """Keep the actor and current plotter mesh reference."""
        self.mesh = mesh
        mapper = pv.DataSetMapper(dataset=mesh)
        mapper.set_active_scalars(kwargs['scalars'], preference=kwargs['preference'])
        self.actor = SimpleNamespace(mapper=mapper)
        return self.actor

    def add_slider_widget(self, *_args, **_kwargs):
        """Reject an internally owned scheduler or widget."""
        msg = 'A time series player must not create a slider.'
        raise AssertionError(msg)

    def render(self):
        """Update the native mapper pipeline."""
        self.actor.mapper.update()

    def remove_actor(self, actor, *, render=False):
        """Release the scene's actor reference."""
        assert render is False
        assert actor is self.actor
        self.actor = None


def make_frame(index):
    """Make an independently owned frame with changing geometry."""
    mesh = pv.Sphere(radius=1 + index / 10, theta_resolution=8, phi_resolution=8)
    mesh['value'] = np.full(mesh.n_points, index, dtype=np.float32)
    return mesh


@pytest.mark.parametrize('volume', [False, True])
def test_from_files_reads_only_selected_frame(tmp_path, monkeypatch, volume):
    """File playback preserves styling and reads no frames ahead of a seek."""
    paths = []
    for index in range(3):
        frame = pv.ImageData(dimensions=(5, 4, 3), spacing=(1, 2, 3), origin=(index, 0, 0))
        frame['value'] = np.full(frame.n_points, index, dtype=np.float32)
        path = tmp_path / f'{index}.pv'
        frame.save(path)
        paths.append(path)
    read = pv.read
    reads = []

    def load(path):
        """Record disk access without bypassing native file IO."""
        reads.append(path)
        return read(path)

    monkeypatch.setattr(pv, 'read', load)
    pl = pv.Plotter()
    player = pv.TimeSeriesPlayer.from_files(
        pl, paths, scalars='value', clim=(0, 2), volume=volume, render=False, cmap='plasma'
    )
    actor = player.actor
    assert isinstance(actor, pv.Volume if volume else pv.Actor)
    expected_colors = pv.LookupTable(cmap='plasma').values[:, :3]
    np.testing.assert_array_equal(actor.mapper.lookup_table.values[:, :3], expected_colors)
    assert reads == [paths[0]]
    assert player.times == (0, 1, 2)
    assert player.cycle_end == 3
    player.seek_index(2)
    assert player.actor is actor
    assert reads == [paths[0], paths[2]]
    assert actor.mapper.dataset == read(paths[2])
    player.close()
    pl.close()


def test_from_files_empty_sequence():
    """Reject an empty source before touching the plotter."""
    with pytest.raises(ValueError, match='nonempty'):
        pv.TimeSeriesPlayer.from_files(Scene(), [], scalars='value', clim=(0, 1))


def test_notebook_view_controls_and_cleanup(monkeypatch):
    """The built-in view owns scheduling, synchronization, and server cleanup."""
    from pyvista.plotting import _time_series_ui

    updates = []
    launches = []
    stops = []
    scene = Scene()
    scene._theme = SimpleNamespace(jupyter_backend='server', trame=pv.global_theme.trame)
    scene.suppress_rendering = False
    scene.reset_camera_clipping_range = lambda: None
    scene.close = lambda: stops.append('plotter')
    callback_indices = []
    player = pv.TimeSeriesPlayer(
        scene,
        [0, 1, 2],
        make_frame,
        cycle_end=3,
        scalars='value',
        clim=(0, 2),
        on_frame=lambda changed: callback_indices.append(changed.index),
    )

    class Viewer:
        """Replace only rendering, keeping real Trame controls and state."""

        def __init__(self, _pl, *, server, suppress_rendering=False):
            """Keep server construction separate from native graphics."""

        def ui(self, **_kwargs):
            """Build controls without creating a native render view."""

        def update(self):
            """Record frames published to the notebook."""
            updates.append(player.index)

    async def start(view, **kwargs):
        """Ready real state callbacks without opening a socket."""
        launches.append(kwargs)
        view.state.ready()

    async def stop_server():
        """Record explicit server shutdown."""
        stops.append('server')

    monkeypatch.setattr(_time_series_ui, 'Viewer', Viewer)
    monkeypatch.setattr(_time_series_ui._TimeSeriesView, 'start', start)

    async def run():
        """Use controls on the same event loop as their plotter."""
        layout = await player.show(12, title='Example', host='127.0.0.1')
        view = player._view
        monkeypatch.setattr(view.server, 'stop', stop_server)
        assert await player.show(12) is layout
        assert launches == [{'host': '127.0.0.1'}]
        with view.state:
            view.state.frame = 2
        assert player.index == 2
        assert updates == [2]
        assert callback_indices == [2]
        player.seek_index(1)
        assert view.state.frame == 1
        assert updates == [2, 1]
        assert callback_indices == [2, 1]
        with view.state:
            view.state.playing = True
        task = player._async_task
        assert task is not None
        assert player.playing
        with view.state:
            view.state.playing = False
        assert not player.playing
        await asyncio.sleep(0)
        assert task.cancelled()
        await player.aclose()
        await player.aclose()
        assert stops == ['server']
        assert player._view is None
        assert player._pl is None
        with view.state:
            view.state.frame = 0

    asyncio.run(run())


def test_aclose_awaits_playback_and_preserves_plotter():
    """Release playback resources while leaving the caller's plotter usable."""
    pl = pv.Plotter()
    callback_indices = []
    player = pv.TimeSeriesPlayer(
        pl,
        [0, 1],
        make_frame,
        cycle_end=2,
        scalars='value',
        clim=(0, 1),
        render=False,
        on_frame=lambda changed: callback_indices.append(changed.index),
    )

    async def run():
        task = player.start(1)
        await asyncio.sleep(0)
        await player.aclose()
        assert task.cancelled()
        assert player._async_task is None
        assert player._on_frame is None
        assert not player._playing
        await player.aclose()

    asyncio.run(run())
    assert not pl._closed
    assert callback_indices == []
    actor = pl.add_mesh(pv.Sphere(), render=False)
    assert actor in pl.renderer.actors.values()
    pl.close()


def test_aclose_drains_task_after_close():
    """A later asynchronous close waits for cancellation cleanup."""
    player = pv.TimeSeriesPlayer(
        Scene(), [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
    )

    async def run():
        task = player.start(1)
        await asyncio.sleep(0)
        player.close()
        await player.aclose()
        assert task.cancelled()
        assert not player._closing_tasks

    asyncio.run(run())


def test_aclose_drains_paused_tasks():
    """Wait for canceled playback tasks even after pause and restart."""
    player = pv.TimeSeriesPlayer(
        Scene(), [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
    )

    async def run():
        release = asyncio.Event()
        drive = player._drive_animation

        async def delayed_cleanup(interval):
            try:
                await drive(interval)
            finally:
                await release.wait()

        player._drive_animation = delayed_cleanup
        first = player.start(1)
        await asyncio.sleep(0)
        player.pause()
        second = player.start(1)
        await asyncio.sleep(0)
        player.pause()
        cleanup = asyncio.create_task(player.aclose())
        await asyncio.sleep(0)
        assert not cleanup.done()
        release.set()
        await cleanup
        assert first.cancelled()
        assert second.cancelled()
        assert not player._closing_tasks

    asyncio.run(run())


def test_aclose_does_not_await_current_task():
    """Closing from the animation task does not await that same task."""
    player = pv.TimeSeriesPlayer(
        Scene(), [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
    )

    async def run():
        player._async_task = asyncio.current_task()
        await player.aclose()
        assert player._pl is None

    asyncio.run(run())


def test_show_during_start_and_close(monkeypatch):
    """Reject overlapping startup and report closure during startup clearly."""
    from pyvista.plotting import _time_series_ui

    player = pv.TimeSeriesPlayer(
        Scene(), [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
    )

    class View:
        def __init__(self, *_args, **_kwargs):
            """Hold startup until the test releases it."""
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.layout = object()

        async def start(self, **_kwargs):
            """Simulate server startup."""
            self.started.set()
            await self.release.wait()

        async def stop(self):
            """Simulate server shutdown."""

    monkeypatch.setattr(_time_series_ui, '_TimeSeriesView', View)

    async def run():
        showing = asyncio.create_task(player.show())
        await asyncio.sleep(0)
        view = player._view
        await view.started.wait()
        with pytest.raises(RuntimeError, match='still starting'):
            await player.show()
        cleanup = asyncio.create_task(player.aclose())
        await asyncio.sleep(0)
        assert not cleanup.done()
        view.release.set()
        with pytest.raises(RuntimeError, match='closed while'):
            await showing
        await cleanup

    asyncio.run(run())


def test_aclose_can_retry_interrupted_view_cleanup():
    """Keep the view available until its asynchronous cleanup succeeds."""
    player = pv.TimeSeriesPlayer(
        Scene(), [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
    )

    class View:
        def __init__(self):
            """Track server shutdown attempts."""
            self.calls = 0
            self.started = asyncio.Event()

        async def stop(self):
            """Wait for cancellation on the first attempt."""
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                await asyncio.Event().wait()

    async def run():
        view = View()
        player._view = view
        cleanup = asyncio.create_task(player.aclose())
        await view.started.wait()
        cleanup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cleanup
        assert player._view is view
        await player.aclose()
        assert view.calls == 2
        assert player._view is None

    asyncio.run(run())


def test_notebook_view_start_failure_restores_player(monkeypatch):
    """A failed server launch restores caller-owned rendering and callbacks."""
    from pyvista.plotting import _time_series_ui

    scene = Scene()
    player = pv.TimeSeriesPlayer(
        scene, [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
    )
    restored = []

    class View:
        """Fail startup after attaching the same resources as a notebook view."""

        def __init__(self, player, **_kwargs):
            """Temporarily take responsibility for frame rendering."""
            player._render = False

        async def start(self, **_kwargs):
            """Simulate a failed notebook server bind."""
            msg = 'Cannot bind server.'
            raise OSError(msg)

        async def stop(self):
            """Restore rendering without closing the caller's player."""
            player._render = True
            restored.append(True)

    monkeypatch.setattr(_time_series_ui, '_TimeSeriesView', View)
    with pytest.raises(OSError, match='Cannot bind'):
        asyncio.run(player.show())
    assert restored == [True]
    assert player._view is None
    assert player._render
    player.seek_index(1)
    assert player.index == 1
    player.close()


def test_notebook_view_start_failure_preserves_error(monkeypatch):
    """A cleanup failure does not replace the server launch error."""
    from pyvista.plotting import _time_series_ui

    player = pv.TimeSeriesPlayer(
        Scene(), [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
    )

    class View:
        def __init__(self, *_args, **_kwargs):
            """Simulate a notebook view."""

        async def start(self, **_kwargs):
            """Fail to launch."""
            msg = 'Cannot bind server.'
            raise OSError(msg)

        async def stop(self):
            """Fail while cleaning up the failed launch."""
            msg = 'Cannot stop server.'
            raise RuntimeError(msg)

    monkeypatch.setattr(_time_series_ui, '_TimeSeriesView', View)
    with pytest.raises(OSError, match='Cannot bind'):
        asyncio.run(player.show())
    assert isinstance(player._view, View)
    player.close()


def test_attached_actor_updates_plotter_mesh_reference():
    """Release the input replaced when attaching to an existing actor."""
    scene = Scene()
    initial = make_frame(0)
    values = np.zeros(initial.n_points, dtype=np.float32)
    initial['value'] = values
    actor = scene.add_mesh(initial, scalars='value', preference='point')
    initial_values = weakref.ref(values)
    del initial, values
    player = pv.TimeSeriesPlayer(
        scene,
        [0, 1],
        make_frame,
        cycle_end=2,
        scalars='value',
        clim=(0, 1),
        actor=actor,
        render=False,
    )
    assert scene.mesh is actor.mapper.dataset
    assert initial_values() is None
    player.seek_index(1)
    assert scene.mesh is actor.mapper.dataset
    player.close()
    assert scene.mesh is None


def test_existing_actor_reuses_mapper():
    """Attach playback without replacing an actor's configured appearance."""
    scene = Scene()
    initial = make_frame(0)
    initial['other'] = np.ones(initial.n_points)
    actor = scene.add_mesh(initial, scalars='other', preference='point')
    mapper = actor.mapper
    renders = []
    scene.render = lambda: renders.append(True)
    player = pv.TimeSeriesPlayer(
        scene,
        [0, 1],
        make_frame,
        cycle_end=2,
        scalars='value',
        clim=(0, 1),
        actor=actor,
        render=False,
    )
    player.seek_index(1)
    assert mapper.array_name == 'value'
    assert player.actor is actor
    assert player.actor.mapper is mapper
    np.testing.assert_array_equal(mapper.dataset['value'], make_frame(1)['value'])
    player.step()
    assert player.index == 0
    player.close()
    player.close()
    assert not renders


@pytest.mark.parametrize('plotter_closed', [False, True])
@pytest.mark.parametrize('other_volume', [False, True])
def test_volume_input_released_on_close(plotter_closed, other_volume):
    """Closing detaches volume buffers even if a caller holds the actor."""

    def load(index):
        mesh = pv.ImageData(dimensions=(4, 3, 2))
        mesh['value'] = np.full(mesh.n_points, index, dtype=np.float32)
        return mesh

    scene = Scene()
    actor = pv.Volume()
    mapper = pv.SmartVolumeMapper()
    actor.mapper = mapper
    mapper.lookup_table = pv.LookupTable()
    scene.actor = actor
    scene.volume = pv.Volume() if other_volume else actor
    cached_volume = scene.volume
    scene.mesh = None
    player = pv.TimeSeriesPlayer(
        scene, [0, 1], load, cycle_end=2, scalars='value', clim=(0, 1), actor=actor, render=False
    )
    player.seek_index(1)
    np.testing.assert_array_equal(actor.mapper.dataset['value'], load(1)['value'])
    scene._closed = plotter_closed
    player.close()
    player.close()
    assert actor.mapper is None
    assert scene.volume is (cached_volume if other_volume else None)


def test_volume_buffers_shared_and_released_repeatedly():
    """Share the displayed buffer and release it without collecting garbage."""
    buffers = []

    def load(index):
        grid = pv.ImageData(dimensions=(4 + index, 3, 2))
        values = np.full(grid.n_points, index, dtype=np.float32)
        buffers.append(weakref.ref(values))
        grid['value'] = values
        return grid

    for _ in range(3):
        scene = Scene()
        scene.mesh = load(0)
        mapper = pv.SmartVolumeMapper()
        mapper.lookup_table = pv.LookupTable()
        mapper.dataset = scene.mesh
        actor = pv.Volume()
        actor.mapper = mapper
        scene.actor = actor
        scene.volume = actor
        player = pv.TimeSeriesPlayer(
            scene,
            [0, 1, 2],
            load,
            cycle_end=3,
            scalars='value',
            clim=(0, 2),
            actor=actor,
            render=False,
        )
        for index in (2, 0, 1, 2, 0):
            player.seek_index(index)
            displayed = mapper.dataset['value']
            assert np.shares_memory(displayed, buffers[-1]())
            np.testing.assert_array_equal(displayed, index)
            del displayed
            assert sum(ref() is not None for ref in buffers) == 1
        player.close()
        player.close()
        assert mapper.dataset is None
        assert scene.mesh is None
        assert scene.volume is None
        assert all(ref() is None for ref in buffers)


def test_exact_seek_and_irregular_timestamps():
    """Select exact and nearest samples without retaining prior frames."""
    calls = []

    def load(index):
        """Record frame requests."""
        calls.append(index)
        return make_frame(index)

    player = pv.TimeSeriesPlayer(
        Scene(), [0, 0.15, 0.3, 0.5, 0.9], load, cycle_end=1, scalars='value', clim=(0, 4)
    )
    actor = player.actor
    player.seek_index(2)
    assert player.index == 2
    assert player.time == 0.3
    np.testing.assert_array_equal(actor.mapper.dataset.points, make_frame(2).points)
    player.seek_time(0.4)
    assert player.index == 2
    player.step(1)
    assert player.index == 3
    player.step(2)
    assert player.index == 0
    assert player.actor is actor
    assert calls == [0, 2, 3, 0]
    player.close()


def test_on_frame_runs_after_changed_frame_render():
    """Notify the caller once the updated frame has rendered."""
    pl = Scene()
    events = []
    pl.render = lambda: events.append(('render', pl.actor.mapper.dataset['value'][0]))

    def on_frame(player):
        events.append(('callback', player.index, player.actor.mapper.dataset['value'][0]))

    player = pv.TimeSeriesPlayer(
        pl,
        [0, 1],
        make_frame,
        cycle_end=2,
        scalars='value',
        clim=(0, 1),
        on_frame=on_frame,
    )
    player.seek_index(0)
    player.seek_index(1)
    assert events == [('render', 1), ('callback', 1, 1)]
    player.close()


def test_failed_load_does_not_call_on_frame():
    """Leave callback state untouched when a replacement frame fails."""
    callbacks = []

    def load(index):
        if index == 1:
            msg = 'Missing frame.'
            raise OSError(msg)
        return make_frame(index)

    player = pv.TimeSeriesPlayer(
        Scene(),
        [0, 1],
        load,
        cycle_end=2,
        scalars='value',
        clim=(0, 1),
        on_frame=callbacks.append,
    )
    with pytest.raises(OSError, match='Missing frame'):
        player.seek_index(1)
    assert callbacks == []
    assert player.index == 0
    player.close()


def test_on_frame_reentrant_seek_keeps_latest_timeline_state():
    """Keep nested callback seeks after the outer seek has finished committing."""
    calls = []

    def on_frame(player):
        calls.append((player.index, player.time))
        if player.index == 1:
            player.seek_index(2)

    player = pv.TimeSeriesPlayer(
        Scene(),
        [0, 1, 2],
        make_frame,
        cycle_end=3,
        scalars='value',
        clim=(0, 2),
        on_frame=on_frame,
    )
    player.seek_index(1)
    assert calls == [(1, 1), (2, 2)]
    assert player.index == 2
    assert player.time == 2
    player.close()


def test_on_frame_pause_during_tick_keeps_paused_state():
    """Keep a pause requested by a frame callback after the outer tick returns."""
    now = [0.0]
    player_holder = {}

    def on_frame(_player):
        player_holder['player'].pause()

    player = pv.TimeSeriesPlayer(
        Scene(),
        [0, 0.5, 0.9],
        make_frame,
        cycle_end=1,
        scalars='value',
        clim=(0, 2),
        clock=lambda: now[0],
        on_frame=on_frame,
    )
    player_holder['player'] = player
    player.play(1)
    now[0] = 0.5
    player.tick()
    assert player.index == 1
    assert player.time == 0.5
    assert not player.playing
    assert player._started_at is None
    player.close()


def test_on_frame_close_during_tick_keeps_closed_state():
    """Do not write playback timing after a callback closes the player."""
    now = [0.0]
    player_holder = {}

    def on_frame(_player):
        player_holder['player'].close()

    player = pv.TimeSeriesPlayer(
        Scene(),
        [0, 0.5, 0.9],
        make_frame,
        cycle_end=1,
        scalars='value',
        clim=(0, 2),
        clock=lambda: now[0],
        on_frame=on_frame,
    )
    player_holder['player'] = player
    player.play(1)
    now[0] = 0.5
    player.tick()
    assert player._pl is None
    assert player._position == 0.5
    assert player._started_at is None
    assert not player._playing


def test_close_releases_on_frame_callback():
    """Drop the callback reference when closing the player."""

    class Callback:
        def __call__(self, _player):
            pass

    callback = Callback()
    reference = weakref.ref(callback)
    player = pv.TimeSeriesPlayer(
        Scene(),
        [0, 1],
        make_frame,
        cycle_end=2,
        scalars='value',
        clim=(0, 1),
        on_frame=callback,
    )
    del callback
    player.close()
    gc.collect()
    assert reference() is None


def test_animate_runs_nonlooping_until_final_dwell(monkeypatch):
    """Drive non-looping playback using the application's asyncio loop."""
    now = [0.0]
    sleeps = []

    async def sleep(interval):
        sleeps.append(interval)
        now[0] += 0.5

    monkeypatch.setattr(asyncio, 'sleep', sleep)
    player = pv.TimeSeriesPlayer(
        Scene(),
        [0, 0.5, 0.9],
        make_frame,
        cycle_end=1,
        scalars='value',
        clim=(0, 2),
        clock=lambda: now[0],
    )
    asyncio.run(player.animate(1, interval=0.1, loop=False))
    assert sleeps == [0.1, 0.1]
    assert player.index == 2
    assert not player.playing
    assert player._async_task is None
    player.close()


def test_animate_rejects_invalid_interval_before_playing():
    """Validate the scheduler interval before starting playback."""
    player = pv.TimeSeriesPlayer(
        Scene(), [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
    )
    with pytest.raises(ValueError, match='Animation interval'):
        asyncio.run(player.animate(1, interval=0))
    assert not player.playing
    player.close()


def test_animate_loader_error_is_not_retried_during_cleanup(monkeypatch):
    """Preserve a loader exception without another fallible cleanup tick."""
    now = [0.0]
    failure = OSError('Missing frame.')
    failed_indices = []

    def load(index):
        if index == 1:
            failed_indices.append(index)
            raise failure
        return make_frame(index)

    async def sleep(_interval):
        now[0] = 0.5

    monkeypatch.setattr(asyncio, 'sleep', sleep)
    player = pv.TimeSeriesPlayer(
        Scene(),
        [0, 0.5, 0.9],
        load,
        cycle_end=1,
        scalars='value',
        clim=(0, 2),
        clock=lambda: now[0],
    )

    async def run():
        task = player.start(1)
        with pytest.raises(OSError, match='Missing frame') as exc_info:
            await task
        assert exc_info.value is failure

    asyncio.run(run())
    assert failed_indices == [1]
    assert player.index == 0
    assert not player._playing
    assert player._started_at is None
    assert player._async_task is None
    player.close()


def test_animate_callback_error_keeps_committed_state(monkeypatch):
    """Propagate a callback error after committing the selected frame and time."""
    now = [0.0]
    failure = RuntimeError('Display update failed.')

    def on_frame(_player):
        raise failure

    async def sleep(_interval):
        now[0] = 0.5

    monkeypatch.setattr(asyncio, 'sleep', sleep)
    player = pv.TimeSeriesPlayer(
        Scene(),
        [0, 0.5, 0.9],
        make_frame,
        cycle_end=1,
        scalars='value',
        clim=(0, 2),
        clock=lambda: now[0],
        on_frame=on_frame,
    )

    async def run():
        task = player.start(1)
        with pytest.raises(RuntimeError, match='Display update failed') as exc_info:
            await task
        assert exc_info.value is failure

    asyncio.run(run())
    assert player.index == 1
    assert player._position == 0.5
    assert not player._playing
    assert player._started_at is None
    assert player._async_task is None
    player.close()


def test_pause_stops_even_when_tick_fails():
    """Clear playback state even if the pause tick cannot load its frame."""
    now = [0.0]
    failure = OSError('Missing frame.')

    def load(index):
        if index == 1:
            raise failure
        return make_frame(index)

    player = pv.TimeSeriesPlayer(
        Scene(),
        [0, 0.5, 0.9],
        load,
        cycle_end=1,
        scalars='value',
        clim=(0, 2),
        clock=lambda: now[0],
    )
    player.play(1)
    now[0] = 0.5
    with pytest.raises(OSError, match='Missing frame') as exc_info:
        player.pause()
    assert exc_info.value is failure
    assert not player._playing
    assert player._started_at is None
    player.close()


def test_animate_cancellation_pauses(monkeypatch):
    """Pause on task cancellation and propagate cancellation to the caller."""
    waiting = asyncio.Event()

    async def sleep(_interval):
        waiting.set()
        await asyncio.Future()

    monkeypatch.setattr(asyncio, 'sleep', sleep)

    async def run():
        player = pv.TimeSeriesPlayer(
            Scene(), [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
        )
        task = asyncio.create_task(player.animate(1))
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not player.playing
        player.close()

    asyncio.run(run())


def test_animate_rejects_concurrent_coroutine(monkeypatch):
    """Prevent two async schedulers from driving the same player."""
    waiting = asyncio.Event()

    async def sleep(_interval):
        waiting.set()
        await asyncio.Future()

    monkeypatch.setattr(asyncio, 'sleep', sleep)

    async def run():
        player = pv.TimeSeriesPlayer(
            Scene(), [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
        )
        task = asyncio.create_task(player.animate(1))
        await waiting.wait()
        with pytest.raises(RuntimeError, match='already running'):
            await player.animate(1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not player.playing
        player.close()

    asyncio.run(run())


def test_animate_stops_after_close(monkeypatch):
    """Exit cleanly when the player closes during an async sleep."""
    waiting = asyncio.Event()
    resume = asyncio.Event()

    async def sleep(_interval):
        waiting.set()
        await resume.wait()

    monkeypatch.setattr(asyncio, 'sleep', sleep)

    async def run():
        player = pv.TimeSeriesPlayer(
            Scene(), [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
        )
        task = asyncio.create_task(player.animate(1))
        await waiting.wait()
        player.close()
        resume.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert player._async_task is None

    asyncio.run(run())


def test_start_returns_same_task_and_pause_cancels(monkeypatch):
    """Reuse one live task and cancel it when playback is paused."""
    waiting = asyncio.Event()

    async def sleep(_interval):
        waiting.set()
        await asyncio.Future()

    monkeypatch.setattr(asyncio, 'sleep', sleep)

    async def run():
        player = pv.TimeSeriesPlayer(
            Scene(), [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
        )
        task = player.start(1)
        assert player.start(9) is task
        await waiting.wait()
        player.pause()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not player.playing
        assert player._async_task is None
        player.close()

    asyncio.run(run())


def test_start_restart_is_not_paused_by_old_cancellation(monkeypatch):
    """Keep a replacement task running as its cancelled predecessor unwinds."""
    first_waiting = asyncio.Event()
    second_waiting = asyncio.Event()

    async def sleep(_interval):
        if not first_waiting.is_set():
            first_waiting.set()
            await asyncio.Future()
        second_waiting.set()
        await asyncio.Future()

    monkeypatch.setattr(asyncio, 'sleep', sleep)

    async def run():
        player = pv.TimeSeriesPlayer(
            Scene(), [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
        )
        old_task = player.start(1)
        await first_waiting.wait()
        player.pause()
        new_task = player.start(1)
        assert new_task is not old_task
        await second_waiting.wait()
        with pytest.raises(asyncio.CancelledError):
            await old_task
        assert player._async_task is new_task
        assert player.playing
        player.pause()
        with pytest.raises(asyncio.CancelledError):
            await new_task
        assert not player.playing
        player.close()

    asyncio.run(run())


def test_start_requires_running_loop_and_valid_interval():
    """Reject invalid start conditions before changing playback state."""
    player = pv.TimeSeriesPlayer(
        Scene(), [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
    )
    with pytest.raises(RuntimeError, match='running event loop'):
        player.start(1)
    assert not player.playing

    async def run():
        with pytest.raises(ValueError, match='Animation interval'):
            player.start(1, interval=float('inf'))
        assert not player.playing

    asyncio.run(run())
    player.close()


def test_start_close_cancels_animation_task(monkeypatch):
    """Cancel an outstanding managed task when closing the player."""
    waiting = asyncio.Event()

    async def sleep(_interval):
        waiting.set()
        await asyncio.Future()

    monkeypatch.setattr(asyncio, 'sleep', sleep)

    async def run():
        player = pv.TimeSeriesPlayer(
            Scene(), [0, 1], make_frame, cycle_end=2, scalars='value', clim=(0, 1)
        )
        task = player.start(1)
        await waiting.wait()
        player.close()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert player._async_task is None

    asyncio.run(run())


@pytest.mark.parametrize('action', ['pause', 'close'])
def test_start_callback_stops_player_before_first_tick(action):
    """Return a canceled task when the restart callback stops playback."""
    now = [0.0]
    restart = [False]
    player_holder = {}

    def on_frame(player):
        if restart[0] and player.index == 0:
            if action == 'pause':
                player_holder['player'].pause()
            else:
                player_holder['player'].close()

    player = pv.TimeSeriesPlayer(
        Scene(),
        [0, 0.5, 0.9],
        make_frame,
        cycle_end=1,
        scalars='value',
        clim=(0, 2),
        clock=lambda: now[0],
        on_frame=on_frame,
    )
    player_holder['player'] = player
    player.play(1, loop=False)
    now[0] = 1.0
    player.tick()
    assert player.index == 2
    assert not player.playing
    restart[0] = True

    async def run():
        task = player.start(1, loop=False)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert player._async_task is None

    asyncio.run(run())
    if action == 'pause':
        assert player.index == 0
        assert not player._playing
        player.close()
    else:
        assert player._pl is None


def test_clock_dwell_pause_resume_and_idempotent_play():
    """Preserve fractional phase across pause and resume, including final dwell."""
    now = [0.0]
    calls = []

    def load(index):
        """Record only frames actually displayed."""
        calls.append(index)
        return make_frame(index)

    player = pv.TimeSeriesPlayer(
        Scene(),
        [0, 0.15, 0.3, 0.5, 0.9],
        load,
        cycle_end=1,
        scalars='value',
        clim=(0, 4),
        clock=lambda: now[0],
    )
    player.play(0.5)
    player.play(0.5)
    now[0] = 0.25
    player.tick()
    assert player.index == 3
    player.pause()
    now[0] = 5.0
    player.tick()
    assert player.index == 3
    player.play(0.5)
    now[0] = 5.20
    player.tick()
    assert player.index == 4
    now[0] = 5.24
    player.tick()
    assert player.index == 4
    now[0] = 5.26
    player.tick()
    assert player.index == 0
    assert calls == [0, 3, 4, 0]
    player.close()


def test_nonlooping_playback_and_failed_frame():
    """Stop at the final frame and keep the old frame when loading fails."""
    now = [0.0]

    def load(index):
        """Fail one replacement without changing the scene."""
        if index == 1:
            msg = 'Missing frame.'
            raise OSError(msg)
        return make_frame(index)

    player = pv.TimeSeriesPlayer(
        Scene(),
        [0, 0.5, 0.9],
        load,
        cycle_end=1,
        scalars='value',
        clim=(0, 2),
        clock=lambda: now[0],
    )
    original = player.actor.mapper.dataset
    with pytest.raises(OSError, match='Missing frame'):
        player.seek_index(1)
    assert player.index == 0
    assert player.actor.mapper.dataset is original
    player.play(1, loop=False)
    now[0] = 2.0
    player.tick()
    assert player.index == 2
    assert not player.playing
    player.close()


def test_nonlooping_mode_survives_pause_and_resume():
    """Keep a selected play mode when resume omits the loop argument."""
    now = [0.0]
    player = pv.TimeSeriesPlayer(
        Scene(),
        [0, 0.5, 0.9],
        make_frame,
        cycle_end=1,
        scalars='value',
        clim=(0, 2),
        clock=lambda: now[0],
    )
    player.play(1, loop=False)
    now[0] = 0.6
    player.pause()
    assert player.index == 1
    now[0] = 2.0
    player.play(1)
    now[0] = 2.5
    player.tick()
    assert player.index == 2
    assert not player.playing
    player.play(1)
    assert player.index == 0
    player.close()


def test_reject_overflowing_cycle_span():
    """Reject finite timestamps whose span has no finite playback arithmetic."""
    with pytest.raises(ValueError, match='Cycle end'):
        pv.TimeSeriesPlayer(
            Scene(),
            (-1e308, 1e308),
            make_frame,
            cycle_end=1.7e308,
            scalars='value',
            clim=(0, 2),
        )


def test_large_finite_tick_does_not_corrupt_position():
    """Keep a looping phase finite when many display cycles elapse."""
    now = [0.0]
    player = pv.TimeSeriesPlayer(
        Scene(),
        [0, 0.5],
        make_frame,
        cycle_end=1,
        scalars='value',
        clim=(0, 1),
        clock=lambda: now[0],
    )
    player.play(1)
    now[0] = 1e308
    player.tick()
    assert player.index == 0
    assert player.playing
    player.close()


def test_close_releases_loader_and_frame():
    """Drop player-owned references while leaving the scene independently usable."""
    pl = Scene()
    player = pv.TimeSeriesPlayer(
        pl, [0, 0.9], make_frame, cycle_end=1, scalars='value', clim=(0, 1)
    )
    frame = weakref.ref(player.actor.mapper.dataset)
    player.seek_index(1)
    gc.collect()
    assert frame() is None
    current = weakref.ref(player.actor.mapper.dataset)
    player.close()
    player.close()
    gc.collect()
    assert current() is None
    assert pl.mesh is None
    assert pl.actor is None
    with pytest.raises(RuntimeError, match='closed'):
        player.tick()


@pytest.mark.parametrize('cycle_end', [0.9, float('nan'), float('inf')])
def test_cycle_end_requires_final_dwell(cycle_end):
    """Reject ambiguous or invalid cycle bounds before adding an actor."""
    pl = Scene()
    with pytest.raises(ValueError, match='Cycle end'):
        pv.TimeSeriesPlayer(
            pl, [0, 0.9], make_frame, cycle_end=cycle_end, scalars='value', clim=(0, 1)
        )
    assert not hasattr(pl, 'actor')
