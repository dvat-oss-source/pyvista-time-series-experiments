"""Check discrete playback, ownership, and externally driven scheduling."""

from __future__ import annotations

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
