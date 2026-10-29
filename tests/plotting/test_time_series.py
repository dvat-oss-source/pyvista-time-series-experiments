"""Check that discrete playback changes the rendered scene."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

import pyvista as pv

pytestmark = pytest.mark.skip_plotting


def test_rendered_seek_and_independent_plotter_close(tmp_path):
    """Show exact file-backed geometry and release a player after its plotter closes."""
    paths = []
    for index, center in enumerate((-0.7, 0.7)):
        mesh = pv.Sphere(center=(center, 0, 0), theta_resolution=12, phi_resolution=12)
        mesh['value'] = np.full(mesh.n_points, index, dtype=np.float32)
        path = tmp_path / f'frame_{index}.vtp'
        mesh.save(path)
        paths.append(path)

    reads = []

    def load(index):
        """Read only a requested frame from its original file."""
        reads.append(index)
        return pv.read(paths[index])

    pl = pv.Plotter(window_size=(200, 160))
    player = pv.TimeSeriesPlayer(pl, (0.0, 0.7), load, cycle_end=1.0, scalars='value', clim=(0, 1))
    try:
        pl.set_background('black')
        pl.camera_position = [(0, -5, 0), (0, 0, 0), (0, 0, 1)]
        pl.show(interactive=False, auto_close=False)
        first_image = pl.screenshot()
        previous = weakref.ref(player.actor.mapper.dataset)
        player.seek_index(1)
        second_image = pl.screenshot()
        gc.collect()
        assert previous() is None
        assert reads == [0, 1]
        assert player.actor.mapper.dataset == pv.read(paths[1])
        assert np.count_nonzero(first_image != second_image) > 100
    finally:
        pl.close()
    with pytest.raises(RuntimeError, match='closed'):
        player.tick()
    player.close()


def test_volume_matches_direct_render(tmp_path):
    """Compressed samples preserve volume appearance, seeking, and actor properties."""
    paths = []
    for index in range(2):
        grid = pv.ImageData(dimensions=(12, 10, 8), spacing=(1, 2, 3))
        grid['density'] = np.linspace(0, 1 + index, grid.n_points, dtype=np.float32)
        path = tmp_path / f'{index}.pv'
        grid.save(path)
        paths.append(path)

    def load(index):
        return pv.read(paths[index])

    pl = pv.Plotter(window_size=(200, 160))
    actor = pl.add_volume(
        pv.read(paths[0]),
        scalars='density',
        clim=(0, 2),
        mapper='smart',
        shade=True,
        show_scalar_bar=False,
    )
    pl.camera_position = 'iso'
    pl.show(interactive=False, auto_close=False)
    direct = []
    for path in paths:
        actor.mapper.dataset = pv.read(path)
        direct.append(pl.screenshot())
    player = pv.TimeSeriesPlayer(
        pl, [0, 1], load, cycle_end=2, scalars='density', clim=(0, 2), actor=actor
    )
    for index in (0, 1, 0, 1):
        player.seek_index(index)
        np.testing.assert_array_equal(pl.screenshot(), direct[index])
    player.close()
    player.close()
    assert actor.mapper is None
    assert pl.volume is None
    pl.show()


@pytest.mark.parametrize('plotter_closed', [False, True])
@pytest.mark.parametrize('keep_mapper', [False, True])
def test_close_releases_rendered_volume_buffers(plotter_closed, keep_mapper):
    """Release native mapper caches even while the volume actor is retained."""
    references = []

    def load(index):
        """Return independently owned float32 values with asymmetric geometry."""
        mesh = pv.ImageData(dimensions=(14, 12, 8), spacing=(1, 2, 3), origin=(index, 0, 0))
        values = np.linspace(0, index + 1, mesh.n_points, dtype=np.float32)
        references.append(weakref.ref(values))
        mesh['density'] = values
        return mesh

    pl = pv.Plotter(window_size=(200, 160))
    actor = pl.add_volume(load(0), scalars='density', clim=(0, 2), mapper='smart')
    player = pv.TimeSeriesPlayer(
        pl, [0, 1], load, cycle_end=2, scalars='density', clim=(0, 2), actor=actor
    )
    pl.camera_position = 'iso'
    pl.show(interactive=False, auto_close=False)
    player.seek_index(1)
    assert references[-1]() is not None
    mapper = actor.mapper if keep_mapper else None
    if plotter_closed:
        pl.close()
    player.close()
    player.close()
    if keep_mapper:
        assert mapper.dataset is None
        assert references[-1]() is not None
        del mapper
    assert all(reference() is None for reference in references)
    if not plotter_closed:
        pl.show()
