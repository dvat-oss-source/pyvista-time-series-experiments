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
