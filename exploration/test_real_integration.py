"""Exercise the temporal prototype with real datasets and file-backed readers."""

from __future__ import annotations

import json

from exploration.time_controller import TimeController
import numpy as np
import pytest
import pyvista as pv

pv.OFF_SCREEN = True


@pytest.fixture
def reader(tmp_path):
    """Create a VTK file series to exercise PyVista's temporal reader adapter."""
    files = []
    for index, time in enumerate([0.0, 0.2, 1.0]):
        mesh = pv.Plane(i_resolution=index + 2, j_resolution=2)
        mesh.points[:, 2] = 0.4 * time * mesh.points[:, 0]
        mesh['temperature'] = mesh.points[:, 0] + time
        name = f'frame_{index}.vtp'
        mesh.save(tmp_path / name)
        files.append({'name': name, 'time': time})
    path = tmp_path / 'frames.vtp.series'
    path.write_text(json.dumps({'file-series-version': '1.0', 'files': files}))
    result = pv.get_reader(path)
    result.set_active_time_value(0.2)
    return result


def test_real_reader_render_updates_and_preserves_camera(reader):
    """Render changing topology while preserving camera, actor and color limits."""
    pl = pv.Plotter(window_size=(400, 300))
    controller = TimeController.from_reader(pl, reader, scalars='temperature', clim=(-0.5, 1.5))
    pl.camera_position = 'iso'
    actor = controller.actor
    pl.show(auto_close=False, interactive=False)
    camera = pl.camera_position.to_list()
    initial = pl.screenshot(return_img=True)
    first_count = actor.mapper.dataset.n_points
    controller.select_time(1)
    assert reader.active_time_value == 0.2
    assert controller.actor is actor
    assert actor.mapper.dataset.n_points != first_count
    np.testing.assert_allclose(
        actor.mapper.dataset['temperature'], actor.mapper.dataset.points[:, 0] + 1
    )
    np.testing.assert_allclose(actor.mapper.scalar_range, (-0.5, 1.5))
    np.testing.assert_allclose(pl.camera_position.to_list(), camera)
    final = pl.screenshot(return_img=True)
    assert np.mean(np.abs(final.astype(float) - initial.astype(float))) > 1
    pl.show(interactive=False)


@pytest.mark.parametrize('clim', [(0, 0), (1, 0), (0,), (0, 1, 2), (0, np.nan), (0, np.inf)])
def test_invalid_color_limits(clim):
    """Reject unusable color limits before loading a dataset."""

    def load(_index):
        """Fail if invalid configuration reaches the data source."""
        pytest.fail('Invalid limits must not load a frame')

    with pytest.raises(ValueError, match='Color limits'):
        TimeController(None, [0], load, scalars='temperature', clim=clim)
