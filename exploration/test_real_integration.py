"""Exercise the temporal prototype with real datasets and file-backed readers."""

from __future__ import annotations

import json

import numpy as np
import pytest
import pyvista as pv

from time_series import TimeSeries


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


def test_real_reader_restores_selection_and_detaches_output(reader):
    """Read varying topology through the actual file-series reader."""
    series = TimeSeries.from_reader(reader)
    assert reader.active_time_value == 0.2
    first = series[0]
    final = series.at_time(1.0)
    assert first.n_points != final.n_points
    assert reader.active_time_value == 0.2
    np.testing.assert_allclose(final['temperature'], final.points[:, 0] + 1)
    final.points[:] = 99
    final['temperature'][:] = 99
    fresh = series[-1]
    np.testing.assert_allclose(fresh['temperature'], fresh.points[:, 0] + 1)
    assert reader.active_time_value == 0.2


def test_composite_snapshots_have_independent_nested_arrays():
    """Preserve spatial blocks and detach arrays at every nesting level."""
    mesh = pv.Cube()
    mesh['values'] = np.arange(mesh.n_points)
    block = pv.MultiBlock({'outer': pv.MultiBlock({'surface': mesh})})
    series = TimeSeries.from_datasets([0], [block])
    mesh['values'][:] = -1
    frame = series[0]
    np.testing.assert_array_equal(frame['outer']['surface']['values'], np.arange(mesh.n_points))
    frame['outer']['surface']['values'][:] = -2
    np.testing.assert_array_equal(
        series[0]['outer']['surface']['values'], np.arange(mesh.n_points)
    )


def test_callback_stays_lazy_and_frames_are_independent():
    """Construction and metadata access must not materialize frames."""
    calls = []
    mesh = pv.Cube()

    def load(index):
        """Record frame requests while reusing one dataset."""
        calls.append(index)
        return mesh

    series = TimeSeries([0, 2], load)
    assert len(series) == 2
    assert series.times == (0, 2)
    assert calls == []
    result = series[-1]
    result.points[:] = 50
    assert calls == [1]
    assert mesh.bounds.x_max == 0.5
