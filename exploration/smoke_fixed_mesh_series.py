"""Run NumPy logic against a minimal mesh double, without PyVista or pytest.

These checks do not verify PyVista array assignment or VTK buffer ownership.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class Mesh:
    """Model only the dataset operations used by the experiment."""

    def __init__(self):
        """Create distinct point and cell counts and a static point field."""
        self.n_points = 4
        self.n_cells = 2
        self.points = np.arange(12).reshape(4, 3)
        self.point_data = {'static': np.arange(4)}
        self.cell_data = {}

    def copy(self, *, deep):
        """Require the prototype to request detached storage."""
        assert deep
        return copy.deepcopy(self)


def check_raises(callback):
    """Require malformed input to raise ValueError."""
    try:
        callback()
    except ValueError:
        return
    msg = 'Expected ValueError'
    raise AssertionError(msg)


def main():
    """Exercise ownership, scalar/vector selection, and rejected inputs."""
    path = Path(__file__).with_name('fixed_mesh_series.py')
    spec = importlib.util.spec_from_file_location('fixed_mesh_experiment', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'pyvista': SimpleNamespace(DataSet=Mesh)}):
        spec.loader.exec_module(module)
    series_type = module.FixedMeshSeries
    mesh = Mesh()
    points = mesh.points.copy()
    values = np.stack([np.zeros(4), np.full(4, 8)])
    vectors = np.stack([np.zeros((2, 3)), np.full((2, 3), 4)])
    times = np.array([1, 5])
    series = series_type(
        mesh, times=times, point_data={'value': values}, cell_data={'value': vectors}
    )
    values[:] = -1
    times[:] = -1
    mesh.points[:] = -1
    frame = series.at_time(2, method='linear')
    np.testing.assert_array_equal(frame.point_data['value'], 2)
    np.testing.assert_array_equal(frame.cell_data['value'], 1)
    np.testing.assert_array_equal(frame.points, points)
    np.testing.assert_array_equal(frame.point_data['static'], np.arange(4))
    frame.point_data['value'][:] = -1
    frame.points[:] = -1
    np.testing.assert_array_equal(series.at_time(5).point_data['value'], 8)
    np.testing.assert_array_equal(series.at_time(1).points, points)
    returned_times = series.times
    returned_times[:] = -1
    np.testing.assert_array_equal(series.times, [1, 5])
    for invalid in [[], [0, 0], [1, 0], [0, np.nan], [[0, 1]]]:
        check_raises(lambda invalid=invalid: series_type(Mesh(), times=invalid))
    for invalid in [np.zeros((2, 3)), np.zeros((2, 4, 2)), np.full((2, 4), 'x')]:
        check_raises(
            lambda invalid=invalid: series_type(Mesh(), times=[0, 1], point_data={'x': invalid})
        )
    for query, method in [(2, 'exact'), (0, 'linear'), (np.nan, 'exact'), (1, 'bad')]:
        check_raises(lambda query=query, method=method: series.at_time(query, method=method))
    print('NumPy stand-in checks passed; see test_real_integration.py for PyVista coverage.')


if __name__ == '__main__':
    main()
