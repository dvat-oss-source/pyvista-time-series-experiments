"""Behavioral checks for the temporal-array experiment."""

import numpy as np
import pytest
import pyvista as pv

from fixed_mesh_series import FixedMeshSeries


def test_scalar_vector_associations_and_interpolation():
    """Interpolate irregular samples and preserve static arrays and geometry."""
    mesh = pv.Plane(i_resolution=2, j_resolution=1)
    mesh.point_data['static'] = np.arange(mesh.n_points)
    scalars = np.stack([np.zeros(mesh.n_points), np.full(mesh.n_points, 8)])
    vectors = np.stack([np.zeros((mesh.n_cells, 3)), np.full((mesh.n_cells, 3), 4)])
    series = FixedMeshSeries(
        mesh, times=[1, 5], point_data={'value': scalars}, cell_data={'value': vectors}
    )
    frame = series.at_time(2, method='linear')
    np.testing.assert_array_equal(frame.point_data['value'], 2)
    np.testing.assert_array_equal(frame.cell_data['value'], 1)
    np.testing.assert_array_equal(frame.point_data['static'], mesh.point_data['static'])
    np.testing.assert_array_equal(frame.points, mesh.points)
    np.testing.assert_array_equal(series.at_time(5).point_data['value'], 8)


def test_ownership():
    """Source mutations and frame mutations cannot change subsequent frames."""
    mesh = pv.Plane(i_resolution=1, j_resolution=1)
    points = mesh.points.copy()
    data = np.arange(8).reshape(2, 4)
    times = np.array([0.0, 1.0])
    series = FixedMeshSeries(mesh, times=times, point_data={'value': data})
    data[:] = -1
    times[:] = -1
    mesh.points[:] = -1
    frame = series.at_time(0)
    frame.point_data['value'][:] = -1
    frame.points[:] = -1
    np.testing.assert_array_equal(series.at_time(0).point_data['value'], np.arange(4))
    np.testing.assert_array_equal(series.at_time(0).points, points)
    returned_times = series.times
    returned_times[:] = -1
    np.testing.assert_array_equal(series.times, [0, 1])


@pytest.mark.parametrize('times', [[], [0, 0], [1, 0], [0, np.nan], [[0, 1]]])
def test_invalid_times(times):
    """Reject ambiguous or unordered coordinates."""
    with pytest.raises(ValueError, match='times must'):
        FixedMeshSeries(pv.Plane(), times=times)


@pytest.mark.parametrize('shape', [(2, 3), (4, 2), (2, 4, 2), (2, 4, 3, 1)])
def test_invalid_shapes(shape):
    """Reject incorrect tuple counts and component layouts."""
    with pytest.raises(ValueError, match='must be real numeric with shape'):
        FixedMeshSeries(
            pv.Plane(i_resolution=1, j_resolution=1),
            times=[0, 1],
            point_data={'value': np.zeros(shape)},
        )


@pytest.mark.parametrize('time,method', [(0.5, 'exact'), (-1, 'linear'), (np.nan, 'exact')])
def test_invalid_selection(time, method):
    """Reject missing exact samples and extrapolation."""
    series = FixedMeshSeries(pv.Plane(), times=[0, 1])
    with pytest.raises(ValueError, match='time must'):
        series.at_time(time, method=method)
