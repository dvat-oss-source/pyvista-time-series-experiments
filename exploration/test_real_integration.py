"""Exercise the temporal prototype with real datasets and file-backed readers."""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

from fixed_mesh_series import FixedMeshSeries


@pytest.mark.parametrize('dtype', [np.int32, np.float32, np.float64])
def test_scalar_and_vector_interpolation_for_both_associations(dtype):
    """Interpolate each component using physical time with no geometry changes."""
    mesh = pv.Plane(i_resolution=2, j_resolution=2)
    point = np.arange(3 * mesh.n_points * 3).reshape(3, mesh.n_points, 3).astype(dtype)
    cell = np.arange(3 * mesh.n_cells).reshape(3, mesh.n_cells).astype(dtype)
    series = FixedMeshSeries(
        mesh, times=[0, 1, 5], point_data={'vector': point}, cell_data={'scalar': cell}
    )
    frame = series.at_time(2, method='linear')
    np.testing.assert_allclose(frame.point_data['vector'], 0.75 * point[1] + 0.25 * point[2])
    np.testing.assert_allclose(frame.cell_data['scalar'], 0.75 * cell[1] + 0.25 * cell[2])
    np.testing.assert_array_equal(frame.points, mesh.points)
    np.testing.assert_array_equal(frame.faces, mesh.faces)
    np.testing.assert_array_equal(series.at_time(5).point_data['vector'], point[2])


@pytest.mark.parametrize('dtype', [bool, complex, str, object])
def test_reject_nonreal_field_types(dtype):
    """Reject field types without the prototype's arithmetic semantics."""
    mesh = pv.Cube()
    with pytest.raises(ValueError, match='real numeric'):
        FixedMeshSeries(
            mesh, times=[0], point_data={'bad': np.ones((1, mesh.n_points), dtype=dtype)}
        )


def test_single_sample_and_static_mesh():
    """Allow one timestamp and a dataset without temporal fields."""
    mesh = pv.Cube()
    series = FixedMeshSeries(mesh, times=[3])
    assert series.at_time(3, method='linear') == mesh
    with pytest.raises(ValueError, match='within the sampled interval'):
        series.at_time(4, method='linear')


def test_invalid_mesh_and_method():
    """Reject invalid objects and unknown selection policies."""
    with pytest.raises(TypeError, match='DataSet'):
        FixedMeshSeries(object(), times=[0])
    series = FixedMeshSeries(pv.Cube(), times=[0])
    with pytest.raises(ValueError, match='method must'):
        series.at_time(0, method='nearest')
