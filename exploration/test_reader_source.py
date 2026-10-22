"""Check saved-frame fidelity, ownership, and controller compatibility."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import pyvista as pv

from reader_source import ReaderSource
from time_controller import TimeController


@pytest.fixture
def result_path(tmp_path):
    """Write irregular saved samples with changing coordinates and topology."""
    entries = []
    for i, time in enumerate((0.0, 0.2, 1.3)):
        mesh = pv.Plane(i_resolution=i + 1, j_resolution=2).cast_to_unstructured_grid()
        mesh.points = mesh.points.astype(np.float64)
        mesh.points[:, 2] = time
        mesh.point_data['value'] = np.arange(mesh.n_points, dtype=np.float64) + time
        mesh.cell_data['material'] = np.arange(mesh.n_cells, dtype=np.int64)
        mesh.field_data['solver_time'] = [time]
        mesh.save(tmp_path / f'{i}.vtu')
        entries.append(f'<DataSet timestep="{time}" part="0" file="{i}.vtu"/>')
    path = tmp_path / 'frames.pvd'
    path.write_text(
        '<VTKFile type="Collection"><Collection>' + ''.join(entries) + '</Collection></VTKFile>'
    )
    return path


def assert_frame_equal(actual, expected):
    """Compare geometry, topology, associations, precision, and all fields."""
    assert type(actual) is type(expected)
    a, b = actual.cast_to_unstructured_grid(), expected.cast_to_unstructured_grid()
    for name in ('points', 'cell_connectivity', 'cell_offsets', 'celltypes'):
        left, right = getattr(a, name), getattr(b, name)
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    for association in ('point_data', 'cell_data', 'field_data'):
        left, right = getattr(actual, association), getattr(expected, association)
        assert set(left) == set(right)
        for name in left:
            assert left[name].dtype == right[name].dtype
            np.testing.assert_array_equal(left[name], right[name])


def test_exact_random_backward_and_owned_frames(result_path):
    """Compare seeks directly with independent original-file reads."""
    reader = pv.get_reader(result_path)
    source = ReaderSource(reader, scalars='value', block=0)
    assert source.times == (0.0, 0.2, 1.3)
    retained = source.frame(0)
    for index in (2, 0, 1, 2, 1, 0):
        actual = source.frame(index)
        expected = pv.read(result_path.parent / f'{index}.vtu')
        assert_frame_equal(actual, expected)
        assert reader.active_time_value == 0.0
        assert_frame_equal(retained, pv.read(result_path.parent / '0.vtu'))
    retained.points[:] = -100
    retained['value'][:] = -100
    assert_frame_equal(source.frame(0), pv.read(result_path.parent / '0.vtu'))


def test_explicit_part_and_missing_field_restore_reader(result_path):
    """Reject ambiguous selection and absent fields without changing time."""
    reader = pv.get_reader(result_path)
    reader.set_active_time_point(1)
    for source, message in (
        (ReaderSource(reader, scalars='value'), 'spatial block'),
        (ReaderSource(reader, scalars='missing', block=0), 'no point array'),
    ):
        with pytest.raises(ValueError, match=message):
            source.frame(2)
        assert reader.active_time_value == 0.2


@pytest.mark.parametrize('index', [-1, 3, True, 1.5])
def test_invalid_indices(result_path, index):
    """Reject requests outside the source contract before reading."""
    source = ReaderSource(pv.get_reader(result_path), scalars='value', block=0)
    with pytest.raises((TypeError, IndexError)):
        source.frame(index)


class Scene:
    """Exercise the real controller and mapper without a render window."""

    def add_mesh(self, mesh, **_kwargs):
        """Create one real mapper for the displayed frame."""
        return SimpleNamespace(mapper=pv.DataSetMapper(dataset=mesh))

    def add_slider_widget(self, callback, *_args, **_kwargs):
        """Retain the selection callback."""
        return callback

    def render(self):
        """Leave rendering to the separate desktop integration check."""


def test_controller_uses_source_without_storage_specific_changes(result_path):
    """Verify the displayed mapper against original files after seeking."""
    source = ReaderSource(pv.get_reader(result_path), scalars='value', block=0)
    controller = TimeController(Scene(), source.times, source.frame, scalars='value', clim=(0, 30))
    actor = controller.actor
    for index in (2, 0, 1):
        controller.select_time(source.times[index])
        assert controller.actor is actor
        assert_frame_equal(actor.mapper.dataset, pv.read(result_path.parent / f'{index}.vtu'))
