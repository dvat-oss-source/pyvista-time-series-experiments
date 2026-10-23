"""Check saved-frame fidelity, ownership, and controller compatibility."""

from __future__ import annotations

import gc
from types import SimpleNamespace
import weakref

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
        mesh.cell_data['material'] = np.arange(mesh.n_cells, dtype=np.int64) + 2**60 + i
        mesh.cell_data['value'] = np.arange(mesh.n_cells, dtype=np.float32) - time
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
    if isinstance(actual, pv.PolyData):
        for name in ('faces', 'lines', 'verts', 'strips'):
            left, right = getattr(actual, name), getattr(expected, name)
            assert left.dtype == right.dtype
            np.testing.assert_array_equal(left, right)
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
    indices = [0, 1, 2, 0, *np.random.default_rng(42).integers(0, 3, 12)]
    for index in indices:
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

    def add_mesh(self, mesh, **kwargs):
        """Create one real mapper for the displayed frame."""
        self.mesh = mesh
        self.mapper = pv.DataSetMapper(dataset=mesh)
        self.mapper.set_active_scalars(kwargs['scalars'], preference=kwargs['preference'])
        self.mapper.update()
        return SimpleNamespace(mapper=self.mapper)

    def add_slider_widget(self, callback, *_args, **_kwargs):
        """Retain the selection callback."""
        return callback

    def render(self):
        """Execute the native scalar pipeline; leave rasterization to desktop checks."""
        self.mapper.update()


def test_controller_uses_source_without_storage_specific_changes(result_path):
    """Verify the displayed mapper against original files after seeking."""
    source = ReaderSource(pv.get_reader(result_path), scalars='value', block=0)
    controller = TimeController(Scene(), source.times, source.frame, scalars='value', clim=(0, 30))
    actor = controller.actor
    for index in (2, 0, 1):
        controller.select_time(source.times[index])
        assert controller.actor is actor
        assert_frame_equal(actor.mapper.dataset, pv.read(result_path.parent / f'{index}.vtu'))


@pytest.mark.parametrize('preference', ['point', 'cell'])
def test_association_dtype_changes_and_lazy_filter(result_path, preference):
    """Compare changing filtered topology and selected association with direct reads."""
    source = ReaderSource(
        pv.get_reader(result_path), scalars='value', block=0, preference=preference
    )

    def filtered(index):
        """Apply the spatial operation to only the requested saved frame."""
        return source.frame(index).clip(normal='x', origin=(0, 0, 0))

    ctl = TimeController(
        Scene(), source.times, filtered, scalars='value', clim=(-5, 30), preference=preference
    )
    for index in (2, 0, 1):
        ctl.select_index(np.int64(index))
        expected = pv.read(result_path.parent / f'{index}.vtu').clip(normal='x', origin=(0, 0, 0))
        assert_frame_equal(ctl.actor.mapper.dataset, expected)
        assert ctl.actor.mapper.dataset.active_scalars_info.association.name.lower() == preference


@pytest.mark.parametrize(
    'bad', ['missing', 'vector', 'complex', 'string', 'empty', 'composite', 'pointset', 'error']
)
def test_failed_frame_preserves_display(result_path, bad):
    """Reject incompatible frames before replacing mapper input or controller state."""
    good = pv.read(result_path.parent / '0.vtu')
    broken = pv.read(result_path.parent / '2.vtu')
    del broken.point_data['value']
    if bad == 'vector':
        broken.point_data['value'] = np.ones((broken.n_points, 3))
    elif bad == 'complex':
        broken.point_data['value'] = np.ones(broken.n_points, dtype=complex)
    elif bad == 'string':
        broken.point_data['value'] = ['solid'] * broken.n_points
    elif bad == 'empty':
        broken = pv.PolyData()
        broken.point_data['value'] = np.empty(0)
    elif bad == 'composite':
        broken = pv.MultiBlock([broken])
    elif bad == 'pointset':
        broken = pv.PointSet(good.points)
        broken['value'] = good.point_data['value']

    def load(index):
        """Supply a failing second sample without disturbing the valid first one."""
        if index and bad == 'error':
            msg = 'Read failed.'
            raise OSError(msg)
        return broken if index else good

    ctl = TimeController(Scene(), [0, 1], load, scalars='value', clim=(0, 30))
    displayed = ctl.actor.mapper.dataset
    with pytest.raises((ValueError, TypeError, OSError)):
        ctl.select_time(1)
    assert ctl.index == 0
    assert ctl.actor.mapper.dataset is displayed
    assert ctl._frame is displayed
    assert_frame_equal(displayed, good)


def test_selection_policy_and_no_redundant_load(result_path):
    """Select exact samples by index, and clamp or choose the earlier nearest time."""
    calls = []

    def load(index):
        """Track original-file requests."""
        calls.append(index)
        return pv.read(result_path.parent / f'{index}.vtu')

    ctl = TimeController(Scene(), [0, 0.25, 1], load, scalars='value', clim=(0, 30))
    for time, expected in ((100, 2), (-100, 0), (0.125, 0), (0.25, 1), (0.625, 1)):
        ctl.select_time(time)
        assert ctl.index == expected
    assert calls == [0, 2, 0, 1]
    for index in (-1, 3, True, 1.5):
        with pytest.raises((TypeError, IndexError)):
            ctl.select_index(index)
    for time in (np.nan, np.inf):
        with pytest.raises(ValueError, match='finite'):
            ctl.select_time(time)
    assert calls == [0, 2, 0, 1]


def test_frame_buffers_are_not_retained(result_path):
    """Observe old Python dataset wrappers and NumPy views becoming unreachable."""
    source = ReaderSource(pv.get_reader(result_path), scalars='value', block=0)
    ctl = TimeController(Scene(), source.times, source.frame, scalars='value', clim=(0, 30))
    refs = []
    for index in [2, 0, 1] * 20:
        previous = ctl.actor.mapper.dataset
        arrays = [previous.points, previous.cell_connectivity, previous.point_data['value']]
        buffers = [previous.GetPoints(), previous.GetCells(), arrays[2].VTKObject]
        refs.extend(weakref.ref(obj) for obj in [previous, *arrays])
        ctl.select_index(index)
        del previous, arrays
        gc.collect()
        # Native reference counts include only these diagnostic references after update.
        assert all(buffer.GetReferenceCount() == 1 for buffer in buffers)
    gc.collect()
    assert all(ref() is None for ref in refs)


class ReusingReader:
    """Model a reader reusing native output buffers, including read failures."""

    time_values = (0.0, 0.25)

    def __init__(self):
        """Create one mutable output and an initial time."""
        self.active_time_value = 0.0
        self.mesh = pv.Plane(i_resolution=1, j_resolution=1)
        self.failure = None

    def set_active_time_point(self, index):
        """Optionally simulate a stale timestep selection."""
        if self.failure != 'stale':
            self.active_time_value = self.time_values[index]

    def set_active_time_value(self, value):
        """Restore time and overwrite the shared output as some readers may do."""
        self.active_time_value = value
        self.mesh.points[:, 2] = value

    def read(self):
        """Reuse the same dataset on each request."""
        if self.failure == 'read':
            msg = 'Read failed.'
            raise OSError(msg)
        self.mesh.points[:, 2] = self.active_time_value
        self.mesh['value'] = np.full(self.mesh.n_points, self.active_time_value)
        return self.mesh


def test_reusing_reader_detaches_before_restoration():
    """Catch aliasing even when restoration mutates reader-owned native buffers."""
    reader = ReusingReader()
    source = ReaderSource(reader, scalars='value')
    retained = source.frame(1)
    np.testing.assert_array_equal(retained.points[:, 2], 0.25)
    assert not np.shares_memory(retained.points, reader.mesh.points)
    source.frame(0)
    np.testing.assert_array_equal(retained['value'], 0.25)
    for failure, error in (('read', OSError), ('stale', ValueError)):
        reader.failure = failure
        with pytest.raises(error):
            source.frame(1)
        assert reader.active_time_value == 0.0


@pytest.mark.parametrize('times', [[], [0, 0], [1, 0], [0, np.nan], [0, np.inf]])
def test_invalid_time_metadata(times):
    """Reject ambiguous timestamp metadata before any frame read."""
    reader = SimpleNamespace(time_values=times)
    with pytest.raises(ValueError, match='times'):
        ReaderSource(reader, scalars='value')
    with pytest.raises(ValueError, match='Times'):
        TimeController(Scene(), times, None, scalars='value', clim=(0, 1))


def test_spatial_block_boundaries_and_convenience_path(result_path):
    """Require explicit selection even for one part and reject nested/null parts."""
    ctl = TimeController.from_reader(
        Scene(),
        pv.get_reader(result_path),
        block=0,
        scalars='material',
        preference='cell',
        clim=(0, 2**61),
    )
    ctl.select_index(2)
    assert_frame_equal(ctl.actor.mapper.dataset, pv.read(result_path.parent / '2.vtu'))
    reader = ReusingReader()
    for block in (pv.MultiBlock([reader.mesh]), None):
        reader.read = lambda block=block: pv.MultiBlock([block])
        with pytest.raises(TypeError, match='null or composite'):
            ReaderSource(reader, scalars='value', block=0).frame(1)
        assert reader.active_time_value == 0


def test_changing_display_types_and_scalar_dtypes():
    """Allow ordinary DataSet types and numeric precision to change between samples."""
    frames = [pv.Plane(), pv.ImageData(dimensions=(3, 3, 2)).cast_to_unstructured_grid()]
    frames[0]['value'] = np.arange(frames[0].n_points, dtype=np.int64) + 2**60
    frames[1]['value'] = np.arange(frames[1].n_points, dtype=np.float32) / 8
    ctl = TimeController(Scene(), [0, 1], frames.__getitem__, scalars='value', clim=(0, 2**61))
    for index in (1, 0):
        ctl.select_index(index)
        assert_frame_equal(ctl.actor.mapper.dataset, frames[index])


@pytest.mark.parametrize('block', [10, 'absent'])
def test_invalid_block_restores_reader(result_path, block):
    """Invalid spatial-part keys must leave the original simulation time selected."""
    reader = pv.get_reader(result_path)
    reader.set_active_time_point(1)
    with pytest.raises((IndexError, KeyError)):
        ReaderSource(reader, scalars='value', block=block).frame(2)
    assert reader.active_time_value == 0.2


def test_restore_failure_preserves_display():
    """Surface reader restoration failures before committing a new displayed frame."""
    reader = ReusingReader()
    ctl = TimeController.from_reader(Scene(), reader, scalars='value', clim=(0, 1))
    displayed = ctl.actor.mapper.dataset

    def failed_restore(_time):
        """Model a reader that cannot restore its previous selection."""
        msg = 'Restore failed.'
        raise OSError(msg)

    reader.set_active_time_value = failed_restore
    with pytest.raises(OSError, match='Restore failed'):
        ctl.select_time(1)
    assert ctl.index == 0
    assert ctl.actor.mapper.dataset is displayed


def test_finite_extreme_time_clamps(result_path):
    """Avoid overflowing every distance when clamping outside the saved interval."""
    ctl = TimeController(
        Scene(),
        [-1.7e308, -1.6e308],
        lambda index: pv.read(result_path.parent / f'{index}.vtu'),
        scalars='value',
        clim=(0, 30),
    )
    ctl.select_time(1.7e308)
    assert ctl.index == 1
