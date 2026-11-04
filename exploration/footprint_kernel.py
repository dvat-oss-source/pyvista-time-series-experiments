"""In-kernel half of ``footprint_tornado.py``: counters, marks and arm factors.

Loaded by path inside the notebook kernel; never imported by the notebook itself.
Every factor is applied at runtime by monkeypatching or wrapping; the notebook,
``pyvista/plotting/time_series.py`` and the cache are not edited.
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import sys
import time
import weakref

import numpy as np

MIB = 2**20
FACTORS = (
    'baseline',
    'u16',
    'u8',
    'crop',
    'inplace',
    'nooverlap',
    'gpu',
    'fb',
    'relief',
    'madvise',
    'mimalloc',
    'jemalloc',
    'lazyimport',
)
# Packages the lazy finder defers (see sitecustomize written by the supervisor).
# Only these stay deferred through this workload: IPython's own boot touches jedi,
# parso, prompt_toolkit and pygments; trame/pyvista touch aiohttp, ipywidgets,
# pooch, scooby, tqdm, xxhash; matplotlib breaks under LazyLoader.
LAZY_MODULES = ('requests', 'filelock')

# --------------------------------------------------------------------------- counters

_libproc = ctypes.CDLL('/usr/lib/libproc.dylib')
_libc = ctypes.CDLL(None)


class RusageInfoV4(ctypes.Structure):
    """``struct rusage_info_v4`` from <sys/resource.h>."""

    _fields_ = [
        ('ri_uuid', ctypes.c_uint8 * 16),
        ('ri_user_time', ctypes.c_uint64),
        ('ri_system_time', ctypes.c_uint64),
        ('ri_pkg_idle_wkups', ctypes.c_uint64),
        ('ri_interrupt_wkups', ctypes.c_uint64),
        ('ri_pageins', ctypes.c_uint64),
        ('ri_wired_size', ctypes.c_uint64),
        ('ri_resident_size', ctypes.c_uint64),
        ('ri_phys_footprint', ctypes.c_uint64),
        ('ri_proc_start_abstime', ctypes.c_uint64),
        ('ri_proc_exit_abstime', ctypes.c_uint64),
        ('ri_child_user_time', ctypes.c_uint64),
        ('ri_child_system_time', ctypes.c_uint64),
        ('ri_child_pkg_idle_wkups', ctypes.c_uint64),
        ('ri_child_interrupt_wkups', ctypes.c_uint64),
        ('ri_child_pageins', ctypes.c_uint64),
        ('ri_child_elapsed_abstime', ctypes.c_uint64),
        ('ri_diskio_bytesread', ctypes.c_uint64),
        ('ri_diskio_byteswritten', ctypes.c_uint64),
        ('ri_cpu_time_qos', ctypes.c_uint64 * 7),
        ('ri_billed_system_time', ctypes.c_uint64),
        ('ri_serviced_system_time', ctypes.c_uint64),
        ('ri_logical_writes', ctypes.c_uint64),
        ('ri_lifetime_max_phys_footprint', ctypes.c_uint64),
        ('ri_instructions', ctypes.c_uint64),
        ('ri_cycles', ctypes.c_uint64),
        ('ri_billed_energy', ctypes.c_uint64),
        ('ri_serviced_energy', ctypes.c_uint64),
        ('ri_interval_max_phys_footprint', ctypes.c_uint64),
        ('ri_runnable_time', ctypes.c_uint64),
    ]


RUSAGE_INFO_V4 = 4


def rusage(pid):
    """Return (phys_footprint, resident_size, lifetime_max_phys_footprint) in bytes, or None."""
    info = RusageInfoV4()
    if _libproc.proc_pid_rusage(int(pid), RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None
    return info.ri_phys_footprint, info.ri_resident_size, info.ri_lifetime_max_phys_footprint


class TaskVmInfo(ctypes.Structure):
    """``task_vm_info_data_t`` through rev1 (``phys_footprint``); mach packs to 4."""

    _pack_ = 4
    _fields_ = [
        ('virtual_size', ctypes.c_uint64),
        ('region_count', ctypes.c_int32),
        ('page_size', ctypes.c_int32),
        ('resident_size', ctypes.c_uint64),
        ('resident_size_peak', ctypes.c_uint64),
        ('device', ctypes.c_uint64),
        ('device_peak', ctypes.c_uint64),
        ('internal', ctypes.c_uint64),
        ('internal_peak', ctypes.c_uint64),
        ('external', ctypes.c_uint64),
        ('external_peak', ctypes.c_uint64),
        ('reusable', ctypes.c_uint64),
        ('reusable_peak', ctypes.c_uint64),
        ('purgeable_volatile_pmap', ctypes.c_uint64),
        ('purgeable_volatile_resident', ctypes.c_uint64),
        ('purgeable_volatile_virtual', ctypes.c_uint64),
        ('compressed', ctypes.c_uint64),
        ('compressed_peak', ctypes.c_uint64),
        ('compressed_lifetime', ctypes.c_uint64),
        ('phys_footprint', ctypes.c_uint64),
    ]


TASK_VM_INFO = 22
_libc.mach_task_self.restype = ctypes.c_uint32
_libc.task_info.argtypes = [
    ctypes.c_uint32,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint32),
]


def task_vm_info():
    """Return this task's TASK_VM_INFO (rev1) or raise."""
    info = TaskVmInfo()
    count = ctypes.c_uint32(ctypes.sizeof(TaskVmInfo) // 4)
    status = _libc.task_info(
        _libc.mach_task_self(), TASK_VM_INFO, ctypes.byref(info), ctypes.byref(count)
    )
    if status != 0:
        msg = f'task_info(TASK_VM_INFO) failed: {status}'
        raise OSError(msg)
    return info


def cross_check(repeats=50):
    """Compare proc_pid_rusage(self) with task_info(TASK_VM_INFO) back to back."""
    rows = []
    for _ in range(repeats):
        before = rusage(os.getpid())[0]
        vm = task_vm_info().phys_footprint
        after = rusage(os.getpid())[0]
        rows.append((before, vm, after))
    exact = sum(before == vm == after for before, vm, after in rows)
    bracketed = sum(min(b, a) <= vm <= max(b, a) for b, vm, a in rows)
    return dict(
        repeats=repeats,
        exact_equal=exact,
        bracketed=bracketed,
        max_abs_difference_bytes=max(max(abs(b - vm), abs(a - vm)) for b, vm, a in rows),
        example=dict(zip(('rusage_before', 'task_vm_info', 'rusage_after'), rows[0], strict=True)),
    )


# --------------------------------------------------------------------------- marks


class Recorder:
    """Write phase marks the supervisor reads, and request snapshots from it."""

    def __init__(self, output):
        """Write to the run folder's ``phases.jsonl``."""
        self.output = Path(output)
        self.path = self.output / 'phases.jsonl'
        self.plotter = None

    def mark(self, phase, *, kind='phase', snapshot=None, **extra):
        """Record a phase boundary (or an event) with in-kernel counters."""
        footprint, rss, lifetime = rusage(os.getpid())
        record = dict(
            phase=phase,
            kind=kind,
            time=time.time(),
            pid=os.getpid(),
            footprint=footprint,
            task_vm_footprint=task_vm_info().phys_footprint,
            rss=rss,
            lifetime_max_footprint=lifetime,
            snapshot=snapshot,
            **extra,
        )
        pl = self.plotter() if self.plotter is not None else None
        if pl is not None and getattr(pl, 'ren_win', None) is not None:
            record['window_size'] = list(pl.window_size)
        with self.path.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        return record

    async def wait_snapshot(self, name, timeout=300.0):
        """Block this cell until the supervisor finished snapshot ``name``."""
        done = self.output / 'snapshots' / f'{name}.done'
        deadline = time.monotonic() + timeout
        while not done.exists():
            if time.monotonic() > deadline:
                msg = f'Supervisor did not finish snapshot {name!r}.'
                raise TimeoutError(msg)
            await asyncio.sleep(0.05)


def allocator_report():
    """Name the malloc zone serving small and large blocks and list allocator dylibs."""
    _libc.malloc.restype = ctypes.c_void_p
    _libc.malloc.argtypes = [ctypes.c_size_t]
    _libc.free.argtypes = [ctypes.c_void_p]
    _libc.malloc_zone_from_ptr.restype = ctypes.c_void_p
    _libc.malloc_zone_from_ptr.argtypes = [ctypes.c_void_p]
    _libc.malloc_get_zone_name.restype = ctypes.c_char_p
    _libc.malloc_get_zone_name.argtypes = [ctypes.c_void_p]

    class DlInfo(ctypes.Structure):
        _fields_ = [
            ('dli_fname', ctypes.c_char_p),
            ('dli_fbase', ctypes.c_void_p),
            ('dli_sname', ctypes.c_char_p),
            ('dli_saddr', ctypes.c_void_p),
        ]

    _libc.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    zones = {}
    for label, size in (('small', 64), ('large', 64 * MIB)):
        pointer = _libc.malloc(size)
        zone = _libc.malloc_zone_from_ptr(pointer)
        name = _libc.malloc_get_zone_name(zone) if zone else None
        zones[label] = name.decode() if name else None
        # The zone's ``malloc`` entry (malloc_zone_t: reserved1, reserved2, size, malloc)
        # names the implementing image even when a replacement keeps the default name.
        info = DlInfo()
        function = ctypes.c_void_p.from_address(zone + 3 * ctypes.sizeof(ctypes.c_void_p)).value
        if zone and _libc.dladdr(function, ctypes.byref(info)):
            zones[label + '_impl'] = info.dli_fname.decode()
        _libc.free(pointer)
    _libc._dyld_image_count.restype = ctypes.c_uint32
    _libc._dyld_get_image_name.restype = ctypes.c_char_p
    _libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    images = [_libc._dyld_get_image_name(i).decode() for i in range(_libc._dyld_image_count())]
    allocators = [p for p in images if 'mimalloc' in p or 'jemalloc' in p]
    # libsystem's virtual default zone forwards to malloc_zones[0]; a replacement such
    # as jemalloc registers itself first, so list every registered zone in order.
    count = ctypes.c_uint()
    table = ctypes.POINTER(ctypes.c_void_p)()
    _libc.malloc_get_all_zones.argtypes = [
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
        ctypes.POINTER(ctypes.c_uint),
    ]
    registered = []
    if _libc.malloc_get_all_zones(0, None, ctypes.byref(table), ctypes.byref(count)) == 0:
        for i in range(count.value):
            zone = table[i]
            info = DlInfo()
            function = ctypes.c_void_p.from_address(zone + 3 * ctypes.sizeof(ctypes.c_void_p))
            _libc.dladdr(function.value, ctypes.byref(info))
            name = _libc.malloc_get_zone_name(zone)
            registered.append(
                [
                    name.decode() if name else None,
                    info.dli_fname.decode() if info.dli_fname else None,
                ]
            )
    return dict(
        registered_zones=registered,
        zone_small=zones['small'],
        zone_large=zones['large'],
        zone_small_impl=zones.get('small_impl'),
        zone_large_impl=zones.get('large_impl'),
        allocator_images=allocators,
    )


# --------------------------------------------------------------------------- frame transforms


def parse_factors(text):
    """Split and validate an ``--arm`` comma list."""
    factors = [f.strip() for f in text.split(',') if f.strip()]
    unknown = [f for f in factors if f not in FACTORS]
    if unknown or not factors:
        msg = f'Unknown factors {unknown}; choose from {FACTORS}.'
        raise ValueError(msg)
    if 'baseline' in factors and len(factors) > 1:
        msg = "'baseline' cannot be combined with other factors."
        raise ValueError(msg)
    if 'u16' in factors and 'u8' in factors:
        msg = "Choose one of 'u16' and 'u8'."
        raise ValueError(msg)
    if 'mimalloc' in factors and 'jemalloc' in factors:
        msg = "Choose one of 'mimalloc' and 'jemalloc'."
        raise ValueError(msg)
    if 'inplace' in factors and 'nooverlap' in factors:
        msg = "'inplace' already replaces selection; it cannot combine with 'nooverlap'."
        raise ValueError(msg)
    return [] if factors == ['baseline'] else factors


def crop_voi(origin, spacing, shape, *, box):
    """Return the inclusive index VOI of one frame inside a world box (identity direction)."""
    voi = []
    for axis in range(3):
        lo = int(np.ceil((box[axis][0] - origin[axis]) / spacing[axis] - 1e-6))
        hi = int(np.floor((box[axis][1] - origin[axis]) / spacing[axis] + 1e-6))
        voi += [max(0, lo), min(shape[axis] - 1, hi)]
    return tuple(voi)


class FrameTransform:
    """Load-time crop and quantization against values fixed from the frame statistics."""

    def __init__(self, factors, stats, *, margin=1):
        """Fix the crop box and quantization range once for all frames."""
        summary = stats['summary']
        self.crop = 'crop' in factors
        self.levels = 65535 if 'u16' in factors else 255 if 'u8' in factors else None
        self.dtype = np.uint16 if 'u16' in factors else np.uint8 if 'u8' in factors else None
        self.range = (float(summary['global_min']), float(summary['global_max']))
        if self.range[0] != 0.0:
            msg = 'Quantization assumes a zero global minimum.'
            raise ValueError(msg)
        spacing = np.array(stats['frames'][0]['spacing'])
        direction = np.array(stats['frames'][0]['direction'])
        if not np.allclose(direction, np.eye(3)):
            msg = 'The crop factor assumes an identity direction matrix.'
            raise ValueError(msg)
        # One fixed world box: the union nonzero box grown by ``margin`` voxels so
        # trilinear interpolation still sees the zero shell around the smoke.
        box = np.array(summary['union_nonzero_bbox_world'])
        self.box = (box + np.outer(spacing, [-margin, margin])).tolist()
        self.margin = margin
        shapes = []
        for record in stats['frames']:
            shape = record['shape']
            if self.crop:
                voi = crop_voi(record['origin'], record['spacing'], shape, box=self.box)
                shape = [voi[1] - voi[0] + 1, voi[3] - voi[2] + 1, voi[5] - voi[4] + 1]
            shapes.append(shape)
        self.max_shape = [int(v) for v in np.max(shapes, axis=0)]
        self.shapes = shapes

    @property
    def scale(self):
        """Density units per stored unit (1.0 for float frames)."""
        return 1.0 if self.levels is None else self.range[1] / self.levels

    def clim(self, clim):
        """Map a density clim into stored units."""
        return tuple(float(v) / self.scale for v in clim)

    def __call__(self, mesh):
        """Return a new frame with the fixed crop and quantization applied."""
        import pyvista as pv

        if self.crop:
            voi = crop_voi(mesh.origin, mesh.spacing, mesh.dimensions, box=self.box)
            mesh = mesh.extract_subset(voi)
        if self.levels is not None:
            values = np.asarray(mesh.point_data['density'])
            out = np.empty(values.shape, dtype=self.dtype)
            step = 1 << 20
            factor = np.float32(self.levels / self.range[1])
            scratch = np.empty(step, dtype=np.float32)
            for start in range(0, values.size, step):
                chunk = values[start : start + step]
                part = scratch[: chunk.size]
                np.multiply(chunk, factor, out=part)
                np.rint(part, out=part)
                np.clip(part, 0, self.levels, out=part)
                out[start : start + chunk.size] = part
            quantized = pv.ImageData(
                dimensions=mesh.dimensions,
                spacing=mesh.spacing,
                origin=mesh.origin,
                direction_matrix=mesh.direction_matrix,
            )
            quantized.point_data['density'] = out
            for name in mesh.field_data.keys():
                quantized.field_data[name] = mesh.field_data[name]
            mesh = quantized
        mesh.set_active_scalars('density')
        return mesh


# --------------------------------------------------------------------------- selection patches


def _placeholder(dtype):
    """Return a tiny all-zero volume used while no frame is attached."""
    import pyvista as pv

    mesh = pv.ImageData(dimensions=(2, 2, 2))
    mesh.point_data['density'] = np.zeros(8, dtype=dtype)
    mesh.set_active_scalars('density')
    return mesh


def make_select_nooverlap():
    """Detach and drop the displayed frame before loading the next one.

    Experiment only: it breaks the documented transactional failed-load
    behavior. With the smart mapper an internal shallow copy keeps the old
    buffer until the next render; combine with ``gpu`` to remove that.
    """

    def _select(self, index):
        pl = self._active()
        if index == self._index:
            return
        mapper = self.actor.mapper
        old = self._frame
        placeholder = _placeholder(old.active_scalars.dtype)
        mapper.dataset = placeholder
        if pl.mesh is old:
            pl.mesh = placeholder
        self._frame = placeholder
        del old
        frame = self._load(index)
        mapper.dataset = frame
        mapper.scalar_range = self.clim
        if pl.mesh is placeholder:
            pl.mesh = frame
        self._frame = frame
        self._index = index
        if self._render:
            pl.render()

    return _select


def make_select_inplace(max_shape):
    """Copy every frame into one persistent, zero-padded image; mark it modified.

    Frames have 97 distinct shapes, so the persistent image has the largest
    (post-crop) shape and each frame fills its low corner; the padding is zero,
    so the extra voxels are fully transparent.
    """
    import pyvista as pv

    def _select(self, index):
        pl = self._active()
        if index == self._index:
            return
        frame = self._load(index)
        target = getattr(self, '_fp_persistent', None)
        source = frame.point_data[self._scalars]
        if target is None or target.point_data[self._scalars].dtype != source.dtype:
            target = pv.ImageData(dimensions=max_shape)
            target.point_data[self._scalars] = np.zeros(int(np.prod(max_shape)), source.dtype)
            target.set_active_scalars(self._scalars, preference=self._preference)
            self._fp_persistent = target
        shape = frame.dimensions
        if any(s > m for s, m in zip(shape, max_shape, strict=True)):
            msg = f'Frame {index} shape {shape} exceeds the persistent shape {max_shape}.'
            raise ValueError(msg)
        destination = np.asarray(target.point_data[self._scalars]).reshape(max_shape[::-1])
        destination[...] = 0
        destination[: shape[2], : shape[1], : shape[0]] = np.asarray(source).reshape(shape[::-1])
        target.origin = frame.origin
        target.spacing = frame.spacing
        target.direction_matrix = frame.direction_matrix
        target.GetPointData().GetScalars().Modified()
        target.Modified()
        del frame, source, destination
        mapper = self.actor.mapper
        old = self._frame
        if mapper.dataset is not target:
            mapper.dataset = target
            mapper.scalar_range = self.clim
            if pl.mesh is old:
                pl.mesh = target
            self._frame = target
        self._index = index
        if self._render:
            pl.render()

    return _select


def pressure_relief():
    """Ask every malloc zone to return free pages; return bytes released."""
    _libc.malloc_zone_pressure_relief.restype = ctypes.c_size_t
    _libc.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return int(_libc.malloc_zone_pressure_relief(None, 0))


class MadvisePool:
    """mmap-backed decode buffers, MADV_FREE_REUSABLE while pooled, MADV_FREE_REUSE on reuse.

    Replaces the decoder's large destinations (``pyvista_zstd`` ``_Segments.allocate``)
    so frame arrays never live in malloc's large-allocation cache.
    """

    threshold = MIB
    granule = 32 * MIB

    def __init__(self, keep=3):
        """Retain at most ``keep`` idle mappings."""
        self.keep = keep
        self.idle = []
        self.stats = dict(mapped=0, reused=0, unmapped=0, reusable_calls=0)

    def _release(self, mapping):
        mapping.madvise(mmap.MADV_FREE_REUSABLE)
        self.stats['reusable_calls'] += 1
        if len(self.idle) < self.keep:
            self.idle.append(mapping)
        else:
            mapping.close()
            self.stats['unmapped'] += 1

    def empty(self, shape, dtype):
        """Return a C-contiguous writable array on a pooled mapping."""
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        mapping = None
        for candidate in self.idle:
            if len(candidate) >= nbytes:
                mapping = candidate
                self.idle.remove(candidate)
                mapping.madvise(mmap.MADV_FREE_REUSE)
                self.stats['reused'] += 1
                break
        if mapping is None:
            # Round up so every decoded frame (20-24 MiB) fits any idle mapping;
            # untouched tail pages cost no footprint.
            mapping = mmap.mmap(-1, -(-max(nbytes, 1) // self.granule) * self.granule)
            self.stats['mapped'] += 1
        owner = _MappingOwner(mapping)
        weakref.finalize(owner, self._release, mapping)
        return np.frombuffer(owner, dtype=dtype, count=int(np.prod(shape))).reshape(shape)


class _MappingOwner:
    """Buffer exporter whose lifetime marks when the decoded frame is gone."""

    def __init__(self, mapping):
        self.mapping = mapping

    def __buffer__(self, flags):
        return memoryview(self.mapping)


def install_madvise():
    """Route large decoder destinations through a :class:`MadvisePool`."""
    from pyvista_zstd import pyvista_zstd as module

    pool = MadvisePool()
    original = module._Segments.allocate

    def allocate(self, name, shape, dtype):  # noqa: PLR0917  (pyvista_zstd's signature)
        if int(np.prod(shape)) * np.dtype(dtype).itemsize >= pool.threshold:
            return pool.empty(shape, dtype)
        return original(self, name, shape, dtype)

    module._Segments.allocate = allocate
    return pool


def apply_pre(factors):
    """Apply factors that must exist before the notebook runs; return their state."""
    import pyvista as pv

    state = {}
    if 'fb' in factors:
        pv.global_theme.multi_samples = 0
        pv.global_theme.anti_aliasing = None
    if 'gpu' in factors:
        original_add_volume = pv.Plotter.add_volume

        def add_volume(self, *args, **kwargs):
            kwargs['mapper'] = 'gpu'
            return original_add_volume(self, *args, **kwargs)

        pv.Plotter.add_volume = add_volume
    select = pv.TimeSeriesPlayer._select
    if 'nooverlap' in factors:
        select = make_select_nooverlap()
    if 'relief' in factors:
        inner = select
        released = state.setdefault('relief_bytes', [])

        def select_relief(self, index):
            # Before loading: releases what the previous render freed.
            released.append(pressure_relief())
            inner(self, index)
            released.append(pressure_relief())

        select_relief.released = released
        select = select_relief
    pv.TimeSeriesPlayer._select = select
    if 'madvise' in factors:
        state['madvise_pool'] = install_madvise()
    return state


def apply_inplace(transform):
    """Install in-place selection after the transform fixes the persistent shape."""
    import pyvista as pv

    current = pv.TimeSeriesPlayer._select
    replacement = make_select_inplace(transform.max_shape)
    if current.__name__ == 'select_relief':
        # Keep relief wrapping around the in-place body.
        released = current.released

        def select(self, index):
            released.append(pressure_relief())
            replacement(self, index)
            released.append(pressure_relief())

        pv.TimeSeriesPlayer._select = select
    else:
        pv.TimeSeriesPlayer._select = replacement


def fix_scalar_bar(pl, density_clim):
    """Label a quantized volume's scalar bar in density units again."""
    import pyvista as pv

    for bar in pl.scalar_bars.values():
        lut = pv.LookupTable()
        lut.DeepCopy(bar.GetLookupTable())
        lut.scalar_range = tuple(density_clim)
        bar.SetLookupTable(lut)


# --------------------------------------------------------------------------- numerical records


def displayed_record(index, dataset, *, scale, original=None):
    """Hash the displayed scalars (raw and canonical) and measure fidelity vs ``original``."""
    values = dataset.active_scalars
    shape = tuple(int(v) for v in dataset.dimensions)
    geometry = dict(
        shape=list(shape),
        origin=[float(v) for v in dataset.origin],
        spacing=[float(v) for v in dataset.spacing],
        direction=np.asarray(dataset.direction_matrix).tolist(),
    )
    raw = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode())
    raw.update(str(values.dtype).encode())
    raw.update(memoryview(np.ascontiguousarray(values)).cast('B'))
    volume = np.asarray(values).reshape(shape[::-1])
    record = dict(index=index, dtype=str(values.dtype), raw_digest=raw.hexdigest(), **geometry)
    nonzero = volume != 0
    canonical = hashlib.sha256()
    if nonzero.any():
        box = []
        for axis in (2, 1, 0):
            other = tuple(a for a in range(3) if a != axis)
            hit = np.flatnonzero(nonzero.any(axis=other))
            box.append((int(hit[0]), int(hit[-1])))
        lo = np.array([b[0] for b in box], dtype=float)
        world = (
            np.asarray(dataset.origin)
            + (lo * np.asarray(dataset.spacing)) @ np.asarray(dataset.direction_matrix).T
        )
        block = volume[
            box[2][0] : box[2][1] + 1, box[1][0] : box[1][1] + 1, box[0][0] : box[0][1] + 1
        ]
        dequantized = np.ascontiguousarray(block, dtype=np.float32) * np.float32(scale)
        canonical.update(
            json.dumps(
                dict(
                    origin=[round(float(v), 6) for v in world],
                    spacing=geometry['spacing'],
                    shape=list(block.shape),
                ),
                sort_keys=True,
            ).encode()
        )
        canonical.update(memoryview(dequantized).cast('B'))
        record['nonzero_bbox_index'] = [list(b) for b in box]
        record['nonzero'] = int(np.count_nonzero(nonzero))
    record['canonical_digest'] = canonical.hexdigest()
    if original is not None:
        # Align the displayed extent with the original frame on its own lattice.
        reference = np.zeros(shape[::-1], dtype=np.float32)
        offset = np.rint(
            (np.asarray(dataset.origin) - np.asarray(original.origin))
            / np.asarray(original.spacing)
        ).astype(int)
        source = np.asarray(original.active_scalars).reshape(tuple(original.dimensions)[::-1])
        lo = np.maximum(offset, 0)
        hi = np.minimum(offset + np.array(shape), np.array(original.dimensions))
        if np.all(hi > lo):
            d_lo = lo - offset
            d_hi = hi - offset
            reference[d_lo[2] : d_hi[2], d_lo[1] : d_hi[1], d_lo[0] : d_hi[0]] = source[
                lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]
            ]
        shown = volume.astype(np.float32) * np.float32(scale)
        error = np.abs(shown - reference)
        record['max_abs_error'] = float(error.max())
        record['rmse'] = float(np.sqrt(np.mean(np.square(error, dtype=np.float64))))
        # Mass the display cannot show: original nonzero voxels outside the displayed extent.
        record['original_sum'] = float(source.sum(dtype=np.float64))
        record['displayed_sum'] = float(shown.sum(dtype=np.float64))
        record['nonzero_lost'] = int(np.count_nonzero((reference != 0) & (shown == 0)))
        record['original_nonzero'] = int(np.count_nonzero(source))
    return record


def lazy_status():
    """Report which deferred packages were actually materialized."""
    status = {}
    for name in LAZY_MODULES:
        module = sys.modules.get(name)
        status[name] = (
            'absent'
            if module is None
            else 'lazy'
            if type(module).__name__ == '_LazyModule'
            else 'loaded'
        )
    return status
