"""Measure the tornado notebook's phys_footprint in a fresh kernel, per arm.

Commands (run with the project interpreter from the repository root)::

    python exploration/footprint_tornado.py run OUTPUT [--arm baseline] [--prewarm]
        [--peak-margin-mib 8] [--cycles 3] [--reopens 3]
        [--notebook tornado_animation.ipynb]
        [--frame-stats exploration/tornado_footprint_data/frame-stats.json]
    python exploration/footprint_tornado.py prewarm
    python exploration/footprint_tornado.py compare RUN_A RUN_B

``run`` executes the unmodified notebook through nbclient in a new kernel
launched with this interpreter, drives the notebook's own Trame state callbacks
(all-frame seek, timed playback cycles, close/reopen, then a diagnostic
numerical/image pass in a reopened viewer), and
samples the kernel's ``phys_footprint`` and resident size every 10 ms from this
supervisor through ``proc_pid_rusage(RUSAGE_INFO_V4)``. ``footprint -v`` and
``vmmap -summary`` snapshots are taken mid third cycle, at the workload's running
maximum (re-taken whenever it grows by ``--peak-margin-mib``) and after the final
``close()``. ``--arm`` is a
comma list of factors applied inside the kernel at runtime; see
``footprint_kernel.FACTORS``. ``OUTPUT`` must not exist. The notebook, the
PyVista sources and the data cache are never written.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
KERNEL_MODULE = HERE / 'footprint_kernel.py'
CACHE_ROOT = Path.home() / 'Library' / 'Caches' / 'pyvista_3' / 'tornado_looping'
DEFAULT_STATS = HERE / 'tornado_footprint_data' / 'frame-stats.json'
ALLOCATORS = {
    'mimalloc': '/opt/homebrew/lib/libmimalloc.dylib',
    'jemalloc': '/opt/homebrew/lib/libjemalloc.dylib',
}
KERNEL_NAME = 'fp-tornado'
# Phases inside a timed interval (preparation_s, startup_s, seek_s, reopen_s). Snapshots
# suspend the kernel about 1 s, so none may run in them: the kernel gates before each
# interval and the supervisor starts no snapshot until the next untimed phase.
TIMED_PHASES = ('preparation', 'transform_ready', 'startup', 'seeking', 'reopen_')
SCHEMA = 'tornado-footprint/1'
MIB = 2**20
IMAGE_FRAMES = (0, 32, 64, 96, 127)


def load_kernel_module():
    """Import the in-kernel helpers by path (exploration/ also holds shadowing names)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location('footprint_kernel', KERNEL_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =========================================================================== worker

SETUP_CELL = r"""
import asyncio, gc, importlib.util, inspect, json, os, sys, time, weakref
from pathlib import Path
_fp_cfg = json.loads(Path(%(config)r).read_text())
for _fp_key in ('DYLD_INSERT_LIBRARIES', 'FP_LAZY_MODULES'):
    os.environ.pop(_fp_key, None)
if 'FP_ORIGINAL_PYTHONPATH' in os.environ:
    os.environ['PYTHONPATH'] = os.environ.pop('FP_ORIGINAL_PYTHONPATH')
else:
    os.environ.pop('PYTHONPATH', None)
_fp_spec = importlib.util.spec_from_file_location('footprint_kernel', _fp_cfg['kernel_module'])
_fpk = importlib.util.module_from_spec(_fp_spec)
sys.modules['footprint_kernel'] = _fpk
_fp_spec.loader.exec_module(_fpk)
_fp = _fpk.Recorder(_fp_cfg['output'])
_fp_factors = _fpk.parse_factors(_fp_cfg['arm'])
_fp_cross = _fpk.cross_check()
_fp_alloc = _fpk.allocator_report()
_fp.mark('start', cross_check=_fp_cross, allocator=_fp_alloc, python=sys.executable,
         python_version=sys.version, lazy=_fpk.lazy_status(),
         lazy_finder=any(type(f).__name__ == '_LazyFinder' for f in sys.meta_path))
_fp_state = _fpk.apply_pre(_fp_factors)
import pyvista as _fp_pv
assert Path(_fp_pv.__file__).resolve().is_relative_to(Path(_fp_cfg['repo'])), _fp_pv.__file__
if 'madvise_pool' in _fp_state:
    _fp.probes['madvise'] = _fp_state['madvise_pool'].snapshot
import numpy as _fp_np
from importlib import metadata as _fp_md
_fp.mark('arm_ready', pyvista=_fp_pv.__file__, factors=_fp_factors,
         imported_versions=dict(pyvista=_fp_pv.__version__,
                                vtk='.'.join(map(str, _fp_pv.vtk_version_info)),
                                pyvista_zstd=_fp_md.version('pyvista-zstd'),
                                numpy=_fp_np.__version__, trame=_fp_md.version('trame')))
"""

TRANSFORM_CELL = r"""
_fp_stats = json.loads(Path(_fp_cfg['frame_stats']).read_text())
_fp_transform = _fpk.FrameTransform(_fp_factors, _fp_stats)
_fp_scale = _fp_transform.scale
_fp_original_load_frame = load_frame
_fp_density_clim = tuple(density_clim)
if _fp_transform.crop or _fp_transform.levels:
    def load_frame(index, _inner=_fp_original_load_frame):
        return _fp_transform(_inner(index))
    density_clim = _fp_transform.clim(density_clim)
if 'inplace' in _fp_factors:
    _fpk.apply_inplace(_fp_transform)
_fp.mark('transform_ready', density_clim=list(density_clim), scale=_fp_scale,
         crop_box=_fp_transform.box if _fp_transform.crop else None,
         persistent_shape=_fp_transform.max_shape)
"""

MAKE_CELL = r"""
_fp_original_make = make_tornado_trame
_fp_render_counts = []
_fp.probes['render_events'] = lambda: [dict(count) for count in _fp_render_counts]
async def make_tornado_trame():
    layout, close = await _fp_original_make()
    pl = inspect.getclosurevars(close).nonlocals['pl']
    if _fp_transform.levels:
        _fpk.fix_scalar_bar(pl, _fp_density_clim)
    player = inspect.getclosurevars(close).nonlocals['player']
    expected_size = (1100, 760)
    count = dict(events=0, first_time_before=pl._first_time,
                 expected_window_size=list(expected_size), render_requests=[],
                 event_source='render_window.RenderEvent')
    _fp_render_counts.append(count)
    def rendered(*_args):
        count['events'] += 1
    # VTK emits RenderEvent on the window, not its renderers.
    pl.render_window.AddObserver('RenderEvent', rendered)
    # A browser normally initializes the view. Without one, render() is gated
    # until show(); initialize publicly inside the startup/reopen interval.
    pl.show(auto_close=False, interactive=False, jupyter_backend='none',
            window_size=expected_size)
    count['initial_events'] = count['events']
    count['initial_index'] = player.index
    count['first_time_after'] = pl._first_time
    count['actual_window_size'] = list(pl.window_size)
    count['suppress_rendering'] = pl.suppress_rendering
    if not count['events'] or pl._first_time or tuple(pl.window_size) != expected_size:
        _fp.mark('viewer_initialization_failed', kind='event')
        raise RuntimeError(f'The headless view did not initialize: {count}.')
    original_render = pl.render
    def checked_render(*args, **kwargs):
        '''Require an actual render for every notebook display request.'''
        index = player.index
        before = count['events']
        result = original_render(*args, **kwargs)
        delta = count['events'] - before
        count['render_requests'].append(dict(index=index, events=delta))
        if delta <= 0:
            raise RuntimeError(f'Render request for frame {index} emitted no renderer event.')
        return result
    pl.render = checked_render
    _fp.plotter = weakref.ref(pl)
    return layout, close
def _fp_nonlocal(name):
    return inspect.getclosurevars(close_tornado_trame).nonlocals[name]
async def _fp_show(index):
    previous = _fp_nonlocal('player').index
    before = _fp_render_counts[-1]['events']
    tornado_ui.server.state.frame = index
    tornado_ui.server.state.flush()
    await asyncio.sleep(0)
    player = _fp_nonlocal('player')
    if player.index != index:
        raise RuntimeError(f'Trame seek to {index} displayed {player.index}.')
    if previous != index and _fp_render_counts[-1]['events'] <= before:
        raise RuntimeError(f'Trame seek to {index} did not emit a renderer event.')
    return player
"""

NUMERICAL_CELL = r"""
# Diagnostic pass, run after every metric phase in a reopened viewer: its
# frame-sized temporaries would otherwise inflate the allocator for the trial.
assert frame_count == 128
_fp.mark('diagnostic')
tornado_ui, close_tornado_trame = await make_tornado_trame()
_fp_frames = []
for _fp_index in range(frame_count):
    _fp_player = await _fp_show(_fp_index)
    _fp_original = _fp_original_load_frame(_fp_index)
    _fp_original.set_active_scalars('density')
    _fp_frames.append(_fpk.displayed_record(
        _fp_index, _fp_player.actor.mapper.dataset, scale=_fp_scale, original=_fp_original))
    del _fp_original
    if _fp_index in _fp_cfg['image_frames']:
        _fp_nonlocal('pl').screenshot(str(Path(_fp_cfg['output']) / f'frame-{_fp_index:03d}.png'))
del _fp_player
(Path(_fp_cfg['output']) / 'frames.json').write_text(json.dumps(_fp_frames, indent=1))
await close_tornado_trame()
del tornado_ui, close_tornado_trame
gc.collect()
_fp_final = dict(lazy=_fpk.lazy_status())
if 'madvise_pool' in _fp_state:
    _fp_final['madvise'] = _fp_state['madvise_pool'].stats
if 'relief_bytes' in _fp_state:
    _fp_released = _fp_state['relief_bytes']
    _fp_final['relief'] = dict(calls=len(_fp_released), bytes_total=sum(_fp_released),
                               bytes_max=max(_fp_released, default=0))
_fp.mark('end', **_fp_final)
"""

SEEK_CELL = r"""
await _fp.gate()
_fp.mark('seeking')
_fp_t0 = time.perf_counter()
for _fp_index in [*range(frame_count), 127, 0, 64, 1, 126, 0]:
    await _fp_show(_fp_index)
_fp_seek_seconds = time.perf_counter() - _fp_t0
_fp.mark('seeking_done', seconds=_fp_seek_seconds)
"""

PLAYBACK_CELL = r"""
_fp.mark('playback_%(cycle)d')
_fp_render_start = _fp_render_counts[-1]['events']
_fp_loop = asyncio.get_running_loop()
_fp_start = _fp_loop.time()
tornado_ui.server.state.playing = True
tornado_ui.server.state.flush()
_fp_seen = []
_fp_mid_done = %(cycle)d != 2
_fp_deadline = _fp_start + display_cycle_seconds + 0.25
while _fp_loop.time() < _fp_deadline:
    await asyncio.sleep(min(0.1, max(0, _fp_deadline - _fp_loop.time())))
    _fp_task = _fp_nonlocal('playback_task')
    if _fp_task is not None and _fp_task.done() and not _fp_task.cancelled():
        _fp_task.result()
    _fp_seen.append(int(tornado_ui.server.state.frame))
    if not _fp_mid_done and _fp_loop.time() >= _fp_start + display_cycle_seconds / 2:
        _fp.mark('third_cycle_mid', kind='event', snapshot='third_cycle')
        _fp_mid_done = True
del _fp_task
if len(set(_fp_seen)) < 2:
    raise RuntimeError('The notebook timer did not advance frames.')
if _fp_render_counts[-1]['events'] <= _fp_render_start:
    raise RuntimeError('The notebook timer emitted no renderer events.')
(Path(_fp_cfg['output']) / 'playback-%(cycle)d.json').write_text(json.dumps(_fp_seen))
tornado_ui.server.state.playing = False
tornado_ui.server.state.flush()
gc.collect()
_fp.mark('playback_%(cycle)d_stop', distinct_frames=len(set(_fp_seen)))
"""

REOPEN_CELL = r"""
_fp.mark('close_%(repeat)d')
await close_tornado_trame()
await close_tornado_trame()
gc.collect()
await _fp.gate()
_fp.mark('reopen_%(repeat)d')
tornado_ui, close_tornado_trame = await make_tornado_trame()
gc.collect()
"""

CLOSE_CELL = r"""
_fp.mark('final_close')
await close_tornado_trame()
await close_tornado_trame()
del tornado_ui, close_tornado_trame
gc.collect()
_fp.mark('after_close')
await asyncio.sleep(1.0)
_fp.mark('after_close_snapshot', snapshot='after_close')
await _fp.wait_snapshot('after_close')
"""


def worker(output):
    """Execute the notebook's original cells plus measurement cells in a new kernel."""
    from nbclient import NotebookClient
    import nbformat

    config = json.loads((output / 'config.json').read_text())
    notebook = Path(config['notebook'])
    nb = nbformat.read(notebook, as_version=4)
    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name=KERNEL_NAME,
        resources={'metadata': {'path': str(notebook.parent)}},
    )
    original_count = len(nb.cells)

    def cell(source):
        nb.cells.append(nbformat.v4.new_code_cell(source))
        client.execute_cell(nb.cells[-1], len(nb.cells) - 1)

    try:
        with client.setup_kernel():
            cell(SETUP_CELL % dict(config=str(output / 'config.json')))
            cell("await _fp.gate()\n_fp.mark('preparation')")
            for index in range(original_count):
                source = nb.cells[index].source
                if nb.cells[index].cell_type != 'code':
                    continue
                startup = 'tornado_ui, close_tornado_trame = await make_tornado_trame()' in source
                if startup:
                    cell("await _fp.gate()\n_fp.mark('startup')")
                client.execute_cell(nb.cells[index], index)
                if 'display_files, frame_records = prepare_display_cache()' in source:
                    cell(TRANSFORM_CELL)
                if source.lstrip().startswith('async def make_tornado_trame'):
                    cell(MAKE_CELL)
                if startup:
                    cell("_fp.mark('viewer_ready')")
            cell(SEEK_CELL)
            for cycle in range(config['cycles']):
                cell(PLAYBACK_CELL % dict(cycle=cycle))
            for repeat in range(config['reopens']):
                cell(REOPEN_CELL % dict(repeat=repeat))
            cell(CLOSE_CELL)
            cell(NUMERICAL_CELL)
    finally:
        buffer = io.StringIO()
        nbformat.write(nb, buffer)
        with gzip.open(output / 'executed.ipynb.gz', 'wt') as stream:
            stream.write(buffer.getvalue())


# =========================================================================== supervisor

LAZY_SITECUSTOMIZE = r'''
"""Defer listed packages with importlib.util.LazyLoader (footprint 'lazyimport' arm)."""
import importlib.abc
import importlib.util
import os
import sys

_NAMES = set(filter(None, os.environ.get('FP_LAZY_MODULES', '').split(',')))


class _LazyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] not in _NAMES:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is None or not hasattr(spec.loader, 'exec_module'):
            return spec
        spec.loader = importlib.util.LazyLoader(spec.loader)
        return spec


if _NAMES:
    sys.meta_path.insert(0, _LazyFinder())
'''


def write_kernelspec(output, factors):
    """Create a private kernelspec: this interpreter, plus per-arm launch environment."""
    kernels = output / 'jupyter' / 'kernels' / KERNEL_NAME
    kernels.mkdir(parents=True)
    paths = [str(REPO)]
    env = {}
    if os.environ.get('PYTHONPATH'):
        env['FP_ORIGINAL_PYTHONPATH'] = os.environ['PYTHONPATH']
    if 'lazyimport' in factors:
        lazy = output / 'jupyter' / 'lazy'
        lazy.mkdir()
        (lazy / 'sitecustomize.py').write_text(LAZY_SITECUSTOMIZE)
        paths.insert(0, str(lazy))
        kernel = load_kernel_module()
        env['FP_LAZY_MODULES'] = ','.join(kernel.LAZY_MODULES)
    env['PYTHONPATH'] = os.pathsep.join(paths)
    for name, library in ALLOCATORS.items():
        if name in factors:
            if not Path(library).is_file():
                msg = f'{library} is missing.'
                raise SystemExit(msg)
            env['DYLD_INSERT_LIBRARIES'] = library
    spec = dict(
        argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name='footprint tornado',
        language='python',
        env=env,
    )
    (kernels / 'kernel.json').write_text(json.dumps(spec, indent=2))
    return spec


def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def provenance(notebook):
    """Record exactly what was run: notebook and source hashes, git state, versions."""
    import importlib.metadata

    def git(*command):
        result = subprocess.run(
            [shutil.which('git') or '/usr/bin/git', '-C', str(REPO), *command],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()

    status = git('status', '--porcelain', '--untracked-files=no')
    versions = {}
    for name in ('pyvista', 'vtk', 'pyvista-zstd', 'numpy', 'trame', 'trame-pyvista', 'nbclient'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(
        notebook_sha256=_sha256(notebook),
        time_series_sha256=_sha256(REPO / 'pyvista' / 'plotting' / 'time_series.py'),
        harness_sha256=_sha256(__file__),
        kernel_module_sha256=_sha256(KERNEL_MODULE),
        git_rev=git('rev-parse', 'HEAD'),
        git_dirty=bool(status),
        git_dirty_files=status.splitlines(),
        interpreter=sys.executable,
        interpreter_version=sys.version,
        # Distribution metadata; the kernel also records imported module versions.
        versions=versions,
    )


def prewarm(cache_root):
    """Read every display frame and the manifest once so trials start warm."""
    start = time.perf_counter()
    total = 0
    files = [
        *sorted((cache_root / 'display-256').glob('*.pv')),
        cache_root / 'display-256' / 'frames.json',
    ]
    for path in files:
        with path.open('rb') as stream:
            while chunk := stream.read(8 * MIB):
                total += len(chunk)
    return dict(files=len(files), bytes=total, seconds=time.perf_counter() - start)


FOOTPRINT_TOTAL = re.compile(r'Footprint:\s+([\d.]+)\s*([KMG]?B)')
UNITS = {'B': 1, 'KB': 2**10, 'MB': 2**20, 'GB': 2**30}
_SIZE = r'[\d.]+[KMG]?'
VMMAP_ROW = re.compile(
    rf'^(?P<name>\S.*?)\s+(?P<virtual>{_SIZE})\s+(?P<resident>{_SIZE})\s+(?P<dirty>{_SIZE})'
    rf'\s+(?P<swapped>{_SIZE})\s+(?P<volatile>{_SIZE})\s+(?P<nonvolatile>{_SIZE})'
    rf'\s+(?P<empty>{_SIZE})\s+(?P<regions>\d+)\b'
)
VMMAP_FOOTPRINT = re.compile(r'^Physical footprint(?P<peak> \(peak\))?:\s+(?P<size>[\d.]+[KMG]?)')
FOOTPRINT_ROW = re.compile(
    r'^\s*(?P<dirty>[\d.]+ [KMG]?B)\s+(?P<clean>[\d.]+ [KMG]?B)\s+'
    r'(?P<reclaimable>[\d.]+ [KMG]?B)\s+(?P<regions>\d+)\s+(?P<category>\S.*)$'
)


def _vmmap_bytes(token):
    """Convert a vmmap size token (``76.3M``, ``0K``, ``824``) to bytes."""
    scale = {'K': 2**10, 'M': 2**20, 'G': 2**30}.get(token[-1])
    return float(token[:-1]) * scale if scale else float(token)


def _footprint_bytes(token):
    """Convert a footprint size token (``310 MB``, ``0 B``) to bytes."""
    value, unit = token.split()
    return float(value) * UNITS[unit]


def parse_vmmap_summary(text):
    """Return the region-type table (bytes) and the physical footprint lines."""
    regions = {}
    header = {}
    for line in text.splitlines():
        match = VMMAP_FOOTPRINT.match(line.strip())
        if match:
            header['peak' if match['peak'] else 'current'] = _vmmap_bytes(match['size'])
            continue
        match = VMMAP_ROW.match(line)
        if match and match['name'] not in regions:
            regions[match['name'].strip()] = {
                key: _vmmap_bytes(match[key])
                for key in (
                    'virtual',
                    'resident',
                    'dirty',
                    'swapped',
                    'volatile',
                    'nonvolatile',
                    'empty',
                )
            } | {'regions': int(match['regions'])}
    return dict(physical_footprint=header, regions=regions)


def parse_footprint(text):
    """Return the 40 largest footprint categories: Dirty/Clean/Reclaimable bytes."""
    categories = {}
    for line in text.splitlines():
        match = FOOTPRINT_ROW.match(line)
        if match:
            categories[match['category'].strip()] = dict(
                dirty=_footprint_bytes(match['dirty']),
                clean=_footprint_bytes(match['clean']),
                reclaimable=_footprint_bytes(match['reclaimable']),
                regions=int(match['regions']),
            )
    # The full table lists every dylib; keep the largest rows (full text is in the .gz).
    top = sorted(categories.items(), key=lambda item: -item[1]['dirty'])[:40]
    return dict(top)


class Snapshots:
    """Serialize footprint/vmmap snapshots in background threads."""

    def __init__(self, output, kernel, *, enabled=True):
        """Write ``snapshots/<name>-<tool>.txt.gz`` and ``<name>.json`` / ``.done``."""
        self.folder = output / 'snapshots'
        self.folder.mkdir()
        self.enabled = enabled
        self.kernel = kernel
        self.lock = threading.Lock()
        self.threads = []
        self.records = {}
        self.peak_history = []
        self.peak_busy = False

    def peak_captured(self):
        """Return the footprint captured by the latest finished peak snapshot."""
        return self.peak_history[-1]['footprint_at_start'] if self.peak_history else None

    def active(self):
        """Return whether a snapshot thread is still running."""
        return any(thread.is_alive() for thread in self.threads)

    def take(self, name, pid, trigger):
        """Start one snapshot without blocking the sampler (no-op when disabled)."""
        if not self.enabled:
            # Release a kernel waiting on this snapshot; no tool is launched.
            (self.folder / f'{name}.done').write_text('')
            return
        if name == 'peak':
            # Re-snapshots overwrite the files; each one is at a higher running max.
            if self.peak_busy:
                return
            self.peak_busy = True
        elif name in self.records:
            return
        self.records[name] = dict(name=name, trigger=trigger, pending=True)
        thread = threading.Thread(target=self._run, args=(name, pid, trigger), daemon=True)
        thread.start()
        self.threads.append(thread)

    def _run(self, name, pid, trigger):
        with self.lock:
            record = dict(name=name, trigger=trigger, started=time.time())
            before = self.kernel.rusage(pid)
            record['footprint_at_start'] = before[0] if before else None
            record['rss_at_start'] = before[1] if before else None
            commands = dict(
                footprint=['/usr/bin/footprint', '-p', str(pid), '-v'],
                vmmap=['/usr/bin/vmmap', '-summary', str(pid)],
            )
            if name == 'allocator':
                commands = dict(vmmap_full=['/usr/bin/vmmap', '-w', str(pid)])
            for tool, command in commands.items():
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                text = result.stdout + result.stderr
                path = self.folder / f'{name}-{tool}.txt.gz'
                with gzip.open(path, 'wt') as stream:
                    stream.write(text)
                record[f'{tool}_file'] = str(path.relative_to(self.folder.parent))
                record[f'{tool}_returncode'] = result.returncode
                if tool == 'footprint':
                    record['footprint_categories'] = parse_footprint(text)
                if tool == 'vmmap':
                    record['vmmap'] = parse_vmmap_summary(text)
                if tool == 'footprint':
                    match = FOOTPRINT_TOTAL.search(text)
                    if match:
                        record['footprint_tool_total'] = (
                            float(match.group(1)) * UNITS[match.group(2)]
                        )
                if tool == 'vmmap_full':
                    record['allocator_lines'] = [
                        line
                        for line in text.splitlines()
                        if 'mimalloc' in line or 'jemalloc' in line
                    ][:20]
            after = self.kernel.rusage(pid)
            record['footprint_at_end'] = after[0] if after else None
            record['finished'] = time.time()
            if name == 'peak':
                self.peak_history.append(
                    {
                        k: record[k]
                        for k in ('trigger', 'started', 'finished', 'footprint_at_start')
                    }
                )
                record['history'] = self.peak_history
            self.records[name] = record
            (self.folder / f'{name}.json').write_text(json.dumps(record, indent=2))
            (self.folder / f'{name}.done').write_text('')
            if name == 'peak':
                self.peak_busy = False

    def join(self):
        """Wait for outstanding snapshots."""
        for thread in self.threads:
            thread.join()


def supervise(args):
    """Launch the worker, sample the kernel every 10 ms and summarize."""
    kernel = load_kernel_module()
    factors = kernel.parse_factors(args.arm)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    warm = prewarm(args.cache_root) if args.prewarm else None
    config = dict(
        schema=SCHEMA,
        arm=args.arm,
        factors=factors,
        notebook=str(args.notebook.resolve()),
        frame_stats=str(args.frame_stats.resolve()),
        kernel_module=str(KERNEL_MODULE),
        repo=str(REPO),
        output=str(output),
        cycles=args.cycles,
        reopens=args.reopens,
        peak_margin_mib=args.peak_margin_mib,
        peak_max_snapshots=args.peak_max_snapshots,
        snapshots=args.snapshots,
        provenance=provenance(args.notebook.resolve()),
        image_frames=list(IMAGE_FRAMES),
        sample_interval=args.interval,
        prewarm=warm,
        python=sys.executable,
    )
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    kernelspec = write_kernelspec(output, factors)
    env = dict(os.environ)
    env['JUPYTER_PATH'] = os.pathsep.join(
        filter(None, [str(output / 'jupyter'), os.environ.get('JUPYTER_PATH')])
    )
    env.pop('DYLD_INSERT_LIBRARIES', None)
    command = [sys.executable, str(Path(__file__).resolve()), 'worker', str(output)]
    margin = args.peak_margin_mib * MIB
    # Peak snapshots follow the running maximum of the workload (everything before the
    # `diagnostic` mark): a new one whenever the footprint exceeds the last peak
    # snapshot by `margin`. Inside timed phases the snapshot (which suspends the
    # kernel for about 1 s) is deferred to the next phase boundary.
    # Tracking starts once the viewer exists (boot and preparation are below every
    # later phase) and stops at the diagnostic pass.
    tracking = False
    timed = False
    current = 0
    peak_snapped = None
    peak_deferred = False
    peak_applied = 0
    blocked = False
    gate_pending = None
    snapshots = None
    samples = []
    marks = []
    pid = None
    launched = time.time()
    (output / 'phases.jsonl').touch()
    with (output / 'worker.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        phases_stream = (output / 'phases.jsonl').open()
        pending = ''
        next_tick = time.perf_counter()
        while process.poll() is None:
            pending += phases_stream.read()
            while '\n' in pending:
                line, pending = pending.split('\n', 1)
                record = json.loads(line)
                marks.append(record)
                if pid is None:
                    pid = record['pid']
                    snapshots = Snapshots(output, kernel, enabled=args.snapshots == 'on')
                    if any(name in factors for name in ALLOCATORS):
                        snapshots.take('allocator', pid, 'start')
                if record['kind'] == 'gate':
                    # A timed interval follows: start nothing and release the kernel
                    # once running snapshots finish.
                    blocked = True
                    gate_pending = record['phase']
                if record.get('snapshot'):
                    if blocked or timed:
                        msg = f'Snapshot {record["snapshot"]!r} requested inside a timed interval.'
                        raise RuntimeError(msg)
                    snapshots.take(record['snapshot'], pid, record['phase'])
                if record['kind'] == 'phase':
                    timed = record['phase'].startswith(TIMED_PHASES)
                    if not timed:
                        blocked = False
                    if record['phase'] == 'viewer_ready':
                        tracking = True
                    if record['phase'] == 'diagnostic':
                        tracking = False
                    if peak_deferred and not blocked and not timed and not snapshots.peak_busy:
                        snapshots.take('peak', pid, f'deferred to {record["phase"]}')
                        # Compare later samples against what this snapshot captures.
                        peak_snapped = current
                        peak_deferred = False
            if gate_pending is not None and not snapshots.active():
                (output / 'snapshots' / f'{gate_pending}.ok').write_text('')
                gate_pending = None
            if pid is not None:
                now = time.time()
                values = kernel.rusage(pid)
                if values is not None:
                    samples.append((now, *values))
                    current = values[0]
                    if tracking and snapshots.enabled:
                        if len(snapshots.peak_history) > peak_applied and not snapshots.peak_busy:
                            # The latest snapshot recorded the footprint when its tools
                            # started, which can be below its trigger on an oscillating series.
                            peak_applied = len(snapshots.peak_history)
                            peak_snapped = snapshots.peak_captured()
                        room = len(snapshots.peak_history) < args.peak_max_snapshots
                        if room and (peak_snapped is None or values[0] >= peak_snapped + margin):
                            if timed or blocked:
                                peak_deferred = True
                            elif not snapshots.peak_busy:
                                snapshots.take(
                                    'peak', pid, f'running max {values[0] / MIB:.1f} MiB'
                                )
                                peak_snapped = values[0]
                                peak_deferred = False
            next_tick += args.interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
        returncode = process.wait()
        pending += phases_stream.read()
        marks.extend(json.loads(line) for line in pending.splitlines() if line.strip())
        phases_stream.close()
    if snapshots is not None:
        snapshots.join()
    with gzip.open(output / 'samples.csv.gz', 'wt', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time', 'footprint', 'rss', 'lifetime_max_footprint'])
        writer.writerows(samples)
    summary = summarize(
        output,
        config=config,
        kernelspec=kernelspec,
        samples=samples,
        marks=marks,
        snapshots=snapshots.records if snapshots else {},
        returncode=returncode,
        launched=launched,
    )
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(headline(summary), indent=2))
    if summary['snapshot_overlaps_timed_phase']:
        msg = (
            'A snapshot overlapped a timed phase; timings in this run are invalid: '
            f'{[w for w in summary["snapshot_windows"] if w["timed_phases"]]}'
        )
        raise SystemExit(msg)
    if returncode:
        msg = f'Notebook run failed; inspect {output / "worker.log"}'
        raise SystemExit(msg)


def _difference(a, b):
    return None if a is None or b is None else a - b


def _stats(values):
    """Median/p95/max in MiB of a list of byte counts."""
    if not values:
        return None
    ordered = sorted(values)
    return dict(
        n=len(ordered),
        median_mib=statistics.median(ordered) / MIB,
        p05_mib=ordered[int(0.05 * (len(ordered) - 1))] / MIB,
        p95_mib=ordered[int(0.95 * (len(ordered) - 1))] / MIB,
        max_mib=ordered[-1] / MIB,
        min_mib=ordered[0] / MIB,
    )


def summarize(output, *, config, kernelspec, samples, marks, snapshots, returncode, launched):
    """Reduce the 10 ms series and marks to the documented summary.json."""
    phases = [m for m in marks if m['kind'] == 'phase']
    events = [m for m in marks if m['kind'] != 'phase']
    by_name = {m['phase']: m for m in marks}

    def phase_of(timestamp):
        name = 'pre_start'
        for mark in phases:
            if mark['time'] <= timestamp:
                name = mark['phase']
            else:
                break
        return name

    labelled = [(phase_of(t), t, fp, rss) for t, fp, rss, _ in samples]
    named_windows = [
        (name, r['started'], r['finished'])
        for name, r in snapshots.items()
        if 'started' in r and 'finished' in r
    ]
    for name, record in snapshots.items():
        if name == 'peak':
            # Earlier peak snapshots were overwritten but still suspended the kernel.
            named_windows += [
                ('peak', h['started'], h['finished']) for h in record.get('history', [])[:-1]
            ]
    windows = [(a, b) for _, a, b in named_windows]

    def overlap(start, end):
        end = float('inf') if end is None else end
        return sum(max(0.0, min(end, b) - max(start, a)) for a, b in windows)

    phase_spans = [
        (
            mark['phase'],
            mark['time'],
            phases[i + 1]['time'] if i + 1 < len(phases) else float('inf'),
        )
        for i, mark in enumerate(phases)
    ]
    snapshot_windows = []
    for name, start, end in sorted(named_windows, key=lambda w: w[1]):
        touched = [phase for phase, a, b in phase_spans if start < b and end > a]
        snapshot_windows.append(
            dict(
                name=name,
                started=start,
                finished=end,
                phases=touched,
                timed_phases=[phase for phase in touched if phase.startswith(TIMED_PHASES)],
            )
        )

    table = []
    for index, mark in enumerate(phases):
        end = phases[index + 1]['time'] if index + 1 < len(phases) else None
        rows = [row for row in labelled if row[0] == mark['phase']]
        table.append(
            dict(
                phase=mark['phase'],
                start=mark['time'],
                seconds=None if end is None else end - mark['time'],
                footprint=_stats([r[2] for r in rows]),
                rss=_stats([r[3] for r in rows]),
                mark_footprint_mib=mark['footprint'] / MIB,
                snapshot_overlap_s=overlap(mark['time'], end),
                mark_rss_mib=mark['rss'] / MIB,
            )
        )

    # Metric samples end where the diagnostic (numerical/image) pass begins.
    workload_end = by_name['diagnostic']['time'] if 'diagnostic' in by_name else float('inf')
    workload = [row for row in labelled if row[1] < workload_end]
    diagnostic = [row for row in labelled if row[1] >= workload_end]

    def peak(column, rows=workload):
        if not rows:
            return None
        row = max(rows, key=lambda r: r[column])
        return dict(mib=row[column] / MIB, time=row[1], phase=row[0])

    def phase_max(column):
        maxima = {}
        for row in labelled:
            maxima[row[0]] = max(maxima.get(row[0], 0), row[column])
        return {name: value / MIB for name, value in maxima.items()}

    def phase_median(name, column):
        rows = [r[column] for r in labelled if r[0] == name]
        return statistics.median(rows) / MIB if rows else None

    footprint_peak = peak(2)
    rss_peak = peak(3)
    peak_phase_rows = (
        [r[2] for r in labelled if r[0] == footprint_peak['phase']] if footprint_peak else []
    )

    def seconds_between(start, end):
        if start in by_name and end in by_name:
            return by_name[end]['time'] - by_name[start]['time']
        return None

    reopen = [
        seconds_between(
            f'reopen_{i}', f'close_{i + 1}' if i + 1 < config['reopens'] else 'final_close'
        )
        for i in range(config['reopens'])
    ]
    # Supervisor vs in-kernel: nearest 10 ms sample to each mark's task_info reading.
    agreement = []
    times = [s[0] for s in samples]
    import bisect

    for mark in marks:
        position = bisect.bisect_left(times, mark['time'])
        near = [samples[i] for i in (position - 1, position) if 0 <= i < len(samples)]
        if near:
            sample = min(near, key=lambda s: abs(s[0] - mark['time']))
            agreement.append(
                dict(
                    phase=mark['phase'],
                    dt_ms=1000 * (sample[0] - mark['time']),
                    task_vm_info_mib=mark['task_vm_footprint'] / MIB,
                    in_kernel_rusage_mib=mark['footprint'] / MIB,
                    supervisor_mib=sample[1] / MIB,
                )
            )
    playback = []
    for cycle in range(config['cycles']):
        path = output / f'playback-{cycle}.json'
        if path.exists():
            seen = json.loads(path.read_text())
            playback.append(dict(cycle=cycle, polls=len(seen), distinct_frames=len(set(seen))))
    frames_path = output / 'frames.json'
    frames = None
    if frames_path.exists():
        records = json.loads(frames_path.read_text())
        frames = dict(
            count=len(records),
            canonical_digest=hashlib.sha256(
                ''.join(r['canonical_digest'] for r in records).encode()
            ).hexdigest(),
            raw_digest=hashlib.sha256(
                ''.join(r['raw_digest'] for r in records).encode()
            ).hexdigest(),
            dtypes=sorted({r['dtype'] for r in records}),
            shapes_distinct=len({tuple(r['shape']) for r in records}),
            max_abs_error=max(r.get('max_abs_error', 0.0) for r in records),
            max_rmse=max(r.get('rmse', 0.0) for r in records),
            nonzero_lost_total=sum(r.get('nonzero_lost', 0) for r in records),
            nonzero_outside_display_total=sum(
                r.get('nonzero_outside_display', 0) for r in records
            ),
            max_abs_mass_difference=max(abs(r.get('mass_difference', 0.0)) for r in records),
            max_displayed_outside_sum=max(
                abs(r.get('displayed_outside_sum', 0.0)) for r in records
            ),
            displayed_bytes_max=max(
                r['shape'][0]
                * r['shape'][1]
                * r['shape'][2]
                * (2 if r['dtype'] == 'uint16' else 1 if r['dtype'] == 'uint8' else 4)
                for r in records
            ),
        )
    intervals = [b[0] - a[0] for a, b in itertools.pairwise(samples)]
    start_mark = by_name.get('start', {})
    end_mark = by_name.get('end', {})
    return dict(
        schema=SCHEMA,
        status='ok' if returncode == 0 and 'end' in by_name else 'failed',
        returncode=returncode,
        arm=config['arm'],
        factors=config['factors'],
        kernel_pid=marks[0]['pid'] if marks else None,
        launched=launched,
        python=start_mark.get('python'),
        python_version=start_mark.get('python_version'),
        pyvista=by_name.get('arm_ready', {}).get('pyvista'),
        kernel_env=kernelspec['env'],
        prewarm=config['prewarm'],
        footprint=dict(
            peak=footprint_peak,
            workload_lifetime_max_mib=(
                by_name['diagnostic']['lifetime_max_footprint'] / MIB
                if 'diagnostic' in by_name
                else None
            ),
            diagnostic_peak=peak(2, diagnostic),
            steady_mib=phase_median('playback_2', 2),
            steady=_stats([r[2] for r in labelled if r[0] == 'playback_2']),
            after_close_mib=phase_median('after_close', 2),
            startup_max_mib=max((r[2] for r in labelled if r[0] == 'startup'), default=0) / MIB,
            peak_phase_p95_mib=_stats(peak_phase_rows)['p95_mib'] if peak_phase_rows else None,
            phase_max_mib=phase_max(2),
            peak_snapshot_footprint_mib=(
                snapshots['peak']['footprint_at_start'] / MIB
                if snapshots.get('peak', {}).get('footprint_at_start')
                else None
            ),
            peak_snapshot_gap_mib=(
                footprint_peak['mib'] - snapshots['peak']['footprint_at_start'] / MIB
                if footprint_peak and snapshots.get('peak', {}).get('footprint_at_start')
                else None
            ),
        ),
        rss=dict(
            peak=rss_peak,
            workload_lifetime_max_mib=(
                by_name['diagnostic']['lifetime_max_rss'] / MIB
                if 'diagnostic' in by_name
                else None
            ),
            steady_mib=phase_median('playback_2', 3),
            after_close_mib=phase_median('after_close', 3),
            phase_max_mib=phase_max(3),
        ),
        footprint_minus_rss_mib=dict(
            steady=_difference(phase_median('playback_2', 2), phase_median('playback_2', 3)),
            after_close=_difference(
                phase_median('after_close', 2), phase_median('after_close', 3)
            ),
        ),
        accounting={
            name: dict(
                footprint_mib=by_name[name]['footprint'] / MIB,
                rss_mib=by_name[name]['rss'] / MIB,
                **{f'{k}_mib': v / MIB for k, v in by_name[name].get('task_vm', {}).items()},
                madvise=by_name[name].get('madvise'),
            )
            for name in ('third_cycle_mid', 'after_close', 'after_close_snapshot', 'diagnostic')
            if name in by_name
        },
        times=dict(
            preparation_s=seconds_between('preparation', 'startup'),
            startup_s=seconds_between('startup', 'viewer_ready'),
            seek_s=by_name.get('seeking_done', {}).get('seconds'),
            reopen_s=reopen,
            diagnostic_s=seconds_between('diagnostic', 'end'),
        ),
        peak_margin_mib=config['peak_margin_mib'],
        provenance=dict(
            **config['provenance'],
            imported_versions=by_name.get('arm_ready', {}).get('imported_versions'),
        ),
        snapshots=snapshots,
        cross_check=dict(
            in_kernel=start_mark.get('cross_check'),
            supervisor_vs_task_vm_info=agreement,
            max_abs_diff_mib_within_10ms=max(
                (
                    abs(a['supervisor_mib'] - a['task_vm_info_mib'])
                    for a in agreement
                    if abs(a['dt_ms']) <= 10
                ),
                default=None,
            ),
        ),
        allocator=start_mark.get('allocator'),
        arm_state=dict(
            transform={
                k: by_name.get('transform_ready', {}).get(k)
                for k in ('density_clim', 'scale', 'crop_box', 'persistent_shape')
            },
            end={k: v for k, v in end_mark.items() if k in ('lazy', 'madvise', 'relief')},
            lazy_at_start=start_mark.get('lazy'),
            lazy_finder_installed=start_mark.get('lazy_finder'),
            window_sizes=sorted({tuple(m['window_size']) for m in marks if 'window_size' in m}),
        ),
        playback=playback,
        frames=frames,
        images=sorted(p.name for p in output.glob('frame-*.png')),
        sampling=dict(
            interval_target_s=config['sample_interval'],
            samples=len(samples),
            interval_median_ms=1000 * statistics.median(intervals) if intervals else None,
            interval_max_ms=1000 * max(intervals) if intervals else None,
        ),
        snapshots_mode=config['snapshots'],
        snapshot_windows=snapshot_windows,
        snapshot_overlaps_timed_phase=any(w['timed_phases'] for w in snapshot_windows),
        timed_phase_prefixes=list(TIMED_PHASES),
        phases=table,
        events=[{k: m[k] for k in ('phase', 'time', 'snapshot')} for m in events],
    )


def headline(summary):
    """Return the few numbers printed after a run."""
    fp = summary['footprint']
    return dict(
        status=summary['status'],
        arm=summary['arm'],
        peak_footprint_mib=fp['peak'] and round(fp['peak']['mib'], 1),
        peak_phase=fp['peak'] and fp['peak']['phase'],
        peak_snapshot_gap_mib=fp['peak_snapshot_gap_mib']
        and round(fp['peak_snapshot_gap_mib'], 1),
        peak_rss_mib=summary['rss']['peak'] and round(summary['rss']['peak']['mib'], 1),
        lifetime_max_rss_mib=summary['rss']['workload_lifetime_max_mib']
        and round(summary['rss']['workload_lifetime_max_mib'], 1),
        workload_lifetime_max_mib=fp['workload_lifetime_max_mib']
        and round(fp['workload_lifetime_max_mib'], 1),
        steady_footprint_mib=fp['steady_mib'] and round(fp['steady_mib'], 1),
        after_close_footprint_mib=fp['after_close_mib'] and round(fp['after_close_mib'], 1),
        steady_rss_mib=summary['rss']['steady_mib'] and round(summary['rss']['steady_mib'], 1),
        seek_s=summary['times']['seek_s'],
        startup_s=summary['times']['startup_s'],
    )


# =========================================================================== compare


def compare(run_a, run_b):
    """Compare numerical records and the five matched images of two runs."""
    sys.path.insert(0, str(REPO))
    import numpy as np
    import pyvista as pv

    frames_a = json.loads((run_a / 'frames.json').read_text())
    frames_b = json.loads((run_b / 'frames.json').read_text())
    canonical = sum(
        a['canonical_digest'] == b['canonical_digest']
        for a, b in zip(frames_a, frames_b, strict=True)
    )
    raw = sum(a['raw_digest'] == b['raw_digest'] for a, b in zip(frames_a, frames_b, strict=True))
    images = {}
    pixels = {}
    for index in IMAGE_FRAMES:
        name = f'frame-{index:03d}.png'
        images[name] = float(pv.compare_images(str(run_a / name), str(run_b / name)))
        a = pv.read(run_a / name).active_scalars.astype(int)
        b = pv.read(run_b / name).active_scalars.astype(int)
        difference = np.abs(a - b).max(axis=-1)
        pixels[name] = dict(max=int(difference.max()), differing=int(np.count_nonzero(difference)))
    result = dict(
        frames=len(frames_a),
        canonical_equal=canonical,
        raw_equal=raw,
        b_max_abs_error=max(r.get('max_abs_error', 0.0) for r in frames_b),
        b_max_rmse=max(r.get('rmse', 0.0) for r in frames_b),
        b_nonzero_lost=sum(r.get('nonzero_lost', 0) for r in frames_b),
        b_nonzero_outside_display=sum(r.get('nonzero_outside_display', 0) for r in frames_b),
        b_max_abs_mass_difference=max(abs(r.get('mass_difference', 0.0)) for r in frames_b),
        image_error=images,
        image_error_max=max(images.values()),
        pixel_difference=pixels,
        # pytest-pyvista defaults: warn above 200, fail above 500.
        images_pass=max(images.values()) <= 500,
    )
    print(json.dumps(result, indent=2))
    return result


def main():
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='one fresh-kernel trial')
    run.add_argument('output', type=Path, help='new folder for this run')
    run.add_argument('--arm', default='baseline', help='comma list of factors')
    run.add_argument('--notebook', type=Path, default=REPO / 'tornado_animation.ipynb')
    run.add_argument('--frame-stats', type=Path, default=DEFAULT_STATS)
    run.add_argument('--cache-root', type=Path, default=CACHE_ROOT)
    run.add_argument(
        '--peak-margin-mib',
        type=float,
        default=8.0,
        help='re-take the peak snapshot when footprint exceeds the last one by this much',
    )
    run.add_argument(
        '--peak-max-snapshots',
        type=int,
        default=8,
        help='bound on peak re-snapshots (each suspends the kernel about 1 s)',
    )
    run.add_argument(
        '--snapshots',
        choices=('on', 'off'),
        default='on',
        help='footprint/vmmap snapshots; use off for timed trials (sampling still runs)',
    )
    run.add_argument('--cycles', type=int, default=3)
    run.add_argument('--reopens', type=int, default=3)
    run.add_argument('--interval', type=float, default=0.01, help=argparse.SUPPRESS)
    run.add_argument('--prewarm', action='store_true', help='read every frame before launching')
    warm = commands.add_parser('prewarm', help='read every frame once before a batch')
    warm.add_argument('--cache-root', type=Path, default=CACHE_ROOT)
    comparison = commands.add_parser('compare', help='numerical and image gate between two runs')
    comparison.add_argument('run_a', type=Path)
    comparison.add_argument('run_b', type=Path)
    hidden = commands.add_parser('worker')
    hidden.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        if args.cycles < 1 or args.reopens < 0:
            parser.error('--cycles must be positive and --reopens nonnegative')
        supervise(args)
    elif args.command == 'prewarm':
        print(json.dumps(prewarm(args.cache_root)))
    elif args.command == 'compare':
        compare(args.run_a, args.run_b)
    else:
        worker(args.output)


if __name__ == '__main__':
    main()
