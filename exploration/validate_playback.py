"""Compare a PVD spatial part with original files and optionally render it.

This diagnostic uses pytest comparison helpers and samples whole-process RSS.
It is not a representative solver benchmark or an image regression test.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import resource
import sys
import threading
import time

from defusedxml import ElementTree
import numpy as np
import psutil
import pyvista as pv

from reader_source import ReaderSource
from test_reader_source import Scene
from test_reader_source import assert_frame_equal
from time_controller import TimeController


def main():
    """Validate direct reads, filtered display, seeks, and retained ownership."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--scalars', required=True)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--preference', choices=('point', 'cell'), default='point')
    parser.add_argument('--component', type=int, help='Explicit component of a vector to display.')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    entries = ElementTree.parse(args.path).getroot().find('Collection').findall('DataSet')
    originals = {}
    for entry in entries:
        attrs = entry.attrib
        originals.setdefault(float(attrs['timestep']), []).append(
            (int(attrs.get('part', 0)), args.path.parent / attrs['file'])
        )
    paths = [sorted(originals[t])[args.block][1] for t in sorted(originals)]
    source = ReaderSource(
        pv.get_reader(args.path),
        scalars=args.scalars,
        block=args.block,
        preference=args.preference,
    )
    assert source.times == tuple(sorted(originals))
    retained = source.frame(0)

    def filtered(frame):
        """Use the same spatial clip on an adapter frame or an independent direct read."""
        result = frame.clip(normal='x', origin=frame.center)
        if args.component is not None:
            data = result.point_data if args.preference == 'point' else result.cell_data
            data['display_component'] = data[args.scalars][:, args.component]
        return result

    def load(index):
        """Filter only the requested saved sample."""
        return filtered(source.frame(index))

    pv.OFF_SCREEN = True
    pl = pv.Plotter(window_size=(480, 360)) if args.render else Scene()
    process = psutil.Process()
    samples = [process.memory_info().rss]
    stop = threading.Event()

    def sample_rss():
        """Include native CPU buffers in the sampled process working set."""
        while not stop.wait(0.002):
            samples.append(process.memory_info().rss)

    thread = threading.Thread(target=sample_rss, daemon=True)
    thread.start()
    try:
        start = time.perf_counter()
        ctl = TimeController(
            pl,
            source.times,
            load,
            scalars='display_component' if args.component is not None else args.scalars,
            preference=args.preference,
            clim=(-1, 1),
        )
        if args.render:
            pl.camera_position = 'iso'
            pl.show(auto_close=False, interactive=False)
        first_ms = (time.perf_counter() - start) * 1000
        indices = [0, len(paths) // 2, len(paths) - 1, 0]
        indices += np.random.default_rng(42).integers(0, len(paths), 12).tolist()
        for index in indices:
            direct = pv.read(paths[index])
            assert_frame_equal(source.frame(index), direct)
            ctl.select_index(index)
            assert_frame_equal(ctl.actor.mapper.dataset, filtered(direct))
            assert_frame_equal(retained, pv.read(paths[0]))
            assert source.reader.active_time_value == source.times[0]
        del direct, retained
        gc.collect()
        rss_before_cycles = process.memory_info().rss
        start = time.perf_counter()
        for index in list(range(len(paths))) * 4:
            ctl.select_index(index)
        cycles_ms = (time.perf_counter() - start) * 1000
        gc.collect()
        rss_after_cycles = process.memory_info().rss
        if args.render:
            pl.screenshot(args.output.with_suffix('.png'))
    finally:
        stop.set()
        thread.join()
        if args.render:
            pl.close()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        'python': sys.version,
        'platform': platform.platform(),
        'pyvista': pv.__version__,
        'pyvista_path': pv.__file__,
        'vtk': tuple(pv.vtk_version_info),
        'numpy': np.__version__,
        'psutil': psutil.__version__,
        'logical_cpus': psutil.cpu_count(),
        'ram_bytes': psutil.virtual_memory().total,
        'path': str(args.path),
        'samples': len(paths),
        'indices': indices,
        'exact_comparisons_passed': True,
        'rendered': args.render,
        'sampled_peak_rss': max(samples),
        'lifetime_peak_rss_bytes': peak if sys.platform == 'darwin' else peak * 1024,
        'rss_before_cycles': rss_before_cycles,
        'rss_after_cycles': rss_after_cycles,
        'first_display_ms': first_ms,
        'four_cycles_ms': cycles_ms,
        'original_bytes_manifest_and_all_parts': args.path.stat().st_size
        + sum(
            path.stat().st_size for path in {p for parts in originals.values() for _, p in parts}
        ),
        'converted_bytes': 0,
        'frame_cache_bytes': 0,
        'conditions': 'Warm/uncontrolled OS cache; imports/reader setup excluded from timings. '
        'First display: read/copy/clip/mapper/show. Cycles: read/copy/clip/update/render. '
        'RSS includes validation reads, native CPU buffers, and screenshots if rendered. '
        'No GPU completion fence or GPU memory measurement.',
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
