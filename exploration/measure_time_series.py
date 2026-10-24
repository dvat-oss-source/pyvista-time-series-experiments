"""Measure the notebook's direct-mapper loop and the discrete player in fresh processes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import threading

import psutil
import pyvista as pv
from pyvista import examples


def measure(mode, output, *, cycles):
    """Render heart cycles using the notebook's GIF and camera settings."""
    pv.OFF_SCREEN = True
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    peak_rss = [initial_rss]
    stop = threading.Event()

    def sample():
        """Retain only the greatest observed resident set size."""
        while not stop.wait(0.002):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    paths = tuple(map(Path, examples.download_heart_motion()))
    read_count = [0]
    logical_read_bytes = [0]

    def load(index):
        """Count a requested read of one original compressed frame."""
        path = paths[index]
        mesh = pv.read(path)
        read_count[0] += 1
        logical_read_bytes[0] += path.stat().st_size
        return mesh

    colors = ['#bb3754', '#4b85c2', '#e6a64c', '#74bdcd', '#e8d6b0', '#df6461']
    lut = pv.LookupTable(cmap=colors, n_values=6, scalar_range=(-0.5, 5.5))
    rgba = lut.values.copy()
    rgba[0, 3] = 70
    lut.values = rgba
    pl = pv.Plotter(notebook=False, window_size=(400, 320))
    player = None
    cycle_rss = []
    digests = []
    gif = output.with_suffix('.gif')
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        if mode == 'direct':
            mesh = load(0)
            actor = pl.add_mesh(
                mesh,
                scalars='part',
                preference='point',
                cmap=lut,
                clim=(-0.5, 5.5),
                show_scalar_bar=False,
            )
        else:
            player = pv.TimeSeriesPlayer(
                pl,
                tuple(index / 10 for index in range(10)),
                load,
                cycle_end=1.0,
                scalars='part',
                clim=(-0.5, 5.5),
            )
            actor = player.actor
            actor.mapper.lookup_table = lut
        mapper = actor.mapper
        parts = mapper.dataset.field_data['part_names'].tolist()
        pl.set_background('#101b2c')
        pl.camera_position = [(40, -180, 180), (0, 0, 0), (-1, 0, 0)]
        pl.enable_parallel_projection()
        pl.camera.zoom(0.85)
        pl.add_legend(list(zip(parts, colors, strict=True)), size=(0.38, 0.30), bcolor='#101b2c')
        pl.add_text(
            'CONRAD phase consensuses | mm inferred',
            position='upper_left',
            font_size=8,
            color='white',
        )
        pl.show(interactive=False, auto_close=False)
        pl.open_gif(str(gif), fps=20)
        for cycle in range(cycles):
            for index in range(10):
                if mode == 'direct':
                    mesh = load(index)
                    mapper.dataset = mesh
                    pl.render()
                else:
                    player.seek_index(index)
                phase = mapper.dataset.field_data['cardiac_phase'].item()
                pl.add_text(
                    f'Cardiac phase {phase:.1f}',
                    name='phase',
                    font_size=10,
                    position='lower_left',
                    color='white',
                )
                if cycle == 0:
                    pl.write_frame()
                    digests.append(hashlib.sha256(mapper.dataset.points.tobytes()).hexdigest())
            gc.collect()
            cycle_rss.append(process.memory_info().rss)
    finally:
        if player is not None:
            player.close()
        pl.close()
        stop.set()
        sampler.join()
    peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
    high = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        high *= 1024
    return {
        'mode': mode,
        'cycles': cycles,
        'python': platform.python_version(),
        'pyvista': pv.__version__,
        'vtk': str(pv.vtk_version_info),
        'original_frame_count': len(paths),
        'original_frame_bytes': sum(path.stat().st_size for path in paths),
        'read_count': read_count[0],
        'logical_compressed_read_bytes': logical_read_bytes[0],
        'initial_rss_bytes': initial_rss,
        'sampled_peak_rss_bytes': peak_rss[0],
        'os_peak_rss_bytes': high,
        'cycle_end_rss_bytes': cycle_rss,
        'post_close_rss_bytes': process.memory_info().rss,
        'gif_bytes': gif.stat().st_size,
        'displayed_coordinate_sha256': digests,
    }


def main():
    """Write one mode's result so a shell can start each mode in a new process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('direct', 'player'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--cycles', type=int, default=5)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error('Cycles must be positive.')
    args.output.write_text(
        json.dumps(measure(args.mode, args.output, cycles=args.cycles), indent=2) + '\n'
    )


if __name__ == '__main__':
    main()
