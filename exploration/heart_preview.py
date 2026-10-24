"""Render exact whole-heart samples with the existing TimeController and PyVista."""

from __future__ import annotations

import gc
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import psutil
import pyvista as pv

from .heart_data import PARTS
from .time_controller import TimeController

COLORS = ('#bb3754', '#4b85c2', '#e6a64c', '#74bdcd', '#e8d6b0', '#df6461')


def preview(
    source, path, *, indices=tuple(range(10)), cycle_seconds=0.5, cycles=3, size=(400, 320)
):
    """Stream a GIF and exercise repeated cycles without retaining previous geometry.

    Physical seconds are a display choice: the source stores cardiac phase only.
    The default 0.5 second period matches the provider's 120 bpm projection demo.
    No frame interpolation, geometric scaling, smoothing, or decimation is applied.
    """
    indices = tuple(indices)
    if not indices or indices[0] != 0 or tuple(sorted(set(indices))) != indices:
        msg = 'Preview indices must start at zero and increase without duplicates.'
        raise ValueError(msg)
    if not set(indices).issubset(range(10)) or cycle_seconds <= 0 or cycles < 1:
        msg = 'Invalid preview phases, display period, or cycle count.'
        raise ValueError(msg)
    if max(size) > 640:
        msg = 'This memory-budgeted demo limits rendering to 640 pixels per dimension.'
        raise ValueError(msg)
    durations = np.diff([source.times[index] for index in indices] + [1.0]) * cycle_seconds
    source.budget.check(256 * 1024**2)
    pl = pv.Plotter(notebook=False, window_size=size)
    cycle_rss = []
    errors = []
    try:
        controller = TimeController(
            pl, source.times, source.frame, scalars='part', clim=(-0.5, 5.5)
        )
        actor = controller.actor
        mapper = actor.mapper
        actor.mapper.lookup_table = pv.LookupTable(
            cmap=list(COLORS), n_values=6, scalar_range=(-0.5, 5.5)
        )
        rgba = actor.mapper.lookup_table.values.copy()
        rgba[0, 3] = 70
        actor.mapper.lookup_table.values = rgba
        pl.remove_scalar_bar()
        pl.clear_slider_widgets()
        pl.set_background('#101b2c')
        pl.camera_position = [(40, -180, 180), (0, 0, 0), (-1, 0, 0)]
        pl.enable_parallel_projection()
        pl.camera.zoom(0.85)
        pl.add_legend(list(zip(PARTS, COLORS, strict=True)), size=(0.38, 0.30), bcolor='#101b2c')
        pl.add_text(
            'CONRAD statistical mean | mm', position='upper_left', font_size=9, color='white'
        )
        pl.show(interactive=False, auto_close=False)
        source.budget.check()
        # The legacy GIF writer writes each image immediately and retains one previous image.
        with imageio.get_writer(
            Path(path), format='GIF-PIL', mode='I', duration=durations.tolist(), loop=0
        ) as writer:
            for index in indices:
                source.budget.check()
                controller.select_index(index)
                pl.add_text(
                    f'Cardiac phase {source.times[index]:.1f}',
                    name='phase',
                    font_size=10,
                    position='lower_left',
                    color='white',
                )
                writer.append_data(pl.screenshot())
                direct = source.frame(index)
                errors.append(float(np.max(np.abs(actor.mapper.dataset.points - direct.points))))
                del direct
        for _ in range(cycles):
            for index in range(10):
                source.budget.check()
                controller.select_index(index)
                if controller.actor is not actor or actor.mapper is not mapper:
                    msg = 'Playback replaced its actor or mapper.'
                    raise RuntimeError(msg)
            gc.collect()
            cycle_rss.append(psutil.Process().memory_info().rss)
        source.budget.check()
    finally:
        pl.close()
    if len(cycle_rss) >= 3 and cycle_rss[-1] - cycle_rss[0] > 32 * 1024**2:
        msg = 'Repeated-cycle RSS grew by more than the 32 MiB accumulation allowance.'
        raise MemoryError(msg)
    return {
        'preview_indices': indices,
        'durations_seconds': durations.tolist(),
        'display_cycle_seconds': cycle_seconds,
        'render_size': size,
        'mapper_coordinate_max_error_mm': max(errors),
        'actor_and_mapper_reused': True,
        'cycle_end_rss_bytes': cycle_rss,
        'cycle_end_growth_bytes': cycle_rss[-1] - cycle_rss[0],
        'interpolation': 'none',
        'time_basis': 'source normalized cardiac phases; chosen display period',
    }
