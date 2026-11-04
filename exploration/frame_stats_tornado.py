"""Summarize the 128 tornado display frames for footprint arithmetic (read-only).

Usage::

    python exploration/frame_stats_tornado.py DISPLAY_DIR OUTPUT.json

``DISPLAY_DIR`` is the ``display-256`` cache folder. Frames are read through
``pv.read``; nothing in the cache is written. The JSON records each frame's
shape, dtype, geometry, value range, nonzero fraction and nonzero bounding box
(index and world), plus their union. When every frame shares spacing and
direction and its origin lies on one common lattice, the union box is also
expressed on that lattice (``lattice``), which gives one fixed crop shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv


def frame_record(index, path):
    """Return statistics for one frame without copying its density buffer."""
    mesh = pv.read(path)
    values = mesh.point_data['density']
    shape = tuple(int(value) for value in mesh.dimensions)
    volume = np.asarray(values).reshape(shape[::-1])  # (z, y, x) view, Fortran order
    nonzero = volume != 0
    count = int(np.count_nonzero(nonzero))
    record = dict(
        index=index,
        file=path.name,
        shape=list(shape),
        dtype=str(values.dtype),
        bytes=int(values.nbytes),
        origin=[float(value) for value in mesh.origin],
        spacing=[float(value) for value in mesh.spacing],
        direction=np.asarray(mesh.direction_matrix).tolist(),
        min=float(values.min()),
        max=float(values.max()),
        nonzero=count,
        nonzero_fraction=count / values.size,
        nonzero_bbox_index=None,
        nonzero_bbox_world=None,
    )
    if count:
        # Axis order of ``volume`` is (z, y, x); report (x, y, z) inclusive ranges.
        box = []
        for axis in (2, 1, 0):
            other = tuple(a for a in range(3) if a != axis)
            hit = np.flatnonzero(nonzero.any(axis=other))
            box.append([int(hit[0]), int(hit[-1])])
        record['nonzero_bbox_index'] = box
        corners = np.array(
            [[box[0][i], box[1][j], box[2][k]] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
            dtype=float,
        )
        world = (
            np.asarray(mesh.origin)
            + (corners * np.asarray(mesh.spacing)) @ np.asarray(mesh.direction_matrix).T
        )
        record['nonzero_bbox_world'] = [
            [float(world[:, a].min()), float(world[:, a].max())] for a in range(3)
        ]
    return record


def lattice(records):
    """Express the union nonzero box on a lattice shared by every frame, if one exists."""
    spacing = np.array([r['spacing'] for r in records])
    direction = np.array([r['direction'] for r in records])
    if not (np.allclose(spacing, spacing[0], rtol=0, atol=1e-9 * spacing[0].max())):
        return dict(common=False, reason='spacing differs between frames')
    if not np.allclose(direction, direction[0], atol=1e-9):
        return dict(common=False, reason='direction differs between frames')
    step = spacing[0]
    axes = direction[0]
    origin0 = np.array(records[0]['origin'])
    offsets = []
    for r in records:
        shift = (axes.T @ (np.array(r['origin']) - origin0)) / step
        if not np.allclose(shift, np.round(shift), atol=1e-4):
            return dict(common=False, reason=f'frame {r["index"]} origin is off-lattice')
        offsets.append(np.round(shift).astype(int))
    lower = []
    upper = []
    for r, offset in zip(records, offsets, strict=True):
        if r['nonzero_bbox_index'] is None:
            continue
        box = np.array(r['nonzero_bbox_index'])
        lower.append(box[:, 0] + offset)
        upper.append(box[:, 1] + offset)
    lo = np.min(lower, axis=0)
    hi = np.max(upper, axis=0)
    extent_lo = np.min(list(offsets), axis=0)
    extent_hi = np.max(
        [o + np.array(r['shape']) - 1 for r, o in zip(records, offsets, strict=True)], axis=0
    )
    return dict(
        common=True,
        spacing=step.tolist(),
        direction=axes.tolist(),
        origin=origin0.tolist(),
        frame_offsets=[o.tolist() for o in offsets],
        union_nonzero_box=[[int(a), int(b)] for a, b in zip(lo, hi, strict=True)],
        union_nonzero_shape=[int(b - a + 1) for a, b in zip(lo, hi, strict=True)],
        union_extent_box=[[int(a), int(b)] for a, b in zip(extent_lo, extent_hi, strict=True)],
        union_extent_shape=[int(b - a + 1) for a, b in zip(extent_lo, extent_hi, strict=True)],
    )


def main():
    """Read every frame once and write the JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('display', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    paths = sorted(args.display.glob('tornado_*.pv'))
    if len(paths) != 128:
        msg = f'Expected 128 frames, found {len(paths)}.'
        raise SystemExit(msg)
    records = [frame_record(index, path) for index, path in enumerate(paths)]
    world = np.array([r['nonzero_bbox_world'] for r in records if r['nonzero_bbox_world']])
    total = sum(np.prod(r['shape']) for r in records)
    nonzero = sum(r['nonzero'] for r in records)
    summary = dict(
        frames=len(records),
        dtypes=sorted({r['dtype'] for r in records}),
        distinct_shapes=len({tuple(r['shape']) for r in records}),
        bytes_min=min(r['bytes'] for r in records),
        bytes_max=max(r['bytes'] for r in records),
        bytes_total=sum(r['bytes'] for r in records),
        global_min=min(r['min'] for r in records),
        global_max=max(r['max'] for r in records),
        nonzero_fraction_overall=nonzero / total,
        nonzero_fraction_min=min(r['nonzero_fraction'] for r in records),
        nonzero_fraction_max=max(r['nonzero_fraction'] for r in records),
        union_nonzero_bbox_world=[
            [float(world[:, a, 0].min()), float(world[:, a, 1].max())] for a in range(3)
        ],
        lattice=lattice(records),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, frames=records), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
