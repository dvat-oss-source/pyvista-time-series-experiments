"""Compare cached playback with independent reads of the original CONRAD meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np
import pyvista as pv

from .heart_data import PARTS
from .heart_data import PHASES
from .heart_data import TRIANGLES
from .heart_data import VERTICES
from .heart_data import HeartSource
from .heart_data import MemoryBudget
from .heart_data import digest


def compare(source, *, indices):
    """Read the original text with NumPy, independently of the extraction parser."""
    with zipfile.ZipFile(source.manifest.parent / 'CardiacModel_v1.1.zip') as archive:
        config = dict(
            line.split(': ', 1)
            for line in archive.read('CardiacModel/cnfg.ccc').decode().splitlines()
        )
    expected = {
        'ANATOMICAL_COMPONENTS': '6',
        'COMPONENT_ORDER': (
            'myocardium rightAtrium leftAtrium rightVentricle aorta leftVentricle'
        ),
        'CARDIAC_PHASES': '10',
        'TOTAL_VERTEX_COUNT': '16430',
        'TOTAL_TRIANGLE_COUNT': '32836',
        'VERTEX_COMPONENT_OFFSETS': '0 4660 6140 7439 11253 12392',
        'TRIANGLE_COMPONENT_OFFSETS': '0 9316 12272 14866 22490 24764',
    }
    if any(config[key] != value for key, value in expected.items()):
        msg = 'The original anatomical configuration differs from the playback mapping.'
        raise AssertionError(msg)
    reference = source.frame(0)
    volumes = []
    displacements = []
    edge_defects = []
    for index in indices:
        source.budget.check()
        with zipfile.ZipFile(source.manifest.parent / 'CardiacModel_v1.1.zip') as archive:
            with archive.open(f'CardiacModel/CardiacModel/phase_{index}.ccm') as stream:
                points = np.loadtxt(stream, skiprows=5, max_rows=49_290, usecols=16).reshape(-1, 3)
                faces = np.loadtxt(stream, skiprows=1, dtype=np.int64)
        faces[:, 1:] += np.repeat(np.cumsum((0, *VERTICES[:-1])), TRIANGLES)[:, None]
        mesh = source.frame(index)
        if not np.array_equal(mesh.points, points) or not np.array_equal(
            mesh.faces, faces.ravel()
        ):
            msg = f'Phase {index} differs from its original consensus mesh.'
            raise AssertionError(msg)
        if mesh.points.dtype != np.float64 or not np.array_equal(mesh['part'], reference['part']):
            msg = 'Coordinate precision or anatomical correspondence changed.'
            raise AssertionError(msg)
        if not np.array_equal(mesh.faces, reference.faces):
            msg = 'Connectivity changes across phases; interpolation would need investigation.'
            raise AssertionError(msg)
        if mesh.field_data['cardiac_phase'].item() != PHASES[index]:
            msg = 'Saved phase metadata differs from the original phase index.'
            raise AssertionError(msg)
        part_volumes = []
        part_displacement = []
        part_edges = []
        point_start = cell_start = 0
        for name, n_points, n_cells in zip(PARTS, VERTICES, TRIANGLES, strict=True):
            xyz = points[point_start : point_start + n_points]
            triangles = faces[cell_start : cell_start + n_cells, 1:] - point_start
            surface = pv.PolyData(xyz, np.column_stack((np.full(n_cells, 3), triangles)))
            part_edges.append(surface.n_open_edges)
            if surface.n_open_edges and name != 'LV myocardium':
                msg = f'{name} is not closed; cavity volume would be unreliable.'
                raise AssertionError(msg)
            a, b, c = (xyz[triangles[:, column]] for column in range(3))
            part_volumes.append(
                None
                if surface.n_open_edges
                else float(abs(np.einsum('ij,ij->', a, np.cross(b, c))) / 6000)
            )
            original = reference.points[point_start : point_start + n_points]
            part_displacement.append(
                float(np.sqrt(np.mean(np.sum((xyz - original) ** 2, axis=1))))
            )
            point_start += n_points
            cell_start += n_cells
            del surface
        volumes.append(part_volumes)
        displacements.append(part_displacement)
        edge_defects.append(part_edges)
        del points, faces, mesh
    return {
        'compared_indices': list(indices),
        'coordinate_max_error_mm': 0.0,
        'connectivity_equal_to_original': True,
        'connectivity_fixed_across_phases': True,
        'float64_preserved': True,
        'phase_metadata_exact': True,
        'source_anatomical_configuration_checked': True,
        'boundary_or_nonmanifold_edge_counts': edge_defects,
        'myocardium_volume': (
            'Not reported: source has an edge defect; source geometry is preserved.'
        ),
        'part_names': PARTS,
        'part_vertex_counts': VERTICES,
        'part_triangle_counts': TRIANGLES,
        'part_volumes_ml': volumes,
        'part_rms_displacement_from_phase_zero_mm': displacements,
        'anatomical_correspondence': (
            'Source registration preserves vertex order; no interpolation used.'
        ),
    }


def main():
    """Run sampled checks first, then optionally render one small preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pv.OFF_SCREEN = True
    with MemoryBudget() as budget:
        source = HeartSource(args.manifest, budget)
        result = compare(source, indices=tuple(range(10)) if args.full else (0, 3, 9))
        if args.render:
            from .heart_preview import preview

            result['render'] = preview(
                source,
                args.output.with_suffix('.gif'),
                indices=tuple(range(10)) if args.full else (0, 2, 4, 6, 8),
                cycles=5 if args.full else 3,
                size=(400, 320) if args.full else (320, 256),
            )
        if args.full:
            lv = np.asarray(result['part_volumes_ml'])[:, 5]
            result['lv_ejection_fraction_percent'] = float(100 * (lv.max() - lv.min()) / lv.max())
            result['lv_ed_es_phases'] = [PHASES[int(lv.argmax())], PHASES[int(lv.argmin())]]
        result['source_archive_sha256'] = digest(args.manifest.parent / 'CardiacModel_v1.1.zip')
        result['source_phases'] = source.times
        result['cache_bytes'] = sum(
            path.stat().st_size for path in args.manifest.parent.glob('*.pv')
        )
    result['memory'] = budget.report()
    result['environment'] = {
        'python': sys.version,
        'pyvista': pv.__version__,
        'vtk': str(pv.vtk_version_info),
        'platform': platform.platform(),
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
