"""Independently compare original PcaIO data with installed-library playback."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from zipfile import ZipFile

import numpy as np

import pyvista as pv
from pyvista import examples

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work-dir', type=Path, required=True)
args = parser.parse_args()
root = args.work_dir.resolve()
paths = examples.download_heart_motion()
assert Path(paths[0]).parent == root / 'fresh-cache/conrad-heart-v1.1'
archive_path = Path(paths[0]).parent / 'CardiacModel_v1.1.zip'
with archive_path.open('rb') as stream:
    archive_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
assert archive_hash == '4c68caa46eb96ee83f604011b2ace79c2e30297220f7e0cb2ae00b0bd7b03b87'
assert archive_path.stat().st_size == 78_152_158
playback = json.loads((root / 'first-use.json').read_text())
warm = json.loads((root / 'cached-offline.json').read_text())
assert playback['displayed_coordinate_sha256'] == warm['displayed_coordinate_sha256']
results = []
baseline_points = baseline_faces = None
with ZipFile(archive_path) as archive:
    config = dict(
        line.split(': ', 1) for line in archive.read('CardiacModel/cnfg.ccc').decode().splitlines()
    )
    order = config['COMPONENT_ORDER'].split()
    assert order == [
        'myocardium',
        'rightAtrium',
        'leftAtrium',
        'rightVentricle',
        'aorta',
        'leftVentricle',
    ]
    assert int(config['CARDIAC_PHASES']) == 10
    point_offsets = np.array([int(x) for x in config['VERTEX_COMPONENT_OFFSETS'].split()])
    cell_offsets = np.array([int(x) for x in config['TRIANGLE_COMPONENT_OFFSETS'].split()])
    vertices = np.diff(np.r_[point_offsets, int(config['TOTAL_VERTEX_COUNT'])])
    triangles = np.diff(np.r_[cell_offsets, int(config['TOTAL_TRIANGLE_COUNT'])])
    assert vertices.tolist() == [4660, 1480, 1299, 3814, 1139, 4038]
    assert triangles.tolist() == [9316, 2956, 2594, 7624, 2274, 8072]
    with archive.open('CardiacModel/CCmModel.ccm') as stream:
        second_level_mean = np.loadtxt(stream, skiprows=5, max_rows=160, usecols=16)
    for index, filename in enumerate(paths):
        with archive.open(f'CardiacModel/CardiacModel/phase_{index}.ccm') as stream:
            points = np.loadtxt(stream, skiprows=5, max_rows=49_290, usecols=16).reshape(-1, 3)
            faces = np.loadtxt(stream, skiprows=1, dtype=np.int64)
        assert np.all(faces[:, 0] == 3)
        faces[:, 1:] += np.repeat(point_offsets, triangles)[:, None]
        mesh = pv.read(filename)
        assert np.isfinite(points).all()
        np.testing.assert_array_equal(
            mesh.field_data['part_names'],
            [
                'LV myocardium',
                'Right atrium',
                'Left atrium',
                'Right ventricle',
                'Aorta',
                'Left ventricle',
            ],
        )
        np.testing.assert_array_equal(mesh.points, points)
        np.testing.assert_array_equal(mesh.faces, faces.ravel())
        np.testing.assert_array_equal(mesh.point_data['part'], np.repeat(np.arange(6), vertices))
        np.testing.assert_array_equal(mesh.cell_data['part'], np.repeat(np.arange(6), triangles))
        assert mesh.points.dtype == np.float64
        assert mesh.field_data['cardiac_phase'].item() == index / 10
        assert 'inferred' in mesh.field_data['coordinate_units'].item()
        assert 'no measured seconds' in mesh.field_data['time_basis'].item()
        assert playback['displayed_coordinate_sha256'][index] == [
            index / 10,
            hashlib.sha256(points.tobytes()).hexdigest(),
        ]
        if baseline_points is None:
            baseline_points = points.copy()
            baseline_faces = faces.copy()
        np.testing.assert_array_equal(faces, baseline_faces)
        measurements = []
        for part, name in enumerate(order):
            start = point_offsets[part]
            xyz = points[start : start + vertices[part]]
            local = faces[cell_offsets[part] : cell_offsets[part] + triangles[part], 1:] - start
            surface = pv.PolyData(xyz, np.column_stack([np.full(len(local), 3), local]))
            edges = surface.n_open_edges
            volume = None
            if part != 0:
                assert edges == 0
                a, b, c = (xyz[local[:, column]] for column in range(3))
                volume = abs(float(np.einsum('ij,ij->', a, np.cross(b, c)))) / 6000
                assert np.isclose(surface.volume / 1000, volume, rtol=1e-10, atol=1e-10)
            displacement = xyz - baseline_points[start : start + vertices[part]]
            measurements.append(
                {
                    'name': name,
                    'boundary_or_nonmanifold_edges': edges,
                    'volume_ml_assuming_mm': volume,
                    'rms_displacement_from_phase_zero_assuming_mm': float(
                        np.sqrt(np.mean(np.sum(displacement**2, axis=1)))
                    ),
                }
            )
            del surface
        results.append(
            {
                'phase': index / 10,
                'maximum_coordinate_error': float(np.abs(mesh.points - points).max()),
                'parts': measurements,
            }
        )
        del mesh, points, faces
    uncompressed_bytes = sum(member.file_size for member in archive.infolist())
    member_count = len(archive.infolist())
    license_members = [
        name
        for name in archive.namelist()
        if 'license' in name.lower() or 'copying' in name.lower()
    ]

volumes = [result['parts'][5]['volume_ml_assuming_mm'] for result in results]
ef = 100 * (max(volumes) - min(volumes)) / max(volumes)
assert all(
    max(row['parts'][part]['rms_displacement_from_phase_zero_assuming_mm'] for row in results) > 0
    for part in range(6)
)
result = {
    'source_archive_sha256': archive_hash,
    'source_archive_bytes': archive_path.stat().st_size,
    'source_uncompressed_bytes': uncompressed_bytes,
    'source_zip_members': member_count,
    'source_archive_license_members': license_members,
    'source_configuration': config,
    'original_to_cache_points_and_faces_exact_all_ten': True,
    'original_to_displayed_points_exact_all_ten_both_runs': True,
    'point_and_cell_labels_match_original_component_offsets': True,
    'phase_metadata_exact': True,
    'connectivity_order_fixed_across_all_ten': True,
    'second_level_consensus_max_abs_weight': float(abs(second_level_mean).max()),
    'second_level_consensus_weight_norm': float(np.linalg.norm(second_level_mean)),
    'measurements': results,
    'lv_edv_ml_assuming_mm': max(volumes),
    'lv_esv_ml_assuming_mm': min(volumes),
    'lv_ed_phase': volumes.index(max(volumes)) / 10,
    'lv_es_phase': volumes.index(min(volumes)) / 10,
    'lv_ef_percent': ef,
    'provider_separate_mean_phantom_ef_percent': 52,
    'ef_difference_percentage_points': abs(52 - ef),
    'independent_volume_method': 'NumPy oriented tetrahedron sum; checked against PyVista volume',
    'coordinate_unit_status': 'mm inferred from CT/model scale; original files untagged',
    'timing_status': 'normalized phase only; display period chosen, no physiological timing claim',
    'anatomy_source': 'original cnfg.ccc component order and offsets',
    'correspondence_evidence': 'provider registration; matching topology alone is insufficient',
    'physiological_validation': False,
    'environment': {
        'python': platform.python_version(),
        'library_path': pv.__file__,
        **{
            name: version(name)
            for name in (
                'pyvista',
                'vtk',
                'numpy',
                'pyvista-zstd',
                'imageio',
                'pillow',
                'nbclient',
                'psutil',
                'ipykernel',
                'pooch',
                'pyvista-validation',
                'nbformat',
                'jupyter-client',
            )
        },
    },
}
(root / 'source_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(
    json.dumps(
        {
            key: value
            for key, value in result.items()
            if key not in ('measurements', 'source_configuration')
        },
        indent=2,
    )
)
