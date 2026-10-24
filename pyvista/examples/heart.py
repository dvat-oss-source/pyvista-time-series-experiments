"""Original, sampled CONRAD cardiac phase consensuses."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from zipfile import ZipFile

import numpy as np
import pooch

import pyvista as pv

_URL = (
    'https://www5.cs.fau.de/fileadmin/research/datasets/ConradCardiacModel/CardiacModel_v1.1.zip'
)
_SHA256 = '4c68caa46eb96ee83f604011b2ace79c2e30297220f7e0cb2ae00b0bd7b03b87'
_ARCHIVE_BYTES = 78_152_158
_VERTICES = (4660, 1480, 1299, 3814, 1139, 4038)
_TRIANGLES = (9316, 2956, 2594, 7624, 2274, 8072)
_PARTS = (
    'LV myocardium',
    'Right atrium',
    'Left atrium',
    'Right ventricle',
    'Aorta',
    'Left ventricle',
)
_ORDER = 'myocardium rightAtrium leftAtrium rightVentricle aorta leftVentricle'


def _digest(path: Path) -> str:
    """Hash a file without retaining its contents."""
    return pooch.file_hash(path, alg='sha256')  # type: ignore[attr-defined]


def _check_configuration(archive: ZipFile) -> None:
    """Verify the source's component order, dimensions, and local index offsets."""
    configuration = dict(
        line.split(': ', 1) for line in archive.read('CardiacModel/cnfg.ccc').decode().splitlines()
    )
    expected = {
        'ANATOMICAL_COMPONENTS': '6',
        'COMPONENT_ORDER': _ORDER,
        'CARDIAC_PHASES': '10',
        'NUMBER_PRINCIPAL_COMPONENTS_EACH_PHASE': ' '.join(['16'] * 10),
        'TOTAL_VERTEX_COUNT': str(sum(_VERTICES)),
        'VERTEX_DIMENSION': '3',
        'VERTEX_COMPONENT_OFFSETS': ' '.join(map(str, np.cumsum((0, *_VERTICES[:-1])))),
        'TOTAL_TRIANGLE_COUNT': str(sum(_TRIANGLES)),
        'TRIANGLE_COMPONENT_OFFSETS': ' '.join(map(str, np.cumsum((0, *_TRIANGLES[:-1])))),
    }
    if configuration != expected:
        msg = 'Unexpected CONRAD anatomical configuration.'
        raise ValueError(msg)


def _read_consensus(archive: ZipFile, index: int) -> pv.PolyData:
    """Read only the final consensus column, preserving float64 and triangle order."""
    with archive.open(f'CardiacModel/CardiacModel/phase_{index}.ccm') as stream:
        if stream.readline().split() != [b'DIMENSION', str(3 * sum(_VERTICES)).encode(), b'16']:
            msg = 'Unexpected CONRAD coordinate dimensions.'
            raise ValueError(msg)
        if stream.readline().split() != [b'POINTDIMENSION', b'3']:
            msg = 'Expected three-dimensional CONRAD coordinates.'
            raise ValueError(msg)
        if stream.readline().strip() != b'EIGENVALUES' or len(stream.readline().split()) != 16:
            msg = 'Unexpected CONRAD eigenvalue header.'
            raise ValueError(msg)
        if stream.readline().strip() != b'EIGENVECTORS | CONSENSUS':
            msg = 'Expected the documented CONRAD consensus column.'
            raise ValueError(msg)
        points = np.empty((sum(_VERTICES), 3), dtype=np.float64)
        for row in range(points.size):
            values = stream.readline().split()
            if len(values) != 17:
                msg = 'Unexpected CONRAD consensus row.'
                raise ValueError(msg)
            points.flat[row] = float(values[-1])
        if stream.readline().split() != [b'CONNECTIVITY', str(sum(_TRIANGLES)).encode(), b'3']:
            msg = 'Unexpected CONRAD connectivity dimensions.'
            raise ValueError(msg)
        faces = np.empty((sum(_TRIANGLES), 4), dtype=np.int64)
        for row in range(len(faces)):
            faces[row] = [int(value) for value in stream.readline().split()]

    point_offset = cell_offset = 0
    for n_points, n_cells in zip(_VERTICES, _TRIANGLES, strict=True):
        part = faces[cell_offset : cell_offset + n_cells]
        if (
            not np.all(part[:, 0] == 3)
            or np.any(part[:, 1:] < 0)
            or np.any(part[:, 1:] >= n_points)
        ):
            msg = 'CONRAD triangles do not match their anatomical component.'
            raise ValueError(msg)
        part[:, 1:] += point_offset
        point_offset += n_points
        cell_offset += n_cells
    if not np.isfinite(points).all():
        msg = 'CONRAD coordinates must be finite.'
        raise ValueError(msg)
    mesh = pv.PolyData(points, faces)
    mesh.point_data['part'] = np.repeat(np.arange(6, dtype=np.uint8), _VERTICES)
    mesh.cell_data['part'] = np.repeat(np.arange(6, dtype=np.uint8), _TRIANGLES)
    mesh.field_data['part_names'] = _PARTS
    mesh.field_data['cardiac_phase'] = [index / 10]
    mesh.field_data['time_basis'] = ['fraction of cardiac cycle; no measured seconds']
    mesh.field_data['coordinate_units'] = ['mm, inferred from CT scale and source implementation']
    return mesh


def download_heart_motion() -> tuple[str, ...]:
    """Download the ten original CONRAD cardiac phase-consensus meshes.

    Return paths instead of loading the sequence. Each file contains all six
    synchronized components and can be read independently with :func:`pyvista.read`.
    The original 78 MB archive is hash-verified and converted one phase at a time
    to lossless ``.pv`` files. This requires the ``pyvista[io]`` extra.

    .. versionadded:: 0.50

    Returns
    -------
    tuple[str, ...]
        Ten cached filenames in normalized cardiac phase order, 0.0 through 0.9.
        Point and cell array ``'part'`` identifies the six anatomical components.
        Field arrays contain ``'part_names'``, ``'cardiac_phase'``, ``'time_basis'``,
        and ``'coordinate_units'``. Cached file hashes are checked on every call.

    Notes
    -----
    These are stored statistical phase consensuses, not a patient recording or
    the provider's full two-level, spline-evaluated phantom. The four cavities,
    LV myocardium, and aorta are included. Pulmonary artery, detailed valves,
    coronaries, pericardium, and separate RV/atrial myocardial walls are absent.
    No interpolation, deformation, smoothing, or coordinate transform is applied.
    Millimeters are inferred from the documented CT/model scale: the original
    files have no unit tag. Phase fractions are documented; measured cycle
    duration is unavailable.

    The provider calls the model open source but supplies no explicit standalone
    dataset license. The GPL source-code license for CONRAD is not asserted to
    license this separate archive. Check permission for your intended use and redistribution.

    See the `provider's model description and citation
    <https://www5.cs.fau.de/conrad/data/heart-model/index.html>`_.
    Citation::

        Unberath et al., "Open-source 4D Statistical Shape Model of the Heart for
        X-ray Projection Imaging", ISBI 2015, pp. 739-742.

    Examples
    --------
    Read one phase without loading the sequence.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> filenames = examples.download_heart_motion()  # doctest: +SKIP
    >>> mesh = pv.read(filenames[0])  # doctest: +SKIP
    >>> mesh.field_data['cardiac_phase']  # doctest: +SKIP
    array([0.])

    """
    from pyvista.examples import downloads  # noqa: PLC0415

    directory = Path(downloads.USER_DATA_PATH) / 'conrad-heart-v1.1'
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = Path(
        pooch.retrieve(  # type: ignore[attr-defined]
            _URL,
            known_hash=f'sha256:{_SHA256}',
            fname='CardiacModel_v1.1.zip',
            path=directory,
        )
    )
    if archive_path.stat().st_size != _ARCHIVE_BYTES:
        msg = 'Unexpected CONRAD archive size.'
        raise ValueError(msg)
    paths = tuple(directory / f'phase_{index:02}.pv' for index in range(10))
    manifest = directory / 'frames.json'
    if manifest.is_file():
        try:
            hashes = json.loads(manifest.read_text())
        except ValueError:
            hashes = []
        if (
            isinstance(hashes, list)
            and len(hashes) == 10
            and all(
                path.is_file() and _digest(path) == checksum
                for path, checksum in zip(paths, hashes, strict=True)
            )
        ):
            return tuple(map(str, paths))

    with ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(dir=directory) as temporary:
        _check_configuration(archive)
        connectivity = None
        hashes = []
        for index, path in enumerate(paths):
            mesh = _read_consensus(archive, index)
            if connectivity is None:
                connectivity = mesh.faces.copy()
            elif not np.array_equal(mesh.faces, connectivity):
                msg = 'CONRAD connectivity differs between phases.'
                raise ValueError(msg)
            staged = Path(temporary) / path.name
            mesh.save(staged)
            hashes.append(_digest(staged))
            del mesh
        manifest.unlink(missing_ok=True)
        for path in paths:
            (Path(temporary) / path.name).replace(path)
        staged_manifest = Path(temporary) / manifest.name
        staged_manifest.write_text(json.dumps(hashes) + '\n')
        staged_manifest.replace(manifest)
    return tuple(map(str, paths))
