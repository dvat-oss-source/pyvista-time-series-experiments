"""Small source-format fixtures exercise the cardiac downloader without network."""

from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.examples import downloads
from pyvista.examples import heart


@pytest.fixture
def cardiac_archive(tmp_path, monkeypatch):
    """Build six asymmetric tetrahedra per phase in the original PcaIO layout."""
    monkeypatch.setattr(heart, '_VERTICES', (4,) * 6)
    monkeypatch.setattr(heart, '_TRIANGLES', (4,) * 6)
    monkeypatch.setattr(downloads, 'USER_DATA_PATH', str(tmp_path / 'cache'))
    path = tmp_path / 'original.zip'
    configuration = (
        'ANATOMICAL_COMPONENTS: 6\n'
        f'COMPONENT_ORDER: {heart._ORDER}\n'
        'CARDIAC_PHASES: 10\n'
        'NUMBER_PRINCIPAL_COMPONENTS_EACH_PHASE: ' + ' '.join(['16'] * 10) + '\n'
        'TOTAL_VERTEX_COUNT: 24\nVERTEX_DIMENSION: 3\n'
        'VERTEX_COMPONENT_OFFSETS: 0 4 8 12 16 20\n'
        'TOTAL_TRIANGLE_COUNT: 24\nTRIANGLE_COMPONENT_OFFSETS: 0 4 8 12 16 20\n'
    )
    points = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 5]], dtype=float)
    faces = np.array([[3, 0, 2, 1], [3, 0, 1, 3], [3, 0, 3, 2], [3, 1, 2, 3]])
    with ZipFile(path, 'w') as archive:
        archive.writestr('CardiacModel/cnfg.ccc', configuration)
        for phase in range(10):
            xyz = np.concatenate([points * (1 + phase / 100) + part * 11 for part in range(6)])
            body = 'DIMENSION 72 16\nPOINTDIMENSION 3\nEIGENVALUES\n'
            body += ' '.join(['1'] * 16) + '\nEIGENVECTORS | CONSENSUS\n'
            body += ''.join('0 ' * 16 + repr(value.item()) + '\n' for value in xyz.flat)
            body += 'CONNECTIVITY 24 3\n'
            body += ''.join(' '.join(map(str, row)) + '\n' for row in np.tile(faces, (6, 1)))
            archive.writestr(f'CardiacModel/CardiacModel/phase_{phase}.ccm', body)
    monkeypatch.setattr(heart, '_ARCHIVE_BYTES', path.stat().st_size)
    monkeypatch.setattr(heart, '_SHA256', heart._digest(path))
    monkeypatch.setattr(heart, '_URL', path.as_uri())

    def retrieve(url, *, known_hash, fname, path):
        """Verify that the downloader requests the pinned archive hash."""
        assert url == heart._URL
        assert known_hash == f'sha256:{heart._SHA256}'
        assert fname == 'CardiacModel_v1.1.zip'
        assert Path(path) == tmp_path / 'cache/conrad-heart-v1.1'
        return str(tmp_path / 'original.zip')

    monkeypatch.setattr(heart.pooch, 'retrieve', retrieve)
    return path, points


def test_download_heart_motion(cardiac_archive):
    """Preserve local component topology, coordinates, labels, and original phases."""
    _, points = cardiac_archive
    paths = examples.download_heart_motion()
    assert len(paths) == 10
    for index, path in enumerate(paths):
        mesh = pv.read(path)
        expected = np.concatenate([points * (1 + index / 100) + part * 11 for part in range(6)])
        np.testing.assert_array_equal(mesh.points, expected)
        np.testing.assert_array_equal(mesh.point_data['part'], np.repeat(range(6), 4))
        np.testing.assert_array_equal(mesh.cell_data['part'], np.repeat(range(6), 4))
        np.testing.assert_array_equal(mesh.field_data['part_names'], heart._PARTS)
        assert mesh.field_data['cardiac_phase'].item() == index / 10
        assert mesh.points.dtype == np.float64
        for part in range(6):
            local = mesh.faces.reshape(-1, 4)[4 * part : 4 * (part + 1), 1:]
            assert local.min() == 4 * part
            assert local.max() == 4 * part + 3


def test_heart_cached_and_corrupt(cardiac_archive, monkeypatch):
    """A warm cache avoids conversion; corrupted cached bytes are regenerated."""
    archive, _ = cardiac_archive
    assert archive.is_file()
    paths = examples.download_heart_motion()
    original = Path(paths[3]).read_bytes()
    parser = heart._read_consensus

    def fail_read(*_args):
        """Reject source parsing while checking the intact warm cache."""
        pytest.fail('Warm cache parsed source geometry')

    monkeypatch.setattr(heart, '_read_consensus', fail_read)
    assert examples.download_heart_motion() == paths
    monkeypatch.setattr(heart, '_read_consensus', parser)
    Path(paths[3]).write_bytes(b'corrupted frame')
    examples.download_heart_motion()
    assert Path(paths[3]).read_bytes() == original


@pytest.mark.parametrize(
    ('old', 'new', 'message'),
    [
        ('COMPONENT_ORDER: myocardium', 'COMPONENT_ORDER: unexpected', 'anatomical configuration'),
        ('DIMENSION 72 16', 'DIMENSION 71 16', 'coordinate dimensions'),
        ('3 0 2 1', '3 0 2 4', 'anatomical component'),
        (' 0.0\n', ' nan\n', 'must be finite'),
    ],
)
def test_heart_rejects_invalid_source(cardiac_archive, monkeypatch, old, new, message):
    """Reject invalid anatomy, dimensions, connectivity, and nonfinite coordinates."""
    path, _ = cardiac_archive
    with ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    with ZipFile(path, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content.replace(old.encode(), new.encode()))
    monkeypatch.setattr(heart, '_ARCHIVE_BYTES', path.stat().st_size)
    monkeypatch.setattr(heart, '_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match=message):
        examples.download_heart_motion()
    assert not (Path(downloads.USER_DATA_PATH) / 'conrad-heart-v1.1/frames.json').exists()


def test_heart_rejects_changed_connectivity(cardiac_archive, monkeypatch):
    """Equal dimensions do not conceal a changed face ordering between phases."""
    path, _ = cardiac_archive
    with ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    name = 'CardiacModel/CardiacModel/phase_5.ccm'
    members[name] = members[name].replace(b'3 0 2 1', b'3 0 1 2')
    with ZipFile(path, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    monkeypatch.setattr(heart, '_SHA256', heart._digest(path))
    with pytest.raises(ValueError, match='connectivity differs'):
        examples.download_heart_motion()
