"""Read the ten original CONRAD phase-consensus meshes, one at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import urllib.request
import zipfile

import numpy as np
import psutil
import pyvista as pv

URL = 'https://www5.cs.fau.de/fileadmin/research/datasets/ConradCardiacModel/CardiacModel_v1.1.zip'
ARCHIVE_BYTES = 78_152_158
ARCHIVE_SHA256 = '4c68caa46eb96ee83f604011b2ace79c2e30297220f7e0cb2ae00b0bd7b03b87'
MIB = 1024**2
PARTS = (
    'LV myocardium',
    'Right atrium',
    'Left atrium',
    'Right ventricle',
    'Aorta',
    'Left ventricle',
)
VERTICES = (4660, 1480, 1299, 3814, 1139, 4038)
TRIANGLES = (9316, 2956, 2594, 7624, 2274, 8072)
PHASES = tuple(index / 10 for index in range(10))


class MemoryBudget:
    """Monitor RSS with constant storage and reserve room before bounded operations."""

    def __init__(self):
        """Set a 1 GiB process ceiling and leave 1 GiB available to the host."""
        self.limit = 1024 * MIB
        self.reserve = 1024 * MIB
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.initial_available = psutil.virtual_memory().available
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        """Keep one maximum rather than an ever-growing list of samples."""
        while not self._stop.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def check(self, allowance=128 * MIB):
        """Stop before another operation if its working allowance no longer fits."""
        rss = self.process.memory_info().rss
        self.peak = max(self.peak, rss)
        if max(self.peak, rss) + allowance >= self.limit:
            msg = 'Heart demo stopped before its 1 GiB RSS budget was exhausted.'
            raise MemoryError(msg)
        if psutil.virtual_memory().available < self.reserve + allowance:
            msg = 'Heart demo stopped to preserve 1 GiB of available host memory.'
            raise MemoryError(msg)

    def __enter__(self):
        """Start sampling only after the initial budget check succeeds."""
        self.check()
        self._thread.start()
        return self

    def __exit__(self, *_):
        """Stop the sampler on success or failure."""
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)

    def report(self):
        """Return sampled RSS and the operating system lifetime high-water mark."""
        if sys.platform == 'win32':
            high = self.process.memory_info().peak_wset
        else:
            import resource

            high = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform != 'darwin':
                high *= 1024
        return {
            'budget_bytes': self.limit,
            'host_reserve_bytes': self.reserve,
            'initial_available_bytes': self.initial_available,
            'sampled_peak_rss_bytes': self.peak,
            'lifetime_peak_rss_bytes': high,
        }


def digest(path):
    """Hash a file without reading it all into RAM."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def retrieve(directory, budget):
    """Stream the fixed-size public archive to disk, without unpacking its PCA arrays."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / 'CardiacModel_v1.1.zip'
    if archive.exists():
        if archive.stat().st_size != ARCHIVE_BYTES or digest(archive) != ARCHIVE_SHA256:
            msg = f'Archive integrity check failed: remove {archive} and retry.'
            raise ValueError(msg)
        return archive
    temporary = archive.with_suffix('.partial')
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as output:
            if int(response.headers['Content-Length']) != ARCHIVE_BYTES:
                msg = 'The published archive size changed; inspect its metadata before retrieval.'
                raise ValueError(msg)
            count = 0
            while chunk := response.read(MIB):
                budget.check()
                count += len(chunk)
                if count > ARCHIVE_BYTES:
                    msg = 'Download exceeded the inspected archive size.'
                    raise ValueError(msg)
                output.write(chunk)
        if count != ARCHIVE_BYTES:
            msg = 'Download ended before the inspected archive size.'
            raise ValueError(msg)
        if digest(temporary) != ARCHIVE_SHA256:
            msg = 'Archive checksum changed; inspect the source before using it.'
            raise ValueError(msg)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def original_frame(archive, index):
    """Extract original consensus coordinates and local triangles from a PcaIO file."""
    if index not in range(10):
        msg = 'CONRAD has exactly ten phases, indexed from zero through nine.'
        raise IndexError(msg)
    member = f'CardiacModel/CardiacModel/phase_{index}.ccm'
    with zipfile.ZipFile(archive) as source, source.open(member) as stream:
        header = stream.readline().split()
        if header != [b'DIMENSION', b'49290', b'16']:
            msg = f'Unexpected CONRAD dimensions: {header!r}.'
            raise ValueError(msg)
        if stream.readline().split() != [b'POINTDIMENSION', b'3']:
            msg = 'Expected three-dimensional CONRAD coordinates.'
            raise ValueError(msg)
        stream.readline()
        stream.readline()
        stream.readline()
        points = np.empty(49_290, dtype=np.float64)
        for row in range(points.size):
            points[row] = float(stream.readline().rsplit(maxsplit=1)[-1])
        if stream.readline().split() != [b'CONNECTIVITY', b'32836', b'3']:
            msg = 'Unexpected CONRAD connectivity dimensions.'
            raise ValueError(msg)
        faces = np.empty((32_836, 4), dtype=np.int64)
        for row in range(len(faces)):
            faces[row] = [int(value) for value in stream.readline().split()]
        # CONRAD stores local vertex indices for each consecutive anatomical component.
        point_offset = cell_offset = 0
        for n_points, n_cells in zip(VERTICES, TRIANGLES, strict=True):
            part = faces[cell_offset : cell_offset + n_cells]
            if not np.all(part[:, 0] == 3) or not np.all(
                (part[:, 1:] >= 0) & (part[:, 1:] < n_points)
            ):
                msg = 'Source triangles do not match their documented anatomical component.'
                raise ValueError(msg)
            part[:, 1:] += point_offset
            point_offset += n_points
            cell_offset += n_cells
        if not np.isfinite(points).all():
            msg = 'Source coordinates contain non-finite values.'
            raise ValueError(msg)
        mesh = pv.PolyData(points.reshape(-1, 3), faces)
        mesh['part'] = np.repeat(np.arange(6, dtype=np.uint8), VERTICES)
        mesh.field_data['cardiac_phase'] = [PHASES[index]]
        return mesh


def prepare(directory, budget):
    """Save lossless PyVista frames sequentially and publish the manifest last."""
    directory = Path(directory)
    manifest = directory / 'sequence.json'
    if manifest.exists():
        return manifest
    archive = retrieve(directory, budget)
    records = []
    for index in range(10):
        budget.check()
        mesh = original_frame(archive, index)
        path = directory / f'phase_{index:02}.pv'
        mesh.save(path)
        records.append({'phase': PHASES[index], 'file': path.name, 'sha256': digest(path)})
        del mesh
    manifest.write_text(
        json.dumps(
            {
                'source_url': URL,
                'archive_sha256': digest(archive),
                'coordinate_unit': 'mm',
                'time_unit': 'fraction of cardiac cycle',
                'parts': PARTS,
                'frames': records,
            },
            indent=2,
        )
        + '\n'
    )
    return manifest


class HeartSource:
    """Load exact, whole-heart samples without caching or assuming tissue correspondence."""

    def __init__(self, manifest, budget):
        """Read only a small manifest; coordinates remain on disk."""
        self.manifest = Path(manifest)
        self.metadata = json.loads(self.manifest.read_text())
        self.frames = self.metadata['frames']
        self.times = tuple(record['phase'] for record in self.frames)
        if self.times != PHASES:
            msg = 'Expected the ten original CONRAD phases from 0.0 through 0.9.'
            raise ValueError(msg)
        self.budget = budget

    def frame(self, index):
        """Read one detached frame using the normal PyVista reader."""
        self.budget.check()
        record = self.frames[index]
        path = self.manifest.parent / record['file']
        if digest(path) != record['sha256']:
            msg = f'Cached frame integrity check failed: {path}.'
            raise ValueError(msg)
        return pv.read(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    with MemoryBudget() as memory:
        print(prepare(args.directory, memory))
    print(json.dumps(memory.report(), indent=2))
