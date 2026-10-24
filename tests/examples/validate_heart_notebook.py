"""Execute the unchanged notebook using the installed package, outside checkout."""

from __future__ import annotations

import argparse
import ast
import base64
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import threading

from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
import nbformat
from PIL import Image
import psutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=('first-use', 'cached-offline'))
parser.add_argument('--notebook', type=Path, required=True)
parser.add_argument('--work-dir', type=Path, required=True)
args = parser.parse_args()
root = args.work_dir.resolve()
root.mkdir(parents=True, exist_ok=True)
mode = args.mode
cache = root / 'fresh-cache'
if mode == 'first-use':
    assert not cache.exists(), 'The first-use test must begin without cached heart assets.'
    cache.mkdir()
directory = root / mode
directory.mkdir(exist_ok=True)
assert not any(directory.iterdir()), 'Execution directory must be empty.'
kernel_directory = root / 'kernels' / 'python3'
kernel_directory.mkdir(parents=True, exist_ok=True)
(kernel_directory / 'kernel.json').write_text(
    json.dumps(
        {
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Python 3 (ipykernel)',
            'language': 'python',
        }
    )
)
book = nbformat.read(args.notebook.resolve(), as_version=4)
client = NotebookClient(
    book,
    timeout=900,
    kernel_name='python3',
    resources={'metadata': {'path': str(directory)}},
)
client.km = client.create_kernel_manager()
client.km.kernel_spec_manager = KernelSpecManager(kernel_dirs=[str(root / 'kernels')])
environment = dict(os.environ)
environment.pop('PYTHONPATH', None)
environment.update(
    {
        'PYVISTA_USERDATA_PATH': str(cache),
        'PYVISTA_OFF_SCREEN': 'true',
        'MPLCONFIGDIR': str(root / 'matplotlib'),
    }
)
if mode == 'cached-offline':
    environment.update(
        {
            'HTTPS_PROXY': 'http://127.0.0.1:9',
            'HTTP_PROXY': 'http://127.0.0.1:9',
            'ALL_PROXY': 'http://127.0.0.1:9',
            'NO_PROXY': '',
        }
    )
stop = threading.Event()
measurements = {
    'peak_kernel_rss_bytes': 0,
    'minimum_host_available_bytes': psutil.virtual_memory().available,
}
with client.setup_kernel(env=environment, cwd=str(directory)):
    process = psutil.Process(client.km.provisioner.pid)

    def sample():
        """Track one peak at 10 ms intervals without retaining a sample history."""
        while not stop.wait(0.01):
            try:
                measurements['peak_kernel_rss_bytes'] = max(
                    measurements['peak_kernel_rss_bytes'], process.memory_info().rss
                )
                measurements['minimum_host_available_bytes'] = min(
                    measurements['minimum_host_available_bytes'], psutil.virtual_memory().available
                )
            except psutil.NoSuchProcess:
                break

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        for index, cell in enumerate(book.cells):
            if cell.cell_type == 'code':
                client.execute_cell(cell, index)
        if platform.system() == 'Windows':
            probe_source = 'print(psutil.Process().memory_info().peak_wset)'
        else:
            multiplier = 1 if platform.system() == 'Darwin' else 1024
            probe_source = (
                'import resource\n'
                f'print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * {multiplier})'
            )
        probe = nbformat.v4.new_code_cell(probe_source)
        book.cells.append(probe)
        client.execute_cell(probe, len(book.cells) - 1)
        measurements['os_lifetime_peak_kernel_rss_bytes'] = int(
            ''.join(
                output.text for output in probe.outputs if output.output_type == 'stream'
            ).strip()
        )
        book.cells.pop()
    finally:
        stop.set()
        sampler.join()

streams = ''.join(
    output.text
    for cell in book.cells
    for output in cell.get('outputs', [])
    if output.output_type == 'stream'
)
print(streams)
assert str(Path(importlib.util.find_spec('pyvista').origin).parent) in streams
assert f'Python: {platform.python_version()}' in streams
assert str(cache / 'conrad-heart-v1.1/phase_00.pv') in streams
digests = ast.literal_eval(
    next(
        line.split(': ', 1)[1]
        for line in streams.splitlines()
        if line.startswith('Displayed phase digests:')
    )
)
cycle_rss = ast.literal_eval(
    next(
        line.split(': ', 1)[1]
        for line in streams.splitlines()
        if line.startswith('Cycle-end RSS (bytes):')
    )
)
encoded = next(
    output.data['image/gif']
    for cell in book.cells
    for output in cell.get('outputs', [])
    if 'image/gif' in output.get('data', {})
)
image = Image.open(BytesIO(base64.b64decode(encoded)))
durations = []
for index in range(image.n_frames):
    image.seek(index)
    durations.append(image.info['duration'])
assert image.n_frames == 10
assert durations == [50] * 10
assert image.size == (400, 320)
assert len(digests) == 10
assert all(cell.execution_count is not None for cell in book.cells if cell.cell_type == 'code')
result = {
    'mode': mode,
    'execution_directory': str(directory),
    'kernel_executable': sys.executable,
    'kernel_isolated_mode': False,
    'installed_library_path_checked': True,
    'code_cells_executed': sum(cell.cell_type == 'code' for cell in book.cells),
    'cell_errors': 0,
    'inline_gif_frames': image.n_frames,
    'inline_gif_size': list(image.size),
    'inline_gif_durations_ms': durations,
    'cycle_end_rss_bytes': cycle_rss,
    'cycle_end_growth_bytes': cycle_rss[-1] - cycle_rss[0],
    'growth_over_last_three_cycle_ends_bytes': cycle_rss[-1] - cycle_rss[-3],
    'displayed_coordinate_sha256': digests,
    'memory': measurements,
    'stdout': streams,
}
(root / f'{mode}.json').write_text(json.dumps(result, indent=2) + '\n')
nbformat.write(book, root / f'{mode}.ipynb')
shutil.copyfile(directory / 'heart_preview.gif', root / f'{mode}.gif')
print(
    json.dumps(
        {
            key: value
            for key, value in result.items()
            if key not in ('stdout', 'displayed_coordinate_sha256')
        },
        indent=2,
    )
)
