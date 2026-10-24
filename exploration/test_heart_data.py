"""Exercise the demo's safeguards without downloading data or creating a renderer."""

from __future__ import annotations

import json
from types import SimpleNamespace

import psutil
import pytest

from .heart_data import PHASES
from .heart_data import HeartSource
from .heart_data import MemoryBudget
from .heart_data import digest


def test_host_memory_reserve(monkeypatch):
    """Refuse work before a renderer or data reader can start on a crowded host."""
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=128 * 1024**2))
    budget = MemoryBudget()
    with pytest.raises(MemoryError, match='available host memory'):
        with budget:
            pytest.fail('The budget allowed work to start.')
    assert not budget._thread.is_alive()


def test_corrupt_cached_frame(tmp_path):
    """Detect corruption before calling the native mesh reader."""
    path = tmp_path / 'phase.pv'
    path.write_bytes(b'original')
    record = {'file': path.name, 'sha256': digest(path)}
    manifest = tmp_path / 'sequence.json'
    manifest.write_text(json.dumps({'frames': [dict(record, phase=phase) for phase in PHASES]}))
    path.write_bytes(b'corrupt')
    source = HeartSource(manifest, MemoryBudget())
    with pytest.raises(ValueError, match='integrity check'):
        source.frame(0)


def test_changed_phase_metadata(tmp_path):
    """Reject a sequence whose sample times were silently changed."""
    manifest = tmp_path / 'sequence.json'
    manifest.write_text(json.dumps({'frames': [{'phase': phase} for phase in reversed(PHASES)]}))
    with pytest.raises(ValueError, match='original CONRAD phases'):
        HeartSource(manifest, MemoryBudget())
