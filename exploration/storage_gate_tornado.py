"""Hash the stored tornado bytes so a run can prove it changed none of them (read-only).

Usage::

    python exploration/storage_gate_tornado.py hash CACHE_ROOT OUTPUT.json
    python exploration/storage_gate_tornado.py compare BEFORE.json AFTER.json

``CACHE_ROOT`` is the ``tornado_looping`` folder. Hashed: every
``display-256/*.pv`` frame, the ``display-256/frames.json`` manifest, every
``vdb/*.vdb`` and ``TornadoLoopingVDB.zip``. ``compare`` exits 1 on any
difference in the set of files, their sizes or their SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def stored_files(root):
    """Return the gated files in a stable order."""
    display = root / 'display-256'
    files = [
        *sorted(display.glob('*.pv')),
        display / 'frames.json',
        *sorted((root / 'vdb').glob('*.vdb')),
        root / 'TornadoLoopingVDB.zip',
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        msg = f'Missing stored files: {missing}'
        raise SystemExit(msg)
    return files


def hash_files(root):
    """Hash every gated file in chunks."""
    rows = []
    for path in stored_files(root):
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        rows.append(
            dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=digest)
        )
    counts = dict(
        pv=sum(r['path'].endswith('.pv') for r in rows),
        vdb=sum(r['path'].endswith('.vdb') for r in rows),
    )
    aggregate = hashlib.sha256(
        '\n'.join(f'{r["path"]} {r["bytes"]} {r["sha256"]}' for r in rows).encode()
    ).hexdigest()
    return dict(
        root=str(root.resolve()),
        counts=counts,
        bytes=sum(r['bytes'] for r in rows),
        aggregate_sha256=aggregate,
        files=rows,
    )


def compare(before, after):
    """Return a list of differences between two hash records."""
    a = {r['path']: r for r in before['files']}
    b = {r['path']: r for r in after['files']}
    problems = [f'missing after: {p}' for p in sorted(a.keys() - b.keys())]
    problems += [f'added after: {p}' for p in sorted(b.keys() - a.keys())]
    problems += [
        f'changed: {p}'
        for p in sorted(a.keys() & b.keys())
        if (a[p]['bytes'], a[p]['sha256']) != (b[p]['bytes'], b[p]['sha256'])
    ]
    return problems


def main():
    """Dispatch the hash or compare command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    hashing = commands.add_parser('hash')
    hashing.add_argument('root', type=Path)
    hashing.add_argument('output', type=Path)
    comparing = commands.add_parser('compare')
    comparing.add_argument('before', type=Path)
    comparing.add_argument('after', type=Path)
    args = parser.parse_args()
    if args.command == 'hash':
        record = hash_files(args.root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2))
        print(json.dumps({k: v for k, v in record.items() if k != 'files'}))
        return
    problems = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    print(json.dumps(dict(identical=not problems, problems=problems)))
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
