# Temporal playback decision and evidence

## Provenance and patch

Branch `feat/saved-result-playback` is based on verified local `main` revision
`06d1ee34775f8b9e4addea0a1d22ba7c421c5509`. Worktree:
`/Users/user/ucsd/pyvista-saved-results`. The original checkout and the
separate dirty `pyvista-reader-playback` worktree were preserved. No worktree pruning,
push, or pull request was needed.

The six existing exploration files matched `0d307a71a` byte for byte; commit
`7d02f3230` restores them here. Its controller also matched supplied commit
`4cbe78ae5a73ad9b6e134195ddc3b7ac45b3a421` exactly. This is restored work, not new
behavior. The subsequent patch hardens those files and adds tests and diagnostics.
No package/core code changes are justified.

New behavior: exact indexed controller selection, explicit point/cell association,
validation before replacement, one reader adaptation path, consistent scalar/theme
options, endpoint clamping without overflow, and release of the controller's
last-added `Plotter.mesh` reference. One reader detachment copy is retained; the
controller still shallow-copies buffers and replaces one actor's mapper input.

Raw evidence: Git prototype sources, original cached files/manifests, actual reader,
mapper, plotter and extension implementations at this revision, raw measurement
JSON/scripts, file sizes, and independently executed assertions. The prior report
was only a map. No representative heart/crash dataset is available or required.

Verified environment: Python 3.13.5, PyVista 0.50.dev0 from this checkout, VTK 9.7.1,
NumPy 2.5.3, pytest 9.0.3, psutil 7.2.2, Ruff 0.17.2; Apple M3 Pro, 11 logical CPUs,
19,327,352,832 bytes RAM (18 GiB), macOS 27.0.1 arm64. Existing interpreter:
`../pyvista/.venv/bin/python`; `PYTHONPATH=.` selects this worktree.

## Assessment of all 26 designs

Status describes suitability here, not the existence of an equivalent public
PyVista temporal API. Selected layers: **9 + 11 + 15 + 18 + 23**.

| #   | Layer / design                                              | Assessment              | Status and concrete reason                                                                                                                                   |
| --- | ----------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Storage: fixed-mesh arrays                                  | Conditional             | Git prototype, not adopted. Eager temporal-array copies and per-frame deep mesh copies; coordinates stay fixed. Requires proven correspondence.              |
| 2   | Source: complete datasets                                   | Suitable, lazy form     | Existing TimeSeries baseline, not added. Independent frames handle topology changes; eager snapshots grow with payload.                                      |
| 3   | Encoding: named time arrays                                 | Unsuitable here         | Ordinary arrays exist; no convention added. Keeps all samples on a mesh and complicates selection.                                                           |
| 4   | Encoding: time components                                   | Conditional             | Multicomponent arrays exist; no temporal schema added. Must distinguish time columns from vectors/tensors.                                                   |
| 5   | Encoding: field-data payload                                | Conditional             | Field data exists; packing/extraction not implemented. Requires ownership/schema rules with no measured gain.                                                |
| 6   | Encoding: blocks as times                                   | Conditional             | MultiBlock exists; no temporal convention added. These readers use spatial blocks, so implicit timestep interpretation is wrong.                             |
| 7   | Geometry: shared connectivity and coordinates/displacements | Conditional             | Point assignment/warp exist; no new encoding added. Needs correspondence and verified reconstruction.                                                        |
| 8   | Storage: checkpoints/deltas                                 | Conditional             | Not implemented. Extra indexing, reconstruction buffers, and seeking complexity; savings unproven.                                                           |
| 9   | I/O: temporal readers/manifests                             | Suitable                | Existing TimeReader/PVD/EnSight readers used by ReaderSource. Reuses original results.                                                                       |
| 10  | I/O: external arrays                                        | Conditional             | Array wrapping exists; no xarray/Zarr/HDF5/memmap adapter added. Useful for authoritative existing storage; conversion adds disk and chunk choices.          |
| 11  | Source: callable                                            | Suitable                | Existing controller interface retained. Reads/reconstructs saved results without a new hierarchy.                                                            |
| 12  | Acquisition: live stream                                    | Unsuitable here         | Not implemented. Recording/buffering is unrelated; bounded live history alone cannot seek evicted samples.                                                   |
| 13  | Representation: tables/trajectories/events                  | Conditional             | Ordinary point/table tools exist; temporal construction not added. Appropriate only for that original representation.                                        |
| 14  | Processing: eager filtering                                 | Unsuitable as default   | Ordinary filters exist; no eager stage added. Retaining/writing a filtered sequence adds RAM/disk.                                                           |
| 15  | Processing: lazy filtering                                  | Complementary           | Normal callable plus PyVista filter; used in tests/diagnostic. Processes only the requested sample.                                                          |
| 16  | Execution: connected temporal pipeline                      | Conditional             | VTK time requests and algorithm inputs exist; full temporal propagation not implemented. Ordinary dataset filters execute eagerly; current adapter suffices. |
| 17  | Analysis: cross-time operators                              | Conditional             | Not implemented. Playback needs none; interpolation/derivatives need correspondence and scientific policies.                                                 |
| 18  | API: standalone temporal object                             | Suitable                | ReaderSource times/frame contract used. Full temporal container semantics unnecessary.                                                                       |
| 19  | API: dataset accessor                                       | Complementary, deferred | `register_dataset_accessor` exists here; no temporal accessor added. One mesh need not own changing-topology history.                                        |
| 20  | API: temporal subclass/core model                           | Unsuitable here         | Not implemented. Copy/filter/serialization semantics exceed the requirement.                                                                                 |
| 21  | I/O: reader plugin                                          | Conditional             | `register_reader` supports readers/callables here. Existing formats need no plugin.                                                                          |
| 22  | UI: plotter component                                       | Complementary, deferred | `register_plotter_component` and cleanup hooks exist. Standalone controller introduces no timer/lifecycle resources.                                         |
| 23  | Display: mapper replacement                                 | Suitable                | Existing controller hardened. Accepts changing geometry/topology/dtypes/filtered output with one actor.                                                      |
| 24  | Display: mutable dataset                                    | Conditional             | Existing mutation benchmark baseline only. Needs stable displayed topology/schema and modification notifications; no useful total-frame gain measured.       |
| 25  | Display: preloaded actors                                   | Unsuitable here         | Visibility exists; sequence convention not added. Retains datasets and potentially GPU resources per frame.                                                  |
| 26  | Scene: shared-clock tracks                                  | Conditional             | No coordinator added. Multiple independently animated scene elements are not demonstrated requirements.                                                      |

## Three combinations investigated deeply

1. FixedMeshSeries arrays + callable + lazy clip + replacement: existing baseline
   eagerly copies time arrays and static mesh; each frame deep-copies geometry and
   does not update coordinates. Its optional integer interpolation loses precision
   (synthetic midpoint `2**60 + 1`). Faster prepared seeks do not justify sequence
   RAM or packing authoritative data just for playback.
2. Complete snapshots + callable + lazy clip + replacement: existing TimeSeries
   eager/lazy baselines and ReaderSource. TimeSeries reader retrieval makes two
   deep copies; the thin adapter needs one. Lazy original-file access has lower
   measured peak RSS and avoids converted storage. **Chosen combination.**
3. Lazy originals + clip + mutable display: prior identical inputs/fields, with
   stable synthetic filtered output. No material total-frame or RSS advantage.
   Changing topology, missing arrays, dtype changes, and changing filtered output
   need extra mutation/reconstruction rules; replacement already handles them.

The array/container prototypes remain recoverable at their supplied commits and
were not copied into this branch. Original files were not removed. The existing
reader/controller are finished rather than replaced by another competing prototype.

Equal counts or unchanged connectivity do not establish physical identity. With
proven fixed correspondence, sharing connectivity can save bytes; with changing
topology, exact independent frames avoid invented mappings. Both fit the same narrow
source contract. Neither needs interpolation or reconstruction merely for playback.

## Frame ownership and peak memory

1. Original files and reader metadata remain authoritative. Readers may allocate
   decompression/reconstruction temporaries, read all enabled fields, and load all
   spatial parts before selection. EnSight has field selection; PVD recreates its
   child readers on time changes. ReaderSource does not promise selective I/O.
2. Reader output may share native buffers. The adapter makes one independent deep
   copy of the selected part before restoring time; it does not copy other parts.
   Restoration may allocate fresh child-reader metadata. Reader internals can retain
   their own output/cache independently of the adapter.
3. The representative clip creates geometry/connectivity and derived field arrays.
   `shares_memory` probes confirmed new point/scalar buffers for this clip; other
   filters may share arrays. Old display, source, filter output, reader buffers,
   and temporaries can coexist during a request.
4. Controller shallow copies share geometry/field buffers but isolate active scalar
   metadata. Mapper assignment reconnects native input. The active-scalar algorithm
   shallow-passes data on update. Previous outputs may survive until update/render.
   The conditional `Plotter.mesh` update releases the initial display reference while
   respecting another actor added later. Controller keeps only its current frame.
5. Rendering can extract surfaces, map colors, and upload GPU buffers. Process RSS
   includes native CPU allocations and allocator retention, not all GPU resources.

No frame cache, eager list, converted storage, or prefetch exists. Steady payload is
reader state + display/filter/native-render buffers; transient payload includes old
and next frames. Timestamp/manifest metadata grows with sequence length. This is
not a guarantee that arbitrary native readers retain bounded internal memory.
PyVista array writes notify VTK; writes through plain NumPy views can bypass those
notifications. Replacement avoids needing them for reconstructed frame buffers.

## Prior measurements verified from raw artifacts

The prior `experiment.py` used fresh processes, two repetitions, 16/64 samples of
the same synthetic 9,409-point/9,216-cell unstructured mesh, float64 coordinates,
point `value`, cell `stress`, and timestamps. Every mode used identical input fields,
a clip, and 480×360 rendering. OS caches were warm/uncontrolled. psutil sampled RSS
every 2 ms; macOS lifetime high-water RSS includes native CPU allocations. Timing
used `perf_counter`; there was no GPU completion fence.

The 16 raw records in `/Users/user/temporal-investigation/measurements.json`
were re-read and these median lifetime peaks recalculated:

| Prior strategy                | 16-sample peak MiB | 64-sample peak MiB |
| ----------------------------- | -----------------: | -----------------: |
| Fixed arrays + replacement    |             252.31 |             265.15 |
| Eager snapshots + replacement |             269.66 |             344.05 |
| Lazy originals + replacement  |             247.60 |             247.70 |
| Lazy originals + mutation     |             247.65 |             247.62 |

Prior first-display medians: replacement 470.41/483.57 ms and mutation
472.83/487.36 ms (16/64 samples). Sequential medians: 5.34/5.30 versus 5.35/5.47 ms;
random medians: 5.27/5.30 versus 5.28/5.32 ms. First display included construction,
GC, load, clip, plotter creation, and first show, excluding imports. Requests timed
load/clip/update/render. Raw scripts retain separate phase timings. These small
observations are not statistically significant speed rankings, current-adapter
measurements, or heart/crash benchmarks. They justify avoiding eager retention and
mutation complexity; rerunning the whole matrix was unnecessary.

All `storage.json` sizes matched actual files: 64 original VTUs total **64,339,890 B**,
existing PVD **3,824 B**, timestamp sidecar **883 B**. Playback reuses them and adds
**zero converted/cache bytes**. The earlier report's 4,592-byte manifest claim was
incorrect. Investigation-only shared NPZs are **10,141,938 B** uncompressed and
**1,012,687 B** compressed. Neither was adopted or used in timing comparisons.

Measured float64 absolute coordinates and displacements both occupy **225,816 B**
per frame. Sharing connectivity/offsets/types saves `(64 - 1) * 377864 = 23,805,432`
raw bytes if correspondence is established; sharing static coordinates saves another
14,226,408 raw bytes. This is payload arithmetic, not RSS prediction. The fixture
is unusually repetitive; its compression ratio cannot predict solver-data savings.
Reference-plus-displacement reconstruction can round; no lossy storage or reduced
precision is used here.

## New validation and reproduction

See [contracts, tutorial, and commands](READER_PLAYBACK.md). Synthetic fixtures and
cached examples establish mechanisms; they do not establish heart/crash field
semantics, physical identity, correctness, contact handling, scale, or performance.

Fresh audits confirmed NACA's 2 times/3 parts/fixed coordinates and connectivity;
wavy's 15 times/changing coordinates/fixed connectivity; dual sphere's 11 times/2
parts/changing coordinates and topology. Fields are point `DENS`, point `z`, and
point `Normals`/cell `cellNormals` vectors respectively.

Tests compare exact saved types, coordinates, connectivity, cell types, dtypes,
all associations, and large integer values against direct original VTUs. They
exercise the real native scalar pipeline and read/restore/filter/validation failure
boundaries. Over 60 seeks, weakrefs check Python reachability and native reference
counts check release of prior points, cells, and selected scalar buffers after update.

The PVD diagnostic independently reads original files and compares source frames and
filtered mapper datasets at first/middle/last, backward, and 12 seeded random seeks.
Wavy passed a real desktop render. Dual sphere part 1 passed numerical comparisons
with an explicitly derived cell-normal component. The sandbox could not create a
macOS OpenGL context; desktop access succeeded without global application changes.
Raw logs/JSON/previews remain outside Git under
`/Users/user/temporal-investigation/saved-results-final/`.

The final wavy render diagnostic measured a **255.25 MiB process lifetime peak**
(267,649,024 bytes). RSS before/after four full playback cycles was
267,436,032/267,452,416 bytes, a 16,384-byte difference over 60 selections.
First display was 88.47 ms and four cycles 150.51 ms. These timings exclude imports,
reader construction, and plotter creation; they include read/copy/clip/mapper/show
or update/render. RSS also includes independent validation reads and screenshot
readback. These boundaries differ from the prior benchmark, so these numbers are
not a before/after speed comparison. OS caches were warm/uncontrolled; GPU completion
was not fenced. Wavy originals plus manifest occupy 1,188,600 bytes; application
converted/cache result storage is zero. The final non-rendered dual-sphere check
peaked at 199,393,280 bytes and read 109,363 original bytes including both parts and
manifest. Tiny examples cannot establish large-reader memory bounds.

Independent review identified implicit RGB-name inference, theme-created auxiliary
geometry, the initial plotter reference, and endpoint-distance overflow. These were
resolved. An additional small desktop probe passed numeric comparisons through
PolyData/unstructured-grid and int64/float32 transitions with scalar `value_rgb` and
a local theme enabling smoothing, silhouettes, and vertices. Only the intended one
mesh actor remained; no global theme or application state was changed. The native
scalar pipeline does support PointSet conversion; the narrower explicit-cast policy
here instead preserves exact source/display type and ownership comparisons.

### Local checks

- 31 focused tests pass; 12 prior mechanism tests pass; 4 upstream PVD-reader tests
  pass with one worker. Ruff check and formatting pass.
- Three diagnostic mutations were tested outside the branch: shallow reader copies,
  skipped timestep selection, and retained initial plotter references. Each made
  its targeted correctness/retention test fail. Test files and failing logs are in
  `saved-results-final/{alias,stale,retained}-mutant*`; branch code was not altered.
- Static doctest-name checking passes: 3,908 examples, zero failures. The tox wrapper
  with `--skip-env-install` hit an existing `toxfile.py` constraint initialization
  error, so its exact command was run directly using the existing environment:
  `PYTHONPATH=. ../pyvista/.tox/doctest-names/bin/python tests/check_doctest_names.py`.
- Full Vale/docstyle passes using the already installed environment:
  `PATH=/private/tmp/pyvista-notebook-verify/bin:$PATH make docstyle`.
  The primary environment lacks Vale. The full package runtime-doctest suite and
  documentation build were not run: this patch changes standalone integration,
  and focused execution plus real rendering keeps validation proportional and RAM low.
- Repository `make lint` required desktop access because taplo panicked accessing
  macOS system configuration in the sandbox. Its first desktop run reformatted
  Markdown/snippets; the final full run passed (`lint-final.log`).

Full lint invocation (reuses the existing environment without syncing dependencies):

```sh
UV_PROJECT_ENVIRONMENT="$PWD/../pyvista/.venv" UV_NO_SYNC=1 make lint
PYTHONPATH=. ../pyvista/.tox/test-core/bin/python -m pytest -q -n 0 \
  tests/core/test_reader.py -k pvd
```

To repeat prior correctness checks without generating data:

```sh
PYTHONPATH=. ../pyvista/.venv/bin/python -m pytest -q \
  /Users/user/temporal-investigation/test_mechanisms.py
```

For a single prior performance cell, use desktop access and
`PYTHONPATH=. ../pyvista/.venv/bin/python /Users/user/temporal-investigation/experiment.py lazy 16`.
Use `arrays`, `eager`, or `mutable` and count `64` only if needed. Do not regenerate
existing fixtures merely to repeat a benchmark.
