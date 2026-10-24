# VTK temporal classes versus discrete playback

## Decision

No VTK temporal class is integrated into `TimeSeriesPlayer` or
`TimeReaderSource`. The current player selects and displays exact stored
samples; its caller drives wall time. The heart example reads ten independent
`.pv` meshes, and the tested `PVDReader` does not expose a VTK algorithm output
port. Applying a VTK temporal filter to either source would require a new
pipeline bridge. A cache reduces reads for repeated nearby seeks once such a
bridge exists, but adds deep-copied resident frames and did not help a forward
scan. That tradeoff does not justify adding the bridge or a cache contract to
this playback API.

## Raw inputs and implementation paths

- The original CONRAD archive contains `phase_0.ccm` through `phase_9.ccm`.
  They are ten independent phase consensuses, not a VTK temporal pipeline.
  The archive is 78,152,158 bytes; the notebook's ten `.pv` frames total
  5,307,414 bytes. A fresh installed-wheel check decoded each original phase
  and matched its points, triangles, `part` labels, and phase scalar against
  its independent `.pv` frame. No heart file was rewritten.
- The reader experiment used `wavy.pvd` and its fifteen original `.vts` files
  (1,187,167 bytes in total). The PVD file lists one spatial part at each
  time 0 through 14. In PyVista, `_PVDReader` parses this XML, creates one
  ordinary reader per active part, and reads it into a `MultiBlock`. Its
  `reader` has no `GetOutputPort`; it cannot be connected directly to
  `vtkTemporalDataSetCache`. `TimeReaderSource.frame` deep-copies the selected
  part before restoring the reader time.
- The player sets `actor.mapper.dataset` to one shallow copy of the requested
  frame and updates `Plotter.mesh` if it names the old frame. The mapper can
  retain its current input. During replacement, the old frame and new load
  coexist through rendering. After the method returns, the player owns one
  displayed frame; the caller, reader, and VTK may own others.
- The VTK behavior below was checked against installed VTK 9.7.1 and the
  matching [VTK source tree](https://github.com/Kitware/VTK/tree/v9.7.1).
  All ten native classes are present in that installation, but none is
  currently re-exported through PyVista's `_vtk` module.
  In particular, [the cache implementation](https://github.com/Kitware/VTK/blob/v9.7.1/Filters/Hybrid/vtkTemporalDataSetCache.cxx)
  defaults to ten entries, deep-copies normal-memory inputs into the cache,
  shallow-copies cached data to output, and can retain an ejected entry in
  addition to its configured entries. Its upstream
  [tests](https://github.com/Kitware/VTK/blob/v9.7.1/Filters/Hybrid/Testing/Cxx/TestTemporalCacheSimple.cxx)
  connect it to VTK algorithms and count source executions.

## Candidate assessment

“Pipeline” means an upstream VTK algorithm advertises and answers time-step
requests. VTK sources instead create such a pipeline; none reads the heart
files by itself.

| Class                              | Concrete use                                                      | Pipeline    | Effect on exact frames, ownership, and memory                                                                                                                                                 |
| ---------------------------------- | ----------------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vtkTemporalStatistics`            | Mean, extrema, or deviation across samples                        | Required    | Produces derived arrays after visiting multiple times. It weights samples equally despite irregular spacing; it does not select an original frame. Accumulators add memory.                   |
| `vtkTemporalPathLineFilter`        | Tracks identified particles across samples                        | Required    | Creates trails, changing the displayed geometry. Trails retain temporal history; the heart workflow has no particle-tracing request.                                                          |
| `vtkTemporalShiftScale`            | Transforms pipeline time coordinates or extends them periodically | Required    | Changes advertised/requested time, not wall-clock scheduling. Its periodic end handling estimates a gap when source steps are irregular; the player's explicit `cycle_end` is less ambiguous. |
| `vtkTemporalSmoothing`             | Averages fields in a sliding sample window                        | Required    | Reads neighboring samples and changes scalar values, violating exact stored-frame display. Its window adds transient data and work.                                                           |
| `vtkTemporalSnapToTimeStep`        | Maps a pipeline time request to a discrete input time             | Required    | Its nearest mode duplicated `seek_time` on the tested irregular times, including earlier-sample ties. It does not provide exact index ownership or external ticks.                            |
| `vtkTemporalArrayOperatorFilter`   | Computes a result from arrays at two times                        | Required    | Produces derived arrays on the first time's mesh. It requires multiple samples and does not solve display selection.                                                                          |
| `vtkTemporalFractal`               | Generates temporal AMR test data                                  | Creates one | Does not read the source files; its composite output is outside the player's one-dataset loader contract.                                                                                     |
| `vtkSpatioTemporalHarmonicsSource` | Generates temporal image data                                     | Creates one | Can supply synthetic tests, but does not load the heart or reader files. No playback need is met.                                                                                             |
| `vtkTemporalDelimitedTextReader`   | Reads time-indexed CSV columns                                    | Creates one | Produces a table, not the spatial dataset the player displays. Converting tables to geometry would be another feature.                                                                        |
| `vtkTemporalDataSetCache`          | Reuses previously requested pipeline steps                        | Required    | Preserves values on a cache hit, but owns up to its configured deep-copied entries plus pipeline/ejected outputs. It cannot attach to this PVD reader or the `.pv` loader without a bridge.   |

The [statistics](https://vtk.org/doc/nightly/html/classvtkTemporalStatistics.html),
[pathline](https://vtk.org/doc/nightly/html/classvtkTemporalPathLineFilter.html),
[shift/scale](https://vtk.org/doc/nightly/html/classvtkTemporalShiftScale.html),
[smoothing](https://vtk.org/doc/nightly/html/classvtkTemporalSmoothing.html),
[snap](https://vtk.org/doc/nightly/html/classvtkTemporalSnapToTimeStep.html),
[array operator](https://vtk.org/doc/nightly/html/classvtkTemporalArrayOperatorFilter.html),
[fractal](https://vtk.org/doc/nightly/html/classvtkTemporalFractal.html),
[harmonics](https://vtk.org/doc/nightly/html/classvtkSpatioTemporalHarmonicsSource.html),
and [CSV reader](https://vtk.org/doc/nightly/html/classvtkTemporalDelimitedTextReader.html)
references describe those outputs. The source and upstream tests were read at
the installed VTK version; the installed classes were exercised directly for
the three plausible playback candidates.

## Executable comparison

Each row below was a fresh process using a normally installed, non-editable
PyVista wheel under Python 3.13.6 with `python -I`, outside this checkout and
without `PYTHONPATH`. The wheel hash, executable output, read index sequences,
per-cycle RSS, and screenshot hashes are in the [raw results](temporal_vtk_evaluation_3136.json).
The research probe used the existing `.vts` files in two ways:

1. **Reader:** `TimeReaderSource(PVDReader, block=0).frame` supplied the player.
2. **Pipeline/cache:** a small experimental VTK temporal source read those same
   files on demand. With or without `vtkTemporalDataSetCache`, its output was
   deep-copied before giving it to the same player, preserving the reader
   adapter's independent-frame ownership. This bridge was measurement-only.

The probe ran thirty cycles of a 15-sample forward scan and thirty cycles of
`0, 1, 2, 1, 0, 2`. Counts include the initial load and final injected-clock
checks. Independent reference reads used for verification are excluded from
source-read counts. Every displayed frame in the first cycle matched an
independent file read for points, structured-grid dimensions, and `z` scalars.
All first-cycle rendered screenshot hashes matched the reader baseline. An
injected failed load preserved the mapper dataset, selected index, and rendered
image in all modes. Injected-clock play, tick, pause, and resume checks also
passed.

| Workload | Mode     | Source reads | Logical compressed bytes | Peak RSS MiB | Cycle 1→30 RSS MiB | Post-close RSS MiB |
| -------- | -------- | -----------: | -----------------------: | -----------: | -----------------: | -----------------: |
| Forward  | Reader   |          453 |               35,852,406 |       238.06 |              +0.77 |             238.03 |
| Forward  | Pipeline |          453 |               35,852,406 |       238.02 |              +0.81 |             237.98 |
| Forward  | Cache 2  |          453 |               35,852,406 |       239.27 |              +0.78 |             239.25 |
| Forward  | Cache 10 |          453 |               35,852,406 |       239.89 |              +0.84 |             239.88 |
| Revisit  | Reader   |          183 |               14,486,556 |       237.80 |              +0.70 |             237.77 |
| Revisit  | Pipeline |          183 |               14,486,556 |       238.34 |              +1.23 |             238.33 |
| Revisit  | Cache 2  |          122 |                9,655,818 |       239.05 |              +0.70 |             239.03 |
| Revisit  | Cache 10 |            5 |                  395,784 |       239.20 |              +0.67 |             239.20 |

An additional irregular-time probe exposed three existing files at times
`(0, 0.25, 1.5)`. `vtkTemporalSnapToTimeStep` chose indices 0 and 1 at exact
midpoints 0.125 and 0.875, matching the player's earlier-tie rule. A
`vtkTemporalShiftScale` scale of 4 advertised `(0, 1, 6)` and requesting 1
read source index 1. Neither provided a missing player behavior. The matching
[VTK snap test](https://github.com/Kitware/VTK/blob/v9.7.1/Filters/Hybrid/Testing/Python/TestTemporalSnapToTimeStep.py)
also checks nearest-step behavior.

The probe files used for this local evaluation are
`/private/tmp/temporal_vtk_probe.py`,
`/private/tmp/temporal_timeline_probe.py`, and
`/private/tmp/heart_verify.py`. They are research-only VTK code; the installed
PyVista package and notebook were not changed. The cache probe can be rerun
with, for example:

```sh
/private/tmp/pv-installed-3136/bin/python -I /private/tmp/temporal_vtk_probe.py \
    cache10 revisit 30 /private/tmp/cache10-revisit.json
```

This experiment shows a read-count benefit for local repeated seeks, not a
general playback improvement. Logical bytes sum file sizes for each counted
source read; they are not physical disk bytes because macOS did not expose
per-process I/O counters here and the OS page cache was uncontrolled. The
2 ms RSS sampler and OS high-water mark cover process CPU memory, including
Python and native allocations, but do not isolate cached arrays from render
allocations or account for GPU memory. Thirty cycles showed no unbounded RSS
growth in this workload; they cannot prove leak freedom. No latency target is
claimed. The [heart comparison](installed_measurements_3136.json) remains the
same installed-wheel direct-versus-player baseline on the independent `.pv`
frames.

That heart baseline used fifty cycles on the same ten files: the direct
notebook-style loop made 501 reads and peaked at 258.91 MiB RSS; the player
made 500 reads and peaked at 258.64 MiB. Both traces leveled off, and that
small RSS difference was not consistent across earlier runs. A VTK cache
would add another owner of full heart frames to this already bounded player
path without a demonstrated end-to-end advantage.

## Local gates

`make lint`, `make docstyle`, `make typecheck`, scoped core tests covering the
player, original heart data, and reader adapter, and the scoped plotting test
all passed. Pre-commit hooks also passed on this new report and its untracked
JSON artifact. The test targets were run sequentially. `make doctest` ran
package-wide: 1,714 passed, 3 skipped, and the same three pre-existing
failures appeared in `PolyData.point_normals`, `Transform.decompose`, and
`PolyDataFilters.collision`. The affected core files are unchanged in this
worktree. The separate doctest name check passed all 3,924 cases. No CI run
was used as a test runner.
