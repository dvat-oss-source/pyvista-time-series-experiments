# General time series container exploration

Branch: `feat/time-series-container`. This is an executable design experiment,
not a proposed public API or a production implementation.

## Proposed shape

```python
import pyvista as pv

from exploration.time_series import TimeSeries

series = TimeSeries.from_datasets([0.0, 2.0], [pv.Sphere(), pv.Cube()])
frame = series.at_time(2.0)
first = series[0]
last = series[-1]

# A reader obtained with pv.get_reader(...) can supply frames lazily.
series = TimeSeries.from_reader(reader)
```

A series owns an ordered time axis and loads one independent data object per
request. This includes MultiBlock output, adaptive meshes and changing topology.
It neither subclasses a dataset nor interprets blocks as times. A dataset can
already represent spatial blocks; a separate temporal axis preserves that meaning.

`times` is an immutable tuple of finite, strictly increasing floating-point
values. Duplicate and descending timestamps are rejected instead of silently
reordering a reader's frame indices. Indexing uses integers, including negative
indices; time selection is explicit and exact. Missing times raise KeyError.
Nearest selection, tolerances, interpolation, units and calendar timestamps need
separate policies and are intentionally absent.

## Ownership and loading

In-memory construction deep-copies the supplied datasets; each access returns
another deep copy. Mutating either the caller's input or a returned frame cannot
change a later access. This deliberately favors correctness over memory savings.
Reader construction captures timestamps without reading datasets. Each access
selects a time, reads and detaches its output, then restores the reader's previous
time even when reading fails. Detaching before restoring matters for readers that
reuse buffers. Generic callbacks must return data with a PyVista-style
`copy(deep=True)` operation. Production code should validate DataObject types.

There is no cache. Reader-backed memory stays roughly proportional to a frame,
with an extra deep copy during access; repeated requests perform repeated reads.
Reader loads currently copy twice: once before restoring reader selection and once
at the common access boundary. Cross-review identified this as a performance cost;
a production adapter should unify ownership without restoring before detachment.
In-memory series cost approximately the sum of all frame sizes plus returned
frames. A later cache must be bounded and have explicit invalidation semantics;
if persisted, use PyVista's native `.pv` format for individual snapshots.

`TimeReader` exposes time metadata and selection, while concrete readers supply
`read()`. The adapter composes these interfaces without changing existing readers.
Some readers need additional state beyond active time (array selection, time-set
selection, file state). Restoring active time is therefore a limited guarantee,
not a claim of full reader transactional safety. Use a dedicated reader; access is
not thread-safe. Source files and reader settings must remain stable for the
series lifetime. An owned reader factory could provide stronger isolation later.

## What this path enables

A shared temporal object could feed both fixed-mesh data and plotting controllers.
Frame-local filters remain explicit (`series[i].clip(...)`); later lazy mapping
could preserve the time axis. Temporal reductions and interpolation need topology
and correspondence checks, so those do not belong in the minimal container.

The main cost is introducing a new object users must learn, along with memory and
lifetime policies that MultiBlock alone does not solve. Start with integer/time
selection plus a dedicated-reader adapter; add plotting as a separate consumer.
This provides the broadest foundation if changing geometry is a real use case.
For fixed geometry alone, the arrays branch can store much less data.

## Validation

Run `python3 exploration/test_time_series.py`. Dependency-free stand-ins check
ownership, changing payload sizes, negative indices, exact lookup, invalid times,
invalid indices, lazy reads, reader selection restoration, output buffer reuse
and failed reads. These establish temporal semantics; they do not replace real
PyVista reader/dataset integration tests.

Actual PyVista import is blocked by the environment's installed
`pyvista_validation.check` lacking `_is_real`. `make lint`, `make doctest`, and
`make test-core ARGS='exploration/test_time_series.py'` were attempted offline;
uv cannot resolve uncached `setuptools>=77.0.3`. `make docstyle` was attempted and
reports Vale missing. No dependencies were installed and no CI was triggered.
`python3 -m compileall -q exploration` passed.
`python3 -m pytest --noconftest -o addopts='' exploration/test_time_series.py`
was also attempted; this Python has no pytest installed.
Before production, run real ImageData/PolyData/MultiBlock and file-backed reader
tests and all repository gates in a compatible development environment.
