# Path 3: plotting time controller

This branch explores a plotting-first feature using existing public PyVista APIs.
The prototype lives outside the package; it adds no `pv` or `Plotter` API.

## Try it

With a working checkout and its dependencies:

```sh
python3 -m exploration.time_controller
```

The demonstration shows three frames at irregular times `0`, `0.2`, and `1`.
A controller creates one actor and a slider. Sliding chooses the nearest sample
by physical time, clamps outside the range, and chooses the earlier sample on
a tie. It reads only the requested frames and avoids repeat reads when the
selected index does not change. A single frame needs no slider.

```python
from exploration.time_controller import TimeController
import pyvista as pv

pl = pv.Plotter()
controller = TimeController(
    pl,
    times=[0.0, 0.2, 1.0],
    load_frame=frames.__getitem__,
    scalars='temperature',
    clim=(-0.5, 1.5),
)
pl.show()
```

`frames` is a caller-provided sequence of PyVista datasets with a point array
named `temperature`. The retained `controller` can also select a frame with
`controller.select_time(0.9)`.

## Reuse existing machinery

`TimeReader` already exposes `time_values`, `active_time_value`, and time-point
selection methods. `TimeController.from_reader(pl, reader, scalars=..., clim=...)`
adapts a temporal reader returning one dataset. Each read borrows the reader's
time selection, copies the output, and restores its prior active time even when
reading raises. This prevents borrowed reader buffers changing the displayed
frame on a subsequent read. The reader remains synchronous and must not be
shared between concurrent tasks during this operation.

The plotting pieces already exist: `Plotter.add_mesh`, `add_slider_widget`,
`actor.mapper.dataset`, `mapper.scalar_range`, and `Plotter.render`. Replacing
the mapper input retains actor styling and the user's camera. Every successful
selection reapplies explicit fixed color limits; the controller does not scan
an entire potentially lazy series to discover a global range.

## What this experiment establishes

A small adapter can add time navigation without waiting for a universal temporal
container. It can consume either future container path through an indexed loader.
The mapper-input strategy allows changing topology between frames without
recreating actors. Point scalars are selected consistently by name, and invalid
frames are checked before changing the displayed dataset.

The slider is a continuous time control snapping displayed data to samples. Its
handle still shows the requested continuous time, which can differ from the
selected frame's timestamp. A production widget should display both the actual
sample time and index, or use a discrete index slider with actual-time labels.

## Deliberate limits and next steps

- This is a surface dataset and point-scalar prototype. Composite data, volumes,
  cell data, missing arrays, changing component counts, and scalar-bar updates
  need a deliberate policy and real mapper integration tests.
- Physical time must be finite and strictly increasing; duplicate timestamps and
  datetime values need explicit semantics. No interpolation is performed.
- Changes execute synchronously on the UI thread at the slider's end event.
  Larger readers need cancellation, prefetching, caching, and visible read errors.
  No threads or application-global rendering state are changed here.
- Playback should use PyVista's existing timer events and derive delays from
  `(times[i + 1] - times[i]) / speed`, with pause and close cleanup. Equal delay
  per frame would misrepresent irregular timestamps. Playback is not implemented.
- Movie writers use a fixed frame rate, so export must explicitly choose constant
  steps or resample physical time, holding frames when necessary. GIF timing and
  interpolation deserve separate options. Export is not implemented.
- Integrate as `pl.add_time_series(...)` only after agreeing on a source protocol,
  widget lifecycle, and whether the returned object owns actors and readers.

## Validation

`python3 -m exploration.test_time_controller` passes five dependency-free
behavioral checks, covering nearest selection for irregular timestamps, tie and
endpoint policies, frame loading, stable actor and camera references, fixed
color limits, invalid timestamps and selections, missing point arrays, reader
restoration after success and failure, and the single-frame case. These doubles
test controller decisions; they do not prove rendering or VTK mapper behavior.
The runnable demonstration ends in `pl.show()`; no image baseline was generated.

Required `make lint`, `make doctest`, and scoped `make test-plotting` were attempted.
Their default UV cache was inaccessible in the sandbox. Repeating with a local
cache and offline mode reached dependency resolution, but `setuptools>=77.0.3`
was unavailable in that cache. `make docstyle` reports Vale is not installed.
The system Python cannot import this checkout because the installed
`pyvista_validation.check` lacks `_is_real`; pytest is also absent. These gates
and a real render remain unverified. No dependencies were downloaded and no CI
was used.

## Recommendation

Use this as the fastest user-facing prototype: it demonstrates a useful feature
without committing to how temporal datasets are stored. Build its production
version on the container path's small read-by-index contract, while keeping the
fixed-mesh path as an optimized source. Gate a public API on real scalar mapper,
widget cleanup, and image-regression coverage.

## Notebook verification

Open `exploration/test_time_series_controller.ipynb` and run all cells in order.
All default code cells passed in a fresh Jupyter kernel using Python 3.13,
NumPy 2.2.6, VTK 9.7.1, this branch's PyVista 0.50.dev0, and the repository-pinned
`pyvista-validation==0.5.4`. The compatible dependency was installed into
`/private/tmp/pyvista-notebook-verify`, a temporary virtual environment inheriting
the machine's existing packages. The global environment was not modified.

The notebook includes saved execution counts, successful assertion outputs, and
one static PNG figure, inspected after execution. No downloads of example data
are needed. Notebook schema and code compilation checks pass.

Real mapper checks exposed a frame-wrapper lifetime issue: the controller retains
the displayed frame and uses a dynamic mapper. Assertions cover displayed scalar
values and points, stable actor/camera/color limits, lazy selection and changed,
nonblank rendered pixels. The test surface avoids an edge-on final view. These
pixel checks are smoke checks, not comparisons against an image baseline.
The optional desktop slider cell remains disabled during automated execution;
manual mouse interaction has not been verified. Cross-review found no blockers.

Repository-wide lint, docstyle and doctest gates remain blocked when reattempted
offline: the uv cache lacks `setuptools>=77.0.3`, and Vale is absent. Notebook
execution passing does not establish that these broader gates pass.
