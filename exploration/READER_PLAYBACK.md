# Saved-result playback

`ReaderSource` and `TimeController` are standalone **project APIs**, not public
PyVista temporal APIs. They display saved results through one ordinary mesh actor.
There is no physics engine, temporal interpolation, cache, converted storage, or
PyVista core change. See [designs and evidence](TEMPORAL_DECISIONS.md).

## Quick start

Create and display a mesh exactly as in the
[PyVista tutorial](https://tutorial.pyvista.org/):

```python
import pyvista as pv

mesh = pv.Sphere()
mesh.plot()
```

Run this in Python with PyVista installed. Mesh creation, filters, and `mesh.plot()`
are ordinary public PyVista APIs and need no playback imports or setup.

To try a time slider without files or downloads, run this complete example from
this worktree's root:

```python
import pyvista as pv
from exploration import TimeController  # Project API


def make_frame(index):
    mesh = pv.Sphere(radius=0.5 + 0.1 * index)
    mesh['height'] = mesh.points[:, 2]
    return mesh


pl = pv.Plotter()
controller = TimeController(pl, [0.0, 0.2, 1.0], make_frame, scalars='height', clim=(-1, 1))
pl.show()
```

Drag the time slider to change the displayed mesh. Only the requested frame is
created; the controller keeps one displayed frame. These generated spheres are a
small UI demonstration, not a physical simulation. `height` supplies the point
values for coloring, and `clim` fixes the color scale across samples. Use ordinary
`pl.add_mesh(...)` calls for any static context before `pl.show()`.

The same example is available as a script:

```sh
../pyvista/.venv/bin/python exploration/time_controller.py
```

## Display saved results

From this worktree's root, using the existing environment:

```sh
../pyvista/.venv/bin/python -m exploration.play_reader \
  "$HOME/Library/Caches/pyvista_3/PVD/wavy.zip.unzip/wavy.pvd" \
  --block 0 --scalars z --clim -1 1
```

Drag the slider to select a saved simulation sample. This is a cached wavy surface,
not a validation of future heart or crash results.

The original `python exploration/play_reader.py ...` command also works. Both
entry points close their plotter if loading or display fails.

Run this short application from the repository root:

```python
import pyvista as pv
from exploration import TimeController  # Project API

pl = pv.Plotter()  # Public PyVista API
controller = TimeController.from_reader(
    pl,
    pv.get_reader('results.pvd'),
    block=0,
    scalars='temperature',
    clim=(0, 100),
)
pl.show()
```

Use `preference='cell'` explicitly for a cell scalar, including when the same name
exists in both associations. Fixed explicit `clim` avoids scanning the sequence at
startup. Category-specific colors and vector display are outside this controller's
contract; color mapping is ordinary numeric scalar mapping.

## Contracts

- Sources expose finite, strictly increasing floating-point `times` and `frame(i)`.
  Simulation time is separate from wall-clock playback speed. `frame(i)` and
  `controller.select_index(i)` select the exact saved sample. Negative, out-of-range,
  boolean, and non-integral indices raise; NumPy integer indices work.
- `select_time(t)` and the slider choose the nearest saved sample, clamp endpoints,
  and choose the earlier sample for equal floating-point distances. Selecting the
  current index does not reload. The selected saved time is
  `controller.times[controller.index]`. The slider thumb shows the requested time
  and is not synchronized after programmatic selection or a failed request.
- ReaderSource uses a dedicated temporal reader synchronously: select time, check
  the selection, read, choose a spatial part, validate the named association, make
  **one deep copy**, then restore previous time in `finally`. Returned datasets
  independently own geometry, topology, arrays, and metadata and may be retained
  or modified. Times are authoritative in `source.times`; file metadata is preserved.
- Reader restoration covers success and failure. Restoration failures propagate;
  a broken reader may need recreation. Internal reader caches are reader-owned.
  Array selections are not changed or restored. Do not share the reader across
  threads, sources, or external time-changing code. Use the plotter's thread.
- Every MultiBlock output requires an explicit top-level `block` index or name,
  even for one part. Blocks are spatial parts, never implicit timesteps. Null or
  nested composite parts fail; non-composite output rejects a block selector.
  The key must identify the intended spatial part at every time.
- The controller supports nonempty ordinary PyVista datasets with a one-component
  real numeric point or cell scalar (integer, unsigned integer, or float). Counts,
  geometry, connectivity, dtype, and dataset types can change. PointSet requires
  explicit `cast_to_polydata()` before display: initial `add_mesh` changes that type,
  breaking exact source/display type comparisons. Empty filtered outputs, composites,
  absent arrays, strings, complex arrays, and vectors fail before replacement.
  Numeric NaNs remain intact for PyVista's usual missing-value color mapping.
- A custom loader must provide buffers that remain valid and unchanged while
  displayed. The controller shallow-copies dataset/attribute containers to isolate
  active scalar metadata and shares arrays; it adds no second deep copy. Editing a
  retained source frame that shares display buffers changes those buffers; deep-copy
  it first for independent edits.
- Read, restore, filter, and validation failures preserve the actor, displayed
  dataset, fixed color range, and selected index. Renderer/driver failures after
  committing mapper input are not rolled back. One actor and camera stay in place.
  Automatic RGB inference, smooth shading, silhouettes, and extra vertex actors
  are explicitly disabled so first and replacement frames use consistent paths.
- Volume display, multiple spatial actors, and derived overlays need their own
  display adapter. Plotter owns ordinary actor/widget cleanup (`pl.close()`). Keep
  the controller alive while using it; no timers or playback threads are installed.

Configure enabled fields on the dedicated reader where its API supports persistent
selection. ReaderSource copies every enabled array in the selected part and may
read all parts first. PVD recreates child readers on time changes; configuring only
today's child reader does not establish persistent selective I/O.

## Filtering and scripted animation

A callable is enough for a project-specific spatial filter:

```python
from exploration import ReaderSource  # Project API

source = ReaderSource(pv.get_reader('results.pvd'), block=0, scalars='temperature')


def load_frame(index):
    return source.frame(index).clip(normal='x', origin=(0, 0, 0))


controller = TimeController(pl, source.times, load_frame, scalars='temperature', clim=(0, 100))
```

Choose a spatial operation valid for the fields. Clipping creates cut points and
derived point values; extraction may be preferable for categorical point state.
Neither establishes physical correspondence across times.

For a simple scripted animation, replace `pl.show()` with:

```python
import time

pl.show(auto_close=False, interactive=False)
try:
    for index in range(len(controller.times)):
        controller.select_index(index)
        time.sleep(0.05)  # Wall-clock pause; simulation timestamps stay unchanged
finally:
    pl.close()
```

This visits every saved sample. Reading and rendering add to each frame's duration;
it is not a real-time scheduler. No IDs, categories, coordinates, or fields are
temporally interpolated.

## Reproduce validation

From the worktree root:

```sh
PYTHONPATH=. ../pyvista/.venv/bin/python -m pytest -q exploration/test_reader_source.py
../pyvista/.tox/test-core/bin/ruff check exploration
../pyvista/.tox/test-core/bin/ruff format --check exploration

PYTHONPATH=. ../pyvista/.venv/bin/python exploration/validate_playback.py \
  "$HOME/Library/Caches/pyvista_3/PVD/wavy.zip.unzip/wavy.pvd" \
  --scalars z --render --output /private/tmp/wavy-playback.json

PYTHONPATH=. ../pyvista/.venv/bin/python exploration/validate_playback.py \
  "$HOME/Library/Caches/pyvista_3/PVD/paraview/dualSphereAnimation.zip.unzip/dualSphereAnimation.pvd" \
  --block 1 --scalars cellNormals --preference cell --component 0 \
  --output /private/tmp/dual-playback.json
```

The diagnostic additionally needs pytest, psutil, and defusedxml, all present in the
existing environment. It independently parses the PVD manifest and uses `pv.read`
on original part files. It compares geometry, connectivity, dtypes, associations,
all fields, retained frames, and filtered mapper datasets at first/middle/last,
backward, and seeded random requests. `--component 0` explicitly derives a display
scalar from the dual-sphere cell-normal vector; original vectors remain unchanged
and are compared exactly. `--render` executes real `show()`/`render()` and writes
a preview next to the JSON. This Mac needs desktop OpenGL access outside the sandbox.

Small synthetic tests cover irregular times, large integer IDs, co-named point/cell
fields, missing data, aliasing, stale reads, restoration failure, changing types,
and retention. They execute a real mapper/native scalar pipeline without a window.
Weakrefs check Python reachability; native reference-count diagnostics check old
points/cells/scalars after update. RSS includes native CPU allocations in the
example diagnostic. The render is an integration check, not an image-regression
baseline or evidence of heart/crash correctness or performance.
