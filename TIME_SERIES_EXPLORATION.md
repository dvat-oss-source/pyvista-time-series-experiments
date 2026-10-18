# Fixed-mesh temporal arrays

Branch: `feat/time-series-arrays`. This is an isolated experiment, outside the
public package. It adds no `pv.TimeSeries` API and registers no global accessor.

## API to try

```python
import numpy as np
import pyvista as pv
from exploration.fixed_mesh_series import FixedMeshSeries

mesh = pv.Plane(i_resolution=2, j_resolution=1)
series = FixedMeshSeries(
    mesh,
    times=[0.0, 0.5, 2.0],
    point_data={'temperature': np.arange(18).reshape(3, 6)},
    cell_data={'velocity': np.arange(18).reshape(3, 2, 3)},
)
frame = series.at_time(0.5)
interpolated = series.at_time(1.0, method='linear')
```

The time axis is always first. Scalars have shape `(n_times, n_tuples)`;
three-component vectors have shape `(n_times, n_tuples, 3)`. Explicit point and
cell dictionaries resolve ambiguity even if both associations have equal tuple
counts or the same field name. Other component counts, complex data, strings,
booleans, and field data are outside this prototype. Vector arrays are stored as
three-component arrays; choosing active vectors/scalars stays an explicit user
operation with existing PyVista methods.

Times must be nonempty, finite, unique, and strictly increasing. Exact selection
is the default and uses exact numeric equality. Linear selection is opt-in,
uses physical time spacing, and rejects extrapolation. Each vector component is
interpolated separately; this does not preserve vector magnitude or categorical
integer meaning. Integer data becomes floating point between samples, so callers
must avoid linear selection for IDs or material labels. Static arrays and
geometry remain unchanged. Units, calendars, tolerances, and nearest selection
need a separate API decision.

## Ownership and cost

The container takes a deep copy of the base mesh and copies every temporal array.
Mutating constructor inputs cannot change its frames. Every selected frame is
also an independent deep copy, so filters and editing a frame cannot corrupt
other time steps. The `times` property returns a copy. A temporal field replaces
a static field of the same name only within its association.

Storage is roughly `geometry + static arrays + T * temporal arrays`, instead of
`T * (geometry + all arrays)`. Constructor copies temporarily increase peak
memory, and each selected frame copies geometry and its active fields. This
prioritizes predictable ownership over maximum playback speed. A shared readonly
geometry representation, caller-managed buffers, memory mapping, and lazy array
providers should be measured before adding a `copy=False` option.

## Limits and integration

All frames use identical coordinates, connectivity, point order, and cell order.
Matching point/cell counts in independently generated meshes would not prove
identity. Remeshing, moving geometry, and changing topology belong in a general
dataset-series container. A displacement field can be selected then passed to
`warp_by_vector`; this prototype does not automatically deform geometry.

The natural integration is a fixed-mesh storage backend for a general temporal
container with `times` and `at_time`. A plotting controller could consume that
protocol and update `actor.mapper.dataset`. A dataset accessor could provide a
constructor convenience, but attaching the entire time cube to ordinary
`point_data` would violate its tuple semantics. Filters should operate on
selected frames; arbitrary filters cannot safely retain the temporal storage
without a rule for remapping all samples. Temporal means need explicit sample
versus time-weighted semantics. Serialization needs a series manifest; independent
mesh caches should use `.pv`, with a chunked store considered for large arrays.

Recommendation: retain this as a specialized backend rather than the only
representation of time series. Its strongest use case is simulation output with
stable topology and many scalar/vector samples; it offers substantial storage
savings with a small conceptual API.

## Validation

Run from the worktree root:

```sh
python3 -m pytest -q -o addopts='' exploration/test_fixed_mesh_series.py
```

The behavioral checks cover irregular-time scalar/vector interpolation, explicit
associations, static arrays, source and returned-frame ownership, invalid shapes,
invalid times, exact-time failures, and extrapolation rejection. Python compilation
and `git diff --check` pass. Runtime tests cannot start because system Python has
no pytest; importing the checkout also encounters the installed
`pyvista_validation` version mismatch (`_is_real` cannot be imported).

Required gates were attempted with `UV_OFFLINE=1` and a temporary uv cache to avoid
network installs. `make lint`, `make doctest`, and scoped `make test-core` stop at
uncached `setuptools>=77.0.3`. `make docstyle` stops because Vale is absent. None of
these gates passed, so this branch is an exploration rather than merge-ready work.

Additional executed smoke verification: `python3 exploration/smoke_fixed_mesh_series.py`
passes using installed NumPy plus a minimal in-memory mesh double, with no PyVista
or pytest dependency. It executes scalar/vector interpolation, irregular time
spacing, association handling, source/frame/timestamp ownership, and invalid
times/shapes/dtypes/methods/selections. This verifies the NumPy logic only; real
PyVista dataset assignment and VTK buffer ownership remain unverified. Query
selection currently expects a scalar numeric time; array-shaped query validation
would need tightening before any public API proposal.
