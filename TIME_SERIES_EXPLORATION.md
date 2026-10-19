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

## Verification

The real-data tests and executed notebook pass. The scoped repository target
passes 26 cases on this branch, and full lint and documentation-style checks
pass. The complete docstring suite passes 1716 examples with 3 skips after the
shared doctest corrections. See [VERIFICATION.md](VERIFICATION.md) for counts, coverage, commands,
environment details and remaining limits.
