# Time-series verification

Branch: `feat/time-series-container`. These are exploratory implementations outside
PyVista's public package API.

## Results

| Check                                 | Result                                                                      |
| ------------------------------------- | --------------------------------------------------------------------------- |
| This branch's scoped `make test-core` | 7 passed                                                                    |
| All three scoped runs                 | 48 passed (39 prototype cases and 3 shared reader cases run on each branch) |
| All three notebooks                   | All default cells passed; one saved static figure each                      |
| `make lint`                           | Passed on all three branches                                                |
| `make docstyle`                       | Passed: 845 files, plus expected-failure fixtures                           |
| Complete docstring execution          | 1713 passed, 3 skipped, 3 failed                                            |
| Docstring name resolution             | Passed                                                                      |
| Manual mouse interaction              | Unverified: Computer Use permission unavailable                             |

The complete docstring and documentation-style runs were performed once because
all three branches have identical package and documentation sources. Each branch
received its own scoped core/plotting and lint runs. No CI was triggered.

## Coverage

- General container: lazy reads, exact and indexed selection, input/frame
  ownership, nested MultiBlock arrays, changing topology, real file-series
  readers, and reader-time restoration after success and simulated failure.
- Fixed mesh: scalar/vector interpolation at irregular times, point/cell
  associations, static geometry and fields, integer/float data, independent
  ownership, invalid shapes/types/times, and extrapolation rejection.
- Controller: real mapper arrays and geometry, changing topology, stable actor,
  camera and color limits, nearest selection, endpoint and tie policies, failed
  selection preserving the scene, single-frame behavior, invalid limits, actual
  reader integration and changed/nonblank rendered pixels.

The rendering checks are smoke comparisons rather than comparisons against
committed image baselines. Desktop-window creation succeeded, but manual slider
interaction could not be performed because Computer Use permission was not
available. The temporary demo process was closed afterward.

## Remaining repository failures

The complete doctest run has three failures. Each reproduces on unchanged
`main` with the same environment:

- `PolyDataFilters.collision`: collision-cell ordering differs from the example.
- `PolyData.point_normals`: a printed normal differs in its final float digits.
- `Transform.decompose`: a near-zero matrix component has different rounding.

These failures are outside the experiments; no package implementation was changed
to suppress them. The overall `make doctest` target therefore remains red, and the
experiments should not be described as passing every repository gate.

An initial run used a tox directory outside the checkout, causing pytest to miss
`pyproject.toml` and its `NUMBER ELLIPSIS` comparison flags. The final counts above
come from a complete rerun with tox in its normal project directory and the
repository configuration loaded.

## Environment and reproduction

Python 3.13.6, NumPy 2.5.3, VTK 9.7.1, PyVista 0.50.dev0,
pyvista-validation 0.5.4, pytest 9.0.3, and Vale 3.22.0 were used. Development
dependencies were installed in `/private/tmp/pyvista-notebook-verify`; the global
Python environment was not modified.

From this worktree, with the dev environment active:

```sh
make lint
make docstyle
make test-core ARGS='exploration tests/core/test_reader.py::test_forbid_invalid_series_file'
make doctest
```

The existing reader case ensures pytest loads the repository's test configuration
when collecting the experiments outside `tests/`. Do not place the doctest tox
environment outside the repository unless pytest is explicitly given its config.

Open `exploration/test_time_series_container.ipynb` in a kernel with this checkout's
dependencies and run all cells. Saved execution outputs are included. The optional
desktop-interaction cell in the controller notebook is disabled by default.

`exploration/ruff.toml` applies the repository's example-style annotation/import
policy to exploratory scripts and its assertion policy to test scripts and
notebooks. All other inherited lint checks remain enabled.

Local detailed logs are available at `/private/tmp/pyvista-gate-container.log`,
`/private/tmp/pyvista-lint-container.log`, `/private/tmp/pyvista-docstyle.log`,
`/private/tmp/pyvista-doctest-corrected.log`,
`/private/tmp/pyvista-baseline-remaining.log`, and
`/private/tmp/pyvista-notebooks-final.log`.
