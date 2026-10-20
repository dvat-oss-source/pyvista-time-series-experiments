# Three time-series versions: local verification

Verified on October 19, 2026. These are three experimental branches, rather than
three PyVista releases. The source fixes remain uncommitted in their restored
worktrees.

| Version                       | Base revision                              | Successful operation                                                                                        | Scoped tests |
| ----------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------- | ------------ |
| `feat/time-series-arrays`     | `f03e34fcadbb101f097d4e16b057a08b78c673fa` | Independently owned frames, exact selection, and scalar/vector interpolation on fixed geometry              | 28 passed    |
| `feat/time-series-container`  | `590242ad73b885418595724e600a78be5620904c` | Lazy snapshots, changing topology, nested ownership, and reader-time restoration                            | 7 passed     |
| `feat/time-series-controller` | `4cbe78ae5a73ad9b6e134195ddc3b7ac45b3a421` | Correct nearest selection and slider display while preserving actor, camera, color limits, and reader state | 26 passed    |

The 61 scoped cases include 52 prototype cases and three shared reader cases run
on each branch. Each worktree is `/private/tmp/pyvista-time-series-<version>`.

## Original environment failure

The system Python could not import PyVista:

```sh
python3 -c 'import pyvista as pv'
```

It raised `ImportError: cannot import name '_is_real' from 'pyvista_validation.check'`.
The installed `pyvista-validation` was 0.2.2; these revisions require exactly 0.5.4.
This was a dependency mismatch. Installing the repository dependencies in
`/private/tmp/pyvista-notebook-verify` resolved it without modifying system packages.
The complete traceback is in
[original-import-failure.log](/private/tmp/pyvista-three-version-verification/original-import-failure.log).

The verified environment uses Python 3.13.5, PyVista 0.50.dev0, VTK 9.7.1,
NumPy 2.5.3, Matplotlib 3.11.1, pytest 9.0.3, and pyvista-validation 0.5.4.
Rendering uses `vtkCocoaRenderWindow` on an Apple M3 Pro. A sandboxed graphics
probe initially failed; the same probe succeeded with macOS graphics access.
Full dependency versions and the backend report are retained with the evidence.

## Reproduced source failures and fixes

- **Fixed arrays:** accepted `float16` fields failed during frame construction with
  `TypeError: Could not find a suitable VTK type for float16`. A two-frame,
  four-point plane reproduces this with `np.ones((2, 4), dtype=np.float16)`.
  The constructor now copies half-precision data to float32, including big-endian
  float16. Tests check interpolated values, exact values, both associations, and
  preservation of the existing supported dtypes. See
  [fixed_mesh_series.py](/private/tmp/pyvista-time-series-arrays/exploration/fixed_mesh_series.py).
- **Nearest selection:** timestamps `[-1.7e308, -1.6e308]` and a request of
  `1.7e308` selected the first frame because both distances overflowed. Selection
  now handles endpoints and exact samples before comparing neighboring distances.
  Exact timestamps, adjacent floating-point samples, midpoint ties, and endpoint
  clamping have regressions. See
  [time_controller.py](/private/tmp/pyvista-time-series-controller/exploration/time_controller.py).
- **Slider display:** `select_time(0.9)` changed the dataset while leaving the
  slider at zero. Successful requests now update its displayed value, including
  requests within the same frame interval. Failed loads retain the previous
  scene and slider state. The fallback retains support for widgets without VTK's
  newer property API.

Each original failure was captured before its fix. Additional byte-order and
adjacent-timestamp and rounded-midpoint regressions exposed gaps in the first fixes and were also
captured before correction. The general container had no reproduced defect and
needed no source change. An independent review found no remaining correctness
findings in the final diff.

## Verification and rendered output

- All three original notebooks execute successfully in fresh kernels.
- The arrays and container figures remain pixel-identical to their original
  saved output. The corrected controller figure differs only within the slider
  rectangle `(287, 17)` to `(623, 57)`; the dataset and color bar are unchanged.
  Only that notebook render cell's saved output and execution metadata were
  refreshed. See the
  [corrected controller figure](/private/tmp/pyvista-three-version-verification/controller/cell-8-0.png).
- Three existing image regression tests passed: the camera widget and two
  scalar-range tests. The camera test used its existing macOS alternative
  baseline after the first comparison differed. No plotting cache was reset.
- `make doctest` passed: 1,716 examples, 3 skips, plus the static name check.
- `make lint` passed on all three branches; `make docstyle` passed.

The branches and primary checkout have identical package and documentation trees:
`b07fc1f4adbdd30c7f8fb2aee023ff6dd208f0ca` and
`729d7837b3fd4e4e3e0a3696a8f33d8aadb780c5`, respectively. Shared docstring and
documentation-style checks therefore ran once against those sources. The changes
are confined to the experiments, their tests, and verification records.

The desktop demo opened. Mouse dragging remains unverified because macOS screen
capture failed before a drag could be checked. The temporary demo was closed.
This run covers the recorded macOS/Python/VTK environment; it is not a platform or
dependency-version matrix.

## Reproduce and inspect

From each restored worktree, use the isolated environment:

```sh
source /private/tmp/pyvista-notebook-verify/bin/activate
export UV_PROJECT_ENVIRONMENT=/private/tmp/pyvista-notebook-verify
export UV_NO_SYNC=true
export PYVISTA_OFF_SCREEN=true
make test-core ARGS='exploration tests/core/test_reader.py::test_forbid_invalid_series_file'
```

For the controller, substitute `make test-plotting` with the same arguments.
The retained [verification runner](/private/tmp/verify_three_versions.py) has
`inventory`, `notebooks`, `gates`, and `final` stages and records the exact commands.

Raw evidence lives in
[/private/tmp/pyvista-three-version-verification](/private/tmp/pyvista-three-version-verification):
complete failing and passing logs, environment snapshots, patch files, executed
notebooks, original/new/difference images, and machine-readable results. These
temporary files should be copied elsewhere for long-term retention.
