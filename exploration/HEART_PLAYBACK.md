# Whole-Heart Playback

The notebook uses the installed library and no `exploration` imports, `sys.path`
changes, or private APIs. One public API is new: `examples.download_heart_motion()`
returns ten verified `.pv` paths. Reading, mapper replacement, `Plotter`,
`open_gif`, and `write_frame` use existing PyVista APIs. The earlier exploration
helpers remain as legacy artifacts and are not notebook dependencies.

## Run the Notebook

The branch is `v0.0.1`; its package version is **0.50.dev0**. Use your existing
tutorial Python (3.10 or newer). Install the library from the clone so its normal
package dependencies and heart downloader are available:

```sh
python3 -m pip install -e '.[io]' jupyterlab psutil
python3 -m jupyter lab heart_animation.ipynb
```

Select the same Python interpreter in the notebook editor and Run All. Restart an
already running kernel after installation. No dedicated environment, custom kernel,
or wheel under `dist/` is required. OpenGL is required. First use downloads the
original archive; later runs reuse the verified cache. Set `PYVISTA_USERDATA_PATH`
before launching Jupyter to choose the cache directory. The notebook writes
`heart_preview.gif` in its working directory and displays it inline.

## Source and Interpretation

Unberath, M.; Maier, A.; Fleischmann, D.; Hornegger, J.; Fahrig, R.
_Open-source 4D Statistical Shape Model of the Heart for X-ray Projection Imaging._
ISBI 2015, 739–742. [Provider and original archive](https://www5.cs.fau.de/conrad/data/heart-model/index.html).

CONRAD v1.1 supplies stored phase consensuses derived from registration-based
segmentation of 20 ten-phase CT angiography cases. This supports synchronized
sampled playback without reconstruction or simulation. It was the smallest
credible directly retrievable whole-cycle source among those checked previously;
no additional dataset was added.

The archive is **78,152,158 bytes**, with 27 members totaling 187,859,952 bytes
uncompressed. Size and ZIP directory were inspected before the original retrieval.
SHA-256: `4c68caa46eb96ee83f604011b2ace79c2e30297220f7e0cb2ae00b0bd7b03b87`.
First use retrieves it and converts one phase at a time; cached calls verify hashes.

**License gap:** neither the archive nor provider page supplies an explicit
standalone dataset license. The GPL code license for CONRAD is not asserted to
license the separate data. This disclosed gap was accepted for the local example;
no redistribution permission is claimed. Heart data are not bundled in the wheel.

The original `cnfg.ccc` fixes component order and offsets. The last column of each
`phase_*.ccm` stores consensus coordinates. These are extracted as float64, with
component-local triangle indices shifted by the documented offsets. No coordinate
transform, smoothing, decimation, repair, deformation, or interpolation is applied.

Interpretation was checked against the paper and CONRAD commit
`9018413e8a8a5ed37cd29d9cbeeabeb75103cfd2`:
[PcaIO format](https://github.com/akmaier/CONRAD/blob/9018413e8a8a5ed37cd29d9cbeeabeb75103cfd2/src/edu/stanford/rsl/conrad/io/PcaIO.java),
[shape evaluation](https://github.com/akmaier/CONRAD/blob/9018413e8a8a5ed37cd29d9cbeeabeb75103cfd2/src/edu/stanford/rsl/conrad/geometry/shapes/activeshapemodels/ActiveShapeModel.java),
[scale restoration](https://github.com/akmaier/CONRAD/blob/9018413e8a8a5ed37cd29d9cbeeabeb75103cfd2/src/edu/stanford/rsl/conrad/geometry/shapes/activeshapemodels/PCA.java),
[phase mapping](https://github.com/akmaier/CONRAD/blob/9018413e8a8a5ed37cd29d9cbeeabeb75103cfd2/src/edu/stanford/rsl/conrad/geometry/shapes/mesh/Mesh4D.java),
and [two-level phantom](https://github.com/akmaier/CONRAD/blob/9018413e8a8a5ed37cd29d9cbeeabeb75103cfd2/src/edu/stanford/rsl/conrad/phantom/asmheart/CONRADCardiacModel4D.java).

Stored phase consensuses differ from the full two-level, spline-evaluated mean
phantom. Its second-level consensus has nonzero phase weights: maximum absolute
weight **0.0347484438480903**, norm **0.160073680030441**. This notebook preserves
the stored meshes without applying those weights or fitting splines.

## Coverage, Timing, and Units

| Component       | Vertices | Triangles |
| --------------- | -------: | --------: |
| LV myocardium   |    4,660 |     9,316 |
| Right atrium    |    1,480 |     2,956 |
| Left atrium     |    1,299 |     2,594 |
| Right ventricle |    3,814 |     7,624 |
| Aorta           |    1,139 |     2,274 |
| Left ventricle  |    4,038 |     8,072 |
| Total per phase |   16,430 |    32,836 |

All six components occur at every phase and move across the cycle. Missing are
pulmonary artery, detailed valves, coronaries, pericardium, and separate RV/atrial
myocardial walls. These statistical shapes are not a patient recording or complete
tissue model. Registration supports anatomical correspondence; equal topology alone
does not establish material-point identity. Playback replaces datasets rather than
interpolating or mutating point buffers.

Source phases are **0.0–0.9 of a cycle**; measured R–R timestamps are unavailable.
The chosen **0.5 s** display period matches the provider's 120 bpm projection demo,
not acquisition timing. Ten frames of 50 ms encode 500 ms without a duplicate
endpoint; browser timing can vary. Millimeters are **inferred** from documented CT
spacing and model scale restoration, not a `.ccm` unit tag. Coordinates are unchanged.

## Original wheel validation (Python 3.13.6)

The results below describe the original wheel run. Clone-install verification
is recorded separately in `heart_validation/clone-install.json`.

| Check                                             | Result                                                                                          |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Independent NumPy reading of every original phase | Exact float64 coordinates and original triangle ordering; maximum error 0                       |
| Displayed mapper coordinates                      | Every phase's hash matches independently parsed originals in both notebook runs                 |
| Anatomy and phase metadata                        | Original offsets, point/cell labels, six moving components, and phases match                    |
| Topology                                          | Fixed across ten phases; four cavities closed; original myocardium edge defect retained         |
| Closed-surface volumes                            | Independent oriented tetrahedron sums agree with PyVista volume                                 |
| Installed-wheel execution                         | Two code cells pass in isolated 3.13.6 kernels outside the checkout; site-packages path checked |
| Retrieval/cache                                   | Empty-cache first use and fresh-kernel offline cached playback pass                             |
| Inline animation                                  | Both runs contain ten 400 × 320 frames, each 50 ms; no cell errors                              |

LV EDV is **149.804 mL** at phase 0.0, ESV
**75.994 mL** at phase 0.4, assuming mm units; EF is
**49.271%**. The provider's separate mean-phantom example says
52%, a **2.729 percentage-point discrepancy**.
The model/spline distinction prevents treating this as the same reference instance;
the discrepancy is not claimed resolved. The provider's remaining surface figure
was checked qualitatively. Its old projection figure is unavailable, and no
reference-animation agreement is claimed. Myocardial volume is omitted because
of the source defect. These checks establish playback fidelity and execution,
not physiological validity, cardiac mechanics, or clinical measurement accuracy.

## Memory and Checks

Five serial cycles reuse one actor and mapper. Geometry is read sequentially;
the GIF encoder can retain ten small raster images. No fixed ceiling was imposed.
A 10 ms sampler and fresh-kernel OS high-water probe measured CPU RSS, not GPU memory.

| Run            |    OS peak | Cycle 1–5 growth | Cycle 3–5 growth |
| -------------- | ---------: | ---------------: | ---------------: |
| Empty-cache    | 285.66 MiB |         1.03 MiB |         0.16 MiB |
| Cached/offline | 277.70 MiB |         0.81 MiB |         0.09 MiB |

These finite-run changes do not prove zero leakage indefinitely. Exact samples,
environment versions, and source measurements are in `heart_validation/*.json`.
Lint, documentation style, static typing, 44 scoped tests, and seven installed-wheel
tests pass. Full doctests: **1,714 passed, three skipped, three failed**; all three
failures reproduce on unchanged `main`. Static docstring names: **3,903 passed**.

## Repeat the Independent Audit

Use a new work directory. The runner launches isolated Python (`-I`) without
`PYTHONPATH`, from empty execution directories; it blocks network proxies for the
cached test. The comparator imports neither the converter parser nor exploration helpers.

```sh
.venv-heart/bin/python -I tests/examples/validate_heart_notebook.py first-use --notebook heart_animation.ipynb --work-dir /tmp/heart-recheck
.venv-heart/bin/python -I tests/examples/validate_heart_notebook.py cached-offline --notebook heart_animation.ipynb --work-dir /tmp/heart-recheck
PYVISTA_USERDATA_PATH=/tmp/heart-recheck/fresh-cache .venv-heart/bin/python -I tests/examples/compare_heart_source.py --work-dir /tmp/heart-recheck
```
