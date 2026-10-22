# Stored-result playback prototype

`ReaderSource` connects a PyVista temporal reader to the existing `TimeController`.
It reads one requested timestep, validates the selected point field, and deep-copies
the selected dataset once before restoring the reader's time. It writes no converted
results and keeps no frame cache. The controller implementation is restored unchanged
from commit `4cbe78ae5`.

The beating heart remains a future application. The working example uses cached
PyVista wavy-surface results: changing coordinates, fixed connectivity, 15 saved frames.
It is not a heart model or a physics engine.

From the repository root, run:

```sh
.venv/bin/python exploration/play_reader.py \
  "$HOME/Library/Caches/pyvista_3/PVD/wavy.zip.unzip/wavy.pvd" \
  --block 0 --scalars z --clim -1 1
```

Move the slider to select an existing simulation sample. Playback uses the current
controller's nearest-sample policy; the adapter performs no interpolation. MultiBlock
readers require an explicit spatial block. Multi-part scenes and cell-field playback
are outside this small adapter's scope. Returned dataset metadata is preserved;
authoritative simulation timestamps are available in `source.times`.

Application usage:

```python
import pyvista as pv
from reader_source import ReaderSource
from time_controller import TimeController

source = ReaderSource(pv.get_reader('results.pvd'), block=0, scalars='temperature')
pl = pv.Plotter()
controller = TimeController(pl, source.times, source.frame, scalars='temperature', clim=(0, 100))
pl.show()
```

These are project prototypes, not public PyVista temporal APIs. Run this snippet from
`exploration`, or put that directory on the Python import path. Use a dedicated reader;
reader access is synchronous and not thread-safe.

Validation:

```sh
.venv/bin/python -m pytest -q exploration/test_reader_source.py
.venv/bin/python /Users/user/temporal-investigation/validate_adapter.py --render
```

Seven focused tests cover exact direct-file comparison, changing topology, irregular
timestamps, backward/random seeking, independent retained frames, explicit block
selection, absent fields, and unchanged-controller compatibility. The cached example
check compares first/middle/last and eight seeded random requests with original PVD
frame files, then exercises real desktop rendering. Results are recorded in
`/Users/user/temporal-investigation/adapter-validation.json`; its preview is
`adapter-preview.png` in that directory. This render check is not an image-regression
baseline or proof of future heart-data performance.
