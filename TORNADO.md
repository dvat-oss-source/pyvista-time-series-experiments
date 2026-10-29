# Tornado time-series experiment

`tornado_animation.ipynb` plays all 128 frames of JangaFX's Looping Tornado
through the experimental `pyvista.TimeSeriesPlayer` API and a Trame volume view.

From this repository's root, using Python 3.13:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tornado.txt
python -m jupyter lab tornado_animation.ipynb
```

Select this environment's kernel. A working OpenGL context is required.
The dependency file records the notebook environment's installed versions and
installs PyVista from this checkout, including its IO and Jupyter extras.

## Data

Download [TornadoLoopingVDB.zip](https://www.mediafire.com/file/fynwmhozanxv821/TornadoLoopingVDB.zip/file)
from [JangaFX's free VDB catalog](https://www.jangafx.com/software/embergen/download/free-vdb-animations)
into `~/Downloads`, then run the notebook cells in order. The notebook verifies
the archive, moves it into PyVista's cache, and extracts all 128 VDB frames.
The source is CC0/public domain; the archive's license and readme are extracted
alongside it.

- Archive size: 1,797,795,595 bytes.
- SHA-256: `492c99321ae911d0ad6c6ff3df6eb53c83e443ba91b6aab7b97ba86e536ac57c`.
- Cache: `Path(pyvista.examples.USER_DATA_PATH) / 'tornado_looping'`.
- Original frames: `vdb/tornado_0000.vdb` through `vdb/tornado_0127.vdb`.
- Converted display frames: `display-256/*.pv`.

Allow additional disk space for extracted and converted frames. On macOS/Linux,
the notebook downloads micromamba and installs Python 3.13, OpenVDB 13.1, and
NumPy in an isolated reader environment inside the cache. Later runs reuse the
downloaded data, reader, and converted frames. These large data files are stored
locally rather than in Git.
