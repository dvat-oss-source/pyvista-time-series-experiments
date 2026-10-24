# Discrete time series playback

`pv.TimeSeriesPlayer` displays one saved dataset at a time through one actor.
`times` are source timestamps; `cycle_end` gives the last sample an explicit
dwell. `play(cycle_seconds)` maps that source cycle to wall-clock seconds.
The application calls `tick()` from its own timer. The player creates no timer,
widget, frame cache, or files.

## What this worktree contained

The untracked `exploration/time_series.py` already proposed a player layered on
the slider-based `TimeController`; its tests existed too. The tracked
`ReaderSource` already adapted `TimeReader`, while `heart_animation.ipynb` read
the ten independent `.pv` files directly for its GIF and Trame views. The
notebook's actual cache is `frames.json` plus ten files totaling **5,307,414 B**;
the original archive is **78,152,158 B**. The older
`exploration/heart_data.py` uses a different `sequence.json` cache and its size
numbers do not describe the notebook cache. A PVD reader returned distinct
buffers for two wavy samples in this environment; the adapter retains its deep
copy because other temporal readers may reuse output buffers.

One player now owns both the selected sample and its fractional timeline
position. A separate clock object would duplicate the selected index and add a
second synchronization boundary without supporting a demonstrated second use.
The injected clock makes scheduling deterministic in tests. The old slider
controller remains available for its existing examples.

## Contract

| Operation                                                                         | Behavior                                                                                                                                                                                                                                                                       |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TimeSeriesPlayer(pl, times, load_frame, *, cycle_end, scalars, clim, clock=...)` | Validates finite, strictly increasing times and a final dwell; loads index 0 only. The caller owns `pl` and the loader.                                                                                                                                                        |
| `seek_index(i)`                                                                   | Displays exact index; invalid indices fail before loading. A seek to the current index does nothing.                                                                                                                                                                           |
| `seek_time(t)`                                                                    | Clamps endpoints and chooses the nearest sample, choosing the earlier one at a midpoint.                                                                                                                                                                                       |
| `step(n)`                                                                         | Moves by sample count and wraps, including negative counts.                                                                                                                                                                                                                    |
| `play(seconds, loop=None)` / `pause()`                                            | Starts or resumes at the retained fractional position; an omitted mode preserves the previous setting (initially looping). Pause first displays the currently due sample. Nonlooping playback stops on the last sample at `cycle_end`, and the next play starts from sample 0. |
| `tick()`                                                                          | Called externally; displays only the sample due now and may skip samples if ticks are late. Clock input must be finite.                                                                                                                                                        |
| `close()`                                                                         | Removes its actor and player-owned references, is idempotent, and leaves the independently owned plotter open. A prior plotter close makes subsequent player operations raise.                                                                                                 |

Source and wall time are distinct. Heart phases are fractions 0.0–0.9 with
`cycle_end=1.0`; the 0.5-second display cycle is a presentation choice, not a
measured heartbeat. Seeking a different sample while playing resets the wall
clock anchor to that sample. A load or validation failure preserves the mapper,
selected index, and fractional position. A renderer failure after replacement
cannot be rolled back.

The player keeps one shallow dataset copy to isolate active scalar metadata.
Before a seek, the player, mapper, and possibly `Plotter.mesh` reference the
current frame. During loading the old display and new loader result coexist.
After mapper replacement and through rendering, the method's local variable
also retains the old frame. When the method returns, the player retains only
the new frame; it updates `Plotter.mesh` if that reference still names its old
frame. VTK's scalar pipeline, a renderer, a reader, or the caller may retain
other buffers. The guaranteed bound is **one player-owned displayed frame**, not
a process RSS or GPU-memory ceiling. `TimeReaderSource` deep-copies one selected
part to detach it before restoring reader time; the reader can still hold its
own output. Neither adapter writes converted files.

## Usage

The [heart notebook](../heart_animation.ipynb) contains executable GIF and
Trame paths. Its GIF loop calls `seek_index(i)` before `pl.write_frame()`; its
Trame slider calls `seek_index(i)`, and an application-owned async task calls
`tick()` while playing. Both load the existing `.pv` files with `pv.read`.

A temporal reader needs only the thin installed adapter:

```python
import pyvista as pv

source = pv.TimeReaderSource(pv.get_reader('results.pvd'), block=0, scalars='temperature')
pl = pv.Plotter()
player = pv.TimeSeriesPlayer(
    pl,
    source.times,
    source.frame,
    cycle_end=source.times[-1] + 1,
    scalars='temperature',
    clim=(0, 100),
)
pl.show(auto_close=False)
player.seek_index(2)
player.close()
pl.close()
```

`block` selects a spatial part, never a time step. The adapter preserves the
reader's files, restores its selected time after each request, and requires a
dedicated synchronous reader.

## Validation and measurements

All ten displayed heart meshes matched independently parsed original CONRAD
coordinates and triangle ordering, plus direct `.pv` reads and point/cell
labels. A real render test changed the screenshot after an exact seek. A
15-sample wavy PVD drove the player through the renderer; exact displayed
geometry and scalars matched independent reads of original `.vts` files.
Injected-clock tests cover irregular timestamps, final dwell, looping,
pause/resume, and a failed seek. Frame loads were counted rather than
preloaded.

Fresh macOS processes used a **non-editable PyVista wheel on Python 3.13.6**,
imported from `site-packages` with `python -I` outside this checkout and no
`PYTHONPATH`. Both modes used the same ten cached files, 400×320 renderer, GIF
writer, camera, and 20 fps as the notebook. The raw
[measurements](installed_measurements_3136.json) include each cycle's RSS and
displayed coordinate hashes. “Logical read bytes” sums compressed file sizes
for each `pv.read` call. It is not physical disk traffic: the OS cache was
uncontrolled and macOS has no per-process `psutil.io_counters()`.

| Mode / cycles        | Reads | Logical read B | Peak RSS MiB | Last-cycle RSS MiB | Cycle 1→last growth MiB | Post-close RSS MiB |
| -------------------- | ----: | -------------: | -----------: | -----------------: | ----------------------: | -----------------: |
| Notebook direct / 5  |    51 |     27,067,699 |       258.47 |             255.52 |                    0.95 |             258.44 |
| Player / 5           |    50 |     26,537,070 |       257.98 |             254.38 |                    0.53 |             257.95 |
| Notebook direct / 50 |   501 |    265,901,329 |       258.91 |             255.88 |                    1.16 |             258.88 |
| Player / 50          |   500 |    265,370,700 |       258.64 |             255.16 |                    0.98 |             258.61 |

The first-cycle displayed coordinate hashes and 342,801-byte GIFs matched
between modes. Both 50-cycle RSS traces leveled off in this run; the player
was not consistently lower than the notebook loop. Peak RSS includes Python
and native CPU allocations, GIF encoding, and allocator retention; it excludes
some GPU allocations. A finite 50-cycle run does not prove leak freedom, and
the 2 ms sampler may miss brief peaks. No cache or latency target is claimed.

Reproduce the measurement in separate processes from the repository root:

```sh
python -m exploration.measure_time_series direct /tmp/heart-direct.json --cycles 50
python -m exploration.measure_time_series player /tmp/heart-player.json --cycles 50
python -m pytest -q tests/core/test_time_series.py exploration/test_reader_source.py
```

The render commands need a working local OpenGL context. On this Mac they ran
with desktop OpenGL access outside the filesystem sandbox.
