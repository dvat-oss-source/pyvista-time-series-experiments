"""
.. _time_series_example:

Display a Time Series
~~~~~~~~~~~~~~~~~~~~~

Display discrete samples through one actor.

Use :class:`~pyvista.TimeSeriesPlayer` with datasets that contain the same named,
one-component real numeric point or cell array. Samples may have different
geometry; they are not interpolated.

For this example, create a few frame files in a temporary directory. Native
``.pv`` files require the ``io`` extra: ``pip install pyvista[io]``.
"""

from pathlib import Path
import tempfile

import numpy as np
import pyvista as pv

# %%
# Create ordered frames and choose fixed color limits. A fixed range makes a
# given color represent the same value at every time, which helps comparisons.

with tempfile.TemporaryDirectory() as directory:
    files = []
    for index, phase in enumerate((0.0, 0.5, 1.0)):
        mesh = pv.Plane(i_resolution=30, j_resolution=30)
        mesh['height'] = np.sin(2 * np.pi * (mesh.points[:, 0] + phase))
        path = Path(directory) / f'frame-{index}.pv'
        mesh.save(path)
        files.append(path)

    pl = pv.Plotter()
    player = pv.TimeSeriesPlayer.from_files(pl, files, scalars='height', clim=(-1, 1))
    player.seek_index(1)
    pl.view_xy()
    pl.show(auto_close=False)
    player.close()
    pl.close()

# %%
# The files are read in the supplied order. ``from_files`` assigns equally
# spaced sample times starting at zero. ``seek_index`` selects an exact sample;
# ``seek_time`` selects the nearest sample. The player reads the selected file
# when needed, while the reader and renderer may retain additional data.
#
# Notebook Playback
# ^^^^^^^^^^^^^^^^^
# In a notebook with the ``jupyter`` and ``io`` extras installed, create the
# files and display the controls in a cell. Press Play to start playback.
#
# .. code-block:: python
#
#    from pathlib import Path
#    import tempfile
#    import numpy as np
#    import pyvista as pv
#
#    pv.set_jupyter_backend('trame')
#    directory = tempfile.TemporaryDirectory()
#    files = []
#    for index, phase in enumerate((0.0, 0.5, 1.0)):
#        mesh = pv.Plane(i_resolution=30, j_resolution=30)
#        mesh['height'] = np.sin(2 * np.pi * (mesh.points[:, 0] + phase))
#        path = Path(directory.name) / f'frame-{index}.pv'
#        mesh.save(path)
#        files.append(path)
#    pl = pv.Plotter()
#    player = pv.TimeSeriesPlayer.from_files(pl, files, scalars='height', clim=(-1, 1))
#    await player.show(cycle_seconds=5)
#
# Keep the files available for as long as the player needs them. Before
# replacing the view or rerunning its cell, stop the notebook view and release
# the player. The plotter is owned by the caller.
#
# .. code-block:: python
#
#    await player.aclose()
#    pl.close()
#    directory.cleanup()
#
# Other Event Loops
# ^^^^^^^^^^^^^^^^^
# In an application with a running ``asyncio`` event loop on the plotter's thread,
# ``player.start(cycle_seconds=5)`` starts a playback task. Await the returned
# task to observe errors; use ``player.pause()`` to stop playback. Alternatively,
# an application with its own timer can call ``player.play(cycle_seconds=5)``
# once and call ``player.tick()`` on each timer event on the plotter's thread.
# Call ``player.close()`` when finished and close the plotter separately.
#
# Physical Timestamps
# ^^^^^^^^^^^^^^^^^^^
# For irregular physical times, use the constructor with a loader accepting a
# sample index. Set ``cycle_end`` later than the last timestamp: the difference
# determines how long the last sample remains visible in source time. For
# example, a temporal reader can supply the samples:
#
# .. code-block:: python
#
#    reader = pv.get_reader('simulation.pvd')
#    source = pv.TimeReaderSource(reader, scalars='temperature', block=0)
#    pl = pv.Plotter()
#    player = pv.TimeSeriesPlayer(
#        pl, source.times, source.frame,
#        cycle_end=source.times[-1] + 0.1,
#        scalars='temperature', clim=(0, 100),
#    )
#
# Choose a ``block`` when the reader returns a MultiBlock dataset. The source
# copies each selected dataset and restores the reader's previous active time;
# use a dedicated reader and do not change its time concurrently.
#
# .. tags:: plot
