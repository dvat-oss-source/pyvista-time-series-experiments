"""Display discrete datasets on an externally driven timeline."""

from __future__ import annotations

import bisect
import math
import operator
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

import pyvista as pv

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from pyvista.core.dataset import DataSet
    from pyvista.core.utilities.reader import TimeReader
    from pyvista.plotting.actor import Actor
    from pyvista.plotting.plotter import Plotter


class TimeSeriesPlayer:
    """Display independently loaded samples through one actor.

    The loader returns one dataset by index and must keep its buffers valid and
    unchanged while that sample is displayed. Only the current sample is retained
    by the player. The caller owns the plotter and calls :meth:`tick` from its own
    event loop or timer. Use the plotter's thread for all player operations.

    Parameters
    ----------
    pl : pyvista.Plotter
        Plotter in which to create one actor. The player does not close it.
    times : sequence of float
        Nonempty, finite, strictly increasing source timestamps.
    load_frame : callable
        Function accepting an integer index and returning one PyVista dataset.
    cycle_end : float
        End of the playback cycle in source time, later than the last sample.
        The gap after the last sample is its display dwell.
    scalars : str
        Name of a one-component real numeric array on every frame.
    clim : sequence of float
        Fixed, increasing color limits for every frame.
    preference : str, default: 'point'
        Association of ``scalars``: ``'point'`` or ``'cell'``.
    clock : callable, optional
        Monotonic clock returning seconds. Defaults to :func:`time.monotonic`.

    Notes
    -----
    Seeking loads only the requested frame. The player makes a shallow copy to
    isolate active-scalar metadata, sharing the loader's geometry and array
    buffers. During replacement, the old and new frames can coexist. VTK, a
    reader, renderer, or caller may retain additional buffers independently.
    A failed load or validation leaves the displayed frame and playback position
    unchanged. Rendering failures after mapper replacement cannot be rolled back.

    .. versionadded:: 0.50

    """

    def __init__(
        self,
        pl: Plotter,
        times: Sequence[float],
        load_frame: Callable[[int], DataSet],
        *,
        cycle_end: float,
        scalars: str,
        clim: Sequence[float],
        preference: Literal['point', 'cell'] = 'point',
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Validate the sequence and display its first sample."""
        self.times = tuple(float(value) for value in times)
        if not self.times or not all(math.isfinite(value) for value in self.times):
            msg = 'Times must be nonempty and finite.'
            raise ValueError(msg)
        if any(right <= left for left, right in zip(self.times, self.times[1:], strict=False)):
            msg = 'Times must be strictly increasing.'
            raise ValueError(msg)
        self.cycle_end = float(cycle_end)
        if (
            not math.isfinite(self.cycle_end)
            or self.cycle_end <= self.times[-1]
            or not math.isfinite(self.cycle_end - self.times[0])
        ):
            msg = 'Cycle end must be finite and later than the final sample.'
            raise ValueError(msg)
        if preference not in ('point', 'cell'):
            msg = "Preference must be 'point' or 'cell'."
            raise ValueError(msg)
        self.clim = tuple(float(value) for value in clim)
        if (
            len(self.clim) != 2
            or not all(math.isfinite(value) for value in self.clim)
            or self.clim[0] >= self.clim[1]
        ):
            msg = 'Color limits must be two finite increasing values.'
            raise ValueError(msg)
        self._pl: Plotter | None = pl
        self._load_frame: Callable[[int], DataSet] | None = load_frame
        self._scalars = scalars
        self._preference = preference
        self._clock: Callable[[], float] | None = time.monotonic if clock is None else clock
        self._index = 0
        self._position = self.times[0]
        self._started_at: float | None = None
        self._cycle_seconds: float | None = None
        self._loop = True
        self._playing = False
        frame = self._load(0)
        self._frame: DataSet | None = frame
        self._actor: Actor | None = pl.add_mesh(
            frame,
            scalars=scalars,
            preference=preference,
            clim=self.clim,
            static=False,
            rgb=False,
            smooth_shading=False,
            silhouette=False,
            show_vertices=False,
            show_scalar_bar=False,
        )

    @property
    def actor(self) -> Actor:
        """Return the actor displaying the current sample."""
        self._active()
        return cast('Actor', self._actor)

    @property
    def index(self) -> int:
        """Return the index of the displayed sample."""
        self._active()
        return self._index

    @property
    def time(self) -> float:
        """Return the source timestamp of the displayed sample."""
        return self.times[self.index]

    @property
    def playing(self) -> bool:
        """Return whether playback is running."""
        self._active()
        return self._playing

    def _active(self) -> Plotter:
        """Return the open plotter or reject use after either owner closes."""
        pl = self._pl
        if pl is None:
            msg = 'The time series player is closed.'
            raise RuntimeError(msg)
        if getattr(pl, '_closed', False):
            self.close()
            msg = 'The player plotter is closed.'
            raise RuntimeError(msg)
        return pl

    def _load(self, index: int) -> DataSet:
        """Validate a replacement before touching the mapper."""
        loaded: object = cast('Callable[[int], DataSet]', self._load_frame)(index)
        if not isinstance(loaded, pv.DataSet) or isinstance(loaded, pv.PointSet):
            msg = 'The frame loader must return one plottable PyVista dataset.'
            raise TypeError(msg)
        data = loaded.point_data if self._preference == 'point' else loaded.cell_data
        if self._scalars not in data:
            msg = f'Frame {index} has no {self._preference} array {self._scalars!r}.'
            raise ValueError(msg)
        values = data[self._scalars]
        if loaded.n_points == 0 or values.size == 0:
            msg = f'Frame {index} is empty.'
            raise ValueError(msg)
        if values.ndim != 1 or values.dtype.kind not in 'iuf':
            msg = f'Frame {index} requires a one-component real numeric scalar array.'
            raise ValueError(msg)
        frame = loaded.copy(deep=False)
        frame.set_active_scalars(self._scalars, preference=self._preference)
        return frame

    def _now(self) -> float:
        """Read a finite value from the supplied monotonic clock."""
        now = float(cast('Callable[[], float]', self._clock)())
        if not math.isfinite(now):
            msg = 'The playback clock must return a finite time.'
            raise ValueError(msg)
        return now

    def _select(self, index: int) -> None:
        """Commit one validated sample and render it."""
        pl = self._active()
        if index == self._index:
            return
        frame = self._load(index)
        old_frame = self._frame
        actor = self.actor
        mapper = actor.mapper
        if mapper is None:
            msg = 'The playback actor has no mapper.'
            raise RuntimeError(msg)
        mapper.dataset = frame
        mapper.scalar_range = self.clim
        if pl.mesh is old_frame:
            pl.mesh = frame
        self._frame = frame
        self._index = index
        pl.render()

    def seek_index(self, index: int) -> None:
        """Display an exact sample, rejecting invalid or out-of-range indices.

        Parameters
        ----------
        index : int
            Index of the sample to display.

        """
        self._active()
        if isinstance(index, bool):
            msg = 'Frame index must be an integer, not a boolean.'
            raise TypeError(msg)
        index = operator.index(index)
        if not 0 <= index < len(self.times):
            msg = 'Frame index is outside the available samples.'
            raise IndexError(msg)
        if index != self._index:
            now = self._now() if self._playing else None
            self._select(index)
            self._position = self.times[index]
            if self._playing:
                self._started_at = now

    def seek_time(self, value: float) -> None:
        """Display the nearest sample, clamping endpoints and breaking ties earlier.

        Parameters
        ----------
        value : float
            Source timestamp to seek.

        """
        value = float(value)
        if not math.isfinite(value):
            msg = 'Selected time must be finite.'
            raise ValueError(msg)
        right = bisect.bisect_left(self.times, value)
        if right == 0:
            index = 0
        elif right == len(self.times) or value <= (
            self.times[right - 1] + (self.times[right] - self.times[right - 1]) / 2
        ):
            index = right - 1
        else:
            index = right
        self.seek_index(index)

    def step(self, count: int = 1) -> None:
        """Move by sample count, wrapping within the available samples.

        Parameters
        ----------
        count : int, default: 1
            Number of samples to move. Negative values step backward.

        """
        if isinstance(count, bool):
            msg = 'Step count must be an integer, not a boolean.'
            raise TypeError(msg)
        count = operator.index(count)
        self.seek_index((self.index + count) % len(self.times))

    def play(self, cycle_seconds: float, *, loop: bool | None = None) -> None:
        """Start or resume playback; the caller remains responsible for ticks.

        Parameters
        ----------
        cycle_seconds : float
            Wall-clock duration of one source cycle, in seconds.
        loop : bool, optional
            Restart from the first sample after the final dwell. If omitted,
            retain the previous mode; the initial mode is looping.

        """
        self._active()
        if self._playing:
            return
        cycle_seconds = float(cycle_seconds)
        if not math.isfinite(cycle_seconds) or cycle_seconds <= 0:
            msg = 'Cycle duration must be finite and positive.'
            raise ValueError(msg)
        now = self._now()
        if self._position == self.cycle_end:
            self.seek_index(0)
            self._position = self.times[0]
        self._cycle_seconds = cycle_seconds
        if loop is not None:
            self._loop = loop
        self._started_at = now
        self._playing = True

    def pause(self) -> None:
        """Display the sample due now, then preserve fractional cycle position."""
        self._active()
        if self._playing:
            self.tick()
            self._playing = False
            self._started_at = None

    def tick(self) -> None:
        """Advance to the sample due at the injected clock's current time."""
        self._active()
        if not self._playing:
            return
        now = self._now()
        started_at = cast('float', self._started_at)
        elapsed = max(0.0, now - started_at)
        if not math.isfinite(elapsed):
            msg = 'The playback clock interval must be finite.'
            raise ValueError(msg)
        span = self.cycle_end - self.times[0]
        offset = (self._position - self.times[0]) / span
        cycle_seconds = cast('float', self._cycle_seconds)
        if self._loop:
            fraction = (offset + (elapsed % cycle_seconds) / cycle_seconds) % 1
            position = self.times[0] + fraction * span
        else:
            position = (
                self.cycle_end
                if elapsed >= (1 - offset) * cycle_seconds
                else self.times[0] + (offset + elapsed / cycle_seconds) * span
            )
        index = bisect.bisect_right(self.times, position) - 1
        self._select(index)
        self._position = position
        self._started_at = max(now, started_at)
        if not self._loop and position >= self.cycle_end:
            self._playing = False
            self._started_at = None

    def close(self) -> None:
        """Remove this actor and release player references without closing the plotter."""
        pl = self._pl
        if pl is None:
            return
        actor = self._actor
        if actor is not None and not getattr(pl, '_closed', False):
            pl.remove_actor(actor, render=False)
            if getattr(pl, 'mapper', None) is actor.mapper:
                pl.mapper = None
        if pl.mesh is self._frame:
            pl.mesh = None
        self._playing = False
        self._started_at = None
        self._frame = None
        self._actor = None
        self._load_frame = None
        self._clock = None
        self._pl = None


class TimeReaderSource:
    """Adapt exact samples from a dedicated temporal reader.

    The source exposes ``times`` and ``frame(index)`` for
    :class:`TimeSeriesPlayer`. A selected MultiBlock part must be named
    explicitly. One deep copy detaches each returned frame before restoring
    the reader's previous active time. The caller owns the reader and must not
    change its time concurrently.

    .. versionadded:: 0.50

    Parameters
    ----------
    reader : pyvista.TimeReader
        Dedicated temporal reader with a ``read`` method.
    scalars : str
        Array required on the selected spatial part.
    block : int or str, optional
        Top-level spatial part of MultiBlock output.
    preference : str, default: 'point'
        Association of ``scalars``: ``'point'`` or ``'cell'``.

    """

    def __init__(
        self,
        reader: TimeReader,
        *,
        scalars: str,
        block: int | str | None = None,
        preference: Literal['point', 'cell'] = 'point',
    ) -> None:
        """Validate reader metadata without reading a result frame."""
        if preference not in ('point', 'cell'):
            msg = "Preference must be 'point' or 'cell'."
            raise ValueError(msg)
        self.times = tuple(float(value) for value in reader.time_values)
        if (
            not self.times
            or not all(math.isfinite(value) for value in self.times)
            or any(right <= left for left, right in zip(self.times, self.times[1:], strict=False))
        ):
            msg = 'Reader times must be finite, nonempty, and strictly increasing.'
            raise ValueError(msg)
        self.reader = reader
        self.scalars = scalars
        self.block = block
        self.preference = preference

    def frame(self, index: int) -> DataSet:
        """Read an exact sample without leaving the reader at another time.

        Parameters
        ----------
        index : int
            Index of the temporal sample to read.

        Returns
        -------
        pyvista.DataSet
            Independently owned selected spatial part.

        """
        if isinstance(index, bool):
            msg = 'Frame index must be an integer, not a boolean.'
            raise TypeError(msg)
        index = operator.index(index)
        if not 0 <= index < len(self.times):
            msg = 'Frame index is outside the available samples.'
            raise IndexError(msg)
        previous = self.reader.active_time_value
        try:
            self.reader.set_active_time_point(index)
            if self.reader.active_time_value != self.times[index]:
                msg = 'Reader did not select the requested simulation timestamp.'
                raise ValueError(msg)
            mesh: object = cast('Any', self.reader).read()
            if isinstance(mesh, pv.MultiBlock):
                if self.block is None:
                    msg = 'Select a spatial block explicitly for MultiBlock output.'
                    raise ValueError(msg)
                mesh = mesh[self.block]
            elif self.block is not None:
                msg = 'A block selection requires MultiBlock reader output.'
                raise ValueError(msg)
            if not isinstance(mesh, pv.DataSet):
                msg = 'The selected spatial part must be a dataset, not a null or composite block.'
                raise TypeError(msg)
            data = mesh.point_data if self.preference == 'point' else mesh.cell_data
            if self.scalars not in data:
                msg = f'Frame {index} has no {self.preference} array {self.scalars!r}.'
                raise ValueError(msg)
            return mesh.copy(deep=True)
        finally:
            self.reader.set_active_time_value(previous)
