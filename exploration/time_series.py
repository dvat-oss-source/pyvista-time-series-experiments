"""Explore time-indexed PyVista snapshots without exporting a package API."""

from collections.abc import Callable
from collections.abc import Sequence
import math
from typing import Protocol


class Snapshot(Protocol):
    """Describe the copying operation supplied by PyVista data objects."""

    def copy(self, *, deep: bool = True) -> 'Snapshot':
        """Return a copy of the data object."""
        ...


class TimeSeries:
    """Hold strictly ordered times and independently owned, lazily loaded snapshots."""

    def __init__(self, times: Sequence[float], load_frame: Callable[[int], Snapshot]):
        """Validate timestamps without loading any frames."""
        self._times = tuple(float(value) for value in times)
        if not self._times or any(not math.isfinite(value) for value in self._times):
            msg = 'Times must be nonempty and finite.'
            raise ValueError(msg)
        if any(left >= right for left, right in zip(self._times, self._times[1:])):
            msg = 'Times must be strictly increasing.'
            raise ValueError(msg)
        self._load_frame = load_frame

    @classmethod
    def from_datasets(cls, times: Sequence[float], datasets: Sequence[Snapshot]) -> 'TimeSeries':
        """Own deep copies of in-memory frames, allowing different topology per frame."""
        if len(times) != len(datasets):
            msg = 'Times and datasets must have equal lengths.'
            raise ValueError(msg)
        series = cls(times, lambda index: frames[index])
        frames = tuple(frame.copy(deep=True) for frame in datasets)
        return series

    @classmethod
    def from_reader(cls, reader) -> 'TimeSeries':
        """Adapt a dedicated TimeReader while restoring its selected time after reads."""
        times = tuple(reader.time_values)

        def load_frame(index: int) -> Snapshot:
            """Detach output before restoring the mutable reader selection."""
            previous = reader.active_time_value
            try:
                reader.set_active_time_value(times[index])
                return reader.read().copy(deep=True)
            finally:
                reader.set_active_time_value(previous)

        return cls(times, load_frame)

    @property
    def times(self) -> tuple[float, ...]:
        """Return immutable timestamp metadata."""
        return self._times

    def __len__(self) -> int:
        """Return the number of snapshots."""
        return len(self._times)

    def __getitem__(self, index: int) -> Snapshot:
        """Load an integer-indexed snapshot and return an independent deep copy."""
        if isinstance(index, bool) or not isinstance(index, int):
            msg = 'Frame indices must be integers.'
            raise TypeError(msg)
        normalized = index if index >= 0 else len(self) + index
        if normalized < 0 or normalized >= len(self):
            msg = 'Frame index out of range.'
            raise IndexError(msg)
        return self._load_frame(normalized).copy(deep=True)

    def at_time(self, time: float) -> Snapshot:
        """Return the exact timestamp match; interpolation requires a separate policy."""
        try:
            index = self._times.index(time)
        except ValueError:
            msg = f'No snapshot at time {time!r}.'
            raise KeyError(msg) from None
        return self[index]
