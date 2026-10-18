"""Check snapshot ownership, lazy access and reader restoration without graphics."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from time_series import TimeSeries


@dataclass
class Frame:
    """Stand in for the deep-copy contract of a PyVista data object."""

    values: list[int]

    def copy(self, *, deep: bool = True) -> Frame:
        """Detach the mutable payload."""
        return Frame(self.values.copy() if deep else self.values)


class Reader:
    """Model a temporal reader that reuses an output buffer."""

    time_values = (0.0, 2.0)
    active_time_value = 0.0

    def __init__(self):
        """Initialize mutable state and an optional read failure."""
        self.output = Frame([])
        self.reads = 0
        self.fail = False

    def set_active_time_value(self, value):
        """Select a frame while simulating reused output storage."""
        self.active_time_value = value
        self.output.values[:] = [int(value)]

    def read(self):
        """Count reads and optionally simulate a file failure."""
        self.reads += 1
        if self.fail:
            msg = 'Cannot read frame.'
            raise OSError(msg)
        return self.output


def test_time_series_smoke():
    """Exercise temporal semantics and meaningful invalid inputs."""
    original = Frame([1])
    series = TimeSeries.from_datasets([0, 2], [original, Frame([2, 3])])
    original.values[0] = 99
    assert series[0].values == [1]
    result = series[0]
    result.values[0] = 88
    assert series[0].values == [1]
    assert series[-1].values == [2, 3]
    for times in ([], [0, 0], [2, 1], [float('nan')], [float('inf')]):
        with pytest.raises(ValueError, match='Times'):
            TimeSeries(times, lambda _index: original)
    with pytest.raises(ValueError, match='equal lengths'):
        TimeSeries.from_datasets([0], [])
    with pytest.raises(KeyError):
        series.at_time(1)
    for index in (2, -3):
        with pytest.raises(IndexError):
            series[index]
    for index in (True, 1.0, slice(None)):
        with pytest.raises(TypeError):
            series[index]
    reader = Reader()
    lazy = TimeSeries.from_reader(reader)
    assert reader.reads == 0
    assert lazy.at_time(2).values == [2]
    assert reader.active_time_value == 0
    assert reader.reads == 1
    reader.fail = True
    with pytest.raises(OSError, match='Cannot read frame'):
        lazy[1]
    assert reader.active_time_value == 0
    print('TimeSeries smoke checks passed.')


if __name__ == '__main__':
    test_time_series_smoke()
