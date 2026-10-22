"""Read independently owned result frames for the playback prototype."""

from __future__ import annotations

import math
import operator

import pyvista as pv


class ReaderSource:
    """Adapt a dedicated TimeReader without caching or converting its files.

    ``block`` explicitly selects one top-level spatial part of MultiBlock
    output. ``scalars`` names the point array consumed by TimeController.
    Returned frames own their buffers and may be modified independently.
    Dataset metadata is preserved; simulation timestamps are in ``times``.
    Reader access is synchronous and must not be shared between threads.
    """

    def __init__(self, reader, *, scalars, block=None):
        """Validate source metadata without reading result frames."""
        self.times = tuple(float(value) for value in reader.time_values)
        if (
            not self.times
            or not all(math.isfinite(value) for value in self.times)
            or any(a >= b for a, b in zip(self.times, self.times[1:], strict=False))
        ):
            msg = 'Reader times must be finite, nonempty, and strictly increasing.'
            raise ValueError(msg)
        self.reader = reader
        self.scalars = scalars
        self.block = block

    def frame(self, index):
        """Read an exact indexed sample and detach it before restoring time."""
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
            mesh = self.reader.read()
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
            if self.scalars not in mesh.point_data:
                msg = f'Frame {index} has no point array {self.scalars!r}.'
                raise ValueError(msg)
            return mesh.copy(deep=True)
        finally:
            self.reader.set_active_time_value(previous)
