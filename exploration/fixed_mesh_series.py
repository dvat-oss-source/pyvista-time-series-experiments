"""Explore temporal arrays without changing PyVista's public namespace."""

import numpy as np
import pyvista as pv


class FixedMeshSeries:
    """Store one mesh and time-major point or cell arrays.

    This experimental container owns copies of all inputs. Frames are independent
    deep copies. Scalar arrays have shape ``(time, tuples)`` and vector arrays
    have shape ``(time, tuples, 3)``. Times are finite and strictly increasing.
    """

    def __init__(self, mesh, *, times, point_data=None, cell_data=None):
        """Validate and copy a fixed mesh and its temporal arrays."""
        if not isinstance(mesh, pv.DataSet):
            msg = 'mesh must be a PyVista DataSet.'
            raise TypeError(msg)
        times = np.array(times, dtype=float, copy=True)
        if (
            times.ndim != 1
            or times.size == 0
            or not np.all(np.isfinite(times))
            or not np.all(np.diff(times) > 0)
        ):
            msg = 'times must be a nonempty, finite, strictly increasing 1D sequence.'
            raise ValueError(msg)
        self._times = times
        self._mesh = mesh.copy(deep=True)
        self._data = {}
        for association, data, count in (
            ('point', point_data, mesh.n_points),
            ('cell', cell_data, mesh.n_cells),
        ):
            arrays = {}
            for name, values in ({} if data is None else data).items():
                values = np.asarray(values)
                if (
                    values.ndim not in (2, 3)
                    or values.shape[:2] != (len(times), count)
                    or (values.ndim == 3 and values.shape[2] != 3)
                    or values.dtype.kind not in 'iuf'
                ):
                    msg = (
                        f'{association} array {name!r} must be real numeric with shape '
                        f'({len(times)}, {count}) or ({len(times)}, {count}, 3).'
                    )
                    raise ValueError(msg)
                arrays[name] = values.copy()
            self._data[association] = arrays

    @property
    def times(self):
        """Return a copy of the available times."""
        return self._times.copy()

    def at_time(self, time, *, method='exact'):
        """Return a frame at an exact time or interpolate real numeric values.

        ``method='linear'`` interpolates components independently, promotes integer
        fields to floating point between samples, and never extrapolates. Static
        arrays and geometry are copied unchanged. Selection has no side effects.
        """
        if method not in ('exact', 'linear'):
            msg = "method must be 'exact' or 'linear'."
            raise ValueError(msg)
        if not np.isfinite(time) or time < self._times[0] or time > self._times[-1]:
            msg = 'time must be finite and within the sampled interval.'
            raise ValueError(msg)
        right = int(np.searchsorted(self._times, time))
        exact = self._times[right] == time
        if not exact and method == 'exact':
            msg = 'time must match a sample when method is exact.'
            raise ValueError(msg)
        frame = self._mesh.copy(deep=True)
        for association, arrays in self._data.items():
            target = frame.point_data if association == 'point' else frame.cell_data
            for name, values in arrays.items():
                if exact:
                    sample = values[right].copy()
                else:
                    left = right - 1
                    weight = (time - self._times[left]) / (self._times[right] - self._times[left])
                    sample = (1 - weight) * values[left] + weight * values[right]
                target[name] = sample
        return frame
