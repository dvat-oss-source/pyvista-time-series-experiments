"""Explore a slider controller without adding public PyVista API."""

from __future__ import annotations

import math
import operator

import pyvista as pv


class TimeController:
    """Connect finite, increasing timestamps and a frame loader to one actor.

    The loader receives an integer frame index and returns a PyVista dataset.
    Supply real numeric scalar arrays and a fixed color range explicitly.
    Loaded buffers must remain valid and unchanged while displayed. The
    controller shallow-copies dataset metadata and shares the loaded buffers.
    Access must be synchronous on the plotter's thread.
    """

    def __init__(self, pl, times, load_frame, *, scalars, clim, preference='point'):
        """Create the actor and a time slider using public plotting methods."""
        if preference not in ('point', 'cell'):
            msg = "Preference must be 'point' or 'cell'."
            raise ValueError(msg)
        self.preference = preference
        self.times = tuple(float(value) for value in times)
        if not self.times or not all(math.isfinite(value) for value in self.times):
            msg = 'Times must be nonempty and finite.'
            raise ValueError(msg)
        if any(right <= left for left, right in zip(self.times, self.times[1:], strict=False)):
            msg = 'Times must be strictly increasing.'
            raise ValueError(msg)
        self.clim = tuple(float(value) for value in clim)
        if (
            len(self.clim) != 2
            or not all(math.isfinite(value) for value in self.clim)
            or self.clim[0] >= self.clim[1]
        ):
            msg = 'Color limits must be two finite increasing values.'
            raise ValueError(msg)
        self.pl = pl
        self.load_frame = load_frame
        self.scalars = scalars
        self.index = 0
        mesh = self._load(0)
        self._frame = mesh
        self.actor = pl.add_mesh(
            mesh,
            scalars=scalars,
            preference=preference,
            clim=self.clim,
            static=False,
            rgb=False,
            smooth_shading=False,
            silhouette=False,
            show_vertices=False,
        )
        self.slider = None
        if len(self.times) > 1:
            self.slider = pl.add_slider_widget(
                self.select_time,
                [self.times[0], self.times[-1]],
                value=self.times[0],
                title='Time (nearest sample)',
                interaction_event='end',
            )

    @classmethod
    def from_reader(cls, pl, reader, *, scalars, clim, block=None, preference='point'):
        """Adapt one explicitly selected spatial part using ReaderSource ownership."""
        if __package__:
            from .reader_source import ReaderSource
        else:
            from reader_source import ReaderSource

        source = ReaderSource(reader, scalars=scalars, block=block, preference=preference)
        return cls(
            pl, source.times, source.frame, scalars=scalars, clim=clim, preference=preference
        )

    def _load(self, index):
        """Validate the frame before changing what the plotter displays."""
        loaded = self.load_frame(index)
        if not isinstance(loaded, pv.DataSet):
            msg = 'The frame loader must return one PyVista dataset.'
            raise TypeError(msg)
        if isinstance(loaded, pv.PointSet):
            msg = 'PointSet frames require an explicit cast_to_polydata() before display.'
            raise TypeError(msg)
        data = loaded.point_data if self.preference == 'point' else loaded.cell_data
        if self.scalars not in data:
            msg = f'Frame {index} has no {self.preference} array {self.scalars!r}.'
            raise ValueError(msg)
        values = data[self.scalars]
        if loaded.n_points == 0 or values.size == 0:
            msg = f'Frame {index} is empty.'
            raise ValueError(msg)
        if values.ndim != 1 or values.dtype.kind not in 'iuf':
            msg = f'Frame {index} requires a one-component real numeric scalar array.'
            raise ValueError(msg)
        mesh = loaded.copy(deep=False)
        mesh.set_active_scalars(self.scalars, preference=self.preference)
        return mesh

    def select_time(self, value):
        """Show the nearest sample, clamping endpoints and breaking ties earlier."""
        value = float(value)
        if not math.isfinite(value):
            msg = 'Selected time must be finite.'
            raise ValueError(msg)
        if value <= self.times[0]:
            index = 0
        elif value >= self.times[-1]:
            index = len(self.times) - 1
        else:
            index = min(range(len(self.times)), key=lambda index: abs(self.times[index] - value))
        self.select_index(index)

    def select_index(self, index):
        """Display an exact saved sample; reject booleans and out-of-range indices."""
        if isinstance(index, bool):
            msg = 'Frame index must be an integer, not a boolean.'
            raise TypeError(msg)
        index = operator.index(index)
        if not 0 <= index < len(self.times):
            msg = 'Frame index is outside the available samples.'
            raise IndexError(msg)
        if index == self.index:
            return
        mesh = self._load(index)
        self.actor.mapper.dataset = mesh
        self.actor.mapper.scalar_range = self.clim
        if getattr(self.pl, 'mesh', None) is self._frame:
            self.pl.mesh = mesh
        self._frame = mesh
        self.index = index
        self.pl.render()


def demo():
    """Display three irregularly spaced frames with changing geometry."""
    mesh = pv.Plane(i_resolution=12, j_resolution=8)
    mesh['temperature'] = mesh.points[:, 0]
    frames = []
    for displacement in (0.0, 0.2, 0.8):
        frame = mesh.copy()
        frame.points[:, 2] = displacement * frame.points[:, 0]
        frame['temperature'] = frame['temperature'] + displacement
        frames.append(frame)
    pl = pv.Plotter()
    controller = TimeController(
        pl, [0.0, 0.2, 1.0], frames.__getitem__, scalars='temperature', clim=(-0.5, 1.5)
    )
    pl.camera_position = 'iso'
    pl.show()
    return controller


if __name__ == '__main__':
    demo()
