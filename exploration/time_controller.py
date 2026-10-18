"""Explore a slider controller without adding public PyVista API."""

import math


class TimeController:
    """Connect finite, increasing timestamps and a frame loader to one actor.

    The loader receives an integer frame index and returns a PyVista dataset.
    Supply point scalars and a fixed color range explicitly to avoid scanning
    a potentially large series just to display its first frame.
    """

    def __init__(self, pl, times, load_frame, *, scalars, clim):
        """Create the actor and a time slider using public plotting methods."""
        self.times = tuple(float(value) for value in times)
        if not self.times or not all(math.isfinite(value) for value in self.times):
            msg = 'Times must be nonempty and finite.'
            raise ValueError(msg)
        if any(right <= left for left, right in zip(self.times, self.times[1:])):
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
        self.actor = pl.add_mesh(mesh, scalars=scalars, preference='point', clim=self.clim)
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
    def from_reader(cls, pl, reader, *, scalars, clim):
        """Adapt a TimeReader that returns one dataset, restoring its active time."""
        times = tuple(reader.time_values)

        def load_frame(index):
            """Read a detached frame without changing the reader's selected time."""
            previous = reader.active_time_value
            try:
                reader.set_active_time_point(index)
                return reader.read().copy(deep=True)
            finally:
                reader.set_active_time_value(previous)

        return cls(pl, times, load_frame, scalars=scalars, clim=clim)

    def _load(self, index):
        """Validate the frame before changing what the plotter displays."""
        mesh = self.load_frame(index).copy(deep=False)
        if self.scalars not in mesh.point_data:
            msg = f'Frame {index} has no point array {self.scalars!r}.'
            raise ValueError(msg)
        mesh.set_active_scalars(self.scalars, preference='point')
        return mesh

    def select_time(self, value):
        """Show the nearest sample, clamping endpoints and breaking ties earlier."""
        value = float(value)
        if not math.isfinite(value):
            msg = 'Selected time must be finite.'
            raise ValueError(msg)
        index = min(range(len(self.times)), key=lambda index: abs(self.times[index] - value))
        if index == self.index:
            return
        mesh = self._load(index)
        self.actor.mapper.dataset = mesh
        self.actor.mapper.scalar_range = self.clim
        self.index = index
        self.pl.render()


def demo():
    """Display three irregularly spaced frames with changing geometry."""
    import pyvista as pv

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
