"""Test controller decisions with plotting doubles; no render baseline is claimed."""

import contextlib
from types import SimpleNamespace

from exploration.time_controller import TimeController


@contextlib.contextmanager
def raises(error, *, match):
    """Check exception type and message without requiring test dependencies."""
    try:
        yield
    except error as exc:
        assert match in str(exc)
    else:
        msg = f"Expected {error.__name__}"
        raise AssertionError(msg)


class Frame:
    """Model the point-array operations needed by the controller."""

    def __init__(self, value):
        """Create an identifiable frame."""
        self.value = value
        self.point_data = {'temperature': [value]}

    def copy(self, *, deep):
        """Return an independent frame wrapper."""
        frame = Frame(self.value)
        frame.point_data = self.point_data.copy()
        return frame

    def set_active_scalars(self, name, *, preference):
        """Check scalar selection arguments."""
        assert name == 'temperature'
        assert preference == 'point'


class Scene:
    """Record actor, camera, and widget behavior without opening a window."""

    def __init__(self):
        """Initialize camera and render records."""
        self.camera = object()
        self.renders = 0
        self.actor_count = 0

    def add_mesh(self, mesh, **kwargs):
        """Record the initial actor and fixed color limits."""
        self.actor_count += 1
        return SimpleNamespace(mapper=SimpleNamespace(dataset=mesh, scalar_range=kwargs['clim']))

    def add_slider_widget(self, callback, rng, **kwargs):
        """Model the immediate callback made when a slider is constructed."""
        callback(kwargs['value'])
        self.callback = callback
        return SimpleNamespace(rng=rng)

    def render(self):
        """Record rendering requested by successful frame changes."""
        self.renders += 1


def test_slider_selects_irregular_time_and_preserves_actor():
    """Select by physical time with an earlier-frame tie and stable color limits."""
    pl = Scene()
    camera = pl.camera
    loaded = []

    def load(index):
        """Record each frame requested."""
        loaded.append(index)
        return Frame(index)

    controller = TimeController(pl, [0, 0.2, 1], load, scalars='temperature', clim=(-1, 3))
    actor = controller.actor
    pl.callback(0.6)
    assert controller.index == 1
    pl.callback(0.9)
    assert controller.index == 2
    assert actor.mapper.dataset.value == 2
    assert actor.mapper.scalar_range == (-1, 3)
    assert controller.actor is actor
    assert pl.actor_count == 1
    assert pl.camera is camera
    assert loaded == [0, 1, 2]
    assert pl.renders == 2
    pl.callback(100)
    assert loaded == [0, 1, 2]
    pl.callback(-100)
    assert controller.index == 0


def test_invalid_times():
    """Reject ambiguous or unusable timestamps before creating an actor."""
    for times in [[], [0, 0], [1, 0], [float('nan')]]:
        pl = Scene()
        with raises(ValueError, match='Times must'):
            TimeController(pl, times, Frame, scalars='temperature', clim=(0, 1))
        assert pl.actor_count == 0


def test_reader_restores_time_after_success_and_failure():
    """A reader error must leave the previous scene and reader selection intact."""
    reader = SimpleNamespace(time_values=[0, 0.2, 1], active_time_value=0.2)

    def select_index(index):
        """Set the reader's active time by index."""
        reader.active_time_value = reader.time_values[index]

    def select_value(value):
        """Restore the reader's active time value."""
        reader.active_time_value = value

    def read():
        """Return one frame, failing on the final sample."""
        if reader.active_time_value == 1:
            msg = 'Unreadable frame'
            raise OSError(msg)
        return Frame(reader.active_time_value)

    reader.set_active_time_point = select_index
    reader.set_active_time_value = select_value
    reader.read = read
    pl = Scene()
    controller = TimeController.from_reader(pl, reader, scalars='temperature', clim=(0, 2))
    assert reader.active_time_value == 0.2
    original = controller.actor.mapper.dataset
    with raises(OSError, match='Unreadable frame'):
        controller.select_time(1)
    assert reader.active_time_value == 0.2
    assert controller.actor.mapper.dataset is original
    assert controller.index == 0
    assert pl.renders == 0


def test_single_frame_has_no_slider():
    """Support a single frame without creating a degenerate slider."""
    controller = TimeController(Scene(), [3], Frame, scalars='temperature', clim=(0, 1))
    assert controller.slider is None


def test_failed_selection_preserves_scene():
    """Reject invalid queries and missing scalars without replacing the dataset."""
    def load(index):
        """Return a second frame without the requested array."""
        frame = Frame(index)
        if index == 1:
            frame.point_data.clear()
        return frame

    pl = Scene()
    controller = TimeController(pl, [0, 1], load, scalars='temperature', clim=(0, 2))
    original = controller.actor.mapper.dataset
    for value in (float('nan'), float('inf')):
        with raises(ValueError, match='Selected time must be finite'):
            controller.select_time(value)
    with raises(ValueError, match='has no point array'):
        controller.select_time(1)
    assert controller.index == 0
    assert controller.actor.mapper.dataset is original
    assert pl.renders == 0


if __name__ == '__main__':
    test_slider_selects_irregular_time_and_preserves_actor()
    test_invalid_times()
    test_reader_restores_time_after_success_and_failure()
    test_single_frame_has_no_slider()
    test_failed_selection_preserves_scene()
    print('5 controller checks passed (including invalid timestamps and selection).')
