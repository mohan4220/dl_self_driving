import numpy as np

from src.hud import render
from src.types import FrameState, Track


def _cfg():
    return {"hud": {"log_lines": 5, "font_scale": 0.5}}


def _state():
    img = np.zeros((360, 640, 3), np.uint8)
    s = FrameState(idx=42, t=1.4, raw=img.copy(), img=img.copy())
    s.control.steer_deg = -7.2
    s.decision.target_speed = 30.0
    s.decision.fsm_state = "LANE_KEEP"
    s.decision.indicator = "right"
    s.decision.log = ["SPEED ZONE 30 DETECTED"]
    s.objects = [Track(id=1, cls="car", bbox=(100, 150, 200, 250), conf=0.9)]
    return s


def test_render_does_not_mutate_input_frame():
    s = _state()
    before = s.img.copy()
    render(s, _cfg())
    assert np.array_equal(s.img, before)


def test_render_returns_same_shape_and_draws_something():
    s = _state()
    out = render(s, _cfg())
    assert out.shape == s.img.shape
    assert out.dtype == np.uint8
    assert out.sum() > 0          # a black frame in, pixels drawn out


def test_render_survives_empty_state():
    """A frame where every perception stage failed must still render."""
    img = np.zeros((360, 640, 3), np.uint8)
    out = render(FrameState(idx=0, t=0.0, raw=img, img=img), _cfg())
    assert out.shape == img.shape


def test_render_log_is_capped():
    s = _state()
    s.decision.log = [f"line {i}" for i in range(50)]
    out = render(s, _cfg())     # must not raise or run off the frame
    assert out.shape == s.img.shape
