import numpy as np

from src.hud import LOG_PANEL_MAX_X2, LOG_STEER_GAP, render
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


def test_log_panel_clears_steering_panel_at_640():
    """At 640px wide (the segmentation size in config.yaml) the log panel
    and the steering panel must not overlap in x. Both panels are drawn
    with cv2 alpha-blending (_panel, alpha=0.45); on a white background a
    point darkened by only ONE panel reads ~140, but a point darkened by
    TWO overlapping panels (blended twice) reads ~77. A non-adaptive log
    panel that still reaches its historical 340px width at this frame
    width would overlap the steering panel and produce that double-dark
    pixel.
    """
    w, h = 640, 360
    cx = w // 2
    steer_x1 = cx - 110  # left edge of the steering panel, same formula as render()

    img = np.full((h, w, 3), 255, np.uint8)
    s = FrameState(idx=0, t=0.0, raw=img.copy(), img=img.copy())
    s.decision.log = ["a" * 60, "b" * 60]
    out = render(s, _cfg())

    y = h - 20  # a row inside both panels' y-range
    probe_x = steer_x1 + LOG_STEER_GAP  # just inside the steering panel
    px = out[y, probe_x]
    assert px[0] > 100, (
        f"pixel at x={probe_x} looks double-darkened (BGR={tuple(int(v) for v in px)}); "
        "the log panel overlaps the steering panel"
    )


def test_log_panel_full_width_unchanged_at_1280():
    """At 1280px wide the log panel must still reach its full historical
    340px width — the narrowing logic must be a no-op on wide frames."""
    w, h = 1280, 720
    img = np.full((h, w, 3), 255, np.uint8)
    s = FrameState(idx=0, t=0.0, raw=img.copy(), img=img.copy())
    s.decision.log = ["a" * 60, "b" * 60]
    out = render(s, _cfg())

    y = h - 20
    inside = out[y, LOG_PANEL_MAX_X2 - 1]
    outside = out[y, LOG_PANEL_MAX_X2 + 5]
    assert inside[0] < 200, f"log panel should still darken up to x={LOG_PANEL_MAX_X2}"
    assert outside[0] > 200, "log panel should stop at its historical 340px width"


def test_detection_box_colour_survives_lane_tint():
    """The lane/road tint must be drawn BEFORE detection boxes, not after,
    so box colours (which carry meaning: red = vulnerable road user) stay
    pure inside the drivable-area tint."""
    img = np.zeros((360, 640, 3), np.uint8)
    s = FrameState(idx=0, t=0.0, raw=img.copy(), img=img.copy())
    s.objects = [Track(id=1, cls="person", bbox=(100, 150, 200, 250), conf=0.9)]
    s.drivable = np.ones((360, 640), np.uint8)   # tint covers the whole frame
    out = render(s, _cfg())

    b, g, r = (int(v) for v in out[150, 150])   # a pixel on the box border
    assert g < 10, f"box colour washed toward green by the tint: BGR=({b},{g},{r})"
    assert r > 200
