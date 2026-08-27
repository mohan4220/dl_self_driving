import numpy as np
from src.types import FrameState, Track, LaneInfo, Decision, Control


def test_framestate_defaults_are_independent():
    """Mutable defaults must not be shared between instances."""
    img = np.zeros((360, 640, 3), np.uint8)
    a = FrameState(idx=0, t=0.0, raw=img, img=img)
    b = FrameState(idx=1, t=0.1, raw=img, img=img)

    a.objects.append(Track(id=1, cls="car", bbox=(0, 0, 10, 10), conf=0.9))
    a.decision.log.append("hello")

    assert b.objects == []
    assert b.decision.log == []


def test_framestate_sensible_defaults():
    img = np.zeros((360, 640, 3), np.uint8)
    s = FrameState(idx=0, t=0.0, raw=img, img=img)

    assert s.weather.label == "clear"
    assert s.weather.visibility == 1.0
    assert s.lanes.valid is False
    assert s.signals.light_state == "none"
    assert s.decision.fsm_state == "LANE_KEEP"
    assert s.decision.indicator == "off"
    assert s.control.steer_deg == 0.0
    assert s.drivable is None


def test_track_defaults_to_unknown_distance():
    t = Track(id=3, cls="person", bbox=(1, 2, 3, 4), conf=0.5)
    assert t.dist_m == float("inf")
    assert t.ttc_s == float("inf")
    assert t.in_path is False
    assert np.isnan(t.lateral_m)
