import pytest

from src.detect import bbox_bottom_centre, update_ttc
from src.types import Track


def test_bbox_bottom_centre():
    assert bbox_bottom_centre((10, 20, 30, 60)) == (20.0, 60.0)


def test_ttc_closing_object():
    """Object closes 5 m in 0.5 s -> 10 m/s closing -> 20 m gap = 2.0 s."""
    t = Track(id=1, cls="car", bbox=(0, 0, 1, 1), conf=0.9, dist_m=20.0)
    prev = update_ttc([t], {1: 25.0}, dt=0.5)
    assert t.ttc_s == pytest.approx(2.0)
    assert prev == {1: 20.0}


def test_ttc_receding_object_is_infinite():
    t = Track(id=1, cls="car", bbox=(0, 0, 1, 1), conf=0.9, dist_m=30.0)
    update_ttc([t], {1: 25.0}, dt=0.5)
    assert t.ttc_s == float("inf")


def test_ttc_unknown_history_is_infinite():
    t = Track(id=7, cls="car", bbox=(0, 0, 1, 1), conf=0.9, dist_m=10.0)
    update_ttc([t], {}, dt=0.5)
    assert t.ttc_s == float("inf")


def test_ttc_ignores_unknown_distance():
    t = Track(id=1, cls="car", bbox=(0, 0, 1, 1), conf=0.9)  # dist_m = inf
    prev = update_ttc([t], {1: 25.0}, dt=0.5)
    assert t.ttc_s == float("inf")
    assert 1 not in prev
