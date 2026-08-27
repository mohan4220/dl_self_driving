import numpy as np
import pytest

from src.segment import fit_lanes, freespace_offset

H, W = 360, 640


def _cfg():
    return {
        "segment": {
            "da_min_px_frac": 0.02,
            "ll_min_px_frac": 0.0015,
            "fit_bands": 12,
            "fit_min_bands": 4,
            "min_curve_radius_m": 80,
        },
        "camera": {
            "lane_width_m": 3.5,
            "seg_w": W,
            "seg_h": H,
            # Same IPM trapezoid as config.yaml's camera block.
            "ipm_src": [[250, 200], [390, 200], [610, 340], [30, 340]],
            "ipm_dst_m": [[-1.75, 30.0], [1.75, 30.0], [1.75, 5.0], [-1.75, 5.0]],
        },
    }


def _straight_lanes(left_x: int, right_x: int, y0: int = 180) -> np.ndarray:
    """Two vertical lane lines from y0 to the bottom of the frame."""
    m = np.zeros((H, W), np.uint8)
    m[y0:, left_x - 2:left_x + 2] = 1
    m[y0:, right_x - 2:right_x + 2] = 1
    return m


def _road(y0: int = 180) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    m[y0:, 100:540] = 1
    return m


def test_centred_straight_lanes_give_zero_offset():
    ll = _straight_lanes(240, 400)          # centre = 320 = image centre
    lanes = fit_lanes(ll, _road(), _cfg())
    assert lanes.valid is True
    assert lanes.offset_m == pytest.approx(0.0, abs=0.05)
    assert abs(lanes.heading_err_rad) < 0.02


def test_lane_shifted_left_means_ego_is_right_of_centre():
    """Lane centre at x=270 but camera at x=320 -> ego is right of centre.

    The offset is no longer a flat pixel-to-metre ratio scaled by lane
    width (that was the defective pixel-proxy algorithm this fix
    replaces) -- it comes from reprojecting the near ground-plane sample
    through the IPM homography built from config.yaml's camera.ipm_src /
    camera.ipm_dst_m. 0.297 m is what src/segment.py's fit_lanes itself
    measures for this synthetic geometry; the real assertion this test
    protects is the sign, per the project's offset_m convention.
    """
    ll = _straight_lanes(190, 350)          # centre = 270
    lanes = fit_lanes(ll, _road(), _cfg())
    assert lanes.offset_m == pytest.approx(0.297, abs=0.05)
    assert lanes.offset_m > 0


def test_empty_lane_mask_is_invalid():
    lanes = fit_lanes(np.zeros((H, W), np.uint8), _road(), _cfg())
    assert lanes.valid is False
    assert lanes.left_fit is None and lanes.right_fit is None


def test_single_line_is_invalid():
    m = np.zeros((H, W), np.uint8)
    m[180:, 238:242] = 1                    # only a left line
    assert fit_lanes(m, _road(), _cfg()).valid is False


def test_freespace_offset_from_centred_road_is_zero():
    assert freespace_offset(_road(), _cfg()) == pytest.approx(0.0, abs=0.1)


def test_freespace_offset_from_shifted_road_is_signed():
    m = np.zeros((H, W), np.uint8)
    m[180:, 0:440] = 1                      # road centroid at x=220, left of centre
    off = freespace_offset(m, _cfg())
    assert off is not None and off > 0      # ego is right of the free space


def test_freespace_offset_none_when_road_missing():
    assert freespace_offset(np.zeros((H, W), np.uint8), _cfg()) is None
