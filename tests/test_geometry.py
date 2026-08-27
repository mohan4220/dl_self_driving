import numpy as np
import pytest

from src.geometry import build_homography, ground_to_image, image_to_ground

# Same trapezoid as config.yaml's camera.ipm_src / camera.ipm_dst_m.
IPM_SRC = [[250, 200], [390, 200], [610, 340], [30, 340]]
IPM_DST_M = [[-1.75, 30.0], [1.75, 30.0], [1.75, 5.0], [-1.75, 5.0]]


def _cfg():
    return {"camera": {"ipm_src": IPM_SRC, "ipm_dst_m": IPM_DST_M}}


def test_calibration_points_map_exactly():
    H = build_homography(_cfg())
    ground = image_to_ground(H, np.array(IPM_SRC, np.float32))
    np.testing.assert_allclose(ground, np.array(IPM_DST_M, np.float32), atol=1e-3)


def test_round_trip_image_ground_image():
    H = build_homography(_cfg())
    pts_px = np.array([[300.0, 250.0], [350.0, 300.0], [200.0, 220.0]], np.float32)
    ground = image_to_ground(H, pts_px)
    back = ground_to_image(H, ground)
    np.testing.assert_allclose(back, pts_px, atol=1e-2)


def test_lower_in_frame_is_closer():
    H = build_homography(_cfg())
    lower = image_to_ground(H, np.array([[320.0, 330.0]], np.float32))[0]
    higher = image_to_ground(H, np.array([[320.0, 210.0]], np.float32))[0]
    assert lower[1] < higher[1]
