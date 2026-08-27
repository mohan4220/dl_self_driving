"""Ground-plane geometry: the bridge between image pixels and metres.

Four road points with known real-world positions fix a homography, so
distances and angles can be computed in metres without camera intrinsics.
The price is an assumption that the road surface is flat.
"""
from __future__ import annotations

import cv2
import numpy as np


def build_homography(cfg: dict) -> np.ndarray:
    """3x3 mapping 640x360 image pixels to ground metres (x_lateral, y_forward)."""
    cam = cfg["camera"]
    src = np.array(cam["ipm_src"], np.float32)
    dst = np.array(cam["ipm_dst_m"], np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def image_to_ground(H: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_px, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def ground_to_image(H: np.ndarray, pts_m: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_m, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)
