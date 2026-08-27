"""Bird's-eye view: turning image pixels into metres on the ground.

Everything the planner reasons about - how far is that car, is it in my
lane, is there room to change lanes - is a question about the ground plane,
not about pixels. One homography, calibrated from four road points with
known real-world positions, converts between the two. No camera intrinsics
and no depth network required; the price is an assumption that the road is
flat, which is fine for the highway and city clips used here.

The homography itself (`build_homography`, `image_to_ground`,
`ground_to_image`) lives in `src.geometry` - it was built there in an
earlier block because lane geometry needed it too. It is re-exported here
so callers can reach it via `src.bev` as well.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.detect import bbox_bottom_centre
from src.geometry import build_homography, ground_to_image, image_to_ground
from src.types import BEVGrid, FrameState, Track

__all__ = [
    "build_homography",
    "image_to_ground",
    "ground_to_image",
    "track_distance",
    "annotate_distances",
    "build_grid",
]


def _to_seg_scale(
    pt: tuple[float, float], frame_size: tuple[int, int], cfg: dict
) -> tuple[float, float]:
    """Boxes arrive in full-frame pixels; the homography is calibrated at 640x360."""
    fw, fh = frame_size
    return (pt[0] * cfg["camera"]["seg_w"] / fw, pt[1] * cfg["camera"]["seg_h"] / fh)


def track_distance(
    H: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_size: tuple[int, int],
    cfg: dict | None = None,
) -> float:
    """Forward distance in metres to where this object touches the road."""
    cfg = cfg or {"camera": {"seg_w": 640, "seg_h": 360}}
    px = _to_seg_scale(bbox_bottom_centre(bbox), frame_size, cfg)
    x_m, y_m = image_to_ground(H, np.array([px], np.float32))[0]
    if not np.isfinite(y_m) or y_m <= 0:
        return float("inf")
    return float(y_m)


def annotate_distances(
    tracks: list[Track], H: np.ndarray, frame_size: tuple[int, int], cfg: dict
) -> None:
    """Fill in `dist_m`, `lateral_m` and `in_path` for every track, in place."""
    corridor = cfg["bev"]["corridor_m"]
    for t in tracks:
        px = _to_seg_scale(bbox_bottom_centre(t.bbox), frame_size, cfg)
        x_m, y_m = image_to_ground(H, np.array([px], np.float32))[0]
        if not np.isfinite(y_m) or y_m <= 0:
            t.dist_m, t.lateral_m, t.in_path = float("inf"), float("nan"), False
            continue
        t.dist_m = float(y_m)
        t.lateral_m = float(x_m)
        t.in_path = abs(float(x_m)) < corridor


def build_grid(state: FrameState, cfg: dict) -> BEVGrid:
    """Rasterise tracked objects into an ego-centric top-down occupancy grid.

    The ego sits at the bottom-centre, looking up the image.
    """
    b = cfg["bev"]
    mpp = b["m_per_px"]
    gh, gw = int(b["range_m"] / mpp), int(b["width_m"] / mpp)
    grid = np.zeros((gh, gw), np.uint8)
    origin = (gw // 2, gh - 1)                      # ego rear axle, in grid pixels

    for t in state.objects:
        if not np.isfinite(t.dist_m) or not np.isfinite(t.lateral_m):
            continue
        px = int(origin[0] + t.lateral_m / mpp)
        py = int(origin[1] - t.dist_m / mpp)
        if 0 <= px < gw and 0 <= py < gh:
            r = max(2, int(1.8 / mpp / 2))
            cv2.circle(grid, (px, py), r, 255, -1)

    return BEVGrid(grid=grid, m_per_px=mpp, origin_px=origin)


if __name__ == "__main__":
    import argparse

    import yaml

    from src.video_io import VideoReader

    ap = argparse.ArgumentParser(
        description="Block 6 demo / IPM calibration. Check the trapezoid hugs the lane."
    )
    ap.add_argument("--video", required=True)
    a = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    H = build_homography(cfg)
    src = np.array(cfg["camera"]["ipm_src"], np.int32)

    for _, _, frame in VideoReader(a.video):
        vis = cv2.resize(frame, (cfg["camera"]["seg_w"], cfg["camera"]["seg_h"]))
        cv2.polylines(vis, [src], True, (0, 255, 255), 2)
        for (x, y), (gx, gy) in zip(src, cfg["camera"]["ipm_dst_m"]):
            cv2.putText(vis, f"({gx},{gy})m", (x - 30, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        # Distance ruler along the lane centre.
        for d in (5, 10, 20, 30, 40):
            p = ground_to_image(H, np.array([[0.0, float(d)]], np.float32))[0]
            if 0 <= p[0] < vis.shape[1] and 0 <= p[1] < vis.shape[0]:
                cv2.line(vis, (int(p[0]) - 25, int(p[1])), (int(p[0]) + 25, int(p[1])),
                         (255, 255, 0), 1)
                cv2.putText(vis, f"{d}m", (int(p[0]) + 30, int(p[1]) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.imshow("bev calib", vis)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
