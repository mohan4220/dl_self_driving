"""Road and lane perception.

TwinLiteNet gives two masks from one forward pass: the drivable area and
the lane lines. The drivable area is what keeps this system honest in bad
weather — when paint is covered by snow or washed out by rain, the lane
fit dies but the free-space mask survives, so we can still steer.

The INT8-quantized model runs at 144 ms/frame on the target CPU versus
774-1350 ms for FP32, at 0.985 / 0.941 mask IoU against FP32.
"""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from src.types import LaneInfo

SEG_W, SEG_H = 640, 360      # fixed by the model; not configurable


class Segmenter:
    def __init__(self, model_path: str, threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(
            model_path, so, providers=["CPUExecutionProvider"]
        )

    def infer(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (drivable_mask, lane_mask), uint8 0/1, both 360x640."""
        small = cv2.resize(bgr, (SEG_W, SEG_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        da, ll = self.sess.run(["da", "ll"], {"images": x})
        return (
            (np.argmax(da[0], 0) == 1).astype(np.uint8),
            (np.argmax(ll[0], 0) == 1).astype(np.uint8),
        )


def _band_centres(mask: np.ndarray, cfg: dict) -> tuple[list, list]:
    """Sample lane pixels in horizontal bands, split into left and right.

    Working in bands rather than fitting all pixels at once keeps far-away
    noise from dominating the fit, because every band contributes equally.
    """
    h, w = mask.shape
    n = cfg["segment"]["fit_bands"]
    cx = w / 2.0
    left, right = [], []
    for i in range(n):
        y0, y1 = int(h * i / n), int(h * (i + 1) / n)
        band = mask[y0:y1]
        ys, xs = np.nonzero(band)
        if xs.size == 0:
            continue
        y = (y0 + y1) / 2.0
        lx, rx = xs[xs < cx], xs[xs >= cx]
        if lx.size >= 3:
            left.append((y, float(np.median(lx))))
        if rx.size >= 3:
            right.append((y, float(np.median(rx))))
    return left, right


def _fit(points: list) -> np.ndarray | None:
    """Fit x = f(y) as a quadratic. Needs at least 3 points."""
    if len(points) < 3:
        return None
    ys = np.array([p[0] for p in points])
    xs = np.array([p[1] for p in points])
    return np.polyfit(ys, xs, 2)


def fit_lanes(ll_mask: np.ndarray, da_mask: np.ndarray, cfg: dict) -> LaneInfo:
    """Turn the lane-line mask into a centreline offset and heading error."""
    seg = cfg["segment"]
    info = LaneInfo()

    if ll_mask.mean() < seg["ll_min_px_frac"]:
        return info                                   # nothing to fit

    left_pts, right_pts = _band_centres(ll_mask, cfg)
    if len(left_pts) < seg["fit_min_bands"] or len(right_pts) < seg["fit_min_bands"]:
        return info

    lf, rf = _fit(left_pts), _fit(right_pts)
    if lf is None or rf is None:
        return info

    h, w = ll_mask.shape
    y_near = h - 1.0
    xl, xr = float(np.polyval(lf, y_near)), float(np.polyval(rf, y_near))
    lane_px = xr - xl
    if lane_px < 40:                                  # degenerate / crossed fits
        return info

    # Pixels to metres, calibrated by the known lane width. This avoids
    # needing camera intrinsics for the lateral measurement.
    m_per_px = cfg["camera"]["lane_width_m"] / lane_px

    lane_cx = (xl + xr) / 2.0
    info.left_fit, info.right_fit = lf, rf
    info.offset_m = (w / 2.0 - lane_cx) * m_per_px    # + = ego right of centre

    # Heading error from the centreline slope dx/dy at the bottom of the frame.
    centre = (lf + rf) / 2.0
    dxdy = float(np.polyval(np.polyder(centre), y_near))
    info.heading_err_rad = float(np.arctan(-dxdy))    # + = lane heads right

    d2 = float(np.polyval(np.polyder(centre, 2), y_near))
    info.curvature_m = (
        float("inf") if abs(d2) < 1e-9
        else abs((1 + dxdy ** 2) ** 1.5 / d2) * m_per_px
    )
    info.valid = True
    return info


def freespace_offset(da_mask: np.ndarray, cfg: dict) -> float | None:
    """Fallback steering target: the centroid of drivable pixels ahead.

    Used when lane paint is invisible (snow, heavy rain, night). Returns
    metres, positive when the ego is right of the free space, or None when
    there is no usable road surface at all.
    """
    if da_mask.mean() < cfg["segment"]["da_min_px_frac"]:
        return None
    h, w = da_mask.shape
    band = da_mask[int(h * 0.55): int(h * 0.85)]      # mid-distance road only
    xs = np.nonzero(band)[1]
    if xs.size == 0:
        return None
    # Scale by an assumed lane width spanning a quarter of the frame width.
    m_per_px = cfg["camera"]["lane_width_m"] / (w / 4.0)
    return float((w / 2.0 - xs.mean()) * m_per_px)


if __name__ == "__main__":
    import argparse
    import time

    import yaml

    from src.video_io import VideoReader

    ap = argparse.ArgumentParser(description="Block 3 demo: road + lane segmentation.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--fp32", action="store_true", help="use the unquantized model")
    a = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    path = cfg["model"]["segmenter_fp32"] if a.fp32 else cfg["model"]["segmenter"]
    seg = Segmenter(path, cfg["model"]["onnx_threads"])
    print(f"model: {path}")

    for idx, _, frame in VideoReader(a.video):
        t0 = time.perf_counter()
        da, ll = seg.infer(frame)
        ms = (time.perf_counter() - t0) * 1000
        lanes = fit_lanes(ll, da, cfg)

        vis = cv2.resize(frame, (SEG_W, SEG_H))
        vis[da.astype(bool)] = (0.6 * vis[da.astype(bool)]
                                + 0.4 * np.array([0, 180, 0])).astype(np.uint8)
        vis[ll.astype(bool)] = (0, 0, 255)
        if lanes.valid:
            for y in range(180, SEG_H, 10):
                xc = int((np.polyval(lanes.left_fit, y)
                          + np.polyval(lanes.right_fit, y)) / 2)
                cv2.circle(vis, (xc, y), 2, (255, 255, 0), -1)
        txt = (f"{ms:.0f}ms  valid={lanes.valid}  "
               f"off={lanes.offset_m:+.2f}m  hdg={np.degrees(lanes.heading_err_rad):+.1f}deg")
        cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("segment", vis)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
