"""Object detection and tracking.

YOLO11n on COCO already covers every class this project needs: vehicles,
people, animals, traffic lights and stop signs. ByteTrack (bundled with
ultralytics) gives stable ids across frames, which is what makes
time-to-collision and once-per-sign OCR possible.
"""
from __future__ import annotations

import numpy as np

from src.types import Track

# COCO ids we care about, mapped to the names our planner reasons about.
COCO_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic_light",
    11: "stop_sign",
    15: "animal", 16: "animal", 17: "animal", 18: "animal",
    19: "animal", 20: "animal", 21: "animal", 22: "animal", 23: "animal",
}

VEHICLE = {"car", "motorcycle", "bus", "truck", "bicycle"}
VULNERABLE = {"person", "animal"}


def bbox_bottom_centre(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Where the object touches the ground, in image pixels."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


TTC_EMA_ALPHA = 0.3   # smoothing factor for per-track distance before differencing


class DistMemo(dict):
    """The distance memo `update_ttc` returns and consumes.

    Behaves exactly like `dict[int, float]` (the raw distance per track id,
    same as before this fix) for every existing caller and test. It also
    carries a hidden `smoothed` attribute with the EMA-smoothed distance per
    track id, used internally to debounce TTC differencing. Instance
    attributes do not participate in dict equality, so `DistMemo({1: 2.0})
    == {1: 2.0}` is still True — no plumbing changes needed at any call site.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.smoothed: dict[int, float] = {}


def update_ttc(
    tracks: list[Track], prev_dist: dict[int, float], dt: float
) -> dict[int, float]:
    """Set `ttc_s` on each track from how fast its distance is shrinking.

    A raw box-bottom edge jitters by a few pixels between frames; at typical
    frame intervals (~0.07s) that jitter differences into tens of m/s of
    spurious "closing" speed for an object that is not actually closing. To
    stop that, distance is smoothed per track id with an EMA (alpha=0.3)
    before differencing. The smoothing state rides along as a hidden
    attribute on the returned memo (see `DistMemo`) rather than as a new
    parameter, so the dict every caller already threads through the
    pipeline is unaffected in shape or equality.

    Returns the distance memo to pass in on the next frame. Tracks with an
    unknown distance, or an unassociated (-1) id, are excluded so they
    cannot poison the next frame.
    """
    prev_smoothed = getattr(prev_dist, "smoothed", {})
    new = DistMemo()
    for t in tracks:
        if not np.isfinite(t.dist_m) or t.id < 0:
            continue
        new[t.id] = t.dist_m

        was_smoothed = prev_smoothed.get(t.id)
        if was_smoothed is None:
            smoothed_now = t.dist_m               # nothing to smooth against yet
            anchor = prev_dist.get(t.id)           # fall back to the raw memo
        else:
            smoothed_now = (
                TTC_EMA_ALPHA * t.dist_m + (1.0 - TTC_EMA_ALPHA) * was_smoothed
            )
            anchor = was_smoothed
        new.smoothed[t.id] = smoothed_now

        if anchor is None or dt <= 0:
            continue
        closing = (anchor - smoothed_now) / dt   # m/s, positive = approaching
        if closing > 0.1:
            t.ttc_s = smoothed_now / closing
    return new


class Detector:
    def __init__(
        self, weights: str, imgsz: int = 640, classes: list[int] | None = None
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.classes = classes

    def track(self, bgr: np.ndarray, conf: float = 0.35) -> list[Track]:
        res = self.model.track(
            bgr,
            imgsz=self.imgsz,
            conf=conf,
            classes=self.classes,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        out: list[Track] = []
        if res.boxes is None:
            return out

        # ByteTrack sometimes finds boxes it cannot associate with an id
        # (id is None for the whole frame). Fall back to id=-1 rather than
        # discarding the detections outright: a dropped-to-empty frame is
        # what clears light_state to "none" and flickers the planner out of
        # STOP_SIGNAL for a frame. -1 boxes still reach the planner; they
        # just never accumulate TTC/OCR history (ids can collide frame to
        # frame), which update_ttc and SignReader both guard against.
        if res.boxes.id is None:
            ids = [-1] * len(res.boxes.xyxy)
        else:
            ids = res.boxes.id.tolist()

        for box, cid, tid, c in zip(
            res.boxes.xyxy.tolist(),
            res.boxes.cls.tolist(),
            ids,
            res.boxes.conf.tolist(),
        ):
            name = COCO_NAMES.get(int(cid))
            if name is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            out.append(
                Track(id=int(tid), cls=name, bbox=(x1, y1, x2, y2), conf=float(c))
            )
        return out


if __name__ == "__main__":
    import argparse

    import cv2
    import yaml

    from src.video_io import VideoReader

    ap = argparse.ArgumentParser(description="Block 2 demo: detection + tracking.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=60)
    a = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    det = Detector(
        cfg["model"]["detector"],
        imgsz=cfg["detect"]["imgsz"],
        classes=cfg["detect"]["classes"],
    )

    for _, _, frame in VideoReader(a.video, max_frames=a.frames):
        for t in det.track(frame, cfg["detect"]["conf"]):
            colour = (0, 0, 255) if t.cls in VULNERABLE else (0, 200, 255)
            x1, y1, x2, y2 = t.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                frame, f"{t.cls}#{t.id}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
            )
        cv2.imshow("detect", frame)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
