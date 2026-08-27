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


def update_ttc(
    tracks: list[Track], prev_dist: dict[int, float], dt: float
) -> dict[int, float]:
    """Set `ttc_s` on each track from how fast its distance is shrinking.

    Returns the distance memo to pass in on the next frame. Tracks with an
    unknown distance are excluded so they cannot poison the next frame.
    """
    new: dict[int, float] = {}
    for t in tracks:
        if not np.isfinite(t.dist_m):
            continue
        new[t.id] = t.dist_m
        was = prev_dist.get(t.id)
        if was is None or dt <= 0:
            continue
        closing = (was - t.dist_m) / dt          # m/s, positive = approaching
        if closing > 0.1:
            t.ttc_s = t.dist_m / closing
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
        if res.boxes is None or res.boxes.id is None:
            return out

        for box, cid, tid, c in zip(
            res.boxes.xyxy.tolist(),
            res.boxes.cls.tolist(),
            res.boxes.id.tolist(),
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
