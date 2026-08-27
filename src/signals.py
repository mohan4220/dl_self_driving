"""Traffic lights, stop signs, and speed zones.

YOLO tells us WHERE a traffic light is but not what colour it shows, so the
state is read with plain HSV colour reasoning inside the box - no extra
model needed. Speed-limit numbers need OCR, which is expensive, so it runs
once per ByteTrack id and is cached: a 60 s clip with 8 signs costs 8 OCR
calls, not 900.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.types import Sign, SignalInfo, Track

# HSV hue ranges. Red wraps around 0, so it needs two windows.
_RED = (((0, 90, 90), (10, 255, 255)), ((170, 90, 90), (180, 255, 255)))
_AMBER = (((11, 90, 90), (32, 255, 255)),)
_GREEN = (((40, 60, 60), (90, 255, 255)),)


def _mask_frac(hsv: np.ndarray, ranges) -> float:
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return float(m.mean() / 255.0)


def classify_light(bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    """Read a traffic light's state from the lit lamp inside its box.

    Colour alone is ambiguous under exposure clipping, so lamp POSITION is
    used as a tie-break: real lights are red-amber-green top to bottom.
    """
    x1, y1, x2, y2 = bbox
    h, w = bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 6:
        return "unknown"

    crop = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    scores = {
        "red": _mask_frac(hsv, _RED),
        "amber": _mask_frac(hsv, _AMBER),
        "green": _mask_frac(hsv, _GREEN),
    }

    best = max(scores, key=scores.get)
    if scores[best] < 0.02:
        return "unknown"

    # Position tie-break: find the brightest third of the box.
    ch = crop.shape[0] // 3
    if ch >= 2:
        v = hsv[..., 2]
        thirds = [v[:ch].mean(), v[ch:2 * ch].mean(), v[2 * ch:].mean()]
        by_pos = ["red", "amber", "green"][int(np.argmax(thirds))]
        if scores[by_pos] > 0.4 * scores[best]:
            return by_pos
    return best


class SignReader:
    """Reads speed-limit numbers, at most once per tracked sign."""

    def __init__(self, min_box_px: int = 22):
        self.min_box_px = min_box_px
        self._cache: dict[int, int | None] = {}
        self._ocr = None

    def _ocr_number(self, crop: np.ndarray) -> int | None:
        """Run OCR and pull out a plausible speed limit."""
        if self._ocr is None:
            from rapidocr import RapidOCR

            self._ocr = RapidOCR()

        res = self._ocr(crop)
        texts = getattr(res, "txts", None) or []
        for txt in texts:
            digits = "".join(c for c in str(txt) if c.isdigit())
            if not digits:
                continue
            val = int(digits[:3])
            if 5 <= val <= 130:
                return val
        return None

    def read_speed_limit(
        self, bgr: np.ndarray, bbox: tuple[int, int, int, int], track_id: int
    ) -> int | None:
        if track_id in self._cache:
            return self._cache[track_id]

        x1, y1, x2, y2 = bbox
        if min(x2 - x1, y2 - y1) < self.min_box_px:
            return None                    # too small to read; retry when closer

        h, w = bgr.shape[:2]
        pad = 4
        crop = bgr[max(0, y1 - pad): min(h, y2 + pad),
                   max(0, x1 - pad): min(w, x2 + pad)]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        val = self._ocr_number(crop)
        self._cache[track_id] = val        # cache misses too: do not retry forever
        return val


def read_signals(
    bgr: np.ndarray,
    tracks: list[Track],
    reader: SignReader,
    prev: SignalInfo,
    cfg: dict,
) -> SignalInfo:
    """Collect light state, signs, and the active speed limit for this frame."""
    out = SignalInfo(speed_limit_kmh=prev.speed_limit_kmh)

    lights = [t for t in tracks if t.cls == "traffic_light"]
    if lights:
        # The nearest light governs us, and nearest means largest on screen.
        nearest = max(
            lights, key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])
        )
        out.light_state = classify_light(bgr, nearest.bbox)

    for t in tracks:
        if t.cls == "stop_sign":
            out.signs.append(Sign(kind="stop", value=None, bbox=t.bbox, track_id=t.id))
            val = reader.read_speed_limit(bgr, t.bbox, t.id)
            if val is not None:
                out.speed_limit_kmh = val
                out.signs.append(
                    Sign(kind="speed_limit", value=val, bbox=t.bbox, track_id=t.id)
                )
    return out


if __name__ == "__main__":
    import argparse

    import yaml

    from src.detect import Detector
    from src.video_io import VideoReader

    ap = argparse.ArgumentParser(description="Block 5 demo: lights, signs, speed zones.")
    ap.add_argument("--video", required=True)
    a = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    det = Detector(cfg["model"]["detector"], cfg["detect"]["imgsz"], cfg["detect"]["classes"])
    reader = SignReader(cfg["signals"]["min_sign_px"])
    info = SignalInfo()

    for _, _, frame in VideoReader(a.video):
        tracks = det.track(frame, cfg["detect"]["conf"])
        info = read_signals(frame, tracks, reader, info, cfg)
        for t in tracks:
            if t.cls in ("traffic_light", "stop_sign"):
                x1, y1, x2, y2 = t.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame, f"LIGHT {info.light_state.upper()}   ZONE {info.speed_limit_kmh} km/h",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2,
        )
        cv2.imshow("signals", frame)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
