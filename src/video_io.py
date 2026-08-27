"""Frame in, frame out, plus the control-signal log.

The control log is the artifact that stands in for a CAN bus trace: one
JSON record per processed frame describing what we would have sent to the
drive hardware.
"""
from __future__ import annotations

import json
from typing import Iterator

import cv2
import numpy as np

from src.types import FrameState


class VideoReader:
    """Iterate a video file, optionally taking every `stride`-th frame.

    Timestamps are derived from the SOURCE fps, so they stay physically
    correct no matter what stride we process at.
    """

    def __init__(self, path: str, stride: int = 1, max_frames: int | None = None):
        self.path = path
        self.stride = max(1, int(stride))
        self.max_frames = max_frames
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Container metadata is not always trustworthy: WebM in particular can
        # report a garbage count (observed: -5.5e17). Treat anything implausible
        # as unknown rather than passing it downstream as if it were real.
        raw = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.n_frames = int(raw) if 0 < raw < 1e9 else 0   # 0 == unknown

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        idx, emitted = 0, 0
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                if idx % self.stride == 0:
                    yield idx, idx / self.fps, frame
                    emitted += 1
                    if self.max_frames is not None and emitted >= self.max_frames:
                        break
                idx += 1
        finally:
            self._cap.release()

    def __len__(self) -> int:
        return self.n_frames // self.stride

    @property
    def expected_frames(self) -> int | None:
        """How many frames this reader will yield, or None if unknowable."""
        if self.n_frames <= 0:
            return None
        n = self.n_frames // self.stride
        return min(n, self.max_frames) if self.max_frames is not None else n


class VideoWriter:
    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        self._w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._w.isOpened():
            raise IOError(f"cannot open writer: {path}")
        self.size = size

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self._w.write(frame)

    def close(self) -> None:
        self._w.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ControlLogWriter:
    """One JSON object per line: the simulated signal to drive hardware."""

    def __init__(self, path: str):
        self._f = open(path, "w")

    def write(self, s: FrameState) -> None:
        rec = {
            "idx": s.idx,
            "t": round(s.t, 3),
            "steer_deg": round(s.control.steer_deg, 2),
            "throttle": round(s.control.throttle, 3),
            "brake": round(s.control.brake, 3),
            "target_speed_kmh": round(s.decision.target_speed, 1),
            "fsm_state": s.decision.fsm_state,
            "indicator": s.decision.indicator,
            "weather": s.weather.label,
            "visibility": round(s.weather.visibility, 3),
            "speed_limit_kmh": s.signals.speed_limit_kmh,
            "light": s.signals.light_state,
            "n_objects": len(s.objects),
            "lane_valid": s.lanes.valid,
            "events": list(s.decision.log),
        }
        self._f.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "ControlLogWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Block 0 demo: read a video, show frame stats.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    r = VideoReader(a.video, stride=a.stride)
    print(f"{a.video}: {r.width}x{r.height} @ {r.fps:.1f} fps, {r.n_frames} frames")
    for idx, t, frame in r:
        print(f"  frame {idx:5d}  t={t:6.2f}s  mean={frame.mean():6.1f}")
