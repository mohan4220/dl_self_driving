"""The overlay: what the driver (or the examiner) sees.

Everything drawn here is a COMMAND, not a measurement. The speed shown is
what the controller asked for, labelled CMD SPD, because ego speed is not
recoverable from a monocular dashcam.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.detect import VULNERABLE
from src.types import FrameState

WHITE = (255, 255, 255)
GREY = (160, 160, 160)
GREEN = (0, 220, 0)
AMBER = (0, 190, 255)
RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
LOG_PANEL_MAX_X2 = 340   # log panel's right edge on a wide frame (unchanged behaviour)
LOG_STEER_GAP = 10       # minimum clearance between the log panel and the steering panel

STATE_COLOUR = {
    "LANE_KEEP": GREEN,
    "FOLLOW": GREEN,
    "PREP_CHANGE_L": AMBER,
    "PREP_CHANGE_R": AMBER,
    "CHANGING": AMBER,
    "STOP_SIGNAL": RED,
    "YIELD_PEDESTRIAN": RED,
    "EMERGENCY_BRAKE": RED,
}


def _panel(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.45):
    """Darken a rectangle so text stays readable over bright road scenes."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = (roi * (1 - alpha)).astype(np.uint8)


def _steering_wheel(img: np.ndarray, cx: int, cy: int, r: int, angle_deg: float):
    """A wheel that physically rotates with the commanded steering angle."""
    cv2.circle(img, (cx, cy), r, WHITE, 2)
    a = np.radians(angle_deg)
    for base in (0.0, np.pi * 2 / 3, np.pi * 4 / 3):
        th = base + a
        cv2.line(
            img, (cx, cy),
            (int(cx + r * np.sin(th)), int(cy - r * np.cos(th))),
            WHITE, 2,
        )
    cv2.circle(img, (cx, cy), 4, WHITE, -1)


def _indicators(img: np.ndarray, cx: int, y: int, which: str, blink_on: bool):
    left = GREEN if (which == "left" and blink_on) else GREY
    right = GREEN if (which == "right" and blink_on) else GREY
    cv2.drawContours(
        img, [np.array([[cx - 60, y], [cx - 35, y - 12], [cx - 35, y + 12]])],
        0, left, -1,
    )
    cv2.drawContours(
        img, [np.array([[cx + 60, y], [cx + 35, y - 12], [cx + 35, y + 12]])],
        0, right, -1,
    )


def render(state: FrameState, cfg: dict) -> np.ndarray:
    """Draw the full HUD. Returns a new image; `state.img` is untouched."""
    img = state.img.copy()
    h, w = img.shape[:2]
    hud = cfg["hud"]
    fs = hud["font_scale"]

    # --- lane overlay (drawn before detections so box colours stay pure) ---
    if state.drivable is not None:
        mask = cv2.resize(state.drivable, (w, h), interpolation=cv2.INTER_NEAREST)
        tint = np.zeros_like(img)
        tint[mask.astype(bool)] = (0, 120, 0)
        cv2.addWeighted(img, 1.0, tint, 0.35, 0, img)
    if state.lane_mask is not None:
        mask = cv2.resize(state.lane_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        img[mask.astype(bool)] = RED

    # --- detections ---
    for t in state.objects:
        x1, y1, x2, y2 = t.bbox
        colour = RED if t.cls in VULNERABLE else (AMBER if t.in_path else (0, 200, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        label = f"{t.cls}#{t.id}"
        if np.isfinite(t.dist_m):
            label += f" {t.dist_m:.0f}m"
        if np.isfinite(t.ttc_s):
            label += f" ttc{t.ttc_s:.1f}s"
        cv2.putText(img, label, (x1, max(12, y1 - 5)), FONT, 0.4, colour, 1)

    # --- top-left status panel ---
    _panel(img, 0, 0, 250, 96)
    st = state.decision.fsm_state
    cv2.putText(img, st, (10, 24), FONT, fs + 0.15, STATE_COLOUR.get(st, WHITE), 2)
    cv2.putText(
        img, f"CMD SPD {state.decision.target_speed:5.1f} km/h",
        (10, 48), FONT, fs, WHITE, 1,
    )
    cv2.putText(
        img, f"ZONE {state.signals.speed_limit_kmh} km/h",
        (10, 68), FONT, fs, WHITE, 1,
    )
    wx = f"{state.weather.label.upper()}  vis {state.weather.visibility:.2f}"
    cv2.putText(img, wx, (10, 88), FONT, fs, AMBER, 1)

    # --- top-right traffic light ---
    if state.signals.light_state not in ("none", ""):
        _panel(img, w - 150, 0, w, 36)
        col = {"red": RED, "amber": AMBER, "green": GREEN}.get(
            state.signals.light_state, GREY
        )
        cv2.circle(img, (w - 130, 18), 10, col, -1)
        cv2.putText(
            img, state.signals.light_state.upper(), (w - 112, 24), FONT, fs, col, 2
        )

    # --- steering wheel + indicators, bottom centre ---
    cx, cy = w // 2, h - 70
    _panel(img, cx - 110, cy - 55, cx + 110, h)
    _steering_wheel(img, cx, cy, 38, state.control.steer_deg)
    cv2.putText(
        img, f"{state.control.steer_deg:+.1f}deg", (cx - 38, cy + 60),
        FONT, fs, WHITE, 2,
    )
    _indicators(img, cx, cy, state.decision.indicator, blink_on=(state.idx // 5) % 2 == 0)

    # --- throttle / brake bars, bottom right ---
    bx = w - 60
    for i, (name, val, col) in enumerate(
        [("THR", state.control.throttle, GREEN), ("BRK", state.control.brake, RED)]
    ):
        y = h - 90 + i * 34
        cv2.putText(img, name, (bx - 42, y + 13), FONT, 0.4, WHITE, 1)
        cv2.rectangle(img, (bx, y), (bx + 40, y + 16), GREY, 1)
        cv2.rectangle(
            img, (bx, y), (bx + int(40 * float(np.clip(val, 0, 1))), y + 16), col, -1
        )

    # --- scrolling event log, bottom left ---
    # Right edge is capped at LOG_PANEL_MAX_X2 on wide frames (unchanged from
    # before), but narrows so it always clears the steering panel's left edge
    # by LOG_STEER_GAP — the two would otherwise overlap below ~900px wide.
    lines = state.decision.log[-hud["log_lines"]:]
    if lines:
        steer_x1 = cx - 110
        log_x2 = min(LOG_PANEL_MAX_X2, max(0, steer_x1 - LOG_STEER_GAP))
        log_fs = fs - 0.05
        max_text_w = max(0, log_x2 - 16)
        _panel(img, 0, h - 22 * len(lines) - 8, log_x2, h)
        for i, line in enumerate(lines):
            y = h - 8 - 22 * (len(lines) - 1 - i)
            text = f"> {line}"
            while text and cv2.getTextSize(text, FONT, log_fs, 1)[0][0] > max_text_w:
                text = text[:-1]
            cv2.putText(img, text, (8, y), FONT, log_fs, AMBER, 1)

    return img


if __name__ == "__main__":
    import argparse

    import yaml

    from src.types import Track
    from src.video_io import VideoReader

    ap = argparse.ArgumentParser(description="Block 1 demo: HUD with fake values.")
    ap.add_argument("--video", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))

    for idx, t, frame in VideoReader(a.video):
        s = FrameState(idx=idx, t=t, raw=frame, img=frame)
        s.control.steer_deg = 20 * np.sin(idx / 25.0)   # fake sweep
        s.control.throttle = 0.5 + 0.5 * np.sin(idx / 30.0)
        s.decision.target_speed = 45.0
        s.decision.indicator = ["off", "left", "right"][(idx // 40) % 3]
        s.decision.log = ["HUD DEMO — VALUES ARE FAKE", f"frame {idx}"]
        s.objects = [Track(id=1, cls="car", bbox=(260, 160, 380, 240), conf=0.9,
                           dist_m=18.0, ttc_s=3.2)]
        cv2.imshow("hud", render(s, cfg))
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
