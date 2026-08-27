"""The one place where commands actually drive something.

The video pipeline is open-loop: our steering angle cannot move a recorded
camera. So the commands are also fed to a kinematic bicycle model driving
in the bird's-eye world we reconstructed, and drawn in a separate window.
That window is where "the car avoided the obstacle" becomes a real claim
rather than an overlay caption.

The bicycle model is the standard low-speed vehicle approximation:
    x'   = v sin(yaw)
    y'   = v cos(yaw)
    yaw' = v tan(delta) / L
which traces a circle of radius L / tan(delta) for a constant steering
angle delta. Tyre slip is ignored, which is valid at the speeds here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from src.types import FrameState


@dataclass
class EgoState:
    x_m: float = 0.0          # + = right of the starting centreline
    y_m: float = 0.0          # + = forward
    yaw_rad: float = 0.0      # 0 = straight ahead, + = turned right
    v_kmh: float = 0.0


def bicycle_step(
    ego: EgoState,
    steer_deg: float,
    throttle: float,
    brake: float,
    dt: float,
    cfg: dict,
) -> EgoState:
    """Advance the vehicle by dt seconds. Returns a new EgoState."""
    c = cfg["control"]
    L = cfg["camera"]["wheelbase_m"]

    delta = np.radians(np.clip(steer_deg, -c["max_steer_deg"], c["max_steer_deg"]))
    accel = throttle * c["accel_max"] - brake * c["decel_max"]      # m/s^2

    v = max(0.0, ego.v_kmh / 3.6 + accel * dt)                      # m/s
    yaw = ego.yaw_rad + (v * np.tan(delta) / L) * dt

    return replace(
        ego,
        x_m=ego.x_m + v * np.sin(yaw) * dt,
        y_m=ego.y_m + v * np.cos(yaw) * dt,
        yaw_rad=float(yaw),
        v_kmh=v * 3.6,
    )


def render_sim(state: FrameState, ego: EgoState, cfg: dict) -> np.ndarray:
    """Top-down view: road corridor, detected objects, and the ego car."""
    b = cfg["bev"]
    mpp = b["m_per_px"]
    gh, gw = int(b["range_m"] / mpp), int(b["width_m"] / mpp)
    canvas = np.full((gh, gw, 3), 28, np.uint8)
    ox, oy = gw // 2, gh - 1

    # Distance rings every 10 m.
    for d in range(10, int(b["range_m"]) + 1, 10):
        y = int(oy - d / mpp)
        cv2.line(canvas, (0, y), (gw, y), (55, 55, 55), 1)
        cv2.putText(canvas, f"{d}m", (4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (90, 90, 90), 1)

    # Ego lane corridor.
    for sign in (-1, 1):
        x = int(ox + sign * b["corridor_m"] / mpp)
        cv2.line(canvas, (x, 0), (x, gh), (70, 70, 70), 1)

    # Detected objects.
    for t in state.objects:
        if not np.isfinite(t.dist_m) or not np.isfinite(t.lateral_m):
            continue
        px, py = int(ox + t.lateral_m / mpp), int(oy - t.dist_m / mpp)
        if not (0 <= px < gw and 0 <= py < gh):
            continue
        colour = (0, 0, 255) if t.in_path else (0, 170, 220)
        w, h = int(1.8 / mpp), int(4.2 / mpp)
        cv2.rectangle(canvas, (px - w // 2, py - h // 2), (px + w // 2, py + h // 2),
                      colour, 2)
        cv2.putText(canvas, f"{t.cls[:3]}{t.id}", (px - w // 2, py - h // 2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)

    # Ego vehicle, drawn at its simulated lateral offset and heading.
    ex, ey = int(ox + ego.x_m / mpp), oy - int(2.0 / mpp)
    w, h = 1.8 / mpp, 4.2 / mpp
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    ca, sa = np.cos(-ego.yaw_rad), np.sin(-ego.yaw_rad)
    rot = corners @ np.array([[ca, -sa], [sa, ca]])
    pts = (rot + [ex, ey]).astype(np.int32)
    cv2.fillPoly(canvas, [pts], (0, 220, 0))
    cv2.line(canvas, (ex, ey),
             (int(ex + 40 * np.sin(ego.yaw_rad)), int(ey - 40 * np.cos(ego.yaw_rad))),
             (255, 255, 255), 2)

    # Readout.
    cv2.rectangle(canvas, (0, 0), (gw, 54), (0, 0, 0), -1)
    cv2.putText(canvas, f"SIM  v={ego.v_kmh:5.1f} km/h", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas,
                f"steer={state.control.steer_deg:+.1f}  x={ego.x_m:+.2f}m  "
                f"yaw={np.degrees(ego.yaw_rad):+.1f}",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
    return canvas


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(open("config.yaml"))
    print("Block 9 demo: bicycle model, constant 15 deg right steer at 40 km/h\n")
    e = EgoState(v_kmh=40.0)
    for i in range(60):
        e = bicycle_step(e, 15.0, 0.0, 0.0, 0.1, cfg)
        if i % 10 == 0:
            print(f"  t={i * 0.1:4.1f}s  x={e.x_m:+7.2f}m  y={e.y_m:7.2f}m  "
                  f"yaw={np.degrees(e.yaw_rad):+7.1f}deg")
    R = cfg["camera"]["wheelbase_m"] / np.tan(np.radians(15.0))
    print(f"\nexpected turn radius = {R:.2f} m")
