"""Turning perception into two numbers: steering angle and pedal.

Lateral control is pure pursuit, the standard geometric path tracker: aim
at a point a fixed distance ahead on the desired path and compute the arc
that reaches it. Longitudinal control is a PID on speed error.

Sign convention throughout the project:
    offset_m  > 0  ->  ego is RIGHT of the lane centre
    steer_deg > 0  ->  steer RIGHT
so a positive offset must produce a negative steering angle.
"""
from __future__ import annotations

import numpy as np


def lookahead_for(speed_kmh: float, cfg: dict) -> float:
    """Look further ahead the faster we go, or the steering oscillates."""
    c = cfg["control"]
    return float(c["lookahead_min_m"] + c["lookahead_k"] * (speed_kmh / 3.6))


def pure_pursuit(
    offset_m: float,
    heading_err_rad: float,
    lookahead_m: float,
    wheelbase_m: float,
    max_steer_deg: float,
) -> float:
    """Steering angle in degrees to rejoin the lane centreline."""
    ld = max(float(lookahead_m), 1e-3)
    # Angle to the lookahead point: heading error plus the geometric term
    # from being laterally displaced.
    alpha = float(heading_err_rad) - np.arctan2(float(offset_m), ld)
    delta = np.arctan2(2.0 * float(wheelbase_m) * np.sin(alpha), ld)
    return float(np.clip(np.degrees(delta), -max_steer_deg, max_steer_deg))


class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        out_min: float = -1.0,
        out_max: float = 1.0,
    ):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        candidate_i = self.integral + error * dt
        raw = self.kp * error + self.ki * candidate_i + self.kd * derivative

        # Conditional integration: only accumulate when doing so would not
        # push us further into saturation. Without this the integral grows
        # unboundedly during a long climb and overshoots badly afterwards.
        if self.out_min < raw < self.out_max:
            self.integral = candidate_i
        else:
            raw = self.kp * error + self.ki * self.integral + self.kd * derivative

        return float(np.clip(raw, self.out_min, self.out_max))


def longitudinal(
    target_kmh: float, current_kmh: float, pid: PID, dt: float
) -> tuple[float, float]:
    """Split a signed PID output into throttle and brake.

    Only one pedal is ever pressed, which is both physically sensible and
    much easier to read on the HUD.

    The error is converted to m/s before it reaches the PID. In km/h, any
    error above ~1.1 km/h saturated the proportional term alone (kp=0.9),
    pinning throttle at 1.000 across the whole ramp and starving the
    integral term (conditional integration only accumulates while
    unsaturated). m/s error is ~3.6x smaller for the same physical gap, so
    `control.pid` in config.yaml is retuned accordingly.
    """
    error_ms = (target_kmh - current_kmh) / 3.6
    u = pid.step(error_ms, dt)
    if u >= 0:
        return float(np.clip(u, 0.0, 1.0)), 0.0
    return 0.0, float(np.clip(-u, 0.0, 1.0))


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(open("config.yaml"))
    c = cfg["control"]

    print("Block 8 demo: pure pursuit response at 50 km/h\n")
    ld = lookahead_for(50.0, cfg)
    print(f"lookahead = {ld:.1f} m,  wheelbase = {cfg['camera']['wheelbase_m']} m\n")
    print(f"{'offset_m':>9} {'heading_deg':>12} {'steer_deg':>10}")
    for off in (-2.0, -1.0, -0.3, 0.0, 0.3, 1.0, 2.0):
        for hdg in (-5.0, 0.0, 5.0):
            s = pure_pursuit(off, np.radians(hdg), ld,
                             cfg["camera"]["wheelbase_m"], c["max_steer_deg"])
            print(f"{off:9.1f} {hdg:12.1f} {s:10.2f}")

    print("\nPID step response, target 50 km/h from standstill:")
    pid = PID(**c["pid"])
    v = 0.0
    for i in range(40):
        thr, brk = longitudinal(50.0, v, pid, 0.1)
        v += (thr * c["accel_max"] - brk * c["decel_max"]) * 0.1 * 3.6
        if i % 5 == 0:
            print(f"  t={i * 0.1:4.1f}s  v={v:6.2f} km/h  thr={thr:.2f} brk={brk:.2f}")
