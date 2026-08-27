"""The decision layer: what should the car DO?

A finite state machine, deliberately, rather than a learned policy. Three
reasons. It needs no training data. Every decision is explainable, which
matters when the whole deliverable is an overlay explaining decisions. And
it degrades predictably when perception fails, which is the core claim of
this project.
"""
from __future__ import annotations

import numpy as np

from src.detect import VEHICLE, VULNERABLE
from src.types import Decision, FrameState
from src.weather import speed_cap_kmh

STATES = (
    "LANE_KEEP", "FOLLOW", "PREP_CHANGE_L", "PREP_CHANGE_R", "CHANGING",
    "STOP_SIGNAL", "YIELD_PEDESTRIAN", "EMERGENCY_BRAKE",
)


def degrade_level(state: FrameState, cfg: dict) -> int:
    """0 = lane lines usable, 1 = free-space fallback, 2 = blind."""
    if state.lanes.valid:
        return 0
    if state.drivable is not None and state.drivable.mean() > 0.02:
        return 1
    return 2


class Planner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._follow_frames = 0
        self._prep_frames = 0
        self._change_frames = 0
        self._manoeuvre: str | None = None      # "left" | "right" | None
        self._prev_state = "LANE_KEEP"

    # --- helpers -------------------------------------------------------

    def _lead_vehicle(self, state: FrameState):
        leads = [
            t for t in state.objects
            if t.in_path and t.cls in VEHICLE and np.isfinite(t.dist_m)
        ]
        return min(leads, key=lambda t: t.dist_m) if leads else None

    def _target_lane_clear(self, state: FrameState, side: str) -> bool:
        """Is there a safe gap in the lane we want to move into?"""
        p = self.cfg["planner"]
        corridor = self.cfg["bev"]["corridor_m"]
        lo, hi = (-3 * corridor, -corridor) if side == "left" else (corridor, 3 * corridor)
        gap_m = p["lane_change_gap_s"] * (state.decision.target_speed / 3.6 or 10.0)
        for t in state.objects:
            if not np.isfinite(t.lateral_m) or not np.isfinite(t.dist_m):
                continue
            if lo <= t.lateral_m <= hi and t.dist_m < max(gap_m, 15.0):
                return False
        return True

    def _speed_caps(self, state: FrameState, level: int) -> list[tuple[float, str]]:
        """Every cap, paired with the reason, so the HUD can say why."""
        p, d = self.cfg["planner"], self.cfg["degrade"]
        caps: list[tuple[float, str]] = [
            (float(state.signals.speed_limit_kmh), f"ZONE {state.signals.speed_limit_kmh}")
        ]

        wcap = speed_cap_kmh(state.weather, self.cfg)
        if wcap < 999:
            caps.append((wcap, f"{state.weather.label.upper()} CAP {wcap:.0f}"))

        if np.isfinite(state.lanes.curvature_m) and state.lanes.curvature_m > 1.0:
            v = np.sqrt(p["lat_accel_max"] * state.lanes.curvature_m) * 3.6
            caps.append((float(v), f"CURVE CAP {v:.0f}"))

        if level == 1:
            caps.append((float(d["freespace_speed_cap_kmh"]), "LANE LOST - FREESPACE MODE"))
        elif level == 2:
            caps.append((float(d["blind_speed_cap_kmh"]), "PERCEPTION DEGRADED"))

        lead = self._lead_vehicle(state)
        if lead is not None:
            gap = p["follow_time_gap_s"]
            if state.weather.visibility < self.cfg["weather"]["visibility_lowvis_below"]:
                gap *= d["lowvis_gap_multiplier"]
            caps.append((float(lead.dist_m / gap * 3.6), f"FOLLOW {lead.dist_m:.0f}m"))

        return caps

    # --- the state machine ---------------------------------------------

    def step(self, state: FrameState) -> Decision:
        p = self.cfg["planner"]
        d = Decision()
        level = degrade_level(state, self.cfg)

        caps = self._speed_caps(state, level)
        target, reason = min(caps, key=lambda c: c[0])
        d.target_speed = max(0.0, target)

        lead = self._lead_vehicle(state)
        hazards = [t for t in state.objects if t.in_path and np.isfinite(t.ttc_s)]
        soonest = min((t.ttc_s for t in hazards), default=float("inf"))
        ped = [
            t for t in state.objects
            if t.in_path and t.cls in VULNERABLE and t.dist_m < p["yield_ped_dist_m"]
        ]

        # 1. Imminent collision.
        if soonest < p["ebrake_ttc_s"]:
            d.fsm_state, d.target_speed, d.indicator = "EMERGENCY_BRAKE", 0.0, "off"
            d.log.append(f"EMERGENCY BRAKE - TTC {soonest:.1f}s")
            self._reset_manoeuvre()

        # 2. Vulnerable road user ahead.
        elif ped:
            d.fsm_state, d.indicator = "YIELD_PEDESTRIAN", "off"
            d.target_speed = 0.0
            d.log.append(f"YIELDING - {ped[0].cls.upper()} AT {ped[0].dist_m:.0f}m")
            self._reset_manoeuvre()

        # 3. Signal says stop.
        elif state.signals.light_state in ("red", "amber", "unknown"):
            d.fsm_state, d.target_speed, d.indicator = "STOP_SIGNAL", 0.0, "off"
            note = ("SIGNAL UNCERTAIN - DECELERATING"
                    if state.signals.light_state == "unknown"
                    else f"{state.signals.light_state.upper()} LIGHT - STOPPING")
            d.log.append(note)
            self._reset_manoeuvre()

        # 4/5. A lane change already in progress.
        elif self._manoeuvre is not None:
            d.indicator = self._manoeuvre
            if self._prep_frames < p["indicator_lead_frames"]:
                self._prep_frames += 1
                d.fsm_state = f"PREP_CHANGE_{self._manoeuvre[0].upper()}"
                d.log.append(f"INDICATOR {self._manoeuvre.upper()} ON - PREPARING")
            elif self._change_frames < p["change_frames"]:
                self._change_frames += 1
                d.fsm_state = "CHANGING"
                d.log.append(f"CHANGING LANE {self._manoeuvre.upper()}")
            else:
                d.fsm_state, d.indicator = "LANE_KEEP", "off"
                d.log.append("LANE CHANGE COMPLETE - INDICATOR OFF")
                self._reset_manoeuvre()

        # 6. Following, and possibly deciding to overtake.
        elif lead is not None and lead.dist_m < (d.target_speed / 3.6) * p["follow_time_gap_s"] + 10.0:
            self._follow_frames += 1
            d.fsm_state = "FOLLOW"
            d.log.append(f"FOLLOWING {lead.cls.upper()} AT {lead.dist_m:.0f}m")
            if (
                self._follow_frames >= p["follow_frames_before_change"]
                and level == 0
                and self._target_lane_clear(state, p["overtake_side"])
            ):
                self._manoeuvre = p["overtake_side"]
                self._prep_frames = 1
                d.fsm_state = f"PREP_CHANGE_{self._manoeuvre[0].upper()}"
                d.indicator = self._manoeuvre
                d.log.append(f"OVERTAKE DECIDED - INDICATOR {self._manoeuvre.upper()} ON")

        # 7. Nothing to react to.
        else:
            self._follow_frames = 0
            d.fsm_state, d.indicator = "LANE_KEEP", "off"

        if reason and d.fsm_state not in ("EMERGENCY_BRAKE", "STOP_SIGNAL", "YIELD_PEDESTRIAN"):
            d.log.append(f"SPEED {reason} - MATCHING {d.target_speed:.0f}")

        self._prev_state = d.fsm_state
        return d

    def _reset_manoeuvre(self) -> None:
        self._manoeuvre = None
        self._prep_frames = 0
        self._change_frames = 0
        self._follow_frames = 0
