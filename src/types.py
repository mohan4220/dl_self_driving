"""The single dataclass that flows through the whole pipeline.

Every pipeline stage is a function FrameState -> FrameState. Keeping all
per-frame state in one object means any stage can be run, printed, and
tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WeatherInfo:
    label: str = "clear"          # clear | rain | fog | night | snow
    confidence: float = 0.0
    visibility: float = 1.0       # 0.0 = blind, 1.0 = perfect


@dataclass
class Track:
    id: int
    cls: str
    bbox: tuple[int, int, int, int]      # x1, y1, x2, y2 in frame pixels
    conf: float
    dist_m: float = float("inf")         # forward distance, metres
    lateral_m: float = float("nan")      # + = right of ego centreline, metres
    ttc_s: float = float("inf")          # time to collision
    in_path: bool = False                # inside the ego lane corridor


@dataclass
class Sign:
    kind: str                            # speed_limit | stop | other
    value: int | None                    # km/h when kind == "speed_limit"
    bbox: tuple[int, int, int, int]
    track_id: int


@dataclass
class LaneInfo:
    # Polynomials in image space: x = f(y), coefficients as np.polyfit returns.
    left_fit: np.ndarray | None = None
    right_fit: np.ndarray | None = None
    offset_m: float = 0.0                # + = ego is right of lane centre
    heading_err_rad: float = 0.0         # + = lane heads right of ego
    curvature_m: float = float("inf")    # radius; inf = straight
    valid: bool = False


@dataclass
class SignalInfo:
    light_state: str = "none"            # none | red | amber | green | unknown
    signs: list[Sign] = field(default_factory=list)
    speed_limit_kmh: int = 50


@dataclass
class BEVGrid:
    grid: np.ndarray | None = None       # uint8 occupancy, 0 = free
    m_per_px: float = 0.1
    origin_px: tuple[int, int] = (0, 0)  # pixel of the ego rear axle


@dataclass
class Decision:
    fsm_state: str = "LANE_KEEP"
    target_speed: float = 0.0            # km/h, commanded
    indicator: str = "off"               # off | left | right
    log: list[str] = field(default_factory=list)


@dataclass
class Control:
    steer_deg: float = 0.0               # + = steer right
    throttle: float = 0.0                # 0..1
    brake: float = 0.0                   # 0..1


@dataclass
class FrameState:
    idx: int
    t: float                             # seconds from clip start
    raw: np.ndarray                      # original BGR frame
    img: np.ndarray                      # weather-enhanced BGR frame
    weather: WeatherInfo = field(default_factory=WeatherInfo)
    objects: list[Track] = field(default_factory=list)
    lanes: LaneInfo = field(default_factory=LaneInfo)
    drivable: np.ndarray | None = None   # binary mask, 360x640
    lane_mask: np.ndarray | None = None  # binary mask, 360x640
    signals: SignalInfo = field(default_factory=SignalInfo)
    bev: BEVGrid = field(default_factory=BEVGrid)
    decision: Decision = field(default_factory=Decision)
    control: Control = field(default_factory=Control)
