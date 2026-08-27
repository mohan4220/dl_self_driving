"""The frame loop: perception, decision, control, overlay.

Stage order matters. Weather runs first because everything downstream uses
the enhanced frame and the visibility score it produces. Segmentation runs
before detection only so the HUD can draw the road under the boxes.

The render path never calls cv2.imshow, so this runs headless on Colab.
The --sim window is strictly a local convenience.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from src.bev import annotate_distances, build_grid, build_homography
from src.control import PID, longitudinal, lookahead_for, pure_pursuit
from src.detect import Detector, update_ttc
from src.hud import render
from src.planner import Planner, degrade_level
from src.segment import Segmenter, fit_lanes, freespace_offset
from src.sim import EgoState, bicycle_step, render_sim
from src.signals import SignReader, read_signals
from src.types import FrameState, SignalInfo, WeatherInfo
from src.video_io import ControlLogWriter, VideoReader, VideoWriter
from src.weather import WeatherClassifier, enhance, visibility_score


class Pipeline:
    def __init__(self, cfg: dict, use_clip: bool = True):
        self.cfg = cfg
        m, w = cfg["model"], cfg["weather"]

        self.seg = Segmenter(m["segmenter"], m["onnx_threads"])
        self.det = Detector(m["detector"], cfg["detect"]["imgsz"], cfg["detect"]["classes"])
        self.signs = SignReader(cfg["signals"]["min_sign_px"])
        self.planner = Planner(cfg)
        self.H = build_homography(cfg)
        self.pid = PID(**cfg["control"]["pid"])

        self.clf = (
            WeatherClassifier(m["clip"], w["labels"], w["prompts"]) if use_clip else None
        )

        self._weather = WeatherInfo()
        self._signals = SignalInfo()
        self._prev_dist: dict[int, float] = {}
        self._prev_t = 0.0
        self._prev_steer = 0.0
        self.ego = EgoState()

    def process(self, idx: int, t: float, frame: np.ndarray) -> FrameState:
        cfg = self.cfg
        dt = max(1e-3, t - self._prev_t)
        self._prev_t = t

        # --- weather: classify rarely, score every frame ---
        if self.clf is not None and idx % cfg["weather"]["clip_every"] == 0:
            self._weather = self.clf.classify(frame)
        else:
            self._weather.visibility = visibility_score(frame)

        img = enhance(frame, self._weather)
        state = FrameState(idx=idx, t=t, raw=frame, img=img)
        state.weather = self._weather

        # --- road and lanes ---
        da, ll = self.seg.infer(img)
        state.drivable, state.lane_mask = da, ll
        state.lanes = fit_lanes(ll, da, cfg)

        # --- objects ---
        low_vis = self._weather.visibility < cfg["weather"]["visibility_lowvis_below"]
        conf = cfg["detect"]["conf_lowvis" if low_vis else "conf"]
        state.objects = self.det.track(img, conf)
        annotate_distances(state.objects, self.H, (img.shape[1], img.shape[0]), cfg)
        self._prev_dist = update_ttc(state.objects, self._prev_dist, dt)

        # --- signals and signs ---
        self._signals = read_signals(img, state.objects, self.signs, self._signals, cfg)
        state.signals = self._signals

        # --- world model ---
        state.bev = build_grid(state, cfg)

        # --- decide ---
        state.decision = self.planner.step(state)

        # --- steer, with the degradation ladder deciding what we aim at ---
        level = degrade_level(state, cfg)
        if level == 0:
            offset, heading = state.lanes.offset_m, state.lanes.heading_err_rad
        elif level == 1:
            offset = freespace_offset(da, cfg) or 0.0
            heading = 0.0
            state.decision.log.append("LANE LOST - STEERING TO FREE SPACE")
        else:
            offset, heading = 0.0, 0.0
            state.decision.log.append("PERCEPTION DEGRADED - HOLDING STEER")

        if level == 2:
            steer = self._prev_steer * 0.85          # decay toward straight
        else:
            steer = pure_pursuit(
                offset, heading,
                lookahead_for(state.decision.target_speed, cfg),
                cfg["camera"]["wheelbase_m"],
                cfg["control"]["max_steer_deg"],
            )
        self._prev_steer = steer
        state.control.steer_deg = steer

        # --- pedals, tracked against the simulated speed ---
        thr, brk = longitudinal(state.decision.target_speed, self.ego.v_kmh, self.pid, dt)
        state.control.throttle, state.control.brake = thr, brk
        self.ego = bicycle_step(self.ego, steer, thr, brk, dt, cfg)
        return state

    def run(
        self,
        video_path: str,
        out_path: str,
        log_path: str,
        sim: bool = False,
        max_frames: int | None = None,
    ) -> dict:
        cfg = self.cfg
        reader = VideoReader(video_path, stride=cfg["video"]["stride"], max_frames=max_frames)
        size = (reader.width, reader.height)

        frames, total_ms = 0, 0.0
        states: dict[str, int] = {}

        with VideoWriter(out_path, cfg["video"]["out_fps"], size) as vw, \
             ControlLogWriter(log_path) as lw:
            for idx, t, frame in reader:
                t0 = time.perf_counter()
                state = self.process(idx, t, frame)
                total_ms += (time.perf_counter() - t0) * 1000

                vw.write(render(state, cfg))
                lw.write(state)
                frames += 1
                states[state.decision.fsm_state] = (
                    states.get(state.decision.fsm_state, 0) + 1
                )

                if sim:
                    cv2.imshow("annotated", render(state, cfg))
                    cv2.imshow("simulator", render_sim(state, self.ego, cfg))
                    if cv2.waitKey(1) == 27:
                        break

        if sim:
            cv2.destroyAllWindows()

        return {
            "frames": frames,
            "mean_ms": total_ms / max(1, frames),
            "fps": 1000.0 / max(1e-9, total_ms / max(1, frames)),
            "states": states,
            "out": out_path,
            "log": log_path,
        }
