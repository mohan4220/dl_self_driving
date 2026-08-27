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

        dev = m.get("device", "auto")
        self.seg = Segmenter(m["segmenter"], m["onnx_threads"])
        self.det = Detector(
            m["detector"], cfg["detect"]["imgsz"], cfg["detect"]["classes"], device=dev
        )
        self.signs = SignReader(cfg["signals"]["min_sign_px"])
        self.planner = Planner(cfg)
        self.H = build_homography(cfg)
        self.pid = PID(**cfg["control"]["pid"])

        self.clf = (
            WeatherClassifier(m["clip"], w["labels"], w["prompts"], device=dev)
            if use_clip
            else None
        )

        self._weather = WeatherInfo()
        self._signals = SignalInfo()
        self._prev_dist: dict[int, float] = {}
        self._prev_t: float | None = None
        self._prev_steer = 0.0
        self._sim = False           # set by run(): only build_grid when --sim needs it
        self.ego = EgoState()

    def process(self, idx: int, t: float, frame: np.ndarray) -> FrameState:
        cfg = self.cfg
        if self._prev_t is None:
            dt = 1.0 / cfg["video"]["out_fps"]     # nominal interval: no prior frame yet
        else:
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

        # --- world model --- (only needed for the --sim top-down window;
        # render_sim rasterises straight from state.objects, nothing else
        # reads state.bev)
        if self._sim:
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

        # A CHANGING manoeuvre biases the pure-pursuit aim point sideways by
        # up to a full lane width, signed by direction. Convention: offset_m
        # > 0 is ego RIGHT of lane centre, and a positive offset yields
        # negative (left) steering -- so moving left means biasing offset
        # positive (makes pure pursuit think we're further right than we
        # are, so it steers left to "correct").
        if level != 2 and state.decision.lane_change_progress > 0:
            sign = 1.0 if state.decision.indicator == "left" else -1.0
            offset += sign * cfg["camera"]["lane_width_m"] * state.decision.lane_change_progress

        if level == 2:
            steer = self._prev_steer * 0.85          # decay toward straight
        else:
            steer = pure_pursuit(
                offset, heading,
                lookahead_for(self.ego.v_kmh, cfg),
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

    def describe(self) -> dict:
        """What each model actually ended up running on.

        Reported at startup because a GPU run that silently fell back to CPU
        looks exactly like a slow GPU run.
        """
        seg_dev = "cuda" if "CUDA" in self.seg.provider else "cpu"
        return {
            "segmenter": f"{self.cfg['model']['segmenter']}  [{self.seg.provider}]",
            "segmenter_device": seg_dev,
            "detector": f"{self.cfg['model']['detector']}  [{self.det.device}]",
            "detector_device": self.det.device,
            "weather": (f"{self.cfg['model']['clip']}  [{self.clf.device}]"
                        if self.clf else "disabled (--no-clip)"),
            "onnx_threads": self.cfg["model"]["onnx_threads"],
        }

    def run(
        self,
        video_path: str,
        out_path: str,
        log_path: str,
        sim: bool = False,
        max_frames: int | None = None,
    ) -> dict:
        cfg = self.cfg
        self._sim = sim
        reader = VideoReader(video_path, stride=cfg["video"]["stride"], max_frames=max_frames)
        size = (reader.width, reader.height)

        frames, total_ms = 0, 0.0
        states: dict[str, int] = {}
        total_expected = reader.expected_frames
        if max_frames is not None:
            total_expected = (max_frames if total_expected is None
                              else min(total_expected, max_frames))
        every = max(1, cfg["video"].get("log_every", 10))
        t_start = time.perf_counter()

        howmany = (f"{total_expected} frames" if total_expected
                   else "frames (total unknown: container reports no usable count)")
        print(f"processing {howmany} "
              f"({reader.width}x{reader.height} @ {reader.fps:.1f}fps, "
              f"stride {cfg['video']['stride']})", flush=True)

        with VideoWriter(out_path, cfg["video"]["out_fps"], size) as vw, \
             ControlLogWriter(log_path) as lw:
            for idx, t, frame in reader:
                t0 = time.perf_counter()
                state = self.process(idx, t, frame)
                total_ms += (time.perf_counter() - t0) * 1000

                overlay = render(state, cfg)
                vw.write(overlay)
                lw.write(state)
                frames += 1
                states[state.decision.fsm_state] = (
                    states.get(state.decision.fsm_state, 0) + 1
                )

                if frames % every == 0 or frames == 1:
                    elapsed = time.perf_counter() - t_start
                    rate = frames / max(1e-9, elapsed)
                    ev = state.decision.log[0] if state.decision.log else ""
                    if total_expected:
                        eta = max(0.0, (total_expected - frames)) / max(1e-9, rate)
                        head = (f"  [{frames:5d}/{total_expected:<5d} "
                                f"{100.0 * frames / total_expected:5.1f}%] "
                                f"t={state.t:6.2f}s  {rate:4.2f} fps  eta {eta:5.0f}s  | ")
                    else:
                        head = (f"  [{frames:5d}] t={state.t:6.2f}s  "
                                f"{rate:4.2f} fps  elapsed {elapsed:5.0f}s  | ")
                    print(
                        head +
                        f"{state.decision.fsm_state:<16} "
                        f"steer {state.control.steer_deg:+6.1f}deg  "
                        f"spd {state.decision.target_speed:5.1f}  "
                        f"obj {len(state.objects):2d}  "
                        f"lane {'ok ' if state.lanes.valid else 'LOST'}  "
                        f"{state.weather.label:<5} | {ev[:38]}",
                        flush=True,
                    )

                if sim:
                    cv2.imshow("annotated", overlay)
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
