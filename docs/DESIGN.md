# All-Weather Self-Driving Car Simulation — Design

**Date:** 2026-08-27
**Author:** Sreedhar Teegala
**Status:** Approved for implementation
**Timeline:** 4 days

---

## 1. Goal

Take a dashcam video as input. Produce an annotated output video showing what a
self-driving controller would command: steering angle, target speed, indicator
state, and a running log of decisions ("SPEED ZONE 30 DETECTED — MATCHING",
"LANE CHANGE RIGHT — INDICATOR ON"). Optionally show a second window with a
top-down simulation in which a virtual car actually executes those commands.

The system must degrade gracefully in rain, fog, night, and snow rather than
failing silently. That is the "all-weather" claim.

## 2. Constraints

| Constraint | Value |
|---|---|
| Hardware | Intel i7-7600U, 2 physical cores, no GPU. Colab GPU for final render. |
| Python | 3.12, venv at `.venv/` |
| Training | **None.** All models pretrained and frozen. |
| Input | Pre-recorded dashcam video (Western roads: US/EU) |
| Timeline | 4 days |
| Deliverable | Annotated MP4 + JSONL control log + report figures |

## 3. Non-Goals

Explicitly out of scope, and why:

- **Closed-loop control on the video.** Video frames are fixed; a steering command
  cannot move the camera. The video path is open-loop and advisory. Closed-loop
  behaviour is demonstrated only in the BEV simulator.
- **Reinforcement learning.** RL requires a reactive environment. A recorded video
  is not one. Rejected for the main pipeline.
- **End-to-end behaviour cloning (PilotNet).** Would require training, and emits
  only a steering scalar — no obstacle, signal, or sign reasoning, so the required
  decision log would be impossible. Discussed in the report as a rejected alternative.
- **Measured ego speed.** Not recoverable from a monocular dashcam without
  calibration and visual odometry. Speed shown in the HUD is the *commanded*
  speed, labelled `CMD SPD`.
- **Quantitative detection accuracy.** Input clips are unlabelled. Evaluation is
  qualitative plus controlled ablations (with/without enhancement, FP32 vs INT8).

## 4. Model Inventory

All pretrained. Latencies measured on the target CPU at 800 MHz (powersave).

| Role | Model | Size | Latency | Source |
|---|---|---|---|---|
| Drivable area + lane lines | TwinLiteNet, INT8-quantized ONNX | 558 KB | **144 ms** | `harrylal/TwinLiteNet-onnxruntime` |
| Objects: vehicles, people, animals, traffic lights, stop signs | YOLO11n (COCO) | 5.4 MB | **82 ms** @640 | ultralytics |
| Multi-object tracking | ByteTrack | — | negligible | bundled with ultralytics |
| Weather classification | CLIP ViT-B/32, zero-shot | ~600 MB | ~300 ms, **every 15th frame** | `openai/clip-vit-base-patch32` |
| Speed-limit digit reading | RapidOCR (PP-OCR) | ~80 MB | ~300 ms, **once per sign track** | `rapidocr` 3.9.2 |

**Per-frame perception budget: ~226 ms ≈ 4.4 FPS.** A 60 s clip sampled at 15 fps
renders in ~3.5 minutes.

### 4.1 INT8 Quantization (engineering contribution)

TwinLiteNet FP32 measured 774–1350 ms/frame — unusable. ONNX Runtime dynamic
quantization to UInt8 reduced this to 144 ms, a **5.4× speedup**, with model size
falling 1.80 MB → 0.558 MB.

Fidelity against FP32 across 7 BDD100K frames:

| Output | mean IoU vs FP32 |
|---|---|
| Drivable area | **0.985** |
| Lane lines | **0.941** |

Input resolution is fixed at 640×360; the encoder contains a hardcoded `Reshape`
to `{1,32,45,80}`, so resolution scaling is not available as a speed lever.

## 5. Architecture

### 5.1 Spine

One dataclass flows through the pipeline. Every module is a pure function
`FrameState -> FrameState`, independently testable and printable.

```python
@dataclass
class FrameState:
    idx: int
    t: float
    raw: np.ndarray                 # original BGR frame
    img: np.ndarray                 # weather-enhanced BGR frame
    weather:  WeatherInfo           # label, confidence, visibility 0..1
    objects:  list[Track]           # id, cls, bbox, conf, dist_m, ttc_s, in_path
    lanes:    LaneInfo              # left_fit, right_fit, offset_m, curvature_m, valid
    drivable: np.ndarray | None     # binary mask, 360x640
    signals:  SignalInfo            # light_state, signs[], speed_limit_kmh
    bev:      BEVGrid               # ego-centric metric occupancy grid
    decision: Decision              # fsm_state, target_speed, indicator, log[]
    control:  Control               # steer_deg, throttle, brake
```

### 5.2 Data flow

```
read → weather(classify + enhance) → segment(lanes + drivable) → detect(+track)
     → signals(lights, signs, speed zones) → bev(project) → planner(FSM)
     → control(pure pursuit + PID) → [sim] → hud(overlay) → write
```

Weather runs first: downstream stages consume the enhanced frame, and the
visibility score sets detector confidence thresholds and speed caps.

### 5.3 Modules

| File | Responsibility | Depends on |
|---|---|---|
| `src/types.py` | `FrameState` and its member dataclasses | — |
| `src/video_io.py` | frame reader/writer, subsampling, FPS accounting | types |
| `src/weather.py` | CLIP classification; CLAHE / dark-channel-prior enhancement | types |
| `src/segment.py` | TwinLiteNet INT8 → drivable mask + lane mask → polynomial fits | types |
| `src/detect.py` | YOLO11n + ByteTrack → tracks with distance and TTC | types |
| `src/signals.py` | HSV traffic-light state; sign detection; RapidOCR speed limits | types, detect |
| `src/bev.py` | inverse perspective mapping, ego-centric occupancy grid | types |
| `src/planner.py` | finite state machine → driving decision | types |
| `src/control.py` | pure pursuit (lateral), PID (longitudinal) | types |
| `src/sim.py` | bicycle kinematic model, top-down `--sim` window | types |
| `src/hud.py` | overlay renderer: gauges, indicators, scrolling log | types |
| `src/pipeline.py` | wires the stages, owns the frame loop | all |
| `run.py` | CLI entry point | pipeline |

Every module exposes a `__main__` demo so it can be run and understood alone:

```bash
python -m src.segment --video data/input/clear.mp4
python -m src.detect  --video data/input/clear.mp4
python -m src.weather --video data/input/rain.mp4
```

This is the teaching unit: one file, one concept, one visual result.

### 5.4 Planner FSM

States: `LANE_KEEP`, `FOLLOW`, `PREP_CHANGE_L`, `PREP_CHANGE_R`, `CHANGING`,
`STOP_SIGNAL`, `YIELD_PEDESTRIAN`, `EMERGENCY_BRAKE`.

Lane-change rules enforced:
- overtake on the left only (configurable per region)
- target lane gap must exceed a time-headway threshold
- indicator asserted for N frames *before* lateral motion begins
- indicator cleared only after the manoeuvre completes

### 5.5 Control

- **Lateral:** pure pursuit against the lane centerline, with a speed-dependent
  lookahead distance. Output: steering angle in degrees.
- **Longitudinal:** target speed is
  `min(sign_limit, weather_cap, curvature_cap, following_gap_cap)`,
  tracked by a PID producing throttle/brake.
- **Simulator ego:** bicycle kinematic model integrating the same commands.

## 6. Degradation Ladder

This is the all-weather claim expressed as code, not prose.

| Trigger | Response | HUD message |
|---|---|---|
| Lane polynomial fit invalid | steer toward drivable-region centroid | `LANE LOST — FREESPACE MODE` |
| Drivable mask also empty | hold last steering, decay toward zero, cap 20 km/h | `PERCEPTION DEGRADED` |
| Visibility score low | detector confidence 0.35 → 0.20; speed cap; following gap ×1.8 | `FOG — SPEED CAPPED 40` |
| Traffic light state ambiguous | treat as amber, decelerate | `SIGNAL UNCERTAIN` |
| OCR returns no digits | retain previous speed limit | `SIGN UNREAD — HOLDING 50` |

## 7. Configuration

All tuning knobs live in `config.yaml`: camera calibration and IPM homography
points, Canny/ROI parameters for the fallback lane detector, per-weather speed
caps, FSM thresholds, HUD colours and layout. No magic numbers in module code —
tuning is expected to consume more time than writing the code.

## 8. Outputs

- `data/output/<name>_annotated.mp4` — HUD-overlaid video
- `data/output/<name>_control.jsonl` — one record per frame:
  `{idx, t, steer_deg, throttle, brake, target_speed, fsm_state, indicator, weather, events[]}`
  This is the simulated signal to the drive hardware, and the primary artifact
  for the report's results section.
- `docs/figures/` — ablation figures for the report

## 9. Testing

`pytest` over the deterministic maths, which is where real bugs hide:

- bicycle model: straight-line and constant-radius trajectories
- pure pursuit: zero cross-track error → zero steering; known offset → known angle
- PID: step response, anti-windup, output clamping
- IPM: image → ground → image round-trip within tolerance
- FSM: full transition table, including illegal transitions being rejected
- degradation ladder: each trigger produces the specified response

Perception is verified visually per module. No labelled ground truth exists for
the input clips, so no mAP/IoU claims are made against them. The one quantitative
result is the INT8-vs-FP32 IoU comparison in §4.1, which needs no labels.

## 10. Plan

| Day | Modules | Done when |
|---|---|---|
| **D0** | source clips (clear + rain/night); verify env | clips play through `video_io` |
| **D1** | `types`, `video_io`, `hud`, `detect` | boxes + fake gauges render on real video |
| **D2** | `segment`, `weather`, `signals` | lanes, road, weather badge, speed limits on screen |
| **D3** | `bev`, `planner`, `control` | steering angle responds correctly to the scene |
| **D4** | `sim`, `pipeline`, `run.py`, tuning, figures | full annotated video + sim window + report figures |

**MVP cut line.** If the schedule slips, this must exist:
`types, video_io, hud, detect, segment, planner, control, pipeline`.
Drop in this order: `sim`, `bev`, `signals`, `weather`.

## 11. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Tuning IPM homography without calibration data | High | Approximate from lane width assumption (3.5 m) and known image geometry; exact metric accuracy is not required for the demo |
| Clips lack clear speed-limit signs | Medium | Source clips deliberately; fall back to stop-sign and traffic-light logic |
| Perception too slow for iteration | Low | Already measured at 4.4 FPS; develop on 15–20 s clips, render final on Colab |
| CLIP download size (~600 MB) | Low | One-time; cache before Day 2 |
| 4 days is tight for 12 modules | **High** | MVP cut line above; Day 4 reserved for tuning, not features |

## 12. Related Work

- **`voldemortX/pytorch-auto-drive`** — benchmark suite covering SCNN, RESA, LSTR,
  LaneATT, BézierLaneNet on CULane/TuSimple. Not deployable here (CUDA ≥9.2,
  Python 3.6, `mmcv-full` 1.x). Cited for its `MODEL_ZOO.md` comparison metrics,
  which justify the architecture choice made in this design.
- **`Zho29/adverse-weather-object-detection`** — learned restoration (WUNET, DSNet)
  ahead of YOLOv8 detection on Foggy Cityscapes. This design instead uses
  dark-channel-prior dehazing for CPU feasibility and zero training cost.
  Learned restoration is noted as future work.
- **TwinLiteNet** — the deployed segmentation model, via
  `harrylal/TwinLiteNet-onnxruntime`.
