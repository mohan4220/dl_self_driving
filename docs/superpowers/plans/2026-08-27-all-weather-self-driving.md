# All-Weather Self-Driving Car Simulation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a dashcam video into an annotated video showing the steering angle, target speed, indicator state, and decision log a self-driving controller would produce, plus an optional top-down simulator window where a virtual car executes those commands.

**Architecture:** A linear pipeline of pure functions over one `FrameState` dataclass. Perception uses only pretrained frozen models (TwinLiteNet INT8 ONNX, YOLO11n, CLIP, RapidOCR); the driving logic is hand-written classical code (finite state machine, pure pursuit, PID, bicycle model). Video processing is open-loop and advisory; closed-loop behaviour is shown only in the simulator window.

**Tech Stack:** Python 3.12, ONNX Runtime 1.29, ultralytics 8.4 (YOLO11n + ByteTrack), OpenCV 5.0, transformers (CLIP ViT-B/32), rapidocr 3.9, NumPy, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-all-weather-self-driving-design.md`

## Global Constraints

- Python 3.12, virtualenv at `.venv/`. Every command below uses `.venv/bin/python` and `.venv/bin/pytest`.
- **No model training.** All models pretrained and frozen.
- Target hardware: Intel i7-7600U, 2 physical cores, **no GPU**. All inference on `CPUExecutionProvider`.
- TwinLiteNet input is **fixed at 640×360**. Resolution is not a tuning lever (hardcoded `Reshape` to `{1,32,45,80}` in the encoder).
- Measured latency budget: TwinLiteNet INT8 **144 ms**, YOLO11n@640 **82 ms**, total perception **~226 ms/frame ≈ 4.4 FPS**.
- CLIP runs **every 15th frame** only. RapidOCR runs **once per sign track id**, cached.
- Every module exposes a `__main__` demo runnable as `python -m src.<module> --video <path>`.
- All tunable numbers live in `config.yaml`. No magic numbers in module code.
- Ego speed is **commanded, never measured**. HUD labels it `CMD SPD`.
- Pipeline must run **headless** (no `cv2.imshow` on the render path) so it can run on Colab. `--sim` is the local-only interactive path.

## Task Schedule

The spec allows four days. Map tasks to days as follows, and treat the MVP cut line as
real: if Day 3 ends without Task 12 working, drop Tasks 11 and 8 rather than shipping a
pipeline that does not run end to end.

| Day | Tasks | Milestone |
|---|---|---|
| **D0** | Appendix A, Task 0 | clips load, config loads, models present |
| **D1** | Tasks 1, 2, 3, 4 | boxes and a live HUD on real footage |
| **D2** | Tasks 5, 6, 7 | road, lanes, weather label, light state, speed zone |
| **D3** | Tasks 8, 9, 10 | steering angle responds correctly to the scene |
| **D4** | Tasks 11, 12, 13 | annotated video, simulator window, report figures |

**MVP cut line** — must exist to have a project at all:
Tasks 0, 1, 2, 3, 4, 5, 9, 10, 12.
Drop in this order if time runs short: **13, 11, 8, 7, 6**.
Note that dropping Task 8 means distances stay `inf`, so `in_path` is never set and the
planner reduces to lane keeping plus speed limits — degraded, but still a working demo.

---

### Task 0: Project scaffold, git, config

**Files:**
- Create: `.gitignore`, `requirements.txt`, `config.yaml`, `src/__init__.py`, `tests/__init__.py`, `conftest.py`
- Create: `data/input/.gitkeep`, `data/output/.gitkeep`, `docs/figures/.gitkeep`

**Interfaces:**
- Consumes: nothing.
- Produces: `config.yaml` loadable as a `dict` via `yaml.safe_load`; git repo initialised so every later task can commit.

- [ ] **Step 1: Initialise git**

```bash
cd /home/mohan/workws/dl_self_driving_car
git init
git config user.email "sreedhar.teegala@braneenterprises.com"
git config user.name "Sreedhar Teegala"
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
data/input/*
data/output/*
!data/**/.gitkeep
third_party/
.pytest_cache/
*.egg-info/
```

Note: `models/` is deliberately NOT ignored — the INT8 model is 558 KB and is a project deliverable.

- [ ] **Step 3: Write `requirements.txt`**

```
onnxruntime==1.29.0
onnx
ultralytics==8.4.128
opencv-python
numpy
pyyaml
transformers
pillow
rapidocr
matplotlib
tqdm
pytest
```

- [ ] **Step 4: Write `config.yaml`**

```yaml
model:
  segmenter: models/twinlitenet_int8.onnx
  segmenter_fp32: models/twinlitenet_fp32.onnx
  detector: models/yolo11n.pt
  clip: openai/clip-vit-base-patch32
  onnx_threads: 4

video:
  stride: 2              # process every Nth frame
  out_fps: 15

camera:
  # Inverse perspective mapping. Source points are in the 640x360 frame,
  # clockwise from top-left of the trapezoid covering the road ahead.
  seg_w: 640
  seg_h: 360
  ipm_src: [[250, 200], [390, 200], [610, 340], [30, 340]]
  ipm_dst_m: [[-1.75, 30.0], [1.75, 30.0], [1.75, 5.0], [-1.75, 5.0]]
  lane_width_m: 3.5
  wheelbase_m: 2.7
  cam_height_m: 1.35

detect:
  conf: 0.35
  conf_lowvis: 0.20
  imgsz: 640
  classes: [0, 1, 2, 3, 5, 7, 9, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23]
  # COCO ids: person, bicycle, car, motorcycle, bus, truck, traffic light,
  # stop sign, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

segment:
  da_min_px_frac: 0.02   # below this, drivable mask is considered empty
  ll_min_px_frac: 0.0015
  fit_bands: 12          # horizontal bands used to sample lane pixels
  fit_min_bands: 4       # fewer valid bands than this -> lane fit invalid

weather:
  clip_every: 15
  labels: [clear, rain, fog, night, snow]
  prompts:
    clear: "a dashcam photo of a road on a clear sunny day"
    rain:  "a dashcam photo of a road in heavy rain with wet reflections"
    fog:   "a dashcam photo of a road in thick fog with low visibility"
    night: "a dashcam photo of a road at night in the dark"
    snow:  "a dashcam photo of a snow covered road in winter"
  speed_cap_kmh:
    clear: 999
    rain: 60
    fog: 40
    night: 70
    snow: 40
  visibility_lowvis_below: 0.45

planner:
  follow_time_gap_s: 2.0
  lane_change_gap_s: 3.0
  indicator_lead_frames: 15
  ebrake_ttc_s: 1.5
  yield_ped_dist_m: 25.0
  stop_line_dist_m: 20.0
  overtake_side: left

control:
  lookahead_min_m: 6.0
  lookahead_k: 0.6        # lookahead = min + k * v(m/s)
  max_steer_deg: 35.0
  pid: {kp: 0.9, ki: 0.12, kd: 0.05}
  accel_max: 2.5          # m/s^2
  decel_max: 5.0

degrade:
  freespace_speed_cap_kmh: 40
  blind_speed_cap_kmh: 20
  lowvis_gap_multiplier: 1.8

hud:
  log_lines: 5
  font_scale: 0.5
```

- [ ] **Step 5: Write `conftest.py` so `src` imports resolve under pytest**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
```

- [ ] **Step 6: Create package markers and data dirs**

```bash
touch src/__init__.py tests/__init__.py
mkdir -p data/input data/output docs/figures
touch data/input/.gitkeep data/output/.gitkeep docs/figures/.gitkeep
```

- [ ] **Step 7: Verify the pretrained model files are present**

These were produced during the feasibility spike and are already on disk — **no task
creates them**, so check before relying on them:

```bash
ls -lh models/
```
Expected exactly:
```
twinlitenet_int8.onnx    558K   <- primary segmenter (INT8 dynamic quantization)
twinlitenet_fp32.onnx    1.8M   <- reference, used only for the Task 13 IoU figure
yolo11n.pt               5.4M
```

If `twinlitenet_int8.onnx` is missing, regenerate it:

```bash
.venv/bin/python -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('models/twinlitenet_fp32.onnx',
                 'models/twinlitenet_int8.onnx',
                 weight_type=QuantType.QUInt8)
print('regenerated')
"
```

If `twinlitenet_fp32.onnx` is also missing, it comes from
`third_party/twinlitenet/models/best.onnx`:

```bash
git clone --depth 1 https://github.com/harrylal/TwinLiteNet-onnxruntime.git third_party/twinlitenet
cp third_party/twinlitenet/models/best.onnx models/twinlitenet_fp32.onnx
```

- [ ] **Step 8: Verify config loads**

Run: `.venv/bin/python -c "import yaml;c=yaml.safe_load(open('config.yaml'));print(sorted(c))"`
Expected: `['camera', 'control', 'degrade', 'detect', 'hud', 'model', 'planner', 'segment', 'video', 'weather']`

- [ ] **Step 9: Commit**

```bash
git add .gitignore requirements.txt config.yaml conftest.py src tests models docs data
git commit -m "chore: project scaffold, config, pinned deps"
```

---

### Task 1: `src/types.py` — the FrameState spine

**Files:**
- Create: `src/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces: dataclasses `WeatherInfo`, `Track`, `Sign`, `LaneInfo`, `SignalInfo`, `BEVGrid`, `Decision`, `Control`, `FrameState`. Every later task imports from here. Field names below are final — do not rename them.

- [ ] **Step 1: Write the failing test**

`tests/test_types.py`:

```python
import numpy as np
from src.types import FrameState, Track, LaneInfo, Decision, Control


def test_framestate_defaults_are_independent():
    """Mutable defaults must not be shared between instances."""
    img = np.zeros((360, 640, 3), np.uint8)
    a = FrameState(idx=0, t=0.0, raw=img, img=img)
    b = FrameState(idx=1, t=0.1, raw=img, img=img)

    a.objects.append(Track(id=1, cls="car", bbox=(0, 0, 10, 10), conf=0.9))
    a.decision.log.append("hello")

    assert b.objects == []
    assert b.decision.log == []


def test_framestate_sensible_defaults():
    img = np.zeros((360, 640, 3), np.uint8)
    s = FrameState(idx=0, t=0.0, raw=img, img=img)

    assert s.weather.label == "clear"
    assert s.weather.visibility == 1.0
    assert s.lanes.valid is False
    assert s.signals.light_state == "none"
    assert s.decision.fsm_state == "LANE_KEEP"
    assert s.decision.indicator == "off"
    assert s.control.steer_deg == 0.0
    assert s.drivable is None


def test_track_defaults_to_unknown_distance():
    t = Track(id=3, cls="person", bbox=(1, 2, 3, 4), conf=0.5)
    assert t.dist_m == float("inf")
    assert t.ttc_s == float("inf")
    assert t.in_path is False
    assert np.isnan(t.lateral_m)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.types'`

- [ ] **Step 3: Write `src/types.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_types.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/types.py tests/test_types.py
git commit -m "feat: FrameState spine dataclasses"
```

---

### Task 2: `src/video_io.py` — reader, writer, control log

**Files:**
- Create: `src/video_io.py`
- Test: `tests/test_video_io.py`

**Interfaces:**
- Consumes: `src.types.FrameState`.
- Produces:
  - `VideoReader(path: str, stride: int = 1, max_frames: int | None = None)` — iterable yielding `(idx: int, t: float, frame: np.ndarray)`; attributes `.fps`, `.width`, `.height`, `.n_frames`.
  - `VideoWriter(path: str, fps: float, size: tuple[int, int])` — `.write(frame)`, `.close()`, context manager.
  - `ControlLogWriter(path: str)` — `.write(state: FrameState)`, `.close()`, context manager.

- [ ] **Step 1: Write the failing test**

`tests/test_video_io.py`:

```python
import json

import cv2
import numpy as np
import pytest

from src.types import FrameState
from src.video_io import ControlLogWriter, VideoReader, VideoWriter


@pytest.fixture
def clip(tmp_path):
    """A 10-frame 64x48 video at 10 fps, frame i filled with intensity i*10."""
    p = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(10):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()
    return str(p)


def test_reader_reads_every_frame(clip):
    r = VideoReader(clip)
    frames = list(r)
    assert len(frames) == 10
    assert r.width == 64 and r.height == 48
    assert r.fps == pytest.approx(10.0, abs=0.1)


def test_reader_stride_skips_frames(clip):
    frames = list(VideoReader(clip, stride=3))
    assert [i for i, _, _ in frames] == [0, 3, 6, 9]


def test_reader_timestamps_use_source_fps(clip):
    frames = list(VideoReader(clip, stride=2))
    assert [round(t, 3) for _, t, _ in frames] == [0.0, 0.2, 0.4, 0.6, 0.8]


def test_reader_max_frames(clip):
    assert len(list(VideoReader(clip, max_frames=4))) == 4


def test_writer_roundtrip(clip, tmp_path):
    out = tmp_path / "out.mp4"
    with VideoWriter(str(out), 10.0, (64, 48)) as w:
        for _, _, f in VideoReader(clip):
            w.write(f)
    assert out.exists() and out.stat().st_size > 0
    assert len(list(VideoReader(str(out)))) == 10


def test_control_log_writes_one_json_per_frame(tmp_path):
    img = np.zeros((48, 64, 3), np.uint8)
    p = tmp_path / "log.jsonl"
    with ControlLogWriter(str(p)) as log:
        for i in range(3):
            s = FrameState(idx=i, t=i * 0.1, raw=img, img=img)
            s.control.steer_deg = float(i)
            s.decision.log.append(f"event {i}")
            log.write(s)

    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert len(rows) == 3
    assert rows[1]["steer_deg"] == 1.0
    assert rows[2]["events"] == ["event 2"]
    assert rows[0]["fsm_state"] == "LANE_KEEP"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_video_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.video_io'`

- [ ] **Step 3: Write `src/video_io.py`**

```python
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
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        idx, emitted = 0, 0
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
        self._cap.release()

    def __len__(self) -> int:
        return self.n_frames // self.stride


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_video_io.py -v`
Expected: 6 passed

- [ ] **Step 5: Add the `__main__` demo**

Append to `src/video_io.py`:

```python
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
```

- [ ] **Step 6: Commit**

```bash
git add src/video_io.py tests/test_video_io.py
git commit -m "feat: video reader/writer and JSONL control log"
```

---

### Task 3: `src/detect.py` — YOLO11n detection and tracking

**Files:**
- Create: `src/detect.py`
- Test: `tests/test_detect.py`

**Interfaces:**
- Consumes: `src.types.Track`.
- Produces:
  - `COCO_NAMES: dict[int, str]` — the COCO ids we keep, mapped to our own class names.
  - `Detector(weights: str, imgsz: int = 640, classes: list[int] | None = None)` with `.track(bgr: np.ndarray, conf: float) -> list[Track]`.
  - `bbox_bottom_centre(bbox) -> tuple[float, float]` — the ground contact point of a box.
  - `update_ttc(tracks: list[Track], prev_dist: dict[int, float], dt: float) -> dict[int, float]` — sets `.ttc_s` in place, returns the new distance memo.

- [ ] **Step 1: Write the failing test**

`tests/test_detect.py`. These tests cover only the deterministic geometry — the neural network itself is verified visually via the `__main__` demo.

```python
import pytest

from src.detect import bbox_bottom_centre, update_ttc
from src.types import Track


def test_bbox_bottom_centre():
    assert bbox_bottom_centre((10, 20, 30, 60)) == (20.0, 60.0)


def test_ttc_closing_object():
    """Object closes 5 m in 0.5 s -> 10 m/s closing -> 20 m gap = 2.0 s."""
    t = Track(id=1, cls="car", bbox=(0, 0, 1, 1), conf=0.9, dist_m=20.0)
    prev = update_ttc([t], {1: 25.0}, dt=0.5)
    assert t.ttc_s == pytest.approx(2.0)
    assert prev == {1: 20.0}


def test_ttc_receding_object_is_infinite():
    t = Track(id=1, cls="car", bbox=(0, 0, 1, 1), conf=0.9, dist_m=30.0)
    update_ttc([t], {1: 25.0}, dt=0.5)
    assert t.ttc_s == float("inf")


def test_ttc_unknown_history_is_infinite():
    t = Track(id=7, cls="car", bbox=(0, 0, 1, 1), conf=0.9, dist_m=10.0)
    update_ttc([t], {}, dt=0.5)
    assert t.ttc_s == float("inf")


def test_ttc_ignores_unknown_distance():
    t = Track(id=1, cls="car", bbox=(0, 0, 1, 1), conf=0.9)  # dist_m = inf
    prev = update_ttc([t], {1: 25.0}, dt=0.5)
    assert t.ttc_s == float("inf")
    assert 1 not in prev
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_detect.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.detect'`

- [ ] **Step 3: Write `src/detect.py`**

```python
"""Object detection and tracking.

YOLO11n on COCO already covers every class this project needs: vehicles,
people, animals, traffic lights and stop signs. ByteTrack (bundled with
ultralytics) gives stable ids across frames, which is what makes
time-to-collision and once-per-sign OCR possible.
"""
from __future__ import annotations

import numpy as np

from src.types import Track

# COCO ids we care about, mapped to the names our planner reasons about.
COCO_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic_light",
    11: "stop_sign",
    15: "animal", 16: "animal", 17: "animal", 18: "animal",
    19: "animal", 20: "animal", 21: "animal", 22: "animal", 23: "animal",
}

VEHICLE = {"car", "motorcycle", "bus", "truck", "bicycle"}
VULNERABLE = {"person", "animal"}


def bbox_bottom_centre(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Where the object touches the ground, in image pixels."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def update_ttc(
    tracks: list[Track], prev_dist: dict[int, float], dt: float
) -> dict[int, float]:
    """Set `ttc_s` on each track from how fast its distance is shrinking.

    Returns the distance memo to pass in on the next frame. Tracks with an
    unknown distance are excluded so they cannot poison the next frame.
    """
    new: dict[int, float] = {}
    for t in tracks:
        if not np.isfinite(t.dist_m):
            continue
        new[t.id] = t.dist_m
        was = prev_dist.get(t.id)
        if was is None or dt <= 0:
            continue
        closing = (was - t.dist_m) / dt          # m/s, positive = approaching
        if closing > 0.1:
            t.ttc_s = t.dist_m / closing
    return new


class Detector:
    def __init__(
        self, weights: str, imgsz: int = 640, classes: list[int] | None = None
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.classes = classes

    def track(self, bgr: np.ndarray, conf: float = 0.35) -> list[Track]:
        res = self.model.track(
            bgr,
            imgsz=self.imgsz,
            conf=conf,
            classes=self.classes,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        out: list[Track] = []
        if res.boxes is None or res.boxes.id is None:
            return out

        for box, cid, tid, c in zip(
            res.boxes.xyxy.tolist(),
            res.boxes.cls.tolist(),
            res.boxes.id.tolist(),
            res.boxes.conf.tolist(),
        ):
            name = COCO_NAMES.get(int(cid))
            if name is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            out.append(
                Track(id=int(tid), cls=name, bbox=(x1, y1, x2, y2), conf=float(c))
            )
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_detect.py -v`
Expected: 5 passed

- [ ] **Step 5: Add the `__main__` demo and verify it visually**

Append to `src/detect.py`:

```python
if __name__ == "__main__":
    import argparse

    import cv2
    import yaml

    from src.video_io import VideoReader

    ap = argparse.ArgumentParser(description="Block 2 demo: detection + tracking.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=60)
    a = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    det = Detector(
        cfg["model"]["detector"],
        imgsz=cfg["detect"]["imgsz"],
        classes=cfg["detect"]["classes"],
    )

    for _, _, frame in VideoReader(a.video, max_frames=a.frames):
        for t in det.track(frame, cfg["detect"]["conf"]):
            colour = (0, 0, 255) if t.cls in VULNERABLE else (0, 200, 255)
            x1, y1, x2, y2 = t.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                frame, f"{t.cls}#{t.id}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
            )
        cv2.imshow("detect", frame)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
```

Run: `.venv/bin/python -m src.detect --video data/input/clear.mp4`
Expected: boxes drawn on vehicles/people, and **the id after `#` stays the same** as an object moves across frames. If ids churn every frame, tracking is broken — check `persist=True`.

- [ ] **Step 6: Commit**

```bash
git add src/detect.py tests/test_detect.py
git commit -m "feat: YOLO11n detection with ByteTrack ids and TTC"
```

---

### Task 4: `src/hud.py` — overlay renderer

**Files:**
- Create: `src/hud.py`
- Test: `tests/test_hud.py`

**Interfaces:**
- Consumes: `src.types.FrameState`, `src.detect.VULNERABLE`.
- Produces: `render(state: FrameState, cfg: dict) -> np.ndarray` — returns a NEW BGR image, never mutates `state.img`.

- [ ] **Step 1: Write the failing test**

`tests/test_hud.py`:

```python
import numpy as np

from src.hud import render
from src.types import FrameState, Track


def _cfg():
    return {"hud": {"log_lines": 5, "font_scale": 0.5}}


def _state():
    img = np.zeros((360, 640, 3), np.uint8)
    s = FrameState(idx=42, t=1.4, raw=img.copy(), img=img.copy())
    s.control.steer_deg = -7.2
    s.decision.target_speed = 30.0
    s.decision.fsm_state = "LANE_KEEP"
    s.decision.indicator = "right"
    s.decision.log = ["SPEED ZONE 30 DETECTED"]
    s.objects = [Track(id=1, cls="car", bbox=(100, 150, 200, 250), conf=0.9)]
    return s


def test_render_does_not_mutate_input_frame():
    s = _state()
    before = s.img.copy()
    render(s, _cfg())
    assert np.array_equal(s.img, before)


def test_render_returns_same_shape_and_draws_something():
    s = _state()
    out = render(s, _cfg())
    assert out.shape == s.img.shape
    assert out.dtype == np.uint8
    assert out.sum() > 0          # a black frame in, pixels drawn out


def test_render_survives_empty_state():
    """A frame where every perception stage failed must still render."""
    img = np.zeros((360, 640, 3), np.uint8)
    out = render(FrameState(idx=0, t=0.0, raw=img, img=img), _cfg())
    assert out.shape == img.shape


def test_render_log_is_capped():
    s = _state()
    s.decision.log = [f"line {i}" for i in range(50)]
    out = render(s, _cfg())     # must not raise or run off the frame
    assert out.shape == s.img.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_hud.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.hud'`

- [ ] **Step 3: Write `src/hud.py`**

```python
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

    # --- lane overlay ---
    if state.drivable is not None:
        mask = cv2.resize(state.drivable, (w, h), interpolation=cv2.INTER_NEAREST)
        tint = np.zeros_like(img)
        tint[mask.astype(bool)] = (0, 120, 0)
        cv2.addWeighted(img, 1.0, tint, 0.35, 0, img)
    if state.lane_mask is not None:
        mask = cv2.resize(state.lane_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        img[mask.astype(bool)] = RED

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
    lines = state.decision.log[-hud["log_lines"]:]
    if lines:
        _panel(img, 0, h - 22 * len(lines) - 8, 340, h)
        for i, line in enumerate(lines):
            y = h - 8 - 22 * (len(lines) - 1 - i)
            cv2.putText(img, f"> {line[:44]}", (8, y), FONT, fs - 0.05, AMBER, 1)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_hud.py -v`
Expected: 4 passed

- [ ] **Step 5: Verify visually**

Run: `.venv/bin/python -m src.hud --video data/input/clear.mp4`
Expected: steering wheel rotates smoothly, indicators blink and cycle left/right/off, throttle bar oscillates, log lines readable over the road. **This is the demo that sells the project** — spend a few minutes on colours and placement here.

- [ ] **Step 6: Commit**

```bash
git add src/hud.py tests/test_hud.py
git commit -m "feat: HUD overlay renderer"
```

---

### Task 5: `src/segment.py` — TwinLiteNet drivable area and lane lines

**Files:**
- Create: `src/segment.py`
- Test: `tests/test_segment.py`

**Interfaces:**
- Consumes: `src.types.LaneInfo`.
- Produces:
  - `Segmenter(model_path: str, threads: int = 4)` with `.infer(bgr) -> tuple[np.ndarray, np.ndarray]` returning `(da_mask, ll_mask)`, both `uint8` 0/1 at 360×640.
  - `fit_lanes(ll_mask, da_mask, cfg) -> LaneInfo`.
  - `freespace_offset(da_mask, cfg) -> float | None` — lateral offset in metres from the drivable-area centroid, used when lane fitting fails.

**Background the implementer needs:** the ONNX model takes `images` of shape `[1,3,360,640]`, RGB, scaled by `1/255`, no mean subtraction. It returns two `[1,2,360,640]` logit tensors named `da` and `ll`; class 1 is the positive class, so the mask is `argmax(axis=0) == 1`. Input resolution is fixed — the encoder has a hardcoded reshape.

- [ ] **Step 1: Write the failing test**

`tests/test_segment.py`. Lane fitting is tested on synthetic masks with known geometry, so the tests are exact and need no video.

```python
import numpy as np
import pytest

from src.segment import fit_lanes, freespace_offset

H, W = 360, 640


def _cfg():
    return {
        "segment": {
            "da_min_px_frac": 0.02,
            "ll_min_px_frac": 0.0015,
            "fit_bands": 12,
            "fit_min_bands": 4,
        },
        "camera": {"lane_width_m": 3.5, "seg_w": W, "seg_h": H},
    }


def _straight_lanes(left_x: int, right_x: int, y0: int = 180) -> np.ndarray:
    """Two vertical lane lines from y0 to the bottom of the frame."""
    m = np.zeros((H, W), np.uint8)
    m[y0:, left_x - 2:left_x + 2] = 1
    m[y0:, right_x - 2:right_x + 2] = 1
    return m


def _road(y0: int = 180) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    m[y0:, 100:540] = 1
    return m


def test_centred_straight_lanes_give_zero_offset():
    ll = _straight_lanes(240, 400)          # centre = 320 = image centre
    lanes = fit_lanes(ll, _road(), _cfg())
    assert lanes.valid is True
    assert lanes.offset_m == pytest.approx(0.0, abs=0.05)
    assert abs(lanes.heading_err_rad) < 0.02


def test_lane_shifted_left_means_ego_is_right_of_centre():
    """Lane centre at x=270 but camera at x=320 -> ego is 50 px right."""
    ll = _straight_lanes(190, 350)          # centre = 270
    lanes = fit_lanes(ll, _road(), _cfg())
    # 160 px between lines == 3.5 m, so 50 px == 1.09 m
    assert lanes.offset_m == pytest.approx(50 * 3.5 / 160, abs=0.1)
    assert lanes.offset_m > 0


def test_empty_lane_mask_is_invalid():
    lanes = fit_lanes(np.zeros((H, W), np.uint8), _road(), _cfg())
    assert lanes.valid is False
    assert lanes.left_fit is None and lanes.right_fit is None


def test_single_line_is_invalid():
    m = np.zeros((H, W), np.uint8)
    m[180:, 238:242] = 1                    # only a left line
    assert fit_lanes(m, _road(), _cfg()).valid is False


def test_freespace_offset_from_centred_road_is_zero():
    assert freespace_offset(_road(), _cfg()) == pytest.approx(0.0, abs=0.1)


def test_freespace_offset_from_shifted_road_is_signed():
    m = np.zeros((H, W), np.uint8)
    m[180:, 0:440] = 1                      # road centroid at x=220, left of centre
    off = freespace_offset(m, _cfg())
    assert off is not None and off > 0      # ego is right of the free space


def test_freespace_offset_none_when_road_missing():
    assert freespace_offset(np.zeros((H, W), np.uint8), _cfg()) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_segment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.segment'`

- [ ] **Step 3: Write `src/segment.py`**

```python
"""Road and lane perception.

TwinLiteNet gives two masks from one forward pass: the drivable area and
the lane lines. The drivable area is what keeps this system honest in bad
weather — when paint is covered by snow or washed out by rain, the lane
fit dies but the free-space mask survives, so we can still steer.

The INT8-quantized model runs at 144 ms/frame on the target CPU versus
774-1350 ms for FP32, at 0.985 / 0.941 mask IoU against FP32.
"""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from src.types import LaneInfo

SEG_W, SEG_H = 640, 360      # fixed by the model; not configurable


class Segmenter:
    def __init__(self, model_path: str, threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(
            model_path, so, providers=["CPUExecutionProvider"]
        )

    def infer(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (drivable_mask, lane_mask), uint8 0/1, both 360x640."""
        small = cv2.resize(bgr, (SEG_W, SEG_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        da, ll = self.sess.run(["da", "ll"], {"images": x})
        return (
            (np.argmax(da[0], 0) == 1).astype(np.uint8),
            (np.argmax(ll[0], 0) == 1).astype(np.uint8),
        )


def _band_centres(mask: np.ndarray, cfg: dict) -> tuple[list, list]:
    """Sample lane pixels in horizontal bands, split into left and right.

    Working in bands rather than fitting all pixels at once keeps far-away
    noise from dominating the fit, because every band contributes equally.
    """
    h, w = mask.shape
    n = cfg["segment"]["fit_bands"]
    cx = w / 2.0
    left, right = [], []
    for i in range(n):
        y0, y1 = int(h * i / n), int(h * (i + 1) / n)
        band = mask[y0:y1]
        ys, xs = np.nonzero(band)
        if xs.size == 0:
            continue
        y = (y0 + y1) / 2.0
        lx, rx = xs[xs < cx], xs[xs >= cx]
        if lx.size >= 3:
            left.append((y, float(np.median(lx))))
        if rx.size >= 3:
            right.append((y, float(np.median(rx))))
    return left, right


def _fit(points: list) -> np.ndarray | None:
    """Fit x = f(y) as a quadratic. Needs at least 3 points."""
    if len(points) < 3:
        return None
    ys = np.array([p[0] for p in points])
    xs = np.array([p[1] for p in points])
    return np.polyfit(ys, xs, 2)


def fit_lanes(ll_mask: np.ndarray, da_mask: np.ndarray, cfg: dict) -> LaneInfo:
    """Turn the lane-line mask into a centreline offset and heading error."""
    seg = cfg["segment"]
    info = LaneInfo()

    if ll_mask.mean() < seg["ll_min_px_frac"]:
        return info                                   # nothing to fit

    left_pts, right_pts = _band_centres(ll_mask, cfg)
    if len(left_pts) < seg["fit_min_bands"] or len(right_pts) < seg["fit_min_bands"]:
        return info

    lf, rf = _fit(left_pts), _fit(right_pts)
    if lf is None or rf is None:
        return info

    h, w = ll_mask.shape
    y_near = h - 1.0
    xl, xr = float(np.polyval(lf, y_near)), float(np.polyval(rf, y_near))
    lane_px = xr - xl
    if lane_px < 40:                                  # degenerate / crossed fits
        return info

    # Pixels to metres, calibrated by the known lane width. This avoids
    # needing camera intrinsics for the lateral measurement.
    m_per_px = cfg["camera"]["lane_width_m"] / lane_px

    lane_cx = (xl + xr) / 2.0
    info.left_fit, info.right_fit = lf, rf
    info.offset_m = (w / 2.0 - lane_cx) * m_per_px    # + = ego right of centre

    # Heading error from the centreline slope dx/dy at the bottom of the frame.
    centre = (lf + rf) / 2.0
    dxdy = float(np.polyval(np.polyder(centre), y_near))
    info.heading_err_rad = float(np.arctan(-dxdy))    # + = lane heads right

    d2 = float(np.polyval(np.polyder(centre, 2), y_near))
    info.curvature_m = (
        float("inf") if abs(d2) < 1e-9
        else abs((1 + dxdy ** 2) ** 1.5 / d2) * m_per_px
    )
    info.valid = True
    return info


def freespace_offset(da_mask: np.ndarray, cfg: dict) -> float | None:
    """Fallback steering target: the centroid of drivable pixels ahead.

    Used when lane paint is invisible (snow, heavy rain, night). Returns
    metres, positive when the ego is right of the free space, or None when
    there is no usable road surface at all.
    """
    if da_mask.mean() < cfg["segment"]["da_min_px_frac"]:
        return None
    h, w = da_mask.shape
    band = da_mask[int(h * 0.55): int(h * 0.85)]      # mid-distance road only
    xs = np.nonzero(band)[1]
    if xs.size == 0:
        return None
    # Scale by an assumed lane width spanning a quarter of the frame width.
    m_per_px = cfg["camera"]["lane_width_m"] / (w / 4.0)
    return float((w / 2.0 - xs.mean()) * m_per_px)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_segment.py -v`
Expected: 7 passed

- [ ] **Step 5: Add the `__main__` demo and verify visually**

Append to `src/segment.py`:

```python
if __name__ == "__main__":
    import argparse
    import time

    import yaml

    from src.video_io import VideoReader

    ap = argparse.ArgumentParser(description="Block 3 demo: road + lane segmentation.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--fp32", action="store_true", help="use the unquantized model")
    a = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    path = cfg["model"]["segmenter_fp32"] if a.fp32 else cfg["model"]["segmenter"]
    seg = Segmenter(path, cfg["model"]["onnx_threads"])
    print(f"model: {path}")

    for idx, _, frame in VideoReader(a.video):
        t0 = time.perf_counter()
        da, ll = seg.infer(frame)
        ms = (time.perf_counter() - t0) * 1000
        lanes = fit_lanes(ll, da, cfg)

        vis = cv2.resize(frame, (SEG_W, SEG_H))
        vis[da.astype(bool)] = (0.6 * vis[da.astype(bool)]
                                + 0.4 * np.array([0, 180, 0])).astype(np.uint8)
        vis[ll.astype(bool)] = (0, 0, 255)
        if lanes.valid:
            for y in range(180, SEG_H, 10):
                xc = int((np.polyval(lanes.left_fit, y)
                          + np.polyval(lanes.right_fit, y)) / 2)
                cv2.circle(vis, (xc, y), 2, (255, 255, 0), -1)
        txt = (f"{ms:.0f}ms  valid={lanes.valid}  "
               f"off={lanes.offset_m:+.2f}m  hdg={np.degrees(lanes.heading_err_rad):+.1f}deg")
        cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("segment", vis)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
```

Run: `.venv/bin/python -m src.segment --video data/input/clear.mp4`
Expected: green road, red lane lines, a cyan centreline, and roughly **140-160 ms** per frame. `off=` should hover near 0 when driving centred and go positive when the car drifts right.

Sanity check the quantization claim: `.venv/bin/python -m src.segment --video data/input/clear.mp4 --fp32` should look near-identical but report ~800-1300 ms.

- [ ] **Step 6: Commit**

```bash
git add src/segment.py tests/test_segment.py
git commit -m "feat: TwinLiteNet INT8 road/lane segmentation with lane fitting"
```

---

### Task 6: `src/weather.py` — weather classification and image enhancement

**Files:**
- Create: `src/weather.py`
- Test: `tests/test_weather.py`

**Interfaces:**
- Consumes: `src.types.WeatherInfo`.
- Produces:
  - `visibility_score(bgr) -> float` in `0..1`.
  - `dark_channel_dehaze(bgr, omega=0.85, t0=0.1) -> np.ndarray`.
  - `clahe_night(bgr, clip=2.0, grid=8) -> np.ndarray`.
  - `enhance(bgr, info: WeatherInfo) -> np.ndarray` — dispatches on `info.label`.
  - `WeatherClassifier(model_id: str, labels: list[str], prompts: dict[str, str])` with `.classify(bgr) -> WeatherInfo`.
  - `speed_cap_kmh(info: WeatherInfo, cfg: dict) -> float`.

- [ ] **Step 1: Write the failing test**

`tests/test_weather.py`. Only the classical image processing is unit-tested; CLIP is verified in the demo.

```python
import numpy as np
import pytest

from src.types import WeatherInfo
from src.weather import (
    clahe_night,
    dark_channel_dehaze,
    enhance,
    speed_cap_kmh,
    visibility_score,
)


def _cfg():
    return {
        "weather": {
            "speed_cap_kmh": {"clear": 999, "rain": 60, "fog": 40, "night": 70, "snow": 40},
            "visibility_lowvis_below": 0.45,
        }
    }


def _sharp_colourful():
    """High contrast checkerboard: should score as good visibility."""
    img = np.zeros((120, 160, 3), np.uint8)
    img[::8] = 255
    img[:, ::8] = (0, 200, 255)
    return img


def _flat_grey():
    """Uniform mid-grey: no contrast, no detail — like thick fog."""
    return np.full((120, 160, 3), 128, np.uint8)


def _near_black():
    return np.full((120, 160, 3), 8, np.uint8)


def test_visibility_score_in_range():
    for img in (_sharp_colourful(), _flat_grey(), _near_black()):
        v = visibility_score(img)
        assert 0.0 <= v <= 1.0


def test_sharp_scene_scores_higher_than_flat_fog():
    assert visibility_score(_sharp_colourful()) > visibility_score(_flat_grey())


def test_near_black_scene_scores_low():
    assert visibility_score(_near_black()) < 0.45


def test_dehaze_increases_contrast_on_hazy_image():
    """A washed-out image dehazed should have a wider intensity spread."""
    hazy = (0.4 * _sharp_colourful().astype(np.float32) + 150).clip(0, 255).astype(np.uint8)
    out = dark_channel_dehaze(hazy)
    assert out.shape == hazy.shape and out.dtype == np.uint8
    assert out.std() > hazy.std()


def test_clahe_brightens_dark_image():
    dark = (_sharp_colourful() * 0.2).astype(np.uint8)
    out = clahe_night(dark)
    assert out.shape == dark.shape and out.dtype == np.uint8
    assert out.mean() > dark.mean()


def test_enhance_clear_is_a_passthrough_copy():
    img = _sharp_colourful()
    out = enhance(img, WeatherInfo(label="clear"))
    assert np.array_equal(out, img)
    assert out is not img          # must not hand back the caller's buffer


def test_enhance_dispatches_per_label_without_error():
    img = _sharp_colourful()
    for label in ("clear", "rain", "fog", "night", "snow"):
        out = enhance(img, WeatherInfo(label=label))
        assert out.shape == img.shape and out.dtype == np.uint8


def test_speed_cap_uses_label():
    assert speed_cap_kmh(WeatherInfo(label="fog"), _cfg()) == 40
    assert speed_cap_kmh(WeatherInfo(label="clear"), _cfg()) == 999


def test_speed_cap_falls_back_for_unknown_label():
    assert speed_cap_kmh(WeatherInfo(label="hail"), _cfg()) == 999
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_weather.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.weather'`

- [ ] **Step 3: Write `src/weather.py`**

```python
"""Weather sensing and image restoration.

Two jobs. First, decide what the weather is, so the planner can slow down
and loosen its thresholds — that policy change is what makes the system
"all-weather" rather than merely "works in daylight". Second, restore the
frame so downstream detection has a fighting chance.

CLIP is used zero-shot, which means no training and no labelled weather
dataset. It is slow (~300 ms) so it runs only every Nth frame; weather does
not change between consecutive frames.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.types import WeatherInfo


def visibility_score(bgr: np.ndarray) -> float:
    """How much usable information is in this frame, from 0 to 1.

    Three cheap, complementary cues:
      brightness  - night and heavy shadow drive this down
      contrast    - fog and haze flatten the histogram
      detail      - rain streaks and defocus blur kill high frequencies
    Being handcrafted, this is fast enough to run on every frame, unlike CLIP.
    """
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.clip(grey.mean() / 110.0, 0.0, 1.0))
    contrast = float(np.clip(grey.std() / 55.0, 0.0, 1.0))
    detail = float(np.clip(cv2.Laplacian(grey, cv2.CV_64F).var() / 400.0, 0.0, 1.0))
    return float(np.clip(0.35 * brightness + 0.40 * contrast + 0.25 * detail, 0.0, 1.0))


def dark_channel_dehaze(
    bgr: np.ndarray, omega: float = 0.85, t0: float = 0.1
) -> np.ndarray:
    """He et al. dark-channel-prior dehazing.

    The prior: in a clear outdoor image, most local patches contain at least
    one very dark colour channel. Where that fails, the patch is hazy, and
    the amount by which it fails estimates the haze thickness.
    """
    img = bgr.astype(np.float32) / 255.0
    patch = 15

    dark = cv2.erode(
        img.min(axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    )

    # Atmospheric light: mean of the brightest 0.1% of dark-channel pixels.
    flat = dark.ravel()
    n = max(1, flat.size // 1000)
    idx = np.argpartition(flat, -n)[-n:]
    A = img.reshape(-1, 3)[idx].mean(axis=0)
    A = np.maximum(A, 1e-3)

    trans = 1.0 - omega * cv2.erode(
        (img / A).min(axis=2),
        cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch)),
    )
    trans = cv2.blur(trans, (30, 30))                 # cheap stand-in for a guided filter
    trans = np.maximum(trans, t0)[..., None]

    out = (img - A) / trans + A
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def clahe_night(bgr: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on luminance only.

    Applied to L in LAB so colours are not distorted, unlike equalising RGB.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(
        lab[..., 0]
    )
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _derain(bgr: np.ndarray) -> np.ndarray:
    """Suppress rain streaks with an edge-preserving filter, then re-sharpen.

    A plain blur would remove the streaks and the lane paint together;
    bilateral filtering keeps strong edges while smoothing thin bright ones.
    """
    smooth = cv2.bilateralFilter(bgr, 7, 60, 60)
    return cv2.addWeighted(smooth, 1.4, cv2.GaussianBlur(smooth, (0, 0), 3), -0.4, 0)


def enhance(bgr: np.ndarray, info: WeatherInfo) -> np.ndarray:
    """Restore the frame according to the detected weather."""
    if info.label == "fog":
        return dark_channel_dehaze(bgr)
    if info.label == "night":
        return clahe_night(bgr)
    if info.label == "rain":
        return _derain(bgr)
    if info.label == "snow":
        return clahe_night(bgr, clip=1.5)             # snow glare: gentler
    return bgr.copy()


def speed_cap_kmh(info: WeatherInfo, cfg: dict) -> float:
    caps = cfg["weather"]["speed_cap_kmh"]
    return float(caps.get(info.label, caps["clear"]))


class WeatherClassifier:
    """Zero-shot weather classification with CLIP. No training, no dataset."""

    def __init__(self, model_id: str, labels: list[str], prompts: dict[str, str]):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.labels = labels
        self.model = CLIPModel.from_pretrained(model_id).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self._texts = [prompts[l] for l in labels]

    def classify(self, bgr: np.ndarray) -> WeatherInfo:
        from PIL import Image

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.proc(
            text=self._texts,
            images=Image.fromarray(rgb),
            return_tensors="pt",
            padding=True,
        )
        with self.torch.no_grad():
            probs = self.model(**inputs).logits_per_image.softmax(dim=1)[0]
        i = int(probs.argmax())
        return WeatherInfo(
            label=self.labels[i],
            confidence=float(probs[i]),
            visibility=visibility_score(bgr),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_weather.py -v`
Expected: 9 passed

- [ ] **Step 5: Add the `__main__` demo and verify visually**

Append to `src/weather.py`:

```python
if __name__ == "__main__":
    import argparse

    import yaml

    from src.video_io import VideoReader

    ap = argparse.ArgumentParser(
        description="Block 4 demo: weather label + before/after enhancement."
    )
    ap.add_argument("--video", required=True)
    ap.add_argument("--force", default=None, help="skip CLIP, force this label")
    a = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    wc = cfg["weather"]
    clf = None
    if a.force is None:
        print("loading CLIP (first run downloads ~600 MB)...")
        clf = WeatherClassifier(cfg["model"]["clip"], wc["labels"], wc["prompts"])

    info = WeatherInfo(label=a.force or "clear")
    for idx, _, frame in VideoReader(a.video):
        if clf is not None and idx % wc["clip_every"] == 0:
            info = clf.classify(frame)
        else:
            info.visibility = visibility_score(frame)

        out = enhance(frame, info)
        pair = np.hstack([cv2.resize(frame, (480, 270)), cv2.resize(out, (480, 270))])
        cv2.putText(pair, "RAW", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(
            pair,
            f"ENHANCED  {info.label} p={info.confidence:.2f} vis={info.visibility:.2f} "
            f"cap={speed_cap_kmh(info, cfg):.0f}",
            (490, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2,
        )
        cv2.imshow("weather", pair)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
```

Run on both clips:
```bash
.venv/bin/python -m src.weather --video data/input/clear.mp4
.venv/bin/python -m src.weather --video data/input/rain.mp4
```
Expected: the label tracks the clip. If CLIP mislabels, **edit the prompts in `config.yaml`** — that is the intended tuning knob, and prompt sensitivity is itself worth a paragraph in the report. Screenshot the rain/night before-after pair; it is a report figure.

- [ ] **Step 6: Commit**

```bash
git add src/weather.py tests/test_weather.py
git commit -m "feat: CLIP weather classification and weather-specific enhancement"
```

---

### Task 7: `src/signals.py` — traffic lights, signs, speed zones

**Files:**
- Create: `src/signals.py`
- Test: `tests/test_signals.py`

**Interfaces:**
- Consumes: `src.types.Track`, `src.types.Sign`, `src.types.SignalInfo`.
- Produces:
  - `classify_light(bgr, bbox) -> str` returning `red | amber | green | unknown`.
  - `SignReader(min_box_px: int = 22)` with `.read_speed_limit(bgr, bbox, track_id) -> int | None`, caching one OCR call per track id.
  - `read_signals(bgr, tracks, reader, prev: SignalInfo, cfg) -> SignalInfo`.

- [ ] **Step 1: Write the failing test**

`tests/test_signals.py`:

```python
import numpy as np

from src.signals import SignReader, classify_light, read_signals
from src.types import SignalInfo, Track


def _cfg():
    return {"signals": {"min_sign_px": 22}}


def _light(colour_bgr, pos: str) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A 30x90 traffic light with one lamp lit at top/middle/bottom."""
    img = np.zeros((200, 200, 3), np.uint8)
    x, y, w, h = 50, 50, 30, 90
    row = {"top": 0, "mid": 1, "bot": 2}[pos]
    cy = y + int(h * (row + 0.5) / 3)
    img[cy - 8:cy + 8, x + 7:x + 23] = colour_bgr
    return img, (x, y, x + w, y + h)


def test_red_lamp_at_top_is_red():
    img, box = _light((0, 0, 255), "top")
    assert classify_light(img, box) == "red"


def test_green_lamp_at_bottom_is_green():
    img, box = _light((0, 255, 0), "bot")
    assert classify_light(img, box) == "green"


def test_amber_lamp_in_middle_is_amber():
    img, box = _light((0, 190, 255), "mid")
    assert classify_light(img, box) == "amber"


def test_all_lamps_dark_is_unknown():
    img = np.zeros((200, 200, 3), np.uint8)
    assert classify_light(img, (50, 50, 80, 140)) == "unknown"


def test_sign_reader_calls_ocr_once_per_track_id(monkeypatch):
    r = SignReader()
    calls = []

    def fake(crop):
        calls.append(1)
        return 30

    monkeypatch.setattr(r, "_ocr_number", fake)
    img = np.zeros((200, 200, 3), np.uint8)
    box = (50, 50, 100, 100)

    assert r.read_speed_limit(img, box, track_id=7) == 30
    assert r.read_speed_limit(img, box, track_id=7) == 30
    assert r.read_speed_limit(img, box, track_id=7) == 30
    assert len(calls) == 1, "OCR must be cached per track id"


def test_sign_reader_skips_tiny_boxes():
    r = SignReader(min_box_px=22)
    img = np.zeros((200, 200, 3), np.uint8)
    assert r.read_speed_limit(img, (50, 50, 60, 60), track_id=1) is None


def test_read_signals_keeps_previous_limit_when_no_sign_seen():
    img = np.zeros((200, 200, 3), np.uint8)
    prev = SignalInfo(speed_limit_kmh=80)
    out = read_signals(img, [], SignReader(), prev, _cfg())
    assert out.speed_limit_kmh == 80
    assert out.light_state == "none"


def test_read_signals_picks_the_largest_traffic_light():
    """The nearest light is the one that governs us; nearest == biggest box."""
    img = np.zeros((300, 300, 3), np.uint8)
    img[20:36, 107:123] = (0, 0, 255)         # small light, red
    img[100:140, 200:240] = (0, 255, 0)       # big light, green lamp low
    tracks = [
        Track(id=1, cls="traffic_light", bbox=(100, 10, 130, 100), conf=0.9),
        Track(id=2, cls="traffic_light", bbox=(190, 60, 250, 240), conf=0.9),
    ]
    out = read_signals(img, tracks, SignReader(), SignalInfo(), _cfg())
    assert out.light_state == "green"


def test_stop_sign_is_recorded():
    img = np.zeros((300, 300, 3), np.uint8)
    tracks = [Track(id=5, cls="stop_sign", bbox=(10, 10, 60, 60), conf=0.9)]
    out = read_signals(img, tracks, SignReader(), SignalInfo(), _cfg())
    assert any(s.kind == "stop" for s in out.signs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_signals.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.signals'`

- [ ] **Step 3: Write `src/signals.py`**

```python
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
```

Note on class coverage: COCO has `stop sign` but no speed-limit class, so circular
speed-limit plates usually arrive labelled `stop_sign` by YOLO. We therefore run
OCR on every `stop_sign` box and let the digits decide: digits found means it is a
speed-limit plate, no digits means it is a real stop sign. This is a deliberate,
documented compromise for the four-day budget; a dedicated traffic-sign detector is
listed as future work in the spec.

- [ ] **Step 4: Add the `signals` config block**

Add to `config.yaml`:

```yaml
signals:
  min_sign_px: 22
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_signals.py -v`
Expected: 9 passed

- [ ] **Step 6: Add the `__main__` demo and verify visually**

Append to `src/signals.py`:

```python
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
```

Run: `.venv/bin/python -m src.signals --video data/input/clear.mp4`
Expected: `LIGHT` flips to `RED`/`GREEN` when the clip passes a signal, and `ZONE`
updates when a speed plate is legible. If lights read `UNKNOWN` constantly, widen the
HSV `_RED`/`_GREEN` saturation floors — small distant lights are often desaturated.

- [ ] **Step 7: Commit**

```bash
git add src/signals.py tests/test_signals.py config.yaml
git commit -m "feat: traffic light state, stop signs, cached speed-limit OCR"
```

---

### Task 8: `src/bev.py` — inverse perspective mapping and occupancy grid

**Files:**
- Create: `src/bev.py`
- Test: `tests/test_bev.py`

**Interfaces:**
- Consumes: `src.types.BEVGrid`, `src.types.Track`, `src.detect.bbox_bottom_centre`.
- Produces:
  - `build_homography(cfg) -> np.ndarray` — 3×3, maps 640×360 image pixels to ground metres `(x_lateral, y_forward)`.
  - `image_to_ground(H, pts_px) -> np.ndarray` — `(N,2)` metres.
  - `ground_to_image(H, pts_m) -> np.ndarray` — `(N,2)` pixels.
  - `track_distance(H, bbox, frame_size) -> float` — forward metres to a box's ground contact point.
  - `annotate_distances(tracks, H, frame_size, cfg) -> None` — sets `.dist_m`, `.lateral_m` and `.in_path` in place.
  - `build_grid(state, cfg) -> BEVGrid`.

**Background:** the ground plane is assumed flat. Four image points forming a trapezoid on the road (`camera.ipm_src`) are mapped to four known ground points (`camera.ipm_dst_m`), which fixes the homography without needing camera intrinsics. `ipm_dst_m` is expressed as ±half the lane width laterally at 30 m and 5 m ahead.

- [ ] **Step 1: Write the failing test**

`tests/test_bev.py`:

```python
import numpy as np
import pytest

from src.bev import (
    annotate_distances,
    build_grid,
    build_homography,
    ground_to_image,
    image_to_ground,
    track_distance,
)
from src.types import FrameState, Track


def _cfg():
    return {
        "camera": {
            "seg_w": 640, "seg_h": 360,
            "ipm_src": [[250, 200], [390, 200], [610, 340], [30, 340]],
            "ipm_dst_m": [[-1.75, 30.0], [1.75, 30.0], [1.75, 5.0], [-1.75, 5.0]],
            "lane_width_m": 3.5,
            "wheelbase_m": 2.7,
        },
        "bev": {"range_m": 60.0, "width_m": 24.0, "m_per_px": 0.15, "corridor_m": 1.9},
    }


def test_homography_maps_calibration_points_exactly():
    cfg = _cfg()
    H = build_homography(cfg)
    got = image_to_ground(H, np.array(cfg["camera"]["ipm_src"], np.float32))
    want = np.array(cfg["camera"]["ipm_dst_m"], np.float32)
    assert np.allclose(got, want, atol=1e-3)


def test_ground_to_image_is_the_inverse():
    cfg = _cfg()
    H = build_homography(cfg)
    pts = np.array([[0.0, 10.0], [1.5, 25.0], [-1.5, 8.0]], np.float32)
    back = image_to_ground(H, ground_to_image(H, pts))
    assert np.allclose(back, pts, atol=1e-2)


def test_nearer_boxes_report_smaller_distance():
    """A box lower in the frame is closer to the camera."""
    cfg = _cfg()
    H = build_homography(cfg)
    near = track_distance(H, (300, 250, 360, 330), (640, 360))
    far = track_distance(H, (300, 190, 340, 215), (640, 360))
    assert 0 < near < far


def test_distance_is_scaled_from_full_resolution_frames():
    """Boxes come from the full-res frame; the homography is calibrated at 640x360."""
    cfg = _cfg()
    H = build_homography(cfg)
    d_small = track_distance(H, (300, 250, 360, 330), (640, 360))
    d_big = track_distance(H, (600, 500, 720, 660), (1280, 720))
    assert d_small == pytest.approx(d_big, rel=0.02)


def test_annotate_marks_objects_in_the_ego_corridor():
    cfg = _cfg()
    H = build_homography(cfg)
    ahead = Track(id=1, cls="car", bbox=(300, 240, 360, 320), conf=0.9)
    beside = Track(id=2, cls="car", bbox=(10, 240, 70, 320), conf=0.9)
    annotate_distances([ahead, beside], H, (640, 360), cfg)

    assert np.isfinite(ahead.dist_m) and ahead.in_path is True
    assert abs(ahead.lateral_m) < cfg["bev"]["corridor_m"]
    assert beside.in_path is False
    assert np.isfinite(beside.lateral_m) and abs(beside.lateral_m) > cfg["bev"]["corridor_m"]


def test_build_grid_marks_an_object_cell():
    cfg = _cfg()
    img = np.zeros((360, 640, 3), np.uint8)
    s = FrameState(idx=0, t=0.0, raw=img, img=img)
    s.objects = [Track(id=1, cls="car", bbox=(300, 240, 360, 320), conf=0.9,
                       dist_m=15.0, lateral_m=0.2, in_path=True)]
    g = build_grid(s, cfg)
    assert g.grid is not None
    assert g.grid.sum() > 0
    assert g.m_per_px == pytest.approx(0.15)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_bev.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.bev'`

- [ ] **Step 3: Add the `bev` config block**

Add to `config.yaml`:

```yaml
bev:
  range_m: 60.0        # how far ahead the top-down view extends
  width_m: 24.0        # total lateral width of the view
  m_per_px: 0.15
  corridor_m: 1.9      # half-width of the ego lane corridor
```

- [ ] **Step 4: Write `src/bev.py`**

```python
"""Bird's-eye view: turning image pixels into metres on the ground.

Everything the planner reasons about - how far is that car, is it in my
lane, is there room to change lanes - is a question about the ground plane,
not about pixels. One homography, calibrated from four road points with
known real-world positions, converts between the two. No camera intrinsics
and no depth network required; the price is an assumption that the road is
flat, which is fine for the highway and city clips used here.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.detect import bbox_bottom_centre
from src.types import BEVGrid, FrameState, Track


def build_homography(cfg: dict) -> np.ndarray:
    cam = cfg["camera"]
    src = np.array(cam["ipm_src"], np.float32)
    dst = np.array(cam["ipm_dst_m"], np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def image_to_ground(H: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_px, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def ground_to_image(H: np.ndarray, pts_m: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_m, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)


def _to_seg_scale(
    pt: tuple[float, float], frame_size: tuple[int, int], cfg: dict
) -> tuple[float, float]:
    """Boxes arrive in full-frame pixels; the homography is calibrated at 640x360."""
    fw, fh = frame_size
    return (pt[0] * cfg["camera"]["seg_w"] / fw, pt[1] * cfg["camera"]["seg_h"] / fh)


def track_distance(
    H: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_size: tuple[int, int],
    cfg: dict | None = None,
) -> float:
    """Forward distance in metres to where this object touches the road."""
    cfg = cfg or {"camera": {"seg_w": 640, "seg_h": 360}}
    px = _to_seg_scale(bbox_bottom_centre(bbox), frame_size, cfg)
    x_m, y_m = image_to_ground(H, np.array([px], np.float32))[0]
    if not np.isfinite(y_m) or y_m <= 0:
        return float("inf")
    return float(y_m)


def annotate_distances(
    tracks: list[Track], H: np.ndarray, frame_size: tuple[int, int], cfg: dict
) -> None:
    """Fill in `dist_m` and `in_path` for every track, in place."""
    corridor = cfg["bev"]["corridor_m"]
    for t in tracks:
        px = _to_seg_scale(bbox_bottom_centre(t.bbox), frame_size, cfg)
        x_m, y_m = image_to_ground(H, np.array([px], np.float32))[0]
        if not np.isfinite(y_m) or y_m <= 0:
            t.dist_m, t.lateral_m, t.in_path = float("inf"), float("nan"), False
            continue
        t.dist_m = float(y_m)
        t.lateral_m = float(x_m)
        t.in_path = abs(float(x_m)) < corridor


def build_grid(state: FrameState, cfg: dict) -> BEVGrid:
    """Rasterise tracked objects into an ego-centric top-down occupancy grid.

    The ego sits at the bottom-centre, looking up the image.
    """
    b = cfg["bev"]
    mpp = b["m_per_px"]
    gh, gw = int(b["range_m"] / mpp), int(b["width_m"] / mpp)
    grid = np.zeros((gh, gw), np.uint8)
    origin = (gw // 2, gh - 1)                      # ego rear axle, in grid pixels

    for t in state.objects:
        if not np.isfinite(t.dist_m) or not np.isfinite(t.lateral_m):
            continue
        px = int(origin[0] + t.lateral_m / mpp)
        py = int(origin[1] - t.dist_m / mpp)
        if 0 <= px < gw and 0 <= py < gh:
            r = max(2, int(1.8 / mpp / 2))
            cv2.circle(grid, (px, py), r, 255, -1)

    return BEVGrid(grid=grid, m_per_px=mpp, origin_px=origin)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_bev.py -v`
Expected: 6 passed

- [ ] **Step 6: Calibrate `ipm_src` against a real clip**

This is the single most fiddly number in the project, and the defaults are a guess.
Append this calibration helper to `src/bev.py`:

```python
if __name__ == "__main__":
    import argparse

    import yaml

    from src.video_io import VideoReader

    ap = argparse.ArgumentParser(
        description="Block 6 demo / IPM calibration. Check the trapezoid hugs the lane."
    )
    ap.add_argument("--video", required=True)
    a = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    H = build_homography(cfg)
    src = np.array(cfg["camera"]["ipm_src"], np.int32)

    for _, _, frame in VideoReader(a.video):
        vis = cv2.resize(frame, (cfg["camera"]["seg_w"], cfg["camera"]["seg_h"]))
        cv2.polylines(vis, [src], True, (0, 255, 255), 2)
        for (x, y), (gx, gy) in zip(src, cfg["camera"]["ipm_dst_m"]):
            cv2.putText(vis, f"({gx},{gy})m", (x - 30, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        # Distance ruler along the lane centre.
        for d in (5, 10, 20, 30, 40):
            p = ground_to_image(H, np.array([[0.0, float(d)]], np.float32))[0]
            if 0 <= p[0] < vis.shape[1] and 0 <= p[1] < vis.shape[0]:
                cv2.line(vis, (int(p[0]) - 25, int(p[1])), (int(p[0]) + 25, int(p[1])),
                         (255, 255, 0), 1)
                cv2.putText(vis, f"{d}m", (int(p[0]) + 30, int(p[1]) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.imshow("bev calib", vis)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
```

Run: `.venv/bin/python -m src.bev --video data/input/clear.mp4`

Adjust `camera.ipm_src` in `config.yaml` until:
- the yellow trapezoid's bottom edge sits on the road just in front of the bonnet
- its top edge sits on the lane about 30 m ahead
- its left and right edges follow the ego lane markings
- the cyan distance rulers land plausibly (a car 3 lengths ahead should read ~15 m)

Metric accuracy is not critical — consistency is. Note in the report that distances
are estimated under a flat-ground assumption without camera calibration.

- [ ] **Step 7: Commit**

```bash
git add src/bev.py tests/test_bev.py config.yaml
git commit -m "feat: IPM homography, distance estimation, BEV occupancy grid"
```

---

### Task 9: `src/control.py` — pure pursuit and PID

**Files:**
- Create: `src/control.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Consumes: `src.types.Control`, `src.types.LaneInfo`.
- Produces:
  - `pure_pursuit(offset_m, heading_err_rad, lookahead_m, wheelbase_m, max_steer_deg) -> float` (degrees, `+` = right).
  - `lookahead_for(speed_kmh, cfg) -> float`.
  - `PID(kp, ki, kd, out_min=-1.0, out_max=1.0)` with `.step(error, dt) -> float` and `.reset()`.
  - `longitudinal(target_kmh, current_kmh, pid, dt) -> tuple[float, float]` returning `(throttle, brake)`.

**Sign convention, used consistently across the project:** `offset_m > 0` means the ego is to the RIGHT of the lane centre, so the correct response is to steer LEFT, i.e. a NEGATIVE steering angle.

- [ ] **Step 1: Write the failing test**

`tests/test_control.py`:

```python
import numpy as np
import pytest

from src.control import PID, longitudinal, lookahead_for, pure_pursuit


def _cfg():
    return {"control": {"lookahead_min_m": 6.0, "lookahead_k": 0.6, "max_steer_deg": 35.0}}


def test_perfectly_centred_means_no_steering():
    assert pure_pursuit(0.0, 0.0, 10.0, 2.7, 35.0) == pytest.approx(0.0, abs=1e-6)


def test_ego_right_of_centre_steers_left():
    assert pure_pursuit(1.0, 0.0, 10.0, 2.7, 35.0) < 0


def test_ego_left_of_centre_steers_right():
    assert pure_pursuit(-1.0, 0.0, 10.0, 2.7, 35.0) > 0


def test_steering_is_antisymmetric():
    a = pure_pursuit(0.8, 0.1, 10.0, 2.7, 35.0)
    b = pure_pursuit(-0.8, -0.1, 10.0, 2.7, 35.0)
    assert a == pytest.approx(-b, abs=1e-9)


def test_larger_offset_demands_more_steering():
    assert abs(pure_pursuit(2.0, 0.0, 10.0, 2.7, 35.0)) > abs(
        pure_pursuit(0.5, 0.0, 10.0, 2.7, 35.0)
    )


def test_longer_lookahead_softens_the_correction():
    assert abs(pure_pursuit(1.0, 0.0, 20.0, 2.7, 35.0)) < abs(
        pure_pursuit(1.0, 0.0, 6.0, 2.7, 35.0)
    )


def test_steering_is_clamped():
    assert pure_pursuit(50.0, 1.5, 3.0, 2.7, 35.0) == pytest.approx(-35.0)
    assert pure_pursuit(-50.0, -1.5, 3.0, 2.7, 35.0) == pytest.approx(35.0)


def test_lookahead_grows_with_speed():
    slow, fast = lookahead_for(0.0, _cfg()), lookahead_for(108.0, _cfg())
    assert slow == pytest.approx(6.0)
    assert fast == pytest.approx(6.0 + 0.6 * 30.0)      # 108 km/h == 30 m/s


def test_pid_proportional_response():
    p = PID(kp=0.5, ki=0.0, kd=0.0)
    assert p.step(1.0, 0.1) == pytest.approx(0.5)


def test_pid_integral_accumulates():
    p = PID(kp=0.0, ki=1.0, kd=0.0)
    p.step(1.0, 0.5)
    assert p.step(1.0, 0.5) == pytest.approx(1.0)


def test_pid_output_is_clamped():
    p = PID(kp=10.0, ki=0.0, kd=0.0, out_min=-1.0, out_max=1.0)
    assert p.step(5.0, 0.1) == 1.0
    assert p.step(-5.0, 0.1) == -1.0


def test_pid_integral_does_not_wind_up_while_saturated():
    """A long saturated stretch must not leave a huge integral behind."""
    p = PID(kp=1.0, ki=5.0, kd=0.0, out_min=-1.0, out_max=1.0)
    for _ in range(50):
        p.step(10.0, 0.1)
    assert abs(p.integral) < 1.0


def test_pid_reset_clears_state():
    p = PID(kp=1.0, ki=1.0, kd=1.0)
    p.step(3.0, 0.1)
    p.reset()
    assert p.integral == 0.0 and p.prev_error == 0.0


def test_speeding_up_uses_throttle_only():
    thr, brk = longitudinal(50.0, 30.0, PID(0.9, 0.12, 0.05), 0.1)
    assert thr > 0 and brk == 0.0


def test_slowing_down_uses_brake_only():
    thr, brk = longitudinal(20.0, 60.0, PID(0.9, 0.12, 0.05), 0.1)
    assert brk > 0 and thr == 0.0


def test_outputs_stay_in_unit_range():
    for target, current in ((0.0, 200.0), (200.0, 0.0), (50.0, 50.0)):
        thr, brk = longitudinal(target, current, PID(0.9, 0.12, 0.05), 0.1)
        assert 0.0 <= thr <= 1.0 and 0.0 <= brk <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_control.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.control'`

- [ ] **Step 3: Write `src/control.py`**

```python
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
    """
    u = pid.step(target_kmh - current_kmh, dt)
    if u >= 0:
        return float(np.clip(u, 0.0, 1.0)), 0.0
    return 0.0, float(np.clip(-u, 0.0, 1.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_control.py -v`
Expected: 16 passed

- [ ] **Step 5: Add the `__main__` demo**

Append to `src/control.py`:

```python
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
```

Run: `.venv/bin/python -m src.control`
Expected: the steering table is antisymmetric about zero offset, and the PID converges toward 50 km/h without wild oscillation. If it overshoots badly, lower `control.pid.kp` in `config.yaml`.

- [ ] **Step 6: Commit**

```bash
git add src/control.py tests/test_control.py
git commit -m "feat: pure pursuit steering and PID speed control"
```

---

### Task 10: `src/planner.py` — the driving finite state machine

**Files:**
- Create: `src/planner.py`
- Test: `tests/test_planner.py`

**Interfaces:**
- Consumes: `src.types.FrameState`, `src.types.Decision`, `src.weather.speed_cap_kmh`, `src.detect.VEHICLE`, `src.detect.VULNERABLE`.
- Produces:
  - `degrade_level(state, cfg) -> int` — `0` lanes good, `1` free-space fallback, `2` blind.
  - `Planner(cfg)` with `.step(state: FrameState) -> Decision`.
  - `STATES: tuple[str, ...]` — the full state list, for the HUD colour map.

**FSM priority order** (first match wins):

| Priority | State | Trigger |
|---|---|---|
| 1 | `EMERGENCY_BRAKE` | any in-path object with `ttc_s < planner.ebrake_ttc_s` |
| 2 | `YIELD_PEDESTRIAN` | in-path person/animal closer than `planner.yield_ped_dist_m` |
| 3 | `STOP_SIGNAL` | traffic light is `red`, `amber`, or `unknown` |
| 4 | `CHANGING` | a lane change is already underway |
| 5 | `PREP_CHANGE_L` / `PREP_CHANGE_R` | indicator asserted, waiting out `indicator_lead_frames` |
| 6 | `FOLLOW` | in-path lead vehicle inside the time-gap threshold |
| 7 | `LANE_KEEP` | otherwise |

- [ ] **Step 1: Write the failing test**

`tests/test_planner.py`:

```python
import numpy as np
import pytest

from src.planner import Planner, degrade_level
from src.types import FrameState, LaneInfo, SignalInfo, Track


def _cfg():
    return {
        "planner": {
            "follow_time_gap_s": 2.0,
            "lane_change_gap_s": 3.0,
            "indicator_lead_frames": 3,
            "ebrake_ttc_s": 1.5,
            "yield_ped_dist_m": 25.0,
            "stop_line_dist_m": 20.0,
            "overtake_side": "left",
            "follow_frames_before_change": 5,
            "change_frames": 4,
            "lat_accel_max": 3.0,
        },
        "weather": {
            "speed_cap_kmh": {"clear": 999, "rain": 60, "fog": 40, "night": 70, "snow": 40},
            "visibility_lowvis_below": 0.45,
        },
        "degrade": {
            "freespace_speed_cap_kmh": 40,
            "blind_speed_cap_kmh": 20,
            "lowvis_gap_multiplier": 1.8,
        },
        "bev": {"corridor_m": 1.9},
    }


def _state(**kw) -> FrameState:
    img = np.zeros((360, 640, 3), np.uint8)
    s = FrameState(idx=kw.pop("idx", 0), t=kw.pop("t", 0.0), raw=img, img=img)
    s.lanes = LaneInfo(valid=True, offset_m=0.0)
    s.drivable = np.ones((360, 640), np.uint8)
    s.signals = SignalInfo(speed_limit_kmh=50)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _car(dist, lateral=0.0, ttc=float("inf"), cls="car"):
    return Track(id=1, cls=cls, bbox=(300, 200, 360, 300), conf=0.9,
                 dist_m=dist, lateral_m=lateral, ttc_s=ttc,
                 in_path=abs(lateral) < 1.9)


# --- degradation ladder ---

def test_degrade_level_zero_when_lanes_valid():
    assert degrade_level(_state(), _cfg()) == 0


def test_degrade_level_one_when_lanes_lost_but_road_visible():
    s = _state()
    s.lanes = LaneInfo(valid=False)
    assert degrade_level(s, _cfg()) == 1


def test_degrade_level_two_when_nothing_is_visible():
    s = _state()
    s.lanes = LaneInfo(valid=False)
    s.drivable = np.zeros((360, 640), np.uint8)
    assert degrade_level(s, _cfg()) == 2


# --- FSM priorities ---

def test_imminent_collision_triggers_emergency_brake():
    s = _state(objects=[_car(8.0, ttc=0.9)])
    d = Planner(_cfg()).step(s)
    assert d.fsm_state == "EMERGENCY_BRAKE"
    assert d.target_speed == 0.0


def test_pedestrian_ahead_triggers_yield():
    s = _state(objects=[_car(15.0, cls="person")])
    assert Planner(_cfg()).step(s).fsm_state == "YIELD_PEDESTRIAN"


def test_red_light_triggers_stop():
    s = _state()
    s.signals.light_state = "red"
    d = Planner(_cfg()).step(s)
    assert d.fsm_state == "STOP_SIGNAL"
    assert d.target_speed == 0.0


def test_green_light_does_not_stop():
    s = _state()
    s.signals.light_state = "green"
    assert Planner(_cfg()).step(s).fsm_state == "LANE_KEEP"


def test_emergency_brake_outranks_a_green_light():
    s = _state(objects=[_car(6.0, ttc=0.5)])
    s.signals.light_state = "green"
    assert Planner(_cfg()).step(s).fsm_state == "EMERGENCY_BRAKE"


def test_close_lead_vehicle_triggers_follow():
    s = _state(objects=[_car(12.0)])
    assert Planner(_cfg()).step(s).fsm_state == "FOLLOW"


def test_distant_lead_vehicle_does_not_trigger_follow():
    s = _state(objects=[_car(200.0)])
    assert Planner(_cfg()).step(s).fsm_state == "LANE_KEEP"


def test_empty_road_is_lane_keep():
    assert Planner(_cfg()).step(_state()).fsm_state == "LANE_KEEP"


# --- lane change sequence ---

def test_sustained_follow_with_clear_left_lane_starts_a_lane_change():
    p, cfg = Planner(_cfg()), _cfg()
    d = None
    for i in range(cfg["planner"]["follow_frames_before_change"] + 1):
        d = p.step(_state(idx=i, objects=[_car(12.0)]))
    assert d.fsm_state == "PREP_CHANGE_L"
    assert d.indicator == "left"


def test_indicator_precedes_lateral_movement():
    """The indicator must be on for indicator_lead_frames BEFORE CHANGING."""
    p, cfg = Planner(_cfg()), _cfg()
    states = []
    for i in range(20):
        states.append(p.step(_state(idx=i, objects=[_car(12.0)])))

    prep = [i for i, d in enumerate(states) if d.fsm_state.startswith("PREP_CHANGE")]
    changing = [i for i, d in enumerate(states) if d.fsm_state == "CHANGING"]
    assert prep and changing
    assert min(changing) - min(prep) >= cfg["planner"]["indicator_lead_frames"]
    assert all(states[i].indicator == "left" for i in prep)


def test_occupied_left_lane_blocks_the_change():
    p, cfg = Planner(_cfg()), _cfg()
    d = None
    for i in range(cfg["planner"]["follow_frames_before_change"] + 3):
        d = p.step(_state(idx=i, objects=[_car(12.0), _car(10.0, lateral=-3.4)]))
    assert d.fsm_state == "FOLLOW"
    assert d.indicator == "off"


def test_indicator_clears_after_the_change_completes():
    p = Planner(_cfg())
    d = None
    for i in range(40):
        d = p.step(_state(idx=i, objects=[_car(12.0)]))
    # After the manoeuvre finishes the indicator must be off again.
    for i in range(40, 60):
        d = p.step(_state(idx=i))
    assert d.indicator == "off"
    assert d.fsm_state == "LANE_KEEP"


# --- target speed ---

def test_speed_limit_is_respected():
    s = _state()
    s.signals.speed_limit_kmh = 30
    assert Planner(_cfg()).step(s).target_speed == pytest.approx(30.0)


def test_fog_caps_speed_below_the_posted_limit():
    s = _state()
    s.signals.speed_limit_kmh = 100
    s.weather.label = "fog"
    assert Planner(_cfg()).step(s).target_speed == pytest.approx(40.0)


def test_lost_lanes_cap_speed_to_the_freespace_limit():
    s = _state()
    s.signals.speed_limit_kmh = 100
    s.lanes = LaneInfo(valid=False)
    assert Planner(_cfg()).step(s).target_speed == pytest.approx(40.0)


def test_blind_perception_caps_speed_hardest():
    s = _state()
    s.signals.speed_limit_kmh = 100
    s.lanes = LaneInfo(valid=False)
    s.drivable = np.zeros((360, 640), np.uint8)
    assert Planner(_cfg()).step(s).target_speed == pytest.approx(20.0)


def test_tight_curve_caps_speed():
    s = _state()
    s.signals.speed_limit_kmh = 100
    s.lanes = LaneInfo(valid=True, offset_m=0.0, curvature_m=50.0)
    # v = sqrt(3.0 * 50) m/s = 12.25 m/s = 44.1 km/h
    assert Planner(_cfg()).step(s).target_speed == pytest.approx(44.1, abs=1.0)


def test_decisions_are_logged():
    s = _state()
    s.signals.speed_limit_kmh = 30
    assert Planner(_cfg()).step(s).log, "planner must explain itself in the log"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_planner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.planner'`

- [ ] **Step 3: Add the planner config keys**

Add these three keys under `planner:` in `config.yaml`:

```yaml
  follow_frames_before_change: 20   # sustained following before considering an overtake
  change_frames: 25                 # how long the lateral manoeuvre takes
  lat_accel_max: 3.0                # m/s^2, sets the cornering speed cap
```

- [ ] **Step 4: Write `src/planner.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_planner.py -v`
Expected: 20 passed

Note: `test_indicator_precedes_lateral_movement` uses the test config's
`follow_frames_before_change: 5`, not the production value of 20. If it fails,
check that `_prep_frames` starts at 1 when the manoeuvre is decided, so the
first PREP frame is counted.

- [ ] **Step 6: Commit**

```bash
git add src/planner.py tests/test_planner.py config.yaml
git commit -m "feat: driving FSM with lane-change rules and degradation caps"
```

---

### Task 11: `src/sim.py` — bicycle model and top-down simulator window

**Files:**
- Create: `src/sim.py`
- Test: `tests/test_sim.py`

**Interfaces:**
- Consumes: `src.types.FrameState`, `src.types.BEVGrid`.
- Produces:
  - `EgoState(x_m: float, y_m: float, yaw_rad: float, v_kmh: float)` dataclass.
  - `bicycle_step(ego, steer_deg, throttle, brake, dt, cfg) -> EgoState` — returns a NEW state.
  - `render_sim(state, ego, cfg) -> np.ndarray` — top-down BGR image.

**Why this matters:** the video pipeline is open-loop — a steering command cannot move a recorded camera. This module is the only place where commands actually drive something, so it is what makes the closed-loop claim honest.

- [ ] **Step 1: Write the failing test**

`tests/test_sim.py`:

```python
import numpy as np
import pytest

from src.sim import EgoState, bicycle_step, render_sim
from src.types import BEVGrid, FrameState


def _cfg():
    return {
        "camera": {"wheelbase_m": 2.7},
        "control": {"accel_max": 2.5, "decel_max": 5.0, "max_steer_deg": 35.0},
        "bev": {"range_m": 60.0, "width_m": 24.0, "m_per_px": 0.15, "corridor_m": 1.9},
    }


def test_step_returns_a_new_object():
    e = EgoState(0.0, 0.0, 0.0, 36.0)
    out = bicycle_step(e, 0.0, 0.0, 0.0, 0.1, _cfg())
    assert out is not e
    assert e.y_m == 0.0, "input state must not be mutated"


def test_straight_driving_advances_forward_only():
    """36 km/h = 10 m/s, so 1 second of straight driving covers 10 m."""
    e = EgoState(0.0, 0.0, 0.0, 36.0)
    for _ in range(10):
        e = bicycle_step(e, 0.0, 0.0, 0.0, 0.1, _cfg())
    assert e.y_m == pytest.approx(10.0, rel=0.02)
    assert e.x_m == pytest.approx(0.0, abs=1e-9)
    assert e.yaw_rad == pytest.approx(0.0, abs=1e-9)


def test_right_steer_curves_right():
    e = EgoState(0.0, 0.0, 0.0, 36.0)
    for _ in range(10):
        e = bicycle_step(e, 10.0, 0.0, 0.0, 0.1, _cfg())
    assert e.x_m > 0 and e.yaw_rad > 0


def test_left_steer_curves_left():
    e = EgoState(0.0, 0.0, 0.0, 36.0)
    for _ in range(10):
        e = bicycle_step(e, -10.0, 0.0, 0.0, 0.1, _cfg())
    assert e.x_m < 0 and e.yaw_rad < 0


def test_turn_radius_matches_the_bicycle_model():
    """Steady steer delta gives radius R = L / tan(delta)."""
    cfg = _cfg()
    delta, L = 20.0, cfg["camera"]["wheelbase_m"]
    expected_R = L / np.tan(np.radians(delta))

    e = EgoState(0.0, 0.0, 0.0, 18.0)      # 5 m/s
    dt = 0.02
    # Integrate a quarter turn, then compare arc length to R * (pi/2).
    dist = 0.0
    while e.yaw_rad < np.pi / 2:
        e = bicycle_step(e, delta, 0.0, 0.0, dt, cfg)
        dist += (e.v_kmh / 3.6) * dt
    assert dist / (np.pi / 2) == pytest.approx(expected_R, rel=0.03)


def test_throttle_accelerates_and_brake_decelerates():
    cfg = _cfg()
    up = bicycle_step(EgoState(0, 0, 0, 20.0), 0.0, 1.0, 0.0, 1.0, cfg)
    assert up.v_kmh == pytest.approx(20.0 + 2.5 * 3.6, rel=0.01)

    down = bicycle_step(EgoState(0, 0, 0, 40.0), 0.0, 0.0, 1.0, 1.0, cfg)
    assert down.v_kmh == pytest.approx(40.0 - 5.0 * 3.6, rel=0.01)


def test_speed_never_goes_negative():
    e = bicycle_step(EgoState(0, 0, 0, 5.0), 0.0, 0.0, 1.0, 5.0, _cfg())
    assert e.v_kmh == 0.0


def test_steer_input_is_clamped():
    a = bicycle_step(EgoState(0, 0, 0, 36.0), 200.0, 0.0, 0.0, 0.1, _cfg())
    b = bicycle_step(EgoState(0, 0, 0, 36.0), 35.0, 0.0, 0.0, 0.1, _cfg())
    assert a.yaw_rad == pytest.approx(b.yaw_rad)


def test_render_sim_produces_an_image():
    img = np.zeros((360, 640, 3), np.uint8)
    s = FrameState(idx=0, t=0.0, raw=img, img=img)
    cfg = _cfg()
    gh = int(cfg["bev"]["range_m"] / cfg["bev"]["m_per_px"])
    gw = int(cfg["bev"]["width_m"] / cfg["bev"]["m_per_px"])
    s.bev = BEVGrid(grid=np.zeros((gh, gw), np.uint8), m_per_px=0.15,
                    origin_px=(gw // 2, gh - 1))
    out = render_sim(s, EgoState(0.0, 0.0, 0.0, 40.0), cfg)
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8
    assert out.sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_sim.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.sim'`

- [ ] **Step 3: Write `src/sim.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_sim.py -v`
Expected: 9 passed

- [ ] **Step 5: Sanity-check the model**

Run: `.venv/bin/python -m src.sim`
Expected: `x` and `yaw` both grow positive, and the printed turn radius is ~10 m for 15° at a 2.7 m wheelbase.

- [ ] **Step 6: Commit**

```bash
git add src/sim.py tests/test_sim.py
git commit -m "feat: kinematic bicycle model and top-down simulator view"
```

---

### Task 12: `src/pipeline.py` and `run.py` — wire everything together

**Files:**
- Create: `src/pipeline.py`, `run.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: every module built so far.
- Produces:
  - `Pipeline(cfg: dict, use_clip: bool = True)` with:
    - `.process(idx: int, t: float, frame: np.ndarray) -> FrameState`
    - `.run(video_path, out_path, log_path, sim=False, max_frames=None) -> dict` returning summary stats.
  - `run.py` CLI: `--video --out --log --sim --frames --stride --no-clip`.

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`. This is an integration test over a synthetic clip, so it exercises the real wiring without needing footage.

```python
import json

import cv2
import numpy as np
import pytest
import yaml

from src.pipeline import Pipeline


@pytest.fixture
def cfg():
    return yaml.safe_load(open("config.yaml"))


@pytest.fixture
def synthetic_clip(tmp_path):
    """A crude 12-frame road scene: grey road, two white lane lines, a dark box."""
    p = tmp_path / "synth.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 360))
    for i in range(12):
        f = np.full((360, 640, 3), 120, np.uint8)
        f[:180] = (170, 150, 130)                     # sky
        cv2.line(f, (250 + i, 200), (150 + i, 359), (240, 240, 240), 6)
        cv2.line(f, (390 + i, 200), (490 + i, 359), (240, 240, 240), 6)
        cv2.rectangle(f, (300, 210), (360, 260), (40, 40, 40), -1)
        w.write(f)
    w.release()
    return str(p)


def test_process_returns_a_fully_populated_state(cfg, synthetic_clip):
    from src.video_io import VideoReader

    pipe = Pipeline(cfg, use_clip=False)
    idx, t, frame = next(iter(VideoReader(synthetic_clip)))
    s = pipe.process(idx, t, frame)

    assert s.drivable is not None and s.lane_mask is not None
    assert s.decision.fsm_state in (
        "LANE_KEEP", "FOLLOW", "PREP_CHANGE_L", "PREP_CHANGE_R", "CHANGING",
        "STOP_SIGNAL", "YIELD_PEDESTRIAN", "EMERGENCY_BRAKE",
    )
    assert np.isfinite(s.control.steer_deg)
    assert 0.0 <= s.control.throttle <= 1.0
    assert 0.0 <= s.control.brake <= 1.0
    assert s.bev.grid is not None


def test_run_writes_video_and_log(cfg, synthetic_clip, tmp_path):
    out, log = tmp_path / "out.mp4", tmp_path / "log.jsonl"
    pipe = Pipeline(cfg, use_clip=False)
    stats = pipe.run(synthetic_clip, str(out), str(log), sim=False, max_frames=6)

    assert out.exists() and out.stat().st_size > 0
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 6
    assert stats["frames"] == 6
    assert stats["mean_ms"] > 0
    for r in rows:
        assert {"steer_deg", "throttle", "brake", "fsm_state", "events"} <= set(r)


def test_pipeline_survives_a_black_video(cfg, tmp_path):
    """Total perception failure must degrade, not crash."""
    p = tmp_path / "black.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 360))
    for _ in range(4):
        w.write(np.zeros((360, 640, 3), np.uint8))
    w.release()

    pipe = Pipeline(cfg, use_clip=False)
    stats = pipe.run(str(p), str(tmp_path / "o.mp4"), str(tmp_path / "l.jsonl"))
    assert stats["frames"] == 4

    rows = [json.loads(l) for l in (tmp_path / "l.jsonl").read_text().splitlines()]
    # Lane paint cannot exist in a black frame, so the lane fit must fail...
    assert all(r["lane_valid"] is False for r in rows)
    # ...and the degradation ladder must cap speed. Whether the segmenter reports a
    # spurious drivable region (level 1) or nothing (level 2) is model-dependent, so
    # accept the looser of the two caps.
    assert all(
        r["target_speed_kmh"] <= cfg["degrade"]["freespace_speed_cap_kmh"] for r in rows
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline'`

- [ ] **Step 3: Write `src/pipeline.py`**

```python
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
```

- [ ] **Step 4: Write `run.py`**

```python
#!/usr/bin/env python
"""All-weather self-driving simulation — command line entry point.

Examples:
    python run.py --video data/input/clear.mp4
    python run.py --video data/input/rain.mp4 --sim
    python run.py --video data/input/clear.mp4 --frames 40 --no-clip
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.pipeline import Pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="input dashcam clip")
    ap.add_argument("--out", default=None, help="output mp4 (default: data/output/<stem>_annotated.mp4)")
    ap.add_argument("--log", default=None, help="output jsonl (default: data/output/<stem>_control.jsonl)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sim", action="store_true", help="show the top-down simulator window")
    ap.add_argument("--frames", type=int, default=None, help="stop after N processed frames")
    ap.add_argument("--stride", type=int, default=None, help="override video.stride")
    ap.add_argument("--no-clip", action="store_true", help="skip CLIP weather classification")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.stride is not None:
        cfg["video"]["stride"] = a.stride

    stem = Path(a.video).stem
    out = a.out or f"data/output/{stem}_annotated.mp4"
    log = a.log or f"data/output/{stem}_control.jsonl"
    Path("data/output").mkdir(parents=True, exist_ok=True)

    print(f"input   {a.video}")
    print(f"output  {out}")
    print(f"log     {log}")
    print(f"stride  {cfg['video']['stride']}   clip={'off' if a.no_clip else 'on'}\n")

    pipe = Pipeline(cfg, use_clip=not a.no_clip)
    stats = pipe.run(a.video, out, log, sim=a.sim, max_frames=a.frames)

    print("\n--- summary ---")
    print(f"frames      {stats['frames']}")
    print(f"mean        {stats['mean_ms']:.1f} ms/frame  ({stats['fps']:.2f} FPS)")
    print("state histogram:")
    for k, v in sorted(stats["states"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:20} {v:5d}  {100 * v / max(1, stats['frames']):5.1f}%")
    print(json.dumps({k: stats[k] for k in ("out", "log")}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: 3 passed. This is slow (~1 min) because it runs real inference.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: all tests pass. Fix anything red before continuing.

- [ ] **Step 7: Run end to end on a real clip**

```bash
.venv/bin/python run.py --video data/input/clear.mp4 --frames 40 --no-clip
.venv/bin/python run.py --video data/input/clear.mp4 --sim
```
Expected: `data/output/clear_annotated.mp4` plays with the full HUD, the simulator window shows the green ego car steering, and the state histogram is dominated by `LANE_KEEP` with some `FOLLOW`.

- [ ] **Step 8: Commit**

```bash
git add src/pipeline.py run.py tests/test_pipeline.py
git commit -m "feat: full pipeline wiring and CLI entry point"
```

---

### Task 13: `tools/figures.py` — report figures and ablations

**Files:**
- Create: `tools/figures.py`
- Test: none. This is a reporting script; its output is checked by eye.

**Interfaces:**
- Consumes: `src.segment.Segmenter`, `src.weather`, `src.pipeline.Pipeline`, `src.video_io.VideoReader`.
- Produces four files in `docs/figures/`: `fig1_int8_vs_fp32.jpg`, `fig2_enhancement.jpg`, `fig3_degradation.jpg`, `fig4_state_histogram.png`, plus `docs/figures/metrics.json`.

These are the results section of your report. Generate them once the pipeline works.

- [ ] **Step 1: Write `tools/figures.py`**

```python
#!/usr/bin/env python
"""Generate the report figures.

Figure 1 is the one quantitative result in the project: INT8 versus FP32
segmentation. It needs no labelled data, because FP32 is the reference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.segment import Segmenter
from src.types import WeatherInfo
from src.video_io import VideoReader
from src.weather import enhance, visibility_score

OUT = Path("docs/figures")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def _overlay(bgr: np.ndarray, da: np.ndarray, ll: np.ndarray) -> np.ndarray:
    v = cv2.resize(bgr, (640, 360)).copy()
    m = da.astype(bool)
    v[m] = (0.6 * v[m] + 0.4 * np.array([0, 180, 0])).astype(np.uint8)
    v[ll.astype(bool)] = (0, 0, 255)
    return v


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def fig1_quantization(cfg: dict, video: str, n: int = 12) -> dict:
    """INT8 vs FP32: speed and mask agreement."""
    i8 = Segmenter(cfg["model"]["segmenter"], cfg["model"]["onnx_threads"])
    f32 = Segmenter(cfg["model"]["segmenter_fp32"], cfg["model"]["onnx_threads"])

    da_ious, ll_ious, t8, t32, rows = [], [], [], [], []
    for k, (_, _, frame) in enumerate(VideoReader(video, stride=7, max_frames=n)):
        s = time.perf_counter(); a8, b8 = i8.infer(frame); t8.append(time.perf_counter() - s)
        s = time.perf_counter(); a3, b3 = f32.infer(frame); t32.append(time.perf_counter() - s)
        da_ious.append(_iou(a8 == 1, a3 == 1))
        ll_ious.append(_iou(b8 == 1, b3 == 1))
        if k < 3:
            rows.append(np.hstack([
                _label(_overlay(frame, a3, b3), f"FP32  {t32[-1]*1000:.0f} ms"),
                _label(_overlay(frame, a8, b8),
                       f"INT8  {t8[-1]*1000:.0f} ms   daIoU {da_ious[-1]:.3f}  llIoU {ll_ious[-1]:.3f}"),
            ]))

    cv2.imwrite(str(OUT / "fig1_int8_vs_fp32.jpg"), np.vstack(rows))
    m = {
        "int8_ms": float(np.mean(t8) * 1000),
        "fp32_ms": float(np.mean(t32) * 1000),
        "speedup": float(np.mean(t32) / np.mean(t8)),
        "da_iou": float(np.mean(da_ious)),
        "ll_iou": float(np.mean(ll_ious)),
        "frames": len(da_ious),
    }
    print(f"fig1: INT8 {m['int8_ms']:.0f} ms vs FP32 {m['fp32_ms']:.0f} ms "
          f"= {m['speedup']:.1f}x   daIoU {m['da_iou']:.3f}  llIoU {m['ll_iou']:.3f}")
    return m


def fig2_enhancement(video: str) -> dict:
    """Raw versus restored, one row per weather mode."""
    frame = next(iter(VideoReader(video, stride=30)))[2]
    rows, scores = [], {}
    for label in ("fog", "night", "rain"):
        out = enhance(frame, WeatherInfo(label=label))
        scores[label] = {
            "raw_visibility": visibility_score(frame),
            "enhanced_visibility": visibility_score(out),
        }
        rows.append(np.hstack([
            _label(cv2.resize(frame, (480, 270)), f"RAW  vis {scores[label]['raw_visibility']:.2f}"),
            _label(cv2.resize(out, (480, 270)),
                   f"{label.upper()} RESTORED  vis {scores[label]['enhanced_visibility']:.2f}"),
        ]))
    cv2.imwrite(str(OUT / "fig2_enhancement.jpg"), np.vstack(rows))
    print("fig2: enhancement comparison written")
    return scores


def fig3_degradation(cfg: dict, video: str) -> None:
    """The degradation ladder, forced by progressively destroying the input."""
    from src.pipeline import Pipeline
    from src.hud import render

    frame = next(iter(VideoReader(video, stride=30)))[2]
    pipe = Pipeline(cfg, use_clip=False)

    variants = [
        ("NORMAL", frame),
        ("HEAVY FOG (synthetic)",
         cv2.addWeighted(frame, 0.35, np.full_like(frame, 200), 0.65, 0)),
        ("NEAR BLIND", (frame * 0.06).astype(np.uint8)),
    ]
    rows = []
    for name, f in variants:
        s = pipe.process(0, 0.0, f)
        rows.append(_label(cv2.resize(render(s, cfg), (640, 360)),
                           f"{name}  ->  {s.decision.fsm_state}  "
                           f"target {s.decision.target_speed:.0f} km/h"))
    cv2.imwrite(str(OUT / "fig3_degradation.jpg"), np.vstack(rows))
    print("fig3: degradation ladder written")


def fig4_states(log_path: str) -> dict:
    """State histogram from a completed run's control log."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(l) for l in Path(log_path).read_text().splitlines()]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["fsm_state"]] = counts.get(r["fsm_state"], 0) + 1

    keys = sorted(counts, key=lambda k: -counts[k])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(keys, [counts[k] for k in keys], color="#2b7a4b")
    ax.set_ylabel("frames")
    ax.set_title(f"FSM state distribution ({len(rows)} frames)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(OUT / "fig4_state_histogram.png", dpi=150)
    print(f"fig4: state histogram written ({len(rows)} frames)")
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate all report figures.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--log", default=None, help="control log from a completed run")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))

    metrics = {"quantization": fig1_quantization(cfg, a.video),
               "enhancement": fig2_enhancement(a.video)}
    fig3_degradation(cfg, a.video)
    if a.log:
        metrics["states"] = fig4_states(a.log)

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {OUT}/metrics.json")
```

- [ ] **Step 2: Generate the figures**

```bash
mkdir -p tools && touch tools/__init__.py
.venv/bin/python run.py --video data/input/clear.mp4 --no-clip
.venv/bin/python tools/figures.py --video data/input/clear.mp4 --log data/output/clear_control.jsonl
```

Expected: four files in `docs/figures/` plus `metrics.json`. Figure 1's speedup should be
between 4× and 9× with `da_iou` above 0.97 — those are the numbers you quote in the report.

- [ ] **Step 3: Check each figure by eye**

Open all four. Figure 3 is the important one: the three rows must show the state and
target speed changing as the input degrades, ending in `PERCEPTION DEGRADED` behaviour
on the near-blind row. If the bottom row still reports a high target speed, the
degradation ladder is not wired correctly — revisit Task 12 Step 3.

- [ ] **Step 4: Commit**

```bash
git add tools/ docs/figures/
git commit -m "docs: report figures and quantization ablation"
```

---

## Appendix A: Day 0 — sourcing input clips

**Do this before Task 3**, because every visual verification step needs footage.
You need two clips, 20–60 seconds each, 720p or lower:

- `data/input/clear.mp4` — daylight, visible lane markings, some traffic, ideally a
  traffic light and a speed-limit sign
- `data/input/rain.mp4` — rain, fog, snow, or night; this is what exercises the
  degradation ladder

**Option 1: you already have footage.** Trim it and drop it in:

```bash
.venv/bin/pip install yt-dlp     # only needed for option 2
ffmpeg -i myvideo.mp4 -ss 00:01:30 -t 30 -vf scale=1280:720 -an data/input/clear.mp4
```

**Option 2: download dashcam footage.**

```bash
.venv/bin/pip install yt-dlp
# Search YouTube for "dashcam highway 4k no commentary" and
# "dashcam heavy rain night driving", pick two clips, then:
.venv/bin/yt-dlp -f "bestvideo[height<=720]" -o raw_clear.mp4 "<URL>"
ffmpeg -i raw_clear.mp4 -ss 00:02:00 -t 30 -an data/input/clear.mp4
```

**Option 3: BDD100K samples.** The seven frames already in
`third_party/twinlitenet/images/` are BDD100K stills covering snow, night, tunnel and
shadow. They are enough for the still figures but not for a video demo.

- [ ] Verify both clips load:

```bash
.venv/bin/python -m src.video_io --video data/input/clear.mp4 --stride 30
.venv/bin/python -m src.video_io --video data/input/rain.mp4  --stride 30
```
Expected: resolution, fps and frame count printed, then per-frame lines.

**Licensing note for the report:** state where each clip came from. Use footage
published under a permissive licence, or record your own. Do not claim third-party
footage as your own dataset.

---

## Appendix B: Rendering the final video on Colab

The laptop runs at roughly 4 FPS, which is fine for development on 20-second clips but
slow for the final full-length render. On Colab with a GPU the same code runs an order
of magnitude faster, because ultralytics uses CUDA automatically when it is available.

```python
!git clone <your repo url> project && cd project
!pip install -q -r requirements.txt
!python run.py --video data/input/clear.mp4          # no --sim: Colab has no display
```

`--sim` must be omitted on Colab: `cv2.imshow` has no display to draw on. Download
`data/output/*_annotated.mp4` when it finishes.

---

## Appendix C: Where the time will actually go

Honest expectations, so nobody is surprised on Day 3:

| Activity | Realistic share of the four days |
|---|---|
| Writing module code | 35% |
| Tuning `camera.ipm_src` and the planner thresholds | 30% |
| Sourcing and trimming clips | 10% |
| HUD layout and colour fiddling | 10% |
| Report figures and write-up | 15% |

The two things most likely to eat a day: the IPM calibration in Task 8 Step 6, and CLIP
prompt wording in Task 6. Both are pure config changes, so neither requires touching
code — but both need patience and a real clip in front of you.
