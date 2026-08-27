# All-Weather Self-Driving Car Simulation

A monocular vision pipeline that takes dashcam video and produces the control
signals a self-driving system would send to the vehicle — steering angle, target
speed, indicator state — overlaid on the video alongside a running log of *why*
each decision was made.

Built for a B.Tech major project. **No model training required**: every neural
network is pretrained and frozen. The contribution is the integration — turning
four off-the-shelf perception models into a coherent driving policy, and
degrading gracefully when perception fails.

---

## What it produces

```bash
python run.py --video data/input/clip.mp4
```

Two artifacts:

**`data/output/<name>_annotated.mp4`** — the input video with a full HUD:
drivable-area tint, lane lines, tracked objects with distance and time-to-collision,
a steering wheel that physically rotates with the commanded angle, indicator
arrows, throttle/brake bars, weather label with a visibility score, and a
scrolling decision log.

**`data/output/<name>_control.jsonl`** — one JSON record per frame. This is the
simulated signal to the drive hardware, and the primary artifact for a report:

```json
{"idx": 84, "t": 2.80, "steer_deg": -2.6, "throttle": 0.42, "brake": 0.0,
 "target_speed_kmh": 40.0, "fsm_state": "LANE_KEEP", "indicator": "off",
 "weather": "clear", "visibility": 0.9, "speed_limit_kmh": 50,
 "light": "none", "n_objects": 2, "lane_valid": false,
 "events": ["LANE LOST - FREESPACE MODE - MATCHING 40"]}
```

Add `--sim` for a second window showing a top-down view where a kinematic
bicycle model actually executes the commands. Local only — it needs a display.

---

## Why not end-to-end, or RL?

Worth stating up front, because it is the first question an examiner asks.

**End-to-end behaviour cloning (NVIDIA PilotNet)** maps pixels straight to a
steering angle. Elegant, but it requires training data, and it emits *only* a
steering scalar — no obstacle reasoning, no signal logic, no explanation. The
decision log that makes this project legible would be impossible.

**Reinforcement learning** needs an environment that reacts to actions. A
recorded video does not react: your steering command cannot move the camera.
RL on the video path is not merely hard, it is ill-posed.

So the architecture is **pretrained perception feeding a hand-written policy**.
Every decision is traceable to a rule and a measurement, which is what makes the
overlay possible and the failure modes analysable.

An honest consequence: **the video path is open-loop.** The commands are advisory.
Closed-loop behaviour is demonstrated only in the `--sim` window, where the
bicycle model genuinely responds. This is stated rather than hidden.

---

## Architecture

One dataclass, `FrameState`, flows through a linear chain of pure functions.
Every stage is `FrameState -> FrameState`, so any stage can be run, printed and
inspected in isolation.

```
read ─► weather ─► segment ─► detect ─► signals ─► bev ─► planner ─► control ─► sim ─► hud ─► write
         │           │          │          │         │        │          │
      classify    drivable   YOLO11n    light     image    FSM       pure
      + restore   + lanes    + track    state    → metres            pursuit
                                       + signs                       + PID
```

Weather runs first: everything downstream consumes the restored frame and the
visibility score it produces, which sets detector thresholds and speed caps.

### Models — all pretrained, all frozen

| Role | Model | Size | Latency (2-core CPU) |
|---|---|---|---|
| Drivable area + lane lines | TwinLiteNet, INT8 ONNX | 558 KB | **144 ms** |
| Objects, tracking | YOLO11n (COCO) + ByteTrack | 5.4 MB | **82 ms** |
| Weather classification | CLIP ViT-B/32, zero-shot | ~600 MB | ~300 ms, every 15th frame |
| Speed-limit digits | RapidOCR (PP-OCR) | ~80 MB | ~300 ms, once per sign, cached |

Everything else — the finite state machine, pure-pursuit steering, PID speed
control, inverse perspective mapping, bicycle model, HUD — is written from
scratch and is where the project's engineering sits.

### Modules

Each is one file, one concept, and each runs standalone so it can be understood
and demonstrated on its own:

```bash
python -m src.segment --video clip.mp4   # road + lane segmentation only
python -m src.detect  --video clip.mp4   # detection + tracking only
python -m src.weather --video clip.mp4   # before/after restoration, split screen
python -m src.bev     --video clip.mp4   # IPM calibration tool (see below)
python -m src.control                    # steering table + PID step response
python -m src.sim                        # bicycle model trajectory
```

| File | Responsibility |
|---|---|
| `src/types.py` | `FrameState` and its members — the spine |
| `src/video_io.py` | frame reader/writer, JSONL control log |
| `src/weather.py` | CLIP classification; CLAHE, dark-channel dehaze, de-rain |
| `src/segment.py` | TwinLiteNet inference, lane polynomial fitting |
| `src/detect.py` | YOLO11n + ByteTrack, distance and TTC |
| `src/signals.py` | traffic-light state from HSV, signs, speed zones |
| `src/geometry.py` | ground-plane homography, pixels ↔ metres |
| `src/bev.py` | distance estimation, ego-centric occupancy grid |
| `src/planner.py` | the driving finite state machine |
| `src/control.py` | pure pursuit (lateral), PID (longitudinal) |
| `src/sim.py` | kinematic bicycle model, top-down view |
| `src/hud.py` | overlay renderer |
| `src/pipeline.py` | stage wiring, frame loop |
| `run.py` | CLI |

~2300 lines of Python. All tunable values live in `config.yaml`.

### The driving policy

Finite state machine, strict priority — first match wins:

| Priority | State | Trigger |
|---|---|---|
| 1 | `EMERGENCY_BRAKE` | in-path object below TTC threshold for N consecutive frames |
| 2 | `YIELD_PEDESTRIAN` | in-path person or animal within yield distance |
| 3 | `STOP_SIGNAL` | red/amber light, or stop sign, inside stop-line distance |
| 4 | `CHANGING` | lane change already underway |
| 5 | `PREP_CHANGE_L/R` | indicator asserted, waiting out the lead-in |
| 6 | `FOLLOW` | in-path lead vehicle inside the time-gap threshold |
| 7 | `LANE_KEEP` | otherwise |

Target speed is the minimum of every applicable cap — posted limit, weather cap,
curvature cap, following-distance cap, degradation cap — and the HUD names which
cap is active, so the number is never unexplained.

Lane-change rules: overtake on the configured side only, require a time-headway
gap in the target lane, assert the indicator for N frames *before* lateral
motion begins, and clear it only after the manoeuvre completes.

### Degradation ladder — the all-weather claim, in code

The point of the project is not that perception always works. It is that failure
is detected and handled.

| Trigger | Response | HUD |
|---|---|---|
| Lane polynomial invalid | steer to drivable-region centroid | `LANE LOST — FREESPACE MODE` |
| Drivable mask also empty | hold last steering, decay to zero, cap 20 km/h | `PERCEPTION DEGRADED` |
| Low visibility score | detector confidence 0.35 → 0.20, speed cap, gap ×1.8 | `FOG — SPEED CAPPED 40` |
| Light state ambiguous | caution speed cap, do not halt | `SIGNAL UNCERTAIN` |

This is why the drivable-area mask matters more than the lane lines. On real
BDD100K frames the drivable mask survived snow, near-darkness, tunnel and deep
shadow, while the lane fit failed on 4 of 7. When paint disappears, free space
remains.

---

## Results

Full numbers in **[docs/RESULTS.md](docs/RESULTS.md)**. Headlines:

**INT8 quantization** — the one quantitative result, and it needs no labelled data
because FP32 is its own reference:

| | FP32 | INT8 | Change |
|---|---|---|---|
| Latency | 774–1350 ms | **144 ms** | **5.4× faster** |
| Size | 1.80 MB | 0.558 MB | 3.2× smaller |
| Drivable IoU vs FP32 | — | **0.993** | negligible loss |
| Lane IoU vs FP32 | — | **0.958** | negligible loss |

FP32 at ~1 FPS is unusable on a 2-core laptop; INT8 at ~7 FPS makes CPU-only
deployment viable.

**Ground-plane geometry** — an early version computed heading error from the raw
pixel slope of the lane centreline. A pixel slope is not a ground angle, because
perspective means one vertical pixel spans far more road than one lateral pixel:

| Frame | Road | Pixel-space | Ground-plane |
|---|---|---|---|
| snowy street | straight | −20.5° | **+0.3°** |
| construction | slight bend | −23.5° | **−4.0°** |
| highway | straight | −69.1° | **+0.9°** |

Worst-case error fell from 69.1° to 4.0°. Fed to pure pursuit, the original
would have saturated the steering clamp every frame.

**Real footage**, two clips:

| | UK motorway | US rural |
|---|---|---|
| Throughput | 1.94 FPS | 2.72 FPS |
| Lane fit valid | 59% | 69% |
| Mean \|steering\| | 4.4° | 2.5° |
| FSM | FOLLOW 60%, CHANGING 22% | LANE_KEEP 92%, FOLLOW 8% |

Figures for a report are generated by `python -m tools.figures`.

---

## How to run

### 1. Install

Requires Python 3.10+ (developed on 3.12). Linux or macOS; Windows works but is
untested.

```bash
git clone <repo-url> dl_self_driving_car
cd dl_self_driving_car

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -U pip wheel

# torch FIRST and CPU-only -- otherwise pip drags in ~2.5 GB of CUDA wheels
# that this project cannot use. Skip this line on Colab, which ships torch+CUDA.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-cpu.txt
```

**Pick the requirements file that matches your hardware.** They differ only in
which ONNX Runtime build they pull, and the two builds cannot coexist:

| File | Use when | Installs |
|---|---|---|
| `requirements-cpu.txt` | no NVIDIA GPU | `onnxruntime` |
| `requirements-gpu.txt` | CUDA GPU, incl. Colab | `onnxruntime-gpu` |
| `requirements-common.txt` | never directly | everything else |

`onnxruntime` and `onnxruntime-gpu` ship the same `onnxruntime/capi/*.so`, so
installing both leaves one binary and two sets of pip metadata — you silently get
whichever came last. The GPU wheel includes `CPUExecutionProvider` as well, so
there is never a reason to have both.

On a GPU you must also point the segmenter at the FP32 model; see
[docs/COLAB.md](docs/COLAB.md).

Roughly 2–3 GB, almost all of it torch. If `python3 -m venv` fails on Ubuntu:
`sudo apt install python3.12-venv`.

**No weight downloads.** All four models are committed under `models/` (7.7 MB):

```
models/twinlitenet_int8.onnx    558K   drivable area + lane lines (primary)
models/twinlitenet_fp32.onnx    1.8M   reference, for the quantization figure
models/yolo11n.pt               5.4M   objects + tracking
```

CLIP (~600 MB) is the one exception — it downloads from HuggingFace on first use.
Pass `--no-clip` to skip it entirely.

### 2. Verify the install

```bash
python -c "import torch, cv2, ultralytics, onnxruntime; print('ok')"
pytest -q                            # 33 tests, ~0.5s
python -m src.control                # no video needed: prints steering + PID tables
```

If those pass, the pipeline is working. `python -m src.control` is the fastest
smoke test because it exercises the control maths with no model loading.

### 3. Footage

**A demo clip is bundled**, so the repo runs out of the box:

```
data/input/footage2.webm    US rural road, 35.6 s, 862x484, 60 fps
```

That is the clip every number in the "real footage" results was measured on.
Skip to step 4 if you just want to see it work.

To use your own: drop any forward-facing dashcam video into `data/input/`.
Landscape, 20–60 s, 720p or lower is ideal. The camera must be fixed, roughly
centred, with the horizon visible.

```bash
# portrait phone video? crop to landscape first -- a 9:16 frame is ~60% sky
ffmpeg -i portrait.mp4 -vf "crop=iw:iw*9/16:0:ih*0.6" -an data/input/clip.mp4
```

No footage to hand? `docs/COLAB.md` and the credits below point at sources;
Pexels and Pixabay are free and unambiguously licensed.

### 4. Run

```bash
# the bundled demo clip -- 60 fps source, so stride 4
python run.py --video data/input/footage2.webm --stride 4

# your own clip
python run.py --video data/input/clip.mp4
```

Outputs land in `data/output/`. On a 2-core CPU expect ~2 FPS, so a 30 s clip at
the default stride takes about 4 minutes.

| Flag | Effect |
|---|---|
| `--video PATH` | input clip (required) |
| `--out PATH` | output video (default `data/output/<stem>_annotated.mp4`) |
| `--log PATH` | control log (default `data/output/<stem>_control.jsonl`) |
| `--sim` | open the top-down simulator window. **Local only** — needs a display |
| `--stride N` | process every Nth frame. Use 4 on 60 fps footage |
| `--frames N` | stop after N frames. Use 40 for a quick look |
| `--no-clip` | skip CLIP weather classification; ~35% faster |
| `--config PATH` | alternative config file |

Typical invocations:

```bash
python run.py --video data/input/clip.mp4 --frames 40 --no-clip   # 20s preview
python run.py --video data/input/clip.mp4 --sim                   # watch it drive
python run.py --video data/input/clip.mp4 --stride 4              # 60fps source
```

The run prints a summary — frame count, ms/frame, FPS, and the FSM state
histogram, which is the quickest way to sanity-check behaviour. Mostly
`LANE_KEEP` on an empty road, `FOLLOW` behind a vehicle. If it is mostly
`CHANGING`, the lane-change cooldown needs raising for that clip.

### 5. Inspect one block at a time

Every module runs standalone, which is how to understand or debug the pipeline:

```bash
python -m src.segment --video data/input/clip.mp4   # road + lanes, with latency
python -m src.detect  --video data/input/clip.mp4   # boxes + track ids
python -m src.weather --video data/input/clip.mp4   # raw vs restored, split screen
python -m src.signals --video data/input/clip.mp4   # light state + speed zone
python -m src.bev     --video data/input/clip.mp4   # IPM calibration overlay
python -m src.hud     --video data/input/clip.mp4   # HUD with synthetic values
python -m src.control                               # steering table, PID response
python -m src.sim                                   # bicycle model trajectory
```

Press <kbd>Esc</kbd> to close any demo window.

### 6. Report figures

```bash
python -m tools.figures --video data/input/clip.mp4 \
                        --log data/output/clip_control.jsonl
cat docs/figures/metrics.json
```

Writes four figures to `docs/figures/`. Note `python -m tools.figures`, not
`python tools/figures.py` — the repo has no packaging config, so a bare script
invocation cannot resolve `from src...` imports.

### 7. Calibration — required for any new camera

Every distance and angle flows through four ground-plane reference points in
`config.yaml` (`camera.ipm_src`). They are camera-specific. Get them wrong and
distances, lane offsets and headings are all wrong together.

```bash
python -m src.bev --video data/input/clip.mp4
```

Adjust `camera.ipm_src` until the yellow trapezoid's bottom edge sits on the road
just ahead of the bonnet, its top edge sits on the lane roughly 30 m out, its
sides follow the ego lane markings, and the cyan distance rulers look plausible —
a car three lengths ahead should read about 15 m.

Budget real time for this. It is the fiddliest part of the project, and the
committed values are calibrated for one specific clip, not for your camera.

### Running on Colab

Roughly 5× faster on a T4. Full guide in **[docs/COLAB.md](docs/COLAB.md)**. Two
things that will bite you: point the config at the **FP32** model (INT8 is a CPU
optimisation and gives nothing on CUDA), and never pass `--sim` — Colab has no
display and the run will fail.

### Troubleshooting

| Symptom | Cause |
|---|---|
| `cv2.error` on `imshow` | no display. Drop `--sim`; the render path is headless |
| Torch download is ~2.5 GB | you skipped the CPU-only index URL |
| `ModuleNotFoundError: rapidocr_onnxruntime` | package is `rapidocr` (v3); import is `from rapidocr import RapidOCR` |
| `ModuleNotFoundError: src` | run from the repo root, and use `python -m tools.figures` |
| Everything is `PERCEPTION DEGRADED` | segmentation is failing — check the clip is landscape and the road is visible |
| Distances look absurd | `camera.ipm_src` is uncalibrated for your camera. See step 7 |
| Constant `EMERGENCY_BRAKE` | raise `planner.ebrake_frames` or `ebrake_ttc_s` in `config.yaml` |
| Mostly `CHANGING` | raise `planner.lane_change_cooldown_frames` |

## Honest limitations

Stated plainly, because a project that names its own weaknesses is stronger than
one that hides them.

**No traffic signs are detected in practice.** COCO — which YOLO11n is trained on
— has exactly two sign classes: `stop sign` and `traffic light`. It has no class
for speed limits, route markers, direction, warning or yield signs. So
`speed_limit_kmh` stays at its default on real footage and the OCR path rarely
fires. A dedicated traffic-sign detector (GTSRB- or Mapillary-pretrained) is the
fix and is the clearest next step.

**Ego speed is commanded, never measured.** A monocular dashcam cannot recover
true vehicle speed without calibration and visual odometry. The HUD is labelled
`CMD SPD` for this reason.

**Distances carry ~25% uncertainty.** Lateral scale is sound, being derived from
the known 3.5 m lane width. Forward distance depends on an assumed field of view.
Do not quote these as exact.

**Curvature is not resolvable.** The IPM baseline spans only a few metres; over
that distance, centimetre-level lane noise produces meaningless radii. Values
below a floor are reported as unresolvable rather than capping the speed on a lie.

**No detection accuracy is claimed.** The clips are unlabelled, so there is no
mAP number anywhere in this project. The INT8 comparison above is the only
quantitative result, and it needs no labels.

**Tested on clear weather only.** Both test clips are clear daylight. The
adverse-weather evidence comes from BDD100K stills, not video.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | every measurement, with methodology and caveats |
| [docs/COLAB.md](docs/COLAB.md) | GPU runtime setup and gotchas |
| [docs/DESIGN.md](docs/DESIGN.md) | the design spec, including rejected alternatives |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | the full implementation plan |
| `docs/figures/` | generated report figures + `metrics.json` |

## Credits

- **TwinLiteNet** — drivable-area and lane segmentation. ONNX model via
  [harrylal/TwinLiteNet-onnxruntime](https://github.com/harrylal/TwinLiteNet-onnxruntime).
  The seven test stills are BDD100K frames from that repository.
- **YOLO11n / ByteTrack** — [Ultralytics](https://github.com/ultralytics/ultralytics).
- **CLIP** — OpenAI, `openai/clip-vit-base-patch32`, used zero-shot.
- **RapidOCR** — [RapidAI/RapidOCR](https://github.com/rapidai/rapidocr), PP-OCR via ONNX Runtime.
- **Dark-channel prior dehazing** — He, Sun & Tang, CVPR 2009.
- **[voldemortX/pytorch-auto-drive](https://github.com/voldemortX/pytorch-auto-drive)** —
  lane-detection benchmark suite (SCNN, RESA, LSTR, LaneATT, BézierLaneNet). Not
  deployed here: it requires CUDA ≥9.2, Python 3.6 and `mmcv-full` 1.x. Its
  `MODEL_ZOO.md` metrics informed the architecture choice.
- **[Zho29/adverse-weather-object-detection](https://github.com/Zho29/adverse-weather-object-detection)** —
  learned restoration (WUNET, DSNet) before YOLOv8 on Foggy Cityscapes. This
  project uses classical dark-channel-prior dehazing instead, for CPU
  feasibility and zero training cost. Learned restoration is future work.

Test footage must be credited to its source, with its licence, in any report.
