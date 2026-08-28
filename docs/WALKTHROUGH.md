# Code Walkthrough

A file-by-file explanation of how this project works, written for someone who
knows Python and a little computer vision but has not seen a self-driving
pipeline before.

Read this alongside the code. Every section names the file it describes, and
every module can be run on its own — those commands are the fastest way to
build intuition, because you see the stage's output in isolation.

---

## 1. The mental model

A self-driving system answers three questions, in order:

1. **What is around me?** — perception
2. **What should I do about it?** — planning
3. **What signal do I send to the car?** — control

Almost every module here belongs to exactly one of those three. If you get lost,
ask which of the three a file is doing, and it will make sense again.

```
   PERCEPTION                      PLANNING          CONTROL
   ┌──────────────────────────┐   ┌──────────┐   ┌─────────────┐
   │ weather   what's the      │   │ planner  │   │ control     │
   │ segment   weather, where  │──►│ FSM:     │──►│ steering    │
   │ detect    is the road,    │   │ what to  │   │ angle,      │
   │ signals   what objects,   │   │ DO       │   │ pedals      │
   │ bev       how far away    │   │          │   │             │
   └──────────────────────────┘   └──────────┘   └─────────────┘
                                                        │
                                          ┌─────────────┴──────────────┐
                                          │ sim  drive a virtual car   │
                                          │ hud  draw it all on screen │
                                          └────────────────────────────┘
```

### The one honest limitation, stated up front

The input is a **recorded video**. When the system decides "steer left", the
camera does not move — it is a recording. So the video path is **open-loop**:
the commands are advisory, drawn on screen but not affecting what happens next.

This is why `sim.py` exists. It takes the same commands and drives a *simulated*
car with them, in a top-down view. That is the only place the loop actually
closes. Being clear about this distinction is worth marks in a viva; pretending
the video path is closed-loop is not defensible.

---

## 2. `src/types.py` — the spine

**Problem it solves:** eleven modules need to share information about one video
frame. Passing fifteen separate arguments around would be unmaintainable.

**The idea:** one object, `FrameState`, holds everything known about a single
frame. Each stage is a function that takes it, fills in its part, and passes it
on:

```python
state = FrameState(idx=0, t=0.0, raw=frame, img=frame)
state.weather  = classify(state)     # weather.py fills this
state.drivable = segment(state)      # segment.py fills this
state.objects  = detect(state)       # detect.py fills this
...
```

**Why this matters practically:** you can stop anywhere and `print(state)` to
see the entire system's belief about the world at that moment. Testing a stage
means constructing a `FrameState` by hand and checking what comes out — no video
required. Most of the test suite works exactly that way.

The dataclasses inside it:

| Type | Holds | Filled by |
|---|---|---|
| `WeatherInfo` | label, confidence, visibility 0–1 | `weather.py` |
| `Track` | one detected object: id, class, box, distance, TTC | `detect.py`, `bev.py` |
| `LaneInfo` | lane polynomials, offset, heading error, curvature | `segment.py` |
| `SignalInfo` | traffic-light state, signs, active speed limit | `signals.py` |
| `BEVGrid` | top-down occupancy grid | `bev.py` |
| `Decision` | FSM state, target speed, indicator | `planner.py` |
| `Control` | steering angle, throttle, brake | `control.py` |

### Sign conventions — memorise these

Three numbers have signs, and mixing them up sends the car off the road. They
are consistent everywhere in the project:

```
offset_m        > 0  ->  the car is RIGHT of the lane centre
heading_err_rad > 0  ->  the lane heads RIGHT relative to where the car points
steer_deg       > 0  ->  steer RIGHT
```

So **a positive offset must produce a negative steering angle**: if you have
drifted right, you correct left. If you ever see the car steer *away* from the
lane centre, a sign is flipped.

---

## 3. `src/video_io.py` — frames in, results out

**Problem:** read a video frame by frame, write an annotated one, and record
what the controller commanded.

Three classes:

- **`VideoReader`** — iterating it yields `(index, timestamp, frame)`. The
  `stride` argument processes every Nth frame, which matters because this
  pipeline runs at ~2 FPS and a 30 fps clip would otherwise take forever.
  Timestamps come from the *source* fps, so they stay physically correct
  whatever stride you use.

- **`VideoWriter`** — writes the annotated frames to an mp4.

- **`ControlLogWriter`** — writes one JSON object per frame to a `.jsonl` file.
  This is the closest thing to a **CAN bus trace**: the actual signal a real car
  would receive. It is the most report-friendly artifact the project produces,
  because you can plot it, tabulate it, and quote it.

### Two bugs here worth learning from

**Releasing the video handle.** The original `__iter__` released the capture
*after* the loop. But if a consumer does `for f in reader: break`, Python raises
`GeneratorExit` at the paused `yield` and the release never runs — leaking an OS
handle. It now uses `try/finally`. This matters because `Pipeline.run` and every
module demo break out of that loop on an ESC keypress.

**Untrustworthy frame counts.** Some containers lie. The original WebM demo clip
reported a frame count of `-5.5e17`. `VideoReader` now treats anything outside a
plausible range as *unknown* and exposes `expected_frames` returning `None`, so
the progress bar says "total unknown" instead of inventing percentages. General
lesson: container metadata is a hint, not a fact.

---

## 4. `src/weather.py` — sensing and restoring

**Problem:** rain, fog, night and snow degrade every downstream stage. The
system needs to know when conditions are bad and do something about it.

Two jobs.

### Job 1: what is the weather? (`WeatherClassifier`)

Uses **CLIP** zero-shot. CLIP was trained to match images with text captions, so
you can classify without training a classifier: give it five candidate sentences
and ask which best matches the image.

```python
"a dashcam photo of a road on a clear sunny day"
"a dashcam photo of a road in heavy rain with wet reflections"
"a dashcam photo of a road in thick fog with low visibility"
...
```

Highest-scoring caption wins. **No training data, no labelled weather dataset.**
The prompts live in `config.yaml` and rewording them changes the results — that
prompt sensitivity is itself worth a paragraph in a report.

CLIP costs ~300 ms, so it runs **every 15th frame**. Weather does not change
between consecutive frames.

### Job 2: how much can we see? (`visibility_score`)

Cheap enough to run on every frame. Combines three hand-crafted cues:

```python
brightness = grey.mean()                      # night drives this down
contrast   = grey.std()                       # fog flattens the histogram
detail     = cv2.Laplacian(grey).var()        # rain and blur kill high frequencies
```

Weighted into a single 0–1 number. When it drops below a threshold, the detector
lowers its confidence bar and the planner increases the following distance.

### Job 3: make the image better (`enhance`)

Classical image processing, dispatched on the weather label:

- **fog → `dark_channel_dehaze`.** Based on He et al.'s *dark channel prior*: in
  a clear outdoor photo, most small patches contain at least one very dark
  colour channel (a shadow, a dark surface). Where that fails, the patch is
  hazy — and *how much* it fails estimates the haze thickness. Subtract the
  estimated haze and contrast returns.
- **night → `clahe_night`.** CLAHE equalises contrast in small tiles rather than
  globally, so a dark road brightens without blowing out headlights. Applied to
  the L channel of LAB so colours are not distorted.
- **rain → `_derain`.** A bilateral filter smooths thin bright streaks while
  preserving strong edges (a plain blur would erase the lane paint too), then
  re-sharpens.

```bash
python -m src.weather --video data/input/footage2.mp4     # raw vs restored, side by side
```

---

## 5. `src/segment.py` — where is the road?

The most important perception module.

**Model:** TwinLiteNet, a small pretrained network that returns **two masks**
from one forward pass:

- **drivable area** — which pixels are road you could drive on
- **lane lines** — which pixels are painted markings

**Why two masks matter.** Lane paint disappears constantly: snow covers it, rain
reflects over it, night hides it, other cars occlude it. The drivable-area mask
survives all of that. So when the lane fit fails, the system falls back to
"steer toward the middle of the free space". **That fallback is the project's
central all-weather claim**, and it is why a segmentation model was chosen over
a lane-lines-only model.

Measured on seven real BDD100K frames covering snow, night, tunnel and deep
shadow: the drivable mask worked on all seven, the lane fit succeeded on three.

### INT8 quantization

The model ships twice: `twinlitenet_fp32.onnx` (1.8 MB) and
`twinlitenet_int8.onnx` (558 KB).

**Quantization** stores weights as 8-bit integers instead of 32-bit floats.
Smaller and faster, at some accuracy cost. Measured here:

| | FP32 | INT8 |
|---|---|---|
| Latency | 774–1350 ms | **144 ms** |
| Drivable IoU vs FP32 | — | 0.993 |
| Lane IoU vs FP32 | — | 0.958 |

**5.4× faster for under 1% mask disagreement.** This is the project's one clean
quantitative result, and it needs no labelled data because FP32 is its own
reference.

Caveat: INT8 is a **CPU** optimisation. It emits integer operations CUDA does
not implement, so on a GPU you must switch to the FP32 file.

### From mask to numbers (`fit_lanes`)

A mask of white pixels is not yet useful. Turning it into a steering target:

1. **Sample in horizontal bands.** Slice the image into 12 rows and take the
   median lane-pixel x in each. Banding stops distant noisy pixels from
   dominating — every band contributes equally.
2. **Split left and right** by which side of the image centre they fall.
3. **Fit a quadratic** `x = f(y)` to each side. Quadratic captures curvature;
   higher orders overfit noise.
4. **Reproject to the ground plane** (see `geometry.py` next) and compute
   offset, heading error and curvature **in metres**.

### The most instructive bug in this project

The first version computed heading error as `arctan(dx/dy)` on the **pixel**
slope of the centreline. That looks perfectly reasonable and is completely
wrong.

In a camera image, perspective means one *vertical* pixel spans metres of road
while one *lateral* pixel spans centimetres. Their ratio is not an angle. On
straight roads this reported **−20.5°, −23.5° and −69.1°**. Fed to the steering
controller it would have saturated the wheel every frame and driven off the road.

After reprojecting through the ground plane: **+0.3°, −4.0°, +0.9°**.

Two lessons worth carrying:

- **All 33 tests passed the whole time.** They exercised synthetic vertical
  lines, where a pixel slope of zero is right by accident. The bug was found by
  *looking at rendered output*, not by testing.
- Geometry must be done in the space where it is meaningful. Pixels are not
  metres, and no amount of arithmetic on pixel ratios produces a real angle.

```bash
python -m src.segment --video data/input/footage2.mp4    # masks + centreline + latency
```

---

## 6. `src/geometry.py` — pixels to metres

Small file, large consequences. It answers: *given a pixel, where is that point
on the road in metres?*

**The trick: a homography.** If you assume the road is flat, the mapping from
image pixels to ground coordinates is a single 3×3 matrix. You do not need the
camera's focal length or lens parameters — you need **four points whose
real-world positions you know**.

So: pick four points forming a trapezoid on the road ahead, and declare where
they are in metres:

```yaml
ipm_src:   [[254,255], [387,255], [478,345], [162,345]]     # pixels
ipm_dst_m: [[-1.75,13.2], [1.75,13.2], [1.75,5.6], [-1.75,5.6]]   # metres
```

Reading that: the pixel `(254,255)` is 1.75 m left of centre and 13.2 m ahead.
`cv2.getPerspectiveTransform` turns those four correspondences into the matrix,
and then any pixel can be converted.

This is **inverse perspective mapping** (IPM), and it is why the system can say
"car at 19 m" from a single camera with no depth sensor.

### Calibration is per-camera, and it is not optional

Those four points are specific to one camera at one mounting angle. Move the
camera and every distance and angle in the system is wrong together.

The committed values were measured, not guessed: lane-line positions were
sampled across seven straight frames of the demo clip, and the two-row
perspective relation solved for the horizon (y≈190) and camera height (≈1.72 m).

For your own footage:

```bash
python -m src.bev --video data/input/your_clip.mp4
```

Adjust `camera.ipm_src` until the yellow trapezoid hugs the ego lane and the
distance rulers look plausible. **Budget real time for this** — it is the
fiddliest part of the project.

Honest limitation: lateral scale is solid because it derives from the known
3.5 m lane width. Forward distances depend on an assumed field of view and carry
roughly 25% uncertainty. Say so rather than quoting them as exact.

---

## 7. `src/detect.py` — what objects are there?

**Model:** YOLO11n, pretrained on COCO. COCO happens to cover almost everything
this project needs — cars, trucks, buses, motorcycles, bicycles, people,
animals, traffic lights, stop signs — so no training is required.

**Tracking:** ByteTrack, bundled with ultralytics, assigns each object a stable
**id** across frames. Ids are what make two things possible:

1. **Time-to-collision.** Comparing an object's distance now against its
   distance last frame gives closing speed, and `distance / closing_speed` is
   the seconds until impact. Without stable ids you cannot tell which "now" goes
   with which "before".
2. **Caching OCR per sign.** Reading a speed-limit sign costs ~300 ms. With
   ids you read each sign **once** and cache by id — 8 signs in a clip means 8
   OCR calls instead of 900.

### The TTC jitter problem

The first version differenced raw per-frame distance. A YOLO box's bottom edge
wobbles 2–5 px between frames, and at 0.067 s intervals that jitter amplifies
into tens of m/s of imaginary closing speed. Measured: a **stationary** car at
29 m produced `ttc = 1.35 s`, tripping full emergency braking.

Two fixes, both needed:

- **Smooth the distance** with an exponential moving average before
  differencing.
- **Require persistence** — the emergency condition must hold for 3 consecutive
  frames before `EMERGENCY_BRAKE` fires.

General lesson: **differentiating a noisy signal amplifies the noise.** If you
must take a derivative of measured data, filter first.

```bash
python -m src.detect --video data/input/footage2.mp4   # boxes; watch the #id stay stable
```

---

## 8. `src/signals.py` — lights and signs

YOLO tells you *where* a traffic light is but not *what colour it shows*. So the
colour is read with plain HSV reasoning inside the box — no extra model:

1. Crop the light's bounding box.
2. Convert to HSV and measure how much red, amber and green it contains. (Red
   needs two hue windows because red wraps around 0°.)
3. Tie-break by **position** — real lights are red-amber-green top to bottom, so
   the brightest third of the box says which lamp is lit.

Distant or backlit lights return `"unknown"`, which the planner treats as
*caution*, not *stop*.

### An important gating fix

The first version stopped the car for **any** traffic light anywhere in frame —
including one 200 m away on a cross street. `read_signals` now only considers a
light that is inside the stop-line distance and within the ego corridor. The
config key `stop_line_dist_m` existed from the start and was read by nothing;
this is a good reminder to check that your config is actually wired up.

### An honest failure

**Speed-limit signs are effectively never detected.** COCO has exactly two
sign classes: `stop sign` and `traffic light`. It has no class for speed limits,
route markers, direction or warning signs. So `speed_limit_kmh` sits at its
default on real footage, and the OCR path rarely fires.

Verified on both test clips: zero traffic lights, and the only "stop sign" hits
were four detections at ~0.25 confidence, below the 0.35 threshold. One of them
was a genuine US route marker — correctly *not* matched, because it is not a
stop sign.

The fix is a dedicated traffic-sign detector (GTSRB or Mapillary pretrained) as
a second head. That is the clearest next step for anyone extending this.

---

## 9. `src/bev.py` — the world in metres

Uses the homography to answer the questions the planner actually cares about:

- **`track_distance`** — how far ahead is that object? Take the **bottom-centre**
  of its bounding box (where it touches the road) and project to the ground.
- **`annotate_distances`** — fills `dist_m`, `lateral_m` and `in_path` on every
  track. `in_path` is true when the object's lateral offset is inside the ego
  lane corridor, and that single boolean is what separates "car ahead of me,
  slow down" from "car in the next lane, ignore".
- **`build_grid`** — rasterises objects into a top-down occupancy grid for the
  simulator window.

Note the bounding-box-bottom trick: it works because objects rest on the road,
and the road is the plane the homography is calibrated to. It fails for anything
airborne or occluded at the base.

---

## 10. `src/planner.py` — the decisions

The brain. A **finite state machine** — a fixed set of states with rules for
moving between them.

**Why an FSM and not a neural network?** Three reasons, and they are the answer
to an obvious viva question:

1. No training data required.
2. Every decision is explainable — which is the whole point, since the output is
   an overlay explaining decisions.
3. It degrades predictably when perception fails.

### The states, in strict priority order

First match wins. Priority *is* the safety policy:

| # | State | Trigger |
|---|---|---|
| 1 | `EMERGENCY_BRAKE` | in-path object below TTC threshold for N consecutive frames |
| 2 | `YIELD_PEDESTRIAN` | in-path person or animal within yield distance |
| 3 | `STOP_SIGNAL` | red/amber light, or stop sign, inside stop-line distance |
| 4 | `CHANGING` | a lane change is already underway |
| 5 | `PREP_CHANGE_L/R` | indicator on, waiting out the lead-in |
| 6 | `FOLLOW` | lead vehicle inside the time-gap threshold |
| 7 | `LANE_KEEP` | nothing to react to |

Ordering matters: an emergency must override a green light, so it sits above it.

### Target speed is a minimum of caps

Rather than one rule, every constraint proposes a speed and the lowest wins:

```python
caps = [
    posted speed limit,
    weather cap        (fog -> 40 km/h),
    curvature cap      (v = sqrt(a_lat * R)),
    following-gap cap  (distance / time_gap),
    degradation cap    (lane lost -> 40, blind -> 20),
]
target = min(caps)
```

The HUD names **which** cap is active, so the number is never unexplained. That
is a small design choice with a big payoff when demoing.

### The degradation ladder

The system's central claim is not that perception always works — it is that
failure is **detected and handled**:

| Trigger | Response | HUD |
|---|---|---|
| Lane fit invalid | steer to drivable-area centroid | `LANE LOST — FREESPACE MODE` |
| Drivable mask empty too | hold last steering, decay to zero, cap 20 km/h | `PERCEPTION DEGRADED` |
| Low visibility | lower detector threshold, cap speed, widen gap | `FOG — SPEED CAPPED 40` |
| Light state ambiguous | caution cap, do **not** halt | `SIGNAL UNCERTAIN` |

Note the last row: a perception failure should slow the car, not stop it dead in
a live lane.

### Lane changes, and a subtle bug

The rules: overtake on the configured side, require a gap in the target lane,
assert the indicator for N frames **before** moving, clear it after completing.

Two things went wrong and are worth understanding:

**The gap check used a stale speed.** `_target_lane_clear` read
`state.decision.target_speed`, which is still the default `0.0` at that moment,
so a fallback produced a 30 m required gap where policy wanted 41.7 m at
50 km/h. It permitted changes into gaps *smaller* than intended. Fixed by
passing the freshly computed speed as a parameter. Lesson: reading a field that
another part of the same function is about to write is a classic source of
silent wrongness.

**The car overtook forever.** On the video path the ego never actually moves, so
after completing a change the lead vehicle is still there and `FOLLOW`
retriggers immediately — 62% of frames were lane-changing. Fixed with a cooldown
after each completed manoeuvre. This is an *artifact of open-loop video*, not a
logic error, and it is a good example of a bug that only appears on real input.

---

## 11. `src/control.py` — the two numbers

Everything above reduces to: a steering angle and a pedal position.

### Lateral: pure pursuit

The standard geometric path tracker, and it is more intuitive than it looks.
Pick a point on the desired path a fixed distance ahead — the **lookahead
point** — then compute the steering angle whose circular arc passes through it.

```python
alpha = heading_err - arctan2(offset, lookahead)     # angle to the target point
delta = arctan2(2 * wheelbase * sin(alpha), lookahead)
```

The `2 * L * sin(alpha) / ld` comes from the geometry of a circle through the
rear axle and the lookahead point.

**Lookahead scales with speed.** Too short and the car weaves, over-correcting
for every small error. Too long and it cuts corners. `lookahead_min_m + k * v`
is the usual compromise.

### Longitudinal: PID

Classic controller. Error is `target_speed - current_speed`, and:

- **P** reacts to the current error
- **I** accumulates past error, removing steady-state offset
- **D** reacts to the rate of change, damping overshoot

### The units bug

The error was originally in **km/h** with `kp = 0.9`. Any error above 1.11 km/h
saturated the proportional term alone, so throttle was pinned at exactly 1.000
across the whole ramp and the integral never accumulated. That is not a PID
controller — it is an on/off switch with extra steps.

Fixed by converting the error to **m/s** and retuning to
`kp=0.3, ki=0.15, kd=0.02`. The step response now behaves: throttle tapers
0.739 → 0.423 → 0.165 → 0, one 5.8% overshoot, settles at exactly 50.00 km/h.

**Lesson: controller gains are unit-dependent.** A gain tuned for m/s is 3.6×
wrong if you feed it km/h.

**Anti-windup:** if the output is already saturated, adding more integral does
nothing but guarantee a large overshoot later. The code only accumulates while
unsaturated.

```bash
python -m src.control    # steering table + PID step response, no video needed
```

---

## 12. `src/sim.py` — the only closed loop

Takes the same commands and drives a virtual car, using the **kinematic bicycle
model** — the standard low-speed vehicle approximation that collapses the four
wheels into two:

```
x'   = v · sin(yaw)
y'   = v · cos(yaw)
yaw' = v · tan(steer) / wheelbase
```

For a constant steering angle this traces a circle of radius `L / tan(δ)`. It
ignores tyre slip, which is fine at these speeds.

`render_sim` draws the top-down view: distance rings, the lane corridor,
detected objects, and the ego car at its simulated offset and heading.

This is where "the car avoided the obstacle" becomes a real claim rather than an
overlay caption.

```bash
python -m src.sim    # prints a trajectory; verify radius ≈ L/tan(δ)
```

---

## 13. `src/hud.py` — what the examiner sees

Draws everything onto the frame: drivable tint, lane lines, detection boxes with
distance and TTC, a steering wheel that physically rotates, indicator arrows,
throttle/brake bars, weather label, and the scrolling decision log.

Three details that are deliberate:

- **`render()` never mutates `state.img`.** It copies first. The pipeline reuses
  that image downstream, and mutating it would corrupt later stages.
- **The speed reads `CMD SPD`, not `SPD`.** Ego speed is *commanded*, never
  measured — a monocular dashcam cannot recover true speed. The label is the
  project's honesty guarantee.
- **Draw order matters.** The road tint is drawn *before* the boxes. When it was
  the other way round the green blend washed out box colours precisely inside
  the drivable area, where they matter most.

---

## 14. `src/pipeline.py` — putting it together

The frame loop. Stage order is not arbitrary:

```python
weather     # FIRST: everything downstream uses the restored image
segment     # road + lanes
detect      # objects, then annotate with distances
signals     # lights and signs (needs distances, so after bev annotation)
planner     # decide
control     # steer + pedals
sim         # advance the virtual car
hud         # draw
```

### How the degradation ladder is wired

```python
level = degrade_level(state, cfg)
if level == 0:      # lanes good
    offset, heading = state.lanes.offset_m, state.lanes.heading_err_rad
elif level == 1:    # lanes lost, road still visible
    offset, heading = freespace_offset(da, cfg) or 0.0, 0.0
else:               # blind
    steer = self._prev_steer * 0.85     # decay toward straight
```

That decay in the blind case is deliberate: a system that cannot see should
gradually straighten and slow, not freeze at whatever angle it last held.

### A subtlety worth noticing

The longitudinal loop is closed against `self.ego.v_kmh` — the **simulated**
speed — while the lateral loop is open against the video. That is the honest
consequence of not being able to measure real speed. The lookahead uses the
simulated speed rather than the commanded one, because during a stop the
commanded speed is 0 and the lookahead would otherwise collapse exactly when the
car is still moving fast.

---

## 15. `run.py` — the CLI

Parses arguments, prints an environment and model banner, runs the pipeline,
prints a summary.

The banner exists because of a real debugging session: a GPU run silently fell
back to CPU and looked identical to a slow GPU run. Now every model reports the
device it actually landed on, and a mismatch prints a diagnostic naming the
cause.

---

## 16. Following one frame end to end

Frame 370 of the demo clip, where a lane change begins:

```
1. video_io   reads frame 740 of the source, t = 24.6 s
2. weather    CLIP said "clear"; visibility 0.88; image passes through
3. segment    TwinLiteNet -> drivable mask, lane mask
              fit_lanes -> offset +0.05 m, heading +0.2°, valid
4. detect     YOLO finds one car; ByteTrack calls it id 3
5. bev        box bottom -> ground: 16 m ahead, 0.3 m lateral -> in_path = True
              update_ttc: distance shrinking slowly -> ttc = 11 s
6. signals    no lights, no signs -> speed limit stays 50
7. planner    no emergency, no pedestrian, no signal
              lead vehicle at 16 m -> FOLLOW
              followed for 20 frames, right lane clear -> decide to overtake
              -> PREP_CHANGE_R, indicator right on
              caps: [50 zone, 29.4 follow] -> target 29.4 km/h
8. control    pure_pursuit(offset, heading, lookahead) -> +0.1°
              longitudinal(29.4, ego 28.9) -> small throttle
9. sim        bicycle model advances the virtual car
10. hud       draws it all; log shows "FOLLOWING CAR AT 16m"
11. video_io  writes the frame and the JSON record
```

Ten frames later the indicator lead-in completes and the state becomes
`CHANGING`, with the steering bias ramping the car sideways. Ten after that:
`LANE CHANGE COMPLETE — INDICATOR OFF`.

You can watch this exact sequence in the run log — it is the clearest single
demonstration that the whole stack works.

---

## 17. How to explore it yourself

Best order for understanding, easiest first:

```bash
python -m src.control                                  # pure maths, no video
python -m src.sim                                      # bicycle model trajectory
python -m src.detect  --video data/input/footage2.mp4  # boxes and track ids
python -m src.segment --video data/input/footage2.mp4  # masks and centreline
python -m src.weather --video data/input/footage2.mp4  # raw vs restored
python -m src.bev     --video data/input/footage2.mp4  # calibration trapezoid
python -m src.hud     --video data/input/footage2.mp4  # HUD with fake values
python run.py --video data/input/footage2.mp4 --stride 2 --sim   # everything
```

Then read the control log, which is where the system's reasoning is easiest to
inspect:

```python
import json
rows = [json.loads(l) for l in open("data/output/footage2_control.jsonl")]
print(rows[370]["fsm_state"], rows[370]["events"])
```

### Things worth trying

- Set `planner.ebrake_ttc_s` very high and watch it brake constantly.
- Break the calibration — change `camera.ipm_src` — and see every distance go
  wrong together.
- Set `weather.speed_cap_kmh.clear` to 20 and watch the HUD explain the cap.
- Point `model.segmenter` at the FP32 file and compare latency.
- Flip a sign in `pure_pursuit` and watch the car steer away from the lane. Then
  put it back.

That last one teaches more about the sign conventions in thirty seconds than
reading about them.

---

## 18. What this project does not do

Being able to state the limits is worth more in a viva than overclaiming:

- **The video path is open-loop.** Commands are advisory. Only `--sim` closes
  the loop.
- **No traffic signs are detected** beyond stop signs and traffic lights,
  because COCO has no other sign classes.
- **Ego speed is commanded, not measured.**
- **Distances carry ~25% uncertainty** and depend on per-camera calibration.
- **Curvature is not resolvable** — the ground baseline is only a few metres,
  over which lane noise swamps the second-order term. Values below a floor are
  reported as unresolvable rather than capping speed on a bad number.
- **No detection accuracy is claimed.** The clips are unlabelled, so there is no
  mAP figure anywhere. The INT8 comparison is the only quantitative result, and
  it needs no labels.
- **Tested on clear weather only.** The adverse-weather evidence comes from
  stills, not video.

See [RESULTS.md](RESULTS.md) for the measurements and
[DESIGN.md](DESIGN.md) for why each alternative was rejected.
