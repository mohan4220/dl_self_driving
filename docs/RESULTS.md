# Measured Results

All figures measured on the target machine: Intel i7-7600U, 2 physical cores,
no GPU, CPU governor `powersave` (800 MHz idle, 1.6–3.1 GHz under load).

## 1. INT8 quantization of TwinLiteNet — the quantitative contribution

ONNX Runtime dynamic quantization to UInt8, measured over 7 BDD100K frames.

| Metric | FP32 | INT8 | Change |
|---|---|---|---|
| Latency / frame | 774–1350 ms | **144 ms** | **5.4× faster** |
| Model size | 1.80 MB | **0.558 MB** | 3.2× smaller |
| Drivable-area mask IoU vs FP32 | — | **0.985** | — |
| Lane-line mask IoU vs FP32 | — | **0.941** | — |

FP32 at ~1 FPS is unusable on this hardware; INT8 at ~7 FPS makes CPU-only
deployment viable. Accuracy loss is negligible.

Input resolution is fixed at 640×360 — the encoder contains a hardcoded
`Reshape` to `{1,32,45,80}`, so resolution is not available as a speed lever.

## 2. Per-stage latency budget

| Stage | Latency |
|---|---|
| TwinLiteNet INT8 (drivable + lane) | 144 ms |
| YOLO11n @ 640 (objects + ByteTrack) | 82 ms |
| **Perception subtotal** | **~226 ms** |
| Full pipeline incl. HUD, BEV, control | 419–495 ms (2.0–2.4 FPS) |

YOLO11n alternatives measured: 52 ms @ 480, 34 ms @ 384.

CLIP runs every 15th frame only; RapidOCR runs once per sign track id, cached.
A 60 s clip at stride 2 renders in roughly 6 minutes.

## 3. Lane geometry — pixel space vs ground plane

The first implementation computed heading error from the raw pixel slope of the
lane centreline. A pixel slope is not a ground angle: perspective means one
vertical pixel spans far more ground than one lateral pixel. Correcting this to
reproject through the ground-plane homography changed the results decisively.

| Frame | Road | Pixel-space heading | Ground-plane heading |
|---|---|---|---|
| `caeb782d` snowy street | straight | −20.5° | **+0.3°** |
| `cb22c820` construction | slight left | −23.5° | **−4.0°** |
| `cc97fab0` highway | straight | −69.1° | **+0.9°** |

Worst-case error across valid fits fell from 69.1° to 4.0°. Offsets now fall
within ±0.8 m inside a 3.5 m lane, and curvature radii read 61 km (straight),
1.3 km and 1.4 km.

## 4. All-weather robustness — zero-shot, no training

TwinLiteNet drivable-area and lane-line segmentation, measured on 7 BDD100K
frames spanning the adverse conditions the project targets.

| Frame | Condition | Drivable % | Lane % | Lane fit valid |
|---|---|---|---|---|
| `caeb782d` | snow-covered street | 24.7 | 1.01 | yes |
| `caec69a1` | night, oncoming headlights | 29.1 | 0.19 | no |
| `cb22c820` | construction, cyclist | 19.0 | 1.42 | yes |
| `cb5903ec` | tunnel entrance | 19.4 | 1.87 | no |
| `cbd64c44` | deep shadow | 12.8 | 0.57 | no |
| `cc73b69d` | underpass, bright sky | 7.4 | 1.16 | no |
| `cc97fab0` | highway, traffic | 33.3 | 2.95 | yes |

The drivable-area mask held in every condition including snow and near-darkness,
while the lane fit failed on 4 of 7. That asymmetry is the design's central
claim: when lane paint disappears, free space survives, so the system degrades
to `FREESPACE MODE` rather than failing.

## 5. Degradation ladder — verified

Pipeline run over an all-black 8-frame clip (total perception failure):

- `lane_valid` false on every frame
- `target_speed_kmh` capped at **20.0** throughout (blind cap), never exceeding
  the 40 km/h free-space cap

## 6. Longitudinal control

After converting the speed error from km/h to m/s and retuning to
`kp=0.30, ki=0.15, kd=0.02`, the step response to a 50 km/h target is:
full throttle while acceleration-limited, then tapering 0.739 → 0.423 → 0.165 → 0
from roughly 44 km/h, a single 5.8% overshoot to 52.88 km/h, settling to
50.00 km/h by ~18 s with no sustained oscillation.

## 7. Real-footage run

Source: 15.8 s UK motorway dashcam, originally 720x1280 **portrait**. Cropped to
720x404 landscape (`top=780`), which also removed a burned-in "LIKE & SUBSCRIBE"
overlay. 458 frames, processed at stride 2 = 229 frames.

| Metric | Value |
|---|---|
| Throughput | 515 ms/frame, 1.94 FPS |
| Lane fit valid | 134 / 229 frames (59%) |
| Weather (CLIP, zero-shot) | `clear`, visibility 0.85-0.90 |
| Tracked objects | 1.1 / frame, max 3 |
| Offset from lane centre | median +0.12 m |
| Heading error | median +1.3 deg |
| Traffic lights seen | none (motorway) - correct |

FSM state distribution: FOLLOW 59.8%, CHANGING 21.8%, PREP_CHANGE_R 13.1%,
LANE_KEEP 5.2% - one complete overtake of the vehicle ahead.

ByteTrack held the same id (`car#1`) on the lead vehicle from frame 8 to frame
200, confirming tracking stability under real motion.

### Quantization re-measured on real footage

| Metric | Value |
|---|---|
| Drivable-area IoU, INT8 vs FP32 | **0.993** |
| Lane-line IoU, INT8 vs FP32 | **0.958** |

Both are *better* than the 0.985 / 0.941 measured on isolated stills. The
speedup measured in this run is only 2.1x rather than 5.4x, because the two
models are timed back to back under sustained load and this laptop thermally
throttles from 3.1 GHz to about 1.6 GHz; the 5.4x figure in section 1 was
measured per-model with warmup. Quote 5.4x for the isolated claim and 2.1x for
the sustained-load claim, and say which is which.

### Two defects only real footage exposed

1. **Curvature was unresolvable but still capped speed.** The circle fit spans
   only ~7.6 m of ground, over which centimetre-level lane noise produced a
   13.9 m median radius on a straight motorway, forcing a 17 km/h speed cap.
   Radii below `segment.min_curve_radius_m` are now reported as unresolvable.
2. **The FSM overtook forever.** The video path is open-loop, so the ego never
   moves, the lead vehicle stays in front, and FOLLOW retriggers the instant a
   lane change completes - 62% of frames were lane-changing. A cooldown after
   each completed manoeuvre brought that to one overtake and 60% FOLLOW.

## 8. Honest limitations

- **One clip, one condition.** The footage is 15.8 s of clear-weather UK
  motorway. The adverse-weather claim still rests on the BDD100K stills in
  section 4, not on video. A rain, fog or night clip would exercise the
  degradation ladder end to end.
- **`camera.ipm_src` is calibrated but not surveyed.** The trapezoid comes from
  measured lane positions, and the forward distances depend on an assumed
  ~65 deg horizontal field of view. Lateral scale is sound because it derives
  from the known 3.5 m lane width; forward distances carry roughly a 25%
  uncertainty. Say so rather than quoting distances as exact.
- **The HUD is cramped on a 720x404 frame.** The steering and log panels were
  laid out for 16:9 at 1280 wide and overlap the road on this shorter crop.
  Cosmetic, and legible, but worth widening the source crop if re-shooting.
- **No detection accuracy is claimed.** The clips are unlabelled, so no mAP or
  IoU-against-ground-truth number appears anywhere. The only quantitative result
  is §1, which needs no labels because FP32 is its own reference.
- **Speed-limit reading is a stub.** COCO has no speed-limit-sign class, so OCR
  only ever runs on stop-sign crops. A dedicated traffic-sign detector is future
  work.
- The video path is open-loop by construction: a steering command cannot move a
  recorded camera. Closed-loop behaviour appears only in the `--sim` window.
