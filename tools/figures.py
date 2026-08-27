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
