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
