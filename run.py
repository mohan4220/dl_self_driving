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

    import torch, onnxruntime as ort
    print("=== environment ===")
    print(f"  torch          {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu            {torch.cuda.get_device_name(0)}")
    print(f"  onnxruntime    {ort.__version__}")
    print(f"  ort providers  {ort.get_available_providers()}")
    print()
    print(f"input   {a.video}")
    print(f"output  {out}")
    print(f"log     {log}")
    print(f"stride  {cfg['video']['stride']}   clip={'off' if a.no_clip else 'on'}\n")

    pipe = Pipeline(cfg, use_clip=not a.no_clip)

    d = pipe.describe()
    print("=== models ===")
    print(f"  segmenter   {d['segmenter']}")
    print(f"  detector    {d['detector']}")
    print(f"  weather     {d['weather']}")
    if torch.cuda.is_available():
        providers = ort.get_available_providers()
        if d["segmenter_device"] != "cuda":
            print()
            print("  WARNING: a GPU is present but the segmenter is on CPU.")
            if "CUDAExecutionProvider" not in providers:
                print("    onnxruntime has no CUDA provider at all -> you have the CPU")
                print("    wheel. Install onnxruntime-gpu and remove plain onnxruntime;")
                print("    they share a binary and cannot coexist.")
            else:
                # The provider is listed but was not selected, which almost always
                # means its shared library failed to load -- ORT logs that above.
                print(f"    CUDA provider is listed but did not load. ORT {ort.__version__}")
                print(f"    needs a CUDA major version matching the host "
                      f"(torch reports CUDA {torch.version.cuda}).")
                print("    onnxruntime-gpu <=1.22 is built for CUDA 12, >=1.23 for CUDA 13.")
                print("    See the 'Require cuDNN/CUDA' line in the ORT error above.")
            if "int8" in str(cfg["model"]["segmenter"]):
                print("    ALSO: model.segmenter points at the INT8 file. INT8 emits")
                print("    integer ops CUDA does not implement, so it stays on CPU even")
                print("    with a working GPU provider. Switch to the FP32 model.")
        if d["detector_device"] != "cuda":
            print("  WARNING: detector is on CPU -> check config model.device is not 'cpu'.")
    print()
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
