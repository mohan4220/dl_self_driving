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
