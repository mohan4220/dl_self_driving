# Running on Google Colab

The pipeline was built headless on purpose, so the render path works on Colab
unchanged. Expect roughly a 5x end-to-end speedup over a 2-core laptop.

**Runtime → Change runtime type → T4 GPU** before you start.

## 1. Get the code up there

The repo has no remote configured. Either push it to GitHub first:

```bash
# on your laptop, once
git remote add origin https://github.com/<you>/dl_self_driving_car.git
git push -u origin feature/all-weather-sdc
```

...or zip and upload (`models/` is only 7.7 MB, so this is fine):

```bash
cd .. && zip -r sdc.zip dl_self_driving_car -x '*/.venv/*' '*/data/*' '*/third_party/*' '*/.git/*'
```

## 2. Setup cell

```python
!git clone -b feature/all-weather-sdc <your-repo-url> dl_self_driving_car
%cd dl_self_driving_car

# Colab ships torch+CUDA and opencv already, so torch is not in this file.
# requirements-gpu.txt pulls onnxruntime-gpu; requirements-cpu.txt would pull
# the CPU build and leave the segmenter on CPU.
!pip install -q -r requirements-gpu.txt

# Belt and braces: if a previous cell (or a stale runtime) left plain
# onnxruntime installed, both wheels are present and you silently get whichever
# came last -- the usual cause of "GPU shows 0.2/15 GB used". Force one:
!pip uninstall -y -q onnxruntime
!pip install -q --force-reinstall --no-deps onnxruntime-gpu==1.29.0

import torch, onnxruntime as ort
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("ort:", ort.__version__)
print("ort providers:", ort.get_available_providers())
```

`CUDAExecutionProvider` **must** appear in that provider list. If it does not,
nothing below will use the GPU for segmentation.

### The provider is listed but still does not work

This is the failure to expect, and it is genuinely misleading: ORT **lists**
`CUDAExecutionProvider` and only fails when it tries to load the library.

```
[E:onnxruntime] Failed to load library libonnxruntime_providers_cuda.so
  with error: libcublasLt.so.13: cannot open shared object file
[W:onnxruntime] Failed to create CUDAExecutionProvider.
  Require cuDNN 9.* and CUDA 13.*
```

**Cause: the ORT version is built for a different CUDA major version than the
host has.** `onnxruntime-gpu` 1.23 and later target CUDA 13; Colab currently
ships CUDA 12.x. So a recent ORT asks for `libcublasLt.so.13` and Colab only
has `.so.12`.

Check the host, then pick the matching wheel:

```python
import torch; print("host CUDA:", torch.version.cuda)     # e.g. 12.8
```

| Host CUDA | Wheel |
|---|---|
| 12.x | `onnxruntime-gpu==1.22.0` — the last CUDA 12 release, and what `requirements-gpu.txt` pins |
| 13.x | `onnxruntime-gpu>=1.23` |

```python
!pip install -q --force-reinstall onnxruntime-gpu==1.22.0
import onnxruntime as ort
ort.InferenceSession("models/twinlitenet_fp32.onnx", providers=["CUDAExecutionProvider"])
print("CUDA provider loads:", ort.get_available_providers())
```

If it still refuses, just leave the segmenter on CPU — the run completes anyway
and YOLO and CLIP keep the GPU. But see the timing note below before assuming
that is good enough.

## 3. Point the segmenter at FP32

The INT8 model is **CPU-only by construction**. Dynamic quantization produces
`ConvInteger`/`MatMulInteger` ops that the CUDA provider does not implement, so
ORT falls back to CPU for those nodes — you get no speedup and a GPU sitting
idle. On a GPU runtime you must switch to FP32:

```python
!sed -i 's|segmenter: models/twinlitenet_int8.onnx|segmenter: models/twinlitenet_fp32.onnx|' config.yaml
!grep -n 'segmenter:' config.yaml
```

Keep INT8 on a CPU machine, FP32 on a GPU one. Section 1 of `RESULTS.md`
explains why.

## 4. Upload footage

```python
from google.colab import files
uploaded = files.upload()          # pick your clip
!mkdir -p data/input data/output
!mv *.mp4 data/input/ 2>/dev/null; ls -lh data/input/
```

Or mount Drive, which is better for anything over ~50 MB:

```python
from google.colab import drive; drive.mount('/content/drive')
!cp "/content/drive/MyDrive/clips/footage2.mp4" data/input/
```

## 5. Run

```python
!python run.py --video data/input/footage2.mp4 --stride 2
```

**Do not pass `--sim`.** It calls `cv2.imshow`, and Colab has no display — the
run will fail. The `--sim` window is a local-only path. Everything else, including
the full HUD in the output video, works.

Drop `--stride` for full frame rate, or raise it to preview quickly.

## 5b. Confirm the GPU is actually being used

`run.py` now prints an environment and model banner before processing starts.
You want to see this:

```
=== models ===
  segmenter   models/twinlitenet_fp32.onnx  [CUDAExecutionProvider]
  detector    models/yolo11n.pt  [cuda]
  weather     openai/clip-vit-base-patch32  [cuda]
```

If any of those says `CPUExecutionProvider` or `[cpu]` while a GPU is present,
the run prints an explicit WARNING telling you which model fell back and why.

Cross-check with the GPU itself, in a second cell while a run is in progress:

```python
!nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```

Expect roughly 1.5-3 GB used and non-zero utilisation. **0.2 GB means only a
CUDA context was created and no model is actually on the device** — that is the
symptom this section exists to fix.

Rough per-model memory: YOLO11n ~0.4 GB, CLIP ~0.9 GB, TwinLiteNet FP32 ~0.3 GB.

## 6. Preview and download

```python
from IPython.display import HTML
from base64 import b64encode
mp4 = open('data/output/footage2_annotated.mp4','rb').read()
HTML(f'<video width=800 controls><source src="data:video/mp4;base64,{b64encode(mp4).decode()}"></video>')
```

```python
from google.colab import files
files.download('data/output/footage2_annotated.mp4')
files.download('data/output/footage2_control.jsonl')
```

## 7. Report figures

```python
!python -m tools.figures --video data/input/footage2.mp4 \
                        --log data/output/footage2_control.jsonl
!cat docs/figures/metrics.json
```

Invoke it as `python -m tools.figures`, not `python tools/figures.py` — the repo
has no packaging config, so a bare script invocation cannot resolve `from src...`.

**Careful with the INT8-vs-FP32 figure on Colab.** Once you have switched the
config to FP32, `fig1` compares FP32 against FP32 and the speedup collapses to
1.0x. Generate that one figure on the laptop, where the comparison is meaningful.
Everything else is fine to generate on Colab.

## A warning about partial GPU use

With the segmenter on CPU and only YOLO and CLIP on the GPU, **Colab can be
slower than a laptop**. Measured on the bundled clip:

| | Segmenter | Throughput |
|---|---|---|
| Laptop, 4 threads, no GPU | CPU, INT8 | **2.83 FPS** |
| Colab T4, ORT CUDA broken | CPU, INT8 | 2.25 FPS |

TwinLiteNet is the single heaviest stage at ~144 ms, and a T4 runtime only gives
you **2 vCPUs** against a laptop's 4 threads. Accelerating the two cheaper models
does not make up for running the expensive one on weaker CPU.

So on Colab it is worth getting the CUDA provider working, not shrugging it off.
Both fixes are required: the CUDA-12 wheel **and** the FP32 model.

## Expected timings

| | Laptop (i7-7600U, no GPU) | Colab T4 |
|---|---|---|
| TwinLiteNet | 144 ms (INT8) | ~15 ms (FP32, CUDA) |
| YOLO11n | 82 ms | ~8 ms |
| Full pipeline | ~500 ms/frame | ~100 ms/frame |
| 35 s clip @ stride 4 | ~3.5 min | ~45 s |

The remaining Colab cost is HUD rendering and BEV rasterisation, which are
OpenCV on CPU and do not benefit from the GPU.

## Gotchas

- **Session timeouts.** Free Colab disconnects after ~90 min idle and caps GPU
  hours. Render in one go rather than leaving it sitting.
- **Everything is wiped on disconnect.** Download outputs immediately or write
  them to Drive.
- **CLIP downloads ~600 MB** on first use. Pass `--no-clip` if you only want the
  driving pipeline; weather then stays `clear` and the weather-adaptive speed
  caps never fire.
- **`opencv-python` vs `opencv-python-headless`.** Colab preinstalls one already;
  do not install the other on top, it breaks the cv2 import.
- **`onnxruntime` and `onnxruntime-gpu` must never both be installed.** They share
  the same `onnxruntime/capi/*.so`, so the second install overwrites the first
  while `pip list` keeps showing both. You land on whichever was installed last,
  usually the CPU build, with no error -- the segmenter runs on CPU and GPU memory
  sits near 0.2 GB. The GPU wheel includes `CPUExecutionProvider` too, so install
  only that one.
- **Use `requirements-gpu.txt`, not `requirements-cpu.txt`.** The CPU file pins
  plain `onnxruntime`; installing it after the GPU wheel clobbers the GPU build.
  The two files exist precisely so this cannot happen by accident. Each is
  self-contained — install exactly one.
