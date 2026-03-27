# MATA CLI Examples

> Shell and PowerShell examples for every `mata` command-line subcommand. _(Added in v1.9.5)_

## Quick Start

```bash
# Install
pip install datamata

# Check version
mata --version

# First detection
mata run detect ../images/000000039769.jpg --model facebook/detr-resnet-50
```

```powershell
# PowerShell equivalent
mata run detect ../images/000000039769.jpg --model facebook/detr-resnet-50
```

## Running the Examples

All scripts are designed to be run from the `examples/cli/` directory:

```bash
cd examples/cli

# Bash (Linux / macOS / WSL)
bash getting_started.sh

# PowerShell (Windows / cross-platform)
.\getting_started.ps1
```

## Scripts

| Bash                                                 | PowerShell                                             | Covers                                                                        |
| ---------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------- |
| [`1_getting_started.sh`](1_getting_started.sh)       | [`1_getting_started.ps1`](1_getting_started.ps1)       | `--version`, `--help`, first detect + classify, save, JSON, verbosity         |
| [`2_run_examples.sh`](2_run_examples.sh)             | [`2_run_examples.ps1`](2_run_examples.ps1)             | `mata run` — all tasks: detect, classify, segment, depth, embed, barcode, vlm |
| [`3_track_examples.sh`](3_track_examples.sh)         | [`3_track_examples.ps1`](3_track_examples.ps1)         | `mata track` — BotSort/ByteTrack, thresholds, ReID, RTSP streams              |
| [`4_val_examples.sh`](4_val_examples.sh)             | [`4_val_examples.ps1`](4_val_examples.ps1)             | `mata val` — detect, classify, segment, depth; plots; JSON metrics            |
| [`5_recognize_examples.sh`](5_recognize_examples.sh) | [`5_recognize_examples.ps1`](5_recognize_examples.ps1) | `mata recognize` — gallery build, top-k, threshold, JSON                      |
| [`6_export_examples.sh`](6_export_examples.sh)       | [`6_export_examples.ps1`](6_export_examples.ps1)       | `mata export` — ONNX / TorchScript _(stub; full support in v2.0)_             |

## Subcommand Reference

### `mata run <task> <image>`

One-shot inference on a single image. Wraps `mata.run()`.

```bash
mata run detect  image.jpg --model facebook/detr-resnet-50 --save
mata run classify image.jpg --model openai/clip-vit-base-patch32 --text "cat,dog"
mata run segment image.jpg --model facebook/detr-resnet-50
mata run depth   image.jpg --model depth-anything/Depth-Anything-V2-Small-hf --save
mata run embed   image.jpg --model openai/clip-vit-base-patch32 --json
mata run barcode image.jpg --model pyzbar
mata run vlm     image.jpg --model Qwen/Qwen3-VL-2B-Instruct --prompt "Describe this"
```

| Flag         | Default          | Description                                 |
| ------------ | ---------------- | ------------------------------------------- |
| `--model`    | registry default | HuggingFace ID, local path, or config alias |
| `--conf`     | —                | Confidence threshold (detect / segment)     |
| `--text`     | —                | Comma-separated labels for zero-shot tasks  |
| `--prompt`   | —                | Text prompt for VLM tasks                   |
| `--save`     | off              | Save annotated result to `--save-dir`       |
| `--save-dir` | `runs/`          | Output directory                            |
| `--json`     | off              | Print raw JSON to stdout                    |
| `--device`   | auto             | `cpu`, `cuda`, `cuda:0`, `mps`              |

### `mata track <source>`

Multi-object tracking. Wraps `mata.track()`.

```bash
mata track video.mp4 --model facebook/detr-resnet-50 --tracker botsort --save
mata track video.mp4 --reid-model openai/clip-vit-base-patch32
mata track rtsp://cam/stream --tracker bytetrack --conf 0.4
mata track 0 --tracker botsort --show
```

| Flag           | Default          | Description                                    |
| -------------- | ---------------- | ---------------------------------------------- |
| `--model`      | registry default | Detection model                                |
| `--tracker`    | `botsort`        | `botsort` or `bytetrack`                       |
| `--conf`       | `0.25`           | Detection confidence threshold                 |
| `--iou`        | `0.7`            | IoU threshold                                  |
| `--reid-model` | —                | ReID model for appearance re-ID (BotSort only) |
| `--save`       | off              | Save annotated output video                    |
| `--show`       | off              | Display live tracking in a window              |
| `--json`       | off              | Per-frame JSON to stdout                       |

### `mata recognize <image>`

Gallery-based identity matching. Wraps `mata.run("recognize", ...)`.

```bash
mata recognize image.jpg --gallery gallery.npz --model openai/clip-vit-base-patch32
mata recognize image.jpg --gallery gallery.npz --top-k 5 --threshold 0.75 --json
```

A `.npz` gallery is required. Build one with Python before running:

```python
import mata
gallery = mata.Gallery(threshold=0.7)
gallery.add("image1.jpg", model="openai/clip-vit-base-patch32", label="person_a")
gallery.save("gallery.npz")
```

| Flag          | Required | Description                           |
| ------------- | -------- | ------------------------------------- |
| `--gallery`   | yes      | Path to `.npz` gallery file           |
| `--model`     | —        | Embed model (HuggingFace ID or alias) |
| `--top-k`     | —        | Number of top matches (default: 1)    |
| `--threshold` | —        | Minimum cosine similarity             |
| `--json`      | —        | Structured JSON output                |

### `mata val <task>`

Dataset evaluation. Wraps `mata.val()`.

```bash
mata val detect   --model facebook/detr-resnet-50 --data configs/coco.yaml --plots
mata val classify --model openai/clip-vit-base-patch32 --data configs/imagenet.yaml
mata val depth    --model depth-anything/Depth-Anything-V2-Small-hf --data configs/diode.yaml
```

Dataset YAML files are in [`../configs/`](../configs/). Edit them to point to your local copies.

### `mata export <task> <model>` _(v2.0 stub)_

```bash
mata export detect facebook/detr-resnet-50 --format onnx
mata export detect ./model.pt --format onnx --quantize int8
```

Full export support is planned for v2.0. In v1.9.5 this prints the pending export summary without writing any files.

## Test Assets

| Asset           | Path                         | Used in                     |
| --------------- | ---------------------------- | --------------------------- |
| COCO val image  | `../images/000000039769.jpg` | `run`, `recognize` examples |
| QR code image   | `../images/sample_qr.png`    | `run barcode` examples      |
| Test video      | `../videos/cup.mp4`          | `track` examples            |
| Dataset configs | `../configs/*.yaml`          | `val` examples              |

Download the COCO test image if not present:

```bash
# Linux / macOS / WSL
curl -o ../images/000000039769.jpg \
     http://images.cocodataset.org/val2017/000000039769.jpg
```

```powershell
# PowerShell
Invoke-WebRequest http://images.cocodataset.org/val2017/000000039769.jpg `
    -OutFile ../images/000000039769.jpg
```
