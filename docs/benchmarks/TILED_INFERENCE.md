# Tiled inference parity

**Question:** does `TiledDetect` reproduce a reference slicer + merge pipeline on the same detections?

**Answer:** yes, exactly for NMS and to within 0.04 % mean IoU for NMM, with lower merge overhead.

## Protocol

| Item | Value |
|---|---|
| Corpus | 24 frames, 2 560 × 1 440 (18 session stills + 6 video frames) |
| Detector | 10-class aerial detector, ONNX, 960 × 960 input, CPU execution provider |
| Slice / overlap | 640 × 640 / 0.2 |
| Confidence | 0.35 |
| Merge IoU | 0.5 |
| Full-image pass | enabled (one extra inference per frame) |
| Inference | shared: the detector runs once per slice; both pipelines merge the identical candidate set |
| Oracle | SAHI 0.11.36: `get_slice_bboxes` + `NMSPostprocess` / `GreedyNMMPostprocess`, `class_agnostic=False` |
| MATA | `mata.core.tiling.merge_instances` (`mode="nms"` / `"nmm"`) |
| Host | AMD Ryzen 7 2700X (8 cores / 16 threads), ONNX Runtime CPU provider |

Candidate extraction (slices + full frame) took **383 s total**, a mean of **1 914 candidates per frame**.

## Results

| Policy | Count delta (mean / max) | F1 (mean) | Mean IoU | MATA merge | Oracle merge |
|---|---|---|---|---|---|
| NMS | 0.000 % / 0.000 % | 1.000 | 1.000 | **321 ms** | 481 ms |
| NMM | 0.000 % / 0.000 % | 1.000 | 0.9999 | **408 ms** | 530 ms |

Per-frame spread:

- NMS: F1 1.000 on all 24 frames; mean IoU 1.000 on all 24.
- NMM: F1 1.000 on all 24; mean IoU 0.99964–0.99999 (union boxes round identically; the residual is float ordering in the union, not a count difference).
- Count delta is 0 on every frame for both policies.

## Tile geometry

`generate_tiles()` was compared with SAHI's `get_slice_bboxes` on **3 000 random** image sizes, slice sizes and overlap ratios: **0 mismatches**. The stride is `slice - int(slice × overlap)` (floor), and edge tiles are shifted inward — the same rule the oracle uses.

## What this proves and what it does not

Proves:

- Slicing geometry is identical to the reference.
- Offset mapping and duplicate handling reproduce the reference output, not just approximately: identical counts and boxes for NMS, and identical counts with sub-0.04 % IoU drift for NMM.
- The MATA merge is 1.3–1.5× faster than the reference postprocess on CPU for ~1 900 candidates.

Does not prove:

- Detector accuracy against ground truth — both pipelines consume the same detections, so this is an agreement test, not an mAP benchmark.
- GPU behaviour or throughput; only CPU was measured.
- Mask/keypoint fidelity: `TiledDetect` is detection-only and drops tile-local masks and keypoints by design.

## Reproduce

```bash
pip install sahi            # optional oracle only; not a runtime dependency
python benchmarks/tiled_parity.py \
  --frames /path/to/frames \
  --model /path/to/model.onnx \
  --limit 24 \
  --json-out /tmp/tiled_parity.json
```

`--frames` accepts a directory, a JSON list of paths, or a comma-separated list.
