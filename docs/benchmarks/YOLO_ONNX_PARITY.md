# YOLO ONNX Parity Benchmark

How closely does MATA's YOLO-family ONNX path reproduce the reference export
pipeline? This document records the protocol, the measured results, and the
limits of the claim.

**Scope:** object detection only; YOLO ONNX exports with the v5/v7 (anchors ×
5+nc), v8/v10/v11/v12 (4+nc × anchors), and end-to-end (NMS baked in, N×6)
layouts.

---

## Protocol

| Item | Value |
|---|---|
| Weights | YOLOv8s-class detector, 11.1 M parameters, VisDrone 10-class checkpoint, trained at 960 |
| Export | ONNX, static 1×3×960×960 input, raw anchors (`[1, 14, 18900]`) |
| Images | 24 aerial frames, 2560×1440, mixed session stills and video frames |
| Confidence | 0.35 |
| NMS IoU | 0.45 |
| Detection cap | 300 per image (reference default) |
| Matching | class-aware greedy matching at IoU ≥ 0.5; unmatched predictions are false positives, unmatched references are false negatives |
| Metrics | precision, recall, F1, mean IoU of matched pairs |

The reference result set is produced by the reference implementation's ONNX
decoder, not by MATA. The comparison is therefore same-weights,
same-backend — it isolates preprocessing, decoding and NMS, not the model.

---

## Results

All rows are capped at 300 detections so the comparison matches the deployed
default.

| Path | Detections | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|---|
| Reference `.pt`, GPU fp32 | 6 689 | — | — | — | — |
| Reference `.pt`, GPU fp16 | 6 697 | — | — | — | — |
| Reference `.onnx`, CPU (parity target) | 6 654 | 1.000 | 1.000 | 1.000 | 1.000 |
| MATA 2.0.0 (stretch resize, PIL) | 6 184 | 0.675 | 0.627 | **0.650** | 0.700 |
| MATA + letterbox (PIL resize) | 6 714 | 0.911 | 0.919 | 0.915 | 0.904 |
| MATA 2.0.1 (letterbox, cv2 semantics) | 6 654 | 1.000 | 1.000 | **1.000** | **1.000** |
| MATA 2.0.1 fed OpenCV BGR without `channel_order="bgr"` | 4 288 | 0.660 | 0.426 | 0.518 | 0.703 |

Cross-backend reference: `.pt` fp32 vs `.onnx` CPU gives F1 0.960 and mean IoU
0.961 on the same frames. That ~4 % gap is ONNX Runtime numeric drift, not an
adapter defect; fp16 costs almost nothing by comparison (`.pt` fp32 vs fp16 →
F1 0.998).

Uncapped, MATA 2.0.0 produced 7 643 boxes, PIL letterbox 8 430, and cv2
letterbox 8 442 against the reference's 6 654 — a +27 % count inflation that
`max_det=300` removes.

---

## Cost

Per 2560×1440 frame, CPU, same host:

| Stage | Median |
|---|---|
| PIL open + RGB convert | 33.5 ms |
| PIL resize to 960×960 | 19.0 ms |
| cv2 letterbox resize + pad | 25.4 ms total preprocessing |
| NumPy transpose + /255 | 1.6 ms |
| Postprocess (decode + NMS) | 13.3 ms |
| `session.run` (960, CPU) | 798 ms |
| MATA Python overhead (total minus session) | **55 ms (6.4 %)** |

On GPU, where the model itself is ~15 ms, the Python layer dominates the frame;
a fast preprocessing path is tracked for the next minor release.

---

## Interpretation

- MATA's decode and NMS are **bit-identical** to the reference when given the
  same input tensor: 6 654 / 6 654 boxes matched at mean IoU 1.000.
- The 2.0.0 gap was entirely upstream of the model: a square stretch instead of
  a letterbox, an absent detection cap, and no channel-order handling.
- "Letterbox" alone is not sufficient: a PIL resize reaches F1 0.915. Exact
  parity requires the reference's cv2 `INTER_LINEAR` resize and integer pad
  rounding.
- The remaining `.pt`→`.onnx` drift (~4 % F1 on dense scenes) belongs to the
  export/runtime pair. It is the ceiling for any ONNX-based path and should be
  measured before switching a `.pt` workload to ONNX.

---

## Limitations

- 24 frames, one checkpoint family, one confidence setting. Not an mAP
  evaluation: no ground-truth annotations are used.
- Measurements were taken on a shared host (RTX 3060, other services resident);
  CPU timings are medians from quiet back-to-back runs, not a controlled
  benchmark rig.
- The committed test fixture is synthetic and guards geometry/decoding, not
  real-model numerics; this document covers the real-model check.

---

## Reproduce

```bash
# Reference JSON: one entry per frame, [[x1,y1,x2,y2,label,score], ...]
python benchmarks/yolo_onnx_parity.py \
  --model /path/to/model.onnx \
  --images /path/to/frames \
  --reference /path/to/reference.json \
  --conf 0.35 --iou 0.45 --max-det 300
```

Expected on the same backend: `"f1": 1.0`, `"mean_iou": 1.0`. Against a `.pt`
reference, expect F1 in the 0.95–0.96 range on dense aerial scenes.
