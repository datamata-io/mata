# Batch inference

**Question:** does batching multiple 640 × 640 images through `predict_batch()` improve CPU throughput by 1.5× at batch 4?

**Answer:** no. On CPU the batch path is neutral-to-slower for a compute-bound YOLO model; the 1.5× gate is **not met**. Batching remains opt-in (`batch_size=1` default) and is intended for accelerator backends, which were not measured here.

## Protocol

| Item | Value |
|---|---|
| Images | 16 crops, 640 × 640, taken from one 2 560 × 1 440 frame |
| Model | 10-class YOLO-family ONNX, 960 × 960 input, dynamic batch dimension |
| Backend | ONNX Runtime 1.x, CPU execution provider |
| Threads | `intra_op_num_threads=8`, `inter_op_num_threads=1` (a 16-thread run is also reported) |
| Warm-up | one call per batch size before timing |
| Timing | 3 rounds, median images/second |
| Host | AMD Ryzen 7 2700X (8 cores / 16 threads); load average ≈ 17 during the run, so absolute numbers are noisy |

The model's batch dimension was made dynamic for the test; the shipped static-batch model falls back to sequential inference. Batch 1 is the sequential reference (one `predict()` per image).

## Results — end to end (`predict_batch`, includes preprocessing)

| Batch | Run A (8 threads) | Speed-up | Run B (16 threads) | Speed-up |
|---|---|---|---|---|
| 1 | 2.46 img/s | 1.00× | 1.49 img/s | 1.00× |
| 2 | 1.09 img/s | 0.44× | — | — |
| 4 | 1.63 img/s | 0.66× | 1.29 img/s | 0.87× |
| 8 | 2.14 img/s | 0.87× | — | — |

## Results — session call only (no preprocessing)

| Batch | ms/image | Speed-up vs batch 1 |
|---|---|---|
| 1 | 419 | 1.00× |
| 2 | 395 | 1.06× |
| 4 | 320 | 1.31× |
| 8 | 382 | 1.10× |

The best case — the session call alone at batch 4 with 8 threads — is 1.31×, below the 1.5× gate. End-to-end is worse because preprocessing stays per-image and larger batches increase memory pressure.

## Numerical parity

Batch output matches the per-image loop to within **1.2 × 10⁻⁴ px** on every box (float ordering in the convolution kernels, not a logic difference). Detection counts and labels match exactly.

## Verdict

- `predict_batch()` is correct and falls back safely for static-batch models.
- It is **not a CPU optimisation** for a compute-bound model at 960 × 960; keep `batch_size=1` on CPU.
- The 6.2 gate is recorded as failed on CPU. A GPU run (`onnxruntime-gpu`, same model, batch 1/4/8) is the missing measurement; do not claim a speed-up until it exists.
- `TiledDetect(batch_size=...)` is opt-in and does not change default behaviour.

## Reproduce

```bash
python benchmarks/batch_inference.py \
  --model /path/to/dynamic-batch.onnx \
  --frame /path/to/frame.jpg \
  --batch-sizes 1,2,4,8 \
  --threads 8
```
