# Letterbox preprocessing

**Question:** can 4K letterbox preprocessing reach ≤15 ms/frame without changing the tensor?

**Answer:** yes — **4.13 ms** from an ndarray, down from 30.47 ms through PIL; tensors are bit-identical.

## Protocol

| Item | Value |
|---|---|
| Frame | 2 560 × 1 440 RGB |
| Model input | 960 × 960 |
| Operations | aspect-preserving `INTER_LINEAR` resize → constant-114 pad → HWC→CHW → float32 ÷ 255 |
| Timing | 20 runs, median wall time |
| Host | AMD Ryzen 7 2700X (8 cores / 16 threads), CPU |

## Results

| Path | Median |
|---|---|
| PIL: `Image.fromarray` → `letterbox` → tensor | 30.47 ms |
| ndarray: `letterbox(array)` → tensor | **4.13 ms** |
| Speed-up | **7.4×** |

Tensor equality: `np.array_equal` → **true** (max delta 0).

## Where the time went

The old path paid for PIL three times:

1. `_load_image` converted every numpy input with `Image.fromarray(array).convert("RGB")` — a full copy into a PIL buffer.
2. `letterbox` called `np.asarray(image.convert("RGB"))` — another convert + copy back to numpy.
3. The tensor step allocated `canvas.astype(np.float32)`, divided, transposed, and made a contiguous copy.

At 2 560 × 1 440 the two PIL conversions alone were ~23 ms and ~30 ms in isolated probes; resize + pad was under 1 ms. The new path passes the array straight to OpenCV, writes CHW directly into a preallocated tensor with `np.copyto`, and normalises in place.

## Changes

- `letterbox()` accepts a PIL image or an HWC RGB array. Array input skips both PIL conversions; PIL input avoids a redundant `convert("RGB")` when the mode is already RGB.
- `ONNXDetectAdapter._preprocess_yolo_letterbox` builds `(1, 3, H, W)` float32 with `np.copyto` + in-place `/= 255` instead of three temporary arrays.
- `predict()` keeps numpy inputs as arrays on the default YOLO letterbox path; BGR inputs get one `cv2.cvtColor` instead of a PIL round-trip.
- `_as_rgb_array` handles HxW, HxWx3, HxWx4 and non-uint8 inputs.

## Limits

- One frame, one input size, CPU. Resize cost scales with source size, but the PIL overhead does not disappear on smaller frames.
- BGR numpy inputs pay one extra `cv2.cvtColor` (~1–2 ms at 4K); the gate still holds.
- The tensor is bit-identical for uint8 RGB input; float inputs are clipped and cast to uint8, matching the previous `Image.fromarray` behaviour.

## Reproduce

```bash
python benchmarks/preprocess.py \
  --frame /path/to/frame.jpg \
  --size 960 \
  --runs 20
```
