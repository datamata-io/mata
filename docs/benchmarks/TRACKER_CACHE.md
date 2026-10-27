# Tracker update latency

**Question:** can `BYTETracker.update()` reach ≤60 ms per frame at ~275 active tracks without changing track IDs?

**Answer:** yes on the median (48.7 ms, down from 98.3 ms); p95 is 60.6 ms. Track IDs are identical across the full clip.

## Protocol

| Item | Value |
|---|---|
| Stream | 132 frames, 1 920 × 1 080, 6 fps, fixed detections (no detector in the loop) |
| Tracks at end | 275 |
| Config | `track_high_thresh=0.25`, `track_low_thresh=0.1`, `new_track_thresh=0.25`, `track_buffer=30`, `match_thresh=0.8`, `fuse_score=True` |
| Timing | per-frame `update()` wall time; median and p95 over 132 frames |
| Host | AMD Ryzen 7 2700X (8 cores / 16 threads), CPU only |
| Baseline | same stream, same config, pre-change code |

## Results

| Config | Before | After | Change |
|---|---|---|---|
| `ultra_parity` (275 tracks) | 98.26 ms | **48.67 ms** | −50.5 % |
| `mata_default` (205 tracks) | 46.01 ms | **20.33 ms** | −55.8 % |

`ultra_parity` p95 after the change: **60.65 ms**. The ≤60 ms gate is met on the median and missed by 0.65 ms on p95.

## Correctness

Replayed against the pre-change output over all 132 frames:

- Track-count mismatch: **0 frames**
- Track-ID differences: **0**
- Maximum box coordinate delta: **3.4 × 10⁻¹³ px** (float reassociation only)

## What changed

1. **Cached box views.** `STrack.tlwh`, `.xyxy`, `.xywh` cache their derived arrays and invalidate on every Kalman state change (`predict`, `multi_predict`, `multi_gmc`, `activate`, `re_activate`, `update`). Callers still receive independent copies, preserving the existing mutation semantics. `BOTrack` uses the same cache and invalidates on its XYWH prediction path.
2. **Vectorized batch prediction.** `multi_predict` now stacks means/covariances and applies the motion model with batched matrix products instead of one Python/Kalman call per track. The process-noise diagonal is rebuilt per batch from each track's height.
3. **Single attribute lookup in IoU.** `_get_xyxy` reads `_xyxy_cache` directly when present and uses `getattr` once instead of `hasattr` + property access (which evaluated every property twice).
4. **float32 IoU.** `iou_batch` keeps boxes in float32; the previous float64 upcast was the largest remaining per-frame cost.

## Limits

- One clip, one tracker config pair, CPU only. GPU is irrelevant here — ByteTrack is CPU/NumPy in this implementation.
- The gate is a median; p95 varies with track-count spikes and host load.
- The cache assumes every Kalman-state mutation goes through the methods above. Any new code that writes `track.mean` directly must call `_invalidate_bbox_cache()`.

## Reproduce

```bash
python benchmarks/tracker_update.py \
  --dets /path/to/dets.json \
  --preset ultra_parity
```

Input format: `{"frames": [{"dets": [[x1, y1, x2, y2, score, cls], ...]}, ...]}`.
