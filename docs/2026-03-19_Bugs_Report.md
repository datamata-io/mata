# Bugs Report — 2026-03-19

Discovered during systematic audit of `examples/` folder.  
**Fixed bugs** were applied immediately. **Complex bugs** are documented here for follow-up.

---

## Summary

| ID          | Severity     | Status     | Description                                                            |
| ----------- | ------------ | ---------- | ---------------------------------------------------------------------- |
| BUG-001     | Critical     | ✅ Fixed   | `MultiResult` missing `__getitem__` — `result['key']` fails            |
| BUG-002     | Critical     | ✅ Fixed   | `MultiResult` missing `__contains__` — `'key' in result` fails         |
| BUG-003     | Minor        | ✅ Fixed   | Private `_mask_to_binary` imported at module level                     |
| BUG-004     | High         | ⬜ Open    | Scenario examples access wrong inner channel name (`.instances`)       |
| BUG-005     | High         | ⬜ Open    | `DepthResult` has no `.shape` — should be `.depth_map.shape`           |
| BUG-006     | High         | ⬜ Open    | `ClassifyResult` subscripted with `[0]` — has no `__getitem__`         |
| BUG-007     | High         | ⬜ Open    | `.classifications` used on `ClassifyResult` — should be `.predictions` |
| REMOVAL-001 | Non-feasible | ⬜ Pending | `valkey_rtsp_pipeline.py` requires unavailable live infra              |

---

## Fixed Bugs

### BUG-001 — `MultiResult` missing `__getitem__`

**File:** `src/mata/core/artifacts/result.py`  
**Severity:** Critical  
**Affects:** All `examples/graph/scenarios/*.py` (24 files)

`MultiResult` (returned by `mata.infer()`) did not implement `__getitem__`, so subscript access `result['channel']` raised `TypeError: 'MultiResult' object is not subscriptable`. All 24 scenario examples relied on this pattern exclusively in their `--real` code paths.

**Root cause:** `MultiResult` implemented `__getattr__` for attribute access but omitted the dict-style subscript interface.

**Fix applied:** Added `__getitem__` to `MultiResult`:

```python
def __getitem__(self, name: str) -> Artifact:
    if name not in self.channels:
        raise KeyError(f"Channel '{name}' not found. Available: {list(self.channels.keys())}")
    return self.channels[name]
```

---

### BUG-002 — `MultiResult` missing `__contains__`

**File:** `src/mata/core/artifacts/result.py`  
**Severity:** Critical  
**Affects:** All scenario files using `if "key" in result`

`"key" in result` raised `TypeError: argument of type 'MultiResult' is not iterable` because `__contains__` was not implemented.

**Fix applied:** Added `__contains__` to `MultiResult`:

```python
def __contains__(self, name: object) -> bool:
    return name in self.channels
```

---

### BUG-003 — Private `_mask_to_binary` imported at module level

**File:** `examples/segment/grounding_sam_pipeline.py`  
**Severity:** Minor  
**Affected line:** `from mata.visualization import _mask_to_binary` (was line 21)

The top-level import of a private (`_`-prefixed) internal function creates a fragile dependency that will silently break if the function is renamed or moved during any internal refactor. The function was only needed as a fallback in the `else` branch of an `if hasattr(instance.mask, "to_binary")` guard.

**Fix applied:** Removed the top-level import; moved it to a lazy import inside the `else` fallback:

```python
else:
    from mata.visualization import _mask_to_binary  # private fallback for raw mask dicts
    mask_array = _mask_to_binary(instance.mask, image_size=image.size)
```

---

## Open Complex Bugs

These bugs require coordinated fixes across scenario files and their corresponding presets. They all manifest only in `--real` mode (guarded by `if USE_REAL:`).

---

### BUG-004 — Scenario examples access wrong inner `MultiResult` channel name

**Files affected:**

- `examples/graph/scenarios/driving_distance_estimation.py`
- `examples/graph/scenarios/driving_road_scene.py`
- `examples/graph/scenarios/agriculture_aerial_crop.py`
- `examples/graph/scenarios/agriculture_pest_detection.py`
- `examples/graph/scenarios/agriculture_disease_classify.py`
- `examples/graph/scenarios/driving_obstacle_vlm.py`
- `examples/graph/scenarios/retail_shelf_analysis.py`
- `examples/graph/scenarios/medical_report_generation.py`

**Severity:** High — `AttributeError` at runtime in `--real` mode

**Root cause:** The `Fuse` node creates an inner `MultiResult` whose channels are named by the kwargs passed to `Fuse(...)`. For example:

```python
# driving.py preset
Fuse(out="final", dets="filtered", depth="depth")
# → inner MultiResult channels: {'dets': VisionResult, 'depth': DepthResult}
```

Examples then access `result['final'].instances`, but 'instances' is not a channel name in the inner `MultiResult`. The inner MultiResult's `__getattr__` raises `AttributeError` because it looks for a channel named 'instances'.

**Symptom:**

```python
# runtime --real mode
len(result['final'].instances)   # AttributeError: No channel 'instances' in MultiResult
for inst in result['final'].instances:  # same
```

**Correct patterns** (preset-specific channel names):

| Scenario file                     | Preset / Fuse config                                                              | Correct access                    |
| --------------------------------- | --------------------------------------------------------------------------------- | --------------------------------- |
| `driving_distance_estimation.py`  | `Fuse(dets="filtered", depth="depth")`                                            | `result['final'].dets.instances`  |
| `driving_road_scene.py`           | `Fuse(dets="filtered", masks="segments", depth="depth", classifications="class")` | `result['final'].dets.instances`  |
| `agriculture_aerial_crop.py`      | `Fuse(masks="segments", depth="depth")`                                           | `result['final'].masks.instances` |
| `agriculture_pest_detection.py`   | `Fuse(dets="filtered", masks="masks_ref")`                                        | `result['final'].dets.instances`  |
| `agriculture_disease_classify.py` | `Fuse(dets="filtered", rois="rois", classifications="classes")`                   | `result['final'].dets.instances`  |

**Note:** An alternative simpler pattern is to bypass the inner `MultiResult` entirely and access the outer result's raw channel directly: `result['filtered'].instances` or `result['dets'].instances` (since `mata.infer()` returns all graph context variables as outer channels).

---

### BUG-005 — `DepthResult` has no `.shape` attribute

**Files affected:**

- `examples/graph/scenarios/driving_distance_estimation.py` (line ~58)
- `examples/graph/scenarios/driving_road_scene.py` (line ~75)
- `examples/graph/scenarios/agriculture_aerial_crop.py` (line ~50)
- `examples/graph/scenarios/driving_obstacle_vlm.py` (line ~77)

**Severity:** High — `AttributeError` at runtime in `--real` mode

**Root cause:** `DepthResult` stores the depth array under `.depth` (raw numpy array) or exposes the array via the `.depth_map` property. It has no direct `.shape` attribute. Accessing `.shape` directly on a `DepthResult` object raises `AttributeError`.

**Symptom:**

```python
# BUG:
result['final'].depth.shape          # DepthResult has no .shape

# Fix:
result['final'].depth.depth_map.shape   # depth_map property returns numpy array
# or
result['final'].depth.depth.shape       # .depth is the raw numpy array
```

**Scope:** All four files contain a print statement reading `result['...'].depth.shape`. Each requires a single-line fix appending `.depth_map` before `.shape`.

---

### BUG-006 — `ClassifyResult` subscripted with integer index

**File:** `examples/graph/scenarios/driving_road_scene.py` (lines ~67-68)

**Severity:** High — `TypeError` at runtime in `--real` mode

**Root cause:** `ClassifyResult` does not implement `__getitem__`. Integer subscripting `result['final'].classifications[0]` raises `TypeError: 'ClassifyResult' object is not subscriptable`. The correct access is via `.predictions` (the field holding the list) or `.top1` (the convenience property).

**Symptom:**

```python
# BUG:
result['final'].classifications[0].label     # TypeError: ClassifyResult not subscriptable

# Fix option A (access list):
result['final'].classifications.predictions[0].label

# Fix option B (use property):
result['final'].classifications.top1.label
```

**Affected code block:**

```python
# driving_road_scene.py ~line 67
print(f"\nScene Type: {result['final'].classifications[0].label} "
      f"(confidence: {result['final'].classifications[0].score:.2f})")
```

---

### BUG-007 — `.classifications` used on `ClassifyResult` (should be `.predictions`)

**Files affected:**

- `examples/graph/scenarios/medical_pathology_triage.py` (lines ~88, 92, 95)
- `examples/graph/scenarios/agriculture_disease_classify.py` (line ~69 in mock print — cosmetic only)
- `examples/graph/scenarios/manufacturing_defect_classify.py` (line ~62 in mock print — cosmetic only)

**Severity:** High (silent failure in `medical_pathology_triage.py`) / Cosmetic in mock print strings

**Root cause:** `ClassifyResult` stores predictions in a field named `predictions` (a `list[Classification]`). There is no `.classifications` attribute. Code that checks `hasattr(result["classes"], "classifications")` always gets `False`, silently skipping the entire analysis block.

**Symptom in `medical_pathology_triage.py`:**

```python
# BUG: hasattr always returns False → block silently skipped
if "classes" in result and hasattr(result["classes"], "classifications"):
    # This block NEVER executes, even when model runs successfully
    for i, classification in enumerate(result["classes"].classifications, 1):
        ...
```

**Fix:**

```python
# Check for correct attribute
if "classes" in result and hasattr(result["classes"], "predictions"):
    for i, classification in enumerate(result["classes"].predictions, 1):
        ...
```

**Note for agriculture/manufacturing mock print strings:** Lines like

```
print("  result['final'].classifications > disease type per crop")
```

are just documentation strings (not executable attribute access), but they're misleading — they should say `.predictions`.

---

## Planned Removal

### REMOVAL-001 — `valkey_rtsp_pipeline.py` requires unavailable live infrastructure

**File:** `examples/graph/valkey_rtsp_pipeline.py`  
**Status:** Scheduled for removal

This example requires BOTH:

1. A live RTSP camera stream at a real URL
2. A running Valkey server at `valkey://localhost:6379`

The example raises `RuntimeError` immediately if the RTSP stream cannot be opened, with no mock fallback or dry-run mode. This makes it impossible to run in any development, CI, or demo environment without real hardware.

**Contrast with similar examples that ARE kept:**

- `examples/graph/valkey_pipeline.py` — has `--mock` flag with sample data
- `examples/track/cross_camera_reid.py` — has full mock mode
- `examples/track/stream_tracking.py` — synthetic frame generator in mock mode

**Resolution:** Remove the file. Any future maintainer interested in Valkey + RTSP integration can refer to `valkey_pipeline.py` and `stream_tracking.py` as starting points with mock support.

---

## Notes

- All bugs in scenarios (BUG-004 through BUG-007) only manifest in `--real` mode (i.e., with `--real <image>` CLI flag). Mock mode (default, no args) runs correctly.
- BUG-004 through BUG-007 should ideally be fixed together in a single pass since they all require cross-referencing scenario files with their corresponding presets.
- `ClassifyResult` could benefit from adding `__getitem__` and `__iter__` convenience methods in a future non-breaking update (similar to how `MultiResult` was extended in this session).
