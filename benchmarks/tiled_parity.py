#!/usr/bin/env python3
"""Compare tiled detection merging against a SAHI slicing + merge oracle.

The script runs the same detector once per slice (and once on the full image),
then applies two merge policies to the identical candidate set:

* MATA: ``mata.core.tiling.merge_instances`` with ``mode="nms"`` or ``"nmm"``.
* Oracle: SAHI ``NMSPostprocess`` or ``GreedyNMMPostprocess`` at the same IoU.

Because inference is shared, the comparison isolates slicing geometry and
duplicate handling — the parts MATA implements.

Usage:
    python benchmarks/tiled_parity.py \
        --frames /path/to/frames \
        --model /path/to/model.onnx \
        --limit 24 \
        --json-out /tmp/tiled_parity.json

SAHI is an optional, test-only oracle (``pip install sahi``); it is not a
runtime dependency of MATA.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from mata.core.artifacts.image import Image
from mata.core.tiling import merge_instances
from mata.core.types import Instance

try:
    from sahi.postprocess.combine import GreedyNMMPostprocess, NMSPostprocess
    from sahi.prediction import ObjectPrediction
    from sahi.slicing import get_slice_bboxes
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("This benchmark needs the optional oracle: pip install sahi") from exc


def _load_frames(source: str, limit: int) -> list[Path]:
    path = Path(source)
    if path.is_dir():
        frames = sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    elif path.suffix == ".json":
        frames = [Path(p) for p in json.loads(path.read_text())]
    else:
        frames = [Path(p) for p in source.split(",")]
    return frames[:limit]


def _slice_boxes(pil: PILImage.Image, slice_size: int, overlap: float) -> list[list[int]]:
    return get_slice_bboxes(
        image_height=pil.height,
        image_width=pil.width,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )


def _collect_candidates(
    adapter: Any, pil: PILImage.Image, slice_size: int, overlap: float, conf: float
) -> list[Instance]:
    """Run the detector on every SAHI slice and the full image; map to source pixels."""
    candidates: list[Instance] = []
    for x1, y1, x2, y2 in _slice_boxes(pil, slice_size, overlap):
        crop = pil.crop((x1, y1, x2, y2))
        result = adapter.predict(Image.from_pil(crop), conf=conf)
        for inst in result.instances:
            if inst.bbox is None:
                continue
            b = inst.bbox
            candidates.append(
                Instance(
                    bbox=(float(b[0]) + x1, float(b[1]) + y1, float(b[2]) + x1, float(b[3]) + y1),
                    score=float(inst.score),
                    label=int(inst.label),
                    label_name=inst.label_name,
                )
            )
    result = adapter.predict(Image.from_pil(pil), conf=conf)
    for inst in result.instances:
        if inst.bbox is None:
            continue
        candidates.append(
            Instance(
                bbox=tuple(float(v) for v in inst.bbox),
                score=float(inst.score),
                label=int(inst.label),
                label_name=inst.label_name,
            )
        )
    return candidates


def _to_object_predictions(instances: Iterable[Instance]) -> list[ObjectPrediction]:
    return [
        ObjectPrediction(
            bbox=[float(v) for v in inst.bbox],
            score=float(inst.score),
            category_id=int(inst.label),
            category_name=str(inst.label_name or inst.label),
        )
        for inst in instances
        if inst.bbox is not None
    ]


def _from_object_predictions(predictions: Iterable[ObjectPrediction]) -> list[Instance]:
    return [
        Instance(
            bbox=tuple(float(v) for v in pred.bbox.to_xyxy()),
            score=float(pred.score.value),
            label=int(pred.category.id),
            label_name=pred.category.name,
        )
        for pred in predictions
    ]


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match(predicted: list[Instance], reference: list[Instance], iou_threshold: float) -> dict[str, float]:
    """Class-aware greedy matching, highest score first."""
    refs = [inst for inst in reference if inst.bbox is not None]
    used: set[int] = set()
    matched_ious: list[float] = []
    tp = 0
    for pred in sorted((p for p in predicted if p.bbox is not None), key=lambda p: float(p.score), reverse=True):
        best_index = -1
        best_iou = 0.0
        for index, ref in enumerate(refs):
            if index in used or int(ref.label) != int(pred.label):
                continue
            value = _iou(tuple(pred.bbox), tuple(ref.bbox))
            if value > best_iou:
                best_iou, best_index = value, index
        if best_index >= 0 and best_iou >= iou_threshold:
            used.add(best_index)
            matched_ious.append(best_iou)
            tp += 1
    fp = len(predicted) - tp
    fn = len(refs) - tp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 1.0
    return {
        "pred": float(len(predicted)),
        "ref": float(len(refs)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "f1": f1,
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 1.0,
        "count_delta_pct": (len(predicted) - len(refs)) / len(refs) * 100 if refs else 0.0,
    }


def _summarize(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    values = [row[key] for row in rows]
    return {"mean": statistics.fmean(values), "max": max(values), "min": min(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, help="Directory, JSON list, or comma-separated image paths")
    parser.add_argument("--model", required=True, help="ONNX/TorchScript checkpoint path")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--slice", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    import mata

    adapter = mata.load("detect", args.model, device=args.device)
    frames = _load_frames(args.frames, args.limit)
    if not frames:
        raise SystemExit("No frames found")

    per_frame: list[dict[str, Any]] = []
    candidate_time = 0.0
    for index, frame in enumerate(frames, 1):
        pil = PILImage.open(frame).convert("RGB")
        start = time.perf_counter()
        candidates = _collect_candidates(adapter, pil, args.slice, args.overlap, args.conf)
        candidate_time += time.perf_counter() - start

        oracle_objects = _to_object_predictions(candidates)
        row: dict[str, Any] = {"frame": str(frame), "size": list(pil.size), "candidates": len(candidates)}
        for policy, oracle in (
            ("nms", NMSPostprocess(match_threshold=args.iou, match_metric="IOU", class_agnostic=False)),
            ("nmm", GreedyNMMPostprocess(match_threshold=args.iou, match_metric="IOU", class_agnostic=False)),
        ):
            start = time.perf_counter()
            mata_merged = merge_instances(candidates, iou_threshold=args.iou, mode=policy)
            mata_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            oracle_merged = _from_object_predictions(oracle(copy.deepcopy(oracle_objects)))
            oracle_ms = (time.perf_counter() - start) * 1000
            row[policy] = {
                "mata_ms": mata_ms,
                "oracle_ms": oracle_ms,
                **_match(mata_merged, oracle_merged, args.iou),
            }
        per_frame.append(row)
        print(
            f"[{index}/{len(frames)}] {frame.name} candidates={len(candidates)} "
            f"nms(f1={row['nms']['f1']:.3f}, iou={row['nms']['mean_iou']:.3f}, "
            f"delta={row['nms']['count_delta_pct']:+.2f}%) "
            f"nmm(f1={row['nmm']['f1']:.3f}, iou={row['nmm']['mean_iou']:.3f}, "
            f"delta={row['nmm']['count_delta_pct']:+.2f}%)"
        )

    summary = {
        policy: {
            "count_delta_pct": _summarize([row[policy] for row in per_frame], "count_delta_pct"),
            "f1": _summarize([row[policy] for row in per_frame], "f1"),
            "mean_iou": _summarize([row[policy] for row in per_frame], "mean_iou"),
            "mata_ms": _summarize([row[policy] for row in per_frame], "mata_ms"),
            "oracle_ms": _summarize([row[policy] for row in per_frame], "oracle_ms"),
        }
        for policy in ("nms", "nmm")
    }
    result = {
        "config": {
            "model": args.model,
            "slice": args.slice,
            "overlap": args.overlap,
            "conf": args.conf,
            "iou": args.iou,
            "frames": len(frames),
            "oracle": "sahi",
        },
        "candidate_time_s": candidate_time,
        "candidates_per_frame": statistics.fmean(row["candidates"] for row in per_frame),
        "summary": summary,
        "per_frame": per_frame,
    }

    print("\npolicy  delta_mean%  delta_max%   f1_mean  iou_mean  mata_ms  oracle_ms")
    for policy in ("nms", "nmm"):
        s = summary[policy]
        print(
            f"{policy:6s}  {s['count_delta_pct']['mean']:+10.3f}  {s['count_delta_pct']['max']:+9.3f}  "
            f"{s['f1']['mean']:.3f}    {s['mean_iou']['mean']:.3f}     {s['mata_ms']['mean']:.1f}     {s['oracle_ms']['mean']:.1f}"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
