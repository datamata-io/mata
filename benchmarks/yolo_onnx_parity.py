"""Score a MATA ONNX detection adapter against a recorded reference result set.

Reference JSON format (one entry per frame):
    {"frames": [{"frame": "name.jpg", "dets": [[x1,y1,x2,y2,label,score], ...]}, ...]}

Usage:
    python benchmarks/yolo_onnx_parity.py \
        --model model.onnx --images frames/ --reference reference.json \
        --conf 0.35 --iou 0.45 --max-det 300
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from mata.adapters.onnx_adapter import ONNXDetectAdapter


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0][:, None], a[:, 1][:, None], a[:, 2][:, None], a[:, 3][:, None]
    bx1, by1, bx2, by2 = b[:, 0][None], b[:, 1][None], b[:, 2][None], b[:, 3][None]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    aa = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    ba = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    return inter / (aa + ba - inter + 1e-9)


def greedy(pred: np.ndarray, ref: np.ndarray, thr: float = 0.5) -> list[tuple[int, int, float]]:
    m = iou_matrix(pred, ref)
    pairs: list[tuple[int, int, float]] = []
    if m.size == 0:
        return pairs
    for flat in np.argsort(-m, axis=None):
        i, j = np.unravel_index(flat, m.shape)
        if m[i, j] < thr:
            break
        if any(p[0] == i for p in pairs) or any(p[1] == j for p in pairs):
            continue
        pairs.append((int(i), int(j), float(m[i, j])))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ref = {
        Path(r["frame"]).name: np.array(r["dets"], float).reshape(-1, 6)
        for r in json.load(open(args.reference))["frames"]
    }
    adapter = ONNXDetectAdapter(
        args.model,
        device=args.device,
        threshold=args.conf,
        iou_threshold=args.iou,
        max_det=args.max_det,
    )

    tp = fp = fn = 0
    ious: list[float] = []
    times: list[float] = []
    for path in sorted(Path(args.images).glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        t0 = time.perf_counter()
        result = adapter.predict(str(path))
        times.append((time.perf_counter() - t0) * 1000)
        pred = np.array(
            [[i.bbox[0], i.bbox[1], i.bbox[2], i.bbox[3], i.label, i.score] for i in result.instances], float
        )
        pred = pred.reshape(-1, 6)
        truth = ref.get(path.name, np.zeros((0, 6)))
        pairs = []
        for cls in set(pred[:, 4].astype(int)) | set(truth[:, 4].astype(int)):
            pi = np.where(pred[:, 4].astype(int) == cls)[0]
            ti = np.where(truth[:, 4].astype(int) == cls)[0]
            pairs.extend(greedy(pred[pi], truth[ti]))
        tp += len(pairs)
        fp += len(pred) - len(pairs)
        fn += len(truth) - len(pairs)
        ious.extend(p[2] for p in pairs)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    print(
        json.dumps(
            {
                "frames": len(times),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
                "median_ms": round(statistics.median(times), 1),
                "p95_ms": round(float(np.percentile(times, 95)), 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
