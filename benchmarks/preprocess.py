#!/usr/bin/env python3
"""Measure letterbox preprocessing latency for PIL vs ndarray inputs.

Both paths produce the same NCHW float32 tensor; only the input plumbing
differs. Passing an ndarray to ``letterbox`` avoids two PIL round-trips.

Usage:
    python benchmarks/preprocess.py --frame frame.jpg --size 960
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mata.core.preprocess import letterbox


def _tensor(canvas: np.ndarray) -> np.ndarray:
    tensor = np.empty((1, 3, canvas.shape[0], canvas.shape[1]), dtype=np.float32)
    np.copyto(tensor[0], canvas.transpose(2, 0, 1))
    tensor /= 255.0
    return tensor


def _pil_path(array: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(array).convert("RGB")
    canvas, _ = letterbox(pil, (size, size))
    return _tensor(canvas)


def _array_path(array: np.ndarray, size: int) -> np.ndarray:
    canvas, _ = letterbox(array, (size, size))
    return _tensor(canvas)


def _bench(fn, runs: int) -> float:
    fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--size", type=int, default=960)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    bgr = cv2.imread(args.frame)
    if bgr is None:
        raise SystemExit(f"Could not read {args.frame}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]

    pil_tensor = _pil_path(rgb, args.size)
    array_tensor = _array_path(rgb, args.size)
    identical = np.array_equal(pil_tensor, array_tensor)

    pil_ms = _bench(lambda: _pil_path(rgb, args.size), args.runs)
    array_ms = _bench(lambda: _array_path(rgb, args.size), args.runs)
    result = {
        "frame": args.frame,
        "source_size": [width, height],
        "input_size": args.size,
        "runs": args.runs,
        "pil_ms": pil_ms,
        "array_ms": array_ms,
        "speedup": pil_ms / array_ms,
        "tensors_identical": bool(identical),
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
