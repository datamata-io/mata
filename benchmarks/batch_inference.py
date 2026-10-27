#!/usr/bin/env python3
"""Measure ONNX detection throughput for batched inference.

Runs the same set of images through ``predict_batch`` at several batch sizes,
after a warm-up, and reports images/second and per-image latency. Batch 1 is
the sequential reference (one ``predict`` call per image).

The model must declare a dynamic batch dimension; static batch-1 models fall
back to sequential inference and will show no gain.

Usage:
    python benchmarks/batch_inference.py \
        --model /path/to/dynamic.onnx \
        --frame /path/to/frame.jpg \
        --batch-sizes 1,2,4,8 \
        --threads 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

import mata


def _crops(frame: Path, size: int, count: int) -> list[np.ndarray]:
    pil = PILImage.open(frame).convert("RGB")
    images: list[np.ndarray] = []
    step = max(size, 1)
    for index in range(count):
        x = (index * step) % max(pil.width - size, 1)
        y = (index * step * 2) % max(pil.height - size, 1)
        images.append(np.array(pil.crop((x, y, x + size, y + size))))
    return images


def _max_bbox_delta(left, right) -> float:
    delta = 0.0
    for a, b in zip(left, right):
        if len(a.instances) != len(b.instances):
            return float("inf")
        for inst_a, inst_b in zip(a.instances, b.instances):
            delta = max(delta, max(abs(x - y) for x, y in zip(inst_a.bbox, inst_b.bbox)))
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    adapter = mata.load(
        "detect",
        args.model,
        device="cpu",
        session_options={"intra_op_num_threads": args.threads, "inter_op_num_threads": 1},
    )
    if not adapter.supports_batch:
        print("WARNING: model declares a static batch dimension; predict_batch will fall back")

    images = _crops(Path(args.frame), args.size, args.images)
    batch_sizes = [int(size) for size in args.batch_sizes.split(",") if size]

    parity = _max_bbox_delta(
        adapter.predict_batch(images[:4], batch_size=4),
        [adapter.predict(image) for image in images[:4]],
    )
    print(f"batch-vs-loop max bbox delta: {parity:.6f}")

    results = {}
    print("\nbatch  images/s  ms/image  speedup")
    baseline = None
    for batch_size in batch_sizes:
        adapter.predict_batch(images[:batch_size], batch_size=batch_size)  # warm-up
        samples = []
        for _ in range(args.rounds):
            start = time.perf_counter()
            out = adapter.predict_batch(images, batch_size=batch_size)
            elapsed = time.perf_counter() - start
            assert len(out) == len(images)
            samples.append(len(images) / elapsed)
        throughput = statistics.median(samples)
        if baseline is None:
            baseline = throughput
        results[batch_size] = {
            "images_per_s": throughput,
            "ms_per_image": 1000.0 / throughput,
            "speedup": throughput / baseline,
        }
        print(f"{batch_size:5d}  {throughput:8.2f}  {1000.0 / throughput:8.1f}  " f"{throughput / baseline:6.2f}x")

    if args.json_out:
        payload = {
            "model": args.model,
            "images": args.images,
            "image_size": args.size,
            "threads": args.threads,
            "max_bbox_delta_vs_loop": parity,
            "results": results,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
