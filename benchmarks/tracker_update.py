#!/usr/bin/env python3
"""Measure BYTETracker.update() latency on a fixed detection stream.

Input JSON format::

    {"frames": [{"dets": [[x1, y1, x2, y2, score, cls], ...]}, ...]}

The same stream is replayed for every run, so changes in median update time
reflect tracker code, not detector variance.

Usage:
    python benchmarks/tracker_update.py --dets dets.json --preset ultra_parity
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from mata.trackers.byte_tracker import BYTETracker, DetectionResults

PRESETS = {
    "mata_default": dict(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    ),
    "ultra_parity": dict(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    ),
}


def _detection_results(rows: list[list[float]]) -> DetectionResults:
    dets = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    xyxy = dets[:, :4]
    conf = dets[:, 4]
    cls = dets[:, 5]
    xywh = (
        np.stack(
            [
                (xyxy[:, 0] + xyxy[:, 2]) / 2,
                (xyxy[:, 1] + xyxy[:, 3]) / 2,
                xyxy[:, 2] - xyxy[:, 0],
                xyxy[:, 3] - xyxy[:, 1],
            ],
            axis=1,
        )
        if len(dets)
        else np.zeros((0, 4), dtype=np.float32)
    )
    return DetectionResults(conf=conf, xyxy=xyxy, xywh=xywh, cls=cls)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dets", required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="ultra_parity")
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    payload = json.loads(Path(args.dets).read_text())
    frames = payload["frames"]
    tracker = BYTETracker(PRESETS[args.preset], frame_rate=args.frame_rate)

    timings: list[float] = []
    last_tracks = 0
    for index, record in enumerate(frames):
        start = time.perf_counter()
        tracks = tracker.update(_detection_results(record["dets"]))
        elapsed = (time.perf_counter() - start) * 1000
        last_tracks = len(tracks)
        if index >= args.warmup:
            timings.append(elapsed)

    result = {
        "preset": args.preset,
        "frames": len(frames),
        "warmup": args.warmup,
        "median_update_ms": statistics.median(timings),
        "p95_update_ms": float(np.percentile(timings, 95)),
        "last_track_count": last_tracks,
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
