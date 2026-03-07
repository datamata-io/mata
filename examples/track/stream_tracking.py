#!/usr/bin/env python3
"""Memory-efficient stream tracking with generator mode.

``mata.track(..., stream=True)`` returns a *generator* that yields one
VisionResult per frame.  The tracker never accumulates the full frame list
in memory, making it suitable for:
  - Long video files (hours of footage)
  - Live RTSP / HTTP streams
  - Webcam capture
  - Edge devices with limited RAM

Demonstrates:
- Generator-mode tracking with ``stream=True``
- Per-frame processing without buffering all results
- Active-track counting per frame
- Early stopping and clean resource cleanup

Usage:
    # Mock mode (no GPU, synthesises detections)
    python examples/track/stream_tracking.py

    # Real video in stream mode
    python examples/track/stream_tracking.py --real examples/videos/cup.mp4

    # Live RTSP stream
    python examples/track/stream_tracking.py --real rtsp://camera/stream

    # Webcam
    python examples/track/stream_tracking.py --real 0

Requirements:
    pip install mata opencv-python
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Generator


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_generator(num_frames: int = 40) -> Generator:
    """Yield synthetic VisionResult objects without any real model."""
    from mata.core.types import Instance, VisionResult

    for n in range(num_frames):
        x = 80 + n * 4
        instances = [
            Instance(bbox=(x, 50, x + 90, 290), label=0,
                     score=0.87, label_name="person", track_id=1),
        ]
        # Add a second track that appears at frame 10
        if n >= 10:
            instances.append(
                Instance(bbox=(380, 130, 500, 270), label=2,
                         score=0.76, label_name="car", track_id=2)
            )
        # Simulate a short-lived third track (frames 20-24)
        if 20 <= n < 25:
            instances.append(
                Instance(bbox=(200, 200, 260, 280), label=0,
                         score=0.65, label_name="person", track_id=3)
            )
        yield VisionResult(
            instances=instances,
            meta={"frame_idx": n, "tracker": "botsort"},
        )


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

def log_frame(frame_idx: int, result) -> None:
    """Print a compact summary for one frame's tracking result."""
    active = [i for i in result.instances if i.track_id is not None]
    id_list = sorted({i.track_id for i in active})
    labels = {i.track_id: i.label_name for i in active}
    label_strs = ", ".join(
        f"#{tid}:{labels[tid]}" for tid in id_list
    )
    print(f"  Frame {frame_idx:4d}  active={len(active):2d}  [{label_strs}]")


def stream_and_count(
    frame_generator,
    *,
    max_frames: int | None = None,
    alert_threshold: int = 3,
) -> dict:
    """Consume a stream of VisionResult objects and produce statistics.

    Args:
        frame_generator: Generator[VisionResult, None, None] from mata.track().
        max_frames: Stop after this many frames (None = consume fully).
        alert_threshold: Print an alert when active tracks exceed this.

    Returns:
        dict with keys: total_frames, unique_track_ids, max_concurrent.
    """
    unique_ids: set[int] = set()
    max_concurrent = 0
    total_frames = 0

    for frame_idx, result in enumerate(frame_generator):
        if max_frames is not None and frame_idx >= max_frames:
            break

        active_tracks = [i for i in result.instances if i.track_id is not None]
        concurrent = len(active_tracks)

        for inst in active_tracks:
            unique_ids.add(inst.track_id)

        max_concurrent = max(max_concurrent, concurrent)

        log_frame(frame_idx, result)

        if concurrent > alert_threshold:
            print(f"  *** ALERT: {concurrent} concurrent tracks "
                  f"(threshold={alert_threshold}) ***")

        total_frames += 1

    return {
        "total_frames": total_frames,
        "unique_track_ids": sorted(unique_ids),
        "max_concurrent": max_concurrent,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stream-mode MATA tracking (memory-efficient generator)"
    )
    parser.add_argument("--real", metavar="SOURCE",
                        help="Video path, RTSP URL, or webcam index (e.g. 0)")
    parser.add_argument("--model", default="facebook/detr-resnet-50",
                        help="Detection model ID")
    parser.add_argument("--tracker", default="botsort",
                        choices=["botsort", "bytetrack"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--alert-threshold", type=int, default=3,
                        help="Warn when active track count exceeds this")
    args = parser.parse_args(argv)

    print("MATA Stream Tracking Example")
    print("=" * 50)

    if args.real:
        # ------------------------------------------------------------------ #
        # Real mode: actual model + video / stream / webcam                   #
        # ------------------------------------------------------------------ #
        import mata

        # Parse webcam index if integer string
        source: str | int = args.real
        try:
            source = int(args.real)
        except ValueError:
            pass

        print(f"[real] Source: {source!r}")
        print(f"       Model: {args.model}")
        print(f"       Tracker: {args.tracker}  conf={args.conf}")
        print()

        # stream=True > returns a generator, never accumulates full result list
        frame_gen = mata.track(
            source,
            model=args.model,
            tracker=args.tracker,
            conf=args.conf,
            stream=True,           # ← key flag for memory efficiency
            show_track_ids=True,
            max_frames=args.max_frames,
        )

    else:
        # ------------------------------------------------------------------ #
        # Mock mode: synthetic generator                                       #
        # ------------------------------------------------------------------ #
        print("[mock] Using synthetic frame generator (no model required).")
        print("       Pass --real <source> to track a real video or stream.\n")

        # Wrap mock generator in patch so mata.track would use it if needed;
        # here we call the generator directly for simplicity.
        num = args.max_frames if args.max_frames is not None else 40
        frame_gen = _make_mock_generator(num_frames=num)

    # ---------------------------------------------------------------------- #
    # Consume the generator — never loads more than one frame at a time       #
    # ---------------------------------------------------------------------- #
    stats = stream_and_count(
        frame_gen,
        max_frames=args.max_frames,
        alert_threshold=args.alert_threshold,
    )

    # ---------------------------------------------------------------------- #
    # Summary                                                                 #
    # ---------------------------------------------------------------------- #
    print()
    print("=" * 50)
    print(f"Stream complete.")
    print(f"  Total frames processed : {stats['total_frames']}")
    print(f"  Unique track IDs seen  : {stats['unique_track_ids']}")
    print(f"  Max concurrent tracks  : {stats['max_concurrent']}")
    print()


if __name__ == "__main__":
    main()
