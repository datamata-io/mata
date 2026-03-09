#!/usr/bin/env python3
"""Basic video tracking with MATA.

Tracks objects in a video file using the combined detect+track API.
Saves an annotated output video with track IDs drawn on each detection.

Demonstrates:
- ``mata.track()`` simple one-call API
- BotSort tracker (default) with track ID rendering
- Iterating per-frame VisionResult objects
- Accessing Instance.track_id for downstream logic

Usage:
    # Mock mode (no models, synthesises detections)
    python examples/track/basic_tracking.py

    # Real video + model
    python examples/track/basic_tracking.py --real examples/videos/cup.mp4

Requirements:
    pip install datamata opencv-python
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Mock helpers (used when --real is not supplied)
# ---------------------------------------------------------------------------

def _build_mock_adapter():
    """Return a lightweight TrackingAdapter-compatible mock.

    Generates a 2-second sequence of synthetic detections so the example
    runs without a GPU or real video file.
    """
    from unittest.mock import Mock

    from mata.core.types import Instance, VisionResult

    call_count = {"n": 0}

    def mock_update(image, **kwargs):
        n = call_count["n"]
        call_count["n"] += 1
        x = 100 + n * 4          # person drifts right
        return VisionResult(
            instances=[
                Instance(bbox=(x, 60, x + 80, 300), label=0,
                            score=0.91, label_name="person", track_id=1),
                Instance(bbox=(400, 100, 540, 280), label=2,
                            score=0.83, label_name="car", track_id=2),
            ],
            meta={"frame_idx": n, "tracker": "botsort"},
        )

    adapter = Mock()
    adapter.update.side_effect = mock_update
    return adapter


def _build_mock_video(path: Path, num_frames: int = 30) -> Path:
    """Write a tiny synthetic AVI to *path* and return it."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return path  # caller will skip video processing

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 15.0, (640, 480))
    for i in range(num_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x = 100 + i * 4
        cv2.rectangle(frame, (x, 60), (x + 80, 300), (0, 200, 0), -1)
        cv2.rectangle(frame, (400, 100), (540, 280), (0, 0, 200), -1)
        writer.write(frame)
    writer.release()
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Basic MATA tracking example")
    parser.add_argument("--real", metavar="VIDEO",
                        help="Path to a real video file (requires a GPU / model download)")
    parser.add_argument("--model", default="facebook/detr-resnet-50",
                        help="Detection model ID (HuggingFace or config alias)")
    parser.add_argument("--tracker", default="botsort",
                        choices=["botsort", "bytetrack"],
                        help="Tracker algorithm")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated video to runs/track/")
    args = parser.parse_args(argv)

    import mata

    if args.real:
        # ------------------------------------------------------------------ #
        # Real mode: actual model + video file                                #
        # ------------------------------------------------------------------ #
        video_path = args.real
        print(f"[real] Tracking '{video_path}' with model='{args.model}' "
                f"tracker='{args.tracker}' conf={args.conf}")

        results = mata.track(
            video_path,
            model=args.model,
            tracker=args.tracker,
            conf=args.conf,
            save=args.save,
            show_track_ids=True,
        )

    else:
        # ------------------------------------------------------------------ #
        # Mock mode: synthetic detections, no GPU required                    #
        # ------------------------------------------------------------------ #
        import tempfile

        print("[mock] Running with synthetic detections (no model required).")
        print("       Pass --real <video.mp4> to use a real model.\n")

        tmp_video = Path(tempfile.mkdtemp()) / "mock_video.avi"
        _build_mock_video(tmp_video, num_frames=30)

        # Patch mata.load so it returns the mock adapter
        from unittest.mock import patch
        mock_adapter = _build_mock_adapter()
        with patch("mata.api.load", return_value=mock_adapter):
            results = mata.track(
                str(tmp_video),
                model=args.model,
                tracker=args.tracker,
                conf=args.conf,
                show_track_ids=True,
            )

    # ---------------------------------------------------------------------- #
    # Inspect results                                                          #
    # ---------------------------------------------------------------------- #
    print(f"\nProcessed {len(results)} frames.\n")

    for frame_idx, result in enumerate(results):
        if not result.instances:
            continue
        for inst in result.instances:
            track_id_str = f"#{inst.track_id}" if inst.track_id is not None else "untracked"
            print(
                f"Frame {frame_idx:4d}  Track {track_id_str:>6}  "
                f"{inst.label_name:<12}  score={inst.score:.2f}  bbox={inst.bbox}"
            )

    # Summary
    unique_ids: set[int] = set()
    for r in results:
        for inst in r.instances:
            if inst.track_id is not None:
                unique_ids.add(inst.track_id)

    print(f"\nSummary: {len(results)} frames, {len(unique_ids)} unique track IDs.\n")
    if args.save and (not args.real):
        print("(Mock mode: no file saved — run with --real to save annotated video.)")


if __name__ == "__main__":
    main()
