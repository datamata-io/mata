#!/usr/bin/env python3
"""Frame-by-frame tracking with persistent state.

Demonstrates the YOLO-like ``persist=True`` pattern where you manage the
frame loop yourself while the tracker keeps state between calls.  Useful when
you need custom pre/post-processing around each frame — e.g. ROI gating,
per-frame business logic, or integration with an existing pipeline.

Demonstrates:
- ``mata.load("track", ...)`` to obtain a reusable adapter
- ``adapter.update(frame, persist=True)`` for manual frame loops
- ``VisionResult.save(show_track_ids=True)`` to annotate a frame
- Graceful OpenCV window display with 'q'-to-quit

Usage:
    # Mock mode (synthesises detections, no GPU)
    python examples/track/persist_tracking.py

    # Real video + model
    python examples/track/persist_tracking.py --real examples/videos/cup.mp4

Requirements:
    pip install datamata opencv-python
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _build_mock_adapter():
    """Mock TrackingAdapter that never needs a real model."""
    from unittest.mock import Mock

    from mata.core.types import Instance, VisionResult

    call_count = {"n": 0}

    def mock_update(image, **kwargs):
        n = call_count["n"]
        call_count["n"] += 1
        x = 80 + n * 5
        return VisionResult(
            instances=[
                Instance(bbox=(x, 50, x + 100, 300), label=0,
                            score=0.88, label_name="person", track_id=1),
                Instance(bbox=(350, 120, 490, 280), label=2,
                            score=0.79, label_name="car", track_id=2),
            ],
            meta={"frame_idx": n, "tracker": "bytetrack"},
        )

    adapter = Mock()
    adapter.update.side_effect = mock_update
    return adapter


def _build_mock_video(path: Path, num_frames: int = 30) -> Path:
    """Write a tiny synthetic AVI to *path*."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 15.0, (640, 480))
    for i in range(num_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x = 80 + i * 5
        cv2.rectangle(frame, (x, 50), (x + 100, 300), (0, 200, 0), -1)
        cv2.rectangle(frame, (350, 120), (490, 280), (200, 0, 0), -1)
        writer.write(frame)
    writer.release()
    return path


# ---------------------------------------------------------------------------
# Core tracking loop
# ---------------------------------------------------------------------------

def run_persist_loop(
    adapter,
    video_path: str,
    *,
    show: bool = False,
    max_frames: int | None = None,
) -> list:
    """Process *video_path* frame-by-frame using *adapter* with persist=True.

    Args:
        adapter: A loaded ``TrackingAdapter`` (or compatible mock).
        video_path: Path to a video file.
        show: Open an OpenCV window to display results live.
        max_frames: Stop after this many frames (None = all frames).

    Returns:
        List of VisionResult objects (one per processed frame).
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for the persist loop. "
            "Install with: pip install opencv-python"
        ) from exc

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path!r}")

    results = []
    frame_idx = 0

    try:
        while cap.isOpened():
            if max_frames is not None and frame_idx >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # -------------------------------------------------------------- #
            # Core pattern: update adapter with persist=True each frame       #
            # -------------------------------------------------------------- #
            result = adapter.update(frame, persist=True)
            results.append(result)

            # Optional: annotate and display
            if show:
                # Build a simple overlay using PIL (no extra deps)
                try:
                    from PIL import Image as PILImage
                    import numpy as np

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = PILImage.fromarray(rgb)
                    annotated_pil = result.save(
                        pil_img, show_track_ids=True, return_image=True
                    )
                    if annotated_pil is not None:
                        annotated_bgr = cv2.cvtColor(
                            annotated_pil if isinstance(annotated_pil, type(frame))
                            else cv2.imdecode(
                                __import__("numpy").frombuffer(
                                    annotated_pil if isinstance(annotated_pil, bytes)
                                    else b"", dtype="uint8"
                                ),
                                cv2.IMREAD_COLOR,
                            ),
                            cv2.COLOR_RGB2BGR,
                        )
                        cv2.imshow("MATA Tracking (persist)", annotated_bgr)
                except Exception:  # noqa: BLE001
                    cv2.imshow("MATA Tracking (persist)", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nUser pressed 'q' — stopping early.")
                    break

            frame_idx += 1

    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Persist-mode MATA tracking (manual frame loop)"
    )
    parser.add_argument("--real", metavar="VIDEO",
                        help="Path to a real video file")
    parser.add_argument("--model", default="facebook/detr-resnet-50",
                        help="Detection model ID")
    parser.add_argument("--tracker", default="bytetrack",
                        choices=["bytetrack", "botsort"],
                        help="Tracker algorithm")
    parser.add_argument("--show", action="store_true",
                        help="Display annotated frames in a window")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    if args.real:
        # ------------------------------------------------------------------ #
        # Real mode                                                            #
        # ------------------------------------------------------------------ #
        import mata

        print(f"[real] Loading model='{args.model}', tracker='{args.tracker}' …")
        adapter = mata.load("track", args.model, tracker=args.tracker)

        print(f"[real] Processing '{args.real}' …")
        results = run_persist_loop(
            adapter,
            args.real,
            show=args.show,
            max_frames=args.max_frames,
        )

    else:
        # ------------------------------------------------------------------ #
        # Mock mode                                                            #
        # ------------------------------------------------------------------ #
        import tempfile

        print("[mock] Running with synthetic detections (no model required).")
        print("       Pass --real <video.mp4> to use a real model.\n")

        tmp_video = Path(tempfile.mkdtemp()) / "mock_persist.avi"
        _build_mock_video(tmp_video, num_frames=30)

        adapter = _build_mock_adapter()
        try:
            import cv2  # noqa: F401
            results = run_persist_loop(
                adapter,
                str(tmp_video),
                show=False,
                max_frames=args.max_frames,
            )
        except ImportError:
            # cv2 unavailable — simulate results directly
            print("(OpenCV not installed — simulating 10 frames without video I/O)\n")
            results = [adapter.update(None) for _ in range(10)]

    # ---------------------------------------------------------------------- #
    # Print summary                                                           #
    # ---------------------------------------------------------------------- #
    print(f"\nProcessed {len(results)} frames.\n")
    for frame_idx, result in enumerate(results[:5]):   # print first 5 frames
        for inst in result.instances:
            tid = f"#{inst.track_id}" if inst.track_id is not None else "untracked"
            print(f"  Frame {frame_idx:3d}  {tid:>6}  {inst.label_name:<12}  "
                    f"score={inst.score:.2f}  bbox={inst.bbox}")
    if len(results) > 5:
        print(f"  … ({len(results) - 5} more frames)\n")

    unique_ids: set[int] = set()
    for r in results:
        for inst in r.instances:
            if inst.track_id is not None:
                unique_ids.add(inst.track_id)
    print(f"Unique track IDs seen: {sorted(unique_ids)}\n")


if __name__ == "__main__":
    main()
