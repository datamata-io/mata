#!/usr/bin/env python3
"""RTSP stream tracking with Valkey result storage.

Demonstrates per-frame BotSort object tracking on an RTSP stream where every
frame's results are:

  - Stored as an indexed snapshot  ``rtsp:cam01:tracks:<frame_idx>`` (TTL 1 h)
  - Written to a rolling latest key ``rtsp:cam01:tracks:latest``  (TTL 15 s)
  - Broadcast on Pub/Sub channel   ``rtsp:cam01:track:events``

A round-trip verification load is performed after the loop.

Requirements:
    pip install datamata[valkey]       # Valkey / Redis client
    pip install datamata opencv-python # RTSP frame capture

Usage:
    python examples/graph/rtsp_pipeline.py

    # Override the Valkey URL or RTSP source
    python examples/graph/rtsp_pipeline.py \\
        --url valkey://myhost:6379 \\
        --rtsp rtsp://example:example@192.168.1.10:8554/Streaming/Channels/102

"""

from __future__ import annotations

import sys


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _arg(flag: str, default: str) -> str:
    """Return the value after *flag* in sys.argv, or *default*."""
    for i, arg in enumerate(sys.argv):
        if arg == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def create_tracker(rtsp_url: str):
    """Load a real RT-DETR detector and wrap it with a BotSort tracker."""
    import mata
    from mata.adapters.tracking_adapter import TrackingAdapter

    print("Loading PekingU/rtdetr_r18vd from HuggingFace (this may take a moment)...")
    detector = mata.load("detect", "PekingU/rtdetr_r18vd")
    tracker = TrackingAdapter(detector, tracker_config="botsort")
    print(f"BotSort tracker ready — RTSP source: {rtsp_url}")
    return tracker


# ---------------------------------------------------------------------------
# Main example
# ---------------------------------------------------------------------------

def run(url: str, rtsp_url: str, tracker, frames: int = 8):
    """Run the per-frame tracking + Valkey storage loop.

    Args:
        url:      Valkey server URL (e.g. ``valkey://localhost:6379``).
        rtsp_url: RTSP stream URL to open with cv2.VideoCapture.
        tracker:  :class:`~mata.adapters.tracking_adapter.TrackingAdapter`.
        frames:   Maximum number of frames to process (0 = unlimited).
    """
    import cv2
    from PIL import Image as PILImage

    from mata.core.exporters import export_valkey, load_valkey, publish_valkey

    TRACK_KEY_PREFIX = "rtsp:cam01:tracks"
    LATEST_KEY       = "rtsp:cam01:tracks:latest"
    CHANNEL          = "rtsp:cam01:track:events"

    print(f"\nOpening RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {rtsp_url}")

    print(f"Processing up to {frames} frame(s)...")
    print(f"  Indexed snapshots : {TRACK_KEY_PREFIX}:<frame_idx>  (TTL 1 h)")
    print(f"  Rolling latest    : {LATEST_KEY}  (TTL 15 s)")
    print(f"  Pub/Sub channel   : {CHANNEL}")
    print()

    track_counts: list[int] = []
    frame_idx = 0

    try:
        while frames == 0 or frame_idx < frames:
            ret, bgr_frame = cap.read()
            if not ret:
                print("Stream ended or frame read failed — stopping.")
                break

            pil_frame = PILImage.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
            result = tracker.update(pil_frame, persist=True)

            active = [inst for inst in result.instances if inst.track_id is not None]
            track_counts.append(len(active))

            # Per-frame indexed snapshot — long TTL for audit / replay
            export_valkey(
                result,
                url=url,
                key=f"{TRACK_KEY_PREFIX}:{frame_idx:06d}",
                ttl=3600,
            )
            # Rolling latest — short TTL so it auto-expires if the stream drops
            export_valkey(result, url=url, key=LATEST_KEY, ttl=15)

            # Broadcast to any live dashboards or alert subscribers
            n_subs = publish_valkey(result, url=url, channel=CHANNEL, serializer="json")

            ids = [str(inst.track_id) for inst in active]
            print(f"  Frame {frame_idx:02d}: {len(active)} track(s)  IDs={ids}"
                    f"  → {n_subs} subscriber(s) notified")

            frame_idx += 1
    finally:
        cap.release()

    if frame_idx == 0:
        print("No frames were processed.")
        return

    # ── Round-trip verification ──────────────────────────────────────────────
    print()
    latest = load_valkey(url=url, key=LATEST_KEY)
    active_latest = [i for i in latest.instances if i.track_id is not None]
    print(f"Latest key '{LATEST_KEY}': {len(active_latest)} active track(s)")
    for inst in active_latest:
        print(f"  Track #{inst.track_id}  {inst.label_name:<10}  "
            f"score={inst.score:.2f}  bbox={inst.bbox}")

    mid_idx = frame_idx // 2
    mid_key = f"{TRACK_KEY_PREFIX}:{mid_idx:06d}"
    mid = load_valkey(url=url, key=mid_key)
    mid_ids = [str(i.track_id) for i in mid.instances if i.track_id is not None]
    print(f"Mid-stream frame [{mid_idx}] ('{mid_key}'): "
            f"{len(mid.instances)} instance(s)  IDs={mid_ids}")

    avg = sum(track_counts) / len(track_counts) if track_counts else 0
    print(f"Average active tracks/frame: {avg:.1f}  over {frame_idx} frames")
    print("✓ RTSP tracking + Valkey storage example complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    url      = _arg("--url",   "valkey://localhost:6379")
    rtsp_url = _arg("--rtsp",  "rtsp://example:example@192.168.1.100:8554/Streaming/Channels/102")
    frames   = int(_arg("--frames", "8"))

    print(f"Valkey server : {url}")
    print(f"RTSP source   : {rtsp_url}")
    print("Make sure valkey-py is installed:  pip install datamata[valkey]")

    tracker = create_tracker(rtsp_url)
    run(url=url, rtsp_url=rtsp_url, tracker=tracker, frames=frames)


if __name__ == "__main__":
    main()
