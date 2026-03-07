#!/usr/bin/env python3
"""Security: Crowd Monitoring with Alert System.

Real-World Problem:
    Security personnel in public spaces (transportation hubs, stadiums, malls)
    need to monitor crowd density in real-time to prevent overcrowding, manage
    foot traffic, and detect potential safety issues. Manual monitoring of
    multiple camera feeds is labor-intensive and prone to missed incidents.

Solution:
    Automated person detection with persistent tracking across video frames.
    BYTETrack assigns unique IDs to each person, enabling accurate crowd counting
    and individual tracking even in dense crowds. BotSort adds Global Motion
    Compensation (GMC) for cameras that pan or zoom. The system can trigger
    alerts when crowd density exceeds thresholds.

Models:
    - Detector: facebook/detr-resnet-50 (person detection)
    - Tracker: ByteTrackWrapper (ByteTrack) or BotSortWrapper (BotSort + GMC)

Graph Flow:
    Detect > Filter(person) > Track > Annotate > Fuse

Usage:
    python security_crowd_monitoring.py                       # Mock mode
    python security_crowd_monitoring.py --real video.mp4      # ByteTrack (default)
    python security_crowd_monitoring.py --real video.mp4 --botsort  # BotSort
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    USE_BOTSORT = "--botsort" in sys.argv
    video_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "surveillance.mp4")

    if USE_REAL:
        import mata
        from mata.nodes.track import BotSortWrapper, ByteTrackWrapper
        from mata.presets import crowd_monitoring, crowd_monitoring_botsort

        detector = mata.load("detect", "facebook/detr-resnet-50")

        if USE_BOTSORT:
            tracker = BotSortWrapper()
            graph = crowd_monitoring_botsort(detection_threshold=0.5)
            tracker_name = "BotSort"
        else:
            tracker = ByteTrackWrapper()
            graph = crowd_monitoring(detection_threshold=0.5)
            tracker_name = "ByteTrack"

        print(f"Processing video: {video_path}")
        print(f"Tracker: {tracker_name}")
        print("=" * 60)

        # For demonstration, process a single frame
        # In production, you would loop over video frames
        result = mata.infer(
            video_path,
            graph,
            providers={"detector": detector, "tracker": tracker},
        )

        print(f"Active tracks: {len(result['final'].instances)}")
        print()
        print("Detected individuals:")
        for inst in result["final"].instances:
            track_id = inst.track_id if hasattr(inst, "track_id") else "N/A"
            print(f"  Track ID {track_id}: {inst.label_name} at {inst.bbox} (score: {inst.score:.2f})")

        # Crowd density alert logic (example)
        crowd_count = len(result["final"].instances)
        CROWD_THRESHOLD = 15  # Example threshold
        if crowd_count > CROWD_THRESHOLD:
            print()
            print(f"⚠️  ALERT: Crowd density high ({crowd_count} persons, threshold: {CROWD_THRESHOLD})")
        else:
            print()
            print(f"✓ Crowd density normal ({crowd_count} persons)")

    else:
        print("=== Security: Crowd Monitoring with Alert System (Mock) ===")
        print()
        print("Graph: Detect > Filter(person) > Track > Annotate > Fuse")
        print("Models: DETR (detector) + ByteTrackWrapper or BotSortWrapper (tracker)")
        print()
        print("Tracker options:")
        print("  ByteTrackWrapper  — default, fast IOu-based tracking")
        print("  BotSortWrapper    — GMC-enabled, better for panning/zooming cameras")
        print()
        print("Expected output structure:")
        print("  result['final'].instances > list of tracked person instances")
        print("  Each instance has:")
        print("    - track_id: unique persistent ID across frames")
        print("    - bbox: bounding box coordinates")
        print("    - score: detection confidence")
        print()
        print("Use case:")
        print("  - Count unique individuals in a scene")
        print("  - Track movement patterns across video frames")
        print("  - Trigger alerts when crowd density exceeds thresholds")
        print("  - Analyze foot traffic in restricted/monitored areas")
        print()

        # Verify both preset constructions
        from mata.presets import crowd_monitoring, crowd_monitoring_botsort

        graph_bt = crowd_monitoring()
        graph_bs = crowd_monitoring_botsort()
        print(f"ByteTrack graph '{graph_bt.name}' constructed with {len(graph_bt._nodes)} nodes")
        print(f"BotSort graph   '{graph_bs.name}' constructed with {len(graph_bs._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph_bt._nodes]}")
        print()
        print("Run with --real <video.mp4> for actual inference.")
        print("Run with --real <video.mp4> --botsort for BotSort tracking.")


if __name__ == "__main__":
    main()
