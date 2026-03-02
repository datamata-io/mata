#!/usr/bin/env python3
"""Autonomous Driving: Multi-Object Traffic Tracking.

Real-World Problem:
    Traffic monitoring and autonomous driving require persistent tracking
    of vehicles and pedestrians across video frames. Each object needs a
    unique track ID that persists even through occlusions, enabling
    trajectory prediction, behavior analysis, and collision avoidance.

Solution:
    Detects traffic participants in each frame, filters to vehicle/pedestrian
    classes, assigns persistent track IDs using BYTETrack or BotSort, and
    annotates frames with tracked bounding boxes. Designed for frame-by-frame
    video processing with temporal consistency. BotSort adds Global Motion
    Compensation (GMC) which is useful for cameras that can pan or tilt.

Models:
    - Detector: PekingU/rtdetr_v2_r18vd (fast real-time detection)
    - Tracker: ByteTrackWrapper (ByteTrack) or BotSortWrapper (BotSort + GMC)

Graph Flow:
    Detect → Filter(vehicle classes) → Track → Annotate → Fuse

Usage:
    python driving_traffic_tracking.py                          # Mock mode (simulates 30 frames)
    python driving_traffic_tracking.py --real video.mp4         # ByteTrack (default)
    python driving_traffic_tracking.py --real video.mp4 --botsort  # BotSort + GMC

Note:
    Real video processing requires opencv-python:
        pip install opencv-python
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    USE_BOTSORT = "--botsort" in sys.argv
    video_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "traffic.mp4")

    if USE_REAL:
        import mata
        from mata.presets import traffic_tracking, traffic_tracking_botsort

        try:
            import cv2
        except ImportError:
            print("Error: opencv-python required for video processing")
            print("Install with: pip install opencv-python")
            return

        # Load models
        detector = mata.load("detect", "PekingU/rtdetr_v2_r18vd")

        # Select tracker — ByteTrack (default) or BotSort (camera-motion robust)
        from mata.nodes.track import BotSortWrapper, ByteTrackWrapper

        if USE_BOTSORT:
            tracker = BotSortWrapper()
            graph = traffic_tracking_botsort(
                detection_threshold=0.5,
                vehicle_labels=["car", "truck", "bus", "person", "bicycle", "motorcycle"],
            )
            tracker_name = "BotSort"
        else:
            tracker = ByteTrackWrapper()
            graph = traffic_tracking(
                detection_threshold=0.5,
                vehicle_labels=["car", "truck", "bus", "person", "bicycle", "motorcycle"],
            )
            tracker_name = "ByteTrack"

        print(f"Tracker: {tracker_name}")
        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        track_history = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            result = mata.infer(
                frame,
                graph,
                providers={"detector": detector, "tracker": tracker},
            )

            # Display tracking results
            frame_count += 1
            num_tracks = len(result["final"].tracks) if hasattr(result["final"], "tracks") else 0

            # Update track history
            for inst in result["final"].instances:
                if hasattr(inst, "track_id") and inst.track_id is not None:
                    if inst.track_id not in track_history:
                        track_history[inst.track_id] = {"first_frame": frame_count, "last_frame": frame_count}
                    else:
                        track_history[inst.track_id]["last_frame"] = frame_count

            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {num_tracks} active tracks, "
                      f"{len(track_history)} total unique tracks")

            # Annotated frame is available in result["final"].image
            # (could display or save here)

        cap.release()

        print(f"\n=== Tracking Summary ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Unique tracks: {len(track_history)}")
        print(f"Average track duration: "
              f"{sum((t['last_frame'] - t['first_frame'] + 1) for t in track_history.values()) / len(track_history):.1f} frames")

    else:
        print("=== Autonomous Driving: Multi-Object Traffic Tracking (Mock) ===")
        print()
        print("Graph: Detect → Filter → Track → Annotate → Fuse")
        print("Models: RT-DETR (detector) + ByteTrackWrapper or BotSortWrapper (tracker)")
        print()
        print("Tracker options:")
        print("  ByteTrackWrapper  — default, fast IoU-based two-stage association")
        print("  BotSortWrapper    — GMC-enabled for panning/tilting cameras")
        print()
        print("Expected output structure (per frame):")
        print("  result['final'].instances → detected objects with track IDs")
        print("  result['final'].tracks → tracking state information")
        print("  result['final'].image → annotated frame with track visualizations")
        print()
        print("Video Processing Pattern:")
        print("  1. Create tracker instance ONCE before video loop")
        print("  2. Process each frame through same graph + tracker")
        print("  3. Track IDs persist across frames (handles occlusions)")
        print()

        # Verify both preset constructions
        from mata.presets import traffic_tracking, traffic_tracking_botsort

        graph_bt = traffic_tracking()
        graph_bs = traffic_tracking_botsort()
        print(f"ByteTrack graph '{graph_bt.name}' constructed with {len(graph_bt._nodes)} nodes")
        print(f"BotSort graph   '{graph_bs.name}' constructed with {len(graph_bs._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph_bt._nodes]}")
        print()
        print("Run with --real <video.mp4> for actual video processing.")
        print("Run with --real <video.mp4> --botsort for BotSort tracking.")


if __name__ == "__main__":
    main()
