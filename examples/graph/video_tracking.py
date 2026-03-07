#!/usr/bin/env python3
"""Video processing with object tracking: Detect > Track across frames.

Demonstrates:
1. Building a detection + tracking graph
2. Using `VideoProcessor` to process video files frame-by-frame
3. Frame policies (`FramePolicyEveryN`, `FramePolicyLatest`, `FramePolicyQueue`)
4. Accessing per-frame results with track IDs
5. Temporal windowing with the `Window` node

Usage:
    python examples/graph/video_tracking.py
    python examples/graph/video_tracking.py --real
    python examples/graph/video_tracking.py --video_path /path/to/video.mp4

Note:
    Requires OpenCV for video file processing:
        pip install opencv-python
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------

def create_mock_providers():
    """Create mock detector and tracker for demo."""
    from unittest.mock import Mock

    from mata.core.types import Instance, VisionResult
    from mata.nodes.track import SimpleIOUTracker

    # Simulate a detector that returns slightly different bboxes each frame
    # (mimicking real object movement)
    call_count = {"n": 0}

    def mock_predict(image, **kwargs):
        n = call_count["n"]
        call_count["n"] += 1
        # Person moves right by 5px each frame
        x_offset = n * 5
        return VisionResult(
            instances=[
                Instance(
                    bbox=(100 + x_offset, 50, 250 + x_offset, 400),
                    label=0, score=0.90, label_name="person",
                ),
                Instance(
                    bbox=(400, 100, 520, 300),
                    label=1, score=0.85, label_name="car",
                ),
            ],
            meta={"model": "mock-detr", "frame": n},
        )

    mock_detector = Mock()
    mock_detector.predict = mock_predict

    # Use SimpleIOUTracker (built-in) as the tracker — returns real Tracks artifacts
    tracker = SimpleIOUTracker()

    return {
        "detect": {"detector": mock_detector},
        "track": {"tracker": tracker},
    }


DEFAULT_VIDEO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "videos", "cup.mp4"
)


def get_video_path() -> str:
    """Return video path from --video_path arg or the default cup.mp4."""
    for i, arg in enumerate(sys.argv):
        if arg == "--video_path" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return DEFAULT_VIDEO_PATH


# ---------------------------------------------------------------------------
# Main example
# ---------------------------------------------------------------------------

def main():
    """Run video tracking example."""
    from mata.core.graph import Graph
    from mata.core.graph.temporal import (
        FramePolicyEveryN,
        FramePolicyLatest,
        FramePolicyQueue,
        VideoProcessor,
    )
    from mata.nodes import Detect, Filter, Fuse, Track

    use_real = "--real" in sys.argv
    if use_real:
        import mata
        from mata.nodes.track import SimpleIOUTracker
        print("Loading real models...")
        detector = mata.load("detect", "PekingU/rtdetr_r50vd")
        # VideoProcessor needs nested {capability: {name: adapter}} format
        providers = {
            "detect": {"detector": detector},
            "track": {"tracker": SimpleIOUTracker()},
        }
    else:
        print("Running with mock providers")
        # VideoProcessor needs nested {capability: {name: adapter}} format
        providers = create_mock_providers()

    # -----------------------------------------------------------------------
    # Build detection + tracking graph
    # -----------------------------------------------------------------------
    graph = (
        Graph("detect_and_track")
        .then(Detect(using="detector", out="dets"))
        .then(Filter(src="dets", score_gt=0.5, out="filtered"))
        .then(Track(using="tracker", dets="filtered", out="tracks"))
        .then(Fuse(detections="filtered", tracks="tracks", out="frame_result"))
    )

    # -----------------------------------------------------------------------
    # Example 1: Frame policies overview
    # -----------------------------------------------------------------------
    print("\n=== Frame Policies ===")

    # Every 3rd frame — good for offline video processing
    policy_every3 = FramePolicyEveryN(n=3)
    print(f"EveryN(3): frame 0={policy_every3.should_process(0)}, "
            f"frame 1={policy_every3.should_process(1)}, "
            f"frame 3={policy_every3.should_process(3)}")

    # Latest frame only — good for real-time (RTSP/webcam)
    policy_latest = FramePolicyLatest()
    print("Latest: Always processes most recent frame")

    # Queue up to 10 frames — balanced approach
    policy_queue = FramePolicyQueue(max_queue=10)
    print("Queue(10): Buffers up to 10 frames")

    # -----------------------------------------------------------------------
    # Example 2: Process video file
    # -----------------------------------------------------------------------
    print("\n=== Video File Processing ===")

    video_path = get_video_path()
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        print("Provide a path with: --video_path /path/to/video.mp4")
    else:
        print(f"Video: {video_path}")
        # providers is nested {capability: {name: adapter}} for VideoProcessor/ExecutionContext.
        # graph.compile() validator needs flat {name: adapter}, so flatten for it.
        flat_providers = {
            name: prov
            for cap_dict in providers.values()
            for name, prov in cap_dict.items()
        }
        compiled = graph.compile(providers=flat_providers)

        # Process with EveryN policy (every 3rd frame)
        processor = VideoProcessor(
            graph=compiled,
            providers=providers,
            frame_policy=FramePolicyEveryN(n=3),
        )

        results = processor.process_video(
            video_path=video_path,
            max_frames=15,  # Limit to 15 frames for demo
        )

        print(f"Processed {len(results)} frames (every 3rd, up to 15)")
        for i, frame_result in enumerate(results):
            channels = list(frame_result.channels.keys())
            print(f"  Frame {i}: channels={channels}")
            # Access track data if available
            if frame_result.has_channel("tracks"):
                tracks = frame_result.get_channel("tracks")
                print(f"    Active tracks: {len(tracks.tracks) if hasattr(tracks, 'tracks') else 'N/A'}")

    # -----------------------------------------------------------------------
    # Example 3: Using the detect_and_track preset
    # -----------------------------------------------------------------------
    print("\n=== Using detect_and_track Preset ===")
    from mata.presets import detect_and_track

    preset_graph = detect_and_track(
        detection_threshold=0.5,
        track_threshold=0.4,
        track_buffer=30,
        match_threshold=0.8,
    )
    print(f"Preset graph: {preset_graph.name}")
    print(f"Nodes: {len(preset_graph._nodes)}")

    # -----------------------------------------------------------------------
    # Example 4: Real-time stream processing (conceptual)
    # -----------------------------------------------------------------------
    print("\n=== Real-time Stream Processing (conceptual) ===")
    print("""
    # For RTSP camera feeds:
    import threading

    compiled = graph.compile(providers=providers)
    processor = VideoProcessor(
        graph=compiled,
        providers=providers,
        frame_policy=FramePolicyLatest(),  # Drop old frames for real-time
    )

    stop_event = threading.Event()

    def on_frame_result(result, frame_num):
        print(f"Frame {frame_num}: {len(result.channels)} channels")
        # Process result, update UI, save to DB, etc.

    # Start stream processing (blocking)
    processor.process_stream(
        source="rtsp://camera_ip:554/stream",
        callback=on_frame_result,
        stop_event=stop_event,
        max_frames=1000,
    )

    # From another thread: stop_event.set() to stop
    """)

    print("Video tracking example complete!")


if __name__ == "__main__":
    main()
