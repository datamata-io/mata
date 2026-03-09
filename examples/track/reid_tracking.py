#!/usr/bin/env python3
"""Single-camera object tracking with appearance-based ReID (v1.9.2).

Demonstrates how to enable ReID in mata.track() to improve track-ID recovery
after occlusion or target re-entry — using BotSort's appearance-distance branch.

Features demonstrated:
- ``mata.track()`` with ``reid_model`` kwarg
- Inspecting ``Instance.embedding`` vectors
- Low-level persistent tracking via ``TrackingAdapter`` with ``reid_encoder``
- ReID config alias in ``.mata/models.yaml``

Usage (mock mode — no GPU or real models required):
    python examples/track/reid_tracking.py

Usage (real video + model):
    python examples/track/reid_tracking.py --real examples/videos/cup.mp4

Requirements:
    pip install mata transformers torch
"""
from __future__ import annotations

import argparse


# ---------------------------------------------------------------------------
# Mock helpers (used when --real is not supplied)
# ---------------------------------------------------------------------------

def _make_mock_detector():
    """Return a minimal detector mock that produces synthetic detections."""
    from unittest.mock import Mock
    from mata.core.types import Instance, VisionResult

    frame_count = {"n": 0}
    _LABEL_NAMES = {0: "person", 2: "car"}

    def mock_predict(image, **kwargs):
        n = frame_count["n"]
        frame_count["n"] += 1
        x = 80 + n * 3  # person drifts right each frame
        return VisionResult(
            instances=[
                Instance(bbox=(x, 50, x + 90, 290), label=0,
                         score=0.92, label_name="person"),
                Instance(bbox=(350, 110, 510, 270), label=2,
                         score=0.85, label_name="car"),
            ],
            meta={"frame_idx": n},
        )

    det = Mock()
    det.predict = mock_predict
    # Provide id2label so TrackingAdapter can resolve label names
    det.id2label = _LABEL_NAMES
    return det


def _make_mock_reid_encoder(embedding_dim: int = 128):
    """Return a mock ReID encoder that produces random unit-norm embeddings."""
    import numpy as np
    from unittest.mock import Mock

    def mock_predict(crops):
        if not crops:
            return np.empty((0, 0), dtype=np.float32)
        n = len(crops)
        raw = np.random.randn(n, embedding_dim).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / np.where(norms == 0, 1.0, norms)

    encoder = Mock()
    encoder.predict = mock_predict
    return encoder


# ---------------------------------------------------------------------------
# Example 1 — mata.track() one-liner with reid_model
# ---------------------------------------------------------------------------

def run_one_liner(video_path: str, *, real: bool = False) -> None:
    """Show mata.track() one-liner API with reid_model."""
    print("\n=== Example 1: mata.track() one-liner with ReID ===\n")

    if not real:
        import numpy as np
        from mata.core.types import Instance, VisionResult

        print("  [mock] mata.track() would be called as:\n")
        print("  results = mata.track(")
        print('      "video.mp4",')
        print('      model="facebook/detr-resnet-50",')
        print('      tracker="botsort",')
        print('      reid_model="openai/clip-vit-base-patch32",')
        print("      conf=0.3,")
        print("      save=False,")
        print("  )\n")
        print("  [mock] Simulating 10 frames of tracked objects...\n")

        for frame_idx in range(10):
            emb = np.random.randn(128).astype(np.float32)
            emb /= np.linalg.norm(emb)
            result = VisionResult(
                instances=[
                    Instance(
                        bbox=(80 + frame_idx * 3, 50, 170 + frame_idx * 3, 290),
                        label=0, score=0.92, label_name="person",
                        track_id=1, embedding=emb,
                    ),
                ],
                meta={"frame_idx": frame_idx},
            )
            for inst in result.instances:
                emb_str = (
                    f"shape=({inst.embedding.shape[0]},) "
                    f"norm={np.linalg.norm(inst.embedding):.4f}"
                    if inst.embedding is not None else "None"
                )
                print(
                    f"  Frame {frame_idx:02d} | Track #{inst.track_id} "
                    f"{inst.label_name:<8} score={inst.score:.2f} | "
                    f"embedding {emb_str}"
                )
        print("\n  ✅ ReID embeddings populated in Instance.embedding\n")
        return

    # Real mode — downloads model on first run
    import mata
    import numpy as np

    results = mata.track(
        video_path,
        model="facebook/detr-resnet-50",
        tracker="botsort",
        reid_model="openai/clip-vit-base-patch32",
        conf=0.3,
        save=False,
    )

    for frame_idx, result in enumerate(results):
        for inst in result.instances:
            emb_info = ""
            if inst.embedding is not None:
                emb_info = (
                    f"embedding shape=({inst.embedding.shape[0]},) "
                    f"norm={np.linalg.norm(inst.embedding):.4f}"
                )
            print(
                f"  Frame {frame_idx:02d} | Track #{inst.track_id} "
                f"{inst.label_name:<10} score={inst.score:.2f} | {emb_info}"
            )


# ---------------------------------------------------------------------------
# Example 2 — Low-level TrackingAdapter with reid_encoder
# ---------------------------------------------------------------------------

def run_low_level(num_frames: int = 5) -> None:
    """Show low-level TrackingAdapter with reid_encoder."""
    import numpy as np
    from mata.adapters.tracking_adapter import TrackingAdapter

    print("\n=== Example 2: Low-level TrackingAdapter with reid_encoder ===\n")

    mock_detector = _make_mock_detector()
    mock_encoder = _make_mock_reid_encoder(embedding_dim=128)

    adapter = TrackingAdapter(
        mock_detector,
        tracker_config={"tracker_type": "botsort"},
        frame_rate=25,
        reid_encoder=mock_encoder,
    )
    print(f"  TrackingAdapter created. reid_encoder set: {adapter._reid_encoder is not None}\n")

    for frame_idx in range(num_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # blank synthetic frame
        result = adapter.update(frame)

        for inst in result.instances:
            emb_str = "None"
            if inst.embedding is not None:
                emb_str = (
                    f"shape=({inst.embedding.shape[0]},) "
                    f"norm={np.linalg.norm(inst.embedding):.4f}"
                )
            label = str(inst.label_name) if inst.label_name is not None else "?"
            print(
                f"  Frame {frame_idx} | Track #{inst.track_id} "
                f"{label:<8} | embedding {emb_str}"
            )

    print("\n  ✅ Low-level ReID tracking complete\n")


# ---------------------------------------------------------------------------
# Example 3 — YAML config alias with reid_model
# ---------------------------------------------------------------------------

def print_config_example() -> None:
    """Print example .mata/models.yaml config for ReID-enabled tracking."""
    print("\n=== Example 3: Config alias with reid_model ===\n")
    print("  Place this in .mata/models.yaml:\n")
    config = """\
  models:
    track:
      smart-cam:
        source: "facebook/detr-resnet-50"
        tracker: botsort
        reid_model: "openai/clip-vit-base-patch32"
        frame_rate: 30
        tracker_config:
          track_high_thresh: 0.6
          appearance_thresh: 0.25
          track_buffer: 60
"""
    print(config)
    print("  Then load with a single call:\n")
    print('  import mata')
    print('  tracker = mata.load("track", "smart-cam")  # ReID loaded automatically')
    print("  result  = tracker.update(frame)\n")
    print("  ✅ Config alias with reid_model demonstrated\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MATA ReID tracking example (v1.9.2)")
    p.add_argument(
        "--real", metavar="VIDEO",
        help="Path to a real video file (downloads model on first run)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    real = bool(args.real)
    video = args.real or "video.mp4"

    run_one_liner(video, real=real)
    run_low_level()
    print_config_example()

    print("=" * 60)
    print("Done.")
    print("See examples/track/cross_camera_reid.py for Valkey cross-camera ReID.")


if __name__ == "__main__":
    main()
