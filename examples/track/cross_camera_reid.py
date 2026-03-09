#!/usr/bin/env python3
"""Cross-camera object re-identification via Valkey (v1.9.2).

Demonstrates how to use ``ReIDBridge`` so that independent tracker instances
running on different camera feeds can resolve the same physical identity across
feeds using shared Valkey embedding storage.

Architecture:
    Camera A tracker
        └─ mata.track(..., reid_bridge=bridge_a)
              └─ bridge_a.publish(track_id, embedding) → Valkey
    Camera B tracker
        └─ mata.track(..., reid_bridge=bridge_b)
              └─ bridge_b.query(embedding) ← Valkey ← cam-a embeddings

Features demonstrated:
- Constructing ``ReIDBridge`` with camera_id, TTL, and similarity_thresh
- Publishing embeddings per track per frame (automatic via reid_bridge kwarg)
- Querying for cross-camera nearest-identity matches
- Mocked Valkey client (no real server required for the basic demo)
- Real Valkey/Redis server usage notes

Usage (mock mode — no Valkey server required):
    python examples/track/cross_camera_reid.py

Usage (real Valkey server):
    python examples/track/cross_camera_reid.py --valkey valkey://localhost:6379

Requirements:
    pip install mata[valkey] transformers torch
    # or:
    pip install mata[redis] transformers torch
"""
from __future__ import annotations

import argparse
import time
from typing import Any


# ---------------------------------------------------------------------------
# Mock Valkey client (used when no --valkey URL is supplied)
# ---------------------------------------------------------------------------

class _MockValkeyClient:
    """In-memory key-value store that mimics the subset of Valkey API used by ReIDBridge."""

    def __init__(self) -> None:
        self._store: dict[bytes | str, bytes] = {}
        self._ttls: dict[bytes | str, float] = {}

    def set(self, key: str, value: bytes, ex: int | None = None) -> None:
        self._store[key] = value
        if ex is not None:
            self._ttls[key] = time.time() + ex

    def get(self, key: str) -> bytes | None:
        if key in self._ttls and time.time() > self._ttls[key]:
            self._store.pop(key, None)
            self._ttls.pop(key, None)
        return self._store.get(key)

    def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                self._ttls.pop(k, None)
                count += 1
        return count

    def scan_iter(self, match: str = "*", count: int = 100):
        """Yield keys matching a simple glob pattern (only ``*`` wildcard)."""
        import fnmatch
        active = [
            k for k, exp in list(self._ttls.items())
            if time.time() <= exp
        ] + [
            k for k in self._store if k not in self._ttls
        ]
        for key in active:
            k_str = key.decode() if isinstance(key, bytes) else key
            if fnmatch.fnmatch(k_str, match):
                yield key

    def ping(self) -> bool:
        return True


def _build_mock_bridge(camera_id: str, similarity_thresh: float = 0.25,
                       shared_store: _MockValkeyClient | None = None):
    """Build a ReIDBridge backed by a mock in-memory client."""
    from unittest.mock import patch
    from mata.trackers.reid_bridge import ReIDBridge

    client = shared_store or _MockValkeyClient()
    bridge = ReIDBridge.__new__(ReIDBridge)
    bridge._client = client
    bridge._camera_id = camera_id
    bridge._ttl = 300
    bridge._similarity_thresh = similarity_thresh
    bridge._prefix = "reid"
    return bridge


# ---------------------------------------------------------------------------
# Example 1 — Publish / Query round-trip (mock)
# ---------------------------------------------------------------------------

def run_publish_query_demo() -> None:
    """Demonstrate publish / query semantics with a mocked Valkey client."""
    import numpy as np

    print("\n=== Example 1: Publish → Query round-trip (mocked Valkey) ===\n")

    shared_store = _MockValkeyClient()

    bridge_cam_a = _build_mock_bridge("cam-a", shared_store=shared_store)
    bridge_cam_b = _build_mock_bridge("cam-b", shared_store=shared_store)

    # Camera A observes person with track_id=7
    emb_a = np.random.randn(128).astype(np.float32)
    emb_a /= np.linalg.norm(emb_a)

    bridge_cam_a.publish(
        track_id=7,
        embedding=emb_a,
        bbox=(120.0, 50.0, 200.0, 280.0),
        label=0,
    )
    print(f"  cam-a: published track #7 (embedding norm={np.linalg.norm(emb_a):.4f})")

    # Camera B receives a detection with very similar appearance
    # (in real usage this would be from a different camera angle of the same person)
    noise = np.random.randn(128).astype(np.float32) * 0.05
    emb_b_query = emb_a + noise
    emb_b_query /= np.linalg.norm(emb_b_query)

    matches = bridge_cam_b.query(emb_b_query, exclude_camera="cam-b", top_k=3)

    if matches:
        best = matches[0]
        print(
            f"  cam-b: best cross-camera match → "
            f"camera={best['camera_id']!r} track_id={best['track_id']} "
            f"similarity={best['similarity']:.4f}"
        )
        print(f"  ✅ Cross-camera identity resolved: cam-b detection → cam-a track #7\n")
    else:
        print("  ⚠ No match found (similarity below threshold)\n")

    # Demonstrate no self-match
    self_matches = bridge_cam_a.query(emb_a, exclude_camera="cam-a")
    assert not self_matches, "Should not return own camera embeddings"
    print("  ✅ Self-camera exclusion works (no same-camera matches)\n")


# ---------------------------------------------------------------------------
# Example 2 — Multi-camera tracking loop (mock)
# ---------------------------------------------------------------------------

def run_multi_camera_loop(num_frames: int = 5) -> None:
    """Simulate two parallel camera trackers publishing to a shared store."""
    import numpy as np
    from mata.adapters.tracking_adapter import TrackingAdapter
    from mata.core.types import Instance, VisionResult

    print("\n=== Example 2: Two-camera tracking loop with shared ReID store ===\n")

    shared_store = _MockValkeyClient()
    bridge_a = _build_mock_bridge("cam-a", shared_store=shared_store)
    bridge_b = _build_mock_bridge("cam-b", shared_store=shared_store)

    def _make_mock_adapter(camera_label: str, track_id: int,
                           base_emb: "np.ndarray", bridge) -> TrackingAdapter:
        """Build a TrackingAdapter mock that auto-publishes to the bridge."""
        from unittest.mock import Mock

        call_count = {"n": 0}

        def mock_update(image, **kw):
            n = call_count["n"]
            call_count["n"] += 1
            # Slightly perturb embedding each frame (EMA simulation)
            emb = base_emb + np.random.randn(128).astype(np.float32) * 0.03
            emb /= np.linalg.norm(emb)
            inst = Instance(
                bbox=(100.0 + n * 2, 50.0, 190.0 + n * 2, 280.0),
                label=0, score=0.90, label_name="person",
                track_id=track_id, embedding=emb,
            )
            result = VisionResult(instances=[inst], meta={"frame_idx": n, "camera": camera_label})
            # Simulate the publish step that TrackingAdapter.update() performs when reid_bridge is set
            for i in result.instances:
                if i.track_id is not None and i.embedding is not None:
                    bridge.publish(i.track_id, i.embedding, bbox=i.bbox, label=i.label)
            return result

        adapter = Mock()
        adapter.update = mock_update
        return adapter

    # Two cameras tracking the same physical person with different track IDs
    base_person_emb = np.random.randn(128).astype(np.float32)
    base_person_emb /= np.linalg.norm(base_person_emb)

    adapter_a = _make_mock_adapter("cam-a", track_id=3, base_emb=base_person_emb, bridge=bridge_a)
    adapter_b = _make_mock_adapter("cam-b", track_id=11, base_emb=base_person_emb, bridge=bridge_b)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for frame_idx in range(num_frames):
        r_a = adapter_a.update(dummy_frame)
        r_b = adapter_b.update(dummy_frame)

        # Camera B queries cross-camera matches for its track #11
        if r_b.instances:
            emb_b = r_b.instances[0].embedding
            matches = bridge_b.query(emb_b, exclude_camera="cam-b", top_k=1)
            match_str = (
                f"→ cam-a track #{matches[0]['track_id']} "
                f"(sim={matches[0]['similarity']:.3f})"
                if matches else "→ no match"
            )
            print(
                f"  Frame {frame_idx} | cam-b track #11 "
                f"(norm={np.linalg.norm(emb_b):.4f}) {match_str}"
            )

    print("\n  ✅ Multi-camera ReID loop complete\n")


# ---------------------------------------------------------------------------
# Example 3 — Real Valkey server usage notes
# ---------------------------------------------------------------------------

def print_real_valkey_notes(valkey_url: str) -> None:
    """Print real-server usage notes (or attempt a live demo if server available)."""
    print("\n=== Example 3: Real Valkey server usage ===\n")

    print(f"  Connection URL: {valkey_url}\n")

    try:
        from mata.trackers.reid_bridge import ReIDBridge
        bridge_test = ReIDBridge(valkey_url, camera_id="test-cam", ttl=10)
        bridge_test._client.ping()
        print("  ✅ Valkey server reachable — live demo mode\n")

        import numpy as np

        emb = np.random.randn(128).astype(np.float32)
        emb /= np.linalg.norm(emb)
        bridge_test.publish(track_id=99, embedding=emb, bbox=(0., 0., 100., 200.), label=0)
        print("  Published test embedding for track #99")

        matches = bridge_test.query(emb, exclude_camera="other-cam", top_k=1)
        print(f"  Self-query (include own camera): would find {len(matches)} match(es)")

        count = bridge_test.clear(camera_id="test-cam")
        print(f"  Cleared {count} test key(s)")
        print("\n  ✅ Live Valkey round-trip successful\n")

    except Exception as exc:
        print(f"  ⚠ Could not connect to Valkey ({exc})")
        print("  Running as notes-only mode\n")

    print("  Full ReIDBridge API reference:\n")
    print("    from mata.trackers import ReIDBridge\n")
    print("    bridge = ReIDBridge(")
    print('        "valkey://localhost:6379",')
    print('        camera_id="cam-front",')
    print("        ttl=300,              # embeddings expire after 5 minutes")
    print("        similarity_thresh=0.25,  # cosine similarity cutoff")
    print("    )\n")
    print("    # In your tracking loop (automatic when passing reid_bridge to mata.track()):")
    print("    bridge.publish(track_id=42, embedding=emb, bbox=(x1,y1,x2,y2), label=0)\n")
    print("    # Query from a different camera process:")
    print('    matches = bridge.query(query_emb, exclude_camera="cam-front", top_k=1)')
    print("    # matches = [{'track_id': 42, 'camera_id': 'cam-front', 'similarity': 0.87, ...}]\n")
    print("    # Attach bridge to mata.track() for automatic publishing:")
    print("    for result in mata.track(")
    print('        "rtsp://cam-front/stream",')
    print('        model="facebook/detr-resnet-50",')
    print('        reid_model="openai/clip-vit-base-patch32",')
    print("        reid_bridge=bridge,")
    print("        stream=True,")
    print("    ):")
    print("        ...")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MATA cross-camera ReID via Valkey example (v1.9.2)"
    )
    p.add_argument(
        "--valkey", metavar="URL",
        help="Valkey/Redis URL for live demo (e.g. valkey://localhost:6379). "
             "Omit to run with mocked client.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    run_publish_query_demo()
    run_multi_camera_loop()
    print_real_valkey_notes(args.valkey or "valkey://localhost:6379")

    print("=" * 60)
    print("Done.")
    print("See examples/track/reid_tracking.py for single-camera ReID usage.")


if __name__ == "__main__":
    main()
