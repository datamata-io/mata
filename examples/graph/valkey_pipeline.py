#!/usr/bin/env python3
"""Valkey / Redis result storage in graph pipelines.

Demonstrates five patterns:

  1. **Basic save/load** — `result.save("valkey://...")` and `load_valkey()`
  2. **`ValkeyStore` sink node** — store mid-pipeline and continue downstream
  3. **Cross-pipeline handoff** — Service A stores, Service B loads via `ValkeyLoad`
  4. **Pub/Sub** — broadcast detection events to live subscribers
  5. **Streaming rolling key** — per-frame overwrite with short TTL for live feeds

For RTSP stream tracking with Valkey see ``examples/graph/rtsp_pipeline.py``.

All examples run in **mock mode** by default (no model downloads, no real Valkey
server required).  When running against a real server, pass ``--real``.

Requirements:
    pip install datamata[valkey]    # or datamata[redis]

Usage:
    # Mock mode — fully self-contained, no server needed
    python examples/graph/valkey_pipeline.py

    # Real mode — requires a Valkey/Redis server on localhost:6379
    python examples/graph/valkey_pipeline.py --real

    # Specify a different server
    python examples/graph/valkey_pipeline.py --real --url valkey://myhost:6379
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _url_from_args(default: str = "valkey://localhost:6379") -> str:
    """Return --url CLI override, or the default."""
    for i, arg in enumerate(sys.argv):
        if arg == "--url" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------

def create_mock_providers():
    """Create a mock detector that returns a small fixed VisionResult."""
    from unittest.mock import Mock

    from mata.core.types import Instance, VisionResult

    mock_detector = Mock()
    mock_detector.predict = Mock(return_value=VisionResult(
        instances=[
            Instance(bbox=(50, 30, 220, 300), label=0, score=0.91, label_name="person"),
            Instance(bbox=(300, 80, 480, 340), label=2, score=0.76, label_name="car"),
            Instance(bbox=(10, 10, 40, 40),   label=3, score=0.12, label_name="noise"),
        ],
        meta={"model": "mock-detector"},
    ))
    return {"detector": mock_detector}


def create_real_providers():
    """Load a real RT-DETR detector from HuggingFace."""
    import mata

    print("Loading PekingU/rtdetr_r18vd from HuggingFace (this may take a moment)...")
    detector = mata.load("detect", "PekingU/rtdetr_r18vd")
    return {"detector": detector}


# ---------------------------------------------------------------------------
# Mock Valkey client  (used when --real is NOT passed)
# ---------------------------------------------------------------------------

class _MockValkeyClient:
    """In-memory stand-in so the example runs without a real server."""

    def __init__(self):
        self._store: dict[str, bytes] = {}
        self._channels: dict[str, list] = {}

    # Key/value ops
    def set(self, key: str, value, ex=None):
        self._store[key] = value if isinstance(value, bytes) else value.encode()

    def setex(self, key: str, ttl: int, value):
        self.set(key, value)

    def get(self, key: str):
        return self._store.get(key)

    def exists(self, key: str) -> int:
        return 1 if key in self._store else 0

    # Pub/Sub
    def publish(self, channel: str, message) -> int:
        subscribers = self._channels.get(channel, [])
        return len(subscribers)


# ---------------------------------------------------------------------------
# Shared inference helper
# ---------------------------------------------------------------------------

IMAGE_PATH = "examples/images/000000039769.jpg"


def _detect(providers: dict) -> "mata.core.types.VisionResult":  # type: ignore[name-defined]
    """Run detection using the pre-loaded adapter in *providers*.

    Calls ``providers["detector"].predict()`` directly so the same code works
    with both mock adapters and real HuggingFace adapters loaded by
    ``create_real_providers()``.  ``mata.run()`` is intentionally not used here
    because it only accepts a model string and loads its own adapter internally.
    """
    from PIL import Image as PILImage
    image = PILImage.open(IMAGE_PATH).convert("RGB")
    return providers["detector"].predict(image)


# Monkey-patch the exporter to use the mock client when not in --real mode
_MOCK_CLIENT = _MockValkeyClient()

def _patch_exporter_with_mock():
    """Replace _get_valkey_client() with our in-memory mock."""
    try:
        import mata.core.exporters.valkey_exporter as _ve
        _ve._get_valkey_client = lambda url, **kw: _MOCK_CLIENT   # type: ignore[attr-defined]
    except Exception:
        pass   # exporter not available — skip patching


# ---------------------------------------------------------------------------
# Example 1 — Basic save / load
# ---------------------------------------------------------------------------

def example_basic_save_load(url: str, providers: dict):
    """result.save('valkey://...') and load_valkey() round-trip."""
    print("\n" + "=" * 60)
    print("Example 1: Basic save / load")
    print("=" * 60)

    from mata.core.exporters import export_valkey, load_valkey

    # Run detection using the pre-loaded adapter (mock or real)
    result = _detect(providers)

    print(f"Detections before save: {len(result.instances)} objects")
    for inst in result.instances:
        print(f"  {inst.label_name:10s}  score={inst.score:.2f}")

    # --- save via result.save() (valkey:// URI scheme) ---
    KEY = "example1:detections:latest"
    result.save(f"{url}/{KEY}", ttl=300)
    print(f"\nSaved to key '{KEY}' (TTL=300s)")

    # --- also via explicit export_valkey() with msgpack ---
    export_valkey(
        result,
        url=url,
        key="example1:detections:msgpack",
        ttl=300,
        serializer="json",   # "msgpack" is faster for large payloads
    )
    print("Saved second copy with export_valkey(serializer='json')")

    # --- load back ---
    loaded = load_valkey(url=url, key=KEY)
    print(f"\nLoaded from key '{KEY}': {len(loaded.instances)} objects  "
            f"(type={type(loaded).__name__})")
    assert len(loaded.instances) == len(result.instances), "Round-trip mismatch!"
    print("✓ Round-trip verified")


# ---------------------------------------------------------------------------
# Example 2 — ValkeyStore sink node in a graph
# ---------------------------------------------------------------------------

def example_valkey_store_node(url: str, providers: dict):
    """ValkeyStore writes mid-pipeline while passing the artifact downstream."""
    print("\n" + "=" * 60)
    print("Example 2: ValkeyStore node in a graph pipeline")
    print("=" * 60)

    import mata
    from mata.core.graph import Graph
    from mata.nodes import Detect, Filter, Fuse, ValkeyStore

    graph = (
        Graph("detection_with_store")
        .then(Detect(using="detector", out="raw_dets"))
        .then(Filter(src="raw_dets", score_gt=0.5, out="filtered"))
        # ── Persist filtered results; {timestamp} resolves to Unix epoch ──
        .then(ValkeyStore(
            src="filtered",
            url=url,
            key="pipeline:filtered:{timestamp}",
            ttl=600,
        ))
        # ── Downstream nodes still receive "filtered" unchanged ──────────
        .then(Fuse(detections="filtered", out="final"))
    )

    result = mata.infer(
        image=IMAGE_PATH,
        graph=graph,
        providers=providers,
    )

    print(f"Graph channels: {list(result.channels.keys())}")
    if result.has_channel("final"):
        final = result.get_channel("final")
        if final.has_channel("detections"):
            dets = final.get_channel("detections")
            print(f"Final detections (post-store): {len(dets.instances)} objects")
            for inst in dets.instances:
                print(f"  {inst.label_name:10s}  score={inst.score:.2f}")
    print("✓ ValkeyStore did not interrupt the downstream pipeline")


# ---------------------------------------------------------------------------
# Example 3 — Cross-pipeline handoff: Store A → Load B
# ---------------------------------------------------------------------------

def example_cross_pipeline(url: str, providers: dict):
    """Service A stores; Service B loads via ValkeyLoad and continues."""
    print("\n" + "=" * 60)
    print("Example 3: Cross-pipeline handoff (ValkeyStore → ValkeyLoad)")
    print("=" * 60)

    import mata
    from mata.core.graph import Graph
    from mata.nodes import Detect, Filter, Fuse, ValkeyLoad, ValkeyStore

    # ── Service A: camera ingestion pipeline ─────────────────────────────
    SHARED_KEY = "cross_pipeline:cam01:latest"

    store_graph = (
        Graph("service_a_ingest")
        .then(Detect(using="detector", out="dets"))
        .then(Filter(src="dets", score_gt=0.3, out="filtered"))
        .then(ValkeyStore(
            src="filtered",
            url=url,
            key=SHARED_KEY,
            ttl=15,   # stays fresh for 15 s before auto-expiry
        ))
    )

    print("Service A: running detection and storing results...")
    mata.infer(
        image=IMAGE_PATH,
        graph=store_graph,
        providers=providers,
    )
    print(f"  Stored filtered detections under '{SHARED_KEY}'")

    # ── Service B: downstream analytics pipeline ─────────────────────────
    load_graph = (
        Graph("service_b_analytics")
        .then(ValkeyLoad(
            url=url,
            key=SHARED_KEY,
            result_type="auto",   # auto-detect VisionResult
            out="loaded_dets",
        ))
        .then(Filter(src="loaded_dets", score_gt=0.7, out="hi_conf"))
        .then(Fuse(detections="hi_conf", out="analytics"))
    )

    print("\nService B: loading stored results and applying hi-conf filter...")
    result_b = mata.infer(
        image=IMAGE_PATH,
        graph=load_graph,
        providers={},   # no detection model needed in Service B
    )

    print(f"  Channels: {list(result_b.channels.keys())}")
    if result_b.has_channel("analytics"):
        anal = result_b.get_channel("analytics")
        if anal.has_channel("detections"):
            hi = anal.get_channel("detections")
            print(f"  Hi-confidence objects: {len(hi.instances)}")
    print("✓ Service B successfully consumed Service A's stored results")


# ---------------------------------------------------------------------------
# Example 4 — Pub/Sub broadcast
# ---------------------------------------------------------------------------

def example_pubsub(url: str, providers: dict):
    """publish_valkey() broadcasts a result to all active channel subscribers."""
    print("\n" + "=" * 60)
    print("Example 4: Pub/Sub event broadcast")
    print("=" * 60)

    from mata.core.exporters import publish_valkey

    result = _detect(providers)

    CHANNEL = "detections:events:cam01"
    n = publish_valkey(
        result=result,
        url=url,
        channel=CHANNEL,
        serializer="json",
    )
    print(f"Published detection event to channel '{CHANNEL}'")
    print(f"Delivered to {n} subscriber(s)")
    print("(In production, start a subscriber before publishing — "
            "Pub/Sub is fire-and-forget.)")

    # ── combined: persist + broadcast ────────────────────────────────────
    print("\nCombined persist + broadcast pattern:")
    from mata.core.exporters import export_valkey
    export_valkey(result, url=url, key="det:latest", ttl=60)
    n = publish_valkey(result, url=url, channel="det:events")
    print(f"  Persisted to 'det:latest' (TTL=60s)")
    print(f"  Broadcast to 'det:events' → {n} subscriber(s)")
    print("✓ Pub/Sub example complete")


# ---------------------------------------------------------------------------
# Example 5 — Streaming per-frame rolling key
# ---------------------------------------------------------------------------

def example_streaming_rolling_key(url: str, providers: dict):
    """Simulate per-frame tracking with a rolling Valkey key (TTL reset each frame)."""
    print("\n" + "=" * 60)
    print("Example 5: Streaming / per-frame rolling key")
    print("=" * 60)

    from mata.core.exporters import export_valkey, load_valkey

    ROLLING_KEY = "stream:track:latest"
    HISTORY_PREFIX = "stream:track:history"
    FRAMES = 5

    print(f"Simulating {FRAMES} frames...")
    for frame_idx in range(FRAMES):
        result = _detect(providers)
        # Rolling latest — overwritten every frame, expires if feed drops
        export_valkey(result, url=url, key=ROLLING_KEY, ttl=10)
        # Indexed history for audit trail
        export_valkey(result, url=url, key=f"{HISTORY_PREFIX}:{frame_idx:06d}", ttl=3600)
        print(f"  Frame {frame_idx:02d}: stored {len(result.instances)} detections")

    # Read back the latest
    latest = load_valkey(url=url, key=ROLLING_KEY)
    print(f"\nLatest key '{ROLLING_KEY}': {len(latest.instances)} detections")

    # Read a specific historical frame
    frame_2 = load_valkey(url=url, key=f"{HISTORY_PREFIX}:000002")
    print(f"History frame 2: {len(frame_2.instances)} detections")
    print("✓ Streaming rolling-key pattern complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    use_real = "--real" in sys.argv
    url = _url_from_args()

    if use_real:
        print(f"Running against real Valkey server: {url}")
        print("Make sure valkey-py is installed:  pip install datamata[valkey]")
        providers = create_real_providers()
    else:
        print("Running in MOCK mode (in-memory store, no server required)")
        print("Pass --real to run against a real Valkey server.")
        _patch_exporter_with_mock()
        providers = create_mock_providers()

    example_basic_save_load(url, providers)
    example_valkey_store_node(url, providers)
    example_cross_pipeline(url, providers)
    example_pubsub(url, providers)
    example_streaming_rolling_key(url, providers)

    print("\n" + "=" * 60)
    print("All Valkey examples completed successfully.")
    print("See docs/VALKEY_GUIDE.md for the full reference.")
    print("=" * 60)


if __name__ == "__main__":
    main()
