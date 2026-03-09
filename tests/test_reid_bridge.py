"""Tests for ReIDBridge — cross-camera embedding store backed by Valkey.

All tests use unittest.mock to mock the Valkey client, so no real Valkey
server is required.

Test coverage:
- publish(): key format, TTL, msgpack serialisation, error handling
- query(): threshold filtering, camera exclusion, top_k, similarity ordering
- clear(): specific camera, all cameras, error handling
- scan_iter usage (not KEYS)
- msgpack roundtrip fidelity
- connection error graceful degradation
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import msgpack
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bridge(
    url: str = "valkey://localhost:6379/0",
    camera_id: str = "cam-1",
    ttl: int = 300,
    similarity_thresh: float = 0.25,
    mock_client: MagicMock | None = None,
):
    """Return a ReIDBridge with a mocked Valkey client."""
    from mata.trackers.reid_bridge import ReIDBridge

    if mock_client is None:
        mock_client = MagicMock()

    with patch(
        "mata.core.exporters.valkey_exporter._get_valkey_client",
        return_value=mock_client,
    ):
        bridge = ReIDBridge(
            url=url,
            camera_id=camera_id,
            ttl=ttl,
            similarity_thresh=similarity_thresh,
        )
    # Swap in the mock so tests can inspect calls made *after* construction
    bridge._client = mock_client
    return bridge, mock_client


def _unit_vec(dim: int = 128, seed: int = 0) -> np.ndarray:
    """Return a deterministic L2-normalised float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _pack_entry(
    track_id: int,
    camera_id: str,
    embedding: np.ndarray,
    bbox: list | None = None,
    label: int = 0,
    timestamp: float | None = None,
) -> bytes:
    """Pack a ReIDBridge-compatible entry with msgpack."""
    return msgpack.packb(
        {
            "track_id": track_id,
            "camera_id": camera_id,
            "embedding": embedding.tolist(),
            "bbox": bbox,
            "label": label,
            "timestamp": timestamp or time.time(),
        }
    )


# ---------------------------------------------------------------------------
# TestReIDBridgePublish
# ---------------------------------------------------------------------------


class TestReIDBridgePublish:
    def test_publish_stores_key_with_ttl(self):
        """publish() calls client.set() with the correct TTL."""
        bridge, client = _make_bridge(camera_id="cam-1", ttl=120)
        emb = _unit_vec()

        bridge.publish(track_id=42, embedding=emb)

        assert client.set.call_count == 1
        _, kwargs = client.set.call_args
        assert kwargs.get("ex") == 120

    def test_publish_key_format(self):
        """publish() uses key pattern 'reid:{camera_id}:{track_id}'."""
        bridge, client = _make_bridge(camera_id="front-door")
        emb = _unit_vec()

        bridge.publish(track_id=7, embedding=emb)

        key_used = client.set.call_args[0][0]
        assert key_used == "reid:front-door:7"

    def test_publish_msgpack_serialises_embedding(self):
        """publish() serialises the embedding correctly via msgpack."""
        bridge, client = _make_bridge(camera_id="cam-2")
        emb = _unit_vec(seed=1)

        bridge.publish(track_id=1, embedding=emb)

        raw_bytes = client.set.call_args[0][1]
        entry = msgpack.unpackb(raw_bytes, raw=False)
        recovered = np.array(entry["embedding"], dtype=np.float32)
        np.testing.assert_allclose(recovered, emb, rtol=1e-5)

    def test_publish_with_bbox(self):
        """publish() includes bbox in the serialised payload."""
        bridge, client = _make_bridge()
        emb = _unit_vec()
        bbox = (10.0, 20.0, 100.0, 200.0)

        bridge.publish(track_id=3, embedding=emb, bbox=bbox)

        raw_bytes = client.set.call_args[0][1]
        entry = msgpack.unpackb(raw_bytes, raw=False)
        assert entry["bbox"] == list(bbox)

    def test_publish_no_bbox_stores_none(self):
        """publish() stores None for bbox when not provided."""
        bridge, client = _make_bridge()
        emb = _unit_vec()

        bridge.publish(track_id=5, embedding=emb, bbox=None)

        raw_bytes = client.set.call_args[0][1]
        entry = msgpack.unpackb(raw_bytes, raw=False)
        assert entry["bbox"] is None

    def test_publish_stores_camera_id_and_label(self):
        """publish() stores camera_id and label in the payload."""
        bridge, client = _make_bridge(camera_id="zone-3")
        emb = _unit_vec()

        bridge.publish(track_id=10, embedding=emb, label=2)

        raw_bytes = client.set.call_args[0][1]
        entry = msgpack.unpackb(raw_bytes, raw=False)
        assert entry["camera_id"] == "zone-3"
        assert entry["label"] == 2

    def test_connection_error_in_publish_logs_warning(self):
        """publish() catches ConnectionError and logs a warning (no exception raised)."""
        bridge, client = _make_bridge()
        client.set.side_effect = ConnectionError("connection refused")
        emb = _unit_vec()

        with patch("mata.trackers.reid_bridge.logger") as mock_logger:
            bridge.publish(track_id=1, embedding=emb)

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "publish" in warning_msg.lower() or "failed" in warning_msg.lower()

    def test_publish_no_exception_on_any_error(self):
        """publish() never propagates exceptions (keeps tracking loop alive)."""
        bridge, client = _make_bridge()
        client.set.side_effect = RuntimeError("unexpected")
        emb = _unit_vec()

        # Must not raise
        bridge.publish(track_id=1, embedding=emb)


# ---------------------------------------------------------------------------
# TestReIDBridgeQuery
# ---------------------------------------------------------------------------


class TestReIDBridgeQuery:
    def _setup_scan(
        self,
        client: MagicMock,
        entries: list[tuple[str, bytes]],
    ) -> None:
        """Configure client.scan_iter and client.get to return given entries."""
        keys = [k for k, _ in entries]
        data = {k: v for k, v in entries}

        client.scan_iter.return_value = iter(keys)
        client.get.side_effect = lambda k: data.get(k)

    def test_query_returns_matches_above_threshold(self):
        """query() returns entries whose cosine similarity ≥ similarity_thresh."""
        bridge, client = _make_bridge(similarity_thresh=0.5)
        query_emb = _unit_vec(seed=0)
        # Highly similar (same vector → sim ≈ 1.0)
        similar_emb = query_emb.copy()
        # Orthogonal → sim ≈ 0.0
        ortho_emb = _unit_vec(seed=99)

        entries = [
            ("reid:cam-2:1", _pack_entry(1, "cam-2", similar_emb)),
            ("reid:cam-2:2", _pack_entry(2, "cam-2", ortho_emb)),
        ]
        self._setup_scan(client, entries)

        results = bridge.query(query_emb, top_k=10)

        assert len(results) == 1
        assert results[0]["track_id"] == 1
        assert results[0]["similarity"] >= 0.5

    def test_query_excludes_own_camera(self):
        """query(exclude_camera=...) filters out same-camera entries."""
        bridge, client = _make_bridge(camera_id="cam-1", similarity_thresh=0.0)
        emb = _unit_vec(seed=0)

        entries = [
            ("reid:cam-1:10", _pack_entry(10, "cam-1", emb)),
            ("reid:cam-2:20", _pack_entry(20, "cam-2", emb)),
        ]
        self._setup_scan(client, entries)

        results = bridge.query(emb, exclude_camera="cam-1", top_k=10)

        assert all(r["camera_id"] != "cam-1" for r in results)
        assert any(r["camera_id"] == "cam-2" for r in results)

    def test_query_empty_store_returns_empty(self):
        """query() returns [] when no keys exist in Valkey."""
        bridge, client = _make_bridge()
        client.scan_iter.return_value = iter([])
        emb = _unit_vec()

        results = bridge.query(emb)

        assert results == []

    def test_similarity_ordering(self):
        """query() returns results sorted by similarity descending."""
        bridge, client = _make_bridge(similarity_thresh=0.0)
        query_emb = _unit_vec(seed=0)

        # Build three embeddings with known similarities
        e1 = query_emb.copy()  # sim = 1.0
        e2 = _unit_vec(seed=5)
        e3 = _unit_vec(seed=10)

        entries = [
            ("reid:cam-2:3", _pack_entry(3, "cam-2", e2)),
            ("reid:cam-2:1", _pack_entry(1, "cam-2", e1)),
            ("reid:cam-2:2", _pack_entry(2, "cam-2", e3)),
        ]
        self._setup_scan(client, entries)

        results = bridge.query(query_emb, top_k=10)

        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True), "Results not sorted by similarity"

    def test_top_k_limits_results(self):
        """query(top_k=2) returns at most 2 results even if more match."""
        bridge, client = _make_bridge(similarity_thresh=0.0)
        emb = _unit_vec(seed=0)

        entries = [(f"reid:cam-2:{i}", _pack_entry(i, "cam-2", _unit_vec(seed=i + 1))) for i in range(5)]
        self._setup_scan(client, entries)

        results = bridge.query(emb, top_k=2)

        assert len(results) <= 2

    def test_query_returns_correct_fields(self):
        """query() result dicts contain the expected keys."""
        bridge, client = _make_bridge(similarity_thresh=0.0)
        emb = _unit_vec(seed=0)
        bbox = [10.0, 20.0, 50.0, 80.0]

        entry = _pack_entry(7, "cam-3", emb, bbox=bbox, label=1)
        self._setup_scan(client, [("reid:cam-3:7", entry)])

        results = bridge.query(emb, top_k=1)

        assert len(results) == 1
        r = results[0]
        assert r["track_id"] == 7
        assert r["camera_id"] == "cam-3"
        assert "similarity" in r
        assert r["bbox"] == bbox
        assert r["label"] == 1

    def test_stale_keys_not_returned(self):
        """When client.get() returns None (TTL expired), the entry is skipped."""
        bridge, client = _make_bridge(similarity_thresh=0.0)
        emb = _unit_vec()

        client.scan_iter.return_value = iter(["reid:cam-2:99"])
        client.get.return_value = None  # simulates expired key

        results = bridge.query(emb, top_k=10)

        assert results == []

    def test_uses_scan_iter_not_keys(self):
        """query() uses scan_iter (not KEYS) for production-safe iteration."""
        bridge, client = _make_bridge()
        client.scan_iter.return_value = iter([])
        emb = _unit_vec()

        bridge.query(emb)

        assert client.scan_iter.called, "scan_iter should be called"
        assert not client.keys.called, "KEYS command must NOT be used"

    def test_connection_error_in_query_logs_warning(self):
        """query() catches ConnectionError, logs warning, returns []."""
        bridge, client = _make_bridge()
        client.scan_iter.side_effect = ConnectionError("unreachable")
        emb = _unit_vec()

        with patch("mata.trackers.reid_bridge.logger") as mock_logger:
            results = bridge.query(emb)

        assert results == []
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "query" in warning_msg.lower() or "failed" in warning_msg.lower()

    def test_query_below_threshold_excluded(self):
        """Entries with similarity < threshold are not returned."""
        bridge, client = _make_bridge(similarity_thresh=0.9)
        query_emb = _unit_vec(seed=0)
        low_sim_emb = _unit_vec(seed=77)  # low cosine similarity to seed=0

        self._setup_scan(client, [("reid:cam-2:1", _pack_entry(1, "cam-2", low_sim_emb))])

        results = bridge.query(query_emb, top_k=10)

        # Result should only appear if sim >= 0.9
        if results:
            assert all(r["similarity"] >= 0.9 for r in results)


# ---------------------------------------------------------------------------
# TestReIDBridgeClear
# ---------------------------------------------------------------------------


class TestReIDBridgeClear:
    def test_clear_removes_keys(self):
        """clear() calls delete() for every key matching the pattern."""
        bridge, client = _make_bridge(camera_id="cam-1")
        client.scan_iter.return_value = iter(
            [
                "reid:cam-1:10",
                "reid:cam-1:11",
            ]
        )

        count = bridge.clear(camera_id="cam-1")

        assert count == 2
        assert client.delete.call_count == 2

    def test_clear_specific_camera_uses_scan_iter(self):
        """clear(camera_id=...) calls scan_iter (not KEYS)."""
        bridge, client = _make_bridge()
        client.scan_iter.return_value = iter([])

        bridge.clear(camera_id="cam-3")

        assert client.scan_iter.called
        assert not client.keys.called

    def test_clear_all_cameras_uses_wildcard_pattern(self):
        """clear(camera_id=None) uses 'reid:*:*' pattern to match all cameras."""
        bridge, client = _make_bridge()
        client.scan_iter.return_value = iter([])

        bridge.clear(camera_id=None)

        scan_kwargs = client.scan_iter.call_args
        pattern_used = scan_kwargs[1].get("match") if scan_kwargs[1] else scan_kwargs[0][0] if scan_kwargs[0] else ""
        assert "*" in pattern_used

    def test_clear_returns_zero_when_no_keys(self):
        """clear() returns 0 when the pattern matches nothing."""
        bridge, client = _make_bridge()
        client.scan_iter.return_value = iter([])

        count = bridge.clear()

        assert count == 0

    def test_clear_connection_error_returns_zero_not_raises(self):
        """clear() catches ConnectionError and returns 0 without raising."""
        bridge, client = _make_bridge()
        client.scan_iter.side_effect = ConnectionError("down")

        count = bridge.clear()

        assert count == 0


# ---------------------------------------------------------------------------
# TestReIDBridgeMsgpackRoundtrip
# ---------------------------------------------------------------------------


class TestReIDBridgeMsgpackRoundtrip:
    def test_msgpack_roundtrip_embedding_fidelity(self):
        """Embedding packed by publish() is faithfully recovered by unpackb()."""
        bridge, client = _make_bridge(camera_id="test-cam")
        original_emb = _unit_vec(dim=512, seed=42)

        bridge.publish(track_id=99, embedding=original_emb)

        raw_bytes = client.set.call_args[0][1]
        entry = msgpack.unpackb(raw_bytes, raw=False)
        recovered = np.array(entry["embedding"], dtype=np.float32)

        np.testing.assert_allclose(recovered, original_emb, rtol=1e-5)

    def test_msgpack_roundtrip_metadata(self):
        """track_id, camera_id, label, and bbox survive msgpack roundtrip."""
        bridge, client = _make_bridge(camera_id="roundtrip-cam")
        emb = _unit_vec()
        bbox = (5.0, 10.0, 55.0, 110.0)

        bridge.publish(
            track_id=7,
            embedding=emb,
            bbox=bbox,
            label=3,
            timestamp=1234567890.0,
        )

        raw_bytes = client.set.call_args[0][1]
        entry = msgpack.unpackb(raw_bytes, raw=False)

        assert entry["track_id"] == 7
        assert entry["camera_id"] == "roundtrip-cam"
        assert entry["label"] == 3
        assert entry["bbox"] == list(bbox)
        assert entry["timestamp"] == pytest.approx(1234567890.0)

    def test_query_correctly_deserialises_cross_camera_entry(self):
        """query() correctly deserialises and computes similarity from packed bytes."""
        bridge, client = _make_bridge(similarity_thresh=0.0)
        query_emb = _unit_vec(seed=0)
        stored_emb = _unit_vec(seed=0)  # identical → sim ≈ 1.0

        raw = _pack_entry(50, "cam-z", stored_emb, bbox=[1, 2, 3, 4])
        client.scan_iter.return_value = iter(["reid:cam-z:50"])
        client.get.return_value = raw

        results = bridge.query(query_emb, top_k=1)

        assert len(results) == 1
        assert results[0]["track_id"] == 50
        assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# TestReIDBridgeProperties
# ---------------------------------------------------------------------------


class TestReIDBridgeProperties:
    def test_camera_id_property(self):
        """camera_id property returns the initialised camera identifier."""
        bridge, _ = _make_bridge(camera_id="lobby")
        assert bridge.camera_id == "lobby"

    def test_ttl_property(self):
        """ttl property returns the configured TTL value."""
        bridge, _ = _make_bridge(ttl=600)
        assert bridge.ttl == 600

    def test_similarity_thresh_property(self):
        """similarity_thresh property returns the configured threshold."""
        bridge, _ = _make_bridge(similarity_thresh=0.6)
        assert bridge.similarity_thresh == pytest.approx(0.6)
