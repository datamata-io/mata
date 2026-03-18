"""Tests for ReID graph node.

Comprehensive unit tests verifying provider resolution, publish/query behaviour,
CrossMatches output, metrics recording, dynamic input mapping, and edge cases.
All tests use mocked ReIDBridge — no Valkey connection required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from mata.core.artifacts.cross_matches import CrossMatches
from mata.core.artifacts.embeddings import Embeddings
from mata.core.artifacts.tracks import Track, Tracks
from mata.nodes.reid import ReID

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_track(track_id: int, state: str = "active") -> Track:
    return Track(
        track_id=track_id,
        bbox=(float(track_id * 10), float(track_id * 10), float(track_id * 10 + 40), float(track_id * 10 + 40)),
        score=0.9,
        label="car",
        age=1,
        state=state,
    )


def _make_tracks(*track_ids: int, state: str = "active") -> Tracks:
    return Tracks(tracks=[_make_track(tid, state=state) for tid in track_ids], frame_id="frame_001")


def _make_embeddings(n: int, dim: int = 128) -> Embeddings:
    vecs = np.random.randn(n, dim).astype(np.float32)
    # L2-normalise each row
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)
    return Embeddings(vectors=vecs, normalized=True)


def _make_bridge(camera_id: str = "cam-1", query_returns: list[dict] | None = None) -> MagicMock:
    bridge = MagicMock()
    bridge.camera_id = camera_id
    bridge.query.return_value = query_returns if query_returns is not None else []
    return bridge


def _make_ctx(bridge: Any, using: str = "bridge") -> MagicMock:
    ctx = MagicMock()
    ctx.get_provider.return_value = bridge
    ctx.record_metric = MagicMock()
    return ctx


# ---------------------------------------------------------------------------
# Protocol / construction tests
# ---------------------------------------------------------------------------


class TestReIDProtocol:
    """Tests for class-level input/output declarations."""

    def test_default_inputs_keys(self):
        node = ReID(using="bridge")
        assert "tracks" in node.inputs
        assert "embeddings" in node.inputs

    def test_default_outputs_keys(self):
        node = ReID(using="bridge")
        assert "cross_matches" in node.outputs

    def test_inputs_artifact_types(self):
        node = ReID(using="bridge")
        assert node.inputs["tracks"] is Tracks
        assert node.inputs["embeddings"] is Embeddings

    def test_outputs_artifact_type(self):
        node = ReID(using="bridge")
        assert node.outputs["cross_matches"] is CrossMatches

    def test_default_name(self):
        node = ReID(using="bridge")
        assert node.name == "ReID"

    def test_custom_name(self):
        node = ReID(using="bridge", name="cross_reid")
        assert node.name == "cross_reid"

    def test_default_top_k(self):
        node = ReID(using="bridge")
        assert node.top_k == 1

    def test_custom_top_k(self):
        node = ReID(using="bridge", top_k=3)
        assert node.top_k == 3


# ---------------------------------------------------------------------------
# Dynamic input mapping
# ---------------------------------------------------------------------------


class TestDynamicInputMapping:
    """Tests for configurable input/output artifact keys."""

    def test_custom_tracks_src(self):
        node = ReID(using="bridge", tracks_src="my_tracks")
        assert "my_tracks" in node.inputs
        assert "tracks" not in node.inputs

    def test_custom_embeddings_src(self):
        node = ReID(using="bridge", embeddings_src="my_embs")
        assert "my_embs" in node.inputs
        assert "embeddings" not in node.inputs

    def test_custom_out_key(self):
        node = ReID(using="bridge", out="reid_out")
        assert "reid_out" in node.outputs
        assert "cross_matches" not in node.outputs

    def test_custom_keys_run(self):
        node = ReID(using="bridge", tracks_src="t", embeddings_src="e", out="cm")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        tracks = _make_tracks(1, 2)
        embs = _make_embeddings(2)

        result = node.run(ctx, t=tracks, e=embs)
        assert "cm" in result

    def test_storing_tracks_src_and_embeddings_src_attrs(self):
        node = ReID(using="bridge", tracks_src="t", embeddings_src="e")
        assert node.tracks_src == "t"
        assert node.embeddings_src == "e"


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


class TestProviderResolution:
    """Tests that the node resolves its provider via ctx.get_provider("reid", name)."""

    def test_get_provider_called_with_reid_capability(self):
        node = ReID(using="my_bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        tracks = _make_tracks(1)
        embs = _make_embeddings(1)

        node.run(ctx, tracks=tracks, embeddings=embs)

        ctx.get_provider.assert_called_once_with("reid", "my_bridge")

    def test_get_provider_respects_using_name(self):
        node = ReID(using="valkey_cam1")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        tracks = _make_tracks(1)
        embs = _make_embeddings(1)

        node.run(ctx, tracks=tracks, embeddings=embs)

        ctx.get_provider.assert_called_once_with("reid", "valkey_cam1")


# ---------------------------------------------------------------------------
# Publish behaviour
# ---------------------------------------------------------------------------


class TestPublishBehaviour:
    """Tests verifying bridge.publish() is called per active track."""

    def test_publish_called_once_for_single_track(self):
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        tracks = _make_tracks(7)
        embs = _make_embeddings(1)

        node.run(ctx, tracks=tracks, embeddings=embs)

        assert bridge.publish.call_count == 1
        call_kwargs = bridge.publish.call_args
        assert call_kwargs.kwargs["track_id"] == 7

    def test_publish_called_per_active_track(self):
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        tracks = _make_tracks(1, 2, 3)
        embs = _make_embeddings(3)

        node.run(ctx, tracks=tracks, embeddings=embs)

        assert bridge.publish.call_count == 3
        published_ids = {c.kwargs["track_id"] for c in bridge.publish.call_args_list}
        assert published_ids == {1, 2, 3}

    def test_publish_not_called_for_lost_tracks(self):
        """Only active tracks (from get_active_tracks()) are published."""
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        active = _make_track(1, state="active")
        lost = _make_track(2, state="lost")
        tracks = Tracks(tracks=[active, lost], frame_id="frame_001")
        embs = _make_embeddings(2)

        node.run(ctx, tracks=tracks, embeddings=embs)

        assert bridge.publish.call_count == 1

    def test_publish_passes_correct_embedding(self):
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        tracks = _make_tracks(1)
        embs = _make_embeddings(1)

        node.run(ctx, tracks=tracks, embeddings=embs)

        call_kwargs = bridge.publish.call_args.kwargs
        np.testing.assert_array_almost_equal(call_kwargs["embedding"], embs.vectors[0])

    def test_publish_passes_bbox(self):
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        tracks = _make_tracks(5)
        embs = _make_embeddings(1)

        node.run(ctx, tracks=tracks, embeddings=embs)

        call_kwargs = bridge.publish.call_args.kwargs
        assert call_kwargs["bbox"] == tracks.tracks[0].bbox


# ---------------------------------------------------------------------------
# Query behaviour
# ---------------------------------------------------------------------------


class TestQueryBehaviour:
    """Tests verifying bridge.query() is called per active track."""

    def test_query_called_once_for_single_track(self):
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        node.run(ctx, tracks=_make_tracks(1), embeddings=_make_embeddings(1))

        assert bridge.query.call_count == 1

    def test_query_called_per_active_track(self):
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        node.run(ctx, tracks=_make_tracks(1, 2, 3), embeddings=_make_embeddings(3))

        assert bridge.query.call_count == 3

    def test_query_called_with_exclude_camera(self):
        node = ReID(using="bridge")
        bridge = _make_bridge(camera_id="cam-1")
        ctx = _make_ctx(bridge)

        node.run(ctx, tracks=_make_tracks(1), embeddings=_make_embeddings(1))

        call_kwargs = bridge.query.call_args.kwargs
        assert call_kwargs["exclude_camera"] == "cam-1"

    def test_top_k_forwarding(self):
        node = ReID(using="bridge", top_k=5)
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        node.run(ctx, tracks=_make_tracks(1), embeddings=_make_embeddings(1))

        call_kwargs = bridge.query.call_args.kwargs
        assert call_kwargs["top_k"] == 5

    def test_default_top_k_forwarded(self):
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        node.run(ctx, tracks=_make_tracks(1), embeddings=_make_embeddings(1))

        call_kwargs = bridge.query.call_args.kwargs
        assert call_kwargs["top_k"] == 1


# ---------------------------------------------------------------------------
# Output: CrossMatches production
# ---------------------------------------------------------------------------


class TestCrossMatchesOutput:
    """Tests verifying CrossMatches artifact is correctly produced."""

    def test_output_key_is_default(self):
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        result = node.run(ctx, tracks=_make_tracks(1), embeddings=_make_embeddings(1))

        assert "cross_matches" in result

    def test_output_is_cross_matches_instance(self):
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        result = node.run(ctx, tracks=_make_tracks(1), embeddings=_make_embeddings(1))

        assert isinstance(result["cross_matches"], CrossMatches)

    def test_no_matches_when_query_returns_empty(self):
        node = ReID(using="bridge")
        bridge = _make_bridge(query_returns=[])
        ctx = _make_ctx(bridge)

        result = node.run(ctx, tracks=_make_tracks(1, 2), embeddings=_make_embeddings(2))

        cm = result["cross_matches"]
        assert len(cm) == 0

    def test_matches_populated_from_query_results(self):
        node = ReID(using="bridge")
        query_response = [
            {
                "camera_id": "cam-2",
                "track_id": 9,
                "similarity": 0.91,
                "bbox": [100.0, 80.0, 150.0, 200.0],
            }
        ]
        bridge = _make_bridge(query_returns=query_response)
        ctx = _make_ctx(bridge)

        result = node.run(ctx, tracks=_make_tracks(3), embeddings=_make_embeddings(1))

        cm = result["cross_matches"]
        assert len(cm) == 1
        match = cm.matches[0]
        assert match.local_track_id == 3
        assert match.remote_camera_id == "cam-2"
        assert match.remote_track_id == 9
        assert match.similarity == 0.91
        assert match.remote_bbox == (100.0, 80.0, 150.0, 200.0)

    def test_matches_from_multiple_tracks(self):
        """Each track has one query hit → two total matches."""
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        def query_side_effect(emb, exclude_camera, top_k):
            call_count = bridge.query.call_count
            return [{"camera_id": "cam-2", "track_id": call_count, "similarity": 0.8, "bbox": None}]

        bridge.query.side_effect = query_side_effect

        result = node.run(ctx, tracks=_make_tracks(1, 2), embeddings=_make_embeddings(2))

        cm = result["cross_matches"]
        assert len(cm) == 2

    def test_camera_id_propagated_from_bridge(self):
        node = ReID(using="bridge")
        bridge = _make_bridge(camera_id="cam-99")
        ctx = _make_ctx(bridge)

        result = node.run(ctx, tracks=_make_tracks(1), embeddings=_make_embeddings(1))

        assert result["cross_matches"].camera_id == "cam-99"

    def test_no_matches_with_no_bbox_in_query_result(self):
        """Query results without bbox → remote_bbox is None in CrossMatch."""
        node = ReID(using="bridge")
        query_response = [
            {"camera_id": "cam-2", "track_id": 4, "similarity": 0.75}
        ]
        bridge = _make_bridge(query_returns=query_response)
        ctx = _make_ctx(bridge)

        result = node.run(ctx, tracks=_make_tracks(1), embeddings=_make_embeddings(1))

        match = result["cross_matches"].matches[0]
        assert match.remote_bbox is None


# ---------------------------------------------------------------------------
# Edge cases: empty inputs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for empty/mismatched inputs."""

    def test_empty_tracks_returns_empty_cross_matches(self):
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        tracks = Tracks(tracks=[], frame_id="frame_001")
        embs = _make_embeddings(3)

        result = node.run(ctx, tracks=tracks, embeddings=embs)

        cm = result["cross_matches"]
        assert isinstance(cm, CrossMatches)
        assert len(cm) == 0
        bridge.publish.assert_not_called()
        bridge.query.assert_not_called()

    def test_empty_embeddings_returns_empty_cross_matches(self):
        """Zero embedding vectors → no publish/query calls."""
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        tracks = _make_tracks(1, 2)
        vecs = np.empty((0, 128), dtype=np.float32)
        embs = Embeddings(vectors=vecs)

        result = node.run(ctx, tracks=tracks, embeddings=embs)

        cm = result["cross_matches"]
        assert isinstance(cm, CrossMatches)
        assert len(cm) == 0
        bridge.publish.assert_not_called()
        bridge.query.assert_not_called()

    def test_mismatched_counts_more_tracks_than_embeddings(self):
        """3 active tracks but only 2 embeddings → process min(2, 3) = 2 tracks."""
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        tracks = _make_tracks(1, 2, 3)
        embs = _make_embeddings(2)

        node.run(ctx, tracks=tracks, embeddings=embs)

        assert bridge.publish.call_count == 2
        assert bridge.query.call_count == 2

    def test_more_embeddings_than_tracks(self):
        """2 active tracks but 5 embeddings → only first 2 embeddings used."""
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        tracks = _make_tracks(1, 2)
        embs = _make_embeddings(5)

        node.run(ctx, tracks=tracks, embeddings=embs)

        assert bridge.publish.call_count == 2
        assert bridge.query.call_count == 2

    def test_only_lost_tracks_returns_empty(self):
        """All tracks lost → get_active_tracks() returns empty → no ops."""
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        tracks = Tracks(
            tracks=[_make_track(1, state="lost"), _make_track(2, state="lost")],
            frame_id="frame_001",
        )
        embs = _make_embeddings(2)

        result = node.run(ctx, tracks=tracks, embeddings=embs)

        assert len(result["cross_matches"]) == 0
        bridge.publish.assert_not_called()

    def test_no_matches_from_bridge_empty_cross_matches(self):
        """bridge.query() returns [] → CrossMatches has no matches."""
        node = ReID(using="bridge")
        bridge = _make_bridge(query_returns=[])
        ctx = _make_ctx(bridge)

        result = node.run(ctx, tracks=_make_tracks(1, 2), embeddings=_make_embeddings(2))

        assert len(result["cross_matches"]) == 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests for ctx.record_metric() calls."""

    def test_num_tracks_published_recorded(self):
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        node.run(ctx, tracks=_make_tracks(1, 2, 3), embeddings=_make_embeddings(3))

        ctx.record_metric.assert_any_call(node.name, "num_tracks_published", 3)

    def test_num_cross_matches_recorded(self):
        node = ReID(using="bridge")
        query_response = [{"camera_id": "cam-2", "track_id": 5, "similarity": 0.8, "bbox": None}]
        bridge = _make_bridge(query_returns=query_response)
        ctx = _make_ctx(bridge)

        node.run(ctx, tracks=_make_tracks(1, 2), embeddings=_make_embeddings(2))

        ctx.record_metric.assert_any_call(node.name, "num_cross_matches", 2)

    def test_zero_tracks_metrics(self):
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        tracks = Tracks(tracks=[], frame_id="frame_001")
        embs = _make_embeddings(0, dim=128) if False else Embeddings(vectors=np.empty((0, 128), dtype=np.float32))

        node.run(ctx, tracks=tracks, embeddings=embs)

        ctx.record_metric.assert_any_call(node.name, "num_tracks_published", 0)
        ctx.record_metric.assert_any_call(node.name, "num_cross_matches", 0)

    def test_metric_recorded_with_custom_node_name(self):
        node = ReID(using="bridge", name="reid_cam1")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        node.run(ctx, tracks=_make_tracks(1), embeddings=_make_embeddings(1))

        calls = [str(c) for c in ctx.record_metric.call_args_list]
        assert any("reid_cam1" in c for c in calls)

    def test_mismatched_counts_num_tracks_published(self):
        """With 3 tracks and 2 embeddings, num_tracks_published should be 2."""
        node = ReID(using="bridge")
        bridge = _make_bridge()
        ctx = _make_ctx(bridge)

        tracks = _make_tracks(1, 2, 3)
        embs = _make_embeddings(2)

        node.run(ctx, tracks=tracks, embeddings=embs)

        ctx.record_metric.assert_any_call(node.name, "num_tracks_published", 2)
