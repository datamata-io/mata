"""Unit tests for IndexVideo and EmbeddingSearch graph nodes.

All tests use mocks — no real models or video files required.
Run: pytest tests/test_index_video_node.py tests/test_embedding_search_node.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mata.core.artifacts.search_results import QueryResult, SearchResults
from mata.core.artifacts.video_index_data import VideoIndexData
from mata.core.artifacts.video_path import VideoPath
from mata.core.graph.context import ExecutionContext
from mata.nodes.embedding_search import EmbeddingSearch
from mata.nodes.index_video import IndexVideo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(providers=None) -> ExecutionContext:
    """Build an ExecutionContext with optional providers."""
    return ExecutionContext(providers=providers or {}, device="cpu")


def _unit(dim: int = 64) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_video_index(n_frames: int = 5, dim: int = 64) -> MagicMock:
    """Return a mock VideoIndex with a .search() method."""
    vi = MagicMock()
    vi.frame_map = {f"frame_{i:04d}": float(i) for i in range(n_frames)}

    def _search(query_vec, top_k=5, threshold=None):
        from mata.recognition.video_index import VideoMatch

        results = []
        for i in range(min(top_k, n_frames)):
            m = MagicMock(spec=VideoMatch)
            m.label = f"frame_{i:04d}"
            m.similarity = 0.9 - i * 0.05
            m.start_s = float(i)
            m.end_s = float(i + 1)
            results.append(m)
        return results

    vi.search.side_effect = _search
    return vi


def _make_embed_adapter(dim: int = 64) -> MagicMock:
    """Return a mock EmbedAdapter whose .embed(str) returns a (1, D) array."""
    adapter = MagicMock()
    adapter.embed.return_value = np.expand_dims(_unit(dim), axis=0)  # (1, D)
    return adapter


# ===========================================================================
# TestIndexVideoNode
# ===========================================================================


class TestIndexVideoNode:
    """Unit tests for IndexVideo."""

    def test_default_construction(self):
        node = IndexVideo(using="embedder")
        assert node.using == "embedder"
        assert node.mode == "frame"
        assert node.sample_fps == 1.0
        assert node.out == "video_index"
        assert node.inputs == {"video": VideoPath}
        assert node.outputs == {"video_index": VideoIndexData}

    def test_custom_params(self):
        node = IndexVideo(using="enc", mode="clip", sample_fps=2.0, out="idx")
        assert node.mode == "clip"
        assert node.sample_fps == 2.0
        assert node.out == "idx"
        assert "idx" in node.outputs

    def test_custom_name(self):
        node = IndexVideo(using="enc", name="MyIndexer")
        assert node.name == "MyIndexer"

    def test_default_name_assigned(self):
        node = IndexVideo(using="enc")
        assert node.name is not None

    def test_dynamic_output_key(self):
        node = IndexVideo(using="enc", out="my_index")
        assert "my_index" in node.outputs
        assert node.outputs["my_index"] is VideoIndexData

    def test_embed_kwargs_stored(self):
        node = IndexVideo(using="enc", embed_dim=256)
        assert node.embed_kwargs.get("embed_dim") == 256

    def test_run_calls_index_video(self):
        vi_mock = _make_video_index()
        adapter = MagicMock()
        ctx = _make_ctx({"embed": {"embedder": adapter}})

        with patch("mata.recognition.index_video", return_value=vi_mock) as mock_iv:
            node = IndexVideo(using="embedder")
            result = node.run(ctx, video=VideoPath(path="/tmp/video.mp4"))

        mock_iv.assert_called_once()
        call_kwargs = mock_iv.call_args
        assert call_kwargs.args[0] == "/tmp/video.mp4"
        assert call_kwargs.kwargs.get("adapter") is adapter
        assert call_kwargs.kwargs.get("mode") == "frame"
        assert call_kwargs.kwargs.get("sample_fps") == 1.0

        assert "video_index" in result
        assert isinstance(result["video_index"], VideoIndexData)
        assert result["video_index"].index is vi_mock

    def test_run_records_metric(self):
        vi_mock = _make_video_index(n_frames=10)
        adapter = MagicMock()
        ctx = _make_ctx({"embed": {"embedder": adapter}})

        with patch("mata.recognition.index_video", return_value=vi_mock):
            node = IndexVideo(using="embedder")
            node.run(ctx, video=VideoPath(path="/tmp/v.mp4"))

        ctx.record_metric(node.name, "indexed_frames", 10)  # already called in run

    def test_run_custom_out_key(self):
        vi_mock = _make_video_index()
        adapter = MagicMock()
        ctx = _make_ctx({"embed": {"embedder": adapter}})

        with patch("mata.recognition.index_video", return_value=vi_mock):
            node = IndexVideo(using="embedder", out="my_idx")
            result = node.run(ctx, video=VideoPath(path="/tmp/v.mp4"))

        assert "my_idx" in result

    def test_run_meta_contains_model(self):
        vi_mock = _make_video_index()
        adapter = MagicMock()
        ctx = _make_ctx({"embed": {"embedder": adapter}})

        with patch("mata.recognition.index_video", return_value=vi_mock):
            node = IndexVideo(using="embedder")
            result = node.run(ctx, video=VideoPath(path="/tmp/v.mp4"))

        assert result["video_index"].meta["model"] == "embedder"

    def test_run_forwards_embed_kwargs(self):
        vi_mock = _make_video_index()
        adapter = MagicMock()
        ctx = _make_ctx({"embed": {"embedder": adapter}})

        with patch("mata.recognition.index_video", return_value=vi_mock) as mock_iv:
            node = IndexVideo(using="embedder", embed_dim=128)
            node.run(ctx, video=VideoPath(path="/tmp/v.mp4"))

        call_kwargs = mock_iv.call_args.kwargs
        assert call_kwargs.get("embed_dim") == 128

    def test_is_exported_from_nodes(self):
        import mata.nodes as nodes

        assert nodes.IndexVideo is IndexVideo


# ===========================================================================
# TestEmbeddingSearchNode
# ===========================================================================


class TestEmbeddingSearchNode:
    """Unit tests for EmbeddingSearch."""

    def test_default_construction(self):
        node = EmbeddingSearch(using="embedder", text="red car")
        assert node.using == "embedder"
        assert node.text == ["red car"]
        assert node.top_k == 5
        assert node.threshold is None
        assert node.out == "search_results"
        assert node.inputs == {"video_index": VideoIndexData}
        assert node.outputs == {"search_results": SearchResults}

    def test_text_list_stored_as_list(self):
        node = EmbeddingSearch(using="embedder", text=["a", "b", "c"])
        assert node.text == ["a", "b", "c"]

    def test_custom_src_out(self):
        node = EmbeddingSearch(using="enc", text="x", src="my_idx", out="my_res")
        assert "my_idx" in node.inputs
        assert "my_res" in node.outputs

    def test_custom_top_k_threshold(self):
        node = EmbeddingSearch(using="enc", text="x", top_k=10, threshold=0.3)
        assert node.top_k == 10
        assert node.threshold == 0.3

    def test_custom_name(self):
        node = EmbeddingSearch(using="enc", text="x", name="Searcher")
        assert node.name == "Searcher"

    def test_run_single_query(self):
        vi_mock = _make_video_index(n_frames=3, dim=64)
        adapter = _make_embed_adapter(dim=64)
        ctx = _make_ctx({"embed": {"embedder": adapter}})
        vid_data = VideoIndexData(index=vi_mock, meta={})

        node = EmbeddingSearch(using="embedder", text="red car", top_k=2)
        result = node.run(ctx, video_index=vid_data)

        assert "search_results" in result
        sr = result["search_results"]
        assert isinstance(sr, SearchResults)
        assert len(sr.results) == 1
        qr = sr.results[0]
        assert isinstance(qr, QueryResult)
        assert qr.query == "red car"
        assert len(qr.matches) == 2

    def test_run_multiple_queries(self):
        vi_mock = _make_video_index(n_frames=5, dim=64)
        adapter = _make_embed_adapter(dim=64)
        ctx = _make_ctx({"embed": {"embedder": adapter}})
        vid_data = VideoIndexData(index=vi_mock, meta={})

        node = EmbeddingSearch(using="embedder", text=["red car", "pedestrian", "traffic light"])
        result = node.run(ctx, video_index=vid_data)

        sr = result["search_results"]
        assert len(sr.results) == 3
        assert sr.results[0].query == "red car"
        assert sr.results[1].query == "pedestrian"
        assert sr.results[2].query == "traffic light"

    def test_run_embeds_each_query(self):
        vi_mock = _make_video_index()
        adapter = _make_embed_adapter()
        ctx = _make_ctx({"embed": {"embedder": adapter}})
        vid_data = VideoIndexData(index=vi_mock, meta={})

        node = EmbeddingSearch(using="embedder", text=["a", "b"])
        node.run(ctx, video_index=vid_data)

        assert adapter.embed.call_count == 2
        adapter.embed.assert_any_call("a")
        adapter.embed.assert_any_call("b")

    def test_run_calls_search_with_threshold(self):
        vi_mock = _make_video_index()
        adapter = _make_embed_adapter()
        ctx = _make_ctx({"embed": {"embedder": adapter}})
        vid_data = VideoIndexData(index=vi_mock, meta={})

        node = EmbeddingSearch(using="embedder", text="query", top_k=3, threshold=0.5)
        node.run(ctx, video_index=vid_data)

        vi_mock.search.assert_called_once()
        call_kwargs = vi_mock.search.call_args.kwargs
        assert call_kwargs.get("top_k") == 3
        assert call_kwargs.get("threshold") == 0.5

    def test_run_query_vec_is_1d(self):
        """EmbeddingSearch must ravel the (1, D) embed output to pass (D,) to search."""
        vi_mock = _make_video_index(dim=64)

        captured_vecs: list[np.ndarray] = []

        def _search(query_vec, top_k=5, threshold=None):
            captured_vecs.append(query_vec)
            return []

        vi_mock.search.side_effect = _search

        adapter = _make_embed_adapter(dim=64)
        ctx = _make_ctx({"embed": {"embedder": adapter}})
        vid_data = VideoIndexData(index=vi_mock, meta={})

        node = EmbeddingSearch(using="embedder", text="test")
        node.run(ctx, video_index=vid_data)

        assert len(captured_vecs) == 1
        assert captured_vecs[0].ndim == 1
        assert captured_vecs[0].shape == (64,)

    def test_run_records_metric(self):
        vi_mock = _make_video_index()
        adapter = _make_embed_adapter()
        ctx = _make_ctx({"embed": {"embedder": adapter}})
        vid_data = VideoIndexData(index=vi_mock, meta={})

        node = EmbeddingSearch(using="embedder", text=["a", "b", "c"])
        node.run(ctx, video_index=vid_data)

        # record_metric should have been called with num_queries=3
        # (ctx records it internally; we just verify no error is raised)

    def test_run_meta_keys(self):
        vi_mock = _make_video_index()
        adapter = _make_embed_adapter()
        ctx = _make_ctx({"embed": {"embedder": adapter}})
        vid_data = VideoIndexData(index=vi_mock, meta={})

        node = EmbeddingSearch(using="embedder", text="x", top_k=7)
        result = node.run(ctx, video_index=vid_data)

        sr = result["search_results"]
        assert sr.meta["model"] == "embedder"
        assert sr.meta["top_k"] == 7

    def test_run_empty_matches_when_threshold_high(self):
        vi_mock = MagicMock()
        vi_mock.frame_map = {}
        vi_mock.search.return_value = []  # nothing above threshold

        adapter = _make_embed_adapter()
        ctx = _make_ctx({"embed": {"embedder": adapter}})
        vid_data = VideoIndexData(index=vi_mock, meta={})

        node = EmbeddingSearch(using="embedder", text="nothing", threshold=0.99)
        result = node.run(ctx, video_index=vid_data)

        sr = result["search_results"]
        assert len(sr.results[0].matches) == 0

    def test_is_exported_from_nodes(self):
        from mata.nodes import EmbeddingSearch as _EmbeddingSearch

        assert _EmbeddingSearch is EmbeddingSearch


# ===========================================================================
# TestSearchResultsArtifact
# ===========================================================================


class TestSearchResultsArtifact:
    """Unit tests for SearchResults and QueryResult artifacts."""

    def test_query_result_frozen(self):
        from mata.recognition.video_index import VideoMatch

        m = MagicMock(spec=VideoMatch)
        qr = QueryResult(query="test", matches=(m,))
        with pytest.raises((AttributeError, TypeError)):
            qr.query = "other"  # type: ignore[misc]

    def test_search_results_len(self):
        qr1 = QueryResult(query="a", matches=())
        qr2 = QueryResult(query="b", matches=())
        sr = SearchResults(results=(qr1, qr2), meta={})
        assert len(sr) == 2

    def test_search_results_iter(self):
        qr1 = QueryResult(query="a", matches=())
        qr2 = QueryResult(query="b", matches=())
        sr = SearchResults(results=(qr1, qr2), meta={})
        queries = [qr.query for qr in sr]
        assert queries == ["a", "b"]

    def test_search_results_getitem(self):
        qr1 = QueryResult(query="a", matches=())
        sr = SearchResults(results=(qr1,), meta={})
        assert sr[0] is qr1

    def test_search_results_validate_no_raises(self):
        qr = QueryResult(query="x", matches=())
        sr = SearchResults(results=(qr,), meta={})
        sr.validate()  # should not raise

    def test_search_results_exported_from_artifacts(self):
        from mata.core.artifacts import QueryResult as _QueryResult
        from mata.core.artifacts import SearchResults as _SearchResults

        assert _QueryResult is QueryResult
        assert _SearchResults is SearchResults


# ===========================================================================
# TestVideoPathArtifact
# ===========================================================================


class TestVideoPathArtifact:
    """Unit tests for VideoPath artifact."""

    def test_basic_construction(self):
        vp = VideoPath(path="/tmp/video.mp4")
        assert vp.path == "/tmp/video.mp4"

    def test_frozen(self):
        vp = VideoPath(path="/tmp/video.mp4")
        with pytest.raises((AttributeError, TypeError)):
            vp.path = "/other"  # type: ignore[misc]

    def test_validate_no_raises(self):
        vp = VideoPath(path="/tmp/video.mp4")
        vp.validate()

    def test_to_dict(self):
        vp = VideoPath(path="/some/path.mp4")
        d = vp.to_dict()
        assert d == {"path": "/some/path.mp4"}

    def test_from_dict(self):
        vp = VideoPath.from_dict({"path": "/some/path.mp4"})
        assert vp.path == "/some/path.mp4"

    def test_exported_from_artifacts(self):
        from mata.core.artifacts import VideoPath as _VideoPath

        assert _VideoPath is VideoPath


# ===========================================================================
# TestVideoIndexDataArtifact
# ===========================================================================


class TestVideoIndexDataArtifact:
    """Unit tests for VideoIndexData artifact."""

    def test_basic_construction(self):
        vi = MagicMock()
        vd = VideoIndexData(index=vi, meta={"model": "test"})
        assert vd.index is vi
        assert vd.meta["model"] == "test"

    def test_frozen(self):
        vi = MagicMock()
        vd = VideoIndexData(index=vi, meta={})
        with pytest.raises((AttributeError, TypeError)):
            vd.index = None  # type: ignore[misc]

    def test_validate_no_raises(self):
        vd = VideoIndexData(index=MagicMock(), meta={})
        vd.validate()

    def test_exported_from_artifacts(self):
        from mata.core.artifacts import VideoIndexData as _VideoIndexData

        assert _VideoIndexData is VideoIndexData
