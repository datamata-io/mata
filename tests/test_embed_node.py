"""Unit and integration tests for the Embed graph node (Task D4).

Tests cover:
- Embed node initialisation (inputs/outputs, dynamic mapping)
- Provider dispatch via ExecutionContext
- Output artifact structure (Embeddings)
- Instance-ID propagation from ROIs → Embeddings
- Empty-ROIs edge case
- Metrics recording
- normalize flag forwarding
- Custom src/out names
- Missing-provider error handling
- Graph pipeline: Detect → ExtractROIs → Embed
- Graph compilation with Embed
- Parallel Embed nodes
- Node export from mata.nodes
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image as PILImage

from mata.core.artifacts.embeddings import Embeddings
from mata.core.artifacts.image import Image
from mata.core.artifacts.rois import ROIs
from mata.core.graph.context import ExecutionContext
from mata.nodes.embed import Embed

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_ctx(providers: dict[str, dict[str, Any]] | None = None) -> ExecutionContext:
    """Build a real ExecutionContext with given providers."""
    return ExecutionContext(providers=providers or {}, device="cpu")


def _make_rois(n: int = 3, crop_size: int = 32) -> ROIs:
    """Create a ROIs artifact with *n* PIL-image crops."""
    roi_images = [PILImage.new("RGB", (crop_size, crop_size), color=(i * 40, 0, 0)) for i in range(n)]
    source_boxes = [(i * 10, i * 10, i * 10 + crop_size, i * 10 + crop_size) for i in range(n)]
    instance_ids = [f"roi_{i:04d}" for i in range(n)]
    return ROIs(
        roi_images=roi_images,
        instance_ids=instance_ids,
        source_boxes=source_boxes,
    )


def _make_mock_embed(n: int = 3, dim: int = 256) -> MagicMock:
    """Create a mock embed provider whose .embed() returns an (n, dim) array."""
    mock = MagicMock()
    mock.embed.return_value = np.zeros((n, dim), dtype=np.float32)
    return mock


def _make_image(w: int = 200, h: int = 200) -> Image:
    """Minimal Image artifact."""
    data = np.zeros((h, w, 3), dtype=np.uint8)
    return Image(data=data, width=w, height=h)


# ===========================================================================
# TestEmbedNode — unit tests
# ===========================================================================


class TestEmbedNode:
    """Unit tests for the Embed graph node."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_init_sets_inputs_outputs(self):
        """Default init wires rois → embeddings."""
        node = Embed(using="encoder")
        assert node.inputs == {"rois": ROIs}
        assert node.outputs == {"embeddings": Embeddings}

    def test_init_stores_using_and_flags(self):
        """Configuration attributes are stored correctly."""
        node = Embed(using="my_encoder", src="crops", out="embs", normalize=False, name="MyEmbed")
        assert node.using == "my_encoder"
        assert node.src == "crops"
        assert node.out == "embs"
        assert node.normalize is False
        assert node.name == "MyEmbed"

    def test_dynamic_artifact_mapping(self):
        """Custom src/out names update inputs and outputs dicts."""
        node = Embed(using="enc", src="region_crops", out="region_embs")
        assert "region_crops" in node.inputs
        assert node.inputs["region_crops"] is ROIs
        assert "region_embs" in node.outputs
        assert node.outputs["region_embs"] is Embeddings

    def test_default_normalize_is_true(self):
        """normalize defaults to True."""
        node = Embed(using="encoder")
        assert node.normalize is True

    # ------------------------------------------------------------------
    # run() — happy path
    # ------------------------------------------------------------------

    def test_run_calls_provider_embed(self):
        """node.run() delegates to provider.embed(rois, normalize=...)."""
        mock_embed = _make_mock_embed(n=3, dim=128)
        ctx = _make_ctx({"embed": {"encoder": mock_embed}})
        node = Embed(using="encoder")
        rois = _make_rois(3)

        node.run(ctx, **{node.src: rois})

        mock_embed.embed.assert_called_once_with(rois, normalize=True)

    def test_run_returns_embeddings_artifact(self):
        """Result value is an Embeddings instance."""
        mock_embed = _make_mock_embed(n=2, dim=512)
        ctx = _make_ctx({"embed": {"encoder": mock_embed}})
        node = Embed(using="encoder")
        rois = _make_rois(2)

        result = node.run(ctx, **{node.src: rois})

        assert node.out in result
        assert isinstance(result[node.out], Embeddings)

    def test_run_output_shape_matches_rois(self):
        """Embeddings shape equals (num_rois, embedding_dim)."""
        n, dim = 4, 256
        mock_embed = _make_mock_embed(n=n, dim=dim)
        ctx = _make_ctx({"embed": {"encoder": mock_embed}})
        node = Embed(using="encoder")
        rois = _make_rois(n)

        result = node.run(ctx, **{node.src: rois})
        embeddings: Embeddings = result[node.out]

        assert len(embeddings) == n
        assert embeddings.embedding_dim == dim

    def test_instance_ids_propagated(self):
        """instance_ids from the input ROIs appear in the output Embeddings."""
        n = 3
        mock_embed = _make_mock_embed(n=n)
        ctx = _make_ctx({"embed": {"encoder": mock_embed}})
        node = Embed(using="encoder")
        rois = _make_rois(n)

        result = node.run(ctx, **{node.src: rois})
        embeddings: Embeddings = result[node.out]

        assert list(embeddings.instance_ids) == list(rois.instance_ids)

    def test_empty_rois_returns_empty_embeddings(self):
        """No crash and empty Embeddings returned for zero-length ROIs."""
        mock_embed = _make_mock_embed(n=0)
        ctx = _make_ctx({"embed": {"encoder": mock_embed}})
        node = Embed(using="encoder")
        # Empty ROIs (no images, no boxes)
        empty_rois = ROIs(roi_images=[], instance_ids=[], source_boxes=[])

        result = node.run(ctx, **{node.src: empty_rois})

        assert node.out in result
        emb: Embeddings = result[node.out]
        assert isinstance(emb, Embeddings)
        assert len(emb) == 0
        # provider.embed() must NOT be called for empty input
        mock_embed.embed.assert_not_called()

    def test_metrics_recorded(self):
        """num_embeddings and embedding_dim are recorded as metrics."""
        n, dim = 3, 512
        mock_embed = _make_mock_embed(n=n, dim=dim)
        ctx = _make_ctx({"embed": {"encoder": mock_embed}})
        ctx.record_metric = MagicMock()  # Spy on record_metric
        node = Embed(using="encoder")
        rois = _make_rois(n)

        node.run(ctx, **{node.src: rois})

        ctx.record_metric.assert_any_call(node.name, "num_embeddings", n)
        ctx.record_metric.assert_any_call(node.name, "embedding_dim", dim)

    def test_normalize_flag_forwarded_false(self):
        """normalize=False is forwarded to provider.embed()."""
        mock_embed = _make_mock_embed(n=2)
        ctx = _make_ctx({"embed": {"encoder": mock_embed}})
        node = Embed(using="encoder", normalize=False)
        rois = _make_rois(2)

        node.run(ctx, **{node.src: rois})

        mock_embed.embed.assert_called_once_with(rois, normalize=False)

    def test_custom_src_out_names(self):
        """Custom src and out names are reflected in result dict key."""
        mock_embed = _make_mock_embed(n=2)
        ctx = _make_ctx({"embed": {"my_enc": mock_embed}})
        node = Embed(using="my_enc", src="my_rois", out="my_embs")
        rois = _make_rois(2)

        result = node.run(ctx, **{"my_rois": rois})

        assert "my_embs" in result
        assert isinstance(result["my_embs"], Embeddings)

    def test_embeddings_meta_contains_model_name(self):
        """Output Embeddings meta has a 'model' key matching the provider name."""
        mock_embed = _make_mock_embed(n=1)
        ctx = _make_ctx({"embed": {"clip_encoder": mock_embed}})
        node = Embed(using="clip_encoder")
        rois = _make_rois(1)

        result = node.run(ctx, **{node.src: rois})
        emb: Embeddings = result[node.out]

        assert emb.meta.get("model") == "clip_encoder"

    def test_provider_not_found_raises(self):
        """KeyError raised when the embed capability is missing from context."""
        ctx = _make_ctx({})  # No "embed" capability
        node = Embed(using="encoder")
        rois = _make_rois(2)

        with pytest.raises(KeyError, match="embed"):
            node.run(ctx, **{node.src: rois})

    def test_provider_name_not_found_raises(self):
        """KeyError raised when provider name is absent in the capability."""
        ctx = _make_ctx({"embed": {"other_enc": MagicMock()}})
        node = Embed(using="encoder")  # "encoder" not registered
        rois = _make_rois(2)

        with pytest.raises(KeyError, match="encoder"):
            node.run(ctx, **{node.src: rois})

    def test_normalized_flag_stored_in_embeddings(self):
        """Embeddings.normalized matches the node's normalize flag."""
        mock_embed = _make_mock_embed(n=2)
        ctx = _make_ctx({"embed": {"enc": mock_embed}})

        for normalize in (True, False):
            node = Embed(using="enc", normalize=normalize)
            rois = _make_rois(2)
            result = node.run(ctx, **{node.src: rois})
            emb: Embeddings = result[node.out]
            assert emb.normalized == normalize

    def test_single_roi_returns_one_embedding(self):
        """Single-ROI input produces (1, D) output."""
        mock_embed = _make_mock_embed(n=1, dim=128)
        ctx = _make_ctx({"embed": {"encoder": mock_embed}})
        node = Embed(using="encoder")
        rois = _make_rois(1)

        result = node.run(ctx, **{node.src: rois})
        emb: Embeddings = result[node.out]

        assert len(emb) == 1
        assert emb.vectors.shape == (1, 128)


# ===========================================================================
# TestEmbedNodeGraph — graph integration tests
# ===========================================================================


class TestEmbedNodeGraph:
    """Graph-level integration tests for the Embed node."""

    def test_embed_importable_from_nodes(self):
        """Embed is exported from the mata.nodes public namespace."""
        from mata.nodes import Embed as EmbedFromNodes

        assert EmbedFromNodes is Embed

    def test_embed_in_all_list(self):
        """Embed appears in mata.nodes.__all__."""
        import mata.nodes as nodes_module

        assert "Embed" in nodes_module.__all__

    def test_embed_in_compiled_graph(self):
        """A graph containing only an Embed node compiles successfully."""
        from mata.core.graph.graph import Graph

        graph = Graph("embed_only").then(Embed(using="encoder", src="rois", out="embs"))
        # compile() accepts flat providers; Embed has no provider_name → no check needed
        compiled = graph.compile(providers={})
        assert compiled is not None
        assert len(compiled.nodes) == 1

    def test_embed_node_validation(self):
        """Embed node's compiled graph has correct execution order."""
        from mata.core.graph.graph import Graph

        node = Embed(using="encoder", src="rois", out="embeddings")
        graph = Graph("val_test").then(node)
        compiled = graph.compile(providers={})

        assert compiled.validation_result.valid

    def test_detect_extract_embed_pipeline(self):
        """Full Detect → ExtractROIs → Embed pipeline executes end-to-end."""
        from mata.core.graph.graph import Graph
        from mata.core.graph.scheduler import SyncScheduler
        from mata.core.types import Instance, VisionResult
        from mata.nodes.detect import Detect
        from mata.nodes.roi import ExtractROIs

        # Mock detector
        mock_det = MagicMock()
        mock_det.predict.return_value = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 60, 80), score=0.9, label=0, label_name="cat"),
                Instance(bbox=(90, 30, 140, 100), score=0.8, label=1, label_name="dog"),
            ]
        )

        # Mock embedder — return (2, 256) vectors
        mock_embed = _make_mock_embed(n=2, dim=256)

        graph = (
            Graph("detect_roi_embed")
            .then(Detect(using="detr", out="dets"))
            .then(ExtractROIs(src_dets="dets", out="rois"))
            .then(Embed(using="encoder", src="rois", out="embeddings"))
        )

        compiled = graph.compile(providers={"detr": mock_det})
        ctx = ExecutionContext(
            providers={
                "detect": {"detr": mock_det},
                "embed": {"encoder": mock_embed},
            },
            device="cpu",
        )

        result = SyncScheduler().execute(compiled, ctx, {"input.image": _make_image()})

        assert result.has_channel("embeddings")
        embeddings = result.channels["embeddings"]
        assert isinstance(embeddings, Embeddings)
        assert len(embeddings) == 2
        assert embeddings.embedding_dim == 256

    def test_parallel_embed_nodes(self):
        """Two independent Embed nodes execute in parallel (pre-injected ROIs)."""
        from mata.core.artifacts.result import MultiResult
        from mata.core.graph.graph import Graph
        from mata.core.graph.scheduler import SyncScheduler

        rois_a = _make_rois(2, crop_size=16)
        rois_b = _make_rois(3, crop_size=16)
        mock_a = _make_mock_embed(n=2, dim=64)
        mock_b = _make_mock_embed(n=3, dim=64)

        graph = Graph("parallel_embed").parallel(
            [
                Embed(using="enc_a", src="rois_a", out="embs_a", name="EmbedA"),
                Embed(using="enc_b", src="rois_b", out="embs_b", name="EmbedB"),
            ]
        )

        compiled = graph.compile(providers={})  # Embed has no provider_name to validate
        ctx = ExecutionContext(
            providers={"embed": {"enc_a": mock_a, "enc_b": mock_b}},
            device="cpu",
        )

        result = SyncScheduler().execute(
            compiled,
            ctx,
            {"input.rois_a": rois_a, "input.rois_b": rois_b},
        )

        assert isinstance(result, MultiResult)
        assert result.has_channel("embs_a")
        assert result.has_channel("embs_b")
        assert len(result.channels["embs_a"]) == 2
        assert len(result.channels["embs_b"]) == 3

    def test_embed_node_graph_with_filter(self):
        """Detect → Filter → ExtractROIs → Embed compiles and runs."""
        from mata.core.graph.graph import Graph
        from mata.core.graph.scheduler import SyncScheduler
        from mata.core.types import Instance, VisionResult
        from mata.nodes.detect import Detect
        from mata.nodes.filter import Filter
        from mata.nodes.roi import ExtractROIs

        mock_det = MagicMock()
        mock_det.predict.return_value = VisionResult(
            instances=[
                Instance(bbox=(5, 5, 50, 50), score=0.95, label=0, label_name="cat"),
                Instance(bbox=(60, 60, 100, 100), score=0.2, label=1, label_name="dog"),
            ]
        )
        mock_embed = _make_mock_embed(n=1, dim=128)  # only 1 passes score filter

        graph = (
            Graph("detect_filter_roi_embed")
            .then(Detect(using="detr", out="dets"))
            .then(Filter(src="dets", score_gt=0.5, out="filtered"))
            .then(ExtractROIs(src_dets="filtered", out="rois"))
            .then(Embed(using="enc", src="rois", out="embeddings"))
        )

        compiled = graph.compile(providers={"detr": mock_det})
        ctx = ExecutionContext(
            providers={"detect": {"detr": mock_det}, "embed": {"enc": mock_embed}},
            device="cpu",
        )

        result = SyncScheduler().execute(compiled, ctx, {"input.image": _make_image()})

        assert result.has_channel("embeddings")
        assert isinstance(result.channels["embeddings"], Embeddings)
