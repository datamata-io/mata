"""Unit tests for Task 5.7: VLM Nodes.

Tests cover:
- VLMDescribe text output and entity extraction
- VLMDetect entity extraction and auto-promotion
- VLMQuery with single image, multi-image, and different output modes
- PromoteEntities exact/fuzzy matching and no-match handling
- Output naming and kwargs passthrough
- Error handling (missing provider)
- Metrics recording
- Artifact conversion from VisionResult
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.graph.context import ExecutionContext
from mata.core.types import Entity, Instance, VisionResult
from mata.nodes.promote_entities import PromoteEntities
from mata.nodes.vlm_describe import VLMDescribe
from mata.nodes.vlm_detect import VLMDetect
from mata.nodes.vlm_query import VLMQuery

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_image():
    """Create a minimal Image artifact for testing."""
    data = np.zeros((100, 200, 3), dtype=np.uint8)
    return Image(data=data, width=200, height=100)


@pytest.fixture
def sample_image_2():
    """Create a second Image artifact for multi-image tests."""
    data = np.ones((100, 200, 3), dtype=np.uint8) * 128
    return Image(data=data, width=200, height=100)


@pytest.fixture
def sample_image_3():
    """Create a third Image artifact for multi-image tests."""
    data = np.ones((100, 200, 3), dtype=np.uint8) * 255
    return Image(data=data, width=200, height=100)


@pytest.fixture
def describe_result():
    """VisionResult with text description (what a VLM describe returns)."""
    return VisionResult(
        instances=[],
        text="A beautiful landscape with mountains and a river.",
        prompt="Describe this image in detail.",
        meta={"model": "qwen3-vl", "output_mode": "describe"},
    )


@pytest.fixture
def detect_entities_result():
    """VisionResult with entities (what a VLM detect returns)."""
    return VisionResult(
        instances=[],
        entities=[
            Entity(label="cat", score=0.9, attributes={"color": "orange"}),
            Entity(label="dog", score=0.8, attributes={"breed": "labrador"}),
            Entity(label="bird", score=0.7),
        ],
        text='{"objects": ["cat", "dog", "bird"]}',
        meta={"model": "qwen3-vl", "output_mode": "detect"},
    )


@pytest.fixture
def detect_promoted_result():
    """VisionResult with both entities and auto-promoted instances."""
    return VisionResult(
        instances=[
            Instance(
                bbox=(10, 20, 100, 150),
                score=0.9,
                label=0,
                label_name="cat",
            ),
            Instance(
                bbox=(200, 50, 300, 200),
                score=0.8,
                label=1,
                label_name="dog",
            ),
        ],
        entities=[
            Entity(label="cat", score=0.9),
            Entity(label="dog", score=0.8),
            Entity(label="bird", score=0.7),
        ],
        text='{"objects": ["cat", "dog", "bird"]}',
        meta={"model": "qwen3-vl", "output_mode": "detect", "auto_promote": True},
    )


@pytest.fixture
def generic_query_result():
    """VisionResult with text for a generic query."""
    return VisionResult(
        instances=[],
        text="The image shows a person walking a dog in a park.",
        prompt="What is happening in this image?",
        meta={"model": "qwen3-vl"},
    )


@pytest.fixture
def multi_image_result():
    """VisionResult for multi-image comparison."""
    return VisionResult(
        instances=[],
        text="Image 1 shows a cat, Image 2 shows a dog. Both are pets.",
        prompt="Compare these images.",
        meta={"model": "qwen3-vl", "num_images": 2},
    )


@pytest.fixture
def spatial_detections():
    """Detections from a spatial detector (GroundingDINO/DETR)."""
    instances = [
        Instance(
            bbox=(10, 20, 100, 150),
            score=0.95,
            label=0,
            label_name="cat",
        ),
        Instance(
            bbox=(200, 50, 300, 200),
            score=0.82,
            label=1,
            label_name="dog",
        ),
        Instance(
            bbox=(350, 100, 450, 250),
            score=0.70,
            label=2,
            label_name="car",
        ),
    ]
    return Detections(
        instances=instances,
        meta={"model": "grounding-dino"},
    )


@pytest.fixture
def vlm_entity_detections():
    """Detections from VLM containing entities."""
    entities = [
        Entity(label="cat", score=0.9, attributes={"color": "orange"}),
        Entity(label="dog", score=0.8, attributes={"breed": "labrador"}),
    ]
    return Detections(
        instances=[],
        entities=entities,
        meta={"model": "qwen3-vl", "output_mode": "detect"},
    )


def _make_ctx(providers: dict[str, dict[str, Any]] | None = None) -> ExecutionContext:
    """Helper to build an ExecutionContext with given providers."""
    return ExecutionContext(providers=providers or {}, device="cpu")


# ===========================================================================
# VLMDescribe node tests
# ===========================================================================


class TestVLMDescribe:
    """Tests for the VLMDescribe node."""

    def test_basic_describe(self, sample_image, describe_result):
        """Mock VLM returns a text description."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = describe_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMDescribe(using="qwen")
        result = node.run(ctx, image=sample_image)

        assert "description" in result
        dets = result["description"]
        assert isinstance(dets, Detections)
        mock_vlm.query.assert_called_once_with(
            sample_image,
            prompt="Describe this image in detail.",
        )

    def test_custom_output_name(self, sample_image, describe_result):
        """Output key respects the `out` parameter."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = describe_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMDescribe(using="qwen", out="my_description")
        result = node.run(ctx, image=sample_image)

        assert "my_description" in result
        assert "description" not in result

    def test_custom_prompt(self, sample_image, describe_result):
        """Custom prompt is forwarded to the VLM."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = describe_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMDescribe(using="qwen", prompt="What animals are here?")
        node.run(ctx, image=sample_image)

        mock_vlm.query.assert_called_once_with(
            sample_image,
            prompt="What animals are here?",
        )

    def test_kwargs_passthrough(self, sample_image, describe_result):
        """Extra kwargs are forwarded to the VLM."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = describe_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMDescribe(using="qwen", max_tokens=512, temperature=0.7)
        node.run(ctx, image=sample_image)

        mock_vlm.query.assert_called_once_with(
            sample_image,
            prompt="Describe this image in detail.",
            max_tokens=512,
            temperature=0.7,
        )

    def test_missing_provider_raises(self, sample_image):
        """KeyError when VLM provider is not registered."""
        ctx = _make_ctx({})
        node = VLMDescribe(using="nonexistent")
        with pytest.raises(KeyError, match="vlm"):
            node.run(ctx, image=sample_image)

    def test_records_metrics(self, sample_image, describe_result):
        """Node records latency and response length."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = describe_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMDescribe(using="qwen", name="describe_node")
        node.run(ctx, image=sample_image)

        metrics = ctx.get_metrics()
        assert "describe_node" in metrics
        assert "latency_ms" in metrics["describe_node"]
        assert metrics["describe_node"]["latency_ms"] >= 0
        assert metrics["describe_node"]["response_length"] > 0

    def test_empty_text_response(self, sample_image):
        """Handles VisionResult with no text gracefully."""
        empty_result = VisionResult(
            instances=[],
            text=None,
            meta={"model": "qwen3-vl"},
        )
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = empty_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMDescribe(using="qwen", name="desc_node")
        result = node.run(ctx, image=sample_image)

        assert "description" in result
        metrics = ctx.get_metrics()
        assert metrics["desc_node"]["response_length"] == 0.0

    def test_default_name(self):
        """Default node name is class name."""
        node = VLMDescribe(using="qwen")
        assert node.name == "VLMDescribe"

    def test_custom_name(self):
        """Custom node name is set correctly."""
        node = VLMDescribe(using="qwen", name="my_describer")
        assert node.name == "my_describer"

    def test_input_output_types(self):
        """Input and output types are declared correctly."""
        assert VLMDescribe.inputs == {"image": Image}
        assert VLMDescribe.outputs == {"description": Detections}


# ===========================================================================
# VLMDetect node tests
# ===========================================================================


class TestVLMDetect:
    """Tests for the VLMDetect node."""

    def test_basic_detect_entities(self, sample_image, detect_entities_result):
        """VLM returns entities, node converts to Detections."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = detect_entities_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMDetect(using="qwen")
        result = node.run(ctx, image=sample_image)

        assert "vlm_dets" in result
        dets = result["vlm_dets"]
        assert isinstance(dets, Detections)
        assert len(dets.entities) == 3
        mock_vlm.query.assert_called_once_with(
            sample_image,
            prompt="List all objects you can identify.",
            output_mode="detect",
            auto_promote=True,
        )

    def test_detect_with_auto_promote(self, sample_image, detect_promoted_result):
        """VLM returns promoted instances alongside entities."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = detect_promoted_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMDetect(using="qwen", auto_promote=True)
        result = node.run(ctx, image=sample_image)

        dets = result["vlm_dets"]
        assert isinstance(dets, Detections)
        assert len(dets.instances) == 2
        assert len(dets.entities) == 3

    def test_detect_without_auto_promote(self, sample_image, detect_entities_result):
        """auto_promote=False only returns entities."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = detect_entities_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMDetect(using="qwen", auto_promote=False)
        node.run(ctx, image=sample_image)

        mock_vlm.query.assert_called_once_with(
            sample_image,
            prompt="List all objects you can identify.",
            output_mode="detect",
            auto_promote=False,
        )

    def test_custom_output_name(self, sample_image, detect_entities_result):
        """Output key respects the `out` parameter."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = detect_entities_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMDetect(using="qwen", out="my_vlm_detections")
        result = node.run(ctx, image=sample_image)

        assert "my_vlm_detections" in result
        assert "vlm_dets" not in result

    def test_custom_prompt(self, sample_image, detect_entities_result):
        """Custom detection prompt is forwarded."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = detect_entities_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMDetect(using="qwen", prompt="Detect all animals.")
        node.run(ctx, image=sample_image)

        mock_vlm.query.assert_called_once_with(
            sample_image,
            prompt="Detect all animals.",
            output_mode="detect",
            auto_promote=True,
        )

    def test_kwargs_passthrough(self, sample_image, detect_entities_result):
        """Extra kwargs are forwarded to the VLM."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = detect_entities_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMDetect(using="qwen", max_tokens=256)
        node.run(ctx, image=sample_image)

        mock_vlm.query.assert_called_once_with(
            sample_image,
            prompt="List all objects you can identify.",
            output_mode="detect",
            auto_promote=True,
            max_tokens=256,
        )

    def test_missing_provider_raises(self, sample_image):
        """KeyError when VLM provider is not registered."""
        ctx = _make_ctx({})
        node = VLMDetect(using="nonexistent")
        with pytest.raises(KeyError, match="vlm"):
            node.run(ctx, image=sample_image)

    def test_records_metrics(self, sample_image, detect_entities_result):
        """Node records latency, entity count, and instance count."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = detect_entities_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMDetect(using="qwen", name="vlm_detect_node")
        node.run(ctx, image=sample_image)

        metrics = ctx.get_metrics()
        assert "vlm_detect_node" in metrics
        assert "latency_ms" in metrics["vlm_detect_node"]
        assert metrics["vlm_detect_node"]["num_entities"] == 3
        assert metrics["vlm_detect_node"]["num_instances"] == 0

    def test_records_metrics_with_instances(self, sample_image, detect_promoted_result):
        """Metrics include both entity and instance counts when promoted."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = detect_promoted_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMDetect(using="qwen", name="vlm_det")
        node.run(ctx, image=sample_image)

        metrics = ctx.get_metrics()
        assert metrics["vlm_det"]["num_entities"] == 3
        assert metrics["vlm_det"]["num_instances"] == 2

    def test_input_output_types(self):
        """Input and output types are declared correctly."""
        assert VLMDetect.inputs == {"image": Image}
        assert VLMDetect.outputs == {"detections": Detections}

    def test_default_name(self):
        """Default node name is class name."""
        node = VLMDetect(using="qwen")
        assert node.name == "VLMDetect"


# ===========================================================================
# VLMQuery node tests
# ===========================================================================


class TestVLMQuery:
    """Tests for the VLMQuery node."""

    def test_single_image_query(self, sample_image, generic_query_result):
        """Query with a single image."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = generic_query_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMQuery(using="qwen", prompt="What is happening?")
        result = node.run(ctx, image=sample_image)

        assert "vlm_result" in result
        dets = result["vlm_result"]
        assert isinstance(dets, Detections)
        # Single image passed directly (not as list)
        mock_vlm.query.assert_called_once_with(
            sample_image,
            prompt="What is happening?",
            output_mode=None,
        )

    def test_multi_image_query(self, sample_image, sample_image_2, multi_image_result):
        """Query with multiple images."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = multi_image_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMQuery(using="qwen", prompt="Compare these images.")
        result = node.run(ctx, image=sample_image, images=[sample_image_2])

        assert "vlm_result" in result
        # Multiple images passed as list
        called_args = mock_vlm.query.call_args
        assert isinstance(called_args[0][0], list)
        assert len(called_args[0][0]) == 2

    def test_three_image_query(self, sample_image, sample_image_2, sample_image_3, multi_image_result):
        """Query with three images."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = multi_image_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMQuery(using="qwen", prompt="Compare all three images.")
        node.run(
            ctx,
            image=sample_image,
            images=[sample_image_2, sample_image_3],
        )

        called_args = mock_vlm.query.call_args
        assert isinstance(called_args[0][0], list)
        assert len(called_args[0][0]) == 3

    def test_output_mode_detect(self, sample_image, detect_entities_result):
        """Query with detect output mode."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = detect_entities_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMQuery(
            using="qwen",
            prompt="List objects.",
            output_mode="detect",
        )
        result = node.run(ctx, image=sample_image)

        dets = result["vlm_result"]
        assert isinstance(dets, Detections)
        mock_vlm.query.assert_called_once_with(
            sample_image,
            prompt="List objects.",
            output_mode="detect",
        )

    def test_output_mode_json(self, sample_image, generic_query_result):
        """Query with json output mode."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = generic_query_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMQuery(
            using="qwen",
            prompt="Describe as JSON.",
            output_mode="json",
        )
        node.run(ctx, image=sample_image)

        mock_vlm.query.assert_called_once_with(
            sample_image,
            prompt="Describe as JSON.",
            output_mode="json",
        )

    def test_output_mode_classify(self, sample_image, generic_query_result):
        """Query with classify output mode."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = generic_query_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMQuery(
            using="qwen",
            prompt="What is this?",
            output_mode="classify",
        )
        node.run(ctx, image=sample_image)

        mock_vlm.query.assert_called_once_with(
            sample_image,
            prompt="What is this?",
            output_mode="classify",
        )

    def test_custom_output_name(self, sample_image, generic_query_result):
        """Output key respects the `out` parameter."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = generic_query_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMQuery(using="qwen", prompt="Hi", out="my_result")
        result = node.run(ctx, image=sample_image)

        assert "my_result" in result
        assert "vlm_result" not in result

    def test_kwargs_passthrough(self, sample_image, generic_query_result):
        """Extra kwargs are forwarded to the VLM."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = generic_query_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMQuery(
            using="qwen",
            prompt="Describe.",
            max_tokens=1024,
            temperature=0.5,
        )
        node.run(ctx, image=sample_image)

        mock_vlm.query.assert_called_once_with(
            sample_image,
            prompt="Describe.",
            output_mode=None,
            max_tokens=1024,
            temperature=0.5,
        )

    def test_missing_provider_raises(self, sample_image):
        """KeyError when VLM provider is not registered."""
        ctx = _make_ctx({})
        node = VLMQuery(using="nonexistent", prompt="Hello")
        with pytest.raises(KeyError, match="vlm"):
            node.run(ctx, image=sample_image)

    def test_records_metrics_single_image(self, sample_image, generic_query_result):
        """Records num_images=1 for single-image query."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = generic_query_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMQuery(using="qwen", prompt="Hi", name="query_node")
        node.run(ctx, image=sample_image)

        metrics = ctx.get_metrics()
        assert "query_node" in metrics
        assert metrics["query_node"]["num_images"] == 1.0

    def test_records_metrics_multi_image(self, sample_image, sample_image_2, multi_image_result):
        """Records num_images=2 for multi-image query."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = multi_image_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMQuery(using="qwen", prompt="Compare", name="query_multi")
        node.run(ctx, image=sample_image, images=[sample_image_2])

        metrics = ctx.get_metrics()
        assert metrics["query_multi"]["num_images"] == 2.0

    def test_input_output_types(self):
        """Input and output types are declared correctly."""
        assert VLMQuery.inputs == {"image": Image}
        assert VLMQuery.outputs == {"result": Detections}

    def test_default_name(self):
        """Default node name is class name."""
        node = VLMQuery(using="qwen", prompt="Hello")
        assert node.name == "VLMQuery"

    def test_none_images_treated_as_single(self, sample_image, generic_query_result):
        """images=None is equivalent to single-image query."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = generic_query_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMQuery(using="qwen", prompt="Hello")
        node.run(ctx, image=sample_image, images=None)

        # Single image, not a list
        mock_vlm.query.assert_called_once_with(
            sample_image,
            prompt="Hello",
            output_mode=None,
        )

    def test_empty_images_list_treated_as_single(self, sample_image, generic_query_result):
        """images=[] is equivalent to single-image query."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = generic_query_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})
        node = VLMQuery(using="qwen", prompt="Hello")
        node.run(ctx, image=sample_image, images=[])

        # Empty list treated as single image
        mock_vlm.query.assert_called_once_with(
            sample_image,
            prompt="Hello",
            output_mode=None,
        )


# ===========================================================================
# PromoteEntities node tests
# ===========================================================================


class TestPromoteEntities:
    """Tests for the PromoteEntities node."""

    def test_fuzzy_matching(self, vlm_entity_detections, spatial_detections):
        """Fuzzy matching promotes entities with label match."""
        ctx = _make_ctx({})
        node = PromoteEntities(match_strategy="label_fuzzy")
        result = node.run(
            ctx,
            entities=vlm_entity_detections,
            spatial=spatial_detections,
        )

        assert "promoted" in result
        promoted = result["promoted"]
        assert isinstance(promoted, Detections)
        # cat and dog should match; car has no matching entity
        assert len(promoted.instances) == 2
        # All entities should be promoted (entities list empty)
        assert len(promoted.entities) == 0

    def test_exact_matching(self, vlm_entity_detections, spatial_detections):
        """Exact matching promotes entities with exact label match."""
        ctx = _make_ctx({})
        node = PromoteEntities(match_strategy="label_exact")
        result = node.run(
            ctx,
            entities=vlm_entity_detections,
            spatial=spatial_detections,
        )

        promoted = result["promoted"]
        assert isinstance(promoted, Detections)
        # Exact match: cat→cat, dog→dog
        assert len(promoted.instances) == 2

    def test_no_matching_entities(self, spatial_detections):
        """When no entities match spatial instances, result is empty."""
        no_match_entities = Detections(
            instances=[],
            entities=[
                Entity(label="elephant", score=0.9),
                Entity(label="giraffe", score=0.8),
            ],
            meta={"model": "qwen3-vl"},
        )

        ctx = _make_ctx({})
        node = PromoteEntities(match_strategy="label_exact")
        result = node.run(
            ctx,
            entities=no_match_entities,
            spatial=spatial_detections,
        )

        promoted = result["promoted"]
        assert len(promoted.instances) == 0
        assert promoted.meta["promoted_count"] == 0

    def test_empty_entities(self, spatial_detections):
        """Empty entities input produces empty result."""
        empty_entities = Detections(
            instances=[],
            entities=[],
            meta={},
        )

        ctx = _make_ctx({})
        node = PromoteEntities()
        result = node.run(
            ctx,
            entities=empty_entities,
            spatial=spatial_detections,
        )

        promoted = result["promoted"]
        assert len(promoted.instances) == 0

    def test_empty_spatial(self, vlm_entity_detections):
        """Empty spatial input produces empty result (nothing to match)."""
        empty_spatial = Detections(
            instances=[],
            meta={},
        )

        ctx = _make_ctx({})
        node = PromoteEntities()
        result = node.run(
            ctx,
            entities=vlm_entity_detections,
            spatial=empty_spatial,
        )

        promoted = result["promoted"]
        assert len(promoted.instances) == 0

    def test_custom_output_name(self, vlm_entity_detections, spatial_detections):
        """Output key respects the `out` parameter."""
        ctx = _make_ctx({})
        node = PromoteEntities(out="my_promoted")
        result = node.run(
            ctx,
            entities=vlm_entity_detections,
            spatial=spatial_detections,
        )

        assert "my_promoted" in result
        assert "promoted" not in result

    def test_instance_ids_generated(self, vlm_entity_detections, spatial_detections):
        """Promoted detections have auto-generated instance IDs."""
        ctx = _make_ctx({})
        node = PromoteEntities()
        result = node.run(
            ctx,
            entities=vlm_entity_detections,
            spatial=spatial_detections,
        )

        promoted = result["promoted"]
        assert len(promoted.instance_ids) == len(promoted.instances)
        # IDs should be unique
        assert len(set(promoted.instance_ids)) == len(promoted.instance_ids)

    def test_meta_contains_strategy(self, vlm_entity_detections, spatial_detections):
        """Meta dict includes match strategy and counts."""
        ctx = _make_ctx({})
        node = PromoteEntities(match_strategy="label_fuzzy")
        result = node.run(
            ctx,
            entities=vlm_entity_detections,
            spatial=spatial_detections,
        )

        promoted = result["promoted"]
        assert promoted.meta["match_strategy"] == "label_fuzzy"
        assert "promoted_count" in promoted.meta
        assert promoted.meta["source_entity_count"] == 2
        assert promoted.meta["source_spatial_count"] == 3

    def test_records_metrics(self, vlm_entity_detections, spatial_detections):
        """Node records promotion metrics."""
        ctx = _make_ctx({})
        node = PromoteEntities(name="promo_node")
        node.run(
            ctx,
            entities=vlm_entity_detections,
            spatial=spatial_detections,
        )

        metrics = ctx.get_metrics()
        assert "promo_node" in metrics
        assert "latency_ms" in metrics["promo_node"]
        assert metrics["promo_node"]["num_entities_input"] == 2
        assert metrics["promo_node"]["num_spatial_input"] == 3
        assert metrics["promo_node"]["num_promoted"] >= 0

    def test_input_output_types(self):
        """Input and output types are declared correctly."""
        assert PromoteEntities.inputs == {
            "entities": Detections,
            "spatial": Detections,
        }
        assert PromoteEntities.outputs == {"detections": Detections}

    def test_default_name(self):
        """Default node name is class name."""
        node = PromoteEntities()
        assert node.name == "PromoteEntities"

    def test_entity_ids_empty_after_promotion(self, vlm_entity_detections, spatial_detections):
        """All entities are promoted, so entity_ids should be empty."""
        ctx = _make_ctx({})
        node = PromoteEntities()
        result = node.run(
            ctx,
            entities=vlm_entity_detections,
            spatial=spatial_detections,
        )

        promoted = result["promoted"]
        assert promoted.entity_ids == []
        assert promoted.entities == []


# ===========================================================================
# Integration tests
# ===========================================================================


class TestVLMNodesIntegration:
    """Integration tests combining VLM nodes."""

    def test_vlm_detect_then_promote_workflow(self, sample_image, spatial_detections):
        """VLMDetect → PromoteEntities workflow."""
        # VLM detect returns entities
        detect_result = VisionResult(
            instances=[],
            entities=[
                Entity(label="cat", score=0.9),
                Entity(label="dog", score=0.8),
            ],
            text='{"objects": ["cat", "dog"]}',
            meta={"model": "qwen3-vl"},
        )

        mock_vlm = MagicMock()
        mock_vlm.query.return_value = detect_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})

        # Step 1: VLM detect
        detect_node = VLMDetect(using="qwen", out="vlm_dets")
        detect_result = detect_node.run(ctx, image=sample_image)
        vlm_dets = detect_result["vlm_dets"]

        # Step 2: Promote entities with spatial
        promote_node = PromoteEntities(match_strategy="label_fuzzy", out="promoted")
        promote_result = promote_node.run(ctx, entities=vlm_dets, spatial=spatial_detections)

        promoted = promote_result["promoted"]
        assert isinstance(promoted, Detections)
        assert len(promoted.instances) == 2
        assert len(promoted.entities) == 0

    def test_all_node_imports(self):
        """All VLM nodes are importable from mata.nodes."""
        from mata.nodes import PromoteEntities as PE  # noqa: N817
        from mata.nodes import VLMDescribe as VD  # noqa: N814
        from mata.nodes import VLMDetect as VDet
        from mata.nodes import VLMQuery as VQ  # noqa: N814

        assert VD is VLMDescribe
        assert VDet is VLMDetect
        assert VQ is VLMQuery
        assert PE is PromoteEntities

    def test_vlm_describe_and_query_have_different_defaults(self):
        """VLMDescribe and VLMQuery have different default output names."""
        describe = VLMDescribe(using="qwen")
        query = VLMQuery(using="qwen", prompt="Hi")

        assert describe.output_name == "description"
        assert query.output_name == "vlm_result"

    def test_vlm_detect_auto_promote_flag(self, sample_image, detect_entities_result):
        """VLMDetect correctly passes auto_promote flag."""
        mock_vlm = MagicMock()
        mock_vlm.query.return_value = detect_entities_result

        ctx = _make_ctx({"vlm": {"qwen": mock_vlm}})

        # With auto_promote=True (default)
        node_true = VLMDetect(using="qwen", auto_promote=True)
        node_true.run(ctx, image=sample_image)
        assert mock_vlm.query.call_args[1]["auto_promote"] is True

        mock_vlm.reset_mock()

        # With auto_promote=False
        node_false = VLMDetect(using="qwen", auto_promote=False)
        node_false.run(ctx, image=sample_image)
        assert mock_vlm.query.call_args[1]["auto_promote"] is False

    def test_promote_entities_preserves_bbox(self, vlm_entity_detections, spatial_detections):
        """Promoted instances retain bounding boxes from spatial source."""
        ctx = _make_ctx({})
        node = PromoteEntities(match_strategy="label_exact")
        result = node.run(
            ctx,
            entities=vlm_entity_detections,
            spatial=spatial_detections,
        )

        promoted = result["promoted"]
        for inst in promoted.instances:
            assert inst.bbox is not None
