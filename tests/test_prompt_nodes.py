"""Unit tests for prompt-based segmentation nodes (Task 5.3).

Tests cover:
- PromptBoxes: box prompt conversion, instance_id alignment, SAM integration
- PromptPoints: point prompt handling, validation, foreground/background
- SegmentEverything: automatic segmentation mode
- Edge cases: empty detections, missing providers, ID alignment mismatch
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.artifacts.masks import Masks
from mata.core.graph.context import ExecutionContext
from mata.core.graph.node import Node
from mata.core.types import Instance
from mata.nodes.prompt_boxes import PromptBoxes
from mata.nodes.prompt_points import PromptPoints
from mata.nodes.segment_everything import SegmentEverything

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_instance(
    bbox: tuple | None = (50, 50, 200, 200),
    mask: Any | None = None,
    score: float = 0.9,
    label: int = 0,
    label_name: str = "object",
) -> Instance:
    """Helper to create an Instance."""
    return Instance(
        bbox=bbox,
        mask=mask,
        score=score,
        label=label,
        label_name=label_name,
    )


def _make_mask_instance(
    bbox: tuple | None = None,
    score: float = 0.9,
    label: int = 0,
    label_name: str = "object",
    mask_val: Any = None,
) -> Instance:
    """Helper to create an Instance with a binary mask."""
    import numpy as np

    mask = mask_val if mask_val is not None else np.ones((100, 100), dtype=bool)
    return Instance(
        bbox=bbox,
        mask=mask,
        score=score,
        label=label,
        label_name=label_name,
    )


def _make_detections(n: int = 3) -> Detections:
    """Create a Detections artifact with N instances."""
    instances = [
        _make_instance(
            bbox=(i * 100, i * 50, i * 100 + 200, i * 50 + 150),
            score=0.9 - i * 0.1,
            label=i,
            label_name=f"cls_{i}",
        )
        for i in range(n)
    ]
    return Detections(
        instances=instances,
        instance_ids=[f"det_{i:04d}" for i in range(n)],
        meta={"source": "test"},
    )


def _make_masks(n: int = 3) -> Masks:
    """Create a Masks artifact with N mask instances."""

    instances = [
        _make_mask_instance(
            score=0.9 - i * 0.1,
            label=i,
            label_name=f"cls_{i}",
        )
        for i in range(n)
    ]
    return Masks(
        instances=instances,
        instance_ids=[f"mask_{i:04d}" for i in range(n)],
        meta={},
    )


def _make_image() -> Image:
    """Create a simple test Image artifact."""
    import numpy as np

    data = np.zeros((480, 640, 3), dtype=np.uint8)
    return Image(data=data, width=640, height=480, color_space="RGB")


def _mock_segmenter(n_masks: int = 3) -> Mock:
    """Create a mock Segmenter provider that returns Masks."""
    segmenter = Mock()
    segmenter.segment = Mock(return_value=_make_masks(n_masks))
    return segmenter


def _make_context(segmenter: Mock | None = None, name: str = "sam") -> ExecutionContext:
    """Create an ExecutionContext with a mock segment provider."""
    seg = segmenter or _mock_segmenter()
    return ExecutionContext(
        providers={"segment": {name: seg}},
        device="cpu",
    )


# ===========================================================================
# PromptBoxes Tests
# ===========================================================================


class TestPromptBoxes:
    """Tests for PromptBoxes node."""

    # --- Construction ---

    def test_init_defaults(self):
        node = PromptBoxes(using="sam")
        assert node.provider_name == "sam"
        assert node.image_src == "image"
        assert node.dets_src == "dets"
        assert node.output_name == "masks"
        assert node.name == "PromptBoxes"

    def test_init_custom(self):
        node = PromptBoxes(
            using="sam3",
            image_src="input_img",
            dets_src="boxes",
            out="seg_masks",
            name="my_prompt_boxes",
            multimask_output=True,
        )
        assert node.provider_name == "sam3"
        assert node.image_src == "input_img"
        assert node.output_name == "seg_masks"
        assert node.name == "my_prompt_boxes"
        assert node.kwargs == {"multimask_output": True}

    # --- Type declarations ---

    def test_input_output_types(self):
        node = PromptBoxes(using="sam")
        assert node.inputs == {"image": Image, "detections": Detections}
        assert node.outputs == {"masks": Masks}
        assert node.required_inputs == {"image", "detections"}
        assert node.provided_outputs == {"masks"}

    def test_is_node_subclass(self):
        assert issubclass(PromptBoxes, Node)

    # --- run() ---

    def test_run_basic(self):
        """PromptBoxes calls segmenter with box_prompts and returns masks."""
        node = PromptBoxes(using="sam")
        segmenter = _mock_segmenter(n_masks=3)
        ctx = _make_context(segmenter)
        image = _make_image()
        dets = _make_detections(3)

        result = node.run(ctx, image=image, detections=dets)

        # Verify segmenter called with correct args
        segmenter.segment.assert_called_once()
        call_kwargs = segmenter.segment.call_args
        assert "box_prompts" in call_kwargs.kwargs or (len(call_kwargs.args) > 1)
        assert call_kwargs.kwargs.get("mode") == "boxes"

        # Check output
        assert node.output_name in result
        masks = result[node.output_name]
        assert isinstance(masks, Masks)

    def test_box_prompt_extraction(self):
        """Bboxes from detections are extracted as box prompts."""
        node = PromptBoxes(using="sam")
        segmenter = _mock_segmenter(n_masks=2)
        ctx = _make_context(segmenter)
        image = _make_image()

        instances = [
            _make_instance(bbox=(10, 20, 100, 200), label_name="cat"),
            _make_instance(bbox=(50, 60, 300, 400), label_name="dog"),
        ]
        dets = Detections(instances=instances, meta={})

        node.run(ctx, image=image, detections=dets)

        call_kwargs = segmenter.segment.call_args.kwargs
        assert call_kwargs["box_prompts"] == [(10, 20, 100, 200), (50, 60, 300, 400)]

    def test_instance_id_alignment(self):
        """Output mask IDs match detection IDs."""
        node = PromptBoxes(using="sam")
        segmenter = _mock_segmenter(n_masks=3)
        ctx = _make_context(segmenter)
        image = _make_image()
        dets = _make_detections(3)

        result = node.run(ctx, image=image, detections=dets)
        masks = result["masks"]

        assert masks.instance_ids == ["det_0000", "det_0001", "det_0002"]

    def test_instance_id_alignment_fewer_masks(self):
        """When segmenter returns fewer masks than prompts, truncate IDs."""
        node = PromptBoxes(using="sam")
        segmenter = _mock_segmenter(n_masks=2)  # 2 masks for 3 detections
        ctx = _make_context(segmenter)
        image = _make_image()
        dets = _make_detections(3)

        result = node.run(ctx, image=image, detections=dets)
        masks = result["masks"]

        assert len(masks.instance_ids) == 2
        assert masks.instance_ids == ["det_0000", "det_0001"]

    def test_instance_id_alignment_more_masks(self):
        """When segmenter returns more masks and keep_best=False, extra keep auto-generated IDs."""
        node = PromptBoxes(using="sam", keep_best=False)
        segmenter = _mock_segmenter(n_masks=5)  # 5 masks for 3 detections
        ctx = _make_context(segmenter)
        image = _make_image()
        dets = _make_detections(3)

        result = node.run(ctx, image=image, detections=dets)
        masks = result["masks"]

        assert len(masks.instance_ids) == 5
        # First 3 aligned, last 2 auto-generated
        assert masks.instance_ids[:3] == ["det_0000", "det_0001", "det_0002"]

    def test_skips_instances_without_bbox(self):
        """Only instances with bbox are used as prompts."""
        import numpy as np

        node = PromptBoxes(using="sam")
        segmenter = _mock_segmenter(n_masks=1)
        ctx = _make_context(segmenter)
        image = _make_image()

        # Instance without bbox but with mask (valid: has at least mask)
        mask_only = Instance(
            bbox=None,
            mask=np.ones((100, 100), dtype=bool),
            score=0.9,
            label=0,
            label_name="no_box",
        )
        has_box = _make_instance(bbox=(10, 20, 100, 200), label_name="has_box")
        instances = [mask_only, has_box]
        dets = Detections(
            instances=instances,
            instance_ids=["id_nobox", "id_hasbox"],
            meta={},
        )

        result = node.run(ctx, image=image, detections=dets)

        call_kwargs = segmenter.segment.call_args.kwargs
        assert call_kwargs["box_prompts"] == [(10, 20, 100, 200)]

        # Aligned ID should be from the instance WITH a bbox
        masks = result["masks"]
        assert masks.instance_ids == ["id_hasbox"]

    def test_empty_detections_raises(self):
        """Raise ValueError when no bounding boxes are available."""
        import numpy as np

        node = PromptBoxes(using="sam")
        ctx = _make_context()
        image = _make_image()

        # Detections with only mask, no bbox
        mask_only = Instance(
            bbox=None,
            mask=np.ones((100, 100), dtype=bool),
            score=0.9,
            label=0,
            label_name="mask_only",
        )
        dets = Detections(instances=[mask_only], meta={})

        with pytest.raises(ValueError, match="no bounding boxes"):
            node.run(ctx, image=image, detections=dets)

    def test_kwargs_passthrough(self):
        """Extra kwargs are forwarded to segmenter.segment()."""
        node = PromptBoxes(using="sam", multimask_output=True, threshold=0.5)
        segmenter = _mock_segmenter(n_masks=1)
        ctx = _make_context(segmenter)
        image = _make_image()
        dets = _make_detections(1)

        node.run(ctx, image=image, detections=dets)

        call_kwargs = segmenter.segment.call_args.kwargs
        assert call_kwargs["multimask_output"] is True
        assert call_kwargs["threshold"] == 0.5

    def test_missing_provider_raises(self):
        """KeyError when segmenter provider is not registered."""
        node = PromptBoxes(using="nonexistent")
        ctx = ExecutionContext(providers={}, device="cpu")
        image = _make_image()
        dets = _make_detections(1)

        with pytest.raises(KeyError, match="segment"):
            node.run(ctx, image=image, detections=dets)

    def test_metrics_recorded(self):
        """Execution metrics are recorded in context."""
        node = PromptBoxes(using="sam")
        segmenter = _mock_segmenter(n_masks=3)
        ctx = _make_context(segmenter)
        image = _make_image()
        dets = _make_detections(3)

        node.run(ctx, image=image, detections=dets)

        metrics = ctx.get_metrics()
        assert "PromptBoxes" in metrics
        assert metrics["PromptBoxes"]["num_box_prompts"] == 3.0
        assert metrics["PromptBoxes"]["num_masks"] == 3.0

    def test_output_meta_includes_prompt_type(self):
        """Output masks meta has prompt_type='boxes'."""
        node = PromptBoxes(using="sam")
        segmenter = _mock_segmenter(n_masks=2)
        ctx = _make_context(segmenter)
        image = _make_image()
        dets = _make_detections(2)

        result = node.run(ctx, image=image, detections=dets)
        masks = result["masks"]
        assert masks.meta.get("prompt_type") == "boxes"

    def test_custom_output_name(self):
        """Output key matches the configured 'out' parameter."""
        node = PromptBoxes(using="sam", out="seg_boxes")
        segmenter = _mock_segmenter(n_masks=1)
        ctx = _make_context(segmenter)
        image = _make_image()
        dets = _make_detections(1)

        result = node.run(ctx, image=image, detections=dets)
        assert "seg_boxes" in result

    def test_repr(self):
        node = PromptBoxes(using="sam", out="m")
        r = repr(node)
        assert "PromptBoxes" in r
        assert "sam" in r
        assert "'m'" in r

    # --- keep_best deduplication ---

    def test_keep_best_defaults(self):
        """keep_best=True and group_size=3 by default."""
        node = PromptBoxes(using="sam")
        assert node.keep_best is True
        assert node.group_size == 3

    def test_keep_best_false(self):
        """keep_best can be disabled."""
        node = PromptBoxes(using="sam", keep_best=False)
        assert node.keep_best is False

    def test_keep_best_deduplicates_multimask(self):
        """When segmenter returns 3× masks (SAM default), keep_best reduces to 1 per prompt."""

        node = PromptBoxes(using="sam")  # keep_best=True by default
        # 2 detections → segmenter returns 6 masks (3 per box)
        mask_instances = [
            _make_mask_instance(score=0.7, label=0),
            _make_mask_instance(score=0.95, label=0),  # best of group 1
            _make_mask_instance(score=0.8, label=0),
            _make_mask_instance(score=0.6, label=0),
            _make_mask_instance(score=0.5, label=0),
            _make_mask_instance(score=0.85, label=0),  # best of group 2
        ]
        segmenter_masks = Masks(
            instances=mask_instances,
            instance_ids=[f"m_{i}" for i in range(6)],
            meta={},
        )
        segmenter = Mock()
        segmenter.segment = Mock(return_value=segmenter_masks)
        ctx = _make_context(segmenter)
        image = _make_image()
        dets = _make_detections(2)

        result = node.run(ctx, image=image, detections=dets)
        masks = result["masks"]

        # Should have 2 masks (best from each group of 3)
        assert len(masks.instances) == 2
        assert masks.instances[0].score == 0.95
        assert masks.instances[1].score == 0.85
        assert masks.meta.get("deduplicated") is True
        assert masks.meta.get("original_count") == 6

    def test_keep_best_disabled_returns_all(self):
        """When keep_best=False, all mask candidates are returned."""
        node = PromptBoxes(using="sam", keep_best=False)
        segmenter = _mock_segmenter(n_masks=6)  # 6 masks for 2 detections
        ctx = _make_context(segmenter)
        image = _make_image()
        dets = _make_detections(2)

        result = node.run(ctx, image=image, detections=dets)
        masks = result["masks"]

        # All 6 masks should be present
        assert len(masks.instances) == 6

    def test_keep_best_no_dedup_when_equal(self):
        """When mask count equals prompt count, no dedup is needed."""
        node = PromptBoxes(using="sam")  # keep_best=True
        segmenter = _mock_segmenter(n_masks=3)  # 3 masks for 3 detections
        ctx = _make_context(segmenter)
        image = _make_image()
        dets = _make_detections(3)

        result = node.run(ctx, image=image, detections=dets)
        masks = result["masks"]

        assert len(masks.instances) == 3
        # No dedup metadata should be set
        assert masks.meta.get("deduplicated") is None

    def test_keep_best_auto_group_size(self):
        """When group_size=None, auto-detect from output/input ratio."""

        node = PromptBoxes(using="sam", group_size=None)
        # 2 prompts → 8 masks (group_size auto-detected as 4)
        mask_instances = [
            _make_mask_instance(score=0.5),
            _make_mask_instance(score=0.6),
            _make_mask_instance(score=0.9),  # best of group 1
            _make_mask_instance(score=0.7),
            _make_mask_instance(score=0.3),
            _make_mask_instance(score=0.4),
            _make_mask_instance(score=0.8),  # best of group 2
            _make_mask_instance(score=0.2),
        ]
        segmenter_masks = Masks(
            instances=mask_instances,
            instance_ids=[f"m_{i}" for i in range(8)],
            meta={},
        )
        segmenter = Mock()
        segmenter.segment = Mock(return_value=segmenter_masks)
        ctx = _make_context(segmenter)
        image = _make_image()
        dets = _make_detections(2)

        result = node.run(ctx, image=image, detections=dets)
        masks = result["masks"]

        assert len(masks.instances) == 2
        assert masks.instances[0].score == 0.9
        assert masks.instances[1].score == 0.8

    def test_keep_best_records_dedup_metric(self):
        """Dedup records raw_masks_before_dedup metric."""
        node = PromptBoxes(using="sam")
        segmenter = _mock_segmenter(n_masks=6)  # 6 masks for 2 dets
        ctx = _make_context(segmenter)
        image = _make_image()
        dets = _make_detections(2)

        node.run(ctx, image=image, detections=dets)

        metrics = ctx.get_metrics()
        assert metrics["PromptBoxes"]["raw_masks_before_dedup"] == 6.0
        assert metrics["PromptBoxes"]["num_masks"] == 2.0

    def test_keep_best_repr_shows_keep_best_false(self):
        """repr shows keep_best when disabled."""
        node = PromptBoxes(using="sam", keep_best=False, out="m")
        r = repr(node)
        assert "keep_best=False" in r

    def test_keep_best_repr_hides_when_default(self):
        """repr omits keep_best when True (default)."""
        node = PromptBoxes(using="sam", out="m")
        r = repr(node)
        assert "keep_best" not in r


# ===========================================================================
# PromptPoints Tests
# ===========================================================================


class TestPromptPoints:
    """Tests for PromptPoints node."""

    # --- Construction ---

    def test_init_defaults(self):
        node = PromptPoints(using="sam", points=[(100, 200, 1)])
        assert node.provider_name == "sam"
        assert node.points == [(100, 200, 1)]
        assert node.image_src == "image"
        assert node.output_name == "masks"
        assert node.name == "PromptPoints"

    def test_init_empty_points(self):
        """Node can be created with no points (will fail at run time)."""
        node = PromptPoints(using="sam")
        assert node.points == []

    def test_init_custom(self):
        node = PromptPoints(
            using="sam3",
            points=[(10, 20, 1), (30, 40, 0)],
            image_src="img",
            out="pt_masks",
            name="my_points",
            multimask_output=False,
        )
        assert node.provider_name == "sam3"
        assert len(node.points) == 2
        assert node.output_name == "pt_masks"
        assert node.kwargs == {"multimask_output": False}

    # --- Type declarations ---

    def test_input_output_types(self):
        node = PromptPoints(using="sam")
        assert node.inputs == {"image": Image}
        assert node.outputs == {"masks": Masks}

    def test_is_node_subclass(self):
        assert issubclass(PromptPoints, Node)

    # --- Validation ---

    def test_invalid_point_format_raises(self):
        """Points must be (x, y, label) tuples."""
        with pytest.raises(ValueError, match="must be a \\(x, y, label\\) tuple"):
            PromptPoints(using="sam", points=[(10, 20)])  # Missing label

    def test_invalid_point_label_raises(self):
        """Point label must be 0 or 1."""
        with pytest.raises(ValueError, match="must be 0 .* or 1"):
            PromptPoints(using="sam", points=[(10, 20, 2)])

    def test_valid_points_accepted(self):
        """Various valid point formats don't raise."""
        node = PromptPoints(
            using="sam",
            points=[(0, 0, 0), (100, 200, 1), (50, 50, 1)],
        )
        assert len(node.points) == 3

    # --- run() ---

    def test_run_basic(self):
        """PromptPoints calls segmenter with point_prompts."""
        points = [(100, 200, 1), (10, 10, 0)]
        node = PromptPoints(using="sam", points=points)
        segmenter = _mock_segmenter(n_masks=1)
        ctx = _make_context(segmenter)
        image = _make_image()

        result = node.run(ctx, image=image)

        segmenter.segment.assert_called_once()
        call_kwargs = segmenter.segment.call_args.kwargs
        assert call_kwargs["point_prompts"] == points
        assert call_kwargs["mode"] == "points"

        assert node.output_name in result
        assert isinstance(result[node.output_name], Masks)

    def test_run_no_points_raises(self):
        """Raise ValueError when no points configured."""
        node = PromptPoints(using="sam")
        ctx = _make_context()
        image = _make_image()

        with pytest.raises(ValueError, match="no point prompts configured"):
            node.run(ctx, image=image)

    def test_kwargs_passthrough(self):
        """Extra kwargs forwarded to segmenter."""
        node = PromptPoints(
            using="sam",
            points=[(50, 50, 1)],
            multimask_output=True,
        )
        segmenter = _mock_segmenter(n_masks=1)
        ctx = _make_context(segmenter)
        image = _make_image()

        node.run(ctx, image=image)

        call_kwargs = segmenter.segment.call_args.kwargs
        assert call_kwargs["multimask_output"] is True

    def test_missing_provider_raises(self):
        """KeyError when provider not found."""
        node = PromptPoints(using="nonexistent", points=[(50, 50, 1)])
        ctx = ExecutionContext(providers={}, device="cpu")
        image = _make_image()

        with pytest.raises(KeyError, match="segment"):
            node.run(ctx, image=image)

    def test_metrics_recorded(self):
        """Execution metrics are recorded."""
        points = [(10, 20, 1), (30, 40, 0), (50, 60, 1)]
        node = PromptPoints(using="sam", points=points)
        segmenter = _mock_segmenter(n_masks=2)
        ctx = _make_context(segmenter)
        image = _make_image()

        node.run(ctx, image=image)

        metrics = ctx.get_metrics()
        assert "PromptPoints" in metrics
        assert metrics["PromptPoints"]["num_point_prompts"] == 3.0
        assert metrics["PromptPoints"]["fg_points"] == 2.0
        assert metrics["PromptPoints"]["bg_points"] == 1.0
        assert metrics["PromptPoints"]["num_masks"] == 2.0

    def test_output_meta_includes_prompt_type(self):
        """Output masks meta has prompt_type='points'."""
        node = PromptPoints(using="sam", points=[(50, 50, 1)])
        segmenter = _mock_segmenter(n_masks=1)
        ctx = _make_context(segmenter)
        image = _make_image()

        result = node.run(ctx, image=image)
        masks = result["masks"]
        assert masks.meta.get("prompt_type") == "points"
        assert masks.meta.get("num_points") == 1

    def test_custom_output_name(self):
        """Output key matches configured 'out'."""
        node = PromptPoints(using="sam", points=[(50, 50, 1)], out="pt_out")
        segmenter = _mock_segmenter(n_masks=1)
        ctx = _make_context(segmenter)
        image = _make_image()

        result = node.run(ctx, image=image)
        assert "pt_out" in result

    def test_repr(self):
        node = PromptPoints(using="sam", points=[(1, 2, 1), (3, 4, 0)], out="m")
        r = repr(node)
        assert "PromptPoints" in r
        assert "sam" in r
        assert "2" in r  # number of points


# ===========================================================================
# SegmentEverything Tests
# ===========================================================================


class TestSegmentEverything:
    """Tests for SegmentEverything node."""

    # --- Construction ---

    def test_init_defaults(self):
        node = SegmentEverything(using="sam")
        assert node.provider_name == "sam"
        assert node.image_src == "image"
        assert node.output_name == "masks"
        assert node.name == "SegmentEverything"

    def test_init_custom(self):
        node = SegmentEverything(
            using="sam3",
            image_src="input",
            out="all_masks",
            name="auto_seg",
            points_per_side=32,
        )
        assert node.provider_name == "sam3"
        assert node.output_name == "all_masks"
        assert node.name == "auto_seg"
        assert node.kwargs == {"points_per_side": 32}

    # --- Type declarations ---

    def test_input_output_types(self):
        node = SegmentEverything(using="sam")
        assert node.inputs == {"image": Image}
        assert node.outputs == {"masks": Masks}

    def test_is_node_subclass(self):
        assert issubclass(SegmentEverything, Node)

    # --- run() ---

    def test_run_basic(self):
        """Calls segmenter with mode='everything'."""
        node = SegmentEverything(using="sam")
        segmenter = _mock_segmenter(n_masks=10)
        ctx = _make_context(segmenter)
        image = _make_image()

        result = node.run(ctx, image=image)

        segmenter.segment.assert_called_once()
        call_kwargs = segmenter.segment.call_args.kwargs
        assert call_kwargs["mode"] == "everything"

        assert node.output_name in result
        masks = result[node.output_name]
        assert isinstance(masks, Masks)
        assert len(masks.instances) == 10

    def test_kwargs_passthrough(self):
        """Extra kwargs forwarded to segmenter."""
        node = SegmentEverything(
            using="sam",
            points_per_side=64,
            pred_iou_thresh=0.88,
        )
        segmenter = _mock_segmenter(n_masks=5)
        ctx = _make_context(segmenter)
        image = _make_image()

        node.run(ctx, image=image)

        call_kwargs = segmenter.segment.call_args.kwargs
        assert call_kwargs["points_per_side"] == 64
        assert call_kwargs["pred_iou_thresh"] == 0.88

    def test_missing_provider_raises(self):
        """KeyError when provider not found."""
        node = SegmentEverything(using="nonexistent")
        ctx = ExecutionContext(providers={}, device="cpu")
        image = _make_image()

        with pytest.raises(KeyError, match="segment"):
            node.run(ctx, image=image)

    def test_metrics_recorded(self):
        """Execution metrics are recorded."""
        node = SegmentEverything(using="sam")
        segmenter = _mock_segmenter(n_masks=7)
        ctx = _make_context(segmenter)
        image = _make_image()

        node.run(ctx, image=image)

        metrics = ctx.get_metrics()
        assert "SegmentEverything" in metrics
        assert metrics["SegmentEverything"]["num_masks"] == 7.0

    def test_output_meta_includes_prompt_type(self):
        """Output masks meta has prompt_type='everything'."""
        node = SegmentEverything(using="sam")
        segmenter = _mock_segmenter(n_masks=3)
        ctx = _make_context(segmenter)
        image = _make_image()

        result = node.run(ctx, image=image)
        masks = result["masks"]
        assert masks.meta.get("prompt_type") == "everything"

    def test_custom_output_name(self):
        """Output key matches configured 'out'."""
        node = SegmentEverything(using="sam", out="auto_masks")
        segmenter = _mock_segmenter(n_masks=2)
        ctx = _make_context(segmenter)
        image = _make_image()

        result = node.run(ctx, image=image)
        assert "auto_masks" in result

    def test_zero_masks_returned(self):
        """Handles segmenter returning zero masks gracefully."""
        node = SegmentEverything(using="sam")
        segmenter = Mock()
        # Return empty Masks — however Masks validates ≥ 1, so mock it
        empty_masks = Mock(spec=Masks)
        empty_masks.instances = []
        empty_masks.instance_ids = []
        empty_masks.meta = {}
        segmenter.segment = Mock(return_value=empty_masks)

        ctx = _make_context(segmenter)
        image = _make_image()

        # The node itself doesn't raise — it trusts the segmenter output.
        # The Masks constructor may raise if empty, so we test with the mock.
        result = node.run(ctx, image=image)
        assert "masks" in result

    def test_repr(self):
        node = SegmentEverything(using="sam", out="m")
        r = repr(node)
        assert "SegmentEverything" in r
        assert "sam" in r
        assert "'m'" in r


# ===========================================================================
# Integration-like tests (cross-node patterns)
# ===========================================================================


class TestPromptNodeIntegration:
    """Integration-like tests combining prompt nodes."""

    def test_prompt_boxes_with_real_artifacts(self):
        """Full flow: Detections → PromptBoxes → Masks with real artifacts."""
        import numpy as np

        # Build real detections
        instances = [
            Instance(bbox=(10, 20, 100, 150), mask=None, score=0.95, label=0, label_name="cat"),
            Instance(bbox=(200, 50, 400, 300), mask=None, score=0.85, label=1, label_name="dog"),
        ]
        dets = Detections(
            instances=instances,
            instance_ids=["cat_001", "dog_002"],
            meta={"model": "detr"},
        )

        # Mock segmenter that returns real Masks
        mask_instances = [
            Instance(
                bbox=None,
                mask=np.ones((480, 640), dtype=bool),
                score=0.92,
                label=0,
                label_name="cat",
            ),
            Instance(
                bbox=None,
                mask=np.ones((480, 640), dtype=bool),
                score=0.88,
                label=1,
                label_name="dog",
            ),
        ]
        real_masks = Masks(
            instances=mask_instances,
            instance_ids=["tmp_0", "tmp_1"],
            meta={"model": "sam"},
        )

        segmenter = Mock()
        segmenter.segment = Mock(return_value=real_masks)

        ctx = _make_context(segmenter)
        image = _make_image()

        node = PromptBoxes(using="sam")
        result = node.run(ctx, image=image, detections=dets)
        masks = result["masks"]

        # IDs should match detections
        assert masks.instance_ids == ["cat_001", "dog_002"]
        assert len(masks.instances) == 2
        assert masks.meta["prompt_type"] == "boxes"

    def test_prompt_points_foreground_and_background(self):
        """Mixed fg/bg points are handled correctly."""
        fg_points = [(100, 200, 1), (150, 250, 1)]
        bg_points = [(0, 0, 0), (639, 0, 0)]
        all_points = fg_points + bg_points

        node = PromptPoints(using="sam", points=all_points)
        segmenter = _mock_segmenter(n_masks=1)
        ctx = _make_context(segmenter)
        image = _make_image()

        result = node.run(ctx, image=image)
        result["masks"]

        # All 4 points should have been sent to the segmenter
        call_kwargs = segmenter.segment.call_args.kwargs
        assert len(call_kwargs["point_prompts"]) == 4

        metrics = ctx.get_metrics()
        assert metrics["PromptPoints"]["fg_points"] == 2.0
        assert metrics["PromptPoints"]["bg_points"] == 2.0

    def test_segment_everything_with_many_masks(self):
        """SegmentEverything handles large number of masks."""
        node = SegmentEverything(using="sam", points_per_side=64)
        segmenter = _mock_segmenter(n_masks=50)
        ctx = _make_context(segmenter)
        image = _make_image()

        result = node.run(ctx, image=image)
        masks = result["masks"]

        assert len(masks.instances) == 50
        assert len(masks.instance_ids) == 50
        assert masks.meta["prompt_type"] == "everything"

    def test_different_provider_names(self):
        """Nodes can target different segmenter providers."""
        seg_sam = _mock_segmenter(n_masks=2)
        seg_sam3 = _mock_segmenter(n_masks=5)

        ctx = ExecutionContext(
            providers={"segment": {"sam": seg_sam, "sam3": seg_sam3}},
            device="cpu",
        )
        image = _make_image()

        node_boxes = PromptBoxes(using="sam")
        node_everything = SegmentEverything(using="sam3")

        dets = _make_detections(2)
        result1 = node_boxes.run(ctx, image=image, detections=dets)
        result2 = node_everything.run(ctx, image=image)

        seg_sam.segment.assert_called_once()
        seg_sam3.segment.assert_called_once()

        assert len(result1["masks"].instances) == 2
        assert len(result2["masks"].instances) == 5
