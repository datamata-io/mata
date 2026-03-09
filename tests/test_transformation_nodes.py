"""Unit tests for Task 5.2: Data Transformation Nodes.

Tests cover:
- Filter node: score filtering, label inclusion/exclusion, fuzzy matching, edge cases
- TopK node: ranking, truncation, validation, edge cases
- ExtractROIs node: cropping, padding, empty inputs, instance_id preservation
- ExpandBoxes node: bbox from mask, alignment by instance_id, fallback, edge cases
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image as PILImage

from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.artifacts.masks import Masks
from mata.core.artifacts.rois import ROIs
from mata.core.graph.context import ExecutionContext
from mata.core.types import Instance
from mata.nodes.expand_boxes import ExpandBoxes, _bbox_from_binary_mask
from mata.nodes.filter import Filter
from mata.nodes.roi import ExtractROIs
from mata.nodes.topk import TopK

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx() -> ExecutionContext:
    """Minimal execution context for testing."""
    return ExecutionContext(providers={}, device="cpu")


def _make_instance(
    label_name: str,
    score: float,
    bbox=(10.0, 20.0, 100.0, 200.0),
    mask=None,
    label: int = 0,
) -> Instance:
    """Helper to build an Instance quickly."""
    return Instance(
        bbox=bbox,
        score=score,
        label=label,
        label_name=label_name,
        mask=mask,
    )


def _make_detections(instances_data: list, ids: list | None = None) -> Detections:
    """Build Detections from a list of (label_name, score) tuples."""
    instances = [_make_instance(name, score) for name, score in instances_data]
    kwargs = {"instances": instances}
    if ids:
        kwargs["instance_ids"] = ids
    return Detections(**kwargs)


@pytest.fixture
def sample_detections() -> Detections:
    """Five detections with varied labels and scores."""
    return _make_detections(
        [
            ("cat", 0.95),
            ("dog", 0.85),
            ("cat", 0.60),
            ("bird", 0.50),
            ("dog", 0.30),
        ],
        ids=["inst_0000", "inst_0001", "inst_0002", "inst_0003", "inst_0004"],
    )


@pytest.fixture
def sample_image() -> Image:
    """100×100 RGB test image."""
    pil = PILImage.new("RGB", (100, 100), color=(128, 128, 128))
    return Image.from_pil(pil)


# ===================================================================
# Filter Node
# ===================================================================


class TestFilter:
    """Tests for the Filter node."""

    def test_filter_by_score(self, ctx: ExecutionContext, sample_detections: Detections):
        node = Filter(score_gt=0.5)
        result = node.run(ctx, detections=sample_detections)

        filtered = result["filtered"]
        assert isinstance(filtered, Detections)
        # filter_by_score uses >= so 0.5 threshold keeps 0.95, 0.85, 0.60, 0.50 (4 instances)
        assert len(filtered.instances) == 4
        assert all(inst.score >= 0.5 for inst in filtered.instances)

    def test_filter_by_label_in(self, ctx: ExecutionContext, sample_detections: Detections):
        node = Filter(label_in=["cat"])
        result = node.run(ctx, detections=sample_detections)

        filtered = result["filtered"]
        assert len(filtered.instances) == 2
        assert all(inst.label_name == "cat" for inst in filtered.instances)

    def test_filter_by_label_not_in(self, ctx: ExecutionContext, sample_detections: Detections):
        node = Filter(label_not_in=["dog"])
        result = node.run(ctx, detections=sample_detections)

        filtered = result["filtered"]
        assert len(filtered.instances) == 3
        assert all(inst.label_name != "dog" for inst in filtered.instances)

    def test_filter_combined(self, ctx: ExecutionContext, sample_detections: Detections):
        node = Filter(score_gt=0.5, label_in=["cat", "dog"])
        result = node.run(ctx, detections=sample_detections)

        filtered = result["filtered"]
        # score >= 0.5: 0.95 cat, 0.85 dog, 0.60 cat  → then label_in cat/dog keeps all 3
        assert len(filtered.instances) == 3

    def test_filter_combined_score_and_exclude(self, ctx: ExecutionContext, sample_detections: Detections):
        node = Filter(score_gt=0.5, label_not_in=["cat"])
        result = node.run(ctx, detections=sample_detections)

        filtered = result["filtered"]
        # score >= 0.5: cat 0.95, dog 0.85, cat 0.60, bird 0.50 → exclude cat → dog 0.85, bird 0.50
        assert len(filtered.instances) == 2
        labels = [inst.label_name for inst in filtered.instances]
        assert "cat" not in labels

    def test_filter_empty_input(self, ctx: ExecutionContext):
        node = Filter(score_gt=0.5)
        empty = Detections(instances=[], instance_ids=[])
        result = node.run(ctx, detections=empty)

        filtered = result["filtered"]
        assert len(filtered.instances) == 0

    def test_filter_preserves_instance_ids(self, ctx: ExecutionContext, sample_detections: Detections):
        node = Filter(label_in=["bird"])
        result = node.run(ctx, detections=sample_detections)

        filtered = result["filtered"]
        assert filtered.instance_ids == ["inst_0003"]

    def test_filter_custom_output_name(self, ctx: ExecutionContext, sample_detections: Detections):
        node = Filter(out="my_output", score_gt=0.9)
        result = node.run(ctx, detections=sample_detections)

        assert "my_output" in result
        assert len(result["my_output"].instances) == 1

    def test_filter_no_criteria_passthrough(self, ctx: ExecutionContext, sample_detections: Detections):
        """No filter criteria → returns all detections."""
        node = Filter()
        result = node.run(ctx, detections=sample_detections)

        filtered = result["filtered"]
        assert len(filtered.instances) == len(sample_detections.instances)

    def test_filter_fuzzy_label_match(self, ctx: ExecutionContext):
        dets = _make_detections([("Cats", 0.9), ("Dogs", 0.8), ("bird", 0.7)])
        node = Filter(label_in=["cat"], fuzzy=True)
        result = node.run(ctx, detections=dets)

        filtered = result["filtered"]
        assert len(filtered.instances) == 1
        assert filtered.instances[0].label_name == "Cats"

    def test_filter_fuzzy_exclude(self, ctx: ExecutionContext):
        dets = _make_detections([("Cats", 0.9), ("Dogs", 0.8), ("bird", 0.7)])
        node = Filter(label_not_in=["cat"], fuzzy=True)
        result = node.run(ctx, detections=dets)

        filtered = result["filtered"]
        assert len(filtered.instances) == 2
        labels = [inst.label_name for inst in filtered.instances]
        assert "Cats" not in labels

    def test_filter_records_metrics(self, ctx: ExecutionContext, sample_detections: Detections):
        node = Filter(score_gt=0.8, name="my_filter")
        node.run(ctx, detections=sample_detections)

        metrics = ctx.get_metrics()
        assert "my_filter" in metrics
        assert metrics["my_filter"]["input_count"] == 5
        assert metrics["my_filter"]["output_count"] == 2

    def test_filter_label_not_in_preserves_none_label(self, ctx: ExecutionContext):
        """Instances with label_name=None should NOT be excluded."""
        inst_no_label = Instance(bbox=(0, 0, 10, 10), score=0.9, label=0, label_name=None)
        inst_dog = _make_instance("dog", 0.8)
        dets = Detections(instances=[inst_no_label, inst_dog])
        node = Filter(label_not_in=["dog"])
        result = node.run(ctx, detections=dets)

        filtered = result["filtered"]
        assert len(filtered.instances) == 1
        assert filtered.instances[0].label_name is None

    def test_filter_immutability(self, ctx: ExecutionContext, sample_detections: Detections):
        """Filtering must not mutate the original Detections."""
        original_count = len(sample_detections.instances)
        node = Filter(score_gt=0.9)
        node.run(ctx, detections=sample_detections)
        assert len(sample_detections.instances) == original_count

    def test_filter_repr(self):
        node = Filter(score_gt=0.5, label_in=["cat"], label_not_in=["dog"])
        r = repr(node)
        assert "Filter" in r
        assert "0.5" in r
        assert "cat" in r
        assert "dog" in r


# ===================================================================
# TopK Node
# ===================================================================


class TestTopK:
    """Tests for the TopK node."""

    def test_topk_basic(self, ctx: ExecutionContext, sample_detections: Detections):
        node = TopK(k=2)
        result = node.run(ctx, detections=sample_detections)

        topk = result["topk"]
        assert isinstance(topk, Detections)
        assert len(topk.instances) == 2
        assert topk.instances[0].score == 0.95
        assert topk.instances[1].score == 0.85

    def test_topk_k_greater_than_count(self, ctx: ExecutionContext, sample_detections: Detections):
        node = TopK(k=100)
        result = node.run(ctx, detections=sample_detections)

        topk = result["topk"]
        assert len(topk.instances) == 5  # All instances kept

    def test_topk_zero(self, ctx: ExecutionContext, sample_detections: Detections):
        node = TopK(k=0)
        result = node.run(ctx, detections=sample_detections)

        topk = result["topk"]
        assert len(topk.instances) == 0

    def test_topk_empty_input(self, ctx: ExecutionContext):
        node = TopK(k=3)
        empty = Detections(instances=[], instance_ids=[])
        result = node.run(ctx, detections=empty)

        topk = result["topk"]
        assert len(topk.instances) == 0

    def test_topk_preserves_instance_ids(self, ctx: ExecutionContext, sample_detections: Detections):
        node = TopK(k=1)
        result = node.run(ctx, detections=sample_detections)

        topk = result["topk"]
        # Highest score is 0.95 (inst_0000)
        assert topk.instance_ids == ["inst_0000"]

    def test_topk_custom_output_name(self, ctx: ExecutionContext, sample_detections: Detections):
        node = TopK(k=1, out="best")
        result = node.run(ctx, detections=sample_detections)

        assert "best" in result

    def test_topk_negative_k_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            TopK(k=-1)

    def test_topk_records_metrics(self, ctx: ExecutionContext, sample_detections: Detections):
        node = TopK(k=2, name="my_topk")
        node.run(ctx, detections=sample_detections)

        metrics = ctx.get_metrics()
        assert metrics["my_topk"]["k"] == 2
        assert metrics["my_topk"]["input_count"] == 5
        assert metrics["my_topk"]["output_count"] == 2

    def test_topk_immutability(self, ctx: ExecutionContext, sample_detections: Detections):
        original_count = len(sample_detections.instances)
        node = TopK(k=1)
        node.run(ctx, detections=sample_detections)
        assert len(sample_detections.instances) == original_count

    def test_topk_repr(self):
        node = TopK(k=5, name="top5_node")
        r = repr(node)
        assert "TopK" in r
        assert "k=5" in r


# ===================================================================
# ExtractROIs Node
# ===================================================================


class TestExtractROIs:
    """Tests for the ExtractROIs node."""

    def test_extract_basic(self, ctx: ExecutionContext, sample_image: Image):
        inst = _make_instance("cat", 0.9, bbox=(10, 20, 50, 60))
        dets = Detections(instances=[inst])

        node = ExtractROIs()
        result = node.run(ctx, image=sample_image, detections=dets)

        rois = result["rois"]
        assert isinstance(rois, ROIs)
        assert len(rois.roi_images) == 1
        assert rois.source_boxes[0] == (10, 20, 50, 60)

    def test_extract_crop_dimensions(self, ctx: ExecutionContext, sample_image: Image):
        inst = _make_instance("cat", 0.9, bbox=(10, 20, 50, 60))
        dets = Detections(instances=[inst])

        node = ExtractROIs()
        result = node.run(ctx, image=sample_image, detections=dets)

        crop = result["rois"].roi_images[0]
        # Crop should be (50-10) x (60-20) = 40 x 40
        assert crop.size == (40, 40)

    def test_extract_with_padding(self, ctx: ExecutionContext, sample_image: Image):
        inst = _make_instance("cat", 0.9, bbox=(10, 20, 50, 60))
        dets = Detections(instances=[inst])

        node = ExtractROIs(padding=5)
        result = node.run(ctx, image=sample_image, detections=dets)

        crop = result["rois"].roi_images[0]
        # With padding: (5, 15, 55, 65) but clamped to (5, 15, 55, 65)
        assert crop.size == (50, 50)

    def test_extract_padding_clamped_to_image(self, ctx: ExecutionContext, sample_image: Image):
        inst = _make_instance("cat", 0.9, bbox=(0, 0, 100, 100))
        dets = Detections(instances=[inst])

        node = ExtractROIs(padding=50)
        result = node.run(ctx, image=sample_image, detections=dets)

        crop = result["rois"].roi_images[0]
        # Padded box (−50, −50, 150, 150) → clamped to (0, 0, 100, 100)
        assert crop.size == (100, 100)

    def test_extract_multiple(self, ctx: ExecutionContext, sample_image: Image):
        dets = _make_detections([("cat", 0.9), ("dog", 0.8)])

        node = ExtractROIs()
        result = node.run(ctx, image=sample_image, detections=dets)

        rois = result["rois"]
        assert len(rois.roi_images) == 2

    def test_extract_preserves_instance_ids(self, ctx: ExecutionContext, sample_image: Image):
        inst = _make_instance("cat", 0.9, bbox=(10, 20, 50, 60))
        dets = Detections(instances=[inst], instance_ids=["my_cat_0"])

        node = ExtractROIs()
        result = node.run(ctx, image=sample_image, detections=dets)

        assert result["rois"].instance_ids == ["my_cat_0"]

    def test_extract_skip_no_bbox(self, ctx: ExecutionContext, sample_image: Image):
        """Instances without bbox are silently skipped."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:50, 10:50] = True
        inst_no_bbox = Instance(mask=mask, score=0.9, label=0, label_name="cat")
        inst_with_bbox = _make_instance("dog", 0.8, bbox=(5, 5, 30, 30))
        dets = Detections(instances=[inst_no_bbox, inst_with_bbox])

        node = ExtractROIs()
        result = node.run(ctx, image=sample_image, detections=dets)

        rois = result["rois"]
        assert len(rois.roi_images) == 1

    def test_extract_empty_detections(self, ctx: ExecutionContext, sample_image: Image):
        empty = Detections(instances=[], instance_ids=[])

        node = ExtractROIs()
        result = node.run(ctx, image=sample_image, detections=empty)

        rois = result["rois"]
        assert len(rois.roi_images) == 0
        assert rois.meta["source_width"] == 100

    def test_extract_custom_output_name(self, ctx: ExecutionContext, sample_image: Image):
        inst = _make_instance("cat", 0.9, bbox=(10, 20, 50, 60))
        dets = Detections(instances=[inst])

        node = ExtractROIs(out="crops")
        result = node.run(ctx, image=sample_image, detections=dets)

        assert "crops" in result

    def test_extract_negative_padding_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ExtractROIs(padding=-1)

    def test_extract_records_metrics(self, ctx: ExecutionContext, sample_image: Image):
        dets = _make_detections([("cat", 0.9), ("dog", 0.8)])
        node = ExtractROIs(name="my_roi")
        node.run(ctx, image=sample_image, detections=dets)

        metrics = ctx.get_metrics()
        assert metrics["my_roi"]["num_rois"] == 2

    def test_extract_meta_contains_source_info(self, ctx: ExecutionContext, sample_image: Image):
        inst = _make_instance("cat", 0.9, bbox=(10, 20, 50, 60))
        dets = Detections(instances=[inst])

        node = ExtractROIs(padding=3)
        result = node.run(ctx, image=sample_image, detections=dets)

        meta = result["rois"].meta
        assert meta["source_width"] == 100
        assert meta["source_height"] == 100
        assert meta["padding"] == 3

    def test_extract_repr(self):
        node = ExtractROIs(padding=5, name="roi_node")
        r = repr(node)
        assert "ExtractROIs" in r
        assert "padding=5" in r


# ===================================================================
# ExpandBoxes Node
# ===================================================================


class TestExpandBoxes:
    """Tests for the ExpandBoxes node."""

    def _make_mask_instance(self, label_name: str, score: float, mask: np.ndarray) -> Instance:
        """Build an Instance with a binary mask (and no bbox)."""
        # Instance requires bbox or mask, provide mask only
        return Instance(mask=mask, score=score, label=0, label_name=label_name)

    def test_expand_basic(self, ctx: ExecutionContext):
        """Bounding box should be recomputed from mask."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:60, 10:50] = True  # y 20-59, x 10-49

        inst = _make_instance("cat", 0.9, bbox=(0, 0, 100, 100))
        mask_inst = self._make_mask_instance("cat", 0.9, mask)
        dets = Detections(instances=[inst], instance_ids=["inst_0000"])
        masks = Masks(instances=[mask_inst], instance_ids=["inst_0000"])

        node = ExpandBoxes()
        result = node.run(ctx, detections=dets, masks=masks)

        expanded = result["expanded"]
        assert isinstance(expanded, Detections)
        assert len(expanded.instances) == 1

        new_bbox = expanded.instances[0].bbox
        assert new_bbox == (10.0, 20.0, 50.0, 60.0)

    def test_expand_preserves_instance_ids(self, ctx: ExecutionContext):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:30, 10:30] = True

        inst = _make_instance("cat", 0.9, bbox=(0, 0, 100, 100))
        mask_inst = self._make_mask_instance("cat", 0.9, mask)
        dets = Detections(instances=[inst], instance_ids=["my_id"])
        masks = Masks(instances=[mask_inst], instance_ids=["my_id"])

        node = ExpandBoxes()
        result = node.run(ctx, detections=dets, masks=masks)

        assert result["expanded"].instance_ids == ["my_id"]

    def test_expand_no_matching_mask_keeps_original(self, ctx: ExecutionContext):
        """Detections without corresponding mask keep their original bbox."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:30, 10:30] = True

        inst = _make_instance("cat", 0.9, bbox=(0, 0, 100, 100))
        mask_inst = self._make_mask_instance("dog", 0.8, mask)
        dets = Detections(instances=[inst], instance_ids=["inst_0"])
        masks = Masks(instances=[mask_inst], instance_ids=["mask_0"])

        node = ExpandBoxes()
        result = node.run(ctx, detections=dets, masks=masks)

        # No instance_id match → original bbox preserved
        assert result["expanded"].instances[0].bbox == (0, 0, 100, 100)

    def test_expand_empty_mask(self, ctx: ExecutionContext):
        """An all-zero mask should leave the original bbox unchanged."""
        mask = np.zeros((100, 100), dtype=bool)  # empty

        inst = _make_instance("cat", 0.9, bbox=(5, 5, 50, 50))
        mask_inst = self._make_mask_instance("cat", 0.9, mask)
        dets = Detections(instances=[inst], instance_ids=["inst_0"])
        masks = Masks(instances=[mask_inst], instance_ids=["inst_0"])

        node = ExpandBoxes()
        result = node.run(ctx, detections=dets, masks=masks)

        # Empty mask → bbox_from_binary_mask returns None → fallback
        assert result["expanded"].instances[0].bbox == inst.bbox

    def test_expand_empty_detections(self, ctx: ExecutionContext):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:30, 10:30] = True
        mask_inst = self._make_mask_instance("cat", 0.9, mask)
        dets = Detections(instances=[], instance_ids=[])
        masks = Masks(instances=[mask_inst], instance_ids=["mask_0"])

        node = ExpandBoxes()
        result = node.run(ctx, detections=dets, masks=masks)

        assert len(result["expanded"].instances) == 0

    def test_expand_multiple_instances(self, ctx: ExecutionContext):
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[0:10, 0:10] = True

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[50:80, 60:90] = True

        inst1 = _make_instance("a", 0.9, bbox=(0, 0, 100, 100))
        inst2 = _make_instance("b", 0.8, bbox=(0, 0, 100, 100))
        mask_inst1 = self._make_mask_instance("a", 0.9, mask1)
        mask_inst2 = self._make_mask_instance("b", 0.8, mask2)

        dets = Detections(
            instances=[inst1, inst2],
            instance_ids=["id_0", "id_1"],
        )
        masks = Masks(
            instances=[mask_inst1, mask_inst2],
            instance_ids=["id_0", "id_1"],
        )

        node = ExpandBoxes()
        result = node.run(ctx, detections=dets, masks=masks)

        expanded = result["expanded"]
        assert expanded.instances[0].bbox == (0.0, 0.0, 10.0, 10.0)
        assert expanded.instances[1].bbox == (60.0, 50.0, 90.0, 80.0)

    def test_expand_preserves_entities(self, ctx: ExecutionContext):
        """Entities pass through unchanged."""
        from mata.core.types import Entity

        mask = np.zeros((100, 100), dtype=bool)
        mask[10:30, 10:30] = True

        inst = _make_instance("cat", 0.9, bbox=(0, 0, 100, 100))
        mask_inst = self._make_mask_instance("cat", 0.9, mask)
        dets = Detections(
            instances=[inst],
            instance_ids=["inst_0"],
            entities=[Entity(label="dog", score=0.7)],
        )
        masks = Masks(instances=[mask_inst], instance_ids=["inst_0"])

        node = ExpandBoxes()
        result = node.run(ctx, detections=dets, masks=masks)

        assert len(result["expanded"].entities) == 1
        assert result["expanded"].entities[0].label == "dog"

    def test_expand_records_metrics(self, ctx: ExecutionContext):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:30, 10:30] = True

        inst = _make_instance("cat", 0.9, bbox=(0, 0, 100, 100))
        mask_inst = self._make_mask_instance("cat", 0.9, mask)
        dets = Detections(instances=[inst], instance_ids=["inst_0"])
        masks = Masks(instances=[mask_inst], instance_ids=["inst_0"])

        node = ExpandBoxes(name="my_expand")
        node.run(ctx, detections=dets, masks=masks)

        metrics = ctx.get_metrics()
        assert metrics["my_expand"]["total_instances"] == 1
        assert metrics["my_expand"]["updated_boxes"] == 1

    def test_expand_custom_output_name(self, ctx: ExecutionContext):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:30, 10:30] = True

        inst = _make_instance("cat", 0.9, bbox=(0, 0, 100, 100))
        mask_inst = self._make_mask_instance("cat", 0.9, mask)
        dets = Detections(instances=[inst], instance_ids=["inst_0"])
        masks = Masks(instances=[mask_inst], instance_ids=["inst_0"])

        node = ExpandBoxes(out="tight")
        result = node.run(ctx, detections=dets, masks=masks)

        assert "tight" in result

    def test_expand_repr(self):
        node = ExpandBoxes(name="expand_node")
        r = repr(node)
        assert "ExpandBoxes" in r


# ===================================================================
# _bbox_from_binary_mask helper
# ===================================================================


class TestBboxFromBinaryMask:
    def test_simple_rect(self):
        mask = np.zeros((50, 80), dtype=bool)
        mask[5:15, 10:30] = True
        bbox = _bbox_from_binary_mask(mask)
        assert bbox == (10.0, 5.0, 30.0, 15.0)

    def test_single_pixel(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[25, 30] = True
        bbox = _bbox_from_binary_mask(mask)
        assert bbox == (30.0, 25.0, 31.0, 26.0)

    def test_full_mask(self):
        mask = np.ones((10, 20), dtype=bool)
        bbox = _bbox_from_binary_mask(mask)
        assert bbox == (0.0, 0.0, 20.0, 10.0)

    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=bool)
        bbox = _bbox_from_binary_mask(mask)
        assert bbox is None


# ===================================================================
# Node type declarations
# ===================================================================


class TestNodeTypeDeclarations:
    """Verify that nodes declare correct input/output types."""

    def test_filter_types(self):
        node = Filter()
        assert node.inputs == {"dets": Detections}  # dynamic: default src="dets"
        assert node.outputs == {"filtered": Detections}  # dynamic: default out="filtered"

    def test_topk_types(self):
        node = TopK(k=1)
        assert node.inputs == {"detections": Detections}
        assert node.outputs == {"detections": Detections}

    def test_extract_rois_types(self):
        node = ExtractROIs()
        assert node.inputs == {"image": Image, "detections": Detections}
        assert node.outputs == {"rois": ROIs}

    def test_expand_boxes_types(self):
        node = ExpandBoxes()
        assert node.inputs == {"detections": Detections, "masks": Masks}
        assert node.outputs == {"detections": Detections}
