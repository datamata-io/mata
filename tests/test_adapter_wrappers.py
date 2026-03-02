"""Comprehensive tests for adapter wrappers (Task 4.3).

Tests all wrapper classes:
- DetectorWrapper (detection adapters → Detector protocol)
- SegmenterWrapper (segment adapters → Segmenter protocol)
- ClassifierWrapper (classification adapters → Classifier protocol)
- DepthWrapper (depth adapters → DepthEstimator protocol)
- SAMWrapper (SAM adapters → Segmenter protocol with prompts)
- VLMWrapper (VLM adapters → VisionLanguageModel protocol)

Each wrapper is tested for:
1. Correct protocol compliance (runtime isinstance check)
2. Image artifact conversion (source_path preferring, PIL fallback)
3. Result type conversion (VisionResult → Detections/Masks, passthrough for Classify/Depth)
4. kwargs passthrough to adapter
5. Error handling (invalid image type, adapter failure, missing predict method)
6. Multi-adapter type support per task
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image as PILImage

from mata.adapters.wrappers import (
    ClassifierWrapper,
    DepthWrapper,
    DetectorWrapper,
    SAMWrapper,
    SegmenterWrapper,
    VLMWrapper,
    wrap_classifier,
    wrap_depth,
    wrap_detector,
    wrap_sam,
    wrap_segmenter,
    wrap_vlm,
)
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.artifacts.masks import Masks
from mata.core.registry.protocols import (
    Classifier,
    DepthEstimator,
    Detector,
    Segmenter,
    VisionLanguageModel,
)
from mata.core.types import (
    Classification,
    ClassifyResult,
    DepthResult,
    Entity,
    Instance,
    VisionResult,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_pil_image():
    """Create a sample PIL image for testing."""
    return PILImage.new("RGB", (640, 480), color=(128, 128, 128))


@pytest.fixture
def sample_image_artifact(sample_pil_image):
    """Create a sample Image artifact."""
    return Image.from_pil(sample_pil_image)


@pytest.fixture
def sample_image_with_path(tmp_path, sample_pil_image):
    """Create a sample Image artifact with source_path."""
    path = str(tmp_path / "test_image.jpg")
    sample_pil_image.save(path)
    return Image.from_path(path)


@pytest.fixture
def sample_instances():
    """Create sample Instance objects for testing."""
    return [
        Instance(
            bbox=(10, 20, 100, 150),
            score=0.95,
            label=0,
            label_name="cat",
        ),
        Instance(
            bbox=(200, 50, 350, 300),
            score=0.87,
            label=1,
            label_name="dog",
        ),
    ]


@pytest.fixture
def sample_mask_instances():
    """Create sample Instance objects with masks for testing."""
    h, w = 480, 640
    mask1 = np.zeros((h, w), dtype=bool)
    mask1[20:150, 10:100] = True
    mask2 = np.zeros((h, w), dtype=bool)
    mask2[50:300, 200:350] = True
    return [
        Instance(
            bbox=(10, 20, 100, 150),
            mask=mask1,
            score=0.92,
            label=0,
            label_name="cat",
        ),
        Instance(
            bbox=(200, 50, 350, 300),
            mask=mask2,
            score=0.85,
            label=1,
            label_name="dog",
        ),
    ]


@pytest.fixture
def sample_vision_result(sample_instances):
    """Create a sample VisionResult."""
    return VisionResult(
        instances=sample_instances,
        meta={"model": "test-model"},
    )


@pytest.fixture
def sample_mask_vision_result(sample_mask_instances):
    """Create a sample VisionResult with masks."""
    return VisionResult(
        instances=sample_mask_instances,
        meta={"model": "test-segment-model"},
    )


@pytest.fixture
def sample_classify_result():
    """Create a sample ClassifyResult."""
    return ClassifyResult(
        predictions=[
            Classification(label=0, score=0.85, label_name="cat"),
            Classification(label=1, score=0.10, label_name="dog"),
            Classification(label=2, score=0.05, label_name="bird"),
        ],
        meta={"model": "test-classify-model"},
    )


@pytest.fixture
def sample_depth_result():
    """Create a sample DepthResult."""
    depth_map = np.random.rand(480, 640).astype(np.float32)
    return DepthResult(
        depth=depth_map,
        normalized=depth_map / depth_map.max(),
        meta={"model": "test-depth-model"},
    )


@pytest.fixture
def sample_vlm_result():
    """Create a sample VisionResult from VLM with entities."""
    return VisionResult(
        instances=[],
        entities=[
            Entity(label="cat", score=0.95, attributes={"color": "orange"}),
            Entity(label="dog", score=0.87, attributes={"size": "large"}),
        ],
        text="I see a cat and a dog in the image.",
        prompt="What objects are in this image?",
        meta={"model": "test-vlm"},
    )


@pytest.fixture
def sample_vlm_result_with_instances(sample_instances):
    """Create a VisionResult from VLM with auto-promoted instances."""
    return VisionResult(
        instances=sample_instances,
        entities=[
            Entity(label="cat", score=0.95),
            Entity(label="dog", score=0.87),
        ],
        text="Objects: cat, dog",
        meta={"model": "test-vlm", "auto_promote": True},
    )


def _make_mock_adapter(return_value):
    """Create a mock adapter with predict() that returns given value."""
    adapter = Mock()
    adapter.predict = Mock(return_value=return_value)
    return adapter


# =============================================================================
# DetectorWrapper Tests
# =============================================================================


class TestDetectorWrapper:
    """Tests for DetectorWrapper."""

    def test_predict_returns_detections(self, sample_image_artifact, sample_vision_result):
        """Wrapper predict() returns Detections artifact."""
        adapter = _make_mock_adapter(sample_vision_result)
        wrapper = DetectorWrapper(adapter)

        result = wrapper.predict(sample_image_artifact)

        assert isinstance(result, Detections)
        assert len(result.instances) == 2
        assert len(result.instance_ids) == 2
        assert result.instances[0].label_name == "cat"
        assert result.instances[1].label_name == "dog"

    def test_predict_converts_image_pil_fallback(self, sample_image_artifact, sample_vision_result):
        """Wrapper converts Image without source_path to PIL."""
        adapter = _make_mock_adapter(sample_vision_result)
        wrapper = DetectorWrapper(adapter)

        wrapper.predict(sample_image_artifact)

        call_args = adapter.predict.call_args
        img_arg = call_args[0][0]
        assert isinstance(img_arg, PILImage.Image)

    def test_predict_converts_image_path_preferred(self, sample_image_with_path, sample_vision_result):
        """Wrapper prefers source_path over PIL conversion."""
        adapter = _make_mock_adapter(sample_vision_result)
        wrapper = DetectorWrapper(adapter)

        wrapper.predict(sample_image_with_path)

        call_args = adapter.predict.call_args
        img_arg = call_args[0][0]
        assert isinstance(img_arg, str)
        assert img_arg == sample_image_with_path.source_path

    def test_predict_passes_kwargs(self, sample_image_artifact, sample_vision_result):
        """Wrapper passes kwargs through to adapter.predict()."""
        adapter = _make_mock_adapter(sample_vision_result)
        wrapper = DetectorWrapper(adapter)

        wrapper.predict(sample_image_artifact, threshold=0.7, nms_iou=0.45)

        call_args = adapter.predict.call_args
        assert call_args[1]["threshold"] == 0.7
        assert call_args[1]["nms_iou"] == 0.45

    def test_predict_passes_text_prompts(self, sample_image_artifact, sample_vision_result):
        """Wrapper passes text_prompts for zero-shot detection."""
        adapter = _make_mock_adapter(sample_vision_result)
        wrapper = DetectorWrapper(adapter)

        wrapper.predict(sample_image_artifact, text_prompts="cat . dog")

        call_args = adapter.predict.call_args
        assert call_args[1]["text_prompts"] == "cat . dog"

    def test_predict_rejects_non_image_input(self, sample_vision_result):
        """Wrapper raises TypeError for non-Image input."""
        adapter = _make_mock_adapter(sample_vision_result)
        wrapper = DetectorWrapper(adapter)

        with pytest.raises(TypeError, match="Expected Image artifact"):
            wrapper.predict("not_an_image.jpg")

    def test_predict_wraps_adapter_errors(self, sample_image_artifact):
        """Wrapper wraps adapter errors in RuntimeError."""
        adapter = Mock()
        adapter.predict = Mock(side_effect=ValueError("Model error"))
        wrapper = DetectorWrapper(adapter)

        with pytest.raises(RuntimeError, match="Detection adapter"):
            wrapper.predict(sample_image_artifact)

    def test_init_rejects_adapter_without_predict(self):
        """Wrapper rejects adapters without predict method."""
        adapter = object()  # No predict() method

        with pytest.raises(TypeError, match="does not have a predict"):
            DetectorWrapper(adapter)

    def test_preserves_metadata(self, sample_image_artifact, sample_vision_result):
        """Wrapper preserves VisionResult metadata in Detections."""
        adapter = _make_mock_adapter(sample_vision_result)
        wrapper = DetectorWrapper(adapter)

        result = wrapper.predict(sample_image_artifact)
        assert result.meta.get("model") == "test-model"

    def test_empty_detections(self, sample_image_artifact):
        """Wrapper handles empty detection results."""
        empty_result = VisionResult(instances=[], meta={})
        adapter = _make_mock_adapter(empty_result)
        wrapper = DetectorWrapper(adapter)

        result = wrapper.predict(sample_image_artifact)
        assert isinstance(result, Detections)
        assert len(result.instances) == 0

    def test_repr(self):
        """Wrapper repr includes adapter type name."""
        adapter = _make_mock_adapter(None)
        wrapper = DetectorWrapper(adapter)
        assert "DetectorWrapper" in repr(wrapper)

    def test_protocol_compliance(self, sample_vision_result):
        """DetectorWrapper satisfies Detector protocol."""
        adapter = _make_mock_adapter(sample_vision_result)
        wrapper = DetectorWrapper(adapter)
        assert isinstance(wrapper, Detector)

    def test_factory_function(self, sample_vision_result):
        """wrap_detector factory returns DetectorWrapper."""
        adapter = _make_mock_adapter(sample_vision_result)
        wrapper = wrap_detector(adapter)
        assert isinstance(wrapper, DetectorWrapper)


# =============================================================================
# SegmenterWrapper Tests
# =============================================================================


class TestSegmenterWrapper:
    """Tests for SegmenterWrapper."""

    def test_segment_returns_masks(self, sample_image_artifact, sample_mask_vision_result):
        """Wrapper segment() returns Masks artifact."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SegmenterWrapper(adapter)

        result = wrapper.segment(sample_image_artifact)

        assert isinstance(result, Masks)
        assert len(result.instances) == 2
        assert len(result.instance_ids) == 2

    def test_segment_converts_image_pil(self, sample_image_artifact, sample_mask_vision_result):
        """Wrapper converts Image without path to PIL."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SegmenterWrapper(adapter)

        wrapper.segment(sample_image_artifact)

        img_arg = adapter.predict.call_args[0][0]
        assert isinstance(img_arg, PILImage.Image)

    def test_segment_prefers_path(self, sample_image_with_path, sample_mask_vision_result):
        """Wrapper uses source_path when available."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SegmenterWrapper(adapter)

        wrapper.segment(sample_image_with_path)

        img_arg = adapter.predict.call_args[0][0]
        assert isinstance(img_arg, str)

    def test_segment_passes_kwargs(self, sample_image_artifact, sample_mask_vision_result):
        """Wrapper passes kwargs through to adapter."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SegmenterWrapper(adapter)

        wrapper.segment(sample_image_artifact, threshold=0.6)

        assert adapter.predict.call_args[1]["threshold"] == 0.6

    def test_segment_rejects_non_image(self, sample_mask_vision_result):
        """Wrapper rejects non-Image input."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SegmenterWrapper(adapter)

        with pytest.raises(TypeError, match="Expected Image artifact"):
            wrapper.segment("not_an_image")

    def test_segment_wraps_errors(self, sample_image_artifact):
        """Wrapper wraps adapter errors."""
        adapter = Mock()
        adapter.predict = Mock(side_effect=RuntimeError("Segment failed"))
        wrapper = SegmenterWrapper(adapter)

        with pytest.raises(RuntimeError, match="Segment adapter"):
            wrapper.segment(sample_image_artifact)

    def test_init_rejects_no_predict(self):
        """Rejects adapters without predict."""
        with pytest.raises(TypeError, match="does not have a predict"):
            SegmenterWrapper(object())

    def test_protocol_compliance(self, sample_mask_vision_result):
        """SegmenterWrapper satisfies Segmenter protocol."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SegmenterWrapper(adapter)
        assert isinstance(wrapper, Segmenter)

    def test_factory_function(self, sample_mask_vision_result):
        """wrap_segmenter factory returns SegmenterWrapper."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = wrap_segmenter(adapter)
        assert isinstance(wrapper, SegmenterWrapper)

    def test_repr(self):
        """Wrapper repr includes adapter type."""
        adapter = _make_mock_adapter(None)
        repr_str = repr(SegmenterWrapper(adapter))
        assert "SegmenterWrapper" in repr_str


# =============================================================================
# ClassifierWrapper Tests
# =============================================================================


class TestClassifierWrapper:
    """Tests for ClassifierWrapper."""

    def test_classify_returns_classify_result(self, sample_image_artifact, sample_classify_result):
        """Wrapper classify() returns ClassifyResult directly."""
        adapter = _make_mock_adapter(sample_classify_result)
        wrapper = ClassifierWrapper(adapter)

        result = wrapper.classify(sample_image_artifact)

        assert isinstance(result, ClassifyResult)
        assert len(result.predictions) == 3
        assert result.predictions[0].label_name == "cat"

    def test_classify_converts_image(self, sample_image_artifact, sample_classify_result):
        """Wrapper converts Image to PIL for adapter."""
        adapter = _make_mock_adapter(sample_classify_result)
        wrapper = ClassifierWrapper(adapter)

        wrapper.classify(sample_image_artifact)

        img_arg = adapter.predict.call_args[0][0]
        assert isinstance(img_arg, PILImage.Image)

    def test_classify_prefers_path(self, sample_image_with_path, sample_classify_result):
        """Wrapper uses source_path when available."""
        adapter = _make_mock_adapter(sample_classify_result)
        wrapper = ClassifierWrapper(adapter)

        wrapper.classify(sample_image_with_path)

        img_arg = adapter.predict.call_args[0][0]
        assert isinstance(img_arg, str)

    def test_classify_passes_kwargs(self, sample_image_artifact, sample_classify_result):
        """Wrapper passes top_k and text_prompts through."""
        adapter = _make_mock_adapter(sample_classify_result)
        wrapper = ClassifierWrapper(adapter)

        wrapper.classify(sample_image_artifact, top_k=5, text_prompts=["cat", "dog"])

        call_kwargs = adapter.predict.call_args[1]
        assert call_kwargs["top_k"] == 5
        assert call_kwargs["text_prompts"] == ["cat", "dog"]

    def test_classify_rejects_non_image(self, sample_classify_result):
        """Wrapper rejects non-Image input."""
        adapter = _make_mock_adapter(sample_classify_result)
        wrapper = ClassifierWrapper(adapter)

        with pytest.raises(TypeError, match="Expected Image artifact"):
            wrapper.classify(42)

    def test_classify_wraps_errors(self, sample_image_artifact):
        """Wrapper wraps adapter errors."""
        adapter = Mock()
        adapter.predict = Mock(side_effect=ValueError("Classify error"))
        wrapper = ClassifierWrapper(adapter)

        with pytest.raises(RuntimeError, match="Classification adapter"):
            wrapper.classify(sample_image_artifact)

    def test_init_rejects_no_predict(self):
        """Rejects adapter without predict method."""
        with pytest.raises(TypeError, match="does not have a predict"):
            ClassifierWrapper(object())

    def test_protocol_compliance(self, sample_classify_result):
        """ClassifierWrapper satisfies Classifier protocol."""
        adapter = _make_mock_adapter(sample_classify_result)
        wrapper = ClassifierWrapper(adapter)
        assert isinstance(wrapper, Classifier)

    def test_factory_function(self, sample_classify_result):
        """wrap_classifier factory returns ClassifierWrapper."""
        adapter = _make_mock_adapter(sample_classify_result)
        wrapper = wrap_classifier(adapter)
        assert isinstance(wrapper, ClassifierWrapper)

    def test_repr(self):
        """Wrapper repr."""
        adapter = _make_mock_adapter(None)
        assert "ClassifierWrapper" in repr(ClassifierWrapper(adapter))


# =============================================================================
# DepthWrapper Tests
# =============================================================================


class TestDepthWrapper:
    """Tests for DepthWrapper."""

    def test_estimate_returns_depth_result(self, sample_image_artifact, sample_depth_result):
        """Wrapper estimate() returns DepthResult directly."""
        adapter = _make_mock_adapter(sample_depth_result)
        wrapper = DepthWrapper(adapter)

        result = wrapper.estimate(sample_image_artifact)

        assert isinstance(result, DepthResult)
        assert result.depth.shape == (480, 640)

    def test_estimate_converts_image(self, sample_image_artifact, sample_depth_result):
        """Wrapper converts Image to PIL."""
        adapter = _make_mock_adapter(sample_depth_result)
        wrapper = DepthWrapper(adapter)

        wrapper.estimate(sample_image_artifact)

        img_arg = adapter.predict.call_args[0][0]
        assert isinstance(img_arg, PILImage.Image)

    def test_estimate_prefers_path(self, sample_image_with_path, sample_depth_result):
        """Wrapper uses source_path when available."""
        adapter = _make_mock_adapter(sample_depth_result)
        wrapper = DepthWrapper(adapter)

        wrapper.estimate(sample_image_with_path)

        img_arg = adapter.predict.call_args[0][0]
        assert isinstance(img_arg, str)

    def test_estimate_passes_kwargs(self, sample_image_artifact, sample_depth_result):
        """Wrapper passes kwargs through."""
        adapter = _make_mock_adapter(sample_depth_result)
        wrapper = DepthWrapper(adapter)

        wrapper.estimate(sample_image_artifact, output_type="normalized")

        assert adapter.predict.call_args[1]["output_type"] == "normalized"

    def test_estimate_rejects_non_image(self, sample_depth_result):
        """Wrapper rejects non-Image input."""
        adapter = _make_mock_adapter(sample_depth_result)
        wrapper = DepthWrapper(adapter)

        with pytest.raises(TypeError, match="Expected Image artifact"):
            wrapper.estimate("not_an_image")

    def test_estimate_wraps_errors(self, sample_image_artifact):
        """Wrapper wraps adapter errors."""
        adapter = Mock()
        adapter.predict = Mock(side_effect=RuntimeError("Depth error"))
        wrapper = DepthWrapper(adapter)

        with pytest.raises(RuntimeError, match="Depth adapter"):
            wrapper.estimate(sample_image_artifact)

    def test_init_rejects_no_predict(self):
        """Rejects adapter without predict."""
        with pytest.raises(TypeError, match="does not have a predict"):
            DepthWrapper(object())

    def test_protocol_compliance(self, sample_depth_result):
        """DepthWrapper satisfies DepthEstimator protocol."""
        adapter = _make_mock_adapter(sample_depth_result)
        wrapper = DepthWrapper(adapter)
        assert isinstance(wrapper, DepthEstimator)

    def test_factory_function(self, sample_depth_result):
        """wrap_depth factory returns DepthWrapper."""
        adapter = _make_mock_adapter(sample_depth_result)
        wrapper = wrap_depth(adapter)
        assert isinstance(wrapper, DepthWrapper)

    def test_repr(self):
        """Wrapper repr."""
        adapter = _make_mock_adapter(None)
        assert "DepthWrapper" in repr(DepthWrapper(adapter))


# =============================================================================
# SAMWrapper Tests
# =============================================================================


class TestSAMWrapper:
    """Tests for SAMWrapper."""

    def test_segment_returns_masks(self, sample_image_artifact, sample_mask_vision_result):
        """Wrapper segment() returns Masks artifact."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SAMWrapper(adapter)

        result = wrapper.segment(sample_image_artifact)

        assert isinstance(result, Masks)
        assert len(result.instances) == 2

    def test_segment_passes_box_prompts(self, sample_image_artifact, sample_mask_vision_result):
        """Wrapper passes box_prompts to adapter."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SAMWrapper(adapter)

        boxes = [(50, 50, 300, 300)]
        wrapper.segment(sample_image_artifact, box_prompts=boxes)

        call_kwargs = adapter.predict.call_args[1]
        assert call_kwargs["box_prompts"] == boxes

    def test_segment_passes_point_prompts(self, sample_image_artifact, sample_mask_vision_result):
        """Wrapper passes point_prompts to adapter."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SAMWrapper(adapter)

        points = [(175, 175, 1)]
        wrapper.segment(sample_image_artifact, point_prompts=points)

        call_kwargs = adapter.predict.call_args[1]
        assert call_kwargs["point_prompts"] == points

    def test_segment_passes_text_prompts(self, sample_image_artifact, sample_mask_vision_result):
        """Wrapper passes text_prompts for SAM3."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SAMWrapper(adapter)

        wrapper.segment(sample_image_artifact, text_prompts="cat")

        call_kwargs = adapter.predict.call_args[1]
        assert call_kwargs["text_prompts"] == "cat"

    def test_segment_passes_threshold(self, sample_image_artifact, sample_mask_vision_result):
        """Wrapper passes threshold to adapter."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SAMWrapper(adapter)

        wrapper.segment(sample_image_artifact, threshold=0.8)

        call_kwargs = adapter.predict.call_args[1]
        assert call_kwargs["threshold"] == 0.8

    def test_segment_passes_box_labels(self, sample_image_artifact, sample_mask_vision_result):
        """Wrapper passes box_labels to adapter."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SAMWrapper(adapter)

        wrapper.segment(
            sample_image_artifact,
            box_prompts=[(50, 50, 300, 300)],
            box_labels=[1],
        )

        call_kwargs = adapter.predict.call_args[1]
        assert call_kwargs["box_labels"] == [1]

    def test_segment_mode_everything(self, sample_image_artifact, sample_mask_vision_result):
        """Wrapper passes mode='everything' for auto mask generation."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SAMWrapper(adapter)

        wrapper.segment(sample_image_artifact, mode="everything")

        call_kwargs = adapter.predict.call_args[1]
        assert call_kwargs["mode"] == "everything"

    def test_segment_passes_extra_kwargs(self, sample_image_artifact, sample_mask_vision_result):
        """Wrapper passes unknown kwargs through too."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SAMWrapper(adapter)

        wrapper.segment(sample_image_artifact, custom_param="value")

        call_kwargs = adapter.predict.call_args[1]
        assert call_kwargs["custom_param"] == "value"

    def test_segment_converts_image(self, sample_image_artifact, sample_mask_vision_result):
        """Wrapper converts Image to PIL."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SAMWrapper(adapter)

        wrapper.segment(sample_image_artifact)

        img_arg = adapter.predict.call_args[0][0]
        assert isinstance(img_arg, PILImage.Image)

    def test_segment_prefers_path(self, sample_image_with_path, sample_mask_vision_result):
        """Wrapper uses source_path when available."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SAMWrapper(adapter)

        wrapper.segment(sample_image_with_path)

        img_arg = adapter.predict.call_args[0][0]
        assert isinstance(img_arg, str)

    def test_segment_rejects_non_image(self, sample_mask_vision_result):
        """Wrapper rejects non-Image input."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SAMWrapper(adapter)

        with pytest.raises(TypeError, match="Expected Image artifact"):
            wrapper.segment("not_an_image")

    def test_segment_wraps_errors(self, sample_image_artifact):
        """Wrapper wraps adapter errors."""
        adapter = Mock()
        adapter.predict = Mock(side_effect=RuntimeError("SAM error"))
        wrapper = SAMWrapper(adapter)

        with pytest.raises(RuntimeError, match="SAM adapter"):
            wrapper.segment(sample_image_artifact)

    def test_init_rejects_no_predict(self):
        """Rejects adapter without predict."""
        with pytest.raises(TypeError, match="does not have a predict"):
            SAMWrapper(object())

    def test_protocol_compliance(self, sample_mask_vision_result):
        """SAMWrapper satisfies Segmenter protocol."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = SAMWrapper(adapter)
        assert isinstance(wrapper, Segmenter)

    def test_factory_function(self, sample_mask_vision_result):
        """wrap_sam factory returns SAMWrapper."""
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = wrap_sam(adapter)
        assert isinstance(wrapper, SAMWrapper)

    def test_repr(self):
        """Wrapper repr."""
        adapter = _make_mock_adapter(None)
        assert "SAMWrapper" in repr(SAMWrapper(adapter))


# =============================================================================
# VLMWrapper Tests
# =============================================================================


class TestVLMWrapper:
    """Tests for VLMWrapper."""

    def test_query_single_image(self, sample_image_artifact, sample_vlm_result):
        """Wrapper query() with single image returns VisionResult."""
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = VLMWrapper(adapter)

        result = wrapper.query(sample_image_artifact, "What is in this image?")

        assert isinstance(result, VisionResult)
        assert result.text == "I see a cat and a dog in the image."
        assert len(result.entities) == 2

    def test_query_multi_image(self, sample_image_artifact, sample_vlm_result):
        """Wrapper query() with multiple images passes additional images."""
        pil1 = PILImage.new("RGB", (640, 480), color=(100, 100, 100))
        pil2 = PILImage.new("RGB", (640, 480), color=(200, 200, 200))
        img1 = Image.from_pil(pil1)
        img2 = Image.from_pil(pil2)
        images = [sample_image_artifact, img1, img2]

        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = VLMWrapper(adapter)

        result = wrapper.query(images, "What changed between these images?")

        assert isinstance(result, VisionResult)
        # Verify adapter received images kwarg with additional images
        call_kwargs = adapter.predict.call_args[1]
        assert "images" in call_kwargs
        assert len(call_kwargs["images"]) == 2  # 2 additional images

    def test_query_single_image_list(self, sample_image_artifact, sample_vlm_result):
        """Wrapper handles single-item image list (no additional images)."""
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = VLMWrapper(adapter)

        result = wrapper.query([sample_image_artifact], "Describe this image.")

        assert isinstance(result, VisionResult)
        call_kwargs = adapter.predict.call_args[1]
        # Single image list → no 'images' kwarg
        assert "images" not in call_kwargs or call_kwargs["images"] is None

    def test_query_with_output_mode(self, sample_image_artifact, sample_vlm_result):
        """Wrapper passes output_mode to adapter."""
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = VLMWrapper(adapter)

        wrapper.query(sample_image_artifact, "List objects.", output_mode="detect")

        call_kwargs = adapter.predict.call_args[1]
        assert call_kwargs["output_mode"] == "detect"

    def test_query_output_modes(self, sample_image_artifact, sample_vlm_result):
        """Test all output_mode values are passed correctly."""
        for mode in [None, "json", "detect", "classify", "describe"]:
            adapter = _make_mock_adapter(sample_vlm_result)
            wrapper = VLMWrapper(adapter)

            wrapper.query(sample_image_artifact, "Test", output_mode=mode)
            call_kwargs = adapter.predict.call_args[1]
            assert call_kwargs["output_mode"] == mode

    def test_query_with_auto_promote_true(self, sample_image_artifact, sample_vlm_result_with_instances):
        """Wrapper passes auto_promote=True to adapter."""
        adapter = _make_mock_adapter(sample_vlm_result_with_instances)
        wrapper = VLMWrapper(adapter)

        result = wrapper.query(
            sample_image_artifact,
            "Detect objects with boxes.",
            output_mode="detect",
            auto_promote=True,
        )

        call_kwargs = adapter.predict.call_args[1]
        assert call_kwargs["auto_promote"] is True
        # VLM result has both entities and instances (auto-promoted)
        assert len(result.instances) == 2
        assert len(result.entities) == 2

    def test_query_with_auto_promote_false(self, sample_image_artifact, sample_vlm_result):
        """Wrapper passes auto_promote=False (default) to adapter."""
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = VLMWrapper(adapter)

        wrapper.query(sample_image_artifact, "What is this?")

        call_kwargs = adapter.predict.call_args[1]
        assert call_kwargs["auto_promote"] is False

    def test_query_preserves_entities(self, sample_image_artifact, sample_vlm_result):
        """Wrapper preserves entities in VisionResult."""
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = VLMWrapper(adapter)

        result = wrapper.query(sample_image_artifact, "List objects.", output_mode="json")

        assert len(result.entities) == 2
        assert result.entities[0].label == "cat"
        assert result.entities[0].score == 0.95
        assert result.entities[0].attributes == {"color": "orange"}
        assert result.entities[1].label == "dog"

    def test_query_preserves_instances(self, sample_image_artifact, sample_vlm_result_with_instances):
        """Wrapper preserves auto-promoted instances in VisionResult."""
        adapter = _make_mock_adapter(sample_vlm_result_with_instances)
        wrapper = VLMWrapper(adapter)

        result = wrapper.query(sample_image_artifact, "Detect.", output_mode="detect", auto_promote=True)

        assert len(result.instances) == 2
        assert result.instances[0].label_name == "cat"

    def test_query_passes_vlm_kwargs(self, sample_image_artifact, sample_vlm_result):
        """Wrapper passes VLM-specific kwargs through."""
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = VLMWrapper(adapter)

        wrapper.query(
            sample_image_artifact,
            "Describe.",
            system_prompt="Be concise.",
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )

        call_kwargs = adapter.predict.call_args[1]
        assert call_kwargs["system_prompt"] == "Be concise."
        assert call_kwargs["max_new_tokens"] == 256
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9

    def test_query_image_converts_pil(self, sample_image_artifact, sample_vlm_result):
        """Wrapper converts Image without path to PIL."""
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = VLMWrapper(adapter)

        wrapper.query(sample_image_artifact, "Test")

        img_arg = adapter.predict.call_args[0][0]
        assert isinstance(img_arg, PILImage.Image)

    def test_query_image_prefers_path(self, sample_image_with_path, sample_vlm_result):
        """Wrapper uses source_path when available."""
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = VLMWrapper(adapter)

        wrapper.query(sample_image_with_path, "Test")

        img_arg = adapter.predict.call_args[0][0]
        assert isinstance(img_arg, str)

    def test_query_rejects_non_image(self, sample_vlm_result):
        """Wrapper rejects non-Image input."""
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = VLMWrapper(adapter)

        with pytest.raises(TypeError, match="Expected Image or List"):
            wrapper.query("not_an_image.jpg", "Test")

    def test_query_rejects_mixed_list(self, sample_image_artifact, sample_vlm_result):
        """Wrapper rejects list with non-Image items."""
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = VLMWrapper(adapter)

        with pytest.raises(TypeError, match="All items in image list"):
            wrapper.query([sample_image_artifact, "not_an_image"], "Test")

    def test_query_rejects_empty_list(self, sample_vlm_result):
        """Wrapper rejects empty image list."""
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = VLMWrapper(adapter)

        with pytest.raises(ValueError, match="Image list cannot be empty"):
            wrapper.query([], "Test")

    def test_query_rejects_empty_prompt(self, sample_image_artifact, sample_vlm_result):
        """Wrapper rejects empty prompt."""
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = VLMWrapper(adapter)

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            wrapper.query(sample_image_artifact, "")

    def test_query_wraps_adapter_errors(self, sample_image_artifact):
        """Wrapper wraps adapter errors in RuntimeError."""
        adapter = Mock()
        adapter.predict = Mock(side_effect=RuntimeError("VLM error"))
        wrapper = VLMWrapper(adapter)

        with pytest.raises(RuntimeError, match="VLM adapter"):
            wrapper.query(sample_image_artifact, "Test")

    def test_init_rejects_no_predict(self):
        """Rejects adapter without predict method."""
        with pytest.raises(TypeError, match="does not have a predict"):
            VLMWrapper(object())

    def test_protocol_compliance(self, sample_vlm_result):
        """VLMWrapper satisfies VisionLanguageModel protocol."""
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = VLMWrapper(adapter)
        assert isinstance(wrapper, VisionLanguageModel)

    def test_factory_function(self, sample_vlm_result):
        """wrap_vlm factory returns VLMWrapper."""
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = wrap_vlm(adapter)
        assert isinstance(wrapper, VLMWrapper)

    def test_repr(self):
        """Wrapper repr."""
        adapter = _make_mock_adapter(None)
        assert "VLMWrapper" in repr(VLMWrapper(adapter))


# =============================================================================
# Cross-Wrapper Tests
# =============================================================================


class TestWrapperInteroperability:
    """Tests for cross-wrapper functionality and edge cases."""

    def test_all_wrappers_implement_protocol(
        self,
        sample_vision_result,
        sample_mask_vision_result,
        sample_classify_result,
        sample_depth_result,
        sample_vlm_result,
    ):
        """All wrappers satisfy their respective protocols."""
        detect_adapter = _make_mock_adapter(sample_vision_result)
        segment_adapter = _make_mock_adapter(sample_mask_vision_result)
        classify_adapter = _make_mock_adapter(sample_classify_result)
        depth_adapter = _make_mock_adapter(sample_depth_result)
        sam_adapter = _make_mock_adapter(sample_mask_vision_result)
        vlm_adapter = _make_mock_adapter(sample_vlm_result)

        assert isinstance(DetectorWrapper(detect_adapter), Detector)
        assert isinstance(SegmenterWrapper(segment_adapter), Segmenter)
        assert isinstance(ClassifierWrapper(classify_adapter), Classifier)
        assert isinstance(DepthWrapper(depth_adapter), DepthEstimator)
        assert isinstance(SAMWrapper(sam_adapter), Segmenter)
        assert isinstance(VLMWrapper(vlm_adapter), VisionLanguageModel)

    def test_image_path_preference_all_wrappers(
        self,
        sample_image_with_path,
        sample_vision_result,
        sample_mask_vision_result,
        sample_classify_result,
        sample_depth_result,
        sample_vlm_result,
    ):
        """All wrappers prefer source_path for image conversion."""
        wrappers_and_results = [
            (DetectorWrapper, sample_vision_result, "predict"),
            (SegmenterWrapper, sample_mask_vision_result, "segment"),
            (ClassifierWrapper, sample_classify_result, "classify"),
            (DepthWrapper, sample_depth_result, "estimate"),
            (SAMWrapper, sample_mask_vision_result, "segment"),
        ]

        for wrapper_cls, result, method_name in wrappers_and_results:
            adapter = _make_mock_adapter(result)
            wrapper = wrapper_cls(adapter)
            method = getattr(wrapper, method_name)
            method(sample_image_with_path)
            img_arg = adapter.predict.call_args[0][0]
            assert isinstance(img_arg, str), f"{wrapper_cls.__name__}: expected str, got {type(img_arg)}"

    def test_image_pil_fallback_all_wrappers(
        self,
        sample_image_artifact,
        sample_vision_result,
        sample_mask_vision_result,
        sample_classify_result,
        sample_depth_result,
        sample_vlm_result,
    ):
        """All wrappers fall back to PIL when no source_path."""
        wrappers_and_results = [
            (DetectorWrapper, sample_vision_result, "predict"),
            (SegmenterWrapper, sample_mask_vision_result, "segment"),
            (ClassifierWrapper, sample_classify_result, "classify"),
            (DepthWrapper, sample_depth_result, "estimate"),
            (SAMWrapper, sample_mask_vision_result, "segment"),
        ]

        for wrapper_cls, result, method_name in wrappers_and_results:
            adapter = _make_mock_adapter(result)
            wrapper = wrapper_cls(adapter)
            method = getattr(wrapper, method_name)
            method(sample_image_artifact)
            img_arg = adapter.predict.call_args[0][0]
            assert isinstance(
                img_arg, PILImage.Image
            ), f"{wrapper_cls.__name__}: expected PIL.Image, got {type(img_arg)}"

    def test_all_wrappers_reject_non_image(
        self,
        sample_vision_result,
        sample_mask_vision_result,
        sample_classify_result,
        sample_depth_result,
        sample_vlm_result,
    ):
        """All wrappers reject non-Image inputs with TypeError."""
        wrappers_and_results = [
            (DetectorWrapper, sample_vision_result, "predict"),
            (SegmenterWrapper, sample_mask_vision_result, "segment"),
            (ClassifierWrapper, sample_classify_result, "classify"),
            (DepthWrapper, sample_depth_result, "estimate"),
            (SAMWrapper, sample_mask_vision_result, "segment"),
        ]

        for wrapper_cls, result, method_name in wrappers_and_results:
            adapter = _make_mock_adapter(result)
            wrapper = wrapper_cls(adapter)
            method = getattr(wrapper, method_name)
            with pytest.raises(TypeError, match="Expected Image artifact"):
                method("not_an_image.jpg")

    def test_all_wrappers_reject_no_predict_adapter(self):
        """All wrappers reject adapters without predict()."""
        wrapper_classes = [
            DetectorWrapper,
            SegmenterWrapper,
            ClassifierWrapper,
            DepthWrapper,
            SAMWrapper,
            VLMWrapper,
        ]
        for cls in wrapper_classes:
            with pytest.raises(TypeError, match="does not have a predict"):
                cls(object())

    def test_multiple_adapter_types_per_task(self, sample_image_artifact, sample_vision_result):
        """DetectorWrapper works with different adapter type names."""
        # Simulate different adapter types
        for adapter_name in ["HuggingFaceDetect", "ONNXDetect", "TorchScriptDetect"]:
            adapter = _make_mock_adapter(sample_vision_result)
            type(adapter).__name__ = adapter_name
            wrapper = DetectorWrapper(adapter)
            result = wrapper.predict(sample_image_artifact)
            assert isinstance(result, Detections)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for all factory (wrap_*) functions."""

    def test_wrap_detector(self, sample_vision_result):
        adapter = _make_mock_adapter(sample_vision_result)
        wrapper = wrap_detector(adapter)
        assert isinstance(wrapper, DetectorWrapper)

    def test_wrap_segmenter(self, sample_mask_vision_result):
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = wrap_segmenter(adapter)
        assert isinstance(wrapper, SegmenterWrapper)

    def test_wrap_classifier(self, sample_classify_result):
        adapter = _make_mock_adapter(sample_classify_result)
        wrapper = wrap_classifier(adapter)
        assert isinstance(wrapper, ClassifierWrapper)

    def test_wrap_depth(self, sample_depth_result):
        adapter = _make_mock_adapter(sample_depth_result)
        wrapper = wrap_depth(adapter)
        assert isinstance(wrapper, DepthWrapper)

    def test_wrap_sam(self, sample_mask_vision_result):
        adapter = _make_mock_adapter(sample_mask_vision_result)
        wrapper = wrap_sam(adapter)
        assert isinstance(wrapper, SAMWrapper)

    def test_wrap_vlm(self, sample_vlm_result):
        adapter = _make_mock_adapter(sample_vlm_result)
        wrapper = wrap_vlm(adapter)
        assert isinstance(wrapper, VLMWrapper)
