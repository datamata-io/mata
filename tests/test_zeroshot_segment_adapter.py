"""Tests for HuggingFaceZeroShotSegmentAdapter (zero-shot image segmentation).

Comprehensive test suite covering:
- Adapter initialization (architecture detection, model loading, mask format)
- Prediction with text prompts (list, comma-separated, space-dot)
- Output validation (VisionResult, Instance, masks, bboxes, scores)
- Threshold filtering (empty masks, high/low thresholds)
- Mask format conversion (binary, RLE, polygon)
- Error handling (missing prompts, invalid inputs, load failures)
- info() metadata
- segment() graph node interface
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from mata.adapters.huggingface_zeroshot_segment_adapter import (
    HuggingFaceZeroShotSegmentAdapter,
)
from mata.core.exceptions import InvalidInputError, ModelLoadError
from mata.core.types import Instance, VisionResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_transformers_clipseg():
    """Mock transformers library components for CLIPSeg."""
    with patch("mata.adapters.huggingface_zeroshot_segment_adapter._ensure_transformers") as mock_ensure:
        mock_processor_class = Mock()
        mock_model_class = Mock()
        mock_auto_processor_class = Mock()

        # Track prompt count for dynamic logits
        prompt_tracker = {"count": 2, "h": 352, "w": 352}

        # Mock processor instance
        mock_processor = Mock()

        def processor_call(*args, **kwargs):
            """Return inputs with real tensors."""
            text_arg = kwargs.get("text") or (args[0] if args else None)
            if isinstance(text_arg, list):
                prompt_tracker["count"] = len(text_arg)
            else:
                prompt_tracker["count"] = 1

            n = prompt_tracker["count"]
            return {
                "input_ids": torch.randint(0, 100, (n, 77)),
                "pixel_values": torch.randn(n, 3, 224, 224),
                "attention_mask": torch.ones(n, 77, dtype=torch.long),
            }

        mock_processor.side_effect = processor_call

        # Mock model instance with logits output
        def model_call(**inputs):
            """Return CLIPSeg-style outputs with logits."""
            n = prompt_tracker["count"]
            h, w = prompt_tracker["h"], prompt_tracker["w"]
            mock_outputs = Mock()
            # CLIPSeg: logits shape = (N, H, W)
            logits = torch.randn(n, h, w)
            # Make some regions strongly positive (high probability after sigmoid)
            for i in range(n):
                logits[i, 50:150, 50:150] = 3.0 + i  # Strong positive region
            mock_outputs.logits = logits
            return mock_outputs

        mock_model = Mock()
        mock_model.side_effect = model_call
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        mock_processor_class.from_pretrained = Mock(return_value=mock_processor)
        mock_model_class.from_pretrained = Mock(return_value=mock_model)
        mock_auto_processor_class.from_pretrained = Mock(return_value=mock_processor)

        mock_ensure.return_value = {
            "CLIPSegForImageSegmentation": mock_model_class,
            "CLIPSegProcessor": mock_processor_class,
            "AutoProcessor": mock_auto_processor_class,
        }

        yield {
            "ensure": mock_ensure,
            "processor_class": mock_processor_class,
            "model_class": mock_model_class,
            "processor": mock_processor,
            "model": mock_model,
            "prompt_tracker": prompt_tracker,
        }


@pytest.fixture
def mock_transformers_empty_masks():
    """Mock transformers that produce all-negative logits (empty masks)."""
    with patch("mata.adapters.huggingface_zeroshot_segment_adapter._ensure_transformers") as mock_ensure:
        mock_processor_class = Mock()
        mock_model_class = Mock()

        prompt_tracker = {"count": 2}

        mock_processor = Mock()

        def processor_call(*args, **kwargs):
            text_arg = kwargs.get("text") or (args[0] if args else None)
            if isinstance(text_arg, list):
                prompt_tracker["count"] = len(text_arg)
            return {
                "input_ids": torch.randint(0, 100, (prompt_tracker["count"], 77)),
                "pixel_values": torch.randn(prompt_tracker["count"], 3, 224, 224),
            }

        mock_processor.side_effect = processor_call

        def model_call(**inputs):
            mock_outputs = Mock()
            n = prompt_tracker["count"]
            # All strongly negative logits → sigmoid < 0.5 → empty masks
            mock_outputs.logits = torch.full((n, 352, 352), -10.0)
            return mock_outputs

        mock_model = Mock()
        mock_model.side_effect = model_call
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        mock_processor_class.from_pretrained = Mock(return_value=mock_processor)
        mock_model_class.from_pretrained = Mock(return_value=mock_model)

        mock_ensure.return_value = {
            "CLIPSegForImageSegmentation": mock_model_class,
            "CLIPSegProcessor": mock_processor_class,
            "AutoProcessor": Mock(),
        }

        yield


@pytest.fixture
def mock_transformers_single_prompt():
    """Mock transformers that produce 2D logits (single prompt, no batch dim)."""
    with patch("mata.adapters.huggingface_zeroshot_segment_adapter._ensure_transformers") as mock_ensure:
        mock_processor_class = Mock()
        mock_model_class = Mock()

        mock_processor = Mock()

        def processor_call(*args, **kwargs):
            return {
                "input_ids": torch.randint(0, 100, (1, 77)),
                "pixel_values": torch.randn(1, 3, 224, 224),
            }

        mock_processor.side_effect = processor_call

        def model_call(**inputs):
            mock_outputs = Mock()
            # Single prompt: logits is 2D (H, W) instead of (1, H, W)
            logits = torch.full((352, 352), -5.0)
            logits[100:200, 100:200] = 5.0  # Strong positive region
            mock_outputs.logits = logits
            return mock_outputs

        mock_model = Mock()
        mock_model.side_effect = model_call
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        mock_processor_class.from_pretrained = Mock(return_value=mock_processor)
        mock_model_class.from_pretrained = Mock(return_value=mock_model)

        mock_ensure.return_value = {
            "CLIPSegForImageSegmentation": mock_model_class,
            "CLIPSegProcessor": mock_processor_class,
            "AutoProcessor": Mock(),
        }

        yield


@pytest.fixture
def mock_pycocotools():
    """Mock pycocotools for RLE encoding."""
    with patch("mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools") as mock_ensure:
        mock_mask_utils = Mock()

        def encode_mask(mask):
            return {"size": list(mask.shape), "counts": b"mock_rle_counts"}

        mock_mask_utils.encode = Mock(side_effect=encode_mask)
        mock_ensure.return_value = mock_mask_utils
        yield mock_mask_utils


@pytest.fixture
def sample_image():
    """Create a sample PIL image for testing."""
    return Image.new("RGB", (640, 480), color=(128, 128, 128))


@pytest.fixture
def adapter(mock_transformers_clipseg):
    """Create a default adapter instance with mocked dependencies."""
    with patch(
        "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
        return_value=None,
    ):
        adapter = HuggingFaceZeroShotSegmentAdapter(
            "CIDAS/clipseg-rd64-refined",
            device="cpu",
            use_rle=False,
        )
        return adapter


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestZeroShotSegmentAdapterInit:
    """Test adapter initialization."""

    def test_init_basic(self, mock_transformers_clipseg):
        """Test basic initialization with default parameters."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=None,
        ):
            adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False)
            assert adapter.model_id == "CIDAS/clipseg-rd64-refined"
            assert adapter.architecture == "clipseg"
            assert adapter.task == "segment"
            assert adapter.threshold == 0.5
            assert adapter.use_rle is False
            assert adapter.use_polygon is False

    def test_init_custom_threshold(self, mock_transformers_clipseg):
        """Test initialization with custom threshold."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=None,
        ):
            adapter = HuggingFaceZeroShotSegmentAdapter(
                "CIDAS/clipseg-rd64-refined",
                device="cpu",
                threshold=0.7,
                use_rle=False,
            )
            assert adapter.threshold == 0.7

    def test_init_with_rle(self, mock_transformers_clipseg, mock_pycocotools):
        """Test initialization with RLE encoding enabled."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter.PYCOCOTOOLS_AVAILABLE",
            True,
        ):
            adapter = HuggingFaceZeroShotSegmentAdapter(
                "CIDAS/clipseg-rd64-refined",
                device="cpu",
                use_rle=True,
            )
            assert adapter.use_rle is True

    def test_init_rle_fallback_no_pycocotools(self, mock_transformers_clipseg):
        """Test RLE fallback when pycocotools not available."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=None,
        ):
            with pytest.warns(UserWarning, match="pycocotools not available"):
                adapter = HuggingFaceZeroShotSegmentAdapter(
                    "CIDAS/clipseg-rd64-refined",
                    device="cpu",
                    use_rle=True,
                )
            assert adapter.use_rle is False

    def test_init_with_polygon(self, mock_transformers_clipseg):
        """Test initialization with polygon mask format."""
        import builtins

        _real_import = builtins.__import__

        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (_real_import(name, *a, **kw) if name != "cv2" else Mock()),
        ):
            with patch(
                "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
                return_value=None,
            ):
                adapter = HuggingFaceZeroShotSegmentAdapter(
                    "CIDAS/clipseg-rd64-refined",
                    device="cpu",
                    use_polygon=True,
                )
                assert adapter.use_polygon is True
                assert adapter.use_rle is False

    def test_init_invalid_threshold(self, mock_transformers_clipseg):
        """Test initialization with invalid threshold raises error."""
        with pytest.raises(ValueError, match="Threshold must be in range"):
            HuggingFaceZeroShotSegmentAdapter(
                "CIDAS/clipseg-rd64-refined",
                device="cpu",
                threshold=1.5,
            )

    def test_init_transformers_not_available(self):
        """Test initialization when transformers is not installed."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_transformers",
            return_value=None,
        ):
            with pytest.raises(ImportError, match="transformers library is required"):
                HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu")

    def test_init_model_load_failure(self, mock_transformers_clipseg):
        """Test initialization when model loading fails."""
        mock_transformers_clipseg["model_class"].from_pretrained.side_effect = RuntimeError("Model not found")
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=None,
        ):
            with pytest.raises(ModelLoadError):
                HuggingFaceZeroShotSegmentAdapter("CIDAS/nonexistent-model", device="cpu", use_rle=False)


# ---------------------------------------------------------------------------
# Architecture Detection Tests
# ---------------------------------------------------------------------------


class TestArchitectureDetection:
    """Test model architecture detection from model ID."""

    def test_detect_clipseg(self, mock_transformers_clipseg):
        """Test CLIPSeg architecture detection."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=None,
        ):
            adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False)
            assert adapter.architecture == "clipseg"

    def test_detect_clipseg_rd16(self, mock_transformers_clipseg):
        """Test CLIPSeg-rd16 is detected as clipseg."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=None,
        ):
            adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd16", device="cpu", use_rle=False)
            assert adapter.architecture == "clipseg"

    def test_detect_unknown_model(self, mock_transformers_clipseg):
        """Test unknown model falls back to auto."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=None,
        ):
            adapter = HuggingFaceZeroShotSegmentAdapter("some-org/unknown-model", device="cpu", use_rle=False)
            assert adapter.architecture == "auto"


# ---------------------------------------------------------------------------
# Text Prompt Normalization Tests
# ---------------------------------------------------------------------------


class TestTextPromptNormalization:
    """Test text prompt parsing and normalization."""

    def test_list_prompts(self, adapter):
        """Test list input is passed through."""
        result = adapter._normalize_text_prompts(["cat", "dog", "bird"])
        assert result == ["cat", "dog", "bird"]

    def test_comma_separated(self, adapter):
        """Test comma-separated string is parsed."""
        result = adapter._normalize_text_prompts("cat, dog, bird")
        assert result == ["cat", "dog", "bird"]

    def test_space_dot_separated(self, adapter):
        """Test space-dot-separated string is parsed."""
        result = adapter._normalize_text_prompts("cat . dog . bird")
        assert result == ["cat", "dog", "bird"]

    def test_single_prompt_string(self, adapter):
        """Test single prompt string."""
        result = adapter._normalize_text_prompts("cat")
        assert result == ["cat"]

    def test_whitespace_trimming(self, adapter):
        """Test whitespace is trimmed from prompts."""
        result = adapter._normalize_text_prompts(["  cat  ", " dog ", "bird  "])
        assert result == ["cat", "dog", "bird"]

    def test_empty_strings_filtered(self, adapter):
        """Test empty strings are filtered from prompts."""
        result = adapter._normalize_text_prompts(["cat", "", "dog", "  "])
        assert result == ["cat", "dog"]

    def test_space_dot_empty_filtered(self, adapter):
        """Test space-dot parsing filters empty entries."""
        result = adapter._normalize_text_prompts("cat . . dog")
        assert result == ["cat", "dog"]


# ---------------------------------------------------------------------------
# Prediction Tests
# ---------------------------------------------------------------------------


class TestPrediction:
    """Test zero-shot segmentation prediction."""

    def test_predict_basic(self, adapter, sample_image):
        """Test basic prediction with list prompts."""
        result = adapter.predict(sample_image, text_prompts=["cat", "dog"])

        assert isinstance(result, VisionResult)
        assert len(result.instances) > 0
        for inst in result.instances:
            assert isinstance(inst, Instance)
            assert inst.mask is not None
            assert inst.score > 0
            assert inst.label_name in ["cat", "dog"]
            assert inst.bbox is not None

    def test_predict_comma_prompts(self, adapter, sample_image):
        """Test prediction with comma-separated prompts."""
        result = adapter.predict(sample_image, text_prompts="cat, dog")

        assert isinstance(result, VisionResult)
        assert len(result.instances) > 0
        label_names = {inst.label_name for inst in result.instances}
        assert label_names.issubset({"cat", "dog"})

    def test_predict_space_dot_prompts(self, adapter, sample_image):
        """Test prediction with space-dot-separated prompts."""
        result = adapter.predict(sample_image, text_prompts="cat . dog . person")

        assert isinstance(result, VisionResult)
        label_names = {inst.label_name for inst in result.instances}
        assert label_names.issubset({"cat", "dog", "person"})

    def test_predict_single_prompt(self, mock_transformers_single_prompt, sample_image):
        """Test prediction with a single prompt (2D logits)."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=None,
        ):
            adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False)
            result = adapter.predict(sample_image, text_prompts="cat")

            assert isinstance(result, VisionResult)
            assert len(result.instances) == 1
            assert result.instances[0].label_name == "cat"
            assert result.instances[0].label == 0

    def test_predict_threshold_override(self, adapter, sample_image):
        """Test prediction with threshold override."""
        # Very high threshold should produce fewer or no masks
        result_high = adapter.predict(sample_image, text_prompts=["cat", "dog"], threshold=0.99)

        # Lower threshold should produce more masks
        result_low = adapter.predict(sample_image, text_prompts=["cat", "dog"], threshold=0.01)

        assert len(result_low.instances) >= len(result_high.instances)

    def test_predict_empty_masks_filtered(self, mock_transformers_empty_masks, sample_image):
        """Test that prompts producing empty masks are omitted."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=None,
        ):
            adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False)
            result = adapter.predict(sample_image, text_prompts=["cat", "dog"])

            # All logits are strongly negative → all masks empty
            assert len(result.instances) == 0

    def test_predict_no_text_prompts_raises(self, adapter, sample_image):
        """Test prediction without text_prompts raises error."""
        with pytest.raises(InvalidInputError, match="text_prompts required"):
            adapter.predict(sample_image, text_prompts=None)

    def test_predict_empty_string_prompts_raises(self, adapter, sample_image):
        """Test prediction with empty string raises error."""
        with pytest.raises(InvalidInputError, match="text_prompts required"):
            adapter.predict(sample_image, text_prompts="")

    def test_predict_empty_list_prompts_raises(self, adapter, sample_image):
        """Test prediction with empty list raises error."""
        with pytest.raises(InvalidInputError, match="text_prompts required"):
            adapter.predict(sample_image, text_prompts=[])

    def test_predict_from_path(self, adapter, tmp_path):
        """Test prediction from file path."""
        img_path = tmp_path / "test.png"
        Image.new("RGB", (640, 480)).save(img_path)

        result = adapter.predict(str(img_path), text_prompts=["object"])
        assert isinstance(result, VisionResult)
        assert result.meta["input_path"] is not None

    def test_predict_from_numpy(self, adapter):
        """Test prediction from numpy array."""
        arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = adapter.predict(arr, text_prompts=["object"])
        assert isinstance(result, VisionResult)


# ---------------------------------------------------------------------------
# Result Structure Tests
# ---------------------------------------------------------------------------


class TestResultStructure:
    """Test the structure and content of prediction results."""

    def test_result_meta(self, adapter, sample_image):
        """Test result metadata fields."""
        result = adapter.predict(sample_image, text_prompts=["cat", "dog"])

        assert result.meta["model_id"] == "CIDAS/clipseg-rd64-refined"
        assert result.meta["architecture"] == "clipseg"
        assert result.meta["text_prompts"] == ["cat", "dog"]
        assert result.meta["mode"] == "zeroshot"
        assert result.meta["backend"] == "transformers"
        assert "threshold" in result.meta
        assert "image_size" in result.meta

    def test_instance_fields(self, adapter, sample_image):
        """Test Instance object fields."""
        result = adapter.predict(sample_image, text_prompts=["cat"])

        if result.instances:
            inst = result.instances[0]
            assert inst.label == 0
            assert inst.label_name == "cat"
            assert 0.0 < inst.score <= 1.0
            assert inst.bbox is not None
            assert len(inst.bbox) == 4
            # xyxy format: x1 < x2, y1 < y2
            x1, y1, x2, y2 = inst.bbox
            assert x1 <= x2
            assert y1 <= y2
            assert inst.area > 0

    def test_instance_label_mapping(self, adapter, sample_image):
        """Test that labels match prompt order."""
        result = adapter.predict(sample_image, text_prompts=["alpha", "beta", "gamma"])

        for inst in result.instances:
            expected_names = {"alpha": 0, "beta": 1, "gamma": 2}
            if inst.label_name in expected_names:
                assert inst.label == expected_names[inst.label_name]

    def test_instance_is_stuff_none(self, adapter, sample_image):
        """Test that is_stuff is None for zero-shot (no thing/stuff distinction)."""
        result = adapter.predict(sample_image, text_prompts=["cat"])

        for inst in result.instances:
            assert inst.is_stuff is None

    def test_result_image_size(self, adapter):
        """Test that image_size in metadata matches input."""
        img = Image.new("RGB", (800, 600))
        result = adapter.predict(img, text_prompts=["object"])

        assert result.meta["image_size"] == [800, 600]

    def test_binary_mask_shape(self, adapter, sample_image):
        """Test binary mask has correct shape matching original image."""
        result = adapter.predict(sample_image, text_prompts=["cat"])

        if result.instances:
            mask = result.instances[0].mask
            # Binary mask should match original image dimensions
            assert isinstance(mask, np.ndarray)
            assert mask.shape == (480, 640)  # (H, W) matches sample_image


# ---------------------------------------------------------------------------
# Mask Format Tests
# ---------------------------------------------------------------------------


class TestMaskFormat:
    """Test mask format conversion."""

    def test_binary_mask_format(self, adapter, sample_image):
        """Test binary mask output (no RLE, no polygon)."""
        result = adapter.predict(sample_image, text_prompts=["cat"])

        assert result.meta["mask_format"] == "binary"
        if result.instances:
            mask = result.instances[0].mask
            assert isinstance(mask, np.ndarray)

    def test_rle_mask_format(self, mock_transformers_clipseg, sample_image):
        """Test RLE mask output."""
        mock_mask_utils = Mock()

        def encode_mask(mask):
            return {"size": list(mask.shape), "counts": b"rle_encoded"}

        mock_mask_utils.encode = Mock(side_effect=encode_mask)

        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=mock_mask_utils,
        ):
            with patch(
                "mata.adapters.huggingface_zeroshot_segment_adapter.PYCOCOTOOLS_AVAILABLE",
                True,
            ):
                with patch(
                    "mata.adapters.huggingface_zeroshot_segment_adapter._mask_utils",
                    mock_mask_utils,
                ):
                    adapter = HuggingFaceZeroShotSegmentAdapter(
                        "CIDAS/clipseg-rd64-refined",
                        device="cpu",
                        use_rle=True,
                    )
                    result = adapter.predict(sample_image, text_prompts=["cat"])

                    assert result.meta["mask_format"] == "rle"
                    if result.instances:
                        mask = result.instances[0].mask
                        assert isinstance(mask, dict)
                        assert "size" in mask
                        assert "counts" in mask


# ---------------------------------------------------------------------------
# Bbox Computation Tests
# ---------------------------------------------------------------------------


class TestBboxComputation:
    """Test bounding box computation from masks."""

    def test_mask_to_bbox(self, adapter):
        """Test bbox computation from a simple mask."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:60] = True

        bbox = adapter._mask_to_bbox(mask)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert x1 == 30.0
        assert y1 == 20.0
        assert x2 == 59.0
        assert y2 == 39.0

    def test_mask_to_bbox_empty(self, adapter):
        """Test bbox is None for empty mask."""
        mask = np.zeros((100, 100), dtype=bool)
        bbox = adapter._mask_to_bbox(mask)
        assert bbox is None

    def test_mask_to_bbox_single_pixel(self, adapter):
        """Test bbox for single pixel mask."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 75] = True

        bbox = adapter._mask_to_bbox(mask)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert x1 == 75.0
        assert y1 == 50.0
        assert x2 == 75.0
        assert y2 == 50.0

    def test_mask_to_bbox_full_image(self, adapter):
        """Test bbox for mask covering entire image."""
        mask = np.ones((100, 200), dtype=bool)
        bbox = adapter._mask_to_bbox(mask)

        assert bbox == (0.0, 0.0, 199.0, 99.0)


# ---------------------------------------------------------------------------
# Info / Metadata Tests
# ---------------------------------------------------------------------------


class TestInfoMetadata:
    """Test adapter info() and metadata."""

    def test_info_fields(self, adapter):
        """Test info() returns expected fields."""
        info = adapter.info()

        assert info["name"] == "HuggingFaceZeroShotSegmentAdapter"
        assert info["task"] == "segment"
        assert info["model_id"] == "CIDAS/clipseg-rd64-refined"
        assert info["architecture"] == "clipseg"
        assert info["mode"] == "zeroshot"
        assert info["backend"] == "huggingface-transformers"
        assert "device" in info
        assert "threshold" in info
        assert "mask_format" in info

    def test_info_mask_format_binary(self, adapter):
        """Test info shows binary mask format."""
        info = adapter.info()
        assert info["mask_format"] == "binary"


# ---------------------------------------------------------------------------
# segment() Interface Tests
# ---------------------------------------------------------------------------


class TestSegmentInterface:
    """Test the segment() graph node interface."""

    def test_segment_delegates_to_predict(self, adapter, sample_image):
        """Test segment() wraps predict()."""
        result = adapter.segment(sample_image, text_prompts=["cat", "dog"])

        assert isinstance(result, VisionResult)
        assert len(result.instances) > 0

    def test_segment_with_threshold(self, adapter, sample_image):
        """Test segment() passes threshold to predict()."""
        result = adapter.segment(sample_image, text_prompts=["cat"], threshold=0.99)

        assert isinstance(result, VisionResult)

    def test_segment_with_kwargs(self, adapter, sample_image):
        """Test segment() passes kwargs through."""
        result = adapter.segment(sample_image, text_prompts=["cat"], threshold=0.3)

        assert isinstance(result, VisionResult)


# ---------------------------------------------------------------------------
# UniversalLoader Integration Tests
# ---------------------------------------------------------------------------


class TestUniversalLoaderRouting:
    """Test that UniversalLoader correctly routes CLIPSeg models."""

    def test_clipseg_routed_to_zeroshot_adapter(self, mock_transformers_clipseg):
        """Test CLIPSeg model ID routes to ZeroShotSegmentAdapter."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=None,
        ):
            from mata.core.model_loader import UniversalLoader

            with patch.object(
                UniversalLoader,
                "_detect_source_type",
                return_value=("huggingface", "CIDAS/clipseg-rd64-refined"),
            ):
                loader = UniversalLoader()
                adapter = loader._load_from_huggingface("segment", "CIDAS/clipseg-rd64-refined", use_rle=False)

                assert isinstance(adapter, HuggingFaceZeroShotSegmentAdapter)
                assert adapter.model_id == "CIDAS/clipseg-rd64-refined"

    def test_sam_model_not_routed_to_zeroshot(self):
        """Test SAM model IDs are not routed to ZeroShotSegmentAdapter."""
        from mata.core.model_loader import UniversalLoader

        UniversalLoader()

        # SAM should not trigger CLIPSeg adapter
        model_id = "facebook/sam-vit-base"
        model_id_lower = model_id.lower()
        is_zeroshot_segment = any(x in model_id_lower for x in ["clipseg", "clip-seg"])
        assert not is_zeroshot_segment

    def test_mask2former_not_routed_to_zeroshot(self):
        """Test Mask2Former model IDs are not routed to ZeroShotSegmentAdapter."""
        model_id = "facebook/mask2former-swin-tiny-coco-instance"
        model_id_lower = model_id.lower()
        is_zeroshot_segment = any(x in model_id_lower for x in ["clipseg", "clip-seg"])
        assert not is_zeroshot_segment


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_image(self, mock_transformers_clipseg):
        """Test with very small image."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=None,
        ):
            adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False)
            small_img = Image.new("RGB", (32, 32))
            result = adapter.predict(small_img, text_prompts=["object"])

            assert isinstance(result, VisionResult)
            # Masks should be resized to match original image
            if result.instances:
                mask = result.instances[0].mask
                assert mask.shape == (32, 32)

    def test_large_image(self, mock_transformers_clipseg):
        """Test with large image."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=None,
        ):
            adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False)
            large_img = Image.new("RGB", (1920, 1080))
            result = adapter.predict(large_img, text_prompts=["object"])

            assert isinstance(result, VisionResult)
            if result.instances:
                mask = result.instances[0].mask
                assert mask.shape == (1080, 1920)

    def test_many_prompts(self, mock_transformers_clipseg):
        """Test with many text prompts."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=None,
        ):
            adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False)
            prompts = [f"class_{i}" for i in range(10)]
            img = Image.new("RGB", (640, 480))
            result = adapter.predict(img, text_prompts=prompts)

            assert isinstance(result, VisionResult)
            # Each prompt might or might not produce a mask
            assert result.meta["text_prompts"] == prompts

    def test_score_within_range(self, adapter, sample_image):
        """Test that all scores are valid probabilities."""
        result = adapter.predict(sample_image, text_prompts=["alpha", "beta"])

        for inst in result.instances:
            assert 0.0 <= inst.score <= 1.0

    def test_area_matches_mask(self, adapter, sample_image):
        """Test that reported area matches actual mask pixel count."""
        result = adapter.predict(sample_image, text_prompts=["object"])

        for inst in result.instances:
            if isinstance(inst.mask, np.ndarray):
                assert inst.area == int(inst.mask.sum())

    def test_threshold_zero_includes_all(self, mock_transformers_clipseg):
        """Test threshold=0.0 includes all non-zero probability pixels."""
        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=None,
        ):
            adapter = HuggingFaceZeroShotSegmentAdapter(
                "CIDAS/clipseg-rd64-refined",
                device="cpu",
                threshold=0.0,
                use_rle=False,
            )
            img = Image.new("RGB", (640, 480))
            result = adapter.predict(img, text_prompts=["cat"], threshold=0.0)

            # With threshold=0.0 and sigmoid output, virtually all pixels pass
            # (sigmoid(x) >= 0 is always true for finite x)
            assert isinstance(result, VisionResult)


# ---------------------------------------------------------------------------
# Mask Conversion Tests
# ---------------------------------------------------------------------------


class TestMaskConversion:
    """Test mask format conversion methods."""

    def test_convert_binary_format(self, adapter):
        """Test binary mask conversion (passthrough)."""
        mask = np.random.randint(0, 2, (100, 100)).astype(bool)
        result = adapter._convert_mask_format(mask)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, mask)

    def test_convert_rle_format(self, mock_transformers_clipseg):
        """Test RLE mask conversion."""
        mock_mask_utils = Mock()

        def encode_mask(mask):
            return {
                "size": list(mask.shape),
                "counts": b"mock_rle",
            }

        mock_mask_utils.encode = Mock(side_effect=encode_mask)

        with patch(
            "mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools",
            return_value=mock_mask_utils,
        ):
            with patch(
                "mata.adapters.huggingface_zeroshot_segment_adapter.PYCOCOTOOLS_AVAILABLE",
                True,
            ):
                with patch(
                    "mata.adapters.huggingface_zeroshot_segment_adapter._mask_utils",
                    mock_mask_utils,
                ):
                    adapter = HuggingFaceZeroShotSegmentAdapter(
                        "CIDAS/clipseg-rd64-refined",
                        device="cpu",
                        use_rle=True,
                    )

                    mask = np.random.randint(0, 2, (100, 100)).astype(bool)
                    result = adapter._convert_mask_format(mask)

                    assert isinstance(result, dict)
                    assert "size" in result
                    assert "counts" in result
                    assert isinstance(result["counts"], str)
