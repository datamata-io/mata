"""Tests for HuggingFaceSAMAdapter (zero-shot segmentation)."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from mata.adapters.huggingface_sam_adapter import HuggingFaceSAMAdapter
from mata.core.exceptions import InvalidInputError, ModelLoadError
from mata.core.types import Instance, VisionResult


@pytest.fixture
def mock_transformers():
    """Mock transformers library components."""
    with patch("mata.adapters.huggingface_sam_adapter._ensure_transformers") as mock_ensure:
        # Create mock classes
        mock_sam_model = Mock()
        mock_sam_processor = Mock()

        # Mock model instance
        mock_model_instance = Mock()
        mock_model_instance.to = Mock(return_value=mock_model_instance)
        mock_model_instance.eval = Mock(return_value=mock_model_instance)

        # Mock processor instance
        mock_processor_instance = Mock()

        # Set up from_pretrained returns
        mock_sam_model.from_pretrained = Mock(return_value=mock_model_instance)
        mock_sam_processor.from_pretrained = Mock(return_value=mock_processor_instance)

        # Return dict with mocked classes
        mock_ensure.return_value = {
            "SamModel": mock_sam_model,
            "SamProcessor": mock_sam_processor,
        }

        yield {
            "ensure": mock_ensure,
            "model_class": mock_sam_model,
            "processor_class": mock_sam_processor,
            "model_instance": mock_model_instance,
            "processor_instance": mock_processor_instance,
        }


@pytest.fixture
def mock_pycocotools():
    """Mock pycocotools for RLE encoding."""
    with patch("mata.adapters.huggingface_sam_adapter._ensure_pycocotools") as mock_ensure:
        mock_mask_utils = Mock()

        # Mock encode function to return RLE dict
        def mock_encode(mask):
            return {"size": [mask.shape[0], mask.shape[1]], "counts": b"mock_rle_data"}

        mock_mask_utils.encode = Mock(side_effect=mock_encode)
        mock_ensure.return_value = mock_mask_utils

        yield mock_mask_utils


class TestSAMAdapterInitialization:
    """Test SAM adapter initialization."""

    def test_init_basic(self, mock_transformers, mock_pycocotools):
        """Test basic SAM adapter initialization."""
        adapter = HuggingFaceSAMAdapter(model_id="facebook/sam-vit-base", device="cpu")

        assert adapter.model_id == "facebook/sam-vit-base"
        # RLE state depends on pycocotools mock state
        # In test environment, pycocotools is mocked successfully so RLE=True
        assert adapter.use_rle is True
        assert adapter.threshold == 0.0  # Default for SAM
        assert adapter.id2label == {0: "object"}  # Class-agnostic

        # Verify model and processor were loaded (with token=None by default)
        mock_transformers["model_class"].from_pretrained.assert_called_once_with("facebook/sam-vit-base", token=None)
        mock_transformers["processor_class"].from_pretrained.assert_called_once_with(
            "facebook/sam-vit-base", token=None
        )

    def test_init_with_custom_threshold(self, mock_transformers):
        """Test initialization with custom threshold."""
        adapter = HuggingFaceSAMAdapter(model_id="facebook/sam-vit-large", threshold=0.8, device="cpu")

        assert adapter.threshold == 0.8

    def test_init_without_rle(self, mock_transformers):
        """Test initialization without RLE encoding."""
        adapter = HuggingFaceSAMAdapter(model_id="facebook/sam-vit-base", use_rle=False, device="cpu")

        assert adapter.use_rle is False

    def test_init_model_not_found(self, mock_transformers):
        """Test initialization with model loading failure."""
        mock_transformers["model_class"].from_pretrained.side_effect = Exception("Model not found")

        with pytest.raises(ModelLoadError) as exc_info:
            HuggingFaceSAMAdapter(model_id="invalid/model", device="cpu")

        assert "Failed to load SAM model" in str(exc_info.value)

    def test_init_warns_non_sam_model(self, mock_transformers):
        """Test warning when model ID doesn't contain 'sam'."""
        with pytest.warns(UserWarning, match="does not contain 'sam'"):
            HuggingFaceSAMAdapter(model_id="facebook/some-other-model", device="cpu")

    def test_init_with_token(self, mock_transformers, mock_pycocotools):
        """Test initialization with HuggingFace token for gated models."""
        # Add SAM3 classes to mock
        mock_sam3_model = Mock()
        mock_sam3_processor = Mock()

        mock_model_instance = Mock()
        mock_model_instance.to = Mock(return_value=mock_model_instance)
        mock_model_instance.eval = Mock(return_value=mock_model_instance)
        mock_processor_instance = Mock()

        mock_sam3_model.from_pretrained = Mock(return_value=mock_model_instance)
        mock_sam3_processor.from_pretrained = Mock(return_value=mock_processor_instance)

        # Add SAM3 classes to transformers mock
        mock_transformers["ensure"].return_value.update(
            {
                "Sam3Model": mock_sam3_model,
                "Sam3Processor": mock_sam3_processor,
            }
        )

        adapter = HuggingFaceSAMAdapter(model_id="facebook/sam3", device="cpu", token="hf_test_token_12345")

        # Verify token was passed to from_pretrained calls
        mock_sam3_model.from_pretrained.assert_called_once_with("facebook/sam3", token="hf_test_token_12345")
        mock_sam3_processor.from_pretrained.assert_called_once_with("facebook/sam3", token="hf_test_token_12345")
        assert adapter.model_id == "facebook/sam3"


class TestSAMAdapterPrediction:
    """Test SAM adapter prediction with prompts."""

    @pytest.fixture
    def adapter(self, mock_transformers, mock_pycocotools):
        """Create SAM adapter for testing."""
        adapter = HuggingFaceSAMAdapter(model_id="facebook/sam-vit-base", device="cpu", threshold=0.0)

        # Store mocks for later use
        adapter._mock_transformers = mock_transformers
        adapter._mock_pycocotools = mock_pycocotools

        return adapter

    def test_predict_point_prompts(self, adapter):
        """Test prediction with point prompts."""
        # Create test image
        test_image = Image.new("RGB", (640, 480), color="red")

        # Mock processor output
        mock_inputs = {
            "pixel_values": Mock(spec=["to"]),
            "input_points": [[100.0, 150.0]],
            "input_labels": [1],
        }
        mock_inputs["pixel_values"].to = Mock(return_value=mock_inputs["pixel_values"])
        adapter.processor.return_value = mock_inputs

        # Mock model output (3 masks per prompt)
        mock_outputs = Mock()
        mock_masks = np.random.rand(1, 3, 480, 640) > 0.5  # (batch, num_masks, H, W)
        mock_iou_scores = np.array([[0.95, 0.85, 0.75]])  # (batch, num_masks)

        # Create torch-like tensors with squeeze and cpu methods
        mock_pred_masks = Mock()
        mock_pred_masks.squeeze = Mock(
            return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=mock_masks[0]))))
        )
        mock_iou_tensor = Mock()
        mock_iou_tensor.squeeze = Mock(
            return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=mock_iou_scores[0]))))
        )

        mock_outputs.pred_masks = mock_pred_masks
        mock_outputs.iou_scores = mock_iou_tensor

        adapter.model.return_value = mock_outputs

        # Run prediction
        result = adapter.predict(test_image, point_prompts=[(100, 150, 1)])  # (x, y, foreground)

        # Verify result
        assert isinstance(result, VisionResult)
        assert len(result.masks) == 3  # SAM returns 3 masks per prompt
        assert result.meta["mode"] == "zeroshot"
        assert result.meta["num_prompts"] == 1
        assert result.meta["prompts"]["points"] == [(100, 150, 1)]

        # Verify masks are sorted by IoU score (descending)
        assert result.masks[0].score >= result.masks[1].score >= result.masks[2].score

        # Verify all masks have correct properties
        for mask in result.masks:
            assert mask.label == 0  # Class-agnostic
            assert mask.label_name == "object"
            assert mask.is_stuff is None  # Not applicable for zero-shot
            assert mask.score > 0.0

    def test_predict_box_prompts(self, adapter):
        """Test prediction with box prompts."""
        test_image = Image.new("RGB", (640, 480), color="blue")

        # Mock processor output
        mock_inputs = {
            "pixel_values": Mock(spec=["to"]),
            "input_boxes": [[50.0, 50.0, 300.0, 300.0]],
        }
        mock_inputs["pixel_values"].to = Mock(return_value=mock_inputs["pixel_values"])
        adapter.processor.return_value = mock_inputs

        # Mock model output
        mock_outputs = Mock()
        mock_masks = np.random.rand(1, 3, 480, 640) > 0.5
        mock_iou_scores = np.array([[0.92, 0.88, 0.80]])

        mock_pred_masks = Mock()
        mock_pred_masks.squeeze = Mock(
            return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=mock_masks[0]))))
        )
        mock_iou_tensor = Mock()
        mock_iou_tensor.squeeze = Mock(
            return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=mock_iou_scores[0]))))
        )

        mock_outputs.pred_masks = mock_pred_masks
        mock_outputs.iou_scores = mock_iou_tensor

        adapter.model.return_value = mock_outputs

        # Run prediction
        result = adapter.predict(test_image, box_prompts=[(50, 50, 300, 300)])  # (x1, y1, x2, y2)

        # Verify result
        assert isinstance(result, VisionResult)
        assert len(result.masks) == 3
        assert result.meta["prompts"]["boxes"] == [(50, 50, 300, 300)]

    def test_predict_combined_prompts(self, adapter):
        """Test prediction with combined point and box prompts."""
        test_image = Image.new("RGB", (640, 480), color="green")

        # Mock processor output
        mock_inputs = {
            "pixel_values": Mock(spec=["to"]),
            "input_points": [[100.0, 150.0], [200.0, 250.0]],
            "input_labels": [1, 0],  # Foreground + background
            "input_boxes": [[50.0, 50.0, 300.0, 300.0]],
        }
        mock_inputs["pixel_values"].to = Mock(return_value=mock_inputs["pixel_values"])
        adapter.processor.return_value = mock_inputs

        # Mock model output
        mock_outputs = Mock()
        mock_masks = np.random.rand(1, 3, 480, 640) > 0.5
        mock_iou_scores = np.array([[0.90, 0.85, 0.70]])

        mock_pred_masks = Mock()
        mock_pred_masks.squeeze = Mock(
            return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=mock_masks[0]))))
        )
        mock_iou_tensor = Mock()
        mock_iou_tensor.squeeze = Mock(
            return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=mock_iou_scores[0]))))
        )

        mock_outputs.pred_masks = mock_pred_masks
        mock_outputs.iou_scores = mock_iou_tensor

        adapter.model.return_value = mock_outputs

        # Run prediction
        result = adapter.predict(
            test_image, point_prompts=[(100, 150, 1), (200, 250, 0)], box_prompts=[(50, 50, 300, 300)]
        )

        # Verify result
        assert isinstance(result, VisionResult)
        assert result.meta["num_prompts"] == 3  # 2 points + 1 box
        assert result.meta["prompts"]["points"] == [(100, 150, 1), (200, 250, 0)]
        assert result.meta["prompts"]["boxes"] == [(50, 50, 300, 300)]

    def test_predict_threshold_filtering(self, adapter):
        """Test threshold filtering of low-quality masks."""
        test_image = Image.new("RGB", (640, 480), color="yellow")

        # Mock processor output
        mock_inputs = {
            "pixel_values": Mock(spec=["to"]),
            "input_points": [[100.0, 150.0]],
            "input_labels": [1],
        }
        mock_inputs["pixel_values"].to = Mock(return_value=mock_inputs["pixel_values"])
        adapter.processor.return_value = mock_inputs

        # Mock model output with varying quality
        mock_outputs = Mock()
        mock_masks = np.random.rand(1, 3, 480, 640) > 0.5
        mock_iou_scores = np.array([[0.95, 0.75, 0.45]])  # High, medium, low

        mock_pred_masks = Mock()
        mock_pred_masks.squeeze = Mock(
            return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=mock_masks[0]))))
        )
        mock_iou_tensor = Mock()
        mock_iou_tensor.squeeze = Mock(
            return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=mock_iou_scores[0]))))
        )

        mock_outputs.pred_masks = mock_pred_masks
        mock_outputs.iou_scores = mock_iou_tensor

        adapter.model.return_value = mock_outputs

        # Run prediction with threshold=0.7
        result = adapter.predict(
            test_image, point_prompts=[(100, 150, 1)], threshold=0.7  # Filter out low-quality mask
        )

        # Verify only high-quality masks returned
        assert len(result.masks) == 2  # Only masks with IoU >= 0.7
        assert all(mask.score >= 0.7 for mask in result.masks)
        assert result.meta["threshold"] == 0.7

    def test_predict_no_prompts_raises_error(self, adapter):
        """Test that prediction without prompts raises InvalidInputError."""
        test_image = Image.new("RGB", (640, 480), color="white")

        with pytest.raises(InvalidInputError) as exc_info:
            adapter.predict(test_image)  # No prompts provided

        assert "At least one prompt type" in str(exc_info.value)

    def test_predict_with_numpy_array(self, adapter):
        """Test prediction with numpy array input."""
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Mock processor output
        mock_inputs = {
            "pixel_values": Mock(spec=["to"]),
            "input_points": [[100.0, 150.0]],
            "input_labels": [1],
        }
        mock_inputs["pixel_values"].to = Mock(return_value=mock_inputs["pixel_values"])
        adapter.processor.return_value = mock_inputs

        # Mock model output
        mock_outputs = Mock()
        mock_masks = np.random.rand(1, 3, 480, 640) > 0.5
        mock_iou_scores = np.array([[0.90, 0.80, 0.70]])

        mock_pred_masks = Mock()
        mock_pred_masks.squeeze = Mock(
            return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=mock_masks[0]))))
        )
        mock_iou_tensor = Mock()
        mock_iou_tensor.squeeze = Mock(
            return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=mock_iou_scores[0]))))
        )

        mock_outputs.pred_masks = mock_pred_masks
        mock_outputs.iou_scores = mock_iou_tensor

        adapter.model.return_value = mock_outputs

        # Run prediction
        result = adapter.predict(test_image, point_prompts=[(100, 150, 1)])

        assert isinstance(result, VisionResult)
        assert len(result.masks) == 3


class TestSAMAdapterInfo:
    """Test SAM adapter info method."""

    def test_info(self, mock_transformers):
        """Test adapter info returns correct metadata."""
        adapter = HuggingFaceSAMAdapter(model_id="facebook/sam-vit-huge", device="cpu", threshold=0.5)

        info = adapter.info()

        assert info["name"] == "huggingface_sam"
        assert info["task"] == "segment"
        assert info["model_id"] == "facebook/sam-vit-huge"
        assert info["mode"] == "zeroshot"
        assert info["threshold"] == 0.5
        assert info["backend"] == "transformers"
        assert info["class_agnostic"] is True


class TestSAMUniversalLoaderIntegration:
    """Test SAM integration with UniversalLoader."""

    @patch("mata.adapters.huggingface_sam_adapter._ensure_transformers")
    def test_load_sam_via_universal_loader(self, mock_transformers):
        """Test loading SAM model through UniversalLoader."""
        from mata.core.model_loader import UniversalLoader

        # Mock transformers
        mock_sam_model = Mock()
        mock_sam_processor = Mock()
        mock_model_instance = Mock()
        mock_model_instance.to = Mock(return_value=mock_model_instance)
        mock_model_instance.eval = Mock(return_value=mock_model_instance)
        mock_processor_instance = Mock()

        mock_sam_model.from_pretrained = Mock(return_value=mock_model_instance)
        mock_sam_processor.from_pretrained = Mock(return_value=mock_processor_instance)

        mock_transformers.return_value = {
            "SamModel": mock_sam_model,
            "SamProcessor": mock_sam_processor,
        }

        loader = UniversalLoader()

        # Load SAM model
        adapter = loader.load("segment", "facebook/sam-vit-base")

        # Verify correct adapter type
        assert isinstance(adapter, HuggingFaceSAMAdapter)
        assert adapter.model_id == "facebook/sam-vit-base"

    @patch("mata.adapters.huggingface_sam_adapter._ensure_transformers")
    def test_load_sam_with_zeroshot_mode(self, mock_transformers):
        """Test loading SAM with explicit zeroshot mode."""
        from mata.core.model_loader import UniversalLoader

        # Mock transformers
        mock_sam_model = Mock()
        mock_sam_processor = Mock()
        mock_model_instance = Mock()
        mock_model_instance.to = Mock(return_value=mock_model_instance)
        mock_model_instance.eval = Mock(return_value=mock_model_instance)
        mock_processor_instance = Mock()

        mock_sam_model.from_pretrained = Mock(return_value=mock_model_instance)
        mock_sam_processor.from_pretrained = Mock(return_value=mock_processor_instance)

        mock_transformers.return_value = {
            "SamModel": mock_sam_model,
            "SamProcessor": mock_sam_processor,
        }

        loader = UniversalLoader()

        # Load with zeroshot mode (forces SAM adapter)
        # Use HuggingFace ID pattern to trigger _load_from_huggingface
        adapter = loader.load("segment", "facebook/some-model", segment_mode="zeroshot")

        # Should route to SAM adapter
        assert isinstance(adapter, HuggingFaceSAMAdapter)


# ============================================================================
# SAM3 Text Prompt Tests
# ============================================================================


def test_sam3_text_prompts(mock_transformers, mock_pycocotools):
    """Test SAM3 with text prompts (Promptable Concept Segmentation)."""
    # Add SAM3 classes to mock
    mock_sam3_model = Mock()
    mock_sam3_processor = Mock()

    mock_model_instance = Mock()
    mock_model_instance.to = Mock(return_value=mock_model_instance)
    mock_model_instance.eval = Mock(return_value=mock_model_instance)
    mock_processor_instance = Mock()

    mock_sam3_model.from_pretrained = Mock(return_value=mock_model_instance)
    mock_sam3_processor.from_pretrained = Mock(return_value=mock_processor_instance)

    # Add SAM3 classes to transformers mock
    mock_transformers["ensure"].return_value.update(
        {
            "Sam3Model": mock_sam3_model,
            "Sam3Processor": mock_sam3_processor,
        }
    )

    # Initialize SAM3 adapter
    adapter = HuggingFaceSAMAdapter("facebook/sam3", device="cpu")

    # Mock inference output
    mock_processor_instance.post_process_instance_segmentation = Mock(
        return_value=[
            {
                "masks": np.array(
                    [
                        [[True, False], [False, True]],  # Mask 1
                        [[False, True], [True, False]],  # Mask 2
                    ]
                ),
                "boxes": np.array([[0, 0, 1, 1], [0, 0, 2, 2]]),
                "scores": np.array([0.9, 0.8]),
            }
        ]
    )

    # Mock processor call
    mock_processor_instance.return_value = {
        "pixel_values": np.zeros((1, 3, 224, 224)),
        "original_sizes": [[224, 224]],
    }

    # Mock model inference
    mock_model_instance.return_value = Mock()  # Outputs object

    # Create test image
    img = Image.new("RGB", (224, 224))

    # Run prediction with text prompt
    result = adapter.predict(img, text_prompts="cat")

    # Verify text prompt was passed to processor
    assert mock_processor_instance.call_count == 1
    call_kwargs = mock_processor_instance.call_args[1]
    assert "text" in call_kwargs
    assert call_kwargs["text"] == "cat"
    assert "return_tensors" in call_kwargs

    # Verify post-processing was called
    assert mock_processor_instance.post_process_instance_segmentation.called

    # Verify result structure
    assert isinstance(result, VisionResult)
    assert len(result.masks) == 2
    assert all(isinstance(m, Instance) for m in result.masks)
    assert result.masks[0].score == 0.9
    assert result.masks[1].score == 0.8


def test_sam3_text_with_negative_boxes(mock_transformers, mock_pycocotools):
    """Test SAM3 with text prompts + negative boxes for refinement."""
    # Add SAM3 classes
    mock_sam3_model = Mock()
    mock_sam3_processor = Mock()

    mock_model_instance = Mock()
    mock_model_instance.to = Mock(return_value=mock_model_instance)
    mock_model_instance.eval = Mock(return_value=mock_model_instance)
    mock_processor_instance = Mock()

    mock_sam3_model.from_pretrained = Mock(return_value=mock_model_instance)
    mock_sam3_processor.from_pretrained = Mock(return_value=mock_processor_instance)

    mock_transformers["ensure"].return_value.update(
        {
            "Sam3Model": mock_sam3_model,
            "Sam3Processor": mock_sam3_processor,
        }
    )

    adapter = HuggingFaceSAMAdapter("facebook/sam3", device="cpu")

    # Mock outputs
    mock_processor_instance.post_process_instance_segmentation = Mock(
        return_value=[
            {
                "masks": np.array([[[True, False], [False, True]]]),
                "boxes": np.array([[0, 0, 1, 1]]),
                "scores": np.array([0.85]),
            }
        ]
    )

    mock_processor_instance.return_value = {
        "pixel_values": np.zeros((1, 3, 224, 224)),
        "original_sizes": [[224, 224]],
    }
    mock_model_instance.return_value = Mock()

    img = Image.new("RGB", (224, 224))

    # Run with text + negative box
    result = adapter.predict(
        img, text_prompts="handle", box_prompts=[(40, 183, 318, 204)], box_labels=[0]  # Negative box
    )

    # Verify both text and box prompts were passed
    call_kwargs = mock_processor_instance.call_args[1]
    assert call_kwargs["text"] == "handle"
    assert "input_boxes" in call_kwargs
    assert "input_boxes_labels" in call_kwargs
    assert call_kwargs["input_boxes_labels"] == [[0]]  # Negative

    assert isinstance(result, VisionResult)
    assert len(result.masks) == 1


def test_sam3_batched_text_prompts(mock_transformers, mock_pycocotools):
    """Test SAM3 with list of text prompts."""
    # Setup SAM3 mocks
    mock_sam3_model = Mock()
    mock_sam3_processor = Mock()

    mock_model_instance = Mock()
    mock_model_instance.to = Mock(return_value=mock_model_instance)
    mock_model_instance.eval = Mock(return_value=mock_model_instance)
    mock_processor_instance = Mock()

    mock_sam3_model.from_pretrained = Mock(return_value=mock_model_instance)
    mock_sam3_processor.from_pretrained = Mock(return_value=mock_processor_instance)

    mock_transformers["ensure"].return_value.update(
        {
            "Sam3Model": mock_sam3_model,
            "Sam3Processor": mock_sam3_processor,
        }
    )

    adapter = HuggingFaceSAMAdapter("facebook/sam3", device="cpu")

    # Mock outputs for multiple concepts
    mock_processor_instance.post_process_instance_segmentation = Mock(
        return_value=[
            {
                "masks": np.array(
                    [
                        [[True, False], [False, True]],
                        [[False, True], [True, False]],
                        [[True, True], [False, False]],
                    ]
                ),
                "boxes": np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 1, 2]]),
                "scores": np.array([0.9, 0.85, 0.8]),
            }
        ]
    )

    mock_processor_instance.return_value = {
        "pixel_values": np.zeros((1, 3, 224, 224)),
        "original_sizes": [[224, 224]],
    }
    mock_model_instance.return_value = Mock()

    img = Image.new("RGB", (224, 224))

    # Run with list of text prompts
    result = adapter.predict(img, text_prompts=["cat", "dog", "person"])

    # Verify list was passed
    call_kwargs = mock_processor_instance.call_args[1]
    assert call_kwargs["text"] == ["cat", "dog", "person"]

    assert isinstance(result, VisionResult)
    assert len(result.masks) == 3


def test_text_prompt_requires_sam3(mock_transformers, mock_pycocotools):
    """Test that text prompts fail on original SAM (not SAM3)."""
    # Initialize original SAM (not SAM3)
    adapter = HuggingFaceSAMAdapter("facebook/sam-vit-base", device="cpu")

    img = Image.new("RGB", (224, 224))

    # Should raise InvalidInputError
    with pytest.raises(InvalidInputError, match="Text prompts are only supported by SAM3"):
        adapter.predict(img, text_prompts="cat")


def test_sam3_unavailable_error(mock_transformers):
    """Test error when SAM3 is requested but not available in transformers."""
    # Don't add SAM3 classes to mock (simulate old transformers)
    mock_transformers["ensure"].return_value = {
        "SamModel": Mock(),
        "SamProcessor": Mock(),
        # No Sam3Model/Sam3Processor
    }

    # Should raise UnsupportedModelError
    from mata.core.exceptions import UnsupportedModelError

    with pytest.raises(UnsupportedModelError, match="SAM3 model requested but not available"):
        HuggingFaceSAMAdapter("facebook/sam3", device="cpu")
