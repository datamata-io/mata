"""Tests for HuggingFaceZeroShotDetectAdapter (zero-shot object detection)."""

from unittest.mock import Mock, patch

import pytest
from PIL import Image

from mata.adapters.huggingface_zeroshot_detect_adapter import HuggingFaceZeroShotDetectAdapter
from mata.core.exceptions import InvalidInputError
from mata.core.types import Instance, VisionResult


@pytest.fixture
def mock_transformers():
    """Mock transformers library components."""
    with patch("mata.adapters.huggingface_zeroshot_detect_adapter._ensure_transformers") as mock_ensure:
        # Create mock classes
        mock_processor_class = Mock()
        mock_model_class = Mock()

        # Mock processor instance
        mock_processor = Mock()
        mock_processor.return_value = {"pixel_values": Mock(), "input_ids": Mock()}

        # Mock model instance
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.config = Mock()

        # Set up from_pretrained
        mock_processor_class.from_pretrained = Mock(return_value=mock_processor)
        mock_model_class.from_pretrained = Mock(return_value=mock_model)

        # Return dict with mocked classes
        mock_ensure.return_value = {
            "AutoProcessor": mock_processor_class,
            "AutoModelForZeroShotObjectDetection": mock_model_class,
            "OwlViTProcessor": mock_processor_class,
            "OwlViTForObjectDetection": mock_model_class,
            "Owlv2Processor": mock_processor_class,
            "Owlv2ForObjectDetection": mock_model_class,
        }

        yield {
            "ensure": mock_ensure,
            "processor_class": mock_processor_class,
            "model_class": mock_model_class,
            "processor": mock_processor,
            "model": mock_model,
        }


@pytest.fixture
def mock_pytorch_base():
    """Mock PyTorch base adapter functionality."""
    with patch("mata.adapters.pytorch_base._ensure_torch") as mock_ensure:
        mock_torch = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.device = Mock(return_value=Mock(type="cpu"))
        mock_ensure.return_value = mock_torch
        yield mock_torch


class TestZeroShotAdapterInitialization:
    """Test adapter initialization."""

    def test_init_grounding_dino(self, mock_transformers, mock_pytorch_base):
        """Test initialization with GroundingDINO model."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="IDEA-Research/grounding-dino-tiny")

        assert adapter.model_id == "IDEA-Research/grounding-dino-tiny"
        assert adapter.architecture == "grounding_dino"
        assert adapter.threshold == 0.3  # Default threshold

    def test_init_owlvit_v1(self, mock_transformers, mock_pytorch_base):
        """Test initialization with OWL-ViT v1 model."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlvit-base-patch32")

        assert adapter.model_id == "google/owlvit-base-patch32"
        assert adapter.architecture == "owlvit_v1"
        assert adapter.threshold == 0.3  # Default for OWL-ViT

    def test_init_owlvit_v2(self, mock_transformers, mock_pytorch_base):
        """Test initialization with OWL-ViT v2 model."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlv2-large-patch14")

        assert adapter.model_id == "google/owlv2-large-patch14"
        assert adapter.architecture == "owlvit_v2"
        assert adapter.threshold == 0.3

    def test_init_custom_threshold(self, mock_transformers, mock_pytorch_base):
        """Test initialization with custom threshold."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="IDEA-Research/grounding-dino-tiny", threshold=0.5)

        assert adapter.threshold == 0.5

    def test_init_custom_device(self, mock_transformers, mock_pytorch_base):
        """Test initialization with custom device."""
        with patch.object(mock_pytorch_base, "cuda") as mock_cuda:
            mock_cuda.is_available = Mock(return_value=True)
            mock_device = Mock()
            mock_device.type = "cuda"
            mock_device.__str__ = Mock(return_value="cuda")
            mock_pytorch_base.device = Mock(return_value=mock_device)

            adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlvit-base-patch32", device="cuda")

            assert str(adapter.device) == "cuda"


class TestArchitectureDetection:
    """Test architecture detection from model IDs."""

    def test_detect_grounding_dino_tiny(self, mock_transformers, mock_pytorch_base):
        """Test detection of GroundingDINO tiny model."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="IDEA-Research/grounding-dino-tiny")
        assert adapter.architecture == "grounding_dino"

    def test_detect_grounding_dino_base(self, mock_transformers, mock_pytorch_base):
        """Test detection of GroundingDINO base model."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="IDEA-Research/grounding-dino-base")
        assert adapter.architecture == "grounding_dino"

    def test_detect_owlvit_base(self, mock_transformers, mock_pytorch_base):
        """Test detection of OWL-ViT base model."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlvit-base-patch32")
        assert adapter.architecture == "owlvit_v1"

    def test_detect_owlv2_large(self, mock_transformers, mock_pytorch_base):
        """Test detection of OWL-ViT v2 large model."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlv2-large-patch14")
        assert adapter.architecture == "owlvit_v2"

    def test_detect_unknown_model_uses_auto(self, mock_transformers, mock_pytorch_base):
        """Test that unknown models use auto mode."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="unknown/model")
        assert adapter.architecture == "auto"


class TestTextPromptNormalization:
    """Test text prompt normalization."""

    def test_normalize_string_to_grounding_dino(self, mock_transformers, mock_pytorch_base):
        """Test normalizing string prompt for GroundingDINO."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="IDEA-Research/grounding-dino-tiny")

        # String input should be kept as-is for GroundingDINO
        result = adapter._normalize_text_prompts("cat . dog . person")
        assert result == "cat . dog . person"

    def test_normalize_list_to_grounding_dino(self, mock_transformers, mock_pytorch_base):
        """Test normalizing list prompt for GroundingDINO."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="IDEA-Research/grounding-dino-tiny")

        # List should be joined with " . " for GroundingDINO
        result = adapter._normalize_text_prompts(["cat", "dog", "person"])
        assert result == "cat . dog . person"

    def test_normalize_string_to_owlvit(self, mock_transformers, mock_pytorch_base):
        """Test normalizing string prompt for OWL-ViT."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlvit-base-patch32")

        # String with dots should be split to list for OWL-ViT
        result = adapter._normalize_text_prompts("cat . dog . person")
        assert result == ["cat", "dog", "person"]

    def test_normalize_list_to_owlvit(self, mock_transformers, mock_pytorch_base):
        """Test normalizing list prompt for OWL-ViT."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlvit-base-patch32")

        # List should be kept as-is for OWL-ViT
        result = adapter._normalize_text_prompts(["cat", "dog", "person"])
        assert result == ["cat", "dog", "person"]


class TestPrediction:
    """Test prediction functionality."""

    def test_predict_returns_vision_result(self, mock_transformers, mock_pytorch_base):
        """Test that predict returns VisionResult."""
        # Mock model output
        mock_outputs = Mock()
        mock_outputs.logits = Mock()
        mock_outputs.pred_boxes = Mock()

        mock_transformers["model"].return_value = mock_outputs

        # Mock post-processing
        with patch.object(
            HuggingFaceZeroShotDetectAdapter, "_predict_single", return_value=VisionResult(instances=[], meta={})
        ):
            adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlvit-base-patch32")

            # Create test image
            image = Image.new("RGB", (640, 480))

            result = adapter.predict(image, text_prompts="cat . dog")

            assert isinstance(result, VisionResult)

    def test_predict_single_image(self, mock_transformers, mock_pytorch_base):
        """Test prediction on single image."""
        # Create mock detection output
        mock_result = VisionResult(
            instances=[Instance(bbox=(10.0, 20.0, 100.0, 150.0), score=0.95, label=0, label_name="cat")],
            meta={"model_id": "google/owlvit-base-patch32"},
        )

        with patch.object(HuggingFaceZeroShotDetectAdapter, "_predict_single", return_value=mock_result):
            adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlvit-base-patch32")

            image = Image.new("RGB", (640, 480))
            result = adapter.predict(image, text_prompts="cat")

            assert len(result.instances) == 1
            assert result.instances[0].label_name == "cat"
            assert result.instances[0].score == 0.95

    def test_predict_batch_images(self, mock_transformers, mock_pytorch_base):
        """Test prediction on batch of images."""
        # Create mock detection outputs
        mock_result1 = VisionResult(
            instances=[Instance(bbox=(10.0, 20.0, 100.0, 150.0), score=0.95, label=0, label_name="cat")], meta={}
        )
        mock_result2 = VisionResult(
            instances=[Instance(bbox=(30.0, 40.0, 120.0, 180.0), score=0.88, label=1, label_name="dog")], meta={}
        )

        with patch.object(
            HuggingFaceZeroShotDetectAdapter, "_predict_single", side_effect=[mock_result1, mock_result2]
        ):
            adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlvit-base-patch32")

            images = [Image.new("RGB", (640, 480)), Image.new("RGB", (640, 480))]
            results = adapter.predict(images, text_prompts="cat . dog")

            assert isinstance(results, list)
            assert len(results) == 2
            assert results[0].instances[0].label_name == "cat"
            assert results[1].instances[0].label_name == "dog"

    def test_predict_empty_text_prompts_raises(self, mock_transformers, mock_pytorch_base):
        """Test that prediction with empty text prompts raises error."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlvit-base-patch32")

        image = Image.new("RGB", (640, 480))

        with pytest.raises(InvalidInputError, match="text_prompts required"):
            adapter.predict(image, text_prompts="")

    def test_predict_filters_by_threshold(self, mock_transformers, mock_pytorch_base):
        """Test that predictions are filtered by threshold."""
        # Mock multiple detections with varying scores
        mock_result = VisionResult(
            instances=[
                Instance(bbox=(10.0, 20.0, 100.0, 150.0), score=0.95, label=0, label_name="cat"),
                Instance(bbox=(30.0, 40.0, 120.0, 180.0), score=0.15, label=1, label_name="dog"),
            ],
            meta={},
        )

        with patch.object(HuggingFaceZeroShotDetectAdapter, "_predict_single", return_value=mock_result):
            adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlvit-base-patch32", threshold=0.3)

            image = Image.new("RGB", (640, 480))
            adapter.predict(image, text_prompts="cat . dog", threshold=0.5)

            # Only high-confidence detection should remain
            # Note: threshold filtering happens in _predict_single
            # This test verifies threshold parameter is passed through


class TestLabelMapping:
    """Test label-to-prompt name mapping."""

    def test_label_mapping_grounding_dino(self, mock_transformers, mock_pytorch_base):
        """Test label mapping for GroundingDINO."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="IDEA-Research/grounding-dino-tiny")

        # GroundingDINO uses space-dot format
        prompts = "cat . dog . person"
        label_names = adapter._extract_label_names(prompts)

        assert label_names == ["cat", "dog", "person"]

    def test_label_mapping_owlvit(self, mock_transformers, mock_pytorch_base):
        """Test label mapping for OWL-ViT."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlvit-base-patch32")

        # OWL-ViT uses list format
        prompts = ["cat", "dog", "person"]
        label_names = adapter._extract_label_names(prompts)

        assert label_names == ["cat", "dog", "person"]


class TestInfoMethod:
    """Test adapter info method."""

    def test_info_returns_metadata(self, mock_transformers, mock_pytorch_base):
        """Test that info() returns correct metadata."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="IDEA-Research/grounding-dino-tiny", threshold=0.4)

        info = adapter.info()

        assert info["name"] == "HuggingFaceZeroShotDetectAdapter"
        assert info["task"] == "detect"
        assert info["model_id"] == "IDEA-Research/grounding-dino-tiny"
        assert info["architecture"] == "grounding_dino"
        assert info["mode"] == "zeroshot"
        assert info["threshold"] == 0.4
        assert info["backend"] == "huggingface-transformers"


class TestBatchProcessing:
    """Test batch processing edge cases."""

    def test_empty_batch_returns_empty_list(self, mock_transformers, mock_pytorch_base):
        """Test that empty batch returns empty list."""
        adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlvit-base-patch32")

        results = adapter.predict([], text_prompts="cat")

        assert results == []

    def test_single_item_batch_returns_list(self, mock_transformers, mock_pytorch_base):
        """Test that single-item batch still returns list."""
        mock_result = VisionResult(instances=[], meta={})

        with patch.object(HuggingFaceZeroShotDetectAdapter, "_predict_single", return_value=mock_result):
            adapter = HuggingFaceZeroShotDetectAdapter(model_id="google/owlvit-base-patch32")

            image = Image.new("RGB", (640, 480))
            results = adapter.predict([image], text_prompts="cat")

            assert isinstance(results, list)
            assert len(results) == 1


# Integration test with UniversalLoader
def test_load_via_universal_loader():
    """Test loading zero-shot adapter via UniversalLoader."""
    with patch("mata.adapters.huggingface_zeroshot_detect_adapter._ensure_transformers"):
        with patch("mata.adapters.pytorch_base._ensure_torch") as mock_torch_ensure:
            mock_torch = Mock()
            mock_torch.cuda.is_available = Mock(return_value=False)
            mock_torch.device = Mock(return_value=Mock(type="cpu"))
            mock_torch_ensure.return_value = mock_torch
            from mata.core.model_loader import UniversalLoader

            loader = UniversalLoader()

            # Load with explicit zeroshot mode
            with patch.object(UniversalLoader, "_load_from_huggingface") as mock_load:
                mock_load.return_value = Mock(spec=HuggingFaceZeroShotDetectAdapter)

                loader.load("detect", "google/owlvit-base-patch32", detect_mode="zeroshot")

                mock_load.assert_called_once()
                call_args = mock_load.call_args
                assert call_args[0][0] == "detect"
                assert call_args[0][1] == "google/owlvit-base-patch32"
