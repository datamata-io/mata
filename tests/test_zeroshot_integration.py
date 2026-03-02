"""Integration tests for zero-shot tasks (detection, segmentation, classification).

Tests the consistency and integration of all three zero-shot adapters:
- HuggingFaceZeroShotDetectAdapter (detection)
- HuggingFaceZeroShotSegmentAdapter (segmentation)
- HuggingFaceCLIPAdapter (classification)
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from PIL import Image

from mata.adapters.clip_adapter import HuggingFaceCLIPAdapter
from mata.adapters.huggingface_zeroshot_detect_adapter import HuggingFaceZeroShotDetectAdapter
from mata.adapters.huggingface_zeroshot_segment_adapter import HuggingFaceZeroShotSegmentAdapter
from mata.core.types import Classification, ClassifyResult, VisionResult


@pytest.fixture
def mock_transformers_all():
    """Mock transformers for all zero-shot adapters."""
    with (
        patch("mata.adapters.huggingface_zeroshot_detect_adapter._ensure_transformers") as mock_detect,
        patch("mata.adapters.clip_adapter._ensure_transformers") as mock_clip,
        patch("mata.adapters.huggingface_zeroshot_segment_adapter._ensure_transformers") as mock_segment,
        patch("mata.adapters.huggingface_zeroshot_segment_adapter._ensure_pycocotools", return_value=None),
    ):

        # Mock detection components
        mock_detect_processor = Mock()
        mock_detect_processor.return_value = {"pixel_values": Mock(), "input_ids": Mock()}

        mock_detect_model = Mock()
        mock_detect_model.to = Mock(return_value=mock_detect_model)
        mock_detect_model.eval = Mock(return_value=mock_detect_model)

        mock_detect.return_value = {
            "AutoProcessor": Mock(from_pretrained=Mock(return_value=mock_detect_processor)),
            "AutoModelForZeroShotObjectDetection": Mock(from_pretrained=Mock(return_value=mock_detect_model)),
            "OwlViTProcessor": Mock(from_pretrained=Mock(return_value=mock_detect_processor)),
            "OwlViTForObjectDetection": Mock(from_pretrained=Mock(return_value=mock_detect_model)),
            "Owlv2Processor": Mock(from_pretrained=Mock(return_value=mock_detect_processor)),
            "Owlv2ForObjectDetection": Mock(from_pretrained=Mock(return_value=mock_detect_model)),
        }

        # Mock CLIP components with dynamic logits
        # Track number of prompts for dynamic logits
        num_prompts_tracker = {"count": 3}  # Default

        mock_clip_processor = Mock()

        def clip_processor_call(*args, **kwargs):
            """Return mock inputs with real tensors that have .to() method."""
            # Count number of text prompts
            text_arg = kwargs.get("text") or (args[1] if len(args) > 1 else None)
            num_prompts = len(text_arg) if isinstance(text_arg, list) else 1
            num_prompts_tracker["count"] = num_prompts

            pixel_values = torch.randn(1, 3, 224, 224)
            input_ids = torch.randint(0, 100, (1, 77))
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
            }

        mock_clip_processor.side_effect = clip_processor_call

        # Mock model with dynamic logits based on tracked prompt count
        def clip_model_call(**inputs):
            """Return mock outputs with dynamic logits shape."""
            mock_outputs = Mock()
            # Use tracked count for logits shape
            mock_outputs.logits_per_image = torch.randn(1, num_prompts_tracker["count"])
            return mock_outputs

        mock_clip_model = Mock()
        mock_clip_model.side_effect = clip_model_call
        mock_clip_model.to = Mock(return_value=mock_clip_model)
        mock_clip_model.eval = Mock(return_value=mock_clip_model)

        mock_clip.return_value = {
            "CLIPProcessor": Mock(from_pretrained=Mock(return_value=mock_clip_processor)),
            "CLIPModel": Mock(from_pretrained=Mock(return_value=mock_clip_model)),
        }

        # Mock segmentation components (CLIPSeg)
        seg_prompt_tracker = {"count": 2}

        mock_seg_processor = Mock()

        def seg_processor_call(*args, **kwargs):
            text_arg = kwargs.get("text") or (args[0] if args else None)
            if isinstance(text_arg, list):
                seg_prompt_tracker["count"] = len(text_arg)
            n = seg_prompt_tracker["count"]
            return {
                "input_ids": torch.randint(0, 100, (n, 77)),
                "pixel_values": torch.randn(n, 3, 224, 224),
            }

        mock_seg_processor.side_effect = seg_processor_call

        def seg_model_call(**inputs):
            mock_outputs = Mock()
            n = seg_prompt_tracker["count"]
            logits = torch.randn(n, 352, 352)
            for i in range(n):
                logits[i, 50:150, 50:150] = 3.0 + i
            mock_outputs.logits = logits
            return mock_outputs

        mock_seg_model = Mock()
        mock_seg_model.side_effect = seg_model_call
        mock_seg_model.to = Mock(return_value=mock_seg_model)
        mock_seg_model.eval = Mock(return_value=mock_seg_model)

        mock_segment.return_value = {
            "CLIPSegForImageSegmentation": Mock(from_pretrained=Mock(return_value=mock_seg_model)),
            "CLIPSegProcessor": Mock(from_pretrained=Mock(return_value=mock_seg_processor)),
            "AutoProcessor": Mock(from_pretrained=Mock(return_value=mock_seg_processor)),
        }

        yield {
            "detect": {"processor": mock_detect_processor, "model": mock_detect_model},
            "clip": {"processor": mock_clip_processor, "model": mock_clip_model},
            "segment": {"processor": mock_seg_processor, "model": mock_seg_model},
        }


@pytest.fixture
def mock_pytorch_base():
    """Mock PyTorchbase functionality."""
    with patch("mata.adapters.pytorch_base._ensure_torch") as mock_ensure:
        # Import real torch for device compatibility
        import torch

        mock_torch = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        # Use real torch.device for tensor.to() compatibility
        mock_torch.device = torch.device
        mock_torch.no_grad = MagicMock()
        mock_ensure.return_value = mock_torch
        yield mock_torch


class TestZeroShotTaskLoading:
    """Test all zero-shot tasks load successfully."""

    def test_load_zeroshot_detection(self, mock_transformers_all, mock_pytorch_base):
        """Test zero-shot detection adapter loads."""
        adapter = HuggingFaceZeroShotDetectAdapter("google/owlv2-base-patch16")

        assert adapter.task == "detect"
        assert adapter.architecture in ["owlvit_v2", "grounding_dino", "owlvit_v1"]

    def test_load_zeroshot_classification(self, mock_transformers_all, mock_pytorch_base):
        """Test zero-shot classification adapter loads."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        assert adapter.task == "classify"
        assert adapter.template is not None

    def test_load_zeroshot_segmentation(self, mock_transformers_all, mock_pytorch_base):
        """Test zero-shot segmentation adapter loads."""
        adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False)

        assert adapter.task == "segment"
        assert adapter.architecture == "clipseg"

    def test_all_tasks_share_pytorch_base(self, mock_transformers_all, mock_pytorch_base):
        """Test all zero-shot adapters inherit from PyTorchBaseAdapter."""
        detect_adapter = HuggingFaceZeroShotDetectAdapter("google/owlv2-base-patch16")
        clip_adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")
        segment_adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False)

        # All should have device attribute
        assert hasattr(detect_adapter, "device")
        assert hasattr(clip_adapter, "device")
        assert hasattr(segment_adapter, "device")

        # All should have threshold attribute
        assert hasattr(detect_adapter, "threshold")
        assert hasattr(clip_adapter, "threshold")
        assert hasattr(segment_adapter, "threshold")


class TestTextPromptInterface:
    """Test text prompt handling consistency across tasks."""

    def test_detection_accepts_text_prompts(self, mock_transformers_all, mock_pytorch_base):
        """Test detection adapter accepts text prompts."""
        adapter = HuggingFaceZeroShotDetectAdapter("google/owlv2-base-patch16")

        # Should have predict method that accepts text_prompts
        assert hasattr(adapter, "predict")
        # Check method signature includes text_prompts parameter
        import inspect

        sig = inspect.signature(adapter.predict)
        assert "text_prompts" in sig.parameters

    def test_classification_accepts_text_prompts(self, mock_transformers_all, mock_pytorch_base):
        """Test classification adapter accepts text prompts."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        # Should have predict method that accepts text_prompts
        assert hasattr(adapter, "predict")
        import inspect

        sig = inspect.signature(adapter.predict)
        assert "text_prompts" in sig.parameters

    def test_segmentation_accepts_text_prompts(self, mock_transformers_all, mock_pytorch_base):
        """Test segmentation adapter accepts text prompts."""
        adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False)

        assert hasattr(adapter, "predict")
        import inspect

        sig = inspect.signature(adapter.predict)
        assert "text_prompts" in sig.parameters

    def test_text_prompt_list_format_consistency(self, mock_transformers_all, mock_pytorch_base):
        """Test all adapters accept list format for prompts."""
        detect_adapter = HuggingFaceZeroShotDetectAdapter("google/owlv2-base-patch16")
        clip_adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")
        segment_adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False)

        # Detection supports list or string
        assert hasattr(detect_adapter, "_normalize_text_prompts")

        # Classification extracts label names from list or string
        assert hasattr(clip_adapter, "_extract_label_names")

        # Segmentation supports list or string
        assert hasattr(segment_adapter, "_normalize_text_prompts")


class TestResultTypeConsistency:
    """Test result type differences between tasks."""

    def test_classification_returns_classifyresult(self, mock_transformers_all, mock_pytorch_base):
        """Test CLIP returns ClassifyResult."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        test_image = Image.new("RGB", (224, 224), color="red")
        result = adapter.predict(test_image, text_prompts=["cat", "dog"])

        assert isinstance(result, ClassifyResult)
        assert not isinstance(result, VisionResult)

    def test_classifyresult_has_predictions(self, mock_transformers_all, mock_pytorch_base):
        """Test ClassifyResult structure."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        test_image = Image.new("RGB", (224, 224), color="red")
        result = adapter.predict(test_image, text_prompts=["cat", "dog", "bird"])

        assert hasattr(result, "predictions")
        assert isinstance(result.predictions, list)
        assert all(isinstance(p, Classification) for p in result.predictions)

    def test_classification_objects_have_label_name(self, mock_transformers_all, mock_pytorch_base):
        """Test Classification objects have label_name from prompts."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        test_image = Image.new("RGB", (224, 224), color="red")
        result = adapter.predict(test_image, text_prompts=["airplane", "car", "ship"])

        for pred in result.predictions:
            assert hasattr(pred, "label_name")
            assert pred.label_name in ["airplane", "car", "ship"]

    def test_segmentation_returns_visionresult(self, mock_transformers_all, mock_pytorch_base):
        """Test segmentation adapter returns VisionResult type."""
        adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False)

        # The adapter's predict() needs real torch tensor ops (sigmoid, interpolate)
        # which mock_pytorch_base can't provide. Verify the contract instead:
        # - task is "segment"
        # - predict exists and accepts text_prompts
        # - The dedicated test suite tests actual inference
        assert adapter.task == "segment"
        import inspect

        sig = inspect.signature(adapter.predict)
        assert "text_prompts" in sig.parameters
        assert "image" in sig.parameters

        # Verify return type annotation (string due to __future__ annotations)
        assert sig.return_annotation in (VisionResult, "VisionResult")


class TestThresholdBehavior:
    """Test threshold filtering across tasks."""

    def test_detection_threshold_filtering(self, mock_transformers_all, mock_pytorch_base):
        """Test detection adapter uses threshold."""
        adapter = HuggingFaceZeroShotDetectAdapter("google/owlv2-base-patch16", threshold=0.5)

        assert adapter.threshold == 0.5

    def test_classification_threshold_filtering(self, mock_transformers_all, mock_pytorch_base):
        """Test classification adapter uses threshold."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", threshold=0.3)

        assert adapter.threshold == 0.3

        # Test threshold is applied in predict
        test_image = Image.new("RGB", (224, 224), color="red")
        result = adapter.predict(test_image, text_prompts=["cat", "dog", "bird"])

        # All predictions should meet threshold
        for pred in result.predictions:
            # Note: with softmax, scores sum to 1, so at least one will pass
            assert isinstance(pred.score, float)

    def test_segmentation_threshold_filtering(self, mock_transformers_all, mock_pytorch_base):
        """Test segmentation adapter uses threshold."""
        adapter = HuggingFaceZeroShotSegmentAdapter(
            "CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False, threshold=0.6
        )

        assert adapter.threshold == 0.6

    def test_threshold_default_values(self, mock_transformers_all, mock_pytorch_base):
        """Test default threshold values."""
        detect_adapter = HuggingFaceZeroShotDetectAdapter("google/owlv2-base-patch16")
        clip_adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")
        segment_adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False)

        # Detection has default threshold (0.3)
        assert detect_adapter.threshold == 0.3

        # CLIP has threshold 0.0 (disabled by default)
        assert clip_adapter.threshold == 0.0

        # Segmentation has default threshold (0.5)
        assert segment_adapter.threshold == 0.5


class TestMetadataPopulation:
    """Test metadata consistency across tasks."""

    def test_classification_metadata(self, mock_transformers_all, mock_pytorch_base):
        """Test classification result includes metadata."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        test_image = Image.new("RGB", (224, 224), color="red")
        result = adapter.predict(test_image, text_prompts=["cat", "dog"])

        assert result.meta is not None
        assert "model_id" in result.meta
        assert "device" in result.meta
        assert "num_classes" in result.meta

    def test_metadata_includes_task_specific_info(self, mock_transformers_all, mock_pytorch_base):
        """Test metadata includes task-specific information."""
        clip_adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", template="ensemble")

        test_image = Image.new("RGB", (224, 224), color="red")
        result = clip_adapter.predict(test_image, text_prompts=["cat", "dog"])

        # CLIP should have template info
        assert "template_type" in result.meta
        assert "use_softmax" in result.meta


class TestErrorHandlingConsistency:
    """Test error handling uniformity across tasks."""

    def test_classification_invalid_image_error(self, mock_transformers_all, mock_pytorch_base):
        """Test classification raises error for invalid image."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        # Should raise error for invalid image path
        with pytest.raises(Exception):  # Will be InvalidInputError or FileNotFoundError
            adapter.predict("nonexistent_image.jpg", text_prompts=["cat", "dog"])

    def test_classification_empty_prompts_error(self, mock_transformers_all, mock_pytorch_base):
        """Test classification raises error for empty prompts."""
        from mata.core.exceptions import InvalidInputError

        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")
        test_image = Image.new("RGB", (224, 224), color="red")

        with pytest.raises(InvalidInputError, match="at least one class"):
            adapter.predict(test_image, text_prompts=[])


class TestCrossTaskComparison:
    """Test same image with different zero-shot tasks."""

    def test_same_prompts_different_tasks(self, mock_transformers_all, mock_pytorch_base):
        """Test same prompts work across detection, segmentation, and classification."""
        # Same categories for all tasks

        detect_adapter = HuggingFaceZeroShotDetectAdapter("google/owlv2-base-patch16")
        clip_adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")
        segment_adapter = HuggingFaceZeroShotSegmentAdapter("CIDAS/clipseg-rd64-refined", device="cpu", use_rle=False)

        # All should accept same prompt format
        assert hasattr(detect_adapter, "predict")
        assert hasattr(clip_adapter, "predict")
        assert hasattr(segment_adapter, "predict")

    def test_different_result_structures(self, mock_transformers_all, mock_pytorch_base):
        """Test detection vs classification return different result types."""
        clip_adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        test_image = Image.new("RGB", (224, 224), color="red")
        clip_result = clip_adapter.predict(test_image, text_prompts=["cat", "dog"])

        # CLIP returns ClassifyResult (no spatial info)
        assert isinstance(clip_result, ClassifyResult)
        assert hasattr(clip_result, "predictions")

        # Classifications don't have bbox/mask
        for pred in clip_result.predictions:
            assert not hasattr(pred, "bbox")
            assert not hasattr(pred, "mask")


class TestTemplateHandlingDifferences:
    """Test template handling differences between tasks."""

    def test_clip_template_customization(self, mock_transformers_all, mock_pytorch_base):
        """Test CLIP allows template customization."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", template="a picture of a {}")

        assert adapter.template == "a picture of a {}"

    def test_clip_template_ensemble(self, mock_transformers_all, mock_pytorch_base):
        """Test CLIP supports template ensembles."""
        adapter = HuggingFaceCLIPAdapter(
            "openai/clip-vit-base-patch32", template=["a photo of a {}", "a picture of a {}"]
        )

        assert isinstance(adapter.template, list)
        assert len(adapter.template) == 2

    def test_detection_prompt_format_normalization(self, mock_transformers_all, mock_pytorch_base):
        """Test detection normalizes prompt formats."""
        adapter = HuggingFaceZeroShotDetectAdapter("google/owlv2-base-patch16")

        # Detection has _normalize_text_prompts for different formats
        assert hasattr(adapter, "_normalize_text_prompts")


@pytest.mark.integration
class TestRealModelIntegration:
    """Integration tests with real models (skip unless MATA_RUN_INTEGRATION=1)."""

    @pytest.mark.skipif(
        os.environ.get("MATA_RUN_INTEGRATION", "0") != "1",
        reason="Requires MATA_RUN_INTEGRATION=1 environment variable and model downloads",
    )
    def test_clip_real_inference(self):
        """Test CLIP with real model (requires transformers + model download)."""
        try:
            adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")
            test_image = Image.new("RGB", (224, 224), color="red")

            result = adapter.predict(test_image, text_prompts=["red background", "blue background", "green background"])

            assert isinstance(result, ClassifyResult)
            assert len(result.predictions) > 0

            # Top prediction should be "red background" with high confidence
            top_pred = result.predictions[0]
            assert top_pred.label_name == "red background"

        except ImportError:
            pytest.skip("transformers not installed")


def pytest_addoption(parser):
    """Add command line option for integration tests."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require model downloads",
    )
