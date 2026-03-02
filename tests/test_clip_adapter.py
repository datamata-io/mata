"""Tests for HuggingFaceCLIPAdapter (zero-shot image classification)."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image

from mata.adapters.clip_adapter import TEMPLATE_SETS, HuggingFaceCLIPAdapter
from mata.core.exceptions import InvalidInputError
from mata.core.types import Classification, ClassifyResult


@pytest.fixture
def mock_transformers_clip():
    """Mock transformers library components for CLIP."""
    with patch("mata.adapters.clip_adapter._ensure_transformers") as mock_ensure:
        # Import torch for real tensors
        import torch

        # Create mock classes
        mock_processor_class = Mock()
        mock_model_class = Mock()

        # Track number of prompts for dynamic logits
        num_prompts_tracker = {"count": 3}  # Default

        # Mock processor instance that returns real tensors
        mock_processor = Mock()

        def processor_call(*args, **kwargs):
            """Return inputs with real tensors that have .to() method."""
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

        mock_processor.side_effect = processor_call

        # Mock model instance with dynamic logits
        def model_call(**inputs):
            """Return mock outputs with dynamic logits shape based on tracked prompt count."""
            mock_outputs = Mock()
            # Use tracked count for logits shape
            mock_outputs.logits_per_image = torch.randn(1, num_prompts_tracker["count"])
            return mock_outputs

        mock_model = Mock()
        mock_model.side_effect = model_call
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        # Set up from_pretrained
        mock_processor_class.from_pretrained = Mock(return_value=mock_processor)
        mock_model_class.from_pretrained = Mock(return_value=mock_model)

        # Return dict with mocked classes
        mock_ensure.return_value = {
            "CLIPProcessor": mock_processor_class,
            "CLIPModel": mock_model_class,
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
        # Import real torch for device compatibility
        import torch

        mock_torch = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        # Use real torch.device for tensor.to() compatibility
        mock_torch.device = torch.device
        mock_torch.no_grad = MagicMock()
        mock_ensure.return_value = mock_torch
        yield mock_torch


class TestCLIPAdapterInitialization:
    """Test CLIP adapter initialization."""

    def test_init_basic(self, mock_transformers_clip, mock_pytorch_base):
        """Test basic initialization with default parameters."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        assert adapter.model_id == "openai/clip-vit-base-patch32"
        assert adapter.top_k == 5
        assert adapter.threshold == 0.0
        assert adapter.use_softmax
        assert adapter.template == "a photo of a {}"

    def test_init_custom_template_string(self, mock_transformers_clip, mock_pytorch_base):
        """Test initialization with custom template string."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", template="this is a {}")

        assert adapter.template == "this is a {}"

    def test_init_custom_template_list(self, mock_transformers_clip, mock_pytorch_base):
        """Test initialization with custom template list."""
        templates = ["a photo of a {}", "a picture of a {}"]
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", template=templates)

        assert adapter.template == templates
        assert isinstance(adapter.template, list)
        assert len(adapter.template) == 2

    def test_init_template_shortcut_basic(self, mock_transformers_clip, mock_pytorch_base):
        """Test initialization with 'basic' template shortcut."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", template="basic")

        assert adapter.template == TEMPLATE_SETS["basic"]
        assert isinstance(adapter.template, list)

    def test_init_template_shortcut_ensemble(self, mock_transformers_clip, mock_pytorch_base):
        """Test initialization with 'ensemble' template shortcut."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", template="ensemble")

        assert adapter.template == TEMPLATE_SETS["ensemble"]
        assert len(adapter.template) == 6

    def test_init_use_softmax_false(self, mock_transformers_clip, mock_pytorch_base):
        """Test initialization with use_softmax=False."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", use_softmax=False)

        assert not adapter.use_softmax

    def test_init_threshold_and_top_k(self, mock_transformers_clip, mock_pytorch_base):
        """Test initialization with custom threshold and top_k."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", threshold=0.1, top_k=3)

        assert adapter.threshold == 0.1
        assert adapter.top_k == 3

    def test_init_invalid_template_shortcut(self, mock_transformers_clip, mock_pytorch_base):
        """Test initialization with invalid template shortcut raises error."""
        # Template doesn't exist in TEMPLATE_SETS and has no {}
        with pytest.raises(ValueError, match="must contain"):
            HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", template="invalid_shortcut_no_placeholder")

    def test_init_device_placement(self, mock_transformers_clip, mock_pytorch_base):
        """Test model is placed on correct device."""
        HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", device="cpu")

        # Verify model.to() was called
        assert mock_transformers_clip["model"].to.called


class TestCLIPTemplateHandling:
    """Test template handling and resolution."""

    def test_template_sets_structure(self):
        """Test TEMPLATE_SETS has expected structure."""
        assert "basic" in TEMPLATE_SETS
        assert "ensemble" in TEMPLATE_SETS
        assert "detailed" in TEMPLATE_SETS

        # All templates should have {} placeholder
        for name, templates in TEMPLATE_SETS.items():
            assert isinstance(templates, list)
            for tmpl in templates:
                assert "{}" in tmpl

    def test_resolve_template_shortcut_basic(self, mock_transformers_clip, mock_pytorch_base):
        """Test resolving 'basic' shortcut."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", template="basic")

        resolved = adapter._resolve_template("basic")
        assert resolved == TEMPLATE_SETS["basic"]

    def test_resolve_template_custom_string(self, mock_transformers_clip, mock_pytorch_base):
        """Test resolving custom template string."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        resolved = adapter._resolve_template("a custom {}")
        assert resolved == "a custom {}"

    def test_resolve_template_custom_list(self, mock_transformers_clip, mock_pytorch_base):
        """Test resolving custom template list."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        custom_templates = ["template1 {}", "template2 {}"]
        resolved = adapter._resolve_template(custom_templates)
        assert resolved == custom_templates

    def test_resolve_template_empty_list_error(self, mock_transformers_clip, mock_pytorch_base):
        """Test empty template list raises error."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        with pytest.raises(ValueError, match="cannot be empty"):
            adapter._resolve_template([])

    def test_resolve_template_missing_placeholder_error(self, mock_transformers_clip, mock_pytorch_base):
        """Test template without {} placeholder raises error."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        with pytest.raises(ValueError, match="must contain"):
            adapter._resolve_template("no placeholder here")

    def test_format_text_prompts_single_template(self, mock_transformers_clip, mock_pytorch_base):
        """Test formatting prompts with single template."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", template="a photo of a {}")

        formatted = adapter._format_text_prompts(["cat", "dog"])
        assert formatted == ["a photo of a cat", "a photo of a dog"]

    def test_format_text_prompts_ensemble(self, mock_transformers_clip, mock_pytorch_base):
        """Test formatting prompts with ensemble templates."""
        adapter = HuggingFaceCLIPAdapter(
            "openai/clip-vit-base-patch32", template=["a photo of a {}", "a picture of a {}"]
        )

        formatted = adapter._format_text_prompts(["cat", "dog"])
        # Should have 2 classes × 2 templates = 4 formatted prompts
        assert len(formatted) == 4
        assert "a photo of a cat" in formatted
        assert "a picture of a cat" in formatted
        assert "a photo of a dog" in formatted
        assert "a picture of a dog" in formatted

    def test_extract_label_names_from_list(self, mock_transformers_clip, mock_pytorch_base):
        """Test extracting label names from list."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        labels = adapter._extract_label_names(["cat", "dog", "bird"])
        assert labels == ["cat", "dog", "bird"]

    def test_extract_label_names_from_string(self, mock_transformers_clip, mock_pytorch_base):
        """Test extracting label names from comma-separated string."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        labels = adapter._extract_label_names("cat, dog, bird")
        assert labels == ["cat", "dog", "bird"]


class TestCLIPPredict:
    """Test CLIP prediction functionality."""

    def test_predict_basic(self, mock_transformers_clip, mock_pytorch_base):
        """Test basic prediction with default parameters."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        # Create test image
        test_image = Image.new("RGB", (224, 224), color="red")

        result = adapter.predict(test_image, text_prompts=["cat", "dog", "bird"])

        assert isinstance(result, ClassifyResult)
        assert len(result.predictions) <= 3
        assert all(isinstance(p, Classification) for p in result.predictions)

        # Predictions should be sorted by score (descending)
        for i in range(len(result.predictions) - 1):
            assert result.predictions[i].score >= result.predictions[i + 1].score

    def test_predict_with_softmax(self, mock_transformers_clip, mock_pytorch_base):
        """Test prediction with softmax enabled."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", use_softmax=True)

        test_image = Image.new("RGB", (224, 224), color="red")
        result = adapter.predict(test_image, text_prompts=["cat", "dog", "bird"])

        # With softmax, scores should sum to ~1.0
        total_score = sum(p.score for p in result.predictions)
        assert abs(total_score - 1.0) < 0.01

        # Scores should be in [0, 1]
        for pred in result.predictions:
            assert 0.0 <= pred.score <= 1.0

    def test_predict_without_softmax(self, mock_transformers_clip, mock_pytorch_base):
        """Test prediction with softmax disabled (raw similarities)."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", use_softmax=False)

        test_image = Image.new("RGB", (224, 224), color="red")
        result = adapter.predict(test_image, text_prompts=["cat", "dog", "bird"], use_softmax=False)

        # Without softmax, scores might not sum to 1.0
        # (they're raw similarity scores from CLIP)
        assert isinstance(result, ClassifyResult)
        assert not result.meta["use_softmax"]

    def test_predict_threshold_filtering(self, mock_transformers_clip, mock_pytorch_base):
        """Test threshold filtering."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", threshold=0.5)  # High threshold

        test_image = Image.new("RGB", (224, 224), color="red")
        result = adapter.predict(test_image, text_prompts=["cat", "dog", "bird"])

        # All predictions should have score >= threshold
        for pred in result.predictions:
            assert pred.score >= 0.5

    def test_predict_top_k_selection(self, mock_transformers_clip, mock_pytorch_base):
        """Test top-k selection."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", top_k=2)

        test_image = Image.new("RGB", (224, 224), color="red")
        result = adapter.predict(test_image, text_prompts=["cat", "dog", "bird", "fish", "horse"])

        # Should return at most top_k predictions
        assert len(result.predictions) <= 2

    def test_predict_combined_threshold_and_top_k(self, mock_transformers_clip, mock_pytorch_base):
        """Test combined threshold and top-k filtering."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", threshold=0.1, top_k=3)

        test_image = Image.new("RGB", (224, 224), color="red")
        result = adapter.predict(test_image, text_prompts=["cat", "dog", "bird", "fish", "horse"])

        # Should apply threshold first, then top-k
        assert len(result.predictions) <= 3
        for pred in result.predictions:
            assert pred.score >= 0.1

    def test_predict_ensemble_template_averaging(self, mock_transformers_clip, mock_pytorch_base):
        """Test ensemble template averaging."""
        adapter = HuggingFaceCLIPAdapter(
            "openai/clip-vit-base-patch32", template=["a photo of a {}", "a picture of a {}"]
        )

        test_image = Image.new("RGB", (224, 224), color="red")

        # Mock will automatically generate logits for 2 classes × 2 templates = 4 prompts
        result = adapter.predict(test_image, text_prompts=["cat", "dog"])

        # Should average across templates
        assert isinstance(result, ClassifyResult)
        assert result.meta["template_type"] == "ensemble"
        assert result.meta["num_templates"] == 2

    def test_predict_with_pil_image(self, mock_transformers_clip, mock_pytorch_base):
        """Test prediction with PIL Image input."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        test_image = Image.new("RGB", (224, 224), color="blue")
        result = adapter.predict(test_image, text_prompts=["cat", "dog"])

        assert isinstance(result, ClassifyResult)

    def test_predict_parameter_override(self, mock_transformers_clip, mock_pytorch_base):
        """Test runtime parameter override."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", top_k=5, threshold=0.0, use_softmax=True)

        test_image = Image.new("RGB", (224, 224), color="green")
        result = adapter.predict(
            test_image,
            text_prompts=["cat", "dog", "bird"],
            top_k=2,  # Override
            threshold=0.2,  # Override
            use_softmax=False,  # Override
        )

        # Should use overridden values
        assert len(result.predictions) <= 2
        assert result.meta["top_k"] == 2
        assert result.meta["threshold"] == 0.2
        assert not result.meta["use_softmax"]

    def test_predict_metadata_population(self, mock_transformers_clip, mock_pytorch_base):
        """Test metadata is properly populated."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", template="ensemble")

        test_image = Image.new("RGB", (224, 224), color="yellow")
        result = adapter.predict(test_image, text_prompts=["cat", "dog", "bird"])

        assert "model_id" in result.meta
        assert "device" in result.meta
        assert "num_classes" in result.meta
        assert "template_type" in result.meta
        assert "num_templates" in result.meta
        assert "use_softmax" in result.meta
        assert "threshold" in result.meta
        assert "top_k" in result.meta

        assert result.meta["model_id"] == "openai/clip-vit-base-patch32"
        assert result.meta["num_classes"] == 3

    def test_predict_empty_prompts_error(self, mock_transformers_clip, mock_pytorch_base):
        """Test prediction with empty prompts raises error."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        test_image = Image.new("RGB", (224, 224), color="white")

        with pytest.raises(InvalidInputError, match="at least one class"):
            adapter.predict(test_image, text_prompts=[])


class TestCLIPResultValidation:
    """Test CLIP result structure and validation."""

    def test_result_structure(self, mock_transformers_clip, mock_pytorch_base):
        """Test ClassifyResult structure is correct."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        test_image = Image.new("RGB", (224, 224), color="red")
        result = adapter.predict(test_image, text_prompts=["cat", "dog"])

        assert isinstance(result, ClassifyResult)
        assert hasattr(result, "predictions")
        assert hasattr(result, "meta")
        assert isinstance(result.predictions, list)
        assert isinstance(result.meta, dict)

    def test_classification_objects(self, mock_transformers_clip, mock_pytorch_base):
        """Test Classification objects have correct fields."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        test_image = Image.new("RGB", (224, 224), color="red")
        result = adapter.predict(test_image, text_prompts=["cat", "dog", "bird"])

        for pred in result.predictions:
            assert isinstance(pred, Classification)
            assert hasattr(pred, "label")
            assert hasattr(pred, "score")
            assert hasattr(pred, "label_name")

            assert isinstance(pred.label, int)
            assert isinstance(pred.score, float)
            assert pred.label_name in ["cat", "dog", "bird"]

    def test_dynamic_id2label(self, mock_transformers_clip, mock_pytorch_base):
        """Test id2label is built dynamically from prompts."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        test_image = Image.new("RGB", (224, 224), color="red")
        result = adapter.predict(test_image, text_prompts=["airplane", "automobile", "ship"])

        # Label names should match input prompts
        label_names = {pred.label_name for pred in result.predictions}
        assert label_names.issubset({"airplane", "automobile", "ship"})

    def test_result_serialization(self, mock_transformers_clip, mock_pytorch_base):
        """Test ClassifyResult can be serialized to JSON."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")

        test_image = Image.new("RGB", (224, 224), color="red")
        result = adapter.predict(test_image, text_prompts=["cat", "dog"])

        # Should have to_json method
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "predictions" in json_str


class TestCLIPAdapterInfo:
    """Test adapter info method."""

    def test_info_basic(self, mock_transformers_clip, mock_pytorch_base):
        """Test info method returns correct metadata."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", top_k=3, threshold=0.1)

        info = adapter.info()

        assert info["name"] == "HuggingFaceCLIPAdapter"
        assert info["task"] == "classify"
        assert info["model_id"] == "openai/clip-vit-base-patch32"
        assert info["architecture"] == "clip"
        assert info["backend"] == "transformers"
        assert info["top_k"] == 3
        assert info["threshold"] == 0.1
        assert info["use_softmax"]

    def test_info_with_ensemble(self, mock_transformers_clip, mock_pytorch_base):
        """Test info method with ensemble template."""
        adapter = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32", template="ensemble")

        info = adapter.info()

        assert "template" in info
        assert "ensemble" in info["template"]
        assert "6 templates" in info["template"]
