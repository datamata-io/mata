"""Unit tests for HuggingFaceClassifyAdapter."""

from unittest.mock import Mock, patch

import pytest


class TestHuggingFaceClassifyAdapter:
    """Test HuggingFaceClassifyAdapter initialization and methods."""

    @patch("mata.adapters.huggingface_classify_adapter._ensure_transformers")
    def test_adapter_initialization_resnet(self, mock_transformers):
        """Test adapter initialization with ResNet model."""
        # Mock transformers
        mock_config = Mock()
        mock_config.model_type = "resnet"
        mock_config.num_labels = 1000
        mock_config.id2label = {i: f"class_{i}" for i in range(1000)}

        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=Mock())),
            "AutoModelForImageClassification": Mock(from_pretrained=Mock(return_value=mock_model)),
        }
        mock_transformers.return_value = mock_transformers_dict

        from mata.adapters.huggingface_classify_adapter import HuggingFaceClassifyAdapter

        # Create adapter
        adapter = HuggingFaceClassifyAdapter(model_id="microsoft/resnet-50", device="cpu", top_k=5)

        assert adapter.model_id == "microsoft/resnet-50"
        assert adapter.top_k == 5
        assert adapter.device.type == "cpu"
        assert len(adapter.id2label) == 1000

    @patch("mata.adapters.huggingface_classify_adapter._ensure_transformers")
    def test_adapter_initialization_vit(self, mock_transformers):
        """Test adapter initialization with ViT model."""
        mock_config = Mock()
        mock_config.model_type = "vit"
        mock_config.num_labels = 1000
        mock_config.id2label = {281: "tabby cat", 282: "tiger cat"}

        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=Mock())),
            "AutoModelForImageClassification": Mock(from_pretrained=Mock(return_value=mock_model)),
        }
        mock_transformers.return_value = mock_transformers_dict

        from mata.adapters.huggingface_classify_adapter import HuggingFaceClassifyAdapter

        adapter = HuggingFaceClassifyAdapter(model_id="google/vit-base-patch16-224", device="cpu")

        assert adapter.model_id == "google/vit-base-patch16-224"
        assert adapter.id2label[281] == "tabby cat"

    @patch("mata.adapters.huggingface_classify_adapter._ensure_transformers")
    def test_adapter_initialization_custom_id2label(self, mock_transformers):
        """Test adapter initialization with custom label mapping."""
        mock_config = Mock()
        mock_config.model_type = "resnet"
        mock_config.num_labels = 10

        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=Mock())),
            "AutoModelForImageClassification": Mock(from_pretrained=Mock(return_value=mock_model)),
        }
        mock_transformers.return_value = mock_transformers_dict

        from mata.adapters.huggingface_classify_adapter import HuggingFaceClassifyAdapter

        # Custom label mapping
        custom_labels = {0: "cat", 1: "dog", 2: "bird"}

        adapter = HuggingFaceClassifyAdapter(model_id="custom/model", device="cpu", id2label=custom_labels)

        assert adapter.id2label == custom_labels

    @patch("mata.adapters.huggingface_classify_adapter._ensure_transformers")
    def test_adapter_initialization_no_transformers(self, mock_transformers):
        """Test adapter raises ImportError when transformers not available."""
        mock_transformers.return_value = None

        from mata.adapters.huggingface_classify_adapter import HuggingFaceClassifyAdapter

        with pytest.raises(ImportError, match="transformers is required"):
            HuggingFaceClassifyAdapter(model_id="microsoft/resnet-50")

    @patch("mata.adapters.huggingface_classify_adapter._ensure_transformers")
    def test_adapter_detect_architecture_resnet(self, mock_transformers):
        """Test architecture detection for ResNet models."""
        mock_config = Mock()
        mock_config.model_type = "resnet"
        mock_config.num_labels = 1000
        mock_config.id2label = {}

        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=Mock())),
            "AutoModelForImageClassification": Mock(from_pretrained=Mock(return_value=mock_model)),
        }
        mock_transformers.return_value = mock_transformers_dict

        from mata.adapters.huggingface_classify_adapter import HuggingFaceClassifyAdapter

        adapter = HuggingFaceClassifyAdapter(model_id="microsoft/resnet-50", device="cpu")
        arch = adapter._detect_architecture()

        assert arch == "resnet"

    @patch("mata.adapters.huggingface_classify_adapter._ensure_transformers")
    def test_adapter_detect_architecture_vit(self, mock_transformers):
        """Test architecture detection for ViT models."""
        mock_config = Mock()
        mock_config.model_type = "vit"
        mock_config.num_labels = 1000
        mock_config.id2label = {}

        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=Mock())),
            "AutoModelForImageClassification": Mock(from_pretrained=Mock(return_value=mock_model)),
        }
        mock_transformers.return_value = mock_transformers_dict

        from mata.adapters.huggingface_classify_adapter import HuggingFaceClassifyAdapter

        adapter = HuggingFaceClassifyAdapter(model_id="google/vit-base-patch16-224", device="cpu")
        arch = adapter._detect_architecture()

        assert arch == "vit"

    @patch("mata.adapters.huggingface_classify_adapter._ensure_transformers")
    def test_predict_basic(self, mock_transformers):
        """Test basic prediction with PIL Image."""
        import torch
        from PIL import Image

        # Mock config and model
        mock_config = Mock()
        mock_config.model_type = "resnet"
        mock_config.num_labels = 1000
        mock_config.id2label = {281: "tabby cat", 282: "tiger cat", 285: "Egyptian cat"}

        # Mock model outputs
        mock_outputs = Mock()
        logits = torch.tensor(
            [
                [
                    -1.0,
                    -2.0,
                    -0.5,  # indices 0, 1, 2
                    *[-3.0] * 278,  # indices 3-280
                    2.5,  # index 281 (highest - tabby cat)
                    1.5,  # index 282 (second - tiger cat)
                    *[-3.0] * 2,  # indices 283-284
                    0.8,  # index 285 (third - Egyptian cat)
                    *[-3.0] * 714,  # remaining indices
                ]
            ]
        )
        mock_outputs.logits = logits

        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.return_value = mock_outputs

        # Mock processor
        mock_processor = Mock()
        mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=mock_processor)),
            "AutoModelForImageClassification": Mock(from_pretrained=Mock(return_value=mock_model)),
        }
        mock_transformers.return_value = mock_transformers_dict

        from mata.adapters.huggingface_classify_adapter import HuggingFaceClassifyAdapter

        adapter = HuggingFaceClassifyAdapter(model_id="microsoft/resnet-50", device="cpu", top_k=3)

        # Create test image
        test_image = Image.new("RGB", (224, 224), color="red")

        result = adapter.predict(test_image)

        assert len(result.predictions) == 3
        assert result.predictions[0].label == 281
        assert result.predictions[0].label_name == "tabby cat"
        assert result.predictions[1].label == 282
        assert result.predictions[2].label == 285
        assert result.meta["model_id"] == "microsoft/resnet-50"
        assert result.meta["top_k"] == 3

    @patch("mata.adapters.huggingface_classify_adapter._ensure_transformers")
    def test_predict_with_runtime_top_k_override(self, mock_transformers):
        """Test prediction with runtime top_k override."""
        import torch
        from PIL import Image

        mock_config = Mock()
        mock_config.model_type = "resnet"
        mock_config.num_labels = 10
        mock_config.id2label = {i: f"class_{i}" for i in range(10)}

        # Create logits with clear ranking
        logits_values = [float(i) for i in range(10)]  # 9 is highest, 0 is lowest
        logits_values.reverse()  # Now 0 is highest

        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([logits_values])

        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.return_value = mock_outputs

        mock_processor = Mock()
        mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=mock_processor)),
            "AutoModelForImageClassification": Mock(from_pretrained=Mock(return_value=mock_model)),
        }
        mock_transformers.return_value = mock_transformers_dict

        from mata.adapters.huggingface_classify_adapter import HuggingFaceClassifyAdapter

        # Adapter with default top_k=5
        adapter = HuggingFaceClassifyAdapter(model_id="microsoft/resnet-50", device="cpu", top_k=5)

        test_image = Image.new("RGB", (224, 224))

        # Override top_k at runtime
        result = adapter.predict(test_image, top_k=3)

        assert len(result.predictions) == 3
        assert result.meta["top_k"] == 3

    @patch("mata.adapters.huggingface_classify_adapter._ensure_transformers")
    def test_predict_from_file_path(self, mock_transformers):
        """Test prediction from image file path."""
        import torch
        from PIL import Image

        mock_config = Mock()
        mock_config.model_type = "resnet"
        mock_config.num_labels = 5
        mock_config.id2label = {i: f"class_{i}" for i in range(5)}

        mock_outputs = Mock()
        mock_outputs.logits = torch.randn(1, 5)

        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.return_value = mock_outputs

        mock_processor = Mock()
        mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=mock_processor)),
            "AutoModelForImageClassification": Mock(from_pretrained=Mock(return_value=mock_model)),
        }
        mock_transformers.return_value = mock_transformers_dict

        from mata.adapters.huggingface_classify_adapter import HuggingFaceClassifyAdapter

        adapter = HuggingFaceClassifyAdapter(model_id="microsoft/resnet-50", device="cpu")

        # Mock _load_image to accept path (returns tuple: Image, path)
        with patch.object(adapter, "_load_image") as mock_load_image:
            mock_load_image.return_value = (Image.new("RGB", (224, 224)), "path/to/image.jpg")

            result = adapter.predict("path/to/image.jpg", top_k=2)

            assert len(result.predictions) == 2
            mock_load_image.assert_called_once_with("path/to/image.jpg")

    @patch("mata.adapters.huggingface_classify_adapter._ensure_transformers")
    def test_predict_scores_are_sorted(self, mock_transformers):
        """Test that predictions are sorted by score in descending order."""
        import torch
        from PIL import Image

        mock_config = Mock()
        mock_config.model_type = "resnet"
        mock_config.num_labels = 5
        mock_config.id2label = {i: f"class_{i}" for i in range(5)}

        # Logits in non-sorted order
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[1.0, 5.0, 2.0, 8.0, 3.0]])  # Sorted: [3, 1, 4, 2, 0]

        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.return_value = mock_outputs

        mock_processor = Mock()
        mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=mock_processor)),
            "AutoModelForImageClassification": Mock(from_pretrained=Mock(return_value=mock_model)),
        }
        mock_transformers.return_value = mock_transformers_dict

        from mata.adapters.huggingface_classify_adapter import HuggingFaceClassifyAdapter

        adapter = HuggingFaceClassifyAdapter(model_id="microsoft/resnet-50", device="cpu", top_k=5)

        result = adapter.predict(Image.new("RGB", (224, 224)))

        # Verify predictions are sorted by score (descending)
        scores = [p.score for p in result.predictions]
        assert scores == sorted(scores, reverse=True)

        # Highest score should be from index 3 (logit=8.0)
        assert result.predictions[0].label == 3
