"""Unit tests for HuggingFaceSegmentAdapter."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from mata.core.exceptions import UnsupportedModelError


class TestHuggingFaceSegmentAdapter:
    """Test HuggingFaceSegmentAdapter initialization and methods."""

    @patch("mata.adapters.huggingface_segment_adapter._ensure_transformers")
    @patch("mata.adapters.huggingface_segment_adapter._ensure_pycocotools")
    def test_adapter_initialization_instance_mode(self, mock_pycocotools, mock_transformers):
        """Test adapter initialization for instance segmentation."""
        # Mock transformers
        mock_config = Mock()
        mock_config.model_type = "mask2former"
        mock_config.id2label = {0: "person", 1: "car"}

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=Mock())),
            "Mask2FormerForUniversalSegmentation": Mock(from_pretrained=Mock(return_value=Mock())),
        }
        mock_transformers.return_value = mock_transformers_dict

        # Mock pycocotools
        mock_pycocotools.return_value = Mock()

        from mata.adapters.huggingface_segment_adapter import HuggingFaceSegmentAdapter

        # Create adapter
        adapter = HuggingFaceSegmentAdapter(
            model_id="facebook/mask2former-swin-tiny-coco-instance",
            device="cpu",
            threshold=0.5,
            segment_mode="instance",
        )

        assert adapter.model_id == "facebook/mask2former-swin-tiny-coco-instance"
        assert adapter.threshold == 0.5
        assert adapter.segment_mode == "instance"
        assert adapter.active_mode == "instance"

    @patch("mata.adapters.huggingface_segment_adapter._ensure_transformers")
    @patch("mata.adapters.huggingface_segment_adapter._ensure_pycocotools")
    def test_adapter_initialization_panoptic_mode(self, mock_pycocotools, mock_transformers):
        """Test adapter initialization for panoptic segmentation."""
        # Mock transformers with is_thing_map
        mock_config = Mock()
        mock_config.model_type = "mask2former"
        mock_config.id2label = {0: "person", 80: "sky"}
        mock_config.is_thing_map = {0: True, 80: False}  # person=instance, sky=stuff

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=Mock())),
            "Mask2FormerForUniversalSegmentation": Mock(from_pretrained=Mock(return_value=Mock())),
        }
        mock_transformers.return_value = mock_transformers_dict

        mock_pycocotools.return_value = Mock()

        from mata.adapters.huggingface_segment_adapter import HuggingFaceSegmentAdapter

        adapter = HuggingFaceSegmentAdapter(
            model_id="facebook/mask2former-swin-tiny-coco-panoptic",
            device="cpu",
            threshold=0.5,
            segment_mode="panoptic",
        )

        assert adapter.active_mode == "panoptic"
        assert adapter.is_thing_map == {0: True, 80: False}

    @patch("mata.adapters.huggingface_segment_adapter._ensure_transformers")
    def test_adapter_initialization_auto_mode_detection(self, mock_transformers):
        """Test auto-detection of segmentation mode from model ID."""
        # Test panoptic detection
        mock_config_panoptic = Mock()
        mock_config_panoptic.model_type = "mask2former"
        mock_config_panoptic.id2label = {}
        mock_config_panoptic.is_thing_map = {0: True, 80: False}  # Mock as dict

        mock_model = Mock()
        mock_model.to.return_value.eval.return_value = Mock()

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config_panoptic)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=Mock())),
            "Mask2FormerForUniversalSegmentation": Mock(from_pretrained=Mock(return_value=mock_model)),
        }
        mock_transformers.return_value = mock_transformers_dict

        from mata.adapters.huggingface_segment_adapter import HuggingFaceSegmentAdapter

        adapter = HuggingFaceSegmentAdapter(
            model_id="facebook/mask2former-swin-tiny-coco-panoptic", device="cpu", segment_mode="auto"
        )
        assert adapter.active_mode == "panoptic"

        # Test instance detection
        mock_config_instance = Mock()
        mock_config_instance.model_type = "mask2former"
        mock_config_instance.id2label = {}

        mock_transformers_dict["AutoConfig"].from_pretrained.return_value = mock_config_instance

        adapter = HuggingFaceSegmentAdapter(
            model_id="facebook/mask2former-swin-tiny-coco-instance", device="cpu", segment_mode="auto"
        )
        assert adapter.active_mode == "instance"

    @patch("mata.adapters.huggingface_segment_adapter._ensure_transformers")
    def test_adapter_initialization_invalid_mode(self, mock_transformers):
        """Test adapter rejects invalid segment_mode."""
        from mata.adapters.huggingface_segment_adapter import HuggingFaceSegmentAdapter

        with pytest.raises(ValueError, match="segment_mode must be"):
            HuggingFaceSegmentAdapter(
                model_id="facebook/mask2former-swin-tiny-coco-instance", segment_mode="invalid_mode"
            )

    @patch("mata.adapters.huggingface_segment_adapter._ensure_transformers")
    def test_adapter_initialization_no_transformers(self, mock_transformers):
        """Test adapter raises ImportError when transformers not available."""
        mock_transformers.return_value = None

        from mata.adapters.huggingface_segment_adapter import HuggingFaceSegmentAdapter

        with pytest.raises(ImportError, match="transformers is required"):
            HuggingFaceSegmentAdapter(model_id="facebook/mask2former-swin-tiny-coco-instance")

    @patch("mata.adapters.huggingface_segment_adapter._ensure_transformers")
    @patch("mata.adapters.huggingface_segment_adapter._ensure_pycocotools")
    def test_adapter_use_rle_fallback(self, mock_pycocotools, mock_transformers):
        """Test adapter falls back to binary masks when pycocotools unavailable."""
        mock_config = Mock()
        mock_config.model_type = "mask2former"
        mock_config.id2label = {}

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=Mock())),
            "Mask2FormerForUniversalSegmentation": Mock(from_pretrained=Mock(return_value=Mock())),
        }
        mock_transformers.return_value = mock_transformers_dict

        # Pycocotools not available
        mock_pycocotools.return_value = None

        from mata.adapters.huggingface_segment_adapter import HuggingFaceSegmentAdapter

        with pytest.warns(UserWarning, match="pycocotools not available"):
            adapter = HuggingFaceSegmentAdapter(
                model_id="facebook/mask2former-swin-tiny-coco-instance", use_rle=True  # Requested, but will fallback
            )

        # Should fallback to binary
        assert adapter.use_rle is False

    def test_detect_architecture_mask2former(self):
        """Test architecture detection for Mask2Former."""
        from mata.adapters.huggingface_segment_adapter import HuggingFaceSegmentAdapter

        mock_config = Mock()
        mock_config.model_type = "mask2former"

        with patch("mata.adapters.huggingface_segment_adapter._ensure_transformers"):
            with patch("mata.adapters.huggingface_segment_adapter._ensure_pycocotools"):
                adapter = HuggingFaceSegmentAdapter.__new__(HuggingFaceSegmentAdapter)
                arch, mode = adapter._detect_architecture(mock_config, "facebook/mask2former-swin-tiny-coco-instance")

        assert arch == "mask2former"
        assert mode == "instance"

    def test_detect_architecture_unsupported(self):
        """Test architecture detection raises error for unsupported models."""
        from mata.adapters.huggingface_segment_adapter import HuggingFaceSegmentAdapter

        mock_config = Mock()
        mock_config.model_type = "unknown_model"

        with patch("mata.adapters.huggingface_segment_adapter._ensure_transformers"):
            with patch("mata.adapters.huggingface_segment_adapter._ensure_pycocotools"):
                adapter = HuggingFaceSegmentAdapter.__new__(HuggingFaceSegmentAdapter)

                with pytest.raises(UnsupportedModelError, match="Could not detect segmentation architecture"):
                    adapter._detect_architecture(mock_config, "unknown/model")

    @patch("mata.adapters.huggingface_segment_adapter._ensure_transformers")
    @patch("mata.adapters.huggingface_segment_adapter._ensure_pycocotools")
    def test_mask_to_bbox(self, mock_pycocotools, mock_transformers):
        """Test bounding box computation from binary mask."""
        mock_config = Mock()
        mock_config.model_type = "mask2former"
        mock_config.id2label = {}

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=Mock())),
            "Mask2FormerForUniversalSegmentation": Mock(from_pretrained=Mock(return_value=Mock())),
        }
        mock_transformers.return_value = mock_transformers_dict
        mock_pycocotools.return_value = Mock()

        from mata.adapters.huggingface_segment_adapter import HuggingFaceSegmentAdapter

        adapter = HuggingFaceSegmentAdapter(model_id="facebook/mask2former-swin-tiny-coco-instance", device="cpu")

        # Create test mask: 5x5 with object in center
        mask = np.array(
            [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=bool
        )

        bbox = adapter._mask_to_bbox(mask)

        # Expected: x_min=1, y_min=1, x_max=3, y_max=3
        assert bbox == (1.0, 1.0, 3.0, 3.0)

    @patch("mata.adapters.huggingface_segment_adapter._ensure_transformers")
    @patch("mata.adapters.huggingface_segment_adapter._ensure_pycocotools")
    def test_mask_to_bbox_empty_mask(self, mock_pycocotools, mock_transformers):
        """Test bounding box computation returns None for empty mask."""
        mock_config = Mock()
        mock_config.model_type = "mask2former"
        mock_config.id2label = {}

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=Mock())),
            "Mask2FormerForUniversalSegmentation": Mock(from_pretrained=Mock(return_value=Mock())),
        }
        mock_transformers.return_value = mock_transformers_dict
        mock_pycocotools.return_value = Mock()

        from mata.adapters.huggingface_segment_adapter import HuggingFaceSegmentAdapter

        adapter = HuggingFaceSegmentAdapter(model_id="facebook/mask2former-swin-tiny-coco-instance", device="cpu")

        # Empty mask
        mask = np.zeros((5, 5), dtype=bool)

        bbox = adapter._mask_to_bbox(mask)

        assert bbox is None

    @patch("mata.adapters.huggingface_segment_adapter._ensure_transformers")
    @patch("mata.adapters.huggingface_segment_adapter._ensure_pycocotools")
    def test_info_method(self, mock_pycocotools, mock_transformers):
        """Test info() method returns correct metadata."""
        mock_config = Mock()
        mock_config.model_type = "mask2former"

        # Set id2label - real dict that will be copied
        id2label_dict = {0: "person", 1: "car"}
        mock_config.configure_mock(**{"id2label": id2label_dict})

        # Make the mock model chainable for .to().eval()
        mock_model = Mock()
        mock_eval = Mock()
        mock_model.to.return_value = Mock(eval=Mock(return_value=mock_eval))

        mock_transformers_dict = {
            "AutoConfig": Mock(from_pretrained=Mock(return_value=mock_config)),
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=Mock())),
            "Mask2FormerForUniversalSegmentation": Mock(from_pretrained=Mock(return_value=mock_model)),
        }
        mock_transformers.return_value = mock_transformers_dict
        mock_pycocotools.return_value = Mock()

        from mata.adapters.huggingface_segment_adapter import HuggingFaceSegmentAdapter

        # Pass id2label directly to force the adapter to use it
        adapter = HuggingFaceSegmentAdapter(
            model_id="facebook/mask2former-swin-tiny-coco-instance",
            device="cpu",
            threshold=0.7,
            use_rle=False,
            id2label=id2label_dict,  # Explicitly pass it
        )

        info = adapter.info()

        assert info["name"] == "huggingface_segment"
        assert info["task"] == "segment"
        assert info["model_id"] == "facebook/mask2former-swin-tiny-coco-instance"
        assert info["mode"] == "instance"
        assert info["threshold"] == 0.7
        assert info["use_rle"] is False
        assert info["num_classes"] == 2
        assert info["backend"] == "transformers"
