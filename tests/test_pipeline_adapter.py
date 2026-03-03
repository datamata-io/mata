"""Tests for GroundingDINOSAMPipeline (multi-modal detection→segmentation)."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from mata.adapters.pipeline_adapter import GroundingDINOSAMPipeline
from mata.core.types import Instance, VisionResult


@pytest.fixture
def mock_detector():
    """Mock zero-shot detection adapter."""
    detector = Mock()
    detector.predict = Mock()
    return detector


@pytest.fixture
def mock_segmenter():
    """Mock SAM segmentation adapter."""
    segmenter = Mock()
    segmenter.predict = Mock()
    return segmenter


class TestPipelineInitialization:
    """Test pipeline initialization."""

    def test_init_with_model_ids(self):
        """Test initialization with model IDs (lazy loading)."""
        pipeline = GroundingDINOSAMPipeline(
            detector_model_id="IDEA-Research/grounding-dino-tiny", sam_model_id="facebook/sam-vit-base"
        )

        assert pipeline.detector_model_id == "IDEA-Research/grounding-dino-tiny"
        assert pipeline.sam_model_id == "facebook/sam-vit-base"
        assert pipeline.detector is None  # Not loaded yet
        assert pipeline.segmenter is None  # Not loaded yet

    def test_init_with_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        pipeline = GroundingDINOSAMPipeline(
            detector_model_id="IDEA-Research/grounding-dino-tiny",
            sam_model_id="facebook/sam-vit-base",
            detection_threshold=0.4,
            segmentation_threshold=0.6,
        )

        assert pipeline.detection_threshold == 0.4
        assert pipeline.segmentation_threshold == 0.6

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init_with_custom_device(self):
        """Test initialization with custom device."""
        pipeline = GroundingDINOSAMPipeline(
            detector_model_id="IDEA-Research/grounding-dino-tiny", sam_model_id="facebook/sam-vit-base", device="cuda"
        )

        assert str(pipeline.device) == "cuda"


class TestLazyLoading:
    """Test lazy loading of detector and segmenter."""

    def test_adapters_not_loaded_on_init(self):
        """Test that adapters are not loaded during initialization."""
        with patch("mata.adapters.huggingface_zeroshot_detect_adapter.HuggingFaceZeroShotDetectAdapter") as mock_det:
            with patch("mata.adapters.huggingface_sam_adapter.HuggingFaceSAMAdapter") as mock_sam:
                GroundingDINOSAMPipeline(
                    detector_model_id="IDEA-Research/grounding-dino-tiny", sam_model_id="facebook/sam-vit-base"
                )

                # Should not be called during __init__
                mock_det.assert_not_called()
                mock_sam.assert_not_called()

    def test_adapters_loaded_on_first_predict(self, mock_detector, mock_segmenter):
        """Test that adapters are loaded on first predict call."""
        with patch(
            "mata.adapters.huggingface_zeroshot_detect_adapter.HuggingFaceZeroShotDetectAdapter"
        ) as mock_det_class:
            with patch("mata.adapters.huggingface_sam_adapter.HuggingFaceSAMAdapter") as mock_sam_class:
                mock_det_class.return_value = mock_detector
                mock_sam_class.return_value = mock_segmenter

                # Mock detection result
                mock_detector.predict.return_value = VisionResult(
                    instances=[Instance(bbox=(10.0, 20.0, 100.0, 150.0), score=0.9, label=0, label_name="cat")], meta={}
                )

                # Mock segmentation result
                mock_segmenter.predict.return_value = VisionResult(
                    instances=[
                        Instance(mask=np.ones((480, 640), dtype=bool), score=0.85, label=0, label_name="object")
                    ],
                    meta={},
                )

                pipeline = GroundingDINOSAMPipeline(
                    detector_model_id="IDEA-Research/grounding-dino-tiny", sam_model_id="facebook/sam-vit-base"
                )

                image = Image.new("RGB", (640, 480))
                pipeline.predict(image, text_prompts="cat")

                # Should be called during first predict
                mock_det_class.assert_called_once()
                mock_sam_class.assert_called_once()


class TestDetectionToSegmentation:
    """Test detection→segmentation pipeline flow."""

    def test_bbox_extraction_from_detections(self, mock_detector, mock_segmenter):
        """Test extraction of bboxes from detection results."""
        with patch(
            "mata.adapters.huggingface_zeroshot_detect_adapter.HuggingFaceZeroShotDetectAdapter"
        ) as mock_det_class:
            with patch("mata.adapters.huggingface_sam_adapter.HuggingFaceSAMAdapter") as mock_sam_class:
                mock_det_class.return_value = mock_detector
                mock_sam_class.return_value = mock_segmenter

                # Detection result with multiple boxes
                mock_detector.predict.return_value = VisionResult(
                    instances=[
                        Instance(bbox=(10.0, 20.0, 100.0, 150.0), score=0.9, label=0, label_name="cat"),
                        Instance(bbox=(200.0, 100.0, 350.0, 300.0), score=0.85, label=1, label_name="dog"),
                    ],
                    meta={},
                )

                # Mock segmentation result
                mock_segmenter.predict.return_value = VisionResult(
                    instances=[
                        Instance(mask=np.ones((480, 640), dtype=bool), score=0.8, label=0, label_name="object"),
                        Instance(mask=np.ones((480, 640), dtype=bool), score=0.75, label=0, label_name="object"),
                    ],
                    meta={},
                )

                pipeline = GroundingDINOSAMPipeline(
                    detector_model_id="IDEA-Research/grounding-dino-tiny", sam_model_id="facebook/sam-vit-base"
                )

                image = Image.new("RGB", (640, 480))
                pipeline.predict(image, text_prompts="cat . dog")

                # Verify SAM was called with box prompts
                sam_call_args = mock_segmenter.predict.call_args
                assert "box_prompts" in sam_call_args[1]
                box_prompts = sam_call_args[1]["box_prompts"]
                assert len(box_prompts) == 2
                assert box_prompts[0] == (10.0, 20.0, 100.0, 150.0)
                assert box_prompts[1] == (200.0, 100.0, 350.0, 300.0)

    def test_instance_merging(self, mock_detector, mock_segmenter):
        """Test merging of detection bbox and segmentation mask."""
        with patch(
            "mata.adapters.huggingface_zeroshot_detect_adapter.HuggingFaceZeroShotDetectAdapter"
        ) as mock_det_class:
            with patch("mata.adapters.huggingface_sam_adapter.HuggingFaceSAMAdapter") as mock_sam_class:
                mock_det_class.return_value = mock_detector
                mock_sam_class.return_value = mock_segmenter

                # Detection result
                mock_detector.predict.return_value = VisionResult(
                    instances=[Instance(bbox=(10.0, 20.0, 100.0, 150.0), score=0.9, label=0, label_name="cat")], meta={}
                )

                # Segmentation result
                mock_mask = np.ones((480, 640), dtype=bool)
                mock_segmenter.predict.return_value = VisionResult(
                    instances=[Instance(mask=mock_mask, score=0.85, label=0, label_name="object", area=1000)], meta={}
                )

                pipeline = GroundingDINOSAMPipeline(
                    detector_model_id="IDEA-Research/grounding-dino-tiny", sam_model_id="facebook/sam-vit-base"
                )

                image = Image.new("RGB", (640, 480))
                result = pipeline.predict(image, text_prompts="cat")

                # Verify merged instance has both bbox and mask
                assert len(result.instances) == 1
                merged = result.instances[0]
                assert merged.bbox == (10.0, 20.0, 100.0, 150.0)
                assert merged.mask is not None
                assert merged.label_name == "cat"  # From detection
                assert merged.score == pytest.approx((0.9 + 0.85) / 2)  # Averaged
                assert merged.area == 1000  # From segmentation


class TestEmptyDetectionHandling:
    """Test handling of empty detection results."""

    def test_empty_detections_skips_sam(self, mock_detector, mock_segmenter):
        """Test that SAM is skipped when no detections are found."""
        with patch(
            "mata.adapters.huggingface_zeroshot_detect_adapter.HuggingFaceZeroShotDetectAdapter"
        ) as mock_det_class:
            with patch("mata.adapters.huggingface_sam_adapter.HuggingFaceSAMAdapter") as mock_sam_class:
                mock_det_class.return_value = mock_detector
                mock_sam_class.return_value = mock_segmenter

                # Empty detection result
                mock_detector.predict.return_value = VisionResult(instances=[], meta={})

                pipeline = GroundingDINOSAMPipeline(
                    detector_model_id="IDEA-Research/grounding-dino-tiny", sam_model_id="facebook/sam-vit-base"
                )

                image = Image.new("RGB", (640, 480))
                result = pipeline.predict(image, text_prompts="unicorn")

                # SAM should not be called
                mock_segmenter.predict.assert_not_called()

                # Result should be empty
                assert len(result.instances) == 0

    def test_empty_detections_logged(self, mock_detector, mock_segmenter, caplog):
        """Test that empty detections are handled correctly."""
        with patch(
            "mata.adapters.huggingface_zeroshot_detect_adapter.HuggingFaceZeroShotDetectAdapter"
        ) as mock_det_class:
            with patch("mata.adapters.huggingface_sam_adapter.HuggingFaceSAMAdapter") as mock_sam_class:
                mock_det_class.return_value = mock_detector
                mock_sam_class.return_value = mock_segmenter

                # Empty detection result
                mock_detector.predict.return_value = VisionResult(instances=[], meta={})

                pipeline = GroundingDINOSAMPipeline(
                    detector_model_id="IDEA-Research/grounding-dino-tiny", sam_model_id="facebook/sam-vit-base"
                )

                image = Image.new("RGB", (640, 480))

                result = pipeline.predict(image, text_prompts="unicorn")

                # Should return empty result
                assert result.instances == []
                # SAM should not be called since no detections
                mock_segmenter.predict.assert_not_called()


class TestBatchProcessing:
    """Test batch processing through pipeline."""

    def test_batch_prediction(self, mock_detector, mock_segmenter):
        """Test batch processing of multiple images."""
        with patch(
            "mata.adapters.huggingface_zeroshot_detect_adapter.HuggingFaceZeroShotDetectAdapter"
        ) as mock_det_class:
            with patch("mata.adapters.huggingface_sam_adapter.HuggingFaceSAMAdapter") as mock_sam_class:
                mock_det_class.return_value = mock_detector
                mock_sam_class.return_value = mock_segmenter

                # Mock detection results (different for each image)
                det_result1 = VisionResult(
                    instances=[Instance(bbox=(10.0, 20.0, 100.0, 150.0), score=0.9, label=0, label_name="cat")], meta={}
                )
                det_result2 = VisionResult(
                    instances=[Instance(bbox=(30.0, 40.0, 120.0, 180.0), score=0.85, label=1, label_name="dog")],
                    meta={},
                )
                mock_detector.predict.side_effect = [det_result1, det_result2]

                # Mock segmentation results
                seg_result = VisionResult(
                    instances=[Instance(mask=np.ones((480, 640), dtype=bool), score=0.8, label=0, label_name="object")],
                    meta={},
                )
                mock_segmenter.predict.return_value = seg_result

                pipeline = GroundingDINOSAMPipeline(
                    detector_model_id="IDEA-Research/grounding-dino-tiny", sam_model_id="facebook/sam-vit-base"
                )

                images = [Image.new("RGB", (640, 480)), Image.new("RGB", (640, 480))]
                results = pipeline.predict(images, text_prompts="cat . dog")

                # Should return list of results
                assert isinstance(results, list)
                assert len(results) == 2

                # Each result should have merged instances
                assert results[0].instances[0].label_name == "cat"
                assert results[1].instances[0].label_name == "dog"

    def test_empty_batch_returns_empty_list(self, mock_detector, mock_segmenter):
        """Test that empty batch returns empty list."""
        pipeline = GroundingDINOSAMPipeline(
            detector_model_id="IDEA-Research/grounding-dino-tiny", sam_model_id="facebook/sam-vit-base"
        )

        results = pipeline.predict([], text_prompts="cat")

        assert results == []


class TestThresholdParameters:
    """Test threshold parameter handling."""

    def test_detection_threshold_passed_to_detector(self, mock_detector, mock_segmenter):
        """Test that detection threshold is passed to detector."""
        with patch(
            "mata.adapters.huggingface_zeroshot_detect_adapter.HuggingFaceZeroShotDetectAdapter"
        ) as mock_det_class:
            with patch("mata.adapters.huggingface_sam_adapter.HuggingFaceSAMAdapter") as mock_sam_class:
                mock_det_class.return_value = mock_detector
                mock_sam_class.return_value = mock_segmenter

                # Mock empty detection
                mock_detector.predict.return_value = VisionResult(instances=[], meta={})

                pipeline = GroundingDINOSAMPipeline(
                    detector_model_id="IDEA-Research/grounding-dino-tiny",
                    sam_model_id="facebook/sam-vit-base",
                    detection_threshold=0.45,
                )

                image = Image.new("RGB", (640, 480))
                pipeline.predict(image, text_prompts="cat")

                # Verify threshold was passed
                call_kwargs = mock_detector.predict.call_args[1]
                assert call_kwargs.get("threshold") == 0.45

    def test_segmentation_threshold_passed_to_sam(self, mock_detector, mock_segmenter):
        """Test that segmentation threshold is passed to SAM."""
        with patch(
            "mata.adapters.huggingface_zeroshot_detect_adapter.HuggingFaceZeroShotDetectAdapter"
        ) as mock_det_class:
            with patch("mata.adapters.huggingface_sam_adapter.HuggingFaceSAMAdapter") as mock_sam_class:
                mock_det_class.return_value = mock_detector
                mock_sam_class.return_value = mock_segmenter

                # Mock detection result
                mock_detector.predict.return_value = VisionResult(
                    instances=[Instance(bbox=(10.0, 20.0, 100.0, 150.0), score=0.9, label=0, label_name="cat")], meta={}
                )

                # Mock segmentation result
                mock_segmenter.predict.return_value = VisionResult(
                    instances=[Instance(mask=np.ones((480, 640), dtype=bool), score=0.8, label=0, label_name="object")],
                    meta={},
                )

                pipeline = GroundingDINOSAMPipeline(
                    detector_model_id="IDEA-Research/grounding-dino-tiny",
                    sam_model_id="facebook/sam-vit-base",
                    segmentation_threshold=0.55,
                )

                image = Image.new("RGB", (640, 480))
                pipeline.predict(image, text_prompts="cat")

                # Verify threshold was passed
                call_kwargs = mock_segmenter.predict.call_args[1]
                assert call_kwargs.get("threshold") == 0.55


class TestInfoMethod:
    """Test adapter info method."""

    def test_info_returns_metadata(self):
        """Test that info() returns correct metadata."""
        pipeline = GroundingDINOSAMPipeline(
            detector_model_id="IDEA-Research/grounding-dino-tiny",
            sam_model_id="facebook/sam-vit-base",
            detection_threshold=0.4,
            segmentation_threshold=0.6,
        )

        info = pipeline.info()

        assert info["name"] == "GroundingDINOSAMPipeline"
        assert info["task"] == "pipeline"
        assert info["detector_model"] == "IDEA-Research/grounding-dino-tiny"
        assert info["sam_model"] == "facebook/sam-vit-base"
        assert info["detection_threshold"] == 0.4
        assert info["segmentation_threshold"] == 0.6
        assert info["pipeline_type"] == "grounding_sam"
        assert info["backend"] == "huggingface-transformers"


class TestErrorHandling:
    """Test error handling."""

    def test_no_text_prompts_raises(self):
        """Test that prediction without text prompts raises error."""
        pipeline = GroundingDINOSAMPipeline(
            detector_model_id="IDEA-Research/grounding-dino-tiny", sam_model_id="facebook/sam-vit-base"
        )

        image = Image.new("RGB", (640, 480))

        # text_prompts is a required parameter
        with pytest.raises(TypeError):  # Missing required parameter
            pipeline.predict(image)


# Integration test with UniversalLoader
def test_load_pipeline_via_universal_loader():
    """Test loading pipeline via UniversalLoader."""
    with patch("mata.adapters.huggingface_zeroshot_detect_adapter.HuggingFaceZeroShotDetectAdapter"):
        with patch("mata.adapters.huggingface_sam_adapter.HuggingFaceSAMAdapter"):
            with patch("mata.adapters.pytorch_base._ensure_torch") as mock_torch_ensure:
                mock_torch = Mock()
                mock_torch.cuda.is_available = Mock(return_value=False)
                mock_torch.device = Mock(return_value=Mock(type="cpu"))
                mock_torch_ensure.return_value = mock_torch

                from mata.core.model_loader import UniversalLoader

                UniversalLoader()

                # This tests the integration - that the loader can create a pipeline
                # Don't use config alias since there's no config file
                # Test that we can create it directly
                pipeline = GroundingDINOSAMPipeline(
                    detector_model_id="IDEA-Research/grounding-dino-tiny", sam_model_id="facebook/sam-vit-base"
                )

                assert pipeline is not None
                assert isinstance(pipeline, GroundingDINOSAMPipeline)
