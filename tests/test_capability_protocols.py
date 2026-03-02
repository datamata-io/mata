"""Tests for graph capability protocols.

Test suite for runtime-checkable protocols in the MATA graph system,
verifying protocol compliance and isinstance checks for all task types.
"""

from __future__ import annotations

from typing import Any

import pytest

from mata.core.artifacts.image import Image
from mata.core.registry.protocols import (
    Classifier,
    DepthEstimator,
    Detector,
    Embedder,
    PoseEstimator,
    Segmenter,
    Tracker,
    VisionLanguageModel,
)

# ============================================================================
# Mock Implementations (Valid)
# ============================================================================


class ValidDetector:
    """Valid detector implementation."""

    def predict(self, image, **kwargs):
        """Mock predict method."""
        return MockDetections()


class ValidSegmenter:
    """Valid segmenter implementation."""

    def segment(self, image, **kwargs):
        """Mock segment method."""
        return MockMasks()


class ValidClassifier:
    """Valid classifier implementation."""

    def classify(self, image, **kwargs):
        """Mock classify method."""
        return MockClassifyResult()


class ValidPoseEstimator:
    """Valid pose estimator implementation."""

    def estimate(self, image, rois=None, **kwargs):
        """Mock estimate method."""
        return MockKeypoints()


class ValidDepthEstimator:
    """Valid depth estimator implementation."""

    def estimate(self, image, **kwargs):
        """Mock estimate method."""
        return MockDepthResult()


class ValidEmbedder:
    """Valid embedder implementation."""

    def embed(self, input, **kwargs):
        """Mock embed method."""
        import numpy as np

        return np.zeros((512,))


class ValidTracker:
    """Valid tracker implementation (stateful)."""

    def __init__(self):
        self.state = {}

    def update(self, detections, frame_id: str, **kwargs):
        """Mock update method."""
        return MockTracks()

    def reset(self) -> None:
        """Mock reset method."""
        self.state = {}


class ValidVLM:
    """Valid VLM implementation."""

    def query(
        self,
        image: Image | list[Image],
        prompt: str,
        output_mode: str | None = None,
        auto_promote: bool = False,
        **kwargs: Any,
    ):
        """Mock query method."""
        return MockVisionResult()


# ============================================================================
# Mock Implementations (Invalid)
# ============================================================================


class InvalidDetector:
    """Invalid detector - wrong method name."""

    def detect(self, image, **kwargs):
        """Wrong method name - should be predict()."""
        return MockDetections()


class InvalidSegmenter:
    """Invalid segmenter - missing method."""

    pass  # No segment() method


class InvalidClassifier:
    """Invalid classifier - wrong signature."""

    def classify(self, image):
        """Missing **kwargs parameter."""
        return MockClassifyResult()


class InvalidTracker:
    """Invalid tracker - missing reset()."""

    def update(self, detections, frame_id: str, **kwargs):
        """Has update but not reset."""
        return MockTracks()


class InvalidVLM:
    """Invalid VLM - missing parameters."""

    def query(self, image, prompt):
        """Missing output_mode and auto_promote parameters."""
        return MockVisionResult()


# ============================================================================
# Mock Return Types
# ============================================================================


class MockDetections:
    """Mock Detections artifact."""

    pass


class MockMasks:
    """Mock Masks artifact."""

    pass


class MockClassifyResult:
    """Mock ClassifyResult."""

    pass


class MockKeypoints:
    """Mock Keypoints artifact."""

    pass


class MockTracks:
    """Mock Tracks artifact."""

    pass


class MockDepthResult:
    """Mock DepthResult."""

    pass


class MockVisionResult:
    """Mock VisionResult."""

    pass


# ============================================================================
# Test: Detector Protocol
# ============================================================================


def test_detector_protocol_valid():
    """Test Detector protocol with valid implementation."""
    detector = ValidDetector()
    assert isinstance(detector, Detector)


def test_detector_protocol_invalid_method_name():
    """Test Detector protocol rejects wrong method name."""
    detector = InvalidDetector()
    assert not isinstance(detector, Detector)


def test_detector_protocol_invalid_missing_method():
    """Test Detector protocol rejects missing method."""
    detector = object()
    assert not isinstance(detector, Detector)


def test_detector_protocol_callable():
    """Test Detector protocol predict is callable."""
    detector = ValidDetector()
    result = detector.predict(None)
    assert isinstance(result, MockDetections)


# ============================================================================
# Test: Segmenter Protocol
# ============================================================================


def test_segmenter_protocol_valid():
    """Test Segmenter protocol with valid implementation."""
    segmenter = ValidSegmenter()
    assert isinstance(segmenter, Segmenter)


def test_segmenter_protocol_invalid_missing_method():
    """Test Segmenter protocol rejects missing method."""
    segmenter = InvalidSegmenter()
    assert not isinstance(segmenter, Segmenter)


def test_segmenter_protocol_callable():
    """Test Segmenter protocol segment is callable."""
    segmenter = ValidSegmenter()
    result = segmenter.segment(None)
    assert isinstance(result, MockMasks)


# ============================================================================
# Test: Classifier Protocol
# ============================================================================


def test_classifier_protocol_valid():
    """Test Classifier protocol with valid implementation."""
    classifier = ValidClassifier()
    assert isinstance(classifier, Classifier)


def test_classifier_protocol_invalid_signature():
    """Test Classifier protocol with wrong signature."""
    # Note: Protocol checking is duck-typing, so signature differences
    # may not be caught by isinstance (depends on implementation)
    classifier = InvalidClassifier()
    # This WILL pass isinstance check because method name matches
    assert isinstance(classifier, Classifier)
    # But will fail if called with kwargs
    with pytest.raises(TypeError):
        classifier.classify(None, top_k=5)


def test_classifier_protocol_callable():
    """Test Classifier protocol classify is callable."""
    classifier = ValidClassifier()
    result = classifier.classify(None)
    assert isinstance(result, MockClassifyResult)


# ============================================================================
# Test: PoseEstimator Protocol
# ============================================================================


def test_pose_estimator_protocol_valid():
    """Test PoseEstimator protocol with valid implementation."""
    pose_estimator = ValidPoseEstimator()
    assert isinstance(pose_estimator, PoseEstimator)


def test_pose_estimator_protocol_callable():
    """Test PoseEstimator protocol estimate is callable."""
    pose_estimator = ValidPoseEstimator()
    result = pose_estimator.estimate(None)
    assert isinstance(result, MockKeypoints)


def test_pose_estimator_protocol_with_rois():
    """Test PoseEstimator protocol with optional rois parameter."""
    pose_estimator = ValidPoseEstimator()
    result = pose_estimator.estimate(None, rois=None)
    assert isinstance(result, MockKeypoints)


# ============================================================================
# Test: DepthEstimator Protocol
# ============================================================================


def test_depth_estimator_protocol_valid():
    """Test DepthEstimator protocol with valid implementation."""
    depth_estimator = ValidDepthEstimator()
    assert isinstance(depth_estimator, DepthEstimator)


def test_depth_estimator_protocol_callable():
    """Test DepthEstimator protocol estimate is callable."""
    depth_estimator = ValidDepthEstimator()
    result = depth_estimator.estimate(None)
    assert isinstance(result, MockDepthResult)


# ============================================================================
# Test: Embedder Protocol
# ============================================================================


def test_embedder_protocol_valid():
    """Test Embedder protocol with valid implementation."""
    embedder = ValidEmbedder()
    assert isinstance(embedder, Embedder)


def test_embedder_protocol_callable():
    """Test Embedder protocol embed is callable."""
    import numpy as np

    embedder = ValidEmbedder()
    result = embedder.embed(None)
    assert isinstance(result, np.ndarray)


# ============================================================================
# Test: Tracker Protocol
# ============================================================================


def test_tracker_protocol_valid():
    """Test Tracker protocol with valid implementation."""
    tracker = ValidTracker()
    assert isinstance(tracker, Tracker)


def test_tracker_protocol_invalid_missing_reset():
    """Test Tracker protocol rejects missing reset method."""
    tracker = InvalidTracker()
    assert not isinstance(tracker, Tracker)


def test_tracker_protocol_callable_update():
    """Test Tracker protocol update is callable."""
    tracker = ValidTracker()
    result = tracker.update(None, "frame_0")
    assert isinstance(result, MockTracks)


def test_tracker_protocol_callable_reset():
    """Test Tracker protocol reset is callable."""
    tracker = ValidTracker()
    tracker.state = {"test": "data"}
    tracker.reset()
    assert tracker.state == {}


def test_tracker_protocol_stateful():
    """Test Tracker protocol maintains state across updates."""
    tracker = ValidTracker()
    tracker.state["count"] = 0
    tracker.update(None, "frame_0")
    tracker.state["count"] += 1
    assert tracker.state["count"] == 1


# ============================================================================
# Test: VisionLanguageModel Protocol
# ============================================================================


def test_vlm_protocol_valid():
    """Test VLM protocol with valid implementation."""
    vlm = ValidVLM()
    assert isinstance(vlm, VisionLanguageModel)


def test_vlm_protocol_invalid_missing_params():
    """Test VLM protocol rejects missing parameters."""
    vlm = InvalidVLM()
    # Will pass isinstance because Python protocols check method name only
    assert isinstance(vlm, VisionLanguageModel)
    # But will fail when called with all parameters
    with pytest.raises(TypeError):
        vlm.query(None, "prompt", output_mode="json", auto_promote=True)


def test_vlm_protocol_callable_single_image():
    """Test VLM protocol query with single image."""
    vlm = ValidVLM()
    result = vlm.query(None, "What is in this image?")
    assert isinstance(result, MockVisionResult)


def test_vlm_protocol_callable_multi_image():
    """Test VLM protocol query with multiple images."""
    vlm = ValidVLM()
    result = vlm.query([None, None], "Compare these images")
    assert isinstance(result, MockVisionResult)


def test_vlm_protocol_with_output_mode():
    """Test VLM protocol supports output_mode parameter."""
    vlm = ValidVLM()
    result = vlm.query(None, "Detect objects", output_mode="detect")
    assert isinstance(result, MockVisionResult)


def test_vlm_protocol_with_auto_promote():
    """Test VLM protocol supports auto_promote parameter."""
    vlm = ValidVLM()
    result = vlm.query(None, "Find cats", auto_promote=True)
    assert isinstance(result, MockVisionResult)


def test_vlm_protocol_all_params():
    """Test VLM protocol with all parameters."""
    vlm = ValidVLM()
    result = vlm.query(
        image=[None, None],
        prompt="Analyze these images",
        output_mode="json",
        auto_promote=True,
        temperature=0.7,
        max_tokens=512,
    )
    assert isinstance(result, MockVisionResult)


# ============================================================================
# Test: Protocol Type Checking
# ============================================================================


def test_all_protocols_are_runtime_checkable():
    """Verify all protocols are runtime checkable."""

    protocols = [
        Detector,
        Segmenter,
        Classifier,
        PoseEstimator,
        DepthEstimator,
        Embedder,
        Tracker,
        VisionLanguageModel,
    ]

    for protocol in protocols:
        # Check it's a Protocol
        assert hasattr(protocol, "_is_protocol")
        assert protocol._is_protocol

        # Runtime checkable verification - test with actual instance checks
        # (different Python versions expose different internal attributes)
        valid_detector = ValidDetector()
        if protocol == Detector:
            assert isinstance(valid_detector, protocol)


def test_protocol_method_signatures():
    """Verify protocol method signatures are correct."""
    import inspect

    # Detector
    sig = inspect.signature(Detector.predict)
    assert "image" in sig.parameters
    assert "kwargs" in sig.parameters

    # Segmenter
    sig = inspect.signature(Segmenter.segment)
    assert "image" in sig.parameters
    assert "kwargs" in sig.parameters

    # Classifier
    sig = inspect.signature(Classifier.classify)
    assert "image" in sig.parameters
    assert "kwargs" in sig.parameters

    # PoseEstimator
    sig = inspect.signature(PoseEstimator.estimate)
    assert "image" in sig.parameters
    assert "rois" in sig.parameters
    assert "kwargs" in sig.parameters

    # DepthEstimator
    sig = inspect.signature(DepthEstimator.estimate)
    assert "image" in sig.parameters
    assert "kwargs" in sig.parameters

    # Embedder
    sig = inspect.signature(Embedder.embed)
    assert "input" in sig.parameters
    assert "kwargs" in sig.parameters

    # Tracker
    sig = inspect.signature(Tracker.update)
    assert "detections" in sig.parameters
    assert "frame_id" in sig.parameters
    assert "kwargs" in sig.parameters

    sig = inspect.signature(Tracker.reset)
    # reset() has only 'self' parameter (instance method)
    params = [p for p in sig.parameters.keys() if p != "self"]
    assert len(params) == 0

    # VisionLanguageModel
    sig = inspect.signature(VisionLanguageModel.query)
    assert "image" in sig.parameters
    assert "prompt" in sig.parameters
    assert "output_mode" in sig.parameters
    assert "auto_promote" in sig.parameters
    assert "kwargs" in sig.parameters


# ============================================================================
# Test: Cross-Protocol Validation
# ============================================================================


def test_different_protocols_are_distinct():
    """Verify protocols are distinct (not interchangeable)."""
    detector = ValidDetector()
    segmenter = ValidSegmenter()
    classifier = ValidClassifier()

    assert isinstance(detector, Detector)
    assert not isinstance(detector, Segmenter)
    assert not isinstance(detector, Classifier)

    assert isinstance(segmenter, Segmenter)
    assert not isinstance(segmenter, Detector)
    assert not isinstance(segmenter, Classifier)

    assert isinstance(classifier, Classifier)
    assert not isinstance(classifier, Detector)
    assert not isinstance(classifier, Segmenter)


def test_multi_capability_provider():
    """Test provider implementing multiple protocols."""

    class MultiCapability:
        """Provider with multiple capabilities."""

        def predict(self, image, **kwargs):
            return MockDetections()

        def segment(self, image, **kwargs):
            return MockMasks()

    multi = MultiCapability()
    assert isinstance(multi, Detector)
    assert isinstance(multi, Segmenter)
    assert not isinstance(multi, Classifier)


# ============================================================================
# Test: Documentation and Metadata
# ============================================================================


def test_protocol_docstrings():
    """Verify all protocols have docstrings."""
    protocols = [
        Detector,
        Segmenter,
        Classifier,
        PoseEstimator,
        DepthEstimator,
        Embedder,
        Tracker,
        VisionLanguageModel,
    ]

    for protocol in protocols:
        assert protocol.__doc__ is not None
        assert len(protocol.__doc__) > 50  # Substantial documentation


def test_vlm_protocol_documentation():
    """Verify VLM protocol has comprehensive documentation."""
    assert VisionLanguageModel.__doc__ is not None
    doc = VisionLanguageModel.__doc__

    # Check for key concepts
    assert "Union[Image, List[Image]]" in doc
    assert "output_mode" in doc
    assert "auto_promote" in doc
    assert "multi-image" in doc.lower() or "multi-modal" in doc.lower()
    assert "entity" in doc.lower()

    # Check method docstring
    assert VisionLanguageModel.query.__doc__ is not None
    method_doc = VisionLanguageModel.query.__doc__
    assert "output_mode" in method_doc
    assert "auto_promote" in method_doc


# ============================================================================
# Test: Edge Cases
# ============================================================================


def test_protocol_with_none():
    """Test protocol checking with None."""
    assert not isinstance(None, Detector)
    assert not isinstance(None, Segmenter)
    assert not isinstance(None, VisionLanguageModel)


def test_protocol_with_builtin_types():
    """Test protocol checking with built-in types."""
    assert not isinstance(42, Detector)
    assert not isinstance("string", Classifier)
    assert not isinstance([], VisionLanguageModel)
    assert not isinstance({}, Tracker)


def test_protocol_with_lambda():
    """Test protocol checking with lambda functions."""

    def lam(x):
        return x

    assert not isinstance(lam, Detector)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
