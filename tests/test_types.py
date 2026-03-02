"""Tests for core data types."""

import json

from mata.core.types import (
    Detection,
    DetectResult,
    Track,
    TrackResult,
)


class TestDetection:
    """Test Detection dataclass."""

    def test_creation(self):
        """Test Detection creation."""
        det = Detection(bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=1, label_name="person")
        assert det.bbox == (10.0, 20.0, 100.0, 200.0)
        assert det.score == 0.95
        assert det.label == 1
        assert det.label_name == "person"

    def test_to_dict(self):
        """Test Detection to_dict."""
        det = Detection(
            bbox=(10.0, 20.0, 100.0, 200.0),
            score=0.95,
            label=1,
        )
        d = det.to_dict()
        assert d["bbox"] == [10.0, 20.0, 100.0, 200.0]
        assert d["score"] == 0.95
        assert d["label"] == 1
        assert d["label_name"] is None


class TestDetectResult:
    """Test DetectResult dataclass."""

    def test_creation(self):
        """Test DetectResult creation."""
        det = Detection(bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=1)
        result = DetectResult(detections=[det], meta={"model": "test"})
        assert len(result.detections) == 1
        assert result.meta["model"] == "test"

    def test_to_json(self):
        """Test JSON serialization."""
        det = Detection(bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=1)
        result = DetectResult(detections=[det])
        json_str = result.to_json()
        assert isinstance(json_str, str)

        # Parse and verify
        data = json.loads(json_str)
        assert len(data["detections"]) == 1
        assert data["detections"][0]["score"] == 0.95

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = """
        {
            "detections": [
                {
                    "bbox": [10.0, 20.0, 100.0, 200.0],
                    "score": 0.95,
                    "label": 1,
                    "label_name": "person"
                }
            ],
            "meta": {"model": "test"}
        }
        """
        result = DetectResult.from_json(json_str)
        assert len(result.detections) == 1
        assert result.detections[0].score == 0.95
        assert result.meta["model"] == "test"

    def test_roundtrip(self):
        """Test JSON serialization roundtrip."""
        det = Detection(bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=1, label_name="person")
        result1 = DetectResult(detections=[det], meta={"model": "test"})

        # Serialize and deserialize
        json_str = result1.to_json()
        result2 = DetectResult.from_json(json_str)

        # Verify equality
        assert len(result2.detections) == 1
        assert result2.detections[0].bbox == det.bbox
        assert result2.detections[0].score == det.score
        assert result2.meta["model"] == "test"


class TestTrack:
    """Test Track dataclass."""

    def test_creation(self):
        """Test Track creation."""
        track = Track(track_id=42, bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=1, age=5)
        assert track.track_id == 42
        assert track.age == 5


class TestTrackResult:
    """Test TrackResult dataclass."""

    def test_to_json(self):
        """Test TrackResult JSON serialization."""
        track = Track(track_id=1, bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=1)
        result = TrackResult(tracks=[track])
        json_str = result.to_json()

        data = json.loads(json_str)
        assert len(data["tracks"]) == 1
        assert data["tracks"][0]["track_id"] == 1
