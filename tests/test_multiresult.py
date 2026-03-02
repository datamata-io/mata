"""Tests for MultiResult artifact.

Comprehensive test suite covering:
- Channel storage and retrieval
- Attribute access patterns
- Instance cross-referencing
- Provenance data structure
- JSON round-trip serialization
- Missing channel handling
- Edge cases
"""

import json

import pytest

from mata.core.artifacts.detections import Detections
from mata.core.artifacts.masks import Masks
from mata.core.artifacts.result import MultiResult
from mata.core.types import Instance


class TestMultiResultBasics:
    """Test basic MultiResult functionality."""

    def test_empty_multiresult(self):
        """Test creating empty MultiResult."""
        result = MultiResult()

        assert len(result.channels) == 0
        assert len(result.provenance) == 0
        assert len(result.metrics) == 0
        assert result.meta == {}

    def test_multiresult_with_channels(self):
        """Test MultiResult with channel artifacts."""
        # Create sample artifacts
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
                Instance(bbox=(50, 60, 150, 250), score=0.85, label=1, label_name="dog"),
            ]
        )

        masks = Masks(
            instances=[
                Instance(
                    bbox=(10, 20, 100, 200),
                    score=0.9,
                    label=0,
                    label_name="cat",
                    mask={"size": [100, 100], "counts": "mock_rle"},
                ),
            ]
        )

        # Create MultiResult
        result = MultiResult(
            channels={
                "detections": dets,
                "masks": masks,
            }
        )

        assert len(result.channels) == 2
        assert "detections" in result.channels
        assert "masks" in result.channels
        assert result.channels["detections"] == dets
        assert result.channels["masks"] == masks

    def test_multiresult_with_provenance(self):
        """Test MultiResult with provenance data."""
        provenance = {
            "models": {
                "detector": "facebook/detr-resnet-50",
                "segmenter": "facebook/sam",
            },
            "graph_hash": "abc123def456",
            "timestamp": "2026-02-12T10:30:00",
            "mata_version": "1.6.0",
        }

        result = MultiResult(provenance=provenance)

        assert result.provenance == provenance
        assert result.provenance["models"]["detector"] == "facebook/detr-resnet-50"
        assert result.provenance["graph_hash"] == "abc123def456"

    def test_multiresult_with_metrics(self):
        """Test MultiResult with execution metrics."""
        metrics = {
            "detect_node": {
                "latency_ms": 45.2,
                "memory_mb": 512,
                "device": "cuda:0",
            },
            "segment_node": {
                "latency_ms": 120.5,
                "memory_mb": 1024,
                "device": "cuda:0",
            },
            "total": {
                "latency_ms": 165.7,
                "memory_peak_mb": 1536,
            },
        }

        result = MultiResult(metrics=metrics)

        assert result.metrics == metrics
        assert result.metrics["detect_node"]["latency_ms"] == 45.2
        assert result.metrics["total"]["latency_ms"] == 165.7


class TestChannelAccess:
    """Test channel access methods."""

    def test_attribute_access(self):
        """Test dynamic attribute access for channels."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ]
        )

        result = MultiResult(channels={"detections": dets})

        # Access via attribute
        accessed_dets = result.detections
        assert accessed_dets == dets
        assert accessed_dets.instances == dets.instances

    def test_attribute_access_missing_channel(self):
        """Test attribute access with missing channel."""
        result = MultiResult()

        with pytest.raises(AttributeError, match="No channel 'nonexistent'"):
            _ = result.nonexistent

    def test_has_channel(self):
        """Test has_channel method."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ]
        )

        result = MultiResult(channels={"detections": dets})

        assert result.has_channel("detections") is True
        assert result.has_channel("masks") is False
        assert result.has_channel("nonexistent") is False

    def test_get_channel_with_default(self):
        """Test get_channel with default value."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ]
        )

        result = MultiResult(channels={"detections": dets})

        # Existing channel
        assert result.get_channel("detections") == dets

        # Missing channel with default
        assert result.get_channel("masks", default=None) is None

        # Missing channel without default
        assert result.get_channel("nonexistent") is None


class TestInstanceCrossReferencing:
    """Test instance cross-referencing across channels."""

    def test_get_instance_artifacts(self):
        """Test getting all artifacts for a specific instance ID."""
        # Create detections and masks with matching instance IDs
        instance_ids = ["inst_0000", "inst_0001"]

        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
                Instance(bbox=(50, 60, 150, 250), score=0.85, label=1, label_name="dog"),
            ],
            instance_ids=instance_ids,
        )

        masks = Masks(
            instances=[
                Instance(
                    bbox=(10, 20, 100, 200),
                    score=0.9,
                    label=0,
                    label_name="cat",
                    mask={"size": [100, 100], "counts": "mock_rle_cat"},
                ),
                Instance(
                    bbox=(50, 60, 150, 250),
                    score=0.85,
                    label=1,
                    label_name="dog",
                    mask={"size": [100, 100], "counts": "mock_rle_dog"},
                ),
            ],
            instance_ids=instance_ids,
        )

        result = MultiResult(
            channels={
                "detections": dets,
                "masks": masks,
            }
        )

        # Get artifacts for first instance
        inst_data = result.get_instance_artifacts("inst_0000")

        assert "detections" in inst_data
        assert "masks" in inst_data
        assert inst_data["detections"].label_name == "cat"
        assert inst_data["masks"].label_name == "cat"
        assert inst_data["masks"].mask["counts"] == "mock_rle_cat"

    def test_get_instance_artifacts_partial_match(self):
        """Test get_instance_artifacts when instance only exists in some channels."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
                Instance(bbox=(50, 60, 150, 250), score=0.85, label=1, label_name="dog"),
            ],
            instance_ids=["inst_0000", "inst_0001"],
        )

        # Masks only has one instance
        masks = Masks(
            instances=[
                Instance(
                    bbox=(10, 20, 100, 200),
                    score=0.9,
                    label=0,
                    label_name="cat",
                    mask={"size": [100, 100], "counts": "mock_rle"},
                ),
            ],
            instance_ids=["inst_0000"],
        )

        result = MultiResult(
            channels={
                "detections": dets,
                "masks": masks,
            }
        )

        # inst_0000 exists in both channels
        inst_data = result.get_instance_artifacts("inst_0000")
        assert "detections" in inst_data
        assert "masks" in inst_data

        # inst_0001 only exists in detections
        inst_data = result.get_instance_artifacts("inst_0001")
        assert "detections" in inst_data
        assert "masks" not in inst_data

    def test_get_instance_artifacts_not_found(self):
        """Test get_instance_artifacts with non-existent instance ID."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ],
            instance_ids=["inst_0000"],
        )

        result = MultiResult(channels={"detections": dets})

        inst_data = result.get_instance_artifacts("nonexistent")
        assert inst_data == {}

    def test_list_instance_ids(self):
        """Test listing all unique instance IDs across channels."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
                Instance(bbox=(50, 60, 150, 250), score=0.85, label=1, label_name="dog"),
            ],
            instance_ids=["inst_0000", "inst_0001"],
        )

        masks = Masks(
            instances=[
                Instance(
                    bbox=(10, 20, 100, 200),
                    score=0.9,
                    label=0,
                    label_name="cat",
                    mask={"size": [100, 100], "counts": "mock_rle"},
                ),
                Instance(
                    bbox=(30, 40, 130, 240),
                    score=0.8,
                    label=2,
                    label_name="bird",
                    mask={"size": [100, 100], "counts": "mock_rle_bird"},
                ),
            ],
            instance_ids=["inst_0000", "inst_0002"],  # Different second ID
        )

        result = MultiResult(
            channels={
                "detections": dets,
                "masks": masks,
            }
        )

        all_ids = result.list_instance_ids()

        assert len(all_ids) == 3
        assert "inst_0000" in all_ids
        assert "inst_0001" in all_ids
        assert "inst_0002" in all_ids


class TestSerialization:
    """Test JSON serialization and deserialization."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ]
        )

        result = MultiResult(
            channels={"detections": dets},
            provenance={"model": "detr"},
            metrics={"latency_ms": 45.2},
            meta={"note": "test"},
        )

        data = result.to_dict()

        assert "channels" in data
        assert "provenance" in data
        assert "metrics" in data
        assert "meta" in data

        assert "detections" in data["channels"]
        assert data["channels"]["detections"]["type"] == "Detections"
        assert "data" in data["channels"]["detections"]

        assert data["provenance"]["model"] == "detr"
        assert data["metrics"]["latency_ms"] == 45.2
        assert data["meta"]["note"] == "test"

    def test_to_json(self):
        """Test to_json conversion."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ]
        )

        result = MultiResult(
            channels={"detections": dets},
            provenance={"model": "detr"},
        )

        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert "detections" in json_str
        assert "Detections" in json_str
        assert "detr" in json_str

        # Verify it's valid JSON
        data = json.loads(json_str)
        assert "channels" in data
        assert "provenance" in data

    def test_to_json_compact(self):
        """Test compact JSON output."""
        result = MultiResult(provenance={"model": "detr"})

        json_compact = result.to_json(indent=None)
        json_pretty = result.to_json(indent=2)

        # Compact should be shorter
        assert len(json_compact) < len(json_pretty)
        assert "\n" not in json_compact
        assert "\n" in json_pretty

    def test_from_dict(self):
        """Test from_dict reconstruction."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ]
        )

        original = MultiResult(
            channels={"detections": dets},
            provenance={"model": "detr"},
            metrics={"latency_ms": 45.2},
        )

        # Convert to dict and back
        data = original.to_dict()
        reconstructed = MultiResult.from_dict(data)

        assert len(reconstructed.channels) == len(original.channels)
        assert "detections" in reconstructed.channels
        assert reconstructed.provenance == original.provenance
        assert reconstructed.metrics == original.metrics

    def test_from_json(self):
        """Test from_json reconstruction."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ]
        )

        original = MultiResult(
            channels={"detections": dets},
            provenance={"model": "detr"},
        )

        # Convert to JSON and back
        json_str = original.to_json()
        reconstructed = MultiResult.from_json(json_str)

        assert len(reconstructed.channels) == len(original.channels)
        assert "detections" in reconstructed.channels
        assert reconstructed.provenance == original.provenance

    def test_json_roundtrip(self):
        """Test full JSON serialization round-trip."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
                Instance(bbox=(50, 60, 150, 250), score=0.85, label=1, label_name="dog"),
            ]
        )

        original = MultiResult(
            channels={"detections": dets},
            provenance={
                "models": {"detector": "detr"},
                "timestamp": "2026-02-12T10:30:00",
            },
            metrics={
                "detect_node": {"latency_ms": 45.2},
                "total": {"latency_ms": 45.2},
            },
        )

        # Round-trip
        json_str = original.to_json()
        reconstructed = MultiResult.from_json(json_str)

        # Verify channels
        assert len(reconstructed.channels) == 1
        assert "detections" in reconstructed.channels

        # Verify provenance
        assert reconstructed.provenance["models"]["detector"] == "detr"

        # Verify metrics
        assert reconstructed.metrics["detect_node"]["latency_ms"] == 45.2


class TestValidation:
    """Test validation functionality."""

    def test_validate_valid_multiresult(self):
        """Test validation passes for valid MultiResult."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ]
        )

        result = MultiResult(
            channels={"detections": dets},
            provenance={"model": "detr"},
            metrics={"latency_ms": 45.2},
            meta={"note": "test"},
        )

        # Should not raise
        result.validate()

    def test_validate_empty_multiresult(self):
        """Test validation passes for empty MultiResult."""
        result = MultiResult()

        # Should not raise
        result.validate()

    def test_validate_invalid_provenance_type(self):
        """Test validation fails for invalid provenance type."""
        # Use object.__setattr__ to bypass frozen dataclass
        result = MultiResult()
        object.__setattr__(result, "provenance", "invalid")

        with pytest.raises(ValueError, match="Provenance must be dict"):
            result.validate()

    def test_validate_invalid_metrics_type(self):
        """Test validation fails for invalid metrics type."""
        result = MultiResult()
        object.__setattr__(result, "metrics", "invalid")

        with pytest.raises(ValueError, match="Metrics must be dict"):
            result.validate()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_channels(self):
        """Test MultiResult with empty channels dict."""
        result = MultiResult(channels={})

        assert len(result.channels) == 0
        assert result.list_instance_ids() == set()

    def test_multiple_channel_types(self):
        """Test MultiResult with multiple different channel types."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ]
        )

        masks = Masks(
            instances=[
                Instance(
                    bbox=(10, 20, 100, 200),
                    score=0.9,
                    label=0,
                    label_name="cat",
                    mask={"size": [100, 100], "counts": "mock_rle"},
                ),
            ]
        )

        result = MultiResult(
            channels={
                "detections": dets,
                "masks": masks,
            }
        )

        assert len(result.channels) == 2
        assert isinstance(result.detections, Detections)
        assert isinstance(result.masks, Masks)

    def test_repr(self):
        """Test string representation."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ]
        )

        result = MultiResult(
            channels={"detections": dets},
            provenance={"model": "detr", "version": "1.0"},
            metrics={"latency": 45.2, "memory": 512},
        )

        repr_str = repr(result)

        assert "MultiResult" in repr_str
        assert "detections" in repr_str
        assert "2 keys" in repr_str  # provenance has 2 keys

    def test_channel_access_with_builtin_attrs(self):
        """Test that builtin attributes don't conflict with channel access."""
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ]
        )

        result = MultiResult(channels={"detections": dets})

        # Builtin attributes should work
        assert hasattr(result, "channels")
        assert hasattr(result, "provenance")
        assert hasattr(result, "metrics")
        assert hasattr(result, "meta")

        # Channel access should also work
        assert result.detections == dets


class TestIntegration:
    """Integration tests with real workflow scenarios."""

    def test_detection_segmentation_workflow(self):
        """Test typical detect → segment workflow result."""
        # Simulate detection results
        dets = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.95, label=0, label_name="cat"),
                Instance(bbox=(50, 60, 150, 250), score=0.88, label=1, label_name="dog"),
            ],
            instance_ids=["inst_0000", "inst_0001"],
        )

        # Simulate segmentation results (same instances with masks)
        masks = Masks(
            instances=[
                Instance(
                    bbox=(10, 20, 100, 200),
                    score=0.95,
                    label=0,
                    label_name="cat",
                    mask={"size": [100, 100], "counts": "rle_cat"},
                ),
                Instance(
                    bbox=(50, 60, 150, 250),
                    score=0.88,
                    label=1,
                    label_name="dog",
                    mask={"size": [100, 100], "counts": "rle_dog"},
                ),
            ],
            instance_ids=["inst_0000", "inst_0001"],
        )

        # Create unified result
        result = MultiResult(
            channels={
                "detections": dets,
                "masks": masks,
            },
            provenance={
                "models": {
                    "detector": "facebook/detr-resnet-50",
                    "segmenter": "facebook/sam",
                },
                "graph_hash": "detect_segment_v1",
                "timestamp": "2026-02-12T10:30:00",
            },
            metrics={
                "detect_node": {"latency_ms": 45.2, "memory_mb": 512},
                "segment_node": {"latency_ms": 120.5, "memory_mb": 1024},
                "total": {"latency_ms": 165.7, "memory_peak_mb": 1536},
            },
        )

        # Access channels
        assert len(result.detections.instances) == 2
        assert len(result.masks.instances) == 2

        # Cross-reference by instance ID
        cat_data = result.get_instance_artifacts("inst_0000")
        assert cat_data["detections"].label_name == "cat"
        assert cat_data["masks"].label_name == "cat"
        assert cat_data["masks"].mask["counts"] == "rle_cat"

        # Check provenance
        assert result.provenance["models"]["detector"] == "facebook/detr-resnet-50"

        # Check metrics
        assert result.metrics["total"]["latency_ms"] == 165.7

        # Serialize
        json_str = result.to_json()
        assert "detections" in json_str
        assert "masks" in json_str

        # Deserialize
        result_copy = MultiResult.from_json(json_str)
        assert len(result_copy.channels) == 2
        assert result_copy.provenance == result.provenance
