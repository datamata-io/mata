"""Tests for Detections artifact.

Comprehensive test suite covering:
- VisionResult conversion (with entities)
- Instance ID and entity ID generation/preservation
- Filtering operations
- Entity promotion workflows
- Property accessors
- Edge cases (empty detections, mixed entities/instances)
- Backward compatibility
"""

import numpy as np
import pytest

from mata.core.artifacts.detections import Detections, _fuzzy_label_match, _generate_id
from mata.core.types import Entity, Instance, VisionResult


class TestDetectionsBasics:
    """Test basic Detections artifact functionality."""

    def test_empty_detections(self):
        """Test creating empty Detections artifact."""
        dets = Detections()

        assert len(dets.instances) == 0
        assert len(dets.instance_ids) == 0
        assert len(dets.entities) == 0
        assert len(dets.entity_ids) == 0
        assert dets.meta == {}

    def test_detections_with_instances(self):
        """Test Detections with spatial instances."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(50, 60, 150, 250), score=0.85, label=1, label_name="dog"),
        ]

        dets = Detections(instances=instances)

        assert len(dets.instances) == 2
        assert len(dets.instance_ids) == 2
        assert all(id.startswith("inst_") for id in dets.instance_ids)

    def test_detections_with_entities(self):
        """Test Detections with semantic entities."""
        entities = [
            Entity("cat", score=0.9),
            Entity("dog", score=0.85),
        ]

        dets = Detections(entities=entities)

        assert len(dets.entities) == 2
        assert len(dets.entity_ids) == 2
        assert all(id.startswith("ent_") for id in dets.entity_ids)

    def test_detections_mixed(self):
        """Test Detections with both instances and entities."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
        ]
        entities = [
            Entity("bird", score=0.75),
        ]

        dets = Detections(instances=instances, entities=entities)

        assert len(dets.instances) == 1
        assert len(dets.instance_ids) == 1
        assert len(dets.entities) == 1
        assert len(dets.entity_ids) == 1

    def test_auto_id_generation(self):
        """Test automatic ID generation."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(50, 60, 150, 250), score=0.85, label=1, label_name="dog"),
        ]

        dets = Detections(instances=instances)

        # IDs should be auto-generated
        assert len(dets.instance_ids) == 2
        assert dets.instance_ids[0] == "inst_0000"
        assert dets.instance_ids[1] == "inst_0001"

    def test_custom_ids(self):
        """Test providing custom IDs."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
        ]
        custom_ids = ["my_cat_001"]

        dets = Detections(instances=instances, instance_ids=custom_ids)

        assert dets.instance_ids == custom_ids

    def test_id_length_mismatch_error(self):
        """Test error when ID length doesn't match."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(50, 60, 150, 250), score=0.85, label=1, label_name="dog"),
        ]
        wrong_ids = ["id1"]  # Only 1 ID for 2 instances

        with pytest.raises(ValueError, match="length mismatch"):
            Detections(instances=instances, instance_ids=wrong_ids)


class TestVisionResultConversion:
    """Test conversion between Detections and VisionResult."""

    def test_from_vision_result_instances_only(self):
        """Test conversion from VisionResult with instances only."""
        result = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
                Instance(bbox=(50, 60, 150, 250), score=0.85, label=1, label_name="dog"),
            ]
        )

        dets = Detections.from_vision_result(result)

        assert len(dets.instances) == 2
        assert len(dets.instance_ids) == 2
        assert len(dets.entities) == 0
        assert dets.instances[0].label_name == "cat"
        assert dets.instances[1].label_name == "dog"

    def test_from_vision_result_entities_only(self):
        """Test conversion from VisionResult with entities only (VLM output)."""
        result = VisionResult(
            instances=[],
            entities=[
                Entity("cat", score=0.9),
                Entity("dog", score=0.85),
            ],
        )

        dets = Detections.from_vision_result(result)

        assert len(dets.instances) == 0
        assert len(dets.entities) == 2
        assert len(dets.entity_ids) == 2
        assert dets.entities[0].label == "cat"
        assert dets.entities[1].label == "dog"

    def test_from_vision_result_mixed(self):
        """Test conversion from VisionResult with both instances and entities."""
        result = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ],
            entities=[
                Entity("bird", score=0.75),
            ],
        )

        dets = Detections.from_vision_result(result)

        assert len(dets.instances) == 1
        assert len(dets.entities) == 1
        assert dets.instances[0].label_name == "cat"
        assert dets.entities[0].label == "bird"

    def test_to_vision_result(self):
        """Test conversion back to VisionResult."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
        ]
        entities = [
            Entity("bird", score=0.75),
        ]

        dets = Detections(instances=instances, entities=entities)
        result = dets.to_vision_result()

        assert len(result.instances) == 1
        assert len(result.entities) == 1
        assert result.instances[0].label_name == "cat"
        assert result.entities[0].label == "bird"

        # IDs should be preserved in metadata
        assert "instance_ids" in result.meta
        assert "entity_ids" in result.meta

    def test_round_trip_conversion(self):
        """Test round-trip conversion preserves data."""
        original = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ],
            entities=[
                Entity("dog", score=0.85, attributes={"color": "brown"}),
            ],
        )

        # Convert to Detections and back
        dets = Detections.from_vision_result(original)
        reconstructed = dets.to_vision_result()

        assert len(reconstructed.instances) == 1
        assert len(reconstructed.entities) == 1
        assert reconstructed.instances[0].label_name == "cat"
        assert reconstructed.entities[0].label == "dog"
        assert reconstructed.entities[0].attributes == {"color": "brown"}


class TestFiltering:
    """Test filtering operations."""

    def test_filter_by_score(self):
        """Test filtering by confidence score."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(50, 60, 150, 250), score=0.5, label=1, label_name="dog"),
            Instance(bbox=(70, 80, 170, 270), score=0.3, label=2, label_name="bird"),
        ]

        dets = Detections(instances=instances)
        filtered = dets.filter_by_score(0.6)

        assert len(filtered.instances) == 1
        assert filtered.instances[0].label_name == "cat"

    def test_filter_by_score_entities(self):
        """Test filtering entities by score."""
        entities = [
            Entity("cat", score=0.9),
            Entity("dog", score=0.5),
            Entity("bird", score=0.3),
        ]

        dets = Detections(entities=entities)
        filtered = dets.filter_by_score(0.6)

        assert len(filtered.entities) == 1
        assert filtered.entities[0].label == "cat"

    def test_filter_by_label_exact(self):
        """Test exact label filtering."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(50, 60, 150, 250), score=0.85, label=1, label_name="dog"),
            Instance(bbox=(70, 80, 170, 270), score=0.8, label=2, label_name="bird"),
        ]

        dets = Detections(instances=instances)
        filtered = dets.filter_by_label(["cat", "dog"], fuzzy=False)

        assert len(filtered.instances) == 2
        assert filtered.instances[0].label_name == "cat"
        assert filtered.instances[1].label_name == "dog"

    def test_filter_by_label_fuzzy(self):
        """Test fuzzy label filtering."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="Cat"),  # Different case
            Instance(bbox=(50, 60, 150, 250), score=0.85, label=1, label_name="dogs"),  # Plural
            Instance(bbox=(70, 80, 170, 270), score=0.8, label=2, label_name="bird"),
        ]

        dets = Detections(instances=instances)
        filtered = dets.filter_by_label(["cat", "dog"], fuzzy=True)

        assert len(filtered.instances) == 2
        assert filtered.instances[0].label_name == "Cat"
        assert filtered.instances[1].label_name == "dogs"

    def test_top_k(self):
        """Test top-k selection."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(50, 60, 150, 250), score=0.7, label=1, label_name="dog"),
            Instance(bbox=(70, 80, 170, 270), score=0.5, label=2, label_name="bird"),
            Instance(bbox=(90, 100, 190, 300), score=0.3, label=3, label_name="fish"),
        ]

        dets = Detections(instances=instances)
        top2 = dets.top_k(2)

        assert len(top2.instances) == 2
        assert top2.instances[0].score == 0.9  # Highest
        assert top2.instances[1].score == 0.7  # Second highest

    def test_filter_empty_result(self):
        """Test filtering that results in empty detections."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.3, label=0, label_name="cat"),
        ]

        dets = Detections(instances=instances)
        filtered = dets.filter_by_score(0.9)

        assert len(filtered.instances) == 0
        assert len(filtered.instance_ids) == 0


class TestEntityPromotion:
    """Test entity promotion workflows (VLM → spatial fusion)."""

    def test_promote_entities_exact_match(self):
        """Test entity promotion with exact label matching."""
        # VLM detects entities
        vlm_entities = [
            Entity("cat", score=0.9),
            Entity("dog", score=0.85),
        ]
        vlm_dets = Detections(entities=vlm_entities)

        # GroundingDINO provides spatial data
        spatial_instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.95, label=0, label_name="cat"),
            Instance(bbox=(50, 60, 150, 250), score=0.9, label=1, label_name="dog"),
        ]
        spatial_dets = Detections(instances=spatial_instances)

        # Promote entities to instances
        promoted = vlm_dets.promote_entities(spatial_dets, match_strategy="label_exact")

        assert len(promoted.instances) == 2
        assert len(promoted.entities) == 0  # All promoted
        assert promoted.instances[0].label_name == "cat"
        assert promoted.instances[0].bbox == (10, 20, 100, 200)
        assert promoted.instances[1].label_name == "dog"

    def test_promote_entities_fuzzy_match(self):
        """Test entity promotion with fuzzy label matching."""
        # VLM detects entities (different case, plural)
        vlm_entities = [
            Entity("Cat", score=0.9),  # Different case
            Entity("dogs", score=0.85),  # Plural
        ]
        vlm_dets = Detections(entities=vlm_entities)

        # GroundingDINO provides spatial data
        spatial_instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.95, label=0, label_name="cat"),
            Instance(bbox=(50, 60, 150, 250), score=0.9, label=1, label_name="dog"),
        ]
        spatial_dets = Detections(instances=spatial_instances)

        # Promote with fuzzy matching
        promoted = vlm_dets.promote_entities(spatial_dets, match_strategy="label_fuzzy")

        assert len(promoted.instances) == 2
        assert promoted.instances[0].label_name == "Cat"  # Preserves original VLM label
        assert promoted.instances[1].label_name == "dogs"

    def test_promote_entities_partial_match(self):
        """Test entity promotion with partial matches."""
        # VLM detects 3 entities
        vlm_entities = [
            Entity("cat", score=0.9),
            Entity("dog", score=0.85),
            Entity("bird", score=0.8),  # No spatial match
        ]
        vlm_dets = Detections(entities=vlm_entities)

        # GroundingDINO only finds 2
        spatial_instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.95, label=0, label_name="cat"),
            Instance(bbox=(50, 60, 150, 250), score=0.9, label=1, label_name="dog"),
        ]
        spatial_dets = Detections(instances=spatial_instances)

        # Promote
        promoted = vlm_dets.promote_entities(spatial_dets, match_strategy="label_exact")

        assert len(promoted.instances) == 2  # Only matched entities promoted
        assert len(promoted.entities) == 0

    def test_promote_entities_no_match(self):
        """Test entity promotion with no matches."""
        vlm_entities = [
            Entity("cat", score=0.9),
        ]
        vlm_dets = Detections(entities=vlm_entities)

        spatial_instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.95, label=0, label_name="dog"),
        ]
        spatial_dets = Detections(instances=spatial_instances)

        promoted = vlm_dets.promote_entities(spatial_dets, match_strategy="label_exact")

        assert len(promoted.instances) == 0  # No matches
        assert len(promoted.entities) == 0

    def test_promote_entities_invalid_strategy(self):
        """Test error with invalid match strategy."""
        vlm_dets = Detections(entities=[Entity("cat", 0.9)])
        spatial_dets = Detections(instances=[])

        with pytest.raises(ValueError, match="Invalid match_strategy"):
            vlm_dets.promote_entities(spatial_dets, match_strategy="invalid")


class TestPropertyAccessors:
    """Test property accessors (boxes, scores, labels)."""

    def test_boxes_property(self):
        """Test boxes property accessor."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(50, 60, 150, 250), score=0.85, label=1, label_name="dog"),
        ]

        dets = Detections(instances=instances)
        boxes = dets.boxes

        assert boxes.shape == (2, 4)
        assert boxes.dtype == np.float32
        assert np.allclose(boxes[0], [10, 20, 100, 200])
        assert np.allclose(boxes[1], [50, 60, 150, 250])

    def test_boxes_property_empty(self):
        """Test boxes property with no instances."""
        dets = Detections()
        boxes = dets.boxes

        assert boxes.shape == (0, 4)

    def test_scores_property(self):
        """Test scores property accessor."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(50, 60, 150, 250), score=0.85, label=1, label_name="dog"),
        ]
        entities = [
            Entity("bird", score=0.75),
        ]

        dets = Detections(instances=instances, entities=entities)
        scores = dets.scores

        assert scores.shape == (3,)
        assert scores.dtype == np.float32
        assert np.allclose(scores, [0.9, 0.85, 0.75])

    def test_labels_property(self):
        """Test labels property accessor."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(50, 60, 150, 250), score=0.85, label=1, label_name="dog"),
        ]
        entities = [
            Entity("bird", score=0.75),
        ]

        dets = Detections(instances=instances, entities=entities)
        labels = dets.labels

        assert labels == ["cat", "dog", "bird"]

    def test_labels_property_without_names(self):
        """Test labels property with instances without label_name."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0),  # No label_name
        ]

        dets = Detections(instances=instances)
        labels = dets.labels

        assert labels == ["class_0"]


class TestSerialization:
    """Test serialization (to_dict/from_dict)."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
        ]
        entities = [
            Entity("dog", score=0.85),
        ]

        dets = Detections(instances=instances, entities=entities, meta={"source": "test"})
        data = dets.to_dict()

        assert "instances" in data
        assert "instance_ids" in data
        assert "entities" in data
        assert "entity_ids" in data
        assert "meta" in data
        assert len(data["instances"]) == 1
        assert len(data["entities"]) == 1
        assert data["meta"]["source"] == "test"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "instances": [
                {
                    "bbox": [10, 20, 100, 200],
                    "score": 0.9,
                    "label": 0,
                    "label_name": "cat",
                    "mask": None,
                    "area": None,
                    "is_stuff": None,
                    "embedding": None,
                    "track_id": None,
                }
            ],
            "instance_ids": ["inst_0000"],
            "entities": [{"label": "dog", "score": 0.85, "attributes": {}}],
            "entity_ids": ["ent_0000"],
            "meta": {"source": "test"},
        }

        dets = Detections.from_dict(data)

        assert len(dets.instances) == 1
        assert len(dets.entities) == 1
        assert dets.instances[0].label_name == "cat"
        assert dets.entities[0].label == "dog"
        assert dets.meta["source"] == "test"

    def test_round_trip_serialization(self):
        """Test round-trip serialization."""
        original = Detections(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ],
            entities=[
                Entity("dog", score=0.85, attributes={"color": "brown"}),
            ],
            meta={"source": "test"},
        )

        # Serialize and deserialize
        data = original.to_dict()
        reconstructed = Detections.from_dict(data)

        assert len(reconstructed.instances) == 1
        assert len(reconstructed.entities) == 1
        assert reconstructed.instances[0].label_name == "cat"
        assert reconstructed.entities[0].label == "dog"
        assert reconstructed.meta["source"] == "test"


class TestValidation:
    """Test validation logic."""

    def test_validate_success(self):
        """Test validation passes for valid data."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
        ]

        dets = Detections(instances=instances)
        dets.validate()  # Should not raise

    def test_validate_invalid_score(self):
        """Test validation fails for invalid scores."""
        # Can't create instance with invalid score due to Instance validation
        # but we can test Detections.validate() catches it
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
        ]
        dets = Detections(instances=instances)

        # Manually modify to invalid score (bypass frozen)
        object.__setattr__(dets.instances[0], "score", 1.5)

        with pytest.raises(ValueError, match="not in"):
            dets.validate()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_fuzzy_label_match_exact(self):
        """Test fuzzy matching with exact labels."""
        assert _fuzzy_label_match("cat", "cat")
        assert not _fuzzy_label_match("cat", "dog")

    def test_fuzzy_label_match_case(self):
        """Test fuzzy matching with different cases."""
        assert _fuzzy_label_match("Cat", "cat")
        assert _fuzzy_label_match("CAT", "cat")

    def test_fuzzy_label_match_plural(self):
        """Test fuzzy matching with plurals."""
        assert _fuzzy_label_match("cats", "cat")
        assert _fuzzy_label_match("cat", "cats")
        assert _fuzzy_label_match("dogs", "dog")

    def test_fuzzy_label_match_articles(self):
        """Test fuzzy matching with articles."""
        assert _fuzzy_label_match("a cat", "cat")
        assert _fuzzy_label_match("the dog", "dog")
        assert _fuzzy_label_match("an apple", "apple")

    def test_generate_id_with_index(self):
        """Test ID generation with index."""
        id1 = _generate_id("test", 0)
        id2 = _generate_id("test", 1)

        assert id1 == "test_0000"
        assert id2 == "test_0001"

    def test_generate_id_random(self):
        """Test random ID generation."""
        id1 = _generate_id("test")
        id2 = _generate_id("test")

        assert id1.startswith("test_")
        assert id2.startswith("test_")
        assert id1 != id2  # Should be different


class TestImmutability:
    """Test immutability of Detections artifact."""

    def test_frozen_dataclass(self):
        """Test that Detections is frozen (immutable)."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
        ]

        dets = Detections(instances=instances)

        # Should not be able to modify
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            dets.instances = []

    def test_filter_returns_new_instance(self):
        """Test that filtering returns new instance, not modifying original."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(50, 60, 150, 250), score=0.5, label=1, label_name="dog"),
        ]

        original = Detections(instances=instances)
        filtered = original.filter_by_score(0.8)

        # Original should be unchanged
        assert len(original.instances) == 2
        assert len(filtered.instances) == 1
        assert original is not filtered


class TestBackwardCompatibility:
    """Test backward compatibility (entities field optional)."""

    def test_detections_without_entities(self):
        """Test creating Detections without entities (old usage)."""
        instances = [
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
        ]

        # Should work without specifying entities
        dets = Detections(instances=instances)

        assert len(dets.instances) == 1
        assert len(dets.entities) == 0  # Default empty list

    def test_vision_result_without_entities(self):
        """Test VisionResult conversion without entities field."""
        # Old VisionResult might not have entities
        result = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            ]
        )

        dets = Detections.from_vision_result(result)

        assert len(dets.instances) == 1
        assert len(dets.entities) == 0
