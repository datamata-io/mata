"""Tests for entity promotion and artifact conversion utilities."""

import pytest

from mata.core.artifacts.converters import (
    _fuzzy_label_match,
    auto_promote_vision_result,
    match_entity_to_instance,
    merge_entity_attributes,
    promote_entities_to_instances,
)
from mata.core.types import Entity, Instance, VisionResult


class TestFuzzyLabelMatch:
    """Test fuzzy label matching utility."""

    def test_exact_match(self):
        """Test exact string match."""
        assert _fuzzy_label_match("cat", "cat")
        assert _fuzzy_label_match("dog", "dog")

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        assert _fuzzy_label_match("Cat", "cat")
        assert _fuzzy_label_match("DOG", "dog")
        assert _fuzzy_label_match("PeRsOn", "person")

    def test_plural_handling(self):
        """Test simple plural form handling (trailing 's')."""
        assert _fuzzy_label_match("cat", "cats")
        assert _fuzzy_label_match("cats", "cat")
        assert _fuzzy_label_match("dog", "dogs")
        # "ss" endings should not be stripped
        assert not _fuzzy_label_match("glass", "glas")

    def test_article_removal(self):
        """Test leading article removal (a, an, the)."""
        assert _fuzzy_label_match("the cat", "cat")
        assert _fuzzy_label_match("a dog", "dog")
        assert _fuzzy_label_match("an apple", "apple")
        assert _fuzzy_label_match("the cats", "cat")  # Article + plural

    def test_whitespace_handling(self):
        """Test extra whitespace normalization."""
        assert _fuzzy_label_match("  cat  ", "cat")
        assert _fuzzy_label_match("cat", "  cat  ")
        assert _fuzzy_label_match("  the   cat  ", "cat")

    def test_no_match(self):
        """Test labels that should not match."""
        assert not _fuzzy_label_match("cat", "dog")
        assert not _fuzzy_label_match("person", "people")  # Irregular plural
        assert not _fuzzy_label_match("child", "children")  # Irregular plural

    def test_empty_strings(self):
        """Test empty string handling."""
        assert _fuzzy_label_match("", "")
        assert not _fuzzy_label_match("cat", "")
        assert not _fuzzy_label_match("", "dog")


class TestMatchEntityToInstance:
    """Test entity-to-instance matching."""

    def test_exact_match_found(self):
        """Test exact label matching finds instance."""
        entity = Entity("cat", 0.9)
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(200, 50, 300, 200), score=0.85, label=1, label_name="dog"),
        ]

        matched = match_entity_to_instance(entity, instances, strategy="label_exact")
        assert matched is not None
        assert matched.label_name == "cat"

    def test_exact_match_not_found(self):
        """Test exact match fails on case mismatch."""
        entity = Entity("Cat", 0.9)  # Capital C
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),
        ]

        matched = match_entity_to_instance(entity, instances, strategy="label_exact")
        assert matched is None  # Exact match is case-sensitive

    def test_fuzzy_match_case_insensitive(self):
        """Test fuzzy matching handles case differences."""
        entity = Entity("Cat", 0.9)
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),
        ]

        matched = match_entity_to_instance(entity, instances, strategy="label_fuzzy")
        assert matched is not None
        assert matched.label_name == "cat"

    def test_fuzzy_match_plural(self):
        """Test fuzzy matching handles plural forms."""
        entity = Entity("cats", 0.9)
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),
        ]

        matched = match_entity_to_instance(entity, instances, strategy="label_fuzzy")
        assert matched is not None
        assert matched.label_name == "cat"

    def test_fuzzy_match_article(self):
        """Test fuzzy matching handles articles."""
        entity = Entity("the dog", 0.85)
        instances = [
            Instance(bbox=(200, 50, 300, 200), score=0.85, label=1, label_name="dog"),
        ]

        matched = match_entity_to_instance(entity, instances, strategy="label_fuzzy")
        assert matched is not None
        assert matched.label_name == "dog"

    def test_no_match_different_labels(self):
        """Test no match when labels differ."""
        entity = Entity("bird", 0.8)
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(200, 50, 300, 200), score=0.85, label=1, label_name="dog"),
        ]

        matched = match_entity_to_instance(entity, instances, strategy="label_fuzzy")
        assert matched is None

    def test_empty_instances_list(self):
        """Test matching with empty instances list."""
        entity = Entity("cat", 0.9)
        instances = []

        matched = match_entity_to_instance(entity, instances, strategy="label_fuzzy")
        assert matched is None

    def test_instance_without_label_name(self):
        """Test instance without label_name is skipped."""
        entity = Entity("cat", 0.9)
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name=None),
        ]

        matched = match_entity_to_instance(entity, instances, strategy="label_fuzzy")
        assert matched is None

    def test_invalid_strategy(self):
        """Test invalid strategy raises ValueError."""
        entity = Entity("cat", 0.9)
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),
        ]

        with pytest.raises(ValueError, match="Invalid strategy"):
            match_entity_to_instance(entity, instances, strategy="invalid")

    def test_embedding_strategy_not_implemented(self):
        """Test embedding strategy raises NotImplementedError."""
        entity = Entity("cat", 0.9)
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),
        ]

        with pytest.raises(NotImplementedError, match="Embedding-based matching"):
            match_entity_to_instance(entity, instances, strategy="embedding")


class TestMergeEntityAttributes:
    """Test entity attribute merging into instance."""

    def test_merge_uses_higher_score(self):
        """Test merging uses higher confidence score."""
        entity = Entity("cat", score=0.95)
        instance = Instance(bbox=(10, 20, 100, 150), score=0.85, label=0, label_name="cat")

        merged = merge_entity_attributes(instance, entity)
        assert merged.score == 0.95  # Entity score higher

    def test_merge_keeps_instance_score_if_higher(self):
        """Test merging keeps instance score if higher."""
        entity = Entity("cat", score=0.75)
        instance = Instance(bbox=(10, 20, 100, 150), score=0.90, label=0, label_name="cat")

        merged = merge_entity_attributes(instance, entity)
        assert merged.score == 0.90  # Instance score higher

    def test_merge_preserves_spatial_data(self):
        """Test merging preserves bbox and mask."""
        entity = Entity("cat", score=0.95, attributes={"color": "orange"})
        bbox = (10, 20, 100, 150)
        mask = {"size": [100, 100], "counts": "test_rle"}
        instance = Instance(bbox=bbox, mask=mask, score=0.85, label=0, label_name="cat")

        merged = merge_entity_attributes(instance, entity)
        assert merged.bbox == bbox
        assert merged.mask == mask
        assert merged.label == 0
        assert merged.label_name == "cat"

    def test_merge_returns_new_instance(self):
        """Test merging returns new instance (immutability)."""
        entity = Entity("cat", score=0.95)
        instance = Instance(bbox=(10, 20, 100, 150), score=0.85, label=0, label_name="cat")

        merged = merge_entity_attributes(instance, entity)

        # New instance created
        assert merged is not instance
        # Original unchanged
        assert instance.score == 0.85


class TestPromoteEntitiesToInstances:
    """Test batch entity promotion."""

    def test_promote_single_match(self):
        """Test promoting single entity with match."""
        entities = [Entity("cat", 0.95)]
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.90, label=0, label_name="cat"),
        ]

        promoted = promote_entities_to_instances(entities, instances)

        assert len(promoted) == 1
        assert promoted[0].label_name == "cat"
        assert promoted[0].bbox == (10, 20, 100, 150)
        assert promoted[0].score == 0.95  # Entity score (higher)

    def test_promote_multiple_matches(self):
        """Test promoting multiple entities."""
        entities = [
            Entity("cat", 0.95),
            Entity("dogs", 0.88),  # Plural
        ]
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.90, label=0, label_name="cat"),
            Instance(bbox=(200, 50, 300, 200), score=0.85, label=1, label_name="dog"),
        ]

        promoted = promote_entities_to_instances(entities, instances, "label_fuzzy")

        assert len(promoted) == 2
        assert promoted[0].label_name == "cat"
        assert promoted[1].label_name == "dog"

    def test_promote_partial_matches(self):
        """Test only matched entities are promoted."""
        entities = [
            Entity("cat", 0.95),
            Entity("bird", 0.88),  # No match
        ]
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.90, label=0, label_name="cat"),
        ]

        promoted = promote_entities_to_instances(entities, instances)

        assert len(promoted) == 1  # Only cat matched
        assert promoted[0].label_name == "cat"

    def test_promote_no_matches(self):
        """Test no promotion when no matches."""
        entities = [Entity("bird", 0.88)]
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.90, label=0, label_name="cat"),
        ]

        promoted = promote_entities_to_instances(entities, instances)

        assert len(promoted) == 0

    def test_promote_empty_entities(self):
        """Test empty entities list."""
        entities = []
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.90, label=0, label_name="cat"),
        ]

        promoted = promote_entities_to_instances(entities, instances)

        assert len(promoted) == 0

    def test_promote_empty_instances(self):
        """Test empty instances list."""
        entities = [Entity("cat", 0.95)]
        instances = []

        promoted = promote_entities_to_instances(entities, instances)

        assert len(promoted) == 0

    def test_promote_exact_strategy(self):
        """Test exact matching strategy."""
        entities = [Entity("Cat", 0.95)]  # Capital C
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.90, label=0, label_name="cat"),
        ]

        # Exact match fails on case
        promoted = promote_entities_to_instances(entities, instances, "label_exact")
        assert len(promoted) == 0

        # Fuzzy match succeeds
        promoted = promote_entities_to_instances(entities, instances, "label_fuzzy")
        assert len(promoted) == 1

    def test_promote_invalid_strategy(self):
        """Test invalid strategy raises ValueError."""
        entities = [Entity("cat", 0.95)]
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.90, label=0, label_name="cat"),
        ]

        with pytest.raises(ValueError, match="Invalid match_strategy"):
            promote_entities_to_instances(entities, instances, "invalid")

    def test_promote_embedding_not_implemented(self):
        """Test embedding strategy raises NotImplementedError."""
        entities = [Entity("cat", 0.95)]
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.90, label=0, label_name="cat"),
        ]

        with pytest.raises(NotImplementedError, match="Embedding-based matching"):
            promote_entities_to_instances(entities, instances, "embedding")

    def test_promote_with_mask(self):
        """Test promotion preserves mask data."""
        entities = [Entity("cat", 0.95)]
        mask = {"size": [100, 100], "counts": "test_rle"}
        instances = [
            Instance(bbox=(10, 20, 100, 150), mask=mask, score=0.90, label=0, label_name="cat"),
        ]

        promoted = promote_entities_to_instances(entities, instances)

        assert promoted[0].mask == mask

    def test_promote_with_attributes(self):
        """Test promotion with entity attributes."""
        entities = [Entity("cat", 0.95, attributes={"color": "orange", "count": 2})]
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.90, label=0, label_name="cat"),
        ]

        promoted = promote_entities_to_instances(entities, instances)

        # Note: Current implementation doesn't preserve attributes
        # (Instance doesn't have metadata field)
        # This test documents the limitation
        assert len(promoted) == 1


class TestAutoPromoteVisionResult:
    """Test automatic VisionResult entity promotion."""

    def test_auto_promote_with_spatial_source(self):
        """Test promotion with explicit spatial source."""
        vlm_result = VisionResult(
            instances=[], entities=[Entity("cat", 0.95), Entity("dog", 0.88)]  # VLM only returns entities
        )
        spatial_result = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 100, 150), score=0.90, label=0, label_name="cat"),
                Instance(bbox=(200, 50, 300, 200), score=0.85, label=1, label_name="dog"),
            ]
        )

        promoted = auto_promote_vision_result(vlm_result, spatial_source=spatial_result)

        assert len(promoted.instances) == 2
        assert len(promoted.entities) == 0  # Entities cleared after promotion
        assert promoted.instances[0].label_name == "cat"
        assert promoted.instances[1].label_name == "dog"

    def test_auto_promote_partial_matches(self):
        """Test promotion with partial matches."""
        vlm_result = VisionResult(instances=[], entities=[Entity("cat", 0.95), Entity("bird", 0.88)])  # Bird no match
        spatial_result = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 100, 150), score=0.90, label=0, label_name="cat"),
            ]
        )

        promoted = auto_promote_vision_result(vlm_result, spatial_source=spatial_result)

        assert len(promoted.instances) == 1  # Only cat matched
        assert promoted.instances[0].label_name == "cat"

    def test_auto_promote_existing_instances(self):
        """Test promotion preserves existing instances in result."""
        vlm_result = VisionResult(
            instances=[
                Instance(bbox=(0, 0, 50, 50), score=0.99, label=2, label_name="existing"),
            ],
            entities=[Entity("cat", 0.95)],
        )
        spatial_result = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 100, 150), score=0.90, label=0, label_name="cat"),
            ]
        )

        promoted = auto_promote_vision_result(vlm_result, spatial_source=spatial_result)

        assert len(promoted.instances) == 2  # Existing + promoted
        assert promoted.instances[0].label_name == "existing"
        assert promoted.instances[1].label_name == "cat"

    def test_auto_promote_no_spatial_source_with_instances(self):
        """Test auto-promotion without spatial source (instances already promoted)."""
        # Simulates VLM with auto_promote=True already returning instances
        result = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 100, 150), score=0.95, label=0, label_name="cat"),
            ],
            entities=[Entity("cat", 0.95)],  # Original entity
        )

        promoted = auto_promote_vision_result(result)

        assert len(promoted.instances) == 1
        assert len(promoted.entities) == 0  # Cleared

    def test_auto_promote_no_spatial_source_no_instances(self):
        """Test no promotion without spatial source and no instances."""
        result = VisionResult(instances=[], entities=[Entity("cat", 0.95), Entity("dog", 0.88)])

        promoted = auto_promote_vision_result(result)

        # No spatial data, entities remain unchanged
        assert len(promoted.entities) == 2
        assert len(promoted.instances) == 0

    def test_auto_promote_no_entities(self):
        """Test with no entities (returns as-is)."""
        result = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 100, 150), score=0.95, label=0, label_name="cat"),
            ]
        )

        promoted = auto_promote_vision_result(result)

        assert promoted == result  # Unchanged

    def test_auto_promote_preserves_metadata(self):
        """Test promotion preserves VisionResult metadata."""
        vlm_result = VisionResult(
            instances=[],
            entities=[Entity("cat", 0.95)],
            meta={"model": "qwen", "timestamp": 123456},
            text="A cat in the image",
            prompt="Detect objects",
        )
        spatial_result = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 100, 150), score=0.90, label=0, label_name="cat"),
            ]
        )

        promoted = auto_promote_vision_result(vlm_result, spatial_source=spatial_result)

        assert promoted.meta == vlm_result.meta
        assert promoted.text == "A cat in the image"
        assert promoted.prompt == "Detect objects"


class TestVisionResultConverters:
    """Test VisionResult conversion utilities."""

    def test_vision_result_to_detections(self):
        """Test VisionResult to Detections conversion."""
        from mata.core.artifacts.converters import vision_result_to_detections

        result = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 100, 150), score=0.95, label=0, label_name="cat"),
            ]
        )

        detections = vision_result_to_detections(result)

        assert len(detections.instances) == 1
        assert len(detections.instance_ids) == 1

    def test_detections_to_vision_result(self):
        """Test Detections to VisionResult conversion."""
        from mata.core.artifacts.converters import detections_to_vision_result
        from mata.core.artifacts.detections import Detections

        result = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 100, 150), score=0.95, label=0, label_name="cat"),
            ]
        )
        detections = Detections.from_vision_result(result)

        converted = detections_to_vision_result(detections)

        assert len(converted.instances) == 1
        assert converted.instances[0].label_name == "cat"


class TestAgentResultConverters:
    """Test AgentResult to artifact conversion utilities (Task B4)."""

    def test_agent_result_to_detections_basic(self):
        """Test basic AgentResult to Detections conversion."""
        from mata.core.agent_loop import AgentResult
        from mata.core.artifacts.converters import agent_result_to_detections
        from mata.core.tool_schema import ToolCall, ToolResult

        # Create simple agent result
        agent_result = AgentResult(
            text="Found 2 cats",
            instances=[
                Instance(bbox=(10, 20, 100, 150), score=0.95, label=0, label_name="cat"),
                Instance(bbox=(200, 50, 300, 200), score=0.87, label=0, label_name="cat"),
            ],
            entities=[Entity("cat", 0.9)],
            iterations=2,
            tool_calls=[ToolCall(tool_name="detect", arguments={}, raw_text="detect")],
            tool_results=[ToolResult(tool_name="detect", success=True, summary="Found 2 cats", artifacts={})],
        )

        detections = agent_result_to_detections(agent_result)

        assert len(detections.instances) == 2
        assert len(detections.entities) == 1
        assert detections.meta["agent_iterations"] == 2
        assert detections.meta["agent_text"] == "Found 2 cats"
        assert "agent_tool_calls" in detections.meta
        assert "agent_tool_results" in detections.meta

    def test_agent_result_to_detections_deduplication(self):
        """Test instance deduplication by IoU."""
        from mata.core.agent_loop import AgentResult
        from mata.core.artifacts.converters import agent_result_to_detections

        # Create overlapping instances (high IoU)
        agent_result = AgentResult(
            text="Found 1 cat",
            instances=[
                Instance(bbox=(10, 10, 50, 50), score=0.95, label=0, label_name="cat"),
                Instance(bbox=(12, 12, 52, 52), score=0.75, label=0, label_name="cat"),  # Overlaps
            ],
            iterations=1,
        )

        detections = agent_result_to_detections(agent_result)

        # Should deduplicate overlapping instances (keep higher score)
        assert len(detections.instances) == 1
        assert detections.instances[0].score == 0.95
        assert detections.meta["deduplication_applied"] is True
        assert detections.meta["pre_dedup_count"] == 2
        assert detections.meta["post_dedup_count"] == 1

    def test_agent_result_to_detections_no_overlap(self):
        """Test that non-overlapping instances are preserved."""
        from mata.core.agent_loop import AgentResult
        from mata.core.artifacts.converters import agent_result_to_detections

        # Create non-overlapping instances
        agent_result = AgentResult(
            text="Found 2 objects",
            instances=[
                Instance(bbox=(10, 10, 50, 50), score=0.95, label=0, label_name="cat"),
                Instance(bbox=(100, 100, 150, 150), score=0.87, label=1, label_name="dog"),
            ],
            iterations=1,
        )

        detections = agent_result_to_detections(agent_result)

        # Both instances should be kept
        assert len(detections.instances) == 2
        assert detections.meta["pre_dedup_count"] == 2
        assert detections.meta["post_dedup_count"] == 2

    def test_agent_result_to_detections_preserves_metadata(self):
        """Test that AgentResult metadata is preserved."""
        from mata.core.agent_loop import AgentResult
        from mata.core.artifacts.converters import agent_result_to_detections
        from mata.core.tool_schema import ToolCall, ToolResult

        tool_call = ToolCall(tool_name="detect", arguments={"threshold": 0.5}, raw_text="detect")
        tool_result = ToolResult(tool_name="detect", success=True, summary="Found 1 cat", artifacts={})

        agent_result = AgentResult(
            text="Analysis complete",
            instances=[Instance(bbox=(10, 20, 100, 150), score=0.95, label=0, label_name="cat")],
            entities=[Entity("cat", 0.9)],
            iterations=3,
            tool_calls=[tool_call],
            tool_results=[tool_result],
            conversation=[
                {"role": "user", "content": "Analyze this image"},
                {"role": "assistant", "content": "I'll use detect tool"},
            ],
            meta={"model": "qwen", "custom_key": "custom_value"},
        )

        detections = agent_result_to_detections(agent_result)

        assert detections.meta["agent_iterations"] == 3
        assert detections.meta["agent_text"] == "Analysis complete"
        assert detections.meta["model"] == "qwen"
        assert detections.meta["custom_key"] == "custom_value"
        assert len(detections.meta["agent_tool_calls"]) == 1
        assert detections.meta["agent_tool_calls"][0]["tool_name"] == "detect"
        assert len(detections.meta["agent_tool_results"]) == 1
        assert detections.meta["agent_tool_results"][0]["success"] is True
        assert len(detections.meta["agent_conversation"]) == 2

    def test_agent_result_to_detections_empty(self):
        """Test conversion with empty instances/entities."""
        from mata.core.agent_loop import AgentResult
        from mata.core.artifacts.converters import agent_result_to_detections

        agent_result = AgentResult(
            text="No objects found",
            instances=[],
            entities=[],
            iterations=1,
        )

        detections = agent_result_to_detections(agent_result)

        assert len(detections.instances) == 0
        assert len(detections.entities) == 0
        assert detections.meta["agent_text"] == "No objects found"

    def test_agent_result_to_detections_type_check(self):
        """Test type checking for invalid input."""
        from mata.core.artifacts.converters import agent_result_to_detections

        with pytest.raises(TypeError, match="Expected AgentResult"):
            agent_result_to_detections("not an agent result")

    def test_agent_result_to_vision_result_basic(self):
        """Test basic AgentResult to VisionResult conversion."""
        from mata.core.agent_loop import AgentResult
        from mata.core.artifacts.converters import agent_result_to_vision_result

        agent_result = AgentResult(
            text="Found objects",
            instances=[
                Instance(bbox=(10, 20, 100, 150), score=0.95, label=0, label_name="cat"),
            ],
            entities=[Entity("cat", 0.9)],
            iterations=2,
        )

        vision_result = agent_result_to_vision_result(agent_result)

        assert len(vision_result.instances) == 1
        assert len(vision_result.entities) == 1
        assert vision_result.text == "Found objects"
        assert vision_result.meta["agent_iterations"] == 2

    def test_agent_result_to_vision_result_deduplication(self):
        """Test that VisionResult conversion also deduplicates."""
        from mata.core.agent_loop import AgentResult
        from mata.core.artifacts.converters import agent_result_to_vision_result

        agent_result = AgentResult(
            text="Found 1 cat",
            instances=[
                Instance(bbox=(10, 10, 50, 50), score=0.95, label=0, label_name="cat"),
                Instance(bbox=(12, 12, 52, 52), score=0.75, label=0, label_name="cat"),
            ],
            iterations=1,
        )

        vision_result = agent_result_to_vision_result(agent_result)

        assert len(vision_result.instances) == 1
        assert vision_result.instances[0].score == 0.95
        assert vision_result.meta["deduplication_applied"] is True

    def test_agent_result_to_vision_result_preserves_all_fields(self):
        """Test that all AgentResult fields are preserved in VisionResult."""
        from mata.core.agent_loop import AgentResult
        from mata.core.artifacts.converters import agent_result_to_vision_result
        from mata.core.tool_schema import ToolCall, ToolResult

        agent_result = AgentResult(
            text="Analysis complete",
            instances=[Instance(bbox=(10, 20, 100, 150), score=0.95, label=0, label_name="cat")],
            entities=[Entity("cat", 0.9), Entity("dog", 0.8)],
            iterations=5,
            tool_calls=[
                ToolCall(tool_name="detect", arguments={}, raw_text="detect"),
                ToolCall(tool_name="classify", arguments={}, raw_text="classify"),
            ],
            tool_results=[
                ToolResult(tool_name="detect", success=True, summary="Found 1 cat", artifacts={}),
                ToolResult(tool_name="classify", success=True, summary="Cat: 0.95", artifacts={}),
            ],
            conversation=[{"role": "user", "content": "test"}],
        )

        vision_result = agent_result_to_vision_result(agent_result)

        assert vision_result.text == "Analysis complete"
        assert len(vision_result.entities) == 2
        assert vision_result.meta["agent_iterations"] == 5
        assert len(vision_result.meta["agent_tool_calls"]) == 2
        assert len(vision_result.meta["agent_tool_results"]) == 2
        assert len(vision_result.meta["agent_conversation"]) == 1

    def test_agent_result_to_vision_result_type_check(self):
        """Test type checking for invalid input."""
        from mata.core.artifacts.converters import agent_result_to_vision_result

        with pytest.raises(TypeError, match="Expected AgentResult"):
            agent_result_to_vision_result({"not": "agent_result"})

    def test_iou_calculation(self):
        """Test IoU calculation helper."""
        from mata.core.artifacts.converters import _calculate_iou

        # Perfect overlap
        box1 = (10, 10, 50, 50)
        box2 = (10, 10, 50, 50)
        assert _calculate_iou(box1, box2) == 1.0

        # No overlap
        box1 = (10, 10, 50, 50)
        box2 = (100, 100, 150, 150)
        assert _calculate_iou(box1, box2) == 0.0

        # Partial overlap
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        iou = _calculate_iou(box1, box2)
        assert 0.0 < iou < 1.0
        # Expected: intersection = 50*50 = 2500, union = 10000 + 10000 - 2500 = 17500
        # IoU = 2500/17500 ≈ 0.1429
        assert abs(iou - 0.1429) < 0.01

    def test_deduplicate_instances(self):
        """Test instance deduplication logic."""
        from mata.core.artifacts.converters import _deduplicate_instances

        instances = [
            Instance(bbox=(10, 10, 50, 50), score=0.95, label=0, label_name="cat"),
            Instance(bbox=(12, 12, 52, 52), score=0.75, label=0, label_name="cat"),  # Overlaps
            Instance(bbox=(100, 100, 150, 150), score=0.87, label=1, label_name="dog"),  # Separate
        ]

        deduped = _deduplicate_instances(instances, iou_threshold=0.7)

        # Should keep instance #1 (highest score) and #3 (no overlap)
        assert len(deduped) == 2
        assert deduped[0].score == 0.95
        assert deduped[1].score == 0.87

    def test_deduplicate_instances_no_bbox(self):
        """Test deduplication preserves instances with masks but no bbox."""
        from mata.core.artifacts.converters import _deduplicate_instances

        # Create RLE mask for instances without bbox
        rle_mask = {"size": [100, 100], "counts": "test"}

        instances = [
            Instance(bbox=None, mask=rle_mask, score=0.9, label=0, label_name="entity1"),
            Instance(bbox=(10, 10, 50, 50), score=0.95, label=1, label_name="cat"),
            Instance(bbox=None, mask=rle_mask, score=0.8, label=2, label_name="entity2"),
        ]

        deduped = _deduplicate_instances(instances, iou_threshold=0.7)

        # All should be kept (can't deduplicate mask-only instances without bbox)
        assert len(deduped) == 3

    def test_deduplicate_instances_empty(self):
        """Test deduplication with empty list."""
        from mata.core.artifacts.converters import _deduplicate_instances

        deduped = _deduplicate_instances([], iou_threshold=0.7)
        assert deduped == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
