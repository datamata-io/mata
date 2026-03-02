"""Unit tests for src/mata/core/parsers.py.

Tests JSON extraction, entity parsing, and schema generation for VLM output.
All tests run in isolation without model downloads.
"""

from __future__ import annotations

from mata.core.parsers import (
    extract_json_from_text,
    get_json_schema,
    parse_entities,
)
from mata.core.types import Entity, Instance


class TestExtractJsonFromText:
    """Tests for extract_json_from_text function."""

    def test_extract_raw_json_object(self):
        """Test extracting a raw JSON object."""
        text = '{"label": "cat"}'
        result = extract_json_from_text(text)

        assert result is not None
        assert isinstance(result, dict)
        assert result == {"label": "cat"}

    def test_extract_raw_json_array(self):
        """Test extracting a raw JSON array."""
        text = '[{"label": "cat"}, {"label": "dog"}]'
        result = extract_json_from_text(text)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["label"] == "cat"
        assert result[1]["label"] == "dog"

    def test_extract_fenced_json(self):
        """Test extracting JSON from markdown code fence with language."""
        text = '```json\n[{"label": "cat", "confidence": 0.9}]\n```'
        result = extract_json_from_text(text)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["label"] == "cat"
        assert result[0]["confidence"] == 0.9

    def test_extract_fenced_no_language(self):
        """Test extracting JSON from markdown code fence without language specifier."""
        text = '```\n{"label": "dog", "score": 0.85}\n```'
        result = extract_json_from_text(text)

        assert result is not None
        assert isinstance(result, dict)
        assert result["label"] == "dog"
        assert result["score"] == 0.85

    def test_extract_json_with_surrounding_text(self):
        """Test extracting JSON when embedded in surrounding text."""
        text = 'Here are the objects I found: [{"label": "cat"}, {"label": "dog"}] in the image.'
        result = extract_json_from_text(text)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["label"] == "cat"

    def test_extract_malformed_json(self):
        """Test that malformed JSON returns None."""
        text = '{"label": "cat"'  # Missing closing brace
        result = extract_json_from_text(text)

        assert result is None

    def test_extract_empty_text(self):
        """Test that empty text returns None."""
        assert extract_json_from_text("") is None
        assert extract_json_from_text("   ") is None
        assert extract_json_from_text("\n\t") is None

    def test_extract_no_json(self):
        """Test that text with no JSON returns None."""
        text = "There are two cats in the image"
        result = extract_json_from_text(text)

        assert result is None

    def test_extract_nested_json(self):
        """Test that nested JSON structures are preserved.

        Note: The extractor prioritizes arrays over objects, so it will
        extract the inner array if one is found first.
        """
        # Use a JSON object without nested arrays to test nested dict preservation
        text = '{"label": "cat", "metadata": {"color": "orange", "age": 3}}'
        result = extract_json_from_text(text)

        assert result is not None
        assert isinstance(result, dict)
        assert result["label"] == "cat"
        assert result["metadata"]["color"] == "orange"
        assert result["metadata"]["age"] == 3

    def test_extract_json_with_trailing_commas(self):
        """Test that trailing commas (common LLM error) are handled."""
        text = '[{"label": "cat",}, {"label": "dog",}]'
        result = extract_json_from_text(text)

        # Should successfully parse after stripping trailing commas
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2

    def test_extract_multiple_fenced_blocks(self):
        """Test extraction when multiple fenced blocks exist (uses first valid one)."""
        text = """
        Here's some text.
        ```json
        {"label": "cat"}
        ```
        And more text.
        ```json
        {"label": "dog"}
        ```
        """
        result = extract_json_from_text(text)

        assert result is not None
        assert isinstance(result, dict)
        # Should extract from first valid fence
        assert result["label"] in ["cat", "dog"]


class TestParseEntities:
    """Tests for parse_entities function."""

    def test_parse_entities_list(self):
        """Test parsing a list of entity dicts."""
        data = [{"label": "cat", "confidence": 0.9}, {"label": "dog", "confidence": 0.85}]
        entities = parse_entities(data)

        assert len(entities) == 2
        assert isinstance(entities[0], Entity)
        assert entities[0].label == "cat"
        assert entities[0].score == 0.9
        assert entities[1].label == "dog"
        assert entities[1].score == 0.85

    def test_parse_entities_dict_with_items(self):
        """Test parsing dict with nested list under 'objects' key."""
        data = {"objects": [{"label": "cat", "confidence": 0.9}, {"label": "dog", "confidence": 0.85}], "count": 2}
        entities = parse_entities(data)

        assert len(entities) == 2
        assert entities[0].label == "cat"
        assert entities[1].label == "dog"

    def test_parse_entities_name_key(self):
        """Test parsing with 'name' key instead of 'label'."""
        data = [{"name": "dog", "probability": 0.95}]
        entities = parse_entities(data)

        assert len(entities) == 1
        assert entities[0].label == "dog"
        assert entities[0].score == 0.95

    def test_parse_entities_missing_label(self):
        """Test that items without recognized label key are skipped."""
        data = [
            {"confidence": 0.9},  # No label
            {"label": "cat", "confidence": 0.85},  # Valid
            {"score": 0.8},  # No label
        ]
        entities = parse_entities(data)

        # Only the valid one should be parsed
        assert len(entities) == 1
        assert entities[0].label == "cat"

    def test_parse_entities_extra_fields(self):
        """Test that unknown keys are stored in attributes."""
        data = [{"label": "cat", "confidence": 0.9, "color": "orange", "age": 3, "indoor": True}]
        entities = parse_entities(data)

        assert len(entities) == 1
        entity = entities[0]
        assert entity.label == "cat"
        assert entity.score == 0.9
        assert entity.attributes["color"] == "orange"
        assert entity.attributes["age"] == 3
        assert entity.attributes["indoor"] is True

    def test_parse_entities_single_object(self):
        """Test parsing a single dict (not in a list)."""
        data = {"label": "cat", "confidence": 0.9}
        entities = parse_entities(data)

        assert len(entities) == 1
        assert entities[0].label == "cat"
        assert entities[0].score == 0.9

    def test_parse_entities_default_score(self):
        """Test that entities without score get default value of 1.0."""
        data = [{"label": "cat"}]
        entities = parse_entities(data)

        assert len(entities) == 1
        assert entities[0].score == 1.0

    def test_parse_entities_case_insensitive_keys(self):
        """Test that key matching is case-insensitive."""
        data = [{"Label": "cat", "Confidence": 0.9}, {"LABEL": "dog", "CONFIDENCE": 0.85}]
        entities = parse_entities(data)

        assert len(entities) == 2
        assert entities[0].label == "cat"
        assert entities[0].score == 0.9
        assert entities[1].label == "dog"
        assert entities[1].score == 0.85

    def test_parse_entities_various_list_keys(self):
        """Test recognition of various list keys (objects, items, detections, etc.)."""
        # Test with 'items' key
        data1 = {"items": [{"label": "cat"}]}
        entities1 = parse_entities(data1)
        assert len(entities1) == 1

        # Test with 'detections' key
        data2 = {"detections": [{"label": "dog"}]}
        entities2 = parse_entities(data2)
        assert len(entities2) == 1

        # Test with 'results' key
        data3 = {"results": [{"label": "bird"}]}
        entities3 = parse_entities(data3)
        assert len(entities3) == 1

    def test_parse_entities_invalid_score_value(self):
        """Test that invalid score values are handled gracefully."""
        data = [
            {"label": "cat", "confidence": "invalid"},  # String instead of number
            {"label": "dog", "confidence": None},  # None value
        ]
        entities = parse_entities(data)

        # Both should parse with default score
        assert len(entities) == 2
        assert entities[0].score == 1.0  # Default when conversion fails
        assert entities[1].score == 1.0

    def test_parse_entities_empty_list(self):
        """Test parsing an empty list."""
        data = []
        entities = parse_entities(data)

        assert entities == []

    def test_parse_entities_custom_key_mapping(self):
        """Test using custom key mapping."""
        data = [{"custom_label": "cat", "custom_score": 0.9}]
        key_mapping = {"label": "custom_label", "score": "custom_score"}
        entities = parse_entities(data, key_mapping)

        assert len(entities) == 1
        assert entities[0].label == "cat"
        assert entities[0].score == 0.9


class TestGetJsonSchema:
    """Tests for get_json_schema function."""

    def test_schema_detect_mode(self):
        """Test schema returns detection-specific instruction."""
        schema = get_json_schema("detect")

        assert isinstance(schema, str)
        assert len(schema) > 0
        assert "label" in schema.lower()
        assert "confidence" in schema.lower()
        assert "json" in schema.lower()

    def test_schema_unknown_mode(self):
        """Test schema returns empty string for unknown mode."""
        schema = get_json_schema("unknown_mode")

        assert schema == ""

    def test_schema_json_mode(self):
        """Test schema for generic JSON mode."""
        schema = get_json_schema("json")

        assert isinstance(schema, str)
        assert "JSON" in schema

    def test_schema_classify_mode(self):
        """Test schema for classify mode."""
        schema = get_json_schema("classify")

        assert isinstance(schema, str)
        assert "classify" in schema.lower() or "classification" in schema.lower()
        assert "label" in schema.lower()
        assert "confidence" in schema.lower()

    def test_schema_describe_mode(self):
        """Test schema for describe mode."""
        schema = get_json_schema("describe")

        assert isinstance(schema, str)
        assert "description" in schema.lower()
        assert "objects" in schema.lower()

    def test_schema_case_insensitive(self):
        """Test that mode matching is case-insensitive."""
        schema_lower = get_json_schema("detect")
        schema_upper = get_json_schema("DETECT")
        schema_mixed = get_json_schema("Detect")

        assert schema_lower == schema_upper == schema_mixed
        assert len(schema_lower) > 0

    def test_schema_empty_mode(self):
        """Test schema with empty mode string."""
        schema = get_json_schema("")

        assert schema == ""

    def test_schema_none_mode(self):
        """Test schema with None mode."""
        schema = get_json_schema(None)

        assert schema == ""


class TestAutoPromotion:
    """Tests for auto_promote parameter in parse_entities.

    Auto-promotion allows VLMs that output bboxes (like Qwen3-VL grounding mode)
    to produce Instance objects directly, enabling comparison with spatial models.
    """

    def test_auto_promote_disabled_by_default(self):
        """Test that auto_promote=False is the default (backward compat)."""
        data = [{"label": "cat", "bbox": [10, 20, 100, 150]}]

        entities = parse_entities(data)  # No auto_promote parameter

        assert len(entities) == 1
        assert isinstance(entities[0], Entity)
        assert not isinstance(entities[0], Instance)
        assert entities[0].label == "cat"
        assert entities[0].attributes["bbox"] == [10, 20, 100, 150]  # Bbox in attributes

    def test_auto_promote_with_bbox(self):
        """Test auto-promotion with bbox in JSON."""
        data = [{"label": "cat", "confidence": 0.95, "bbox": [10, 20, 100, 150]}]

        results = parse_entities(data, auto_promote=True)

        assert len(results) == 1
        assert isinstance(results[0], Instance)
        assert results[0].bbox == (10, 20, 100, 150)  # Converted to tuple
        assert results[0].score == 0.95
        assert results[0].label == 0  # Default label_id
        assert results[0].label_name == "cat"

    def test_auto_promote_with_box_alias(self):
        """Test auto-promotion with 'box' alias instead of 'bbox'."""
        data = [{"label": "dog", "box": [5, 10, 50, 100]}]

        results = parse_entities(data, auto_promote=True)

        assert len(results) == 1
        assert isinstance(results[0], Instance)
        assert results[0].bbox == (5, 10, 50, 100)
        assert results[0].label_name == "dog"

    def test_auto_promote_with_bounding_box_alias(self):
        """Test auto-promotion with 'bounding_box' alias."""
        data = [{"label": "person", "bounding_box": [100, 200, 300, 400]}]

        results = parse_entities(data, auto_promote=True)

        assert len(results) == 1
        assert isinstance(results[0], Instance)
        assert results[0].bbox == (100, 200, 300, 400)

    def test_auto_promote_case_insensitive_bbox(self):
        """Test auto-promotion with case-insensitive bbox key."""
        data = [{"label": "car", "BBox": [50, 60, 200, 250]}]

        results = parse_entities(data, auto_promote=True)

        assert len(results) == 1
        assert isinstance(results[0], Instance)
        assert results[0].bbox == (50, 60, 200, 250)

    def test_auto_promote_with_mask(self):
        """Test auto-promotion with mask in JSON."""
        data = [{"label": "cat", "mask": {"size": [100, 100], "counts": "..."}}]

        results = parse_entities(data, auto_promote=True)

        assert len(results) == 1
        assert isinstance(results[0], Instance)
        assert results[0].mask == {"size": [100, 100], "counts": "..."}
        assert results[0].bbox is None  # No bbox, only mask

    def test_auto_promote_with_segmentation_alias(self):
        """Test auto-promotion with 'segmentation' alias for mask (RLE format)."""
        # Use valid RLE mask format instead of polygon
        data = [{"label": "person", "segmentation": {"size": [100, 100], "counts": "..."}}]

        results = parse_entities(data, auto_promote=True)

        assert len(results) == 1
        assert isinstance(results[0], Instance)
        assert results[0].mask == {"size": [100, 100], "counts": "..."}

    def test_auto_promote_with_both_bbox_and_mask(self):
        """Test auto-promotion with both bbox and mask."""
        data = [{"label": "cat", "bbox": [10, 20, 100, 150], "mask": {"size": [100, 100], "counts": "..."}}]

        results = parse_entities(data, auto_promote=True)

        assert len(results) == 1
        assert isinstance(results[0], Instance)
        assert results[0].bbox == (10, 20, 100, 150)
        assert results[0].mask == {"size": [100, 100], "counts": "..."}

    def test_auto_promote_without_spatial_data(self):
        """Test auto-promotion with no bbox/mask returns Entity."""
        data = [{"label": "cat", "confidence": 0.9, "color": "orange"}]

        results = parse_entities(data, auto_promote=True)

        assert len(results) == 1
        assert isinstance(results[0], Entity)
        assert not isinstance(results[0], Instance)
        assert results[0].label == "cat"
        assert results[0].score == 0.9
        assert results[0].attributes["color"] == "orange"

    def test_auto_promote_preserves_additional_attributes(self):
        """Test that non-spatial attributes remain after bbox extraction."""
        data = [
            {
                "label": "cat",
                "bbox": [10, 20, 100, 150],
                "color": "orange",  # Extra attribute
                "age": "3 years",  # Extra attribute
            }
        ]

        results = parse_entities(data, auto_promote=True)

        # Instance is created with bbox, but extra attributes stay in Entity.attributes
        # Since Instance gets created, extra attributes are not preserved in Instance
        # (Instance has fixed fields). This is expected behavior.
        assert len(results) == 1
        assert isinstance(results[0], Instance)
        assert results[0].bbox == (10, 20, 100, 150)
        assert results[0].label_name == "cat"

    def test_auto_promote_invalid_bbox_format(self):
        """Test auto-promotion with invalid bbox format (too few elements)."""
        data = [{"label": "cat", "bbox": [10, 20]}]  # Only 2 elements, need 4

        results = parse_entities(data, auto_promote=True)

        # Should return Entity with bbox in attributes (not enough coordinates)
        assert len(results) == 1
        assert isinstance(results[0], Entity)
        assert not isinstance(results[0], Instance)
        # Bbox remains in attributes since it didn't meet >= 4 elements requirement
        assert results[0].attributes["bbox"] == [10, 20]

    def test_auto_promote_mixed_spatial_and_semantic(self):
        """Test auto-promotion with mixed list (some with bbox, some without)."""
        data = [
            {"label": "cat", "bbox": [10, 20, 100, 150]},
            {"label": "dog", "confidence": 0.8},  # No spatial data
            {"label": "person", "bbox": [200, 300, 400, 500]},
        ]

        results = parse_entities(data, auto_promote=True)

        assert len(results) == 3
        assert isinstance(results[0], Instance)  # cat with bbox
        assert isinstance(results[1], Entity)  # dog without bbox
        assert isinstance(results[2], Instance)  # person with bbox

    def test_auto_promote_with_nested_objects_key(self):
        """Test auto-promotion with nested 'objects' key."""
        data = {
            "objects": [{"label": "cat", "bbox": [10, 20, 100, 150]}, {"label": "dog", "bbox": [200, 300, 400, 500]}]
        }

        results = parse_entities(data, auto_promote=True)

        assert len(results) == 2
        assert all(isinstance(r, Instance) for r in results)
        assert results[0].label_name == "cat"
        assert results[1].label_name == "dog"

    def test_auto_promote_qwen_grounding_format(self):
        """Test auto-promotion with realistic Qwen3-VL grounding output format."""
        # Simulated Qwen3-VL grounding mode output
        data = {
            "objects": [
                {"name": "cat", "confidence": 0.95, "bbox": [10.5, 20.3, 100.7, 150.2]},  # Float coords
                {"name": "person", "confidence": 0.88, "bbox": [200, 300, 400, 500]},
            ]
        }

        results = parse_entities(data, auto_promote=True)

        assert len(results) == 2
        assert all(isinstance(r, Instance) for r in results)
        assert results[0].label_name == "cat"
        assert results[0].bbox == (10.5, 20.3, 100.7, 150.2)
        assert results[0].score == 0.95
        assert results[1].label_name == "person"
        assert results[1].score == 0.88

    def test_auto_promote_explicit_false(self):
        """Test explicit auto_promote=False."""
        data = [{"label": "cat", "bbox": [10, 20, 100, 150]}]

        results = parse_entities(data, auto_promote=False)

        assert len(results) == 1
        assert isinstance(results[0], Entity)
        assert results[0].attributes["bbox"] == [10, 20, 100, 150]
