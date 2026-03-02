"""Unit tests for classification type system (Classification and ClassifyResult)."""

import json

import pytest

from mata.core.types import Classification, ClassifyResult


class TestClassification:
    """Test Classification dataclass."""

    def test_creation_with_all_fields(self):
        """Test creating Classification with all fields."""
        classification = Classification(label=281, score=0.95, label_name="tabby cat")

        assert classification.label == 281
        assert classification.score == 0.95
        assert classification.label_name == "tabby cat"

    def test_creation_without_label_name(self):
        """Test creating Classification without label name."""
        classification = Classification(label=5, score=0.87)

        assert classification.label == 5
        assert classification.score == 0.87
        assert classification.label_name is None

    def test_to_dict_with_label_name(self):
        """Test serializing Classification to dictionary."""
        classification = Classification(label=10, score=0.92, label_name="dog")

        result = classification.to_dict()

        assert result == {"label": 10, "score": 0.92, "label_name": "dog"}

    def test_to_dict_without_label_name(self):
        """Test serializing Classification without label name."""
        classification = Classification(label=15, score=0.78)

        result = classification.to_dict()

        assert result == {"label": 15, "score": 0.78, "label_name": None}

    def test_immutability(self):
        """Test that Classification is immutable (frozen dataclass)."""
        classification = Classification(label=0, score=0.5)

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            classification.label = 1


class TestClassifyResult:
    """Test ClassifyResult dataclass."""

    def test_creation_with_predictions(self):
        """Test creating ClassifyResult with predictions."""
        predictions = [
            Classification(label=281, score=0.95, label_name="tabby cat"),
            Classification(label=282, score=0.03, label_name="tiger cat"),
            Classification(label=285, score=0.01, label_name="Egyptian cat"),
        ]

        result = ClassifyResult(predictions=predictions)

        assert len(result.predictions) == 3
        assert result.predictions[0].label == 281
        assert result.predictions[0].score == 0.95
        assert result.meta is None

    def test_creation_with_metadata(self):
        """Test creating ClassifyResult with metadata."""
        predictions = [Classification(label=0, score=0.85, label_name="class_0")]
        meta = {"model_id": "microsoft/resnet-50", "device": "cuda", "top_k": 5}

        result = ClassifyResult(predictions=predictions, meta=meta)

        assert result.meta == meta
        assert result.meta["model_id"] == "microsoft/resnet-50"

    def test_to_dict(self):
        """Test serializing ClassifyResult to dictionary."""
        predictions = [
            Classification(label=281, score=0.95, label_name="tabby cat"),
            Classification(label=282, score=0.03, label_name="tiger cat"),
        ]
        meta = {"model_id": "test-model"}

        result = ClassifyResult(predictions=predictions, meta=meta)
        result_dict = result.to_dict()

        assert result_dict == {
            "predictions": [
                {"label": 281, "score": 0.95, "label_name": "tabby cat"},
                {"label": 282, "score": 0.03, "label_name": "tiger cat"},
            ],
            "meta": {"model_id": "test-model"},
        }

    def test_to_json(self):
        """Test serializing ClassifyResult to JSON string."""
        predictions = [Classification(label=10, score=0.88, label_name="dog")]

        result = ClassifyResult(predictions=predictions)
        json_str = result.to_json()

        # Parse JSON to verify structure
        parsed = json.loads(json_str)
        assert parsed["predictions"][0]["label"] == 10
        assert parsed["predictions"][0]["score"] == 0.88
        assert parsed["predictions"][0]["label_name"] == "dog"

    def test_to_json_with_indent(self):
        """Test JSON serialization with custom formatting."""
        predictions = [Classification(label=0, score=0.9)]

        result = ClassifyResult(predictions=predictions)
        json_str = result.to_json(indent=2)

        # Verify indentation is applied
        assert "\n" in json_str
        assert "  " in json_str

    def test_from_dict(self):
        """Test deserializing ClassifyResult from dictionary."""
        data = {
            "predictions": [
                {"label": 281, "score": 0.95, "label_name": "tabby cat"},
                {"label": 282, "score": 0.03, "label_name": "tiger cat"},
            ],
            "meta": {"model_id": "test-model"},
        }

        result = ClassifyResult.from_dict(data)

        assert len(result.predictions) == 2
        assert result.predictions[0].label == 281
        assert result.predictions[0].score == 0.95
        assert result.predictions[0].label_name == "tabby cat"
        assert result.predictions[1].label == 282
        assert result.meta == {"model_id": "test-model"}

    def test_from_dict_without_label_names(self):
        """Test deserialization when label_name is missing."""
        data = {
            "predictions": [
                {"label": 5, "score": 0.8},
            ]
        }

        result = ClassifyResult.from_dict(data)

        assert result.predictions[0].label == 5
        assert result.predictions[0].score == 0.8
        assert result.predictions[0].label_name is None

    def test_from_json(self):
        """Test deserializing ClassifyResult from JSON string."""
        json_str = """
        {
            "predictions": [
                {"label": 10, "score": 0.88, "label_name": "dog"},
                {"label": 15, "score": 0.06, "label_name": "cat"}
            ],
            "meta": {"device": "cpu"}
        }
        """

        result = ClassifyResult.from_json(json_str)

        assert len(result.predictions) == 2
        assert result.predictions[0].label == 10
        assert result.predictions[0].label_name == "dog"
        assert result.meta["device"] == "cpu"

    def test_get_top1_with_predictions(self):
        """Test getting top-1 prediction."""
        predictions = [
            Classification(label=281, score=0.95, label_name="tabby cat"),
            Classification(label=282, score=0.03, label_name="tiger cat"),
        ]

        result = ClassifyResult(predictions=predictions)
        top1 = result.get_top1()

        assert top1 is not None
        assert top1.label == 281
        assert top1.score == 0.95
        assert top1.label_name == "tabby cat"

    def test_get_top1_empty_predictions(self):
        """Test getting top-1 from empty predictions."""
        result = ClassifyResult(predictions=[])
        top1 = result.get_top1()

        assert top1 is None

    def test_filter_by_score(self):
        """Test filtering predictions by score threshold."""
        predictions = [
            Classification(label=0, score=0.95, label_name="class_0"),
            Classification(label=1, score=0.80, label_name="class_1"),
            Classification(label=2, score=0.60, label_name="class_2"),
            Classification(label=3, score=0.40, label_name="class_3"),
            Classification(label=4, score=0.10, label_name="class_4"),
        ]

        result = ClassifyResult(predictions=predictions, meta={"model": "test"})
        filtered = result.filter_by_score(0.70)

        assert len(filtered.predictions) == 2
        assert filtered.predictions[0].label == 0
        assert filtered.predictions[1].label == 1
        assert filtered.meta == {"model": "test"}  # Metadata preserved

    def test_filter_by_score_all_below_threshold(self):
        """Test filtering when all predictions are below threshold."""
        predictions = [
            Classification(label=0, score=0.30),
            Classification(label=1, score=0.20),
        ]

        result = ClassifyResult(predictions=predictions)
        filtered = result.filter_by_score(0.50)

        assert len(filtered.predictions) == 0

    def test_filter_by_score_all_above_threshold(self):
        """Test filtering when all predictions are above threshold."""
        predictions = [
            Classification(label=0, score=0.95),
            Classification(label=1, score=0.90),
        ]

        result = ClassifyResult(predictions=predictions)
        filtered = result.filter_by_score(0.10)

        assert len(filtered.predictions) == 2

    def test_immutability(self):
        """Test that ClassifyResult is immutable (frozen dataclass)."""
        result = ClassifyResult(predictions=[])

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            result.predictions = []


class TestClassifyResultRoundTrip:
    """Test serialization/deserialization round-trip consistency."""

    def test_dict_round_trip(self):
        """Test converting to dict and back preserves data."""
        original = ClassifyResult(
            predictions=[
                Classification(label=281, score=0.95, label_name="tabby cat"),
                Classification(label=282, score=0.03, label_name="tiger cat"),
            ],
            meta={"model_id": "microsoft/resnet-50", "device": "cuda"},
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = ClassifyResult.from_dict(data)

        # Verify predictions match
        assert len(restored.predictions) == len(original.predictions)
        for orig_pred, rest_pred in zip(original.predictions, restored.predictions):
            assert orig_pred.label == rest_pred.label
            assert orig_pred.score == rest_pred.score
            assert orig_pred.label_name == rest_pred.label_name

        # Verify metadata matches
        assert restored.meta == original.meta

    def test_json_round_trip(self):
        """Test converting to JSON and back preserves data."""
        original = ClassifyResult(
            predictions=[
                Classification(label=10, score=0.88, label_name="dog"),
                Classification(label=15, score=0.06, label_name="cat"),
            ],
            meta={"top_k": 5},
        )

        # Convert to JSON and back
        json_str = original.to_json()
        restored = ClassifyResult.from_json(json_str)

        # Verify predictions match
        assert len(restored.predictions) == len(original.predictions)
        assert restored.predictions[0].label == 10
        assert restored.predictions[1].label == 15

        # Verify metadata matches
        assert restored.meta == original.meta

    def test_round_trip_empty_predictions(self):
        """Test round-trip with empty predictions list."""
        original = ClassifyResult(predictions=[], meta={"note": "empty"})

        json_str = original.to_json()
        restored = ClassifyResult.from_json(json_str)

        assert len(restored.predictions) == 0
        assert restored.meta == {"note": "empty"}
