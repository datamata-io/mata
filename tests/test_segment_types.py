"""Unit tests for segmentation type system (SegmentMask and SegmentResult)."""

import numpy as np
import pytest

from mata.core.types import SegmentMask, SegmentResult


class TestSegmentMask:
    """Test SegmentMask dataclass."""

    def test_creation_with_binary_mask(self):
        """Test creating SegmentMask with numpy binary mask."""
        binary_mask = np.array([[0, 1, 1], [1, 1, 0]], dtype=bool)

        mask = SegmentMask(
            mask=binary_mask,
            score=0.95,
            label=0,
            label_name="person",
            bbox=(10.0, 20.0, 50.0, 80.0),
            is_stuff=False,
            area=5,
        )

        assert mask.score == 0.95
        assert mask.label == 0
        assert mask.label_name == "person"
        assert mask.bbox == (10.0, 20.0, 50.0, 80.0)
        assert mask.is_stuff is False
        assert mask.area == 5
        assert mask.is_binary()
        assert not mask.is_rle()

    def test_creation_with_rle_mask(self):
        """Test creating SegmentMask with RLE format."""
        rle_mask = {"size": [480, 640], "counts": "eNqVkLuN"}

        mask = SegmentMask(mask=rle_mask, score=0.87, label=15, label_name="cat", is_stuff=False)

        assert mask.score == 0.87
        assert mask.label == 15
        assert mask.is_rle()
        assert not mask.is_binary()

    def test_creation_with_polygon(self):
        """Test creating SegmentMask with polygon format."""
        polygon = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

        mask = SegmentMask(mask=polygon, score=0.92, label=5, label_name="dog")

        assert mask.is_polygon()
        assert not mask.is_rle()
        assert not mask.is_binary()

    def test_invalid_binary_mask_shape(self):
        """Test validation rejects non-2D binary masks."""
        # 3D array should fail
        invalid_mask = np.zeros((2, 3, 4), dtype=bool)

        with pytest.raises(ValueError, match="Binary mask must be 2D"):
            SegmentMask(mask=invalid_mask, score=0.8, label=0)

    def test_invalid_rle_format(self):
        """Test validation rejects invalid RLE dictionaries."""
        # Missing 'counts' key
        invalid_rle = {"size": [480, 640]}

        with pytest.raises(ValueError, match="RLE mask dict must contain"):
            SegmentMask(mask=invalid_rle, score=0.8, label=0)

    def test_invalid_polygon_format(self):
        """Test validation rejects odd-length polygon coordinates."""
        # Odd number of coordinates
        invalid_polygon = [10.0, 20.0, 30.0]  # Missing y coordinate

        with pytest.raises(ValueError, match="Polygon must have even number"):
            SegmentMask(mask=invalid_polygon, score=0.8, label=0)

    def test_to_dict_binary_mask(self):
        """Test serializing binary mask to dictionary."""
        binary_mask = np.array([[0, 1], [1, 0]], dtype=bool)

        mask = SegmentMask(
            mask=binary_mask, score=0.95, label=0, label_name="person", bbox=(10.0, 20.0, 50.0, 80.0), area=2
        )

        result = mask.to_dict()

        assert result["mask"]["format"] == "binary"
        assert result["mask"]["data"] == [[False, True], [True, False]]
        assert result["mask"]["shape"] == [2, 2]
        assert result["score"] == 0.95
        assert result["label"] == 0
        assert result["label_name"] == "person"
        assert result["bbox"] == [10.0, 20.0, 50.0, 80.0]
        assert result["area"] == 2

    def test_to_dict_rle_mask(self):
        """Test serializing RLE mask to dictionary."""
        rle_mask = {"size": [480, 640], "counts": b"eNqVkLuN"}  # Bytes

        mask = SegmentMask(mask=rle_mask, score=0.87, label=15, label_name="cat")

        result = mask.to_dict()

        assert result["mask"]["format"] == "rle"
        assert result["mask"]["data"]["size"] == [480, 640]
        # Bytes should be decoded to string
        assert result["mask"]["data"]["counts"] == "eNqVkLuN"

    def test_to_dict_rle_mask_string_counts(self):
        """Test serializing RLE mask with string counts (already decoded)."""
        rle_mask = {"size": [480, 640], "counts": "eNqVkLuN"}  # String

        mask = SegmentMask(mask=rle_mask, score=0.87, label=15)

        result = mask.to_dict()

        assert result["mask"]["data"]["counts"] == "eNqVkLuN"

    def test_to_dict_polygon_mask(self):
        """Test serializing polygon mask to dictionary."""
        polygon = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

        mask = SegmentMask(mask=polygon, score=0.92, label=5, label_name="dog")

        result = mask.to_dict()

        assert result["mask"]["format"] == "polygon"
        assert result["mask"]["data"] == polygon


class TestSegmentResult:
    """Test SegmentResult dataclass."""

    def test_creation_empty(self):
        """Test creating empty SegmentResult."""
        result = SegmentResult(masks=[], meta={})

        assert len(result.masks) == 0
        assert result.meta == {}

    def test_creation_with_masks(self):
        """Test creating SegmentResult with masks."""
        masks = [
            SegmentMask(
                mask={"size": [480, 640], "counts": "abc"}, score=0.95, label=0, label_name="person", is_stuff=False
            ),
            SegmentMask(
                mask={"size": [480, 640], "counts": "def"}, score=0.87, label=15, label_name="cat", is_stuff=False
            ),
        ]

        result = SegmentResult(masks=masks, meta={"mode": "instance", "threshold": 0.5})

        assert len(result.masks) == 2
        assert result.meta["mode"] == "instance"
        assert result.meta["threshold"] == 0.5

    def test_to_dict(self):
        """Test converting SegmentResult to dictionary."""
        masks = [SegmentMask(mask={"size": [480, 640], "counts": "abc"}, score=0.95, label=0, label_name="person")]

        result = SegmentResult(masks=masks, meta={"mode": "instance"})

        result_dict = result.to_dict()

        assert "masks" in result_dict
        assert len(result_dict["masks"]) == 1
        assert result_dict["masks"][0]["label"] == 0
        assert result_dict["meta"]["mode"] == "instance"

    def test_to_json_roundtrip(self):
        """Test JSON serialization and deserialization."""
        masks = [
            SegmentMask(
                mask={"size": [480, 640], "counts": "abc"},
                score=0.95,
                label=0,
                label_name="person",
                bbox=(10.0, 20.0, 50.0, 80.0),
                is_stuff=False,
                area=1000,
            )
        ]

        original = SegmentResult(masks=masks, meta={"mode": "instance", "threshold": 0.5})

        # Serialize to JSON
        json_str = original.to_json()
        assert isinstance(json_str, str)

        # Deserialize from JSON
        restored = SegmentResult.from_json(json_str)

        assert len(restored.masks) == 1
        assert restored.masks[0].score == 0.95
        assert restored.masks[0].label == 0
        assert restored.masks[0].label_name == "person"
        assert restored.masks[0].bbox == (10.0, 20.0, 50.0, 80.0)
        assert restored.masks[0].is_stuff is False
        assert restored.masks[0].area == 1000
        assert restored.meta["mode"] == "instance"
        assert restored.meta["threshold"] == 0.5

    def test_from_dict_binary_mask(self):
        """Test reconstructing SegmentResult from dict with binary mask."""
        data = {
            "masks": [
                {
                    "mask": {
                        "format": "binary",
                        "data": [[False, True], [True, False]],
                        "shape": [2, 2],
                        "dtype": "bool",
                    },
                    "score": 0.95,
                    "label": 0,
                    "label_name": "person",
                    "bbox": [10.0, 20.0, 50.0, 80.0],
                    "is_stuff": False,
                    "area": 2,
                }
            ],
            "meta": {"mode": "instance"},
        }

        result = SegmentResult.from_dict(data)

        assert len(result.masks) == 1
        mask = result.masks[0]
        assert isinstance(mask.mask, np.ndarray)
        assert mask.mask.shape == (2, 2)
        assert mask.score == 0.95

    def test_filter_by_score(self):
        """Test filtering masks by confidence threshold."""
        masks = [
            SegmentMask(mask={"size": [10, 10], "counts": "a"}, score=0.9, label=0),
            SegmentMask(mask={"size": [10, 10], "counts": "b"}, score=0.7, label=1),
            SegmentMask(mask={"size": [10, 10], "counts": "c"}, score=0.5, label=2),
            SegmentMask(mask={"size": [10, 10], "counts": "d"}, score=0.3, label=3),
        ]

        result = SegmentResult(masks=masks)

        # Filter by threshold 0.6
        filtered = result.filter_by_score(0.6)

        assert len(filtered.masks) == 2
        assert filtered.masks[0].score == 0.9
        assert filtered.masks[1].score == 0.7

    def test_get_instances(self):
        """Test getting only instance masks (is_stuff=False)."""
        masks = [
            SegmentMask(mask={"size": [10, 10], "counts": "a"}, score=0.9, label=0, is_stuff=False),
            SegmentMask(mask={"size": [10, 10], "counts": "b"}, score=0.8, label=1, is_stuff=True),
            SegmentMask(mask={"size": [10, 10], "counts": "c"}, score=0.7, label=2, is_stuff=False),
            SegmentMask(mask={"size": [10, 10], "counts": "d"}, score=0.6, label=3, is_stuff=None),
        ]

        result = SegmentResult(masks=masks)
        instances = result.get_instances()

        # Should include is_stuff=False and is_stuff=None
        assert len(instances) == 3
        assert instances[0].is_stuff is False
        assert instances[1].is_stuff is False
        assert instances[2].is_stuff is None

    def test_get_stuff(self):
        """Test getting only stuff masks (is_stuff=True)."""
        masks = [
            SegmentMask(mask={"size": [10, 10], "counts": "a"}, score=0.9, label=0, is_stuff=False),
            SegmentMask(mask={"size": [10, 10], "counts": "b"}, score=0.8, label=1, is_stuff=True),
            SegmentMask(mask={"size": [10, 10], "counts": "c"}, score=0.7, label=2, is_stuff=False),
            SegmentMask(mask={"size": [10, 10], "counts": "d"}, score=0.6, label=3, is_stuff=True),
        ]

        result = SegmentResult(masks=masks)
        stuff = result.get_stuff()

        assert len(stuff) == 2
        assert stuff[0].is_stuff is True
        assert stuff[1].is_stuff is True

    def test_panoptic_mode_metadata(self):
        """Test SegmentResult with panoptic mode metadata."""
        masks = [
            SegmentMask(mask={"size": [10, 10], "counts": "a"}, score=0.9, label=0, is_stuff=False),
            SegmentMask(mask={"size": [10, 10], "counts": "b"}, score=0.8, label=80, is_stuff=True),
        ]

        result = SegmentResult(
            masks=masks,
            meta={"mode": "panoptic", "threshold": 0.5, "model_id": "facebook/mask2former-swin-tiny-coco-panoptic"},
        )

        assert result.meta["mode"] == "panoptic"
        assert len(result.get_instances()) == 1
        assert len(result.get_stuff()) == 1
