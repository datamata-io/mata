"""Test suite for result save() functionality.

Tests all save formats (JSON, CSV, image overlays, crops) for all result types.
Uses mocks to avoid actual file I/O during testing.
"""

import json
from unittest.mock import Mock, patch

import pytest

# Import result types
from mata.core.types import (
    Classification,
    ClassifyResult,
    Detection,
    DetectResult,
    Instance,
    SegmentMask,
    SegmentResult,
    VisionResult,
)


class TestVisionResultSave:
    """Test VisionResult.save() method."""

    def test_save_json(self, tmp_path):
        """Test JSON export."""
        result = VisionResult(
            instances=[Instance(bbox=(10, 20, 100, 200), score=0.95, label=0, label_name="cat")],
            meta={"model_id": "test", "input_path": "test.jpg"},
        )

        output_path = tmp_path / "result.json"
        result.save(str(output_path))

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "instances" in data
        assert len(data["instances"]) == 1
        assert data["instances"][0]["label_name"] == "cat"

    def test_save_csv_detections(self, tmp_path):
        """Test CSV export for detections."""
        result = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 100, 200), score=0.95, label=0, label_name="cat"),
                Instance(bbox=(150, 30, 250, 180), score=0.87, label=1, label_name="dog"),
            ],
            meta={"input_path": "test.jpg"},
        )

        output_path = tmp_path / "result.csv"
        result.save(str(output_path))

        assert output_path.exists()
        csv_content = output_path.read_text()
        assert "cat" in csv_content
        assert "dog" in csv_content
        assert "0.95" in csv_content

    @patch("mata.core.exporters.image_exporter._load_image")
    @patch("mata.core.exporters.image_exporter._draw_bounding_boxes")
    def test_save_image_overlay(self, mock_draw, mock_load, tmp_path):
        """Test image overlay export."""
        from PIL import Image

        # Mock image loading
        mock_img = Mock(spec=Image.Image)
        mock_img.save = Mock()
        mock_load.return_value = mock_img
        mock_draw.return_value = mock_img

        result = VisionResult(
            instances=[Instance(bbox=(10, 20, 100, 200), score=0.95, label=0, label_name="cat")],
            meta={"input_path": "test.jpg"},
        )

        output_path = tmp_path / "overlay.png"
        result.save(str(output_path))

        mock_load.assert_called_once()
        mock_img.save.assert_called_once()

    @patch("mata.core.exporters.crop_exporter._load_image")
    def test_save_crops(self, mock_load, tmp_path):
        """Test crop extraction."""
        from PIL import Image

        # Mock image
        mock_img = Mock(spec=Image.Image)
        mock_img.size = (640, 480)
        mock_crop = Mock(spec=Image.Image)
        mock_crop.save = Mock()
        mock_img.crop = Mock(return_value=mock_crop)
        mock_load.return_value = mock_img

        result = VisionResult(
            instances=[Instance(bbox=(10, 20, 100, 200), score=0.95, label=0, label_name="cat")],
            meta={"input_path": "test.jpg"},
        )

        output_path = tmp_path / "crops.png"
        result.save(str(output_path), crop_dir="test_crops")

        mock_load.assert_called_once()
        mock_img.crop.assert_called_once()

    def test_save_invalid_format(self):
        """Test error on unsupported format."""
        result = VisionResult(instances=[], meta={})

        with pytest.raises(ValueError, match="Unsupported file format"):
            result.save("output.xyz")

    def test_get_input_path(self):
        """Test get_input_path() convenience method."""
        result = VisionResult(instances=[], meta={"input_path": "test.jpg"})

        assert result.get_input_path() == "test.jpg"

        # No input_path
        result_no_path = VisionResult(instances=[], meta={})
        assert result_no_path.get_input_path() is None

    def test_save_empty_result(self, tmp_path):
        """Test saving empty result (no instances)."""
        result = VisionResult(instances=[], meta={})

        output_path = tmp_path / "empty.json"
        result.save(str(output_path))

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["instances"] == []


class TestDetectResultSave:
    """Test DetectResult.save() method (legacy)."""

    def test_save_json(self, tmp_path):
        """Test JSON export."""
        result = DetectResult(
            detections=[Detection(bbox=(10, 20, 100, 200), score=0.95, label=0, label_name="cat")],
            meta={"input_path": "test.jpg"},
        )

        output_path = tmp_path / "detect.json"
        result.save(str(output_path))

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "detections" in data
        assert len(data["detections"]) == 1

    def test_save_csv(self, tmp_path):
        """Test CSV export."""
        result = DetectResult(
            detections=[Detection(bbox=(10, 20, 100, 200), score=0.95, label=0, label_name="cat")], meta={}
        )

        output_path = tmp_path / "detect.csv"
        result.save(str(output_path))

        assert output_path.exists()
        csv_content = output_path.read_text()
        assert "cat" in csv_content


class TestSegmentResultSave:
    """Test SegmentResult.save() method (legacy)."""

    def test_save_json(self, tmp_path):
        """Test JSON export."""
        import numpy as np

        result = SegmentResult(
            masks=[SegmentMask(mask=np.array([[0, 1], [1, 1]], dtype=bool), score=0.95, label=0, label_name="cat")],
            meta={"input_path": "test.jpg"},
        )

        output_path = tmp_path / "segment.json"
        result.save(str(output_path))

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "masks" in data

    def test_save_csv(self, tmp_path):
        """Test CSV export."""
        import numpy as np

        result = SegmentResult(
            masks=[
                SegmentMask(mask=np.array([[0, 1], [1, 1]], dtype=bool), score=0.95, label=0, label_name="cat", area=3)
            ],
            meta={},
        )

        output_path = tmp_path / "segment.csv"
        result.save(str(output_path))

        assert output_path.exists()
        csv_content = output_path.read_text()
        assert "cat" in csv_content
        assert "yes" in csv_content  # has_mask


class TestClassifyResultSave:
    """Test ClassifyResult.save() method."""

    def test_save_json(self, tmp_path):
        """Test JSON export."""
        result = ClassifyResult(
            predictions=[
                Classification(label=0, score=0.95, label_name="cat"),
                Classification(label=1, score=0.03, label_name="dog"),
            ],
            meta={"input_path": "test.jpg"},
        )

        output_path = tmp_path / "classify.json"
        result.save(str(output_path))

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    def test_save_csv(self, tmp_path):
        """Test CSV export."""
        result = ClassifyResult(
            predictions=[
                Classification(label=0, score=0.95, label_name="cat"),
                Classification(label=1, score=0.03, label_name="dog"),
            ],
            meta={},
        )

        output_path = tmp_path / "classify.csv"
        result.save(str(output_path))

        assert output_path.exists()
        csv_content = output_path.read_text()
        assert "rank" in csv_content
        assert "cat" in csv_content
        assert "0.95" in csv_content

    def test_save_bar_chart(self, tmp_path):
        """Test bar chart visualization."""
        # Skip if matplotlib not installed
        pytest.importorskip("matplotlib")

        result = ClassifyResult(
            predictions=[
                Classification(label=0, score=0.95, label_name="cat"),
                Classification(label=1, score=0.03, label_name="dog"),
            ],
            meta={},
        )

        output_path = tmp_path / "chart.png"
        result.save(str(output_path))

        assert output_path.exists()


class TestInputPathInjection:
    """Test that input_path is correctly injected from adapters."""

    def test_adapter_stores_input_path(self):
        """Test that adapters store input_path in meta."""
        # This is tested via integration tests with real adapters
        # The unit test would require mocking entire adapter initialization
        # which is better suited for integration testing
        pytest.skip("Tested via integration tests")


class TestExporterFunctions:
    """Test individual exporter functions."""

    def test_export_json_function(self, tmp_path):
        """Test export_json function directly."""
        from mata.core.exporters import export_json

        result = VisionResult(instances=[Instance(bbox=(10, 20, 100, 200), score=0.95, label=0)], meta={})

        output_path = tmp_path / "test.json"
        export_json(result, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "instances" in data

    def test_export_csv_function(self, tmp_path):
        """Test export_csv function directly."""
        from mata.core.exporters import export_csv

        result = VisionResult(
            instances=[Instance(bbox=(10, 20, 100, 200), score=0.95, label=0, label_name="cat")], meta={}
        )

        output_path = tmp_path / "test.csv"
        export_csv(result, output_path)

        assert output_path.exists()
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2  # Header + 1 detection
        assert "cat" in lines[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
