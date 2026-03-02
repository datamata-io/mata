"""Unit tests for Image artifact.

Tests image artifact functionality including format conversions, color space handling,
metadata preservation, and file loading.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from mata.core.artifacts.image import Image


class TestImageConstruction:
    """Test image artifact construction from various sources."""

    def test_from_pil_rgb(self):
        """Test creating Image from PIL RGB image."""
        pil_img = PILImage.new("RGB", (640, 480), color=(255, 0, 0))
        img = Image.from_pil(pil_img)

        assert img.width == 640
        assert img.height == 480
        assert img.color_space == "RGB"
        assert isinstance(img.data, PILImage.Image)

    def test_from_pil_grayscale(self):
        """Test creating Image from PIL grayscale image."""
        pil_img = PILImage.new("L", (320, 240), color=128)
        img = Image.from_pil(pil_img)

        assert img.width == 320
        assert img.height == 240
        assert img.color_space == "GRAY"
        assert isinstance(img.data, PILImage.Image)

    def test_from_pil_rgba(self):
        """Test creating Image from PIL RGBA image."""
        pil_img = PILImage.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        img = Image.from_pil(pil_img)

        assert img.width == 100
        assert img.height == 100
        assert img.color_space == "RGBA"

    def test_from_pil_with_metadata(self):
        """Test creating Image from PIL with metadata."""
        pil_img = PILImage.new("RGB", (640, 480))
        img = Image.from_pil(pil_img, timestamp_ms=1000, frame_id="frame_001", source_path="/path/to/image.jpg")

        assert img.timestamp_ms == 1000
        assert img.frame_id == "frame_001"
        assert img.source_path == "/path/to/image.jpg"

    def test_from_numpy_rgb(self):
        """Test creating Image from numpy array (RGB)."""
        array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.from_numpy(array, color_space="RGB")

        assert img.width == 640
        assert img.height == 480
        assert img.color_space == "RGB"
        assert isinstance(img.data, np.ndarray)
        assert img.data.shape == (480, 640, 3)

    def test_from_numpy_bgr(self):
        """Test creating Image from numpy array (BGR)."""
        array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.from_numpy(array, color_space="BGR")

        assert img.color_space == "BGR"

    def test_from_numpy_grayscale(self):
        """Test creating Image from numpy array (grayscale)."""
        array = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        img = Image.from_numpy(array, color_space="GRAY")

        assert img.width == 640
        assert img.height == 480
        assert img.color_space == "GRAY"
        assert img.data.shape == (480, 640)

    def test_from_numpy_float_normalized(self):
        """Test creating Image from float array normalized to [0, 1]."""
        array = np.random.rand(480, 640, 3).astype(np.float32)
        img = Image.from_numpy(array)

        # Should be converted to uint8
        assert img.data.dtype == np.uint8
        assert np.min(img.data) >= 0
        assert np.max(img.data) <= 255

    def test_from_numpy_invalid_shape(self):
        """Test error on invalid numpy array shape."""
        array = np.random.rand(640)  # 1D array

        with pytest.raises(ValueError, match="Invalid array shape"):
            Image.from_numpy(array)

    def test_from_numpy_invalid_channels(self):
        """Test error on invalid number of channels."""
        array = np.random.randint(0, 255, (480, 640, 5), dtype=np.uint8)

        with pytest.raises(ValueError, match="Invalid number of channels"):
            Image.from_numpy(array)

    def test_from_path_jpg(self):
        """Test loading Image from JPG file."""
        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
        # File is now closed, safe to use on Windows

        pil_img = PILImage.new("RGB", (320, 240), color=(128, 128, 128))
        pil_img.save(temp_path)

        try:
            img = Image.from_path(temp_path)

            assert img.width == 320
            assert img.height == 240
            assert img.color_space == "RGB"
            assert img.source_path == str(Path(temp_path).absolute())
        finally:
            os.unlink(temp_path)

    def test_from_path_png(self):
        """Test loading Image from PNG file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
        # File is now closed, safe to use on Windows

        pil_img = PILImage.new("RGB", (100, 100))
        pil_img.save(temp_path)

        try:
            img = Image.from_path(temp_path)
            assert img.width == 100
            assert img.height == 100
        finally:
            os.unlink(temp_path)

    def test_from_path_with_metadata(self):
        """Test loading from path with metadata."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
        # File is now closed, safe to use on Windows

        pil_img = PILImage.new("RGB", (640, 480))
        pil_img.save(temp_path)

        try:
            img = Image.from_path(temp_path, timestamp_ms=2000, frame_id="frame_002")

            assert img.timestamp_ms == 2000
            assert img.frame_id == "frame_002"
        finally:
            os.unlink(temp_path)

    def test_from_path_not_found(self):
        """Test error when file not found."""
        with pytest.raises(FileNotFoundError):
            Image.from_path("/nonexistent/path/image.jpg")

    def test_from_path_invalid_file(self):
        """Test error when file is not a valid image."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="wb") as f:
            temp_path = f.name
            f.write(b"not an image")
        # File is now closed, safe to use on Windows

        try:
            with pytest.raises(ValueError, match="Failed to load image"):
                Image.from_path(temp_path)
        finally:
            os.unlink(temp_path)


class TestImageConversion:
    """Test conversions between image formats."""

    def test_pil_to_pil(self):
        """Test PIL → PIL conversion (should be no-op)."""
        pil_img = PILImage.new("RGB", (640, 480))
        img = Image.from_pil(pil_img)

        result = img.to_pil()
        assert result is pil_img  # Should be same object

    def test_pil_to_numpy(self):
        """Test PIL → numpy conversion."""
        pil_img = PILImage.new("RGB", (640, 480), color=(255, 128, 0))
        img = Image.from_pil(pil_img)

        array = img.to_numpy()
        assert isinstance(array, np.ndarray)
        assert array.shape == (480, 640, 3)
        assert array.dtype == np.uint8
        # Check color (note: PIL is RGB)
        assert array[0, 0, 0] == 255  # R
        assert array[0, 0, 1] == 128  # G
        assert array[0, 0, 2] == 0  # B

    def test_pil_to_tensor(self):
        """Test PIL → tensor conversion."""
        pytest.importorskip("torch")  # Skip if torch not installed

        pil_img = PILImage.new("RGB", (640, 480))
        img = Image.from_pil(pil_img)

        tensor = img.to_tensor()
        import torch

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 480, 640)  # CHW format
        assert tensor.dtype == torch.float32

    def test_numpy_to_numpy(self):
        """Test numpy → numpy conversion (should be no-op)."""
        array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.from_numpy(array)

        result = img.to_numpy()
        assert result is array  # Should be same object

    def test_numpy_to_pil(self):
        """Test numpy → PIL conversion."""
        array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.from_numpy(array)

        pil_img = img.to_pil()
        assert isinstance(pil_img, PILImage.Image)
        assert pil_img.size == (640, 480)
        assert pil_img.mode == "RGB"

    def test_numpy_to_tensor(self):
        """Test numpy → tensor conversion."""
        pytest.importorskip("torch")

        array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.from_numpy(array)

        tensor = img.to_tensor()
        import torch

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 480, 640)

    def test_numpy_dtype_conversion(self):
        """Test numpy dtype conversion."""
        array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.from_numpy(array)

        # Convert to float32
        float_array = img.to_numpy(dtype=np.float32)
        assert float_array.dtype == np.float32
        assert np.min(float_array) >= 0.0
        assert np.max(float_array) <= 1.0  # Should be normalized

    def test_round_trip_pil_numpy_pil(self):
        """Test round-trip conversion: PIL → numpy → PIL."""
        pil_img = PILImage.new("RGB", (100, 100), color=(128, 64, 32))
        img = Image.from_pil(pil_img)

        # Convert to numpy and back
        array = img.to_numpy()
        img2 = Image.from_numpy(array)
        pil_img2 = img2.to_pil()

        # Should preserve data
        assert pil_img2.size == pil_img.size
        assert pil_img2.mode == pil_img.mode

        # Check pixel values
        arr1 = np.array(pil_img)
        arr2 = np.array(pil_img2)
        np.testing.assert_array_equal(arr1, arr2)

    def test_round_trip_numpy_pil_numpy(self):
        """Test round-trip conversion: numpy → PIL → numpy."""
        array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.from_numpy(array)

        pil_img = img.to_pil()
        img2 = Image.from_pil(pil_img)
        array2 = img2.to_numpy()

        np.testing.assert_array_equal(array, array2)

    def test_grayscale_pil_to_numpy(self):
        """Test grayscale PIL → numpy conversion."""
        pil_img = PILImage.new("L", (320, 240), color=128)
        img = Image.from_pil(pil_img)

        array = img.to_numpy()
        assert array.shape == (240, 320)  # 2D array
        assert array.dtype == np.uint8

    def test_grayscale_numpy_to_pil(self):
        """Test grayscale numpy → PIL conversion."""
        array = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        img = Image.from_numpy(array, color_space="GRAY")

        pil_img = img.to_pil()
        assert pil_img.mode == "L"
        assert pil_img.size == (320, 240)


class TestColorSpaceHandling:
    """Test color space conversions."""

    def test_bgr_to_numpy(self):
        """Test BGR image to numpy conversion."""
        # Create RGB array, mark as BGR
        array_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        array_rgb[:, :, 0] = 255  # Set red channel

        img = Image.from_numpy(array_rgb, color_space="BGR")

        # When converting to numpy with BGR, should get original
        array_out = img.to_numpy()
        np.testing.assert_array_equal(array_out, array_rgb)

    def test_bgr_to_pil(self):
        """Test BGR image to PIL conversion (should convert to RGB)."""
        # Create BGR array (blue channel = 255)
        array_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        array_bgr[:, :, 0] = 255  # Blue in BGR

        img = Image.from_numpy(array_bgr, color_space="BGR")
        pil_img = img.to_pil()

        # PIL should have RGB, so blue should be in last channel
        array_rgb = np.array(pil_img)
        assert array_rgb[0, 0, 2] == 255  # Blue in RGB
        assert array_rgb[0, 0, 0] == 0  # Red in RGB

    def test_rgb_from_path(self):
        """Test loading RGB image from path."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
        # File is now closed, safe to use on Windows

        pil_img = PILImage.new("RGB", (100, 100))
        pil_img.save(temp_path)

        try:
            img = Image.from_path(temp_path, color_space="RGB")
            assert img.color_space == "RGB"
        finally:
            os.unlink(temp_path)

    def test_gray_from_path(self):
        """Test loading grayscale image from path."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
        # File is now closed, safe to use on Windows

        pil_img = PILImage.new("L", (100, 100))
        pil_img.save(temp_path)

        try:
            img = Image.from_path(temp_path, color_space="GRAY")
            assert img.color_space == "GRAY"
        finally:
            os.unlink(temp_path)

    def test_gray_from_path_rgb_source(self):
        """Test loading grayscale from RGB source (conversion)."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
        # File is now closed, safe to use on Windows

        # Save as RGB
        pil_img = PILImage.new("RGB", (100, 100), color=(128, 128, 128))
        pil_img.save(temp_path)

        try:
            # Load as GRAY (should convert)
            img = Image.from_path(temp_path, color_space="GRAY")
            assert img.color_space == "GRAY"
        finally:
            os.unlink(temp_path)

    def test_bgr_from_path(self):
        """Test loading BGR from path."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
        # File is now closed, safe to use on Windows

        pil_img = PILImage.new("RGB", (100, 100))
        pil_img.save(temp_path)

        try:
            img = Image.from_path(temp_path, color_space="BGR")
            assert img.color_space == "BGR"
        finally:
            os.unlink(temp_path)

    def test_bgr_from_path_gray_source(self):
        """Test loading BGR from grayscale source (conversion)."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
        # File is now closed, safe to use on Windows

        # Save as grayscale
        pil_img = PILImage.new("L", (100, 100))
        pil_img.save(temp_path)

        try:
            # Load as BGR (should convert to RGB first)
            img = Image.from_path(temp_path, color_space="BGR")
            assert img.color_space == "BGR"
        finally:
            os.unlink(temp_path)


class TestMetadataPreservation:
    """Test metadata preservation through conversions."""

    def test_metadata_in_constructor(self):
        """Test metadata is stored correctly."""
        pil_img = PILImage.new("RGB", (640, 480))
        img = Image.from_pil(pil_img, timestamp_ms=5000, frame_id="frame_123", source_path="/path/to/source.jpg")

        assert img.timestamp_ms == 5000
        assert img.frame_id == "frame_123"
        assert img.source_path == "/path/to/source.jpg"

    def test_metadata_preserved_after_conversion(self):
        """Test metadata is preserved through format conversions."""
        array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.from_numpy(array, timestamp_ms=1000, frame_id="test")

        # Convert to PIL
        _ = img.to_pil()

        # Metadata should still be there
        assert img.timestamp_ms == 1000
        assert img.frame_id == "test"

    def test_serialization_preserves_metadata(self):
        """Test metadata preserved through serialization."""
        array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.from_numpy(array, timestamp_ms=3000, frame_id="serialize_test", source_path="/test/path.jpg")

        # Serialize
        data = img.to_dict()

        # Deserialize
        img2 = Image.from_dict(data)

        assert img2.timestamp_ms == 3000
        assert img2.frame_id == "serialize_test"
        assert img2.source_path == "/test/path.jpg"


class TestValidation:
    """Test image artifact validation."""

    def test_invalid_color_space(self):
        """Test error on invalid color space."""
        pil_img = PILImage.new("RGB", (640, 480))

        with pytest.raises(ValueError, match="Invalid color_space"):
            Image.from_pil(pil_img, color_space="INVALID")

    def test_invalid_dimensions(self):
        """Test error on invalid dimensions."""
        array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Invalid dimensions"):
            Image(data=array, width=0, height=480, color_space="RGB")

    def test_validate_dimension_mismatch_pil(self):
        """Test validation catches dimension mismatch for PIL."""
        pil_img = PILImage.new("RGB", (640, 480))

        # Create with wrong dimensions
        img = Image(data=pil_img, width=320, height=240, color_space="RGB")  # Wrong!

        with pytest.raises(ValueError, match="Dimension mismatch"):
            img.validate()

    def test_validate_dimension_mismatch_numpy(self):
        """Test validation catches dimension mismatch for numpy."""
        array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        img = Image(data=array, width=100, height=100, color_space="RGB")  # Wrong!

        with pytest.raises(ValueError, match="Dimension mismatch"):
            img.validate()

    def test_validate_invalid_data_type(self):
        """Test validation catches invalid data type."""
        img = Image(data="not an image", width=640, height=480, color_space="RGB")  # type: ignore

        with pytest.raises(ValueError, match="Invalid data type"):
            img.validate()


class TestSerialization:
    """Test image serialization and deserialization."""

    def test_to_dict(self):
        """Test image to dictionary conversion."""
        array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.from_numpy(array, timestamp_ms=1000)

        data = img.to_dict()

        assert "data" in data
        assert data["width"] == 100
        assert data["height"] == 100
        assert data["color_space"] == "RGB"
        assert data["timestamp_ms"] == 1000

    def test_from_dict(self):
        """Test image from dictionary construction."""
        data = {
            "data": np.random.randint(0, 255, (100, 100, 3)).tolist(),
            "width": 100,
            "height": 100,
            "color_space": "RGB",
            "timestamp_ms": 2000,
            "frame_id": "test",
        }

        img = Image.from_dict(data)

        assert img.width == 100
        assert img.height == 100
        assert img.timestamp_ms == 2000
        assert img.frame_id == "test"

    def test_round_trip_serialization(self):
        """Test serialization round-trip."""
        array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        img = Image.from_numpy(array, frame_id="round_trip")

        # Serialize
        data = img.to_dict()

        # Deserialize
        img2 = Image.from_dict(data)

        # Check equality
        assert img2.width == img.width
        assert img2.height == img.height
        assert img2.color_space == img.color_space
        assert img2.frame_id == img.frame_id

        # Check data
        np.testing.assert_array_equal(img2.to_numpy(), img.to_numpy())


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_small_image(self):
        """Test very small image (1x1)."""
        array = np.array([[[255, 128, 0]]], dtype=np.uint8)
        img = Image.from_numpy(array)

        assert img.width == 1
        assert img.height == 1

    def test_large_dimensions(self):
        """Test image with large dimensions."""
        # Don't actually create large array, just test metadata
        pil_img = PILImage.new("RGB", (4096, 2160))
        img = Image.from_pil(pil_img)

        assert img.width == 4096
        assert img.height == 2160

    def test_repr(self):
        """Test string representation."""
        pil_img = PILImage.new("RGB", (640, 480))
        img = Image.from_pil(pil_img)

        repr_str = repr(img)
        assert "640x480" in repr_str
        assert "RGB" in repr_str

    def test_to_tensor_without_torch(self, monkeypatch):
        """Test to_tensor error when torch not available."""
        # Mock torch import failure
        import sys

        monkeypatch.setitem(sys.modules, "torch", None)

        pil_img = PILImage.new("RGB", (100, 100))
        img = Image.from_pil(pil_img)

        with pytest.raises(ImportError, match="PyTorch is required"):
            img.to_tensor()

    def test_rgba_image(self):
        """Test RGBA image (4 channels)."""
        array = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        img = Image.from_numpy(array, color_space="RGBA")

        assert img.color_space == "RGBA"
        assert img.width == 100
        assert img.height == 100

    def test_rgba_pil_to_numpy(self):
        """Test RGBA PIL to numpy conversion."""
        pil_img = PILImage.new("RGBA", (100, 100), color=(255, 128, 0, 200))
        img = Image.from_pil(pil_img)

        array = img.to_numpy()
        assert array.shape == (100, 100, 4)
        assert array[0, 0, 3] == 200  # Alpha channel

    def test_from_pil_auto_convert(self):
        """Test PIL auto-conversion of unsupported modes."""
        # Create image in mode that's not RGB/L/RGBA
        pil_img = PILImage.new("P", (100, 100))  # Palette mode
        img = Image.from_pil(pil_img)

        # Should auto-convert to RGB
        assert img.color_space == "RGB"

    def test_grayscale_to_tensor(self):
        """Test grayscale image to tensor conversion."""
        pytest.importorskip("torch")

        array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img = Image.from_numpy(array, color_space="GRAY")

        tensor = img.to_tensor()
        import torch

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 100, 100)  # 1 channel for grayscale

    def test_from_numpy_int16_dtype(self):
        """Test creating image from non-uint8 dtype that needs conversion."""
        array = np.random.randint(0, 255, (100, 100, 3), dtype=np.int16)
        img = Image.from_numpy(array)

        # Should be converted to uint8
        assert img.data.dtype == np.uint8

    def test_validate_with_tensor(self):
        """Test validation with torch tensor data."""
        pytest.importorskip("torch")
        import torch

        tensor = torch.rand(3, 100, 100)
        img = Image(data=tensor, width=100, height=100, color_space="RGB")

        # Should not raise
        img.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
