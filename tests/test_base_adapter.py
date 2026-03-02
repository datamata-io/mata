"""Tests for BaseAdapter URL image loading functionality.

Tests the URL image loading feature added in v1.5.4 (Task 2.4).
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, Mock, patch
from urllib.error import URLError

import numpy as np
import pytest
from PIL import Image

from mata.adapters.base import BaseAdapter
from mata.core.exceptions import InvalidInputError


class ConcreteAdapter(BaseAdapter):
    """Concrete implementation of BaseAdapter for testing.

    BaseAdapter is abstract, so we need a concrete subclass to test
    the _load_image() and _load_images() methods.
    """

    def predict(self, *args, **kwargs):
        """Dummy predict implementation."""
        pass

    def info(self) -> dict:
        """Dummy info implementation."""
        return {"type": "test"}


@pytest.fixture
def adapter():
    """Create a concrete adapter instance for testing."""
    return ConcreteAdapter()


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    return Image.new("RGB", (100, 100), color="red")


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for mocking URL responses."""
    img = Image.new("RGB", (100, 100), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


class TestLoadImageURL:
    """Tests for URL image loading in _load_image()."""

    def test_load_image_from_http_url(self, adapter, sample_image_bytes):
        """Test loading image from HTTP URL."""
        url = "http://example.com/image.jpg"

        # Mock urllib.request.urlopen
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            mock_response.read = Mock(return_value=sample_image_bytes)
            mock_urlopen.return_value = mock_response

            pil_image, image_path = adapter._load_image(url)

            # Verify urlopen was called with correct URL and timeout
            mock_urlopen.assert_called_once_with(url, timeout=30)

            # Verify returned values
            assert isinstance(pil_image, Image.Image)
            assert pil_image.mode == "RGB"
            assert pil_image.size == (100, 100)
            assert image_path == url

    def test_load_image_from_https_url(self, adapter, sample_image_bytes):
        """Test loading image from HTTPS URL."""
        url = "https://example.com/secure/image.png"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            mock_response.read = Mock(return_value=sample_image_bytes)
            mock_urlopen.return_value = mock_response

            pil_image, image_path = adapter._load_image(url)

            mock_urlopen.assert_called_once_with(url, timeout=30)
            assert isinstance(pil_image, Image.Image)
            assert pil_image.mode == "RGB"
            assert image_path == url

    def test_load_image_url_with_30_second_timeout(self, adapter, sample_image_bytes):
        """Test that URL loading uses 30-second timeout."""
        url = "https://example.com/image.jpg"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            mock_response.read = Mock(return_value=sample_image_bytes)
            mock_urlopen.return_value = mock_response

            adapter._load_image(url)

            # Verify timeout parameter
            _, kwargs = mock_urlopen.call_args
            assert kwargs.get("timeout") == 30

    def test_load_image_url_network_failure(self, adapter):
        """Test that network failures raise InvalidInputError with clear message."""
        url = "http://example.com/nonexistent.jpg"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("Network error")

            with pytest.raises(InvalidInputError) as exc_info:
                adapter._load_image(url)

            assert "Failed to load image from URL" in str(exc_info.value)
            assert "Network error" in str(exc_info.value)

    def test_load_image_url_timeout(self, adapter):
        """Test that timeout errors are handled gracefully."""
        url = "http://slow-server.com/image.jpg"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("Connection timeout")

            with pytest.raises(InvalidInputError) as exc_info:
                adapter._load_image(url)

            assert "Failed to load image from URL" in str(exc_info.value)

    def test_load_image_url_invalid_image_data(self, adapter):
        """Test that invalid image data from URL raises InvalidInputError."""
        url = "http://example.com/not-an-image.txt"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            mock_response.read = Mock(return_value=b"This is not an image")
            mock_urlopen.return_value = mock_response

            with pytest.raises(InvalidInputError) as exc_info:
                adapter._load_image(url)

            assert "Failed to load image from URL" in str(exc_info.value)

    def test_load_image_url_converts_to_rgb(self, adapter):
        """Test that URL-loaded images are converted to RGB mode."""
        url = "http://example.com/grayscale.jpg"

        # Create grayscale image
        gray_img = Image.new("L", (100, 100), color=128)
        buffer = io.BytesIO()
        gray_img.save(buffer, format="PNG")
        buffer.seek(0)

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            mock_response.read = Mock(return_value=buffer.getvalue())
            mock_urlopen.return_value = mock_response

            pil_image, _ = adapter._load_image(url)

            assert pil_image.mode == "RGB"


class TestLoadImageBackwardCompatibility:
    """Tests to ensure URL support doesn't break existing functionality."""

    def test_load_image_file_path_still_works(self, adapter, tmp_path):
        """Test that file path loading still works after URL support."""
        # Create temporary test image
        img_path = tmp_path / "test.jpg"
        test_img = Image.new("RGB", (50, 50), color="green")
        test_img.save(img_path)

        pil_image, image_path = adapter._load_image(str(img_path))

        assert isinstance(pil_image, Image.Image)
        assert pil_image.mode == "RGB"
        assert pil_image.size == (50, 50)
        assert image_path == str(img_path)

    def test_load_image_path_object_still_works(self, adapter, tmp_path):
        """Test that Path object loading still works."""
        img_path = tmp_path / "test.jpg"
        test_img = Image.new("RGB", (50, 50), color="green")
        test_img.save(img_path)

        pil_image, image_path = adapter._load_image(img_path)

        assert isinstance(pil_image, Image.Image)
        assert image_path == str(img_path)

    def test_load_image_pil_image_still_works(self, adapter, sample_image):
        """Test that PIL Image input still works."""
        pil_image, image_path = adapter._load_image(sample_image)

        assert isinstance(pil_image, Image.Image)
        assert pil_image.mode == "RGB"
        assert image_path is None

    def test_load_image_numpy_array_still_works(self, adapter):
        """Test that numpy array input still works."""
        np_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        pil_image, image_path = adapter._load_image(np_array)

        assert isinstance(pil_image, Image.Image)
        assert pil_image.mode == "RGB"
        assert pil_image.size == (100, 100)
        assert image_path is None

    def test_load_image_unsupported_type_still_raises(self, adapter):
        """Test that unsupported types still raise InvalidInputError."""
        with pytest.raises(InvalidInputError) as exc_info:
            adapter._load_image(12345)

        assert "Unsupported image type" in str(exc_info.value)


class TestLoadImagesBasic:
    """Tests for _load_images() basic functionality (Task 3.3)."""

    def test_load_images_list_of_paths(self, adapter, tmp_path, sample_image):
        """Test that _load_images() returns list of (PIL, path) tuples."""
        # Create multiple test images
        img1_path = tmp_path / "img1.jpg"
        img2_path = tmp_path / "img2.jpg"
        img3_path = tmp_path / "img3.jpg"

        sample_image.save(img1_path)
        sample_image.save(img2_path)
        sample_image.save(img3_path)

        results = adapter._load_images([str(img1_path), str(img2_path), str(img3_path)])

        assert len(results) == 3
        for i, (pil_img, path) in enumerate(results):
            assert isinstance(pil_img, Image.Image)
            assert pil_img.mode == "RGB"
            assert path == str(tmp_path / f"img{i+1}.jpg")

    def test_load_images_empty_list_raises(self, adapter):
        """Test that empty list raises InvalidInputError."""
        with pytest.raises(InvalidInputError) as exc_info:
            adapter._load_images([])

        assert "images list cannot be empty" in str(exc_info.value)

    def test_load_images_mixed_types(self, adapter, tmp_path, sample_image):
        """Test that mix of str + PIL images works correctly."""
        # Create one file
        img_path = tmp_path / "file.jpg"
        sample_image.save(img_path)

        # Create a PIL image
        pil_img = Image.new("RGB", (50, 50), color="blue")

        # Create a numpy array
        np_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        results = adapter._load_images([str(img_path), pil_img, np_array])

        assert len(results) == 3
        # File path should have path
        assert isinstance(results[0][0], Image.Image)
        assert results[0][1] == str(img_path)
        # PIL image should have no path
        assert isinstance(results[1][0], Image.Image)
        assert results[1][1] is None
        # Numpy array should have no path
        assert isinstance(results[2][0], Image.Image)
        assert results[2][1] is None

    def test_load_images_invalid_item_error_includes_index(self, adapter, tmp_path, sample_image):
        """Test that error message includes index when loading fails."""
        # Create one valid image
        img1_path = tmp_path / "img1.jpg"
        sample_image.save(img1_path)

        # Second item is invalid (unsupported type)
        invalid_input = 12345

        with pytest.raises(InvalidInputError) as exc_info:
            adapter._load_images([str(img1_path), invalid_input])

        # Error should mention index 1 (second item)
        assert "index 1" in str(exc_info.value)
        assert "Failed to load image" in str(exc_info.value)


class TestLoadImagesWithURL:
    """Tests for URL support in _load_images() method."""

    def test_load_images_with_url_in_list(self, adapter, sample_image_bytes, tmp_path):
        """Test that _load_images() works with URLs in the list."""
        # Create a local file
        img_path = tmp_path / "local.jpg"
        test_img = Image.new("RGB", (50, 50), color="yellow")
        test_img.save(img_path)

        url = "http://example.com/remote.jpg"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            mock_response.read = Mock(return_value=sample_image_bytes)
            mock_urlopen.return_value = mock_response

            results = adapter._load_images([str(img_path), url])

            assert len(results) == 2
            # First image from file
            assert isinstance(results[0][0], Image.Image)
            assert results[0][1] == str(img_path)
            # Second image from URL
            assert isinstance(results[1][0], Image.Image)
            assert results[1][1] == url

    def test_load_images_multiple_urls(self, adapter, sample_image_bytes):
        """Test loading multiple images from URLs."""
        urls = [
            "http://example.com/image1.jpg",
            "https://example.com/image2.png",
            "http://example.com/image3.jpg",
        ]

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            mock_response.read = Mock(return_value=sample_image_bytes)
            mock_urlopen.return_value = mock_response

            results = adapter._load_images(urls)

            assert len(results) == 3
            assert mock_urlopen.call_count == 3
            for i, (pil_img, path) in enumerate(results):
                assert isinstance(pil_img, Image.Image)
                assert path == urls[i]

    def test_load_images_url_failure_includes_index(self, adapter):
        """Test that URL loading failures in list include index in error."""
        urls = [
            "http://example.com/good.jpg",
            "http://example.com/bad.jpg",  # This will fail
        ]

        call_count = 0

        def mock_urlopen_side_effect(url, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First URL succeeds
                img = Image.new("RGB", (100, 100), color="blue")
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)
                mock_resp = MagicMock()
                mock_resp.__enter__ = Mock(return_value=mock_resp)
                mock_resp.__exit__ = Mock(return_value=False)
                mock_resp.read = Mock(return_value=buffer.getvalue())
                return mock_resp
            else:
                # Second URL fails
                raise URLError("Network error")

        with patch("urllib.request.urlopen", side_effect=mock_urlopen_side_effect):
            with pytest.raises(InvalidInputError) as exc_info:
                adapter._load_images(urls)

            # Error should mention index 1 (second image)
            assert "index 1" in str(exc_info.value)
            assert "Failed to load image from URL" in str(exc_info.value)


class TestEdgeCases:
    """Edge case tests for URL handling."""

    def test_load_image_ftp_url_not_supported(self, adapter):
        """Test that FTP URLs are treated as file paths (not URLs)."""
        ftp_url = "ftp://example.com/image.jpg"

        # Should try to open as file path and fail
        with pytest.raises(InvalidInputError) as exc_info:
            adapter._load_image(ftp_url)

        # Should not attempt URL download
        assert "Failed to load image" in str(exc_info.value)

    def test_load_image_http_in_middle_of_string(self, adapter, tmp_path):
        """Test that http in middle of string is not treated as URL."""
        # Create a file with "http" in the name
        img_path = tmp_path / "my_http_image.jpg"
        test_img = Image.new("RGB", (50, 50), color="purple")
        test_img.save(img_path)

        # Should load as file path, not URL
        pil_image, image_path = adapter._load_image(str(img_path))

        assert isinstance(pil_image, Image.Image)
        assert image_path == str(img_path)

    def test_load_image_empty_url_string(self, adapter):
        """Test that empty string raises appropriate error."""
        with pytest.raises(InvalidInputError):
            adapter._load_image("")

    def test_load_image_url_with_query_params(self, adapter, sample_image_bytes):
        """Test loading image from URL with query parameters."""
        url = "https://example.com/image.jpg?size=large&format=png"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            mock_response.read = Mock(return_value=sample_image_bytes)
            mock_urlopen.return_value = mock_response

            pil_image, image_path = adapter._load_image(url)

            assert isinstance(pil_image, Image.Image)
            assert image_path == url
            mock_urlopen.assert_called_once_with(url, timeout=30)
