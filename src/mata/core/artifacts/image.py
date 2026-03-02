"""Image artifact for graph system.

Provides a versatile image container supporting PIL, numpy, and torch tensor formats
with lazy conversion, metadata preservation, and color space handling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

import numpy as np
from PIL import Image as PILImage

from mata.core.artifacts.base import Artifact


@dataclass(frozen=True)
class Image(Artifact):
    """Image artifact supporting multiple formats with lazy conversion.

    The Image artifact is the primary data container for image data in the graph system.
    It supports three internal formats:
    - PIL.Image.Image (for compatibility with PIL/Pillow operations)
    - np.ndarray (for numpy-based processing)
    - torch.Tensor (for PyTorch models)

    Conversions between formats are lazy - they only occur when explicitly requested
    via to_pil(), to_numpy(), or to_tensor() methods. This avoids unnecessary copies
    and memory overhead.

    Color space handling:
    - RGB: Standard RGB format (PIL default, most models)
    - BGR: OpenCV format (used by some vision libraries)
    - GRAY: Grayscale images

    Metadata fields:
    - timestamp_ms: Unix timestamp in milliseconds (for video frames)
    - frame_id: String identifier for frame (for video sequences)
    - source_path: Original file path (if loaded from disk)

    Example:
        ```python
        # Load from file
        img = Image.from_path("photo.jpg")

        # Load from PIL
        pil_img = PILImage.open("photo.jpg")
        img = Image.from_pil(pil_img)

        # Load from numpy
        array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.from_numpy(array)

        # Convert between formats (lazy)
        pil_img = img.to_pil()
        array = img.to_numpy()
        tensor = img.to_tensor()  # requires torch

        # Access metadata
        print(f"Image size: {img.width}x{img.height}")
        print(f"Color space: {img.color_space}")
        ```
    """

    data: PILImage.Image | np.ndarray | torch.Tensor
    width: int
    height: int
    color_space: str = "RGB"  # RGB, BGR, GRAY
    timestamp_ms: int | None = None
    frame_id: str | None = None
    source_path: str | None = None

    def __post_init__(self):
        """Validate image data on construction."""
        # Validate color space
        valid_color_spaces = ["RGB", "BGR", "GRAY", "L", "RGBA"]
        if self.color_space not in valid_color_spaces:
            raise ValueError(f"Invalid color_space '{self.color_space}'. " f"Valid values: {valid_color_spaces}")

        # Validate dimensions
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid dimensions: {self.width}x{self.height}. " f"Width and height must be positive.")

    @classmethod
    def from_path(
        cls, path: str, color_space: str = "RGB", timestamp_ms: int | None = None, frame_id: str | None = None
    ) -> Image:
        """Load image from file path.

        Args:
            path: Path to image file (supports common formats: jpg, png, bmp, etc.)
            color_space: Color space of the image ("RGB", "BGR", "GRAY")
            timestamp_ms: Optional timestamp in milliseconds
            frame_id: Optional frame identifier

        Returns:
            Image artifact

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file cannot be loaded as an image

        Example:
            ```python
            img = Image.from_path("photo.jpg")
            img = Image.from_path("frame.png", frame_id="frame_001")
            ```
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

        try:
            pil_img = PILImage.open(path)
            # Force load to close file handle (important for Windows)
            pil_img.load()

            # Convert to specified color space
            if color_space == "RGB" and pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            elif color_space == "GRAY" and pil_img.mode != "L":
                pil_img = pil_img.convert("L")
            elif color_space == "BGR":
                # PIL doesn't have BGR mode, convert via numpy
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                # Note: We'll store as RGB and handle BGR conversion in to_numpy()

            width, height = pil_img.size

            return cls(
                data=pil_img,
                width=width,
                height=height,
                color_space=color_space,
                timestamp_ms=timestamp_ms,
                frame_id=frame_id,
                source_path=str(Path(path).absolute()),
            )
        except Exception as e:
            raise ValueError(f"Failed to load image from {path}: {e}")

    @classmethod
    def from_pil(
        cls,
        pil_image: PILImage.Image,
        color_space: str | None = None,
        timestamp_ms: int | None = None,
        frame_id: str | None = None,
        source_path: str | None = None,
    ) -> Image:
        """Create Image artifact from PIL Image.

        Args:
            pil_image: PIL Image object
            color_space: Color space override (auto-detected if None)
            timestamp_ms: Optional timestamp in milliseconds
            frame_id: Optional frame identifier
            source_path: Optional source file path

        Returns:
            Image artifact

        Example:
            ```python
            pil_img = PILImage.open("photo.jpg")
            img = Image.from_pil(pil_img)
            ```
        """
        width, height = pil_image.size

        # Auto-detect color space from PIL mode if not specified
        if color_space is None:
            if pil_image.mode == "RGB":
                color_space = "RGB"
            elif pil_image.mode == "L":
                color_space = "GRAY"
            elif pil_image.mode == "RGBA":
                color_space = "RGBA"
            else:
                # Convert to RGB for unsupported modes
                pil_image = pil_image.convert("RGB")
                color_space = "RGB"

        return cls(
            data=pil_image,
            width=width,
            height=height,
            color_space=color_space,
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            source_path=source_path,
        )

    @classmethod
    def from_numpy(
        cls,
        array: np.ndarray,
        color_space: str = "RGB",
        timestamp_ms: int | None = None,
        frame_id: str | None = None,
        source_path: str | None = None,
    ) -> Image:
        """Create Image artifact from numpy array.

        Args:
            array: Numpy array with shape (H, W) for grayscale or (H, W, C) for color
            color_space: Color space of the array ("RGB", "BGR", "GRAY")
            timestamp_ms: Optional timestamp in milliseconds
            frame_id: Optional frame identifier
            source_path: Optional source file path

        Returns:
            Image artifact

        Raises:
            ValueError: If array has invalid shape or dtype

        Example:
            ```python
            array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img = Image.from_numpy(array, color_space="RGB")
            ```
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(array)}")

        # Validate shape
        if array.ndim == 2:
            # Grayscale (H, W)
            height, width = array.shape
            if color_space not in ["GRAY", "L"]:
                color_space = "GRAY"
        elif array.ndim == 3:
            # Color (H, W, C)
            height, width, channels = array.shape
            if channels not in [3, 4]:
                raise ValueError(f"Invalid number of channels: {channels}. Expected 3 (RGB/BGR) or 4 (RGBA)")
            if channels == 4 and color_space not in ["RGBA"]:
                color_space = "RGBA"
        else:
            raise ValueError(
                f"Invalid array shape: {array.shape}. " f"Expected (H, W) for grayscale or (H, W, C) for color"
            )

        # Ensure uint8 dtype
        if array.dtype != np.uint8:
            if np.max(array) <= 1.0:
                # Assume normalized [0, 1], scale to [0, 255]
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)

        return cls(
            data=array,
            width=width,
            height=height,
            color_space=color_space,
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            source_path=source_path,
        )

    def to_pil(self) -> PILImage.Image:
        """Convert image to PIL Image format.

        Returns:
            PIL Image object

        Note:
            Conversion is lazy - if data is already PIL, returns it directly.
            For BGR color space, converts to RGB for PIL compatibility.
        """
        if isinstance(self.data, PILImage.Image):
            return self.data

        # Convert from numpy or torch tensor
        if hasattr(self.data, "cpu"):
            # Torch tensor - convert to numpy first
            array = self.data.cpu().numpy()
        else:
            array = self.data

        # Handle color space conversion
        if self.color_space == "BGR":
            # Convert BGR to RGB for PIL
            array = array[..., ::-1]  # Reverse channel order
        elif self.color_space in ["GRAY", "L"] and array.ndim == 2:
            # Grayscale - PIL expects 2D array
            return PILImage.fromarray(array, mode="L")

        # Create PIL image
        if self.color_space == "RGBA":
            return PILImage.fromarray(array, mode="RGBA")
        else:
            return PILImage.fromarray(array, mode="RGB")

    def to_numpy(self, dtype: type = np.uint8) -> np.ndarray:
        """Convert image to numpy array format.

        Args:
            dtype: Desired numpy dtype (default: np.uint8)

        Returns:
            Numpy array with shape (H, W) for grayscale or (H, W, C) for color

        Note:
            Conversion is lazy - if data is already numpy, returns it directly
            (unless dtype conversion is needed).
            When converting uint8 -> float32, values are normalized to [0, 1].
        """
        if isinstance(self.data, np.ndarray):
            if self.data.dtype == dtype:
                return self.data
            # Handle dtype conversion with normalization
            if dtype == np.float32 and self.data.dtype == np.uint8:
                # Normalize to [0, 1]
                return self.data.astype(np.float32) / 255.0
            else:
                return self.data.astype(dtype)

        # Convert from PIL or torch tensor
        if hasattr(self.data, "cpu"):
            # Torch tensor
            array = self.data.cpu().numpy()
        else:
            # PIL Image
            array = np.array(self.data)

        # Handle color space
        if self.color_space == "BGR" and array.ndim == 3:
            # PIL gives us RGB, convert to BGR
            array = array[..., ::-1]  # Reverse channel order

        if array.dtype != dtype:
            if dtype == np.float32 and array.dtype == np.uint8:
                # Normalize to [0, 1]
                array = array.astype(np.float32) / 255.0
            else:
                array = array.astype(dtype)

        return array

    def to_tensor(self) -> torch.Tensor:
        """Convert image to PyTorch tensor format.

        Returns:
            PyTorch tensor with shape (C, H, W) in float32 format, normalized to [0, 1]

        Raises:
            ImportError: If PyTorch is not installed

        Note:
            Conversion is lazy - if data is already a tensor, returns it directly.
            Output tensor is in (C, H, W) format (channels first), which is
            standard for PyTorch models.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for to_tensor(). " "Install with: pip install torch")

        if isinstance(self.data, torch.Tensor):
            return self.data

        # Convert to numpy first
        array = self.to_numpy(dtype=np.float32)

        # Handle grayscale
        if array.ndim == 2:
            # (H, W) -> (1, H, W)
            array = array[np.newaxis, ...]
        else:
            # (H, W, C) -> (C, H, W)
            array = np.transpose(array, (2, 0, 1))

        # Convert to tensor
        tensor = torch.from_numpy(array)

        return tensor

    def to_dict(self) -> dict[str, Any]:
        """Convert image to dictionary representation.

        Returns:
            Dictionary with image metadata and numpy array representation

        Note:
            The image data is converted to numpy array for serialization.
            For large images, consider saving to file and storing only the path.
        """
        return {
            "data": self.to_numpy().tolist(),  # Convert to list for JSON compatibility
            "width": self.width,
            "height": self.height,
            "color_space": self.color_space,
            "timestamp_ms": self.timestamp_ms,
            "frame_id": self.frame_id,
            "source_path": self.source_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Image:
        """Construct image from dictionary representation.

        Args:
            data: Dictionary containing image data and metadata

        Returns:
            Image artifact

        Example:
            ```python
            # Serialize
            img_dict = img.to_dict()

            # Deserialize
            restored = Image.from_dict(img_dict)
            ```
        """
        array = np.array(data["data"], dtype=np.uint8)

        return cls.from_numpy(
            array=array,
            color_space=data.get("color_space", "RGB"),
            timestamp_ms=data.get("timestamp_ms"),
            frame_id=data.get("frame_id"),
            source_path=data.get("source_path"),
        )

    def validate(self) -> None:
        """Validate image data.

        Raises:
            ValueError: If image data is invalid
        """
        # Check data type
        valid_types = (PILImage.Image, np.ndarray)

        # Check for torch.Tensor if torch is available
        try:
            import torch

            valid_types = (PILImage.Image, np.ndarray, torch.Tensor)
        except ImportError:
            pass

        if not isinstance(self.data, valid_types):
            raise ValueError(
                f"Invalid data type: {type(self.data)}. " f"Expected PIL Image, numpy array, or torch Tensor"
            )

        # Validate dimensions match data
        if isinstance(self.data, PILImage.Image):
            width, height = self.data.size
            if width != self.width or height != self.height:
                raise ValueError(
                    f"Dimension mismatch: data is {width}x{height}, " f"but metadata says {self.width}x{self.height}"
                )
        elif isinstance(self.data, np.ndarray):
            if self.data.ndim == 2:
                h, w = self.data.shape
            else:
                h, w = self.data.shape[:2]
            if w != self.width or h != self.height:
                raise ValueError(
                    f"Dimension mismatch: data is {w}x{h}, " f"but metadata says {self.width}x{self.height}"
                )

    def __repr__(self) -> str:
        """String representation of Image artifact."""
        data_type = type(self.data).__name__
        return f"Image(size={self.width}x{self.height}, " f"color_space={self.color_space}, " f"data_type={data_type})"
