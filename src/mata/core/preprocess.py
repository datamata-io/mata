"""Shared preprocessing geometry for detection adapters."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

#: Pad value used by the common YOLO export pipeline (RGB).
PAD_VALUE: tuple[int, int, int] = (114, 114, 114)


@dataclass(frozen=True)
class LetterboxTransform:
    """Geometry of a letterbox resize, used to map model boxes back."""

    ratio: float
    pad_left: int
    pad_top: int
    input_width: int
    input_height: int
    orig_width: int
    orig_height: int


def letterbox(
    image: Image.Image | np.ndarray,
    size: tuple[int, int],
    pad_value: tuple[int, int, int] = PAD_VALUE,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize preserving aspect ratio and pad to ``size`` = (width, height).

    Accepts a PIL image or an HWC RGB uint8 array. Passing an array avoids the
    PIL round-trip entirely; the geometry and output pixels are identical.

    Mirrors the reference export pipeline: OpenCV ``INTER_LINEAR`` resize,
    integer pad rounding, constant 114 border.
    """
    target_w, target_h = int(size[0]), int(size[1])
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an HWC 3-channel array, got shape {image.shape}")
        orig_h, orig_w = image.shape[:2]
        rgb = image
    else:
        orig_w, orig_h = image.size
        # convert() is a no-op for RGB images but still copies; skip it.
        rgb = np.asarray(image if image.mode == "RGB" else image.convert("RGB"))
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image size: {(orig_w, orig_h)}")

    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = round(orig_w * ratio), round(orig_h * ratio)
    dw, dh = (target_w - new_w) / 2.0, (target_h - new_h) / 2.0
    left, right = round(dw - 0.1), round(dw + 0.1)
    top, bottom = round(dh - 0.1), round(dh + 0.1)

    if (new_w, new_h) != (orig_w, orig_h):
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
    if canvas.shape[0] != target_h or canvas.shape[1] != target_w:
        canvas = cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return canvas, LetterboxTransform(
        ratio=ratio,
        pad_left=left,
        pad_top=top,
        input_width=target_w,
        input_height=target_h,
        orig_width=orig_w,
        orig_height=orig_h,
    )
