"""Tests for the shared letterbox preprocessing helper."""

import numpy as np
from PIL import Image

from mata.core.preprocess import letterbox


def test_letterbox_wide_image_pads_top_bottom():
    image = Image.new("RGB", (1280, 720), (255, 0, 0))
    canvas, tf = letterbox(image, (640, 640))
    assert canvas.shape == (640, 640, 3)
    assert tf.ratio == 0.5
    assert (tf.pad_left, tf.pad_top) == (0, 140)
    assert np.allclose(canvas[0, 0], (114, 114, 114))
    assert np.allclose(canvas[140, 320], (255, 0, 0))


def test_letterbox_tall_image_pads_left_right():
    image = Image.new("RGB", (720, 1280), (0, 255, 0))
    canvas, tf = letterbox(image, (640, 640))
    assert canvas.shape == (640, 640, 3)
    assert tf.ratio == 0.5
    assert (tf.pad_left, tf.pad_top) == (140, 0)
    assert np.allclose(canvas[320, 140], (0, 255, 0))


def test_letterbox_square_image_is_unchanged():
    image = Image.new("RGB", (640, 640), (10, 20, 30))
    canvas, tf = letterbox(image, (640, 640))
    assert canvas.shape == (640, 640, 3)
    assert tf.ratio == 1.0
    assert (tf.pad_left, tf.pad_top) == (0, 0)
    assert np.allclose(canvas[100, 100], (10, 20, 30))
