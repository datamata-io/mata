"""Unit tests for the per-model transformers floors."""

from __future__ import annotations

import pytest

from mata.adapters.transformers_compat import (
    MIN_TRANSFORMERS,
    check_min_transformers,
    floor_for,
    parse_version,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("5.21.0", (5, 21, 0)),
        ("5.21.0.dev0", (5, 21, 0, 0)),
        ("4.35.0", (4, 35, 0)),
    ],
)
def test_parse_version(value, expected):
    assert parse_version(value) == expected


def test_floor_for_exact_and_prefix():
    assert floor_for("openbmb/MiniCPM-OCR-1.0") == "5.21.0"
    assert floor_for("LiquidAI/LFM3-VL-2B-fp8") == "5.21.0"
    assert floor_for("facebook/detr-resnet-50") is None


def test_check_min_transformers_passes_at_or_above_floor():
    assert check_min_transformers("openbmb/MiniCPM-OCR-1.0", "5.21.0") is None
    assert check_min_transformers("openbmb/MiniCPM-OCR-1.0", "5.22.1") is None


def test_check_min_transformers_explains_upgrade():
    message = check_min_transformers("openbmb/MiniCPM-OCR-1.0", "5.20.2")
    assert message is not None
    assert "transformers>=5.21.0" in message
    assert "installed: 5.20.2" in message


def test_gemma4_requires_transformers_55():
    # Gemma 4 multimodal landed upstream in 5.5.0; on 5.0-5.4 the load fails
    # deep inside AutoModel with a message that does not name the requirement.
    assert floor_for("google/gemma-4-E2B-it") == "5.5.0"
    message = check_min_transformers("google/gemma-4-E2B-it", "5.4.1")
    assert message is not None and "transformers>=5.5.0" in message


def test_every_floor_is_above_the_base_pin():
    for model_id, floor in MIN_TRANSFORMERS.items():
        assert parse_version(floor) >= (4, 35, 0), model_id
