"""Shared transformers version floors for adapter model support.

Some checkpoints are only supported from a specific transformers release
(e.g. a model merged upstream in 5.21, or a checkpoint that ships custom code
requiring a newer API). MATA keeps the global pin in ``pyproject.toml`` wide
and gates individual models here instead. A missing floor means the model is
expected to work on the base pin; an entry here is enforced with a
``ModelLoadError`` before any weights are fetched.
"""

from __future__ import annotations

# model id -> minimum transformers version, enforced before the model is loaded
MIN_TRANSFORMERS: dict[str, str] = {
    "google/gemma-4-E2B-it": "5.5.0",
    "LiquidAI/LFM3-VL-2B": "5.21.0",
    "openbmb/MiniCPM-OCR-1.0": "5.21.0",
}


def parse_version(value: str) -> tuple[int, ...]:
    """'5.21.0' -> (5, 21, 0); tolerates suffixes like '5.21.0.dev0'."""
    parts = []
    for chunk in str(value).split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def floor_for(model_id: str) -> str | None:
    """Return the minimum transformers version for a model id, if any."""
    floor = MIN_TRANSFORMERS.get(model_id)
    if floor is not None:
        return floor
    for known, value in MIN_TRANSFORMERS.items():
        if model_id.startswith(known):
            return value
    return None


def check_min_transformers(model_id: str, installed: str) -> str | None:
    """Return an actionable message when ``installed`` is below the model floor."""
    floor = floor_for(model_id)
    if floor is None:
        return None
    if parse_version(installed) < parse_version(floor):
        return (
            f"requires transformers>={floor} (installed: {installed}). "
            f"Upgrade with: pip install -U 'transformers>={floor}'"
        )
    return None
