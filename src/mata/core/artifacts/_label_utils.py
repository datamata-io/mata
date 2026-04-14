"""Shared label-matching utilities for artifact modules."""

from __future__ import annotations

import re


def _fuzzy_label_match(label1: str, label2: str) -> bool:
    """Fuzzy label matching for entity promotion.

    Handles:
    - Case insensitivity
    - Plural forms (simple heuristic: trailing 's')
    - Leading articles (a, an, the)
    - Extra whitespace

    Args:
        label1: First label
        label2: Second label

    Returns:
        True if labels match fuzzy criteria

    Example:
        >>> _fuzzy_label_match("cat", "cats")
        True
        >>> _fuzzy_label_match("The Dog", "dog")
        True
        >>> _fuzzy_label_match("person", "people")
        False  # Complex plural not handled
    """

    def normalize(text: str) -> str:
        """Normalize label for fuzzy matching."""
        # Lowercase and strip whitespace
        text = text.lower().strip()

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Remove leading articles
        for article in ["a ", "an ", "the "]:
            if text.startswith(article):
                text = text[len(article) :]
                break

        # Remove trailing 's' for simple plural handling
        # (doesn't handle irregular plurals like "people", "children")
        if text.endswith("s") and len(text) > 1 and not text.endswith("ss"):
            text = text[:-1]

        return text

    norm1 = normalize(label1)
    norm2 = normalize(label2)

    return norm1 == norm2
