"""OCR recognition metrics (OCRMetrics) — Task A1/A2 implementation.

Implements standard recognition-only evaluation metrics for OCR:

  * CER (Character Error Rate) — Levenshtein / max(|gt|, 1)
  * WER (Word Error Rate)      — word-level Levenshtein / max(|gt_words|, 1)
  * Accuracy                   — exact-match ratio (higher is better)

Running-accumulation pattern: call ``process_batch()`` for each image,
then call ``finalize()`` once to average over all images.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Standard O(n*m) dynamic programming implementation with O(min(n, m))
    space optimisation (single-row).
    Used internally for CER and WER computation.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Minimum number of single-character edits (insertions,
        deletions, substitutions) to transform *a* into *b*.
    """
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _levenshtein_seq(a: list, b: list) -> int:
    """Compute Levenshtein edit distance between two token sequences.

    Identical to :func:`_levenshtein` but operates on arbitrary lists of
    tokens (e.g. words) rather than character strings.  Used internally
    for WER computation.

    Args:
        a: First token sequence.
        b: Second token sequence.

    Returns:
        Minimum number of single-token edits (insertions, deletions,
        substitutions) to transform *a* into *b*.
    """
    if len(a) < len(b):
        return _levenshtein_seq(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a):
        curr = [i + 1]
        for j, tb in enumerate(b):
            cost = 0 if ta == tb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


@dataclass
class OCRMetrics:
    """OCR recognition metrics — CER, WER, and exact-match accuracy.

    Running-accumulation pattern: call ``process_batch()`` for each image,
    then call ``finalize()`` once to average over all images.

    Usage::

        metrics = OCRMetrics(case_sensitive=False)
        for pred_text, gt_text in dataset:
            metrics.process_batch(pred_text, gt_text)
        metrics.finalize()

        print(metrics.cer)       # mean character error rate
        print(metrics.accuracy)  # exact-match accuracy

    Args:
        case_sensitive: When ``False`` (default), both predicted and
            ground-truth text are lowercased before comparison.
            Matches ICDAR benchmark convention.

    Attributes:
        cer:      Mean Character Error Rate — lower is better.
        wer:      Mean Word Error Rate — lower is better.
        accuracy: Exact-match accuracy — higher is better.
        speed:    Timing breakdown in ms/image.
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    #: When ``False`` (default), text is lowercased before comparison.
    case_sensitive: bool = False

    # ------------------------------------------------------------------
    # Accumulated metric fields (set by finalize())
    # ------------------------------------------------------------------

    #: Mean Character Error Rate — Levenshtein(pred, gt) / max(|gt|, 1).
    cer: float = 0.0
    #: Mean Word Error Rate — word-level Levenshtein / max(|gt_words|, 1).
    wer: float = 0.0
    #: Exact-match accuracy — fraction of images with pred == gt.
    accuracy: float = 0.0

    #: Timing breakdown in ms/image.
    speed: dict[str, float] = field(
        default_factory=lambda: {
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
        }
    )

    # ------------------------------------------------------------------
    # Private accumulators (not part of the public API)
    # ------------------------------------------------------------------

    _cer_sum: float = field(default=0.0, repr=False)
    _wer_sum: float = field(default=0.0, repr=False)
    _exact_matches: int = field(default=0, repr=False)
    _count: int = field(default=0, repr=False)

    # ------------------------------------------------------------------
    # Core accumulation
    # ------------------------------------------------------------------

    def process_batch(self, pred_text: str, gt_text: str) -> None:
        """Accumulate metrics for one image.

        Args:
            pred_text: Predicted text (full image transcription).
            gt_text:   Ground-truth text (full image transcription).
        """
        # Normalize case
        if not self.case_sensitive:
            pred_text = pred_text.lower()
            gt_text = gt_text.lower()

        # CER: character-level edit distance / GT length
        char_dist = _levenshtein(pred_text, gt_text)
        cer = char_dist / max(len(gt_text), 1)

        # WER: word-level edit distance / GT word count
        pred_words = pred_text.split()
        gt_words = gt_text.split()
        word_dist = _levenshtein_seq(pred_words, gt_words)
        wer = word_dist / max(len(gt_words), 1)

        # Exact match
        exact = 1 if pred_text.strip() == gt_text.strip() else 0

        self._cer_sum += cer
        self._wer_sum += wer
        self._exact_matches += exact
        self._count += 1

    def finalize(self) -> None:
        """Average accumulated metrics over all processed images.

        Must be called once after all ``process_batch()`` calls.
        Calling ``finalize()`` on a fresh (zero-image) accumulator sets
        all metrics to 0.0.  Calling it a second time overwrites previous
        values — always call after completing a full evaluation loop.
        """
        n = max(self._count, 1)
        self.cer = self._cer_sum / n
        self.wer = self._wer_sum / n
        self.accuracy = self._exact_matches / n

    # ------------------------------------------------------------------
    # Legacy / Validator interface
    # ------------------------------------------------------------------

    def update(self, pred_text: str, gt_text: str) -> None:
        """Alias for :meth:`process_batch` (Validator compatibility)."""
        self.process_batch(pred_text, gt_text)

    def mean_results(self) -> list[float]:
        """Return ``[cer, wer, accuracy]``.

        Used by the console printer to populate the summary row.
        """
        return [self.cer, self.wer, self.accuracy]

    # ------------------------------------------------------------------
    # Fitness & diagnostics
    # ------------------------------------------------------------------

    @property
    def fitness(self) -> float:
        """Scalar quality score: exact-match accuracy.

        Higher is better.

        Returns:
            Float in ``[0, 1]``.
        """
        return self.accuracy

    @property
    def keys(self) -> list[str]:
        """Metric keys (without ``"fitness"``) for use by loggers."""
        return ["metrics/cer", "metrics/wer", "metrics/accuracy"]

    @property
    def results_dict(self) -> dict[str, float]:
        """Flat metrics dict for logging and external tools.

        Returns a dict with exactly 4 keys::

            {
                "metrics/cer":      <float>,
                "metrics/wer":      <float>,
                "metrics/accuracy": <float>,
                "fitness":          <float>,
            }
        """
        return {
            "metrics/cer": self.cer,
            "metrics/wer": self.wer,
            "metrics/accuracy": self.accuracy,
            "fitness": self.fitness,
        }

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def summary(self) -> list[dict[str, Any]]:
        """Return a single-row summary list for human consumption.

        The list always has exactly one entry — OCR evaluation produces
        a global scalar summary rather than per-class rows.

        Returns:
            A list with one dict containing all three metrics and fitness.
        """
        return [
            {
                "cer": round(self.cer, 6),
                "wer": round(self.wer, 6),
                "accuracy": round(self.accuracy, 6),
                "fitness": round(self.fitness, 6),
            }
        ]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of all metrics."""
        return {
            "results": self.results_dict,
            "speed": self.speed,
            "summary": self.summary(),
        }

    def to_json(self) -> str:
        """Serialise metrics to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_csv(self) -> str:
        """Serialise metrics to a CSV string.

        Returns a header row followed by one data row with all metrics.
        """
        fieldnames = ["cer", "wer", "accuracy", "fitness"]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in self.summary():
            writer.writerow(row)
        return buf.getvalue()
