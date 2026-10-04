"""MATA Training & Fine-Tuning Module.

Provides mata.train() and mata.finetune() APIs for training and fine-tuning
detection, classification, and segmentation models on custom datasets.
"""

from __future__ import annotations

try:
    from .config import TrainingConfig
except ImportError:  # pragma: no cover
    TrainingConfig = None  # type: ignore[assignment,misc]

try:
    from .result import TrainingResult
except ImportError:  # pragma: no cover
    TrainingResult = None  # type: ignore[assignment,misc]

try:
    from .trainer import TrainingOrchestrator
except ImportError:  # pragma: no cover
    TrainingOrchestrator = None  # type: ignore[assignment,misc]

try:
    from .callbacks import EarlyStoppingCallback, LoggingCallback, ValidationCallback
except ImportError:  # pragma: no cover
    EarlyStoppingCallback = None  # type: ignore[assignment,misc]
    LoggingCallback = None  # type: ignore[assignment,misc]
    ValidationCallback = None  # type: ignore[assignment,misc]

try:
    from .checkpoint import CheckpointManager
except ImportError:  # pragma: no cover
    CheckpointManager = None  # type: ignore[assignment,misc]

__all__ = [
    "CheckpointManager",
    "TrainingConfig",
    "TrainingResult",
    "TrainingOrchestrator",
    "ValidationCallback",
    "LoggingCallback",
    "EarlyStoppingCallback",
]
