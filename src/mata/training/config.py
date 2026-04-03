"""Training configuration dataclass for mata.train() and mata.finetune()."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mata.core.exceptions import ConfigurationError

_VALID_TASKS = {"detect", "classify", "segment"}
_VALID_OPTIMIZERS = {"adamw", "sgd", "adam"}
_VALID_SCHEDULERS = {"cosine", "linear", "step", "none"}
_DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda|cuda:\d+)$")


@dataclass
class TrainingConfig:
    """Training configuration for mata.train() and mata.finetune().

    Attributes:
        task: Task type — "detect", "classify", or "segment"
        model: Model source (same as mata.load() — HF ID, torchvision/*, path, alias)
        data: Dataset path (YAML config, directory, or COCO JSON)
        val_data: Optional separate validation dataset path
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        optimizer: Optimizer type — "adamw", "sgd", "adam"
        weight_decay: L2 regularization
        scheduler: LR scheduler — "cosine", "linear", "step", "none"
        warmup_epochs: Number of warmup epochs
        device: Device — "auto", "cuda", "cpu"
        amp: Automatic mixed precision
        save_dir: Directory for checkpoints and logs
        save_every: Save checkpoint every N epochs (0 = best + last only)
        val_every: Validate every N epochs
        patience: Early stopping patience (0 = disabled)
        freeze_backbone: Freeze backbone parameters (for fine-tuning)
        freeze_layers: Specific layer name patterns to freeze
        augment: Enable data augmentation
        augment_config: Custom augmentation config dict
        resume: Checkpoint path to resume from
        num_workers: DataLoader worker count
        gradient_accumulation_steps: Accumulate gradients over N steps before updating weights.
            Effective batch size = batch_size × gradient_accumulation_steps.
            Useful to simulate larger batches without extra VRAM.
        gradient_checkpointing: Recompute activations during the backward pass instead of
            storing them. Reduces VRAM at the cost of ~20–30 % extra compute.
        max_grad_norm: Maximum gradient norm for clipping. Set to 0.0 to disable.
            The DETR/RT-DETR family of models benefits from a tight clip (e.g. 0.1)
            to prevent the occasional inf grad_norm seen during warmup.
        seed: Random seed for reproducibility
        verbose: Print progress to console
    """

    task: str = ""
    model: str = ""
    data: str | dict = ""
    val_data: str | dict | None = None
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-4
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    device: str = "auto"
    amp: bool = True
    save_dir: str = "runs/train"
    save_every: int = 0
    val_every: int = 1
    patience: int = 0
    freeze_backbone: bool = False
    freeze_layers: list[str] | None = None
    augment: bool = True
    augment_config: dict[str, Any] | None = None
    resume: str | None = None
    num_workers: int = 4
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    seed: int = 42
    verbose: bool = True

    def __post_init__(self) -> None:
        # Windows uses 'spawn' for multiprocessing, so worker sub-processes cannot
        # safely be started outside of a `if __name__ == '__main__':` guard.
        # Force num_workers=0 (main-process loading) on Windows to avoid the
        # "bootstrapping phase" RuntimeError from PyTorch DataLoader.
        if sys.platform == "win32" and self.num_workers > 0:
            self.num_workers = 0

    def validate(self) -> None:
        """Validate all configuration fields.

        Raises:
            ConfigurationError: On the first invalid field, with an actionable message.
        """
        # task
        if self.task not in _VALID_TASKS:
            raise ConfigurationError(
                f"Invalid task '{self.task}'. " f"Must be one of: {', '.join(sorted(_VALID_TASKS))}."
            )

        # optimizer
        if self.optimizer not in _VALID_OPTIMIZERS:
            raise ConfigurationError(
                f"Invalid optimizer '{self.optimizer}'. " f"Must be one of: {', '.join(sorted(_VALID_OPTIMIZERS))}."
            )

        # scheduler
        if self.scheduler not in _VALID_SCHEDULERS:
            raise ConfigurationError(
                f"Invalid scheduler '{self.scheduler}'. " f"Must be one of: {', '.join(sorted(_VALID_SCHEDULERS))}."
            )

        # device
        if not _DEVICE_PATTERN.match(self.device):
            raise ConfigurationError(
                f"Invalid device '{self.device}'. " f"Must be 'auto', 'cpu', 'cuda', or 'cuda:<index>' (e.g. 'cuda:0')."
            )

        # numeric ranges
        if self.epochs <= 0:
            raise ConfigurationError(f"epochs must be > 0, got {self.epochs}.")
        if self.batch_size <= 0:
            raise ConfigurationError(f"batch_size must be > 0, got {self.batch_size}.")
        if self.lr <= 0:
            raise ConfigurationError(f"lr must be > 0, got {self.lr}.")
        if self.weight_decay < 0:
            raise ConfigurationError(f"weight_decay must be >= 0, got {self.weight_decay}.")
        if self.warmup_epochs < 0:
            raise ConfigurationError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}.")
        if self.warmup_epochs >= self.epochs:
            raise ConfigurationError(f"warmup_epochs ({self.warmup_epochs}) must be less than epochs ({self.epochs}).")
        if self.save_every < 0:
            raise ConfigurationError(f"save_every must be >= 0, got {self.save_every}.")
        if self.val_every <= 0:
            raise ConfigurationError(f"val_every must be > 0, got {self.val_every}.")
        if self.patience < 0:
            raise ConfigurationError(f"patience must be >= 0, got {self.patience}.")
        if self.num_workers < 0:
            raise ConfigurationError(f"num_workers must be >= 0, got {self.num_workers}.")
        if self.gradient_accumulation_steps < 1:
            raise ConfigurationError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}."
            )
        if self.max_grad_norm < 0:
            raise ConfigurationError(f"max_grad_norm must be >= 0 (0 = disabled), got {self.max_grad_norm}.")

        # resume path must exist if specified
        if self.resume is not None and not Path(self.resume).exists():
            raise ConfigurationError(
                f"resume path '{self.resume}' does not exist. "
                f"Provide the path to a valid MATA checkpoint directory."
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load a TrainingConfig from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A populated TrainingConfig instance.

        Raises:
            ConfigurationError: If the file cannot be read or contains invalid YAML.
        """
        path = Path(path)
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigurationError(f"Training config file not found: {path}")
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Failed to parse training config YAML at '{path}': {exc}")

        if not isinstance(data, dict):
            raise ConfigurationError(f"Training config YAML must be a mapping, got {type(data).__name__}.")

        # Filter to known fields only; unknown keys are silently ignored
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered)
