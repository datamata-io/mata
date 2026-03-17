"""Training Orchestrator — dispatches to HFTrainingEngine or TorchTrainingEngine.

Engine detection mirrors UniversalLoader._detect_source_type():
- ``"torchvision/*"`` → TorchTrainingEngine
- ``"org/model"`` (contains ``"/"``) → HFTrainingEngine
- Config alias → resolve to underlying source and re-detect
- Local checkpoint → detect engine from ``config.json``
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

import yaml

from mata.core.exceptions import ModelNotFoundError, TrainingError
from mata.core.logging import get_logger
from mata.training.config import TrainingConfig
from mata.training.result import TrainingResult

logger = get_logger(__name__)

# Matches a local file extension that torch/HF saves state into
_LOCAL_FILE_SUFFIXES = {".pt", ".pth", ".onnx", ".bin", ".trt", ".engine"}


def _auto_save_dir(base: str, task: str) -> Path:
    """Return an auto-incremented save directory.

    E.g.::

        runs/train/detect   → runs/train/detect
        runs/train/detect   → runs/train/detect2   (if first exists)
        runs/train/detect   → runs/train/detect3   (if first two exist)
    """
    base_path = Path(base) / task
    if not base_path.exists():
        return base_path
    counter = 2
    while True:
        candidate = Path(base) / f"{task}{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


class TrainingOrchestrator:
    """Dispatches training to HFTrainingEngine or TorchTrainingEngine.

    Engine selection mirrors the strategy used by ``UniversalLoader``:

    1. ``source.startswith("torchvision/")`` → ``TorchTrainingEngine``
    2. ``"/" in source`` → ``HFTrainingEngine``
    3. Config alias → resolve via ``ModelRegistry``, then re-detect
    4. Local checkpoint directory (contains ``config.json``) → engine from
       ``config.json``
    5. Otherwise → raise ``TrainingError``

    Args:
        config: Fully validated :class:`~mata.training.config.TrainingConfig`.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Engine detection
    # ------------------------------------------------------------------

    def _detect_engine(self, model_source: str) -> str:
        """Detect which engine to use for *model_source*.

        Args:
            model_source: Model identifier — HF ID, ``torchvision/*``,
                          config alias, or local checkpoint path.

        Returns:
            ``"huggingface"`` or ``"torchvision"``

        Raises:
            :class:`~mata.core.exceptions.TrainingError`: If the engine
                cannot be determined.
        """
        resolved = self._resolve_alias(model_source)
        return self._classify_source(resolved)

    def _resolve_alias(self, source: str) -> str:
        """Resolve a config alias to its underlying source string.

        If *source* is not an alias it is returned unchanged.
        Alias chains are followed (same behaviour as ``UniversalLoader``).
        """
        try:
            from mata.core.model_registry import ModelRegistry
        except ImportError:
            return source

        registry = ModelRegistry()
        visited: set[str] = set()
        current = source

        while registry.has_alias(self.config.task, current):
            if current in visited:
                logger.warning(
                    f"Circular alias detected for '{current}'; stopping resolution."
                )
                break
            visited.add(current)
            cfg = registry.get_config(self.config.task, current)
            sub = cfg.get("source", current)
            if sub == current:
                break
            current = sub

        return current

    def _classify_source(self, source: str) -> str:
        """Classify *source* as ``"torchvision"`` or ``"huggingface"``.

        Args:
            source: Already-resolved source string.

        Returns:
            ``"torchvision"`` or ``"huggingface"``

        Raises:
            :class:`~mata.core.exceptions.TrainingError`: If source type is
                unrecognised.
        """
        # Local checkpoint directory — read engine from config.json
        ckpt_path = Path(source)
        if ckpt_path.is_dir() and (ckpt_path / "config.json").is_file():
            return self._engine_from_checkpoint(ckpt_path)

        # Torchvision
        if source.startswith("torchvision/"):
            return "torchvision"

        # Has a path suffix — local file, not a trainable source
        if Path(source).suffix.lower() in _LOCAL_FILE_SUFFIXES and not ckpt_path.is_dir():
            raise TrainingError(
                f"Cannot train from a plain weight file: '{source}'. "
                "Provide an HF model ID (e.g. 'facebook/detr-resnet-50') or a "
                "torchvision model key (e.g. 'torchvision/fasterrcnn_resnet50_fpn')."
            )

        # HuggingFace — contains '/'
        if "/" in source:
            return "huggingface"

        raise TrainingError(
            f"Cannot determine training engine for model source '{source}'. "
            "Use an HuggingFace model ID (e.g. 'facebook/detr-resnet-50'), "
            "a torchvision key (e.g. 'torchvision/fasterrcnn_resnet50_fpn'), "
            "or a config alias defined in .mata/models.yaml."
        )

    def _engine_from_checkpoint(self, ckpt_dir: Path) -> str:
        """Read ``config.json`` in *ckpt_dir* and return the engine type."""
        config_path = ckpt_dir / "config.json"
        try:
            with config_path.open() as fh:
                meta = json.load(fh)
            engine = meta.get("engine", "")
            if engine in ("huggingface", "torchvision"):
                return engine
            # Fallback: inspect model source field
            raw_source = meta.get("model", meta.get("model_source", ""))
            if raw_source:
                return self._classify_source(raw_source)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Could not read checkpoint config.json: {exc}")

        # Last resort: presence of HF-format files
        if (ckpt_dir / "model.safetensors").exists() or (ckpt_dir / "pytorch_model.bin").exists():
            return "huggingface"
        return "torchvision"

    # ------------------------------------------------------------------
    # Save directory helpers
    # ------------------------------------------------------------------

    def _prepare_save_dir(self) -> Path:
        """Create and return the auto-incremented save directory."""
        save_dir = _auto_save_dir(self.config.save_dir, self.config.task)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Save directory: {save_dir}")
        return save_dir

    def _write_config(self, save_dir: Path) -> None:
        """Serialise the current :class:`~mata.training.config.TrainingConfig`
        to ``{save_dir}/config.yaml`` for reproducibility."""
        import dataclasses

        config_path = save_dir / "config.yaml"
        raw = dataclasses.asdict(self.config)
        with config_path.open("w") as fh:
            yaml.safe_dump(raw, fh, default_flow_style=False, sort_keys=False)
        logger.debug(f"Config written to {config_path}")

    # ------------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seeds(seed: int) -> None:
        """Set random seeds for reproducibility."""
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Dataset & augmentation building
    # ------------------------------------------------------------------

    def _build_datasets(self, save_dir: Path) -> tuple[Any, Any, Any]:
        """Build train/val datasets and the collate function.

        Returns:
            ``(train_dataset, val_dataset_or_None, collate_fn)``
        """
        from mata.training.augmentations.factory import AugmentationFactory
        from mata.training.datasets.factory import DatasetFactory

        train_aug = (
            AugmentationFactory.create(
                self.config.task,
                config=self.config.augment_config,
                train=True,
            )
            if self.config.augment
            else None
        )
        val_aug = (
            AugmentationFactory.create(
                self.config.task,
                config=self.config.augment_config,
                train=False,
            )
            if self.config.augment
            else None
        )

        train_dataset, collate_fn = DatasetFactory.create(
            self.config.task,
            self.config.data,
            split="train",
            transforms=train_aug,
        )

        val_dataset = None
        val_source = self.config.val_data or self.config.data
        if val_source:
            try:
                val_dataset, _ = DatasetFactory.create(
                    self.config.task,
                    val_source,
                    split="val",
                    transforms=val_aug,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not build validation dataset: {exc}")
                val_dataset = None

        return train_dataset, val_dataset, collate_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> TrainingResult:
        """Run the full training pipeline.

        Steps:

        1. Set random seeds.
        2. Create save directory (auto-incremented).
        3. Write ``config.yaml`` for reproducibility.
        4. Build datasets via :class:`~mata.training.datasets.factory.DatasetFactory`.
        5. Build augmentations via
           :class:`~mata.training.augmentations.factory.AugmentationFactory`.
        6. Detect the training engine.
        7. Dispatch to :class:`~mata.training.hf_trainer.HFTrainingEngine` or
           :class:`~mata.training.torch_trainer.TorchTrainingEngine`.
        8. Return :class:`~mata.training.result.TrainingResult`.

        Returns:
            :class:`~mata.training.result.TrainingResult` with metrics,
            checkpoint paths, and per-epoch history.

        Raises:
            :class:`~mata.core.exceptions.TrainingError`: On unrecoverable
                training failures.
            :class:`~mata.core.exceptions.ConfigurationError`: If the config
                is invalid.
        """
        self._set_seeds(self.config.seed)

        save_dir = self._prepare_save_dir()
        # Patch config so engines write into the resolved directory
        self.config.save_dir = str(save_dir)

        self._write_config(save_dir)

        train_dataset, val_dataset, _collate_fn = self._build_datasets(save_dir)

        engine_name = self._detect_engine(self.config.model)
        logger.info(f"Detected engine: {engine_name}")

        result = self._dispatch(engine_name, train_dataset, val_dataset)
        return result

    def finetune(self) -> TrainingResult:
        """Convenience wrapper — enables backbone freezing and delegates to
        :meth:`train`.

        Sets ``config.freeze_backbone = True`` before dispatching.

        Returns:
            :class:`~mata.training.result.TrainingResult`
        """
        self.config.freeze_backbone = True
        return self.train()

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        engine_name: str,
        train_dataset: Any,
        val_dataset: Any,
    ) -> TrainingResult:
        """Instantiate the chosen engine and run ``train()``."""
        model_source = self._resolve_alias(self.config.model)

        if engine_name == "torchvision":
            from mata.training.torch_trainer import TorchTrainingEngine

            engine = TorchTrainingEngine(
                task=self.config.task,
                model_name=model_source,
                config=self.config,
            )
        elif engine_name == "huggingface":
            from mata.training.hf_trainer import HFTrainingEngine

            engine = HFTrainingEngine(
                task=self.config.task,
                model_id=model_source,
                config=self.config,
            )
        else:
            raise TrainingError(
                f"Unknown engine '{engine_name}'. Expected 'huggingface' or 'torchvision'."
            )

        return engine.train(train_dataset, val_dataset)
