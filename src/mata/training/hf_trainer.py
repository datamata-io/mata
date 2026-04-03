"""HuggingFace training engine for MATA — detect, classify, and segment tasks.

Uses ``transformers.Trainer`` as the training backend.  Supports:

- **Detection**: DETR / RT-DETR via ``AutoModelForObjectDetection``
- **Classification**: ResNet / ViT / ConvNeXt via ``AutoModelForImageClassification``
- **Segmentation**: Mask2Former via ``Mask2FormerForUniversalSegmentation``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mata.core.exceptions import TrainingError
from mata.core.logging import get_logger
from mata.training.config import TrainingConfig
from mata.training.result import TrainingResult

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy transformers import
# ---------------------------------------------------------------------------

_transformers: dict[str, Any] | None = None


def _ensure_transformers() -> dict[str, Any]:
    """Lazily import transformers and return a mapping of required classes.

    Raises:
        ImportError: If transformers is not installed.
    """
    global _transformers
    if _transformers is not None:
        return _transformers

    try:
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoModelForImageClassification,
            AutoModelForObjectDetection,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )

        # Mask2Former is optional — not all transformers installs include it.
        try:
            from transformers import Mask2FormerForUniversalSegmentation
        except ImportError:
            Mask2FormerForUniversalSegmentation = None  # type: ignore[assignment]

        _transformers = {
            "AutoConfig": AutoConfig,
            "AutoImageProcessor": AutoImageProcessor,
            "AutoModelForObjectDetection": AutoModelForObjectDetection,
            "AutoModelForImageClassification": AutoModelForImageClassification,
            "Mask2FormerForUniversalSegmentation": Mask2FormerForUniversalSegmentation,
            "TrainingArguments": TrainingArguments,
            "Trainer": Trainer,
            "TrainerCallback": TrainerCallback,
        }
    except ImportError as exc:
        raise ImportError(
            "transformers is required for HFTrainingEngine. " "Install with: pip install transformers torch"
        ) from exc

    return _transformers


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_TASKS = {"detect", "classify", "segment"}

# Parameter name substrings that form the task-specific head (kept trainable
# when freeze_backbone=True).
_TASK_HEAD_PATTERNS: dict[str, set[str]] = {
    "detect": {"class_labels_classifier", "bbox_predictor"},
    "classify": {"classifier"},
    "segment": {"class_predictor", "mask_embedder"},
}

# transformers optimizer names
_OPT_MAP = {
    "adamw": "adamw_torch",
    "adam": "adam",
    "sgd": "sgd",
}

# transformers LR scheduler names
_SCHED_MAP = {
    "cosine": "cosine",
    "linear": "linear",
    "step": "constant_with_warmup",
    "none": "constant",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _HistoryCallback:
    """Minimal per-epoch metrics accumulator (not a TrainerCallback subclass).

    Registered as a real TrainerCallback inside :class:`HFTrainingEngine` via
    the ``_HistoryCBWrapper`` adapter defined in :meth:`train`.
    """

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {}

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any
    ) -> None:  # noqa: ARG002
        """Collect numeric log entries."""
        if not logs:
            return
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.history.setdefault(key, []).append(float(value))


class _ClassificationCollator:
    """Collate classification samples for HF Trainer.

    Accepts batches of ``(PIL.Image | Tensor, {"label": int})`` and returns
    a dict with ``pixel_values`` and ``labels`` ready for an HF image
    classification model.
    """

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, batch: list) -> dict[str, Any]:
        import torch

        images, targets = zip(*batch)

        # Process images — handle both PIL Images and pre-converted tensors.
        if hasattr(images[0], "mode"):  # PIL Image
            encoding = self.processor(images=list(images), return_tensors="pt")
        else:
            encoding = {"pixel_values": torch.stack(list(images))}

        labels = torch.tensor(
            [t["label"] if isinstance(t, dict) else int(t) for t in targets],
            dtype=torch.long,
        )
        encoding["labels"] = labels
        return dict(encoding)


class _DetectionCollator:
    """Collate detection / segmentation samples for HF Trainer.

    Accepts batches of
    ``(PIL.Image | Tensor, {"boxes": Tensor[N,4], "labels": Tensor[N], ...})``
    and returns a dict with ``pixel_values``, ``pixel_mask`` (if produced by
    the processor), and ``labels`` in HF DETR format.

    HF DETR label format per image::

        {
            "class_labels": LongTensor[N],
            "boxes": FloatTensor[N, 4],  # xyxy absolute, passed to processor
        }

    The processor converts absolute xyxy → normalised cxcywh internally when
    it prepares the inputs.
    """

    def __init__(self, processor: Any, task: str = "detect") -> None:
        self.processor = processor
        self.task = task

    def __call__(self, batch: list) -> dict[str, Any]:
        import torch

        images, targets = zip(*batch)

        # Build HF-format annotations list (one entry per image).
        # Detection  → {"class_labels": LongTensor[N], "boxes": FloatTensor[N,4]}
        # Segment    → {"class_labels": LongTensor[N], "mask_labels": FloatTensor[N,H,W]}
        # Mask2Former requires `mask_labels` to trigger its internal loss;
        # DETR-family models require `boxes`.
        annotations: list[dict[str, Any]] = []
        for target in targets:
            if not isinstance(target, dict):
                if self.task == "segment":
                    annotations.append(
                        {
                            "class_labels": torch.zeros(0, dtype=torch.long),
                            "mask_labels": torch.zeros((0, 1, 1), dtype=torch.float32),
                        }
                    )
                else:
                    annotations.append(
                        {
                            "class_labels": torch.zeros(0, dtype=torch.long),
                            "boxes": torch.zeros((0, 4), dtype=torch.float32),
                        }
                    )
                continue

            labels = target.get("labels")
            class_labels = (
                labels
                if isinstance(labels, torch.Tensor)
                else torch.tensor(labels if labels is not None else [], dtype=torch.long)
            )

            if self.task == "segment":
                masks = target.get("masks")  # Tensor[N, H, W] uint8 from COCOSegmentationDataset
                if masks is not None and isinstance(masks, torch.Tensor) and masks.numel() > 0:
                    mask_labels = masks.float()
                else:
                    mask_labels = torch.zeros((0, 1, 1), dtype=torch.float32)
                annotations.append({"class_labels": class_labels, "mask_labels": mask_labels})
            else:
                boxes = target.get("boxes")
                if boxes is None or labels is None:
                    annotations.append(
                        {
                            "class_labels": torch.zeros(0, dtype=torch.long),
                            "boxes": torch.zeros((0, 4), dtype=torch.float32),
                        }
                    )
                    continue
                boxes_t = boxes if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.float32)
                annotations.append({"class_labels": class_labels, "boxes": boxes_t})

        # Encode images; attempt to let the processor pad them.
        if hasattr(images[0], "mode"):  # PIL Images
            try:
                encoding = self.processor(images=list(images), return_tensors="pt")
            except Exception:  # pragma: no cover — processor may not support it
                encoding = {}
        else:
            try:
                encoding = self.processor(images=list(images), return_tensors="pt")
            except Exception:
                encoding = {"pixel_values": torch.stack(list(images))}

        encoding = dict(encoding)
        encoding["labels"] = annotations
        return encoding


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class HFTrainingEngine:
    """Training engine for HuggingFace transformers models.

    Uses ``transformers.Trainer`` internally which provides:

    - Automatic mixed precision (AMP / fp16)
    - Gradient accumulation
    - LR scheduling with warmup
    - Logging and progress bars
    - HuggingFace checkpoint saving

    MATA-format checkpoints are saved alongside HF checkpoints via
    :class:`~mata.training.checkpoint.CheckpointManager`.

    Supported tasks:

    +----------+------------------------------------------+
    | Task     | AutoModel class                          |
    +==========+==========================================+
    | detect   | AutoModelForObjectDetection              |
    +----------+------------------------------------------+
    | classify | AutoModelForImageClassification          |
    +----------+------------------------------------------+
    | segment  | Mask2FormerForUniversalSegmentation      |
    +----------+------------------------------------------+

    Examples::

        from mata.training import TrainingConfig
        from mata.training.hf_trainer import HFTrainingEngine

        config = TrainingConfig(
            task="classify",
            model="microsoft/resnet-50",
            data="flowers/",
            epochs=5,
            batch_size=16,
            lr=1e-4,
        )
        engine = HFTrainingEngine("classify", "microsoft/resnet-50", config)
        result = engine.train(train_dataset, val_dataset)
    """

    def __init__(self, task: str, model_id: str, config: TrainingConfig) -> None:
        """Initialise the engine.

        Args:
            task: Vision task — ``"detect"``, ``"classify"``, or ``"segment"``.
            model_id: HuggingFace model identifier (e.g. ``"facebook/detr-resnet-50"``).
            config: Populated :class:`~mata.training.config.TrainingConfig`.

        Raises:
            ValueError: If *task* is not supported.
            ImportError: If ``transformers`` is not installed.
        """
        if task not in _SUPPORTED_TASKS:
            raise ValueError(
                f"HFTrainingEngine: unsupported task '{task}'. "
                f"Expected one of: {', '.join(sorted(_SUPPORTED_TASKS))}"
            )

        self.task = task
        self.model_id = model_id
        self.config = config

        # Populated by _load_model_for_training()
        self.model: Any = None
        self.processor: Any = None
        self._device: str = "cpu"

        # Validate transformers is available at construction time so the
        # caller gets a clear ImportError immediately.
        _ensure_transformers()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model_for_training(self, id2label: dict[int, str] | None = None) -> None:
        """Load the HF model with gradients enabled for training.

        Important: this method intentionally does **not** call ``.eval()``
        or wrap the model in ``torch.no_grad()``, as both would disable
        gradient computation required for back-propagation.

        Args:
            id2label: For classify task — label mapping discovered from the
                training dataset.  When provided the model head is replaced to
                match the number of dataset classes (``ignore_mismatched_sizes=True``).

        Raises:
            TrainingError: If model loading fails.
            ImportError: If required libraries are not installed.
        """
        tf = _ensure_transformers()
        logger.info(
            "Loading HF model for training: task=%s, model=%s",
            self.task,
            self.model_id,
        )

        try:
            import torch
        except ImportError as exc:
            raise TrainingError("PyTorch is required for training.") from exc

        # Resolve device
        if self.config.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.config.device

        try:
            if self.task == "detect":
                self.processor = tf["AutoImageProcessor"].from_pretrained(self.model_id, use_fast=True)
                self.model = tf["AutoModelForObjectDetection"].from_pretrained(self.model_id)

            elif self.task == "classify":
                self.processor = tf["AutoImageProcessor"].from_pretrained(self.model_id, use_fast=True)
                cls_kwargs: dict = {}
                if id2label:
                    label2id = {v: k for k, v in id2label.items()}
                    cls_kwargs = {
                        "num_labels": len(id2label),
                        "id2label": id2label,
                        "label2id": label2id,
                        "ignore_mismatched_sizes": True,
                    }
                    logger.info(
                        "Fine-tuning classification head: %d classes %s",
                        len(id2label),
                        list(id2label.values()),
                    )
                self.model = tf["AutoModelForImageClassification"].from_pretrained(self.model_id, **cls_kwargs)

            else:  # segment
                # Mask2FormerForUniversalSegmentation is not covered by the
                # standard AutoModel API — use the class loaded in _ensure_transformers.
                Mask2Former = tf.get("Mask2FormerForUniversalSegmentation")
                if Mask2Former is None:
                    raise ImportError(
                        "transformers with Mask2Former support is required for "
                        "segment training.  Install with: pip install transformers torch"
                    )
                self.processor = tf["AutoImageProcessor"].from_pretrained(self.model_id, use_fast=True)
                self.model = Mask2Former.from_pretrained(self.model_id)

        except (TrainingError, ImportError):
            raise
        except Exception as exc:
            raise TrainingError(
                f"Failed to load model '{self.model_id}' for task '{self.task}': " f"{type(exc).__name__}: {exc}"
            ) from exc

        # Move to the target device; model stays in train mode (the default
        # after from_pretrained()).
        self.model = self.model.to(self._device)
        logger.info("Model loaded on %s (train mode, gradients enabled)", self._device)

    # ------------------------------------------------------------------
    # TrainingArguments builder
    # ------------------------------------------------------------------

    def _build_training_args(self, train_size: int | None = None) -> Any:
        """Map :class:`TrainingConfig` fields to ``transformers.TrainingArguments``.

        Args:
            train_size: Number of training samples. When provided, warmup is
                expressed in steps (exact) rather than as a ratio.

        Returns:
            A configured ``TrainingArguments`` instance.
        """
        tf = _ensure_transformers()
        cfg = self.config

        eval_strategy = "epoch" if cfg.val_every > 0 else "no"

        optim = _OPT_MAP.get(cfg.optimizer, "adamw_torch")
        lr_scheduler_type = _SCHED_MAP.get(cfg.scheduler, "cosine")

        # Mixed precision — only enable fp16 when CUDA is available.
        fp16 = False
        try:
            import torch

            fp16 = bool(cfg.amp and torch.cuda.is_available())
        except ImportError:
            pass

        hf_save_dir = Path(cfg.save_dir) / "hf_trainer"
        hf_save_dir.mkdir(parents=True, exist_ok=True)

        # Compute warmup_steps from dataset size when available so we don't
        # trigger the warmup_ratio deprecation warning in transformers ≥ v5.2.
        if train_size is not None and train_size > 0:
            steps_per_epoch = max(
                1,
                -(-train_size // (cfg.batch_size * cfg.gradient_accumulation_steps)),  # ceil
            )
            warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
        else:
            # Fallback: ratio-based (no dataset size available at config time)
            warmup_steps = 0

        # max_grad_norm=0 means disabled; HF Trainer uses -1 for that.
        hf_max_grad_norm = cfg.max_grad_norm if cfg.max_grad_norm > 0 else -1.0

        return tf["TrainingArguments"](
            output_dir=str(hf_save_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            gradient_checkpointing=cfg.gradient_checkpointing,
            learning_rate=cfg.lr,
            weight_decay=cfg.weight_decay,
            warmup_steps=warmup_steps,
            max_grad_norm=hf_max_grad_norm,
            optim=optim,
            lr_scheduler_type=lr_scheduler_type,
            fp16=fp16,
            eval_strategy=eval_strategy,
            save_strategy="epoch",
            save_total_limit=3,
            logging_strategy="epoch",
            report_to="none",
            dataloader_num_workers=cfg.num_workers,
            seed=cfg.seed,
            remove_unused_columns=False,
            # MATA manages best-checkpoint selection itself (CheckpointManager).
            # Disabling load_best_model_at_end avoids HF Trainer reloading an
            # intermediate .safetensors checkpoint whose key names may differ
            # from the live model (e.g. backbone conv_encoder vs model rename)
            # which produces a large spurious missing/unexpected keys dump.
            load_best_model_at_end=False,
        )

    # ------------------------------------------------------------------
    # Data collator
    # ------------------------------------------------------------------

    def _build_data_collator(self) -> Any:
        """Return a task-appropriate data collator for use with HF Trainer.

        - **classify**: :class:`_ClassificationCollator` — stacks images and labels.
        - **detect / segment**: :class:`_DetectionCollator` — pads images and
          converts MATA-format targets to HF DETR label dicts.

        Returns:
            A callable collator, or ``None`` to use the Trainer default.
        """
        if self.task == "classify":
            return _ClassificationCollator(processor=self.processor)
        return _DetectionCollator(processor=self.processor, task=self.task)

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------

    def _build_compute_metrics(self) -> Any:
        """Return a ``compute_metrics`` callback for the task, or ``None``.

        For classification: returns accuracy.
        For detection / segmentation: returns ``None``; full evaluation should
        be carried out via ``mata.val()`` after training.
        """
        if self.task == "classify":

            def compute_metrics(eval_pred: Any) -> dict[str, float]:
                import numpy as np

                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                accuracy = float((predictions == labels).mean())
                return {"accuracy": accuracy}

            return compute_metrics

        return None

    # ------------------------------------------------------------------
    # Backbone / layer freezing
    # ------------------------------------------------------------------

    def _freeze_backbone(self, model: Any) -> None:
        """Freeze all model parameters except the task-specific head.

        After this call only the parameters whose names contain one of the
        head substrings defined in ``_TASK_HEAD_PATTERNS`` will have
        ``requires_grad=True``.

        Task → head patterns:

        +----------+----------------------------------------------+
        | detect   | class_labels_classifier, bbox_predictor      |
        +----------+----------------------------------------------+
        | classify | classifier                                   |
        +----------+----------------------------------------------+
        | segment  | class_predictor, mask_embedder               |
        +----------+----------------------------------------------+

        Args:
            model: The ``nn.Module`` to modify in-place.
        """
        # Freeze all parameters first.
        for param in model.parameters():
            param.requires_grad = False

        head_patterns = _TASK_HEAD_PATTERNS.get(self.task, set())

        # Selectively unfreeze head parameters.
        unfrozen = 0
        for name, param in model.named_parameters():
            for pattern in head_patterns:
                if pattern in name:
                    param.requires_grad = True
                    unfrozen += 1
                    break

        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        logger.info(
            "Backbone frozen: %d params frozen, %d params trainable",
            frozen,
            unfrozen,
        )

    def _freeze_layers(self, model: Any, patterns: list[str]) -> None:
        """Freeze parameters whose names contain any of the given substrings.

        Args:
            model: The ``nn.Module`` to modify in-place.
            patterns: List of name substrings.  A parameter is frozen if its
                fully-qualified name contains **any** of the patterns.
        """
        count = 0
        for name, param in model.named_parameters():
            for pattern in patterns:
                if pattern in name:
                    param.requires_grad = False
                    count += 1
                    break
        logger.info("Froze %d parameters matching patterns: %s", count, patterns)

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: Any,
        val_dataset: Any = None,
    ) -> TrainingResult:
        """Run training via ``transformers.Trainer``.

        Steps:

        1. Load model (``_load_model_for_training``).
        2. Apply backbone / layer freezing if configured.
        3. Build ``TrainingArguments`` (``_build_training_args``).
        4. Build data collator and ``compute_metrics``.
        5. Attach a history-collection callback.
        6. Instantiate and run ``transformers.Trainer``.
        7. Extract per-epoch history from callback logs.
        8. Save MATA-format checkpoint via ``CheckpointManager``.
        9. Return populated ``TrainingResult``.

        Args:
            train_dataset: Any PyTorch ``Dataset`` returning
                ``(image, target)`` pairs.
            val_dataset: Optional validation ``Dataset``.  When provided,
                evaluation is run every ``config.val_every`` epochs.

        Returns:
            A populated :class:`~mata.training.result.TrainingResult`.

        Raises:
            TrainingError: If model loading or the training loop fails.
        """
        from mata.training.checkpoint import CheckpointManager

        # ── 1. Load model ───────────────────────────────────────────────
        # For classification, derive id2label from the dataset so the model
        # head is replaced to match the actual number of classes rather than
        # the pretrained head (e.g. 1000 ImageNet classes for ResNet-50).
        train_id2label: dict[int, str] | None = None
        if self.task == "classify" and hasattr(train_dataset, "class_names"):
            train_id2label = train_dataset.class_names  # {0: "circle", 1: "square", …}
        self._load_model_for_training(id2label=train_id2label)

        # ── 2. Freezing ─────────────────────────────────────────────────
        if self.config.freeze_backbone:
            self._freeze_backbone(self.model)
        if self.config.freeze_layers:
            self._freeze_layers(self.model, self.config.freeze_layers)

        # ── 3. TrainingArguments ────────────────────────────────────────
        training_args = self._build_training_args(train_size=len(train_dataset) if train_dataset is not None else None)

        # ── 4. Collator + metrics ────────────────────────────────────────
        data_collator = self._build_data_collator()
        compute_metrics = self._build_compute_metrics()

        # ── 5. History callback ──────────────────────────────────────────
        history_cb = _HistoryCallback()
        tf = _ensure_transformers()
        TrainerCallback = tf["TrainerCallback"]

        class _HistoryCBWrapper(TrainerCallback):
            """Thin TrainerCallback subclass delegating to _HistoryCallback."""

            def on_log(self_cb, args, state, control, logs=None, **kwargs):  # noqa: N805
                history_cb.on_log(args, state, control, logs=logs, **kwargs)

        # ── 6. Build and run Trainer ────────────────────────────────────
        trainer_kwargs: dict[str, Any] = {
            "model": self.model,
            "args": training_args,
            "train_dataset": train_dataset,
            "data_collator": data_collator,
            "compute_metrics": compute_metrics,
            "callbacks": [_HistoryCBWrapper()],
        }
        if val_dataset is not None:
            trainer_kwargs["eval_dataset"] = val_dataset

        # Mask2Former's forward() takes separate `mask_labels` / `class_labels`
        # args rather than a unified `labels` dict.  The standard HF Trainer
        # passes whatever the collator puts in `labels` verbatim, so it never
        # reaches Mask2Former's loss path.  We subclass Trainer locally to
        # unpack the list-of-dicts that our collator builds.
        TrainerBase = tf["Trainer"]
        if self.task == "segment":

            class _SegmentTrainer(TrainerBase):  # type: ignore[valid-type]
                """Unpacks collated `labels` into Mask2Former-native args."""

                def compute_loss(self_t, model, inputs, return_outputs=False, **kwargs):  # noqa: N805
                    labels = inputs.pop("labels", None)
                    if labels is not None:
                        mask_labels_list = [item.get("mask_labels") for item in labels]
                        class_labels_list = [item.get("class_labels") for item in labels]
                        outputs = model(
                            **inputs,
                            mask_labels=mask_labels_list,
                            class_labels=class_labels_list,
                        )
                    else:
                        outputs = model(**inputs)
                    loss = outputs.loss
                    if loss is None:
                        raise ValueError(
                            "Mask2Former returned no loss.  Ensure mask_labels and "
                            "class_labels are correctly formed in the collator."
                        )
                    return (loss, outputs) if return_outputs else loss

                def prediction_step(self_t, model, inputs, prediction_loss_only, ignore_keys=None):  # noqa: N805
                    """Eval step: compute only loss.

                    HF Trainer's default prediction_step (with
                    prediction_loss_only=False) tries to stack the per-image
                    ``labels`` list-of-dicts as a tensor across batches, which
                    fails silently and causes ``eval_loss`` to be dropped.
                    Override to always return (loss, None, None).
                    """
                    import torch

                    inputs = self_t._prepare_inputs(inputs)
                    labels = inputs.pop("labels", None)
                    with torch.no_grad():
                        if labels is not None:
                            mask_labels = [item.get("mask_labels") for item in labels]
                            class_labels = [item.get("class_labels") for item in labels]
                            outputs = model(
                                **inputs,
                                mask_labels=mask_labels,
                                class_labels=class_labels,
                            )
                        else:
                            outputs = model(**inputs)
                    loss = outputs.loss
                    if loss is not None:
                        loss = loss.detach()
                    return (loss, None, None)

            TrainerCls = _SegmentTrainer
        else:
            TrainerCls = TrainerBase

        try:
            trainer = TrainerCls(**trainer_kwargs)
        except Exception as exc:
            raise TrainingError(f"Failed to build HF Trainer: {type(exc).__name__}: {exc}") from exc

        resume_from = self.config.resume or None
        try:
            trainer.train(resume_from_checkpoint=resume_from)
        except Exception as exc:
            raise TrainingError(f"Training failed: {type(exc).__name__}: {exc}") from exc

        # ── 7. Normalise history keys ────────────────────────────────────
        # Primary source: the _HistoryCBWrapper callback (fires on every logged
        # step during real training).  Fallback: trainer.state.log_history
        # (always populated by HF Trainer and available after training ends).
        raw_history = history_cb.history
        if not raw_history:
            # Reconstruct from the authoritative log_history list.
            log_history = trainer.state.log_history
            for entry in (log_history if isinstance(log_history, list) else []):
                for key, value in entry.items():
                    if isinstance(value, (int, float)):
                        raw_history.setdefault(key, []).append(float(value))

        history: dict[str, list[float]] = {}
        for key, vals in raw_history.items():
            if key == "loss":
                history["train_loss"] = vals
            elif key == "eval_loss":
                history["val_loss"] = vals
            elif key.startswith("eval_"):
                history[f"val_{key[5:]}"] = vals
            else:
                history[key] = vals

        # ── 8. MATA checkpoint ───────────────────────────────────────────
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        ckpt_mgr = CheckpointManager()
        best_ckpt_path = ""
        last_ckpt_path = ""
        try:
            last_epoch = int(trainer.state.epoch or self.config.epochs)
            last_log = trainer.state.log_history[-1] if trainer.state.log_history else {}

            last_ckpt = ckpt_mgr.save(
                model=self.model,
                optimizer=trainer.optimizer,
                scheduler=None,
                epoch=last_epoch,
                metrics=last_log,
                config=self.config,
                path=save_dir / "last",
            )
            last_ckpt_path = str(last_ckpt)

            # Also export an HF-format directory loadable by mata.load()
            ckpt_mgr.export_for_inference(
                checkpoint_dir=last_ckpt,
                output_dir=save_dir / "best",
                model=self.model,
                processor=self.processor,
            )
            best_ckpt_path = str(save_dir / "best")
        except Exception as exc:
            logger.warning("Checkpoint save failed (non-fatal): %s", exc)

        # ── 9. Build result ──────────────────────────────────────────────
        epoch_val = trainer.state.epoch
        epochs_completed = int(epoch_val if isinstance(epoch_val, (int, float)) else self.config.epochs)
        final_metrics: dict[str, float] | None = None
        log_history = trainer.state.log_history
        if isinstance(log_history, list) and log_history:
            final_metrics = {k: v for k, v in log_history[-1].items() if isinstance(v, (int, float))}

        return TrainingResult(
            best_metrics=final_metrics,
            final_metrics=final_metrics,
            best_checkpoint=best_ckpt_path,
            last_checkpoint=last_ckpt_path,
            history=history,
            config=self.config,
            epochs_completed=epochs_completed,
        )
