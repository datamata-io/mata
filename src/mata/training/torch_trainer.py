"""Custom PyTorch training loop for torchvision detection models.

Supports all seven models in TorchvisionDetectAdapter.MODEL_BUILDERS:
    - Faster R-CNN (v1 / v2)
    - RetinaNet (v1 / v2)
    - FCOS
    - SSD300-VGG16
    - SSDLite320-MobileNetV3-Large

Torchvision models return a loss dict in ``.train()`` mode::

    losses = model(images, targets)
    # {"loss_classifier": ..., "loss_box_reg": ...,
    #  "loss_objectness": ..., "loss_rpn_box_reg": ...}

This engine handles that API; HuggingFace models (detect/classify/segment)
should use HFTrainingEngine instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mata.core.exceptions import TrainingError
from mata.core.logging import get_logger
from mata.training.checkpoint import CheckpointManager
from mata.training.config import TrainingConfig
from mata.training.result import TrainingResult

logger = get_logger(__name__)

# ── Model key sets (strip the "torchvision/" prefix) ──────────────────────────
_FASTERRCNN_KEYS = frozenset({"fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2"})
_RETINANET_KEYS = frozenset({"retinanet_resnet50_fpn", "retinanet_resnet50_fpn_v2"})
_FCOS_KEYS = frozenset({"fcos_resnet50_fpn"})
_SSD_KEYS = frozenset({"ssd300_vgg16", "ssdlite320_mobilenet_v3_large"})


def _get_tqdm() -> Any:
    """Return tqdm class or None if not installed."""
    try:
        from tqdm import tqdm  # type: ignore[import]

        return tqdm
    except ImportError:
        return None


def _extract_val_metric(metrics: Any) -> float:
    """Extract a scalar validation metric for best-checkpoint tracking.

    Preference order: map50 > map > fitness > 0.0.
    Handles both dict-style and object-style metrics.
    """
    if metrics is None:
        return 0.0
    if isinstance(metrics, dict):
        for key in ("map50", "map", "fitness"):
            if key in metrics and metrics[key] is not None:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    pass
        return 0.0
    for attr in ("map50", "map", "fitness"):
        val = getattr(metrics, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


def _move_targets_to_device(targets: list[dict[str, Any]], device: Any) -> list[dict[str, Any]]:
    """Move only tensor target fields to a device.

    Torchvision detection targets may include scalar metadata fields
    (e.g., ``image_id`` as ``int``) that do not implement ``.to()``.
    """
    import torch

    moved: list[dict[str, Any]] = []
    for target in targets:
        moved_target: dict[str, Any] = {}
        for key, value in target.items():
            moved_target[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        moved.append(moved_target)
    return moved


class TorchTrainingEngine:
    """Custom PyTorch training loop for torchvision detection models.

    Torchvision models return a loss dict when in ``.train()`` mode::

        losses = model(images, targets)
        # e.g. {"loss_classifier": tensor, "loss_box_reg": tensor, ...}

    This engine handles all seven models in
    :data:`~mata.adapters.torchvision_detect_adapter.TorchvisionDetectAdapter.MODEL_BUILDERS`.

    Args:
        task: Must be ``"detect"`` — torchvision only supports detection.
        model_name: Full model name including optional ``"torchvision/"`` prefix,
            e.g. ``"torchvision/fasterrcnn_resnet50_fpn"``.
        config: :class:`~mata.training.config.TrainingConfig` with all
            hyperparameters.

    Raises:
        TrainingError: If *task* is not ``"detect"`` or *model_name* is
            unrecognised.

    Example::

        from mata.training.config import TrainingConfig
        from mata.training.torch_trainer import TorchTrainingEngine

        config = TrainingConfig(
            task="detect",
            model="torchvision/fasterrcnn_resnet50_fpn",
            data="coco.yaml",
            epochs=10,
            batch_size=4,
        )
        engine = TorchTrainingEngine("detect", "torchvision/fasterrcnn_resnet50_fpn", config)
        result = engine.train(train_dataset, val_dataset)
    """

    def __init__(self, task: str, model_name: str, config: TrainingConfig) -> None:
        if task != "detect":
            raise TrainingError(
                f"TorchTrainingEngine only supports task='detect', got '{task}'. "
                "Use HFTrainingEngine for HuggingFace models (classify/segment)."
            )

        self.task = task
        self.model_name = model_name
        self.config = config
        self.ckpt_manager = CheckpointManager()

        # Strip "torchvision/" prefix for internal lookups
        self.model_key = model_name.replace("torchvision/", "").strip()

        # Validate against known models
        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        if self.model_key not in TorchvisionDetectAdapter.MODEL_BUILDERS:
            available = ", ".join(
                f"torchvision/{k}" for k in sorted(TorchvisionDetectAdapter.MODEL_BUILDERS)
            )
            raise TrainingError(
                f"Unknown torchvision model '{model_name}'. "
                f"Supported models: {available}"
            )

    # ── Device resolution ──────────────────────────────────────────────────────

    def _resolve_device(self) -> Any:
        """Resolve ``config.device`` to a ``torch.device``."""
        import torch

        device_str = self.config.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device_str)

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model_for_training(self) -> Any:
        """Load the torchvision model with pretrained weights for fine-tuning.

        Does **not** call ``.eval()`` or wrap in ``torch.no_grad()`` — the
        model is returned in training mode ready for head replacement and
        optimisation.

        Returns:
            ``nn.Module`` in training mode (default after construction).

        Raises:
            TrainingError: If torchvision is not installed or model cannot
                be loaded.
        """
        try:
            import torchvision.models.detection as detection_models
        except ImportError as exc:
            raise TrainingError(
                "torchvision is required for TorchTrainingEngine. "
                "Install with: pip install torchvision"
            ) from exc

        builder_fn = getattr(detection_models, self.model_key)

        try:
            # New torchvision API (>=0.13): weights="DEFAULT"
            model = builder_fn(weights="DEFAULT")
        except TypeError:
            # Old API fallback: pretrained=True
            model = builder_fn(pretrained=True)  # type: ignore[call-arg]

        logger.info("Loaded torchvision model '%s' with pretrained COCO weights", self.model_key)
        # Model is in train mode by default after construction
        return model

    # ── Head replacement ───────────────────────────────────────────────────────

    def _modify_head(self, model: Any, num_classes: int) -> None:
        """Replace the classification head for transfer learning.

        Handles all supported model families:

        * **Faster R-CNN** (v1/v2): replaces ``model.roi_heads.box_predictor``
          with a new ``FastRCNNPredictor``.
        * **RetinaNet** (v1/v2): replaces ``cls_logits`` Conv2d and updates
          ``num_classes`` on the classification head.
        * **FCOS**: replaces ``cls_logits`` Conv2d (anchor-free, 1 anchor/cell)
          and updates ``num_classes``.
        * **SSD300-VGG16**: replaces each Conv2d in
          ``head.classification_head.module_list``.
        * **SSDLite320-MobileNetV3**: replaces the final pointwise Conv2d
          inside each depthwise-separable Sequential block.

        Args:
            model: The loaded torchvision detection model.
            num_classes: Number of target classes (including background if
                the model counts it — Faster R-CNN does).
        """
        import torch.nn as nn

        if self.model_key in _FASTERRCNN_KEYS:
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            logger.debug(
                "Replaced Faster R-CNN head: in_features=%d, num_classes=%d",
                in_features,
                num_classes,
            )

        elif self.model_key in _RETINANET_KEYS:
            cls_head = model.head.classification_head
            in_channels = cls_head.cls_logits.in_channels
            num_anchors = cls_head.num_anchors
            cls_head.cls_logits = nn.Conv2d(
                in_channels, num_anchors * num_classes, kernel_size=3, padding=1
            )
            cls_head.num_classes = num_classes
            logger.debug(
                "Replaced RetinaNet head: in_channels=%d, num_anchors=%d, num_classes=%d",
                in_channels,
                num_anchors,
                num_classes,
            )

        elif self.model_key in _FCOS_KEYS:
            cls_head = model.head.classification_head
            # FCOS is anchor-free: cls_logits out_channels = num_classes
            in_channels = cls_head.cls_logits.in_channels
            cls_head.cls_logits = nn.Conv2d(
                in_channels, num_classes, kernel_size=3, padding=1
            )
            cls_head.num_classes = num_classes
            logger.debug(
                "Replaced FCOS head: in_channels=%d, num_classes=%d",
                in_channels,
                num_classes,
            )

        elif self.model_key in _SSD_KEYS:
            cls_head = model.head.classification_head
            # SSDClassificationHead does not expose num_classes as a plain attribute.
            # Infer it from the paired regression head: each prediction Conv2d has
            # out_channels = num_anchors * 4 (regression) or num_anchors * C (cls).
            def _last_pred_conv(m: Any) -> Any:
                if isinstance(m, nn.Conv2d):
                    return m
                for layer in reversed(list(m.children())):
                    if isinstance(layer, nn.Conv2d):
                        return layer
                return None

            _reg_conv = _last_pred_conv(next(iter(model.head.regression_head.module_list), None))
            _cls_conv = _last_pred_conv(next(iter(cls_head.module_list), None))
            if _reg_conv is not None and _cls_conv is not None:
                old_num_classes = _cls_conv.out_channels // max(_reg_conv.out_channels // 4, 1)
            else:
                old_num_classes = getattr(cls_head, "num_classes", 91)
            new_modules: list[Any] = []

            for module in cls_head.module_list:
                if isinstance(module, nn.Conv2d):
                    # SSD300: each feature-map level has a direct Conv2d
                    in_ch = module.in_channels
                    num_anchors_lvl = module.out_channels // old_num_classes
                    new_modules.append(
                        nn.Conv2d(in_ch, num_anchors_lvl * num_classes, kernel_size=3, padding=1)
                    )
                elif isinstance(module, nn.Sequential):
                    # SSDLite: Sequential[depthwise_conv, ..., pointwise_conv]
                    layers = list(module.children())
                    last_conv = layers[-1]
                    in_ch_last = last_conv.in_channels
                    num_anchors_lvl = last_conv.out_channels // old_num_classes
                    layers[-1] = nn.Conv2d(
                        in_ch_last, num_anchors_lvl * num_classes, kernel_size=1
                    )
                    new_modules.append(nn.Sequential(*layers))
                else:
                    # Unknown module type — leave unchanged (defensive)
                    new_modules.append(module)

            cls_head.module_list = nn.ModuleList(new_modules)
            cls_head.num_columns = num_classes  # used by SSDScoringHead.forward() reshape
            cls_head.num_classes = num_classes
            logger.debug(
                "Replaced SSD/SSDLite head for model '%s': num_classes=%d",
                self.model_key,
                num_classes,
            )

        else:
            logger.warning(
                "No head replacement implemented for model '%s'; "
                "training will continue with the original head (COCO classes).",
                self.model_key,
            )

    # ── Backbone freezing ──────────────────────────────────────────────────────

    def _freeze_backbone(self, model: Any) -> None:
        """Freeze all parameters in ``model.backbone``.

        Args:
            model: The torchvision detection model.
        """
        if not hasattr(model, "backbone"):
            logger.warning(
                "Model '%s' has no 'backbone' attribute; skipping backbone freeze.",
                self.model_key,
            )
            return

        frozen = 0
        for param in model.backbone.parameters():
            param.requires_grad = False
            frozen += 1

        logger.info("Froze %d backbone parameter tensors in '%s'.", frozen, self.model_key)

    def _apply_freeze_layers(self, model: Any) -> None:
        """Freeze parameters whose names contain any pattern in ``config.freeze_layers``.

        Args:
            model: The torchvision detection model.
        """
        if not self.config.freeze_layers:
            return

        frozen = 0
        for name, param in model.named_parameters():
            for pattern in self.config.freeze_layers:
                if pattern in name:
                    param.requires_grad = False
                    frozen += 1
                    break

        logger.info(
            "Froze %d parameters matching freeze_layers patterns: %s",
            frozen,
            self.config.freeze_layers,
        )

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def _build_optimizer(self, model: Any) -> Any:
        """Build an optimizer from ``config.optimizer``.

        Only parameters with ``requires_grad=True`` are included.

        Args:
            model: The torchvision detection model (after head modification
                and optional freezing).

        Returns:
            A ``torch.optim.Optimizer`` instance.

        Raises:
            TrainingError: If no trainable parameters are found, or if the
                optimizer name is unrecognised.
        """
        import torch.optim as optim

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise TrainingError(
                "No trainable parameters found. "
                "Check freeze_backbone and freeze_layers settings — "
                "every parameter appears to be frozen."
            )

        opt_name = self.config.optimizer.lower()
        kwargs = dict(lr=self.config.lr, weight_decay=self.config.weight_decay)

        if opt_name == "adamw":
            return optim.AdamW(trainable, **kwargs)
        elif opt_name == "adam":
            return optim.Adam(trainable, **kwargs)
        elif opt_name == "sgd":
            return optim.SGD(trainable, momentum=0.9, **kwargs)
        else:
            raise TrainingError(
                f"Unknown optimizer '{opt_name}'. Supported: adamw, adam, sgd."
            )

    # ── LR Scheduler ──────────────────────────────────────────────────────────

    def _build_scheduler(self, optimizer: Any) -> Any:
        """Build a learning-rate scheduler from ``config.scheduler``.

        Args:
            optimizer: The optimizer to wrap.

        Returns:
            A ``torch.optim.lr_scheduler._LRScheduler`` instance.

        Raises:
            TrainingError: If the scheduler name is unrecognised.
        """
        import torch.optim.lr_scheduler as lr_sched

        sched_name = self.config.scheduler.lower()
        epochs = self.config.epochs

        if sched_name == "cosine":
            return lr_sched.CosineAnnealingLR(optimizer, T_max=epochs)

        elif sched_name == "linear":
            def _linear_lambda(epoch: int) -> float:
                if epochs <= 1:
                    return 1.0
                return max(0.0, 1.0 - epoch / epochs)

            return lr_sched.LambdaLR(optimizer, lr_lambda=_linear_lambda)

        elif sched_name == "step":
            step_size = max(1, epochs // 3)
            return lr_sched.StepLR(optimizer, step_size=step_size, gamma=0.1)

        elif sched_name == "none":
            return lr_sched.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        else:
            raise TrainingError(
                f"Unknown scheduler '{sched_name}'. Supported: cosine, linear, step, none."
            )

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate(
        self,
        model: Any,
        val_dataset: Any,
        device: Any,
        epoch: int,
    ) -> dict[str, float] | None:
        """Run a validation pass using mata.val() integration.

        Temporarily sets the model to eval mode, saves a temporary checkpoint
        so that ``mata.val()`` can load it, runs eval, then restores training
        mode.

        If the temporary checkpoint cannot be created or ``mata.val()`` fails,
        the error is logged as a warning and ``None`` is returned so training
        can continue.

        Args:
            model: The torchvision detection model (in training mode).
            val_dataset: Validation dataset instance.
            device: ``torch.device`` used for training.
            epoch: Current epoch index (0-based), used for logging only.

        Returns:
            Dict with at least ``"map50"`` key, or ``None`` on failure.
        """
        import tempfile

        import torch

        model.eval()
        try:
            # ── Attempt mata.val() integration ────────────────────────────────
            try:
                import mata
                from mata.adapters.torchvision_detect_adapter import (
                    TorchvisionDetectAdapter,
                )

                with tempfile.TemporaryDirectory(prefix="mata_val_") as tmp_dir:
                    tmp_weights = str(Path(tmp_dir) / "model_state.pth")
                    torch.save(model.state_dict(), tmp_weights)

                    adapter = TorchvisionDetectAdapter(
                        model_name=self.model_name,
                        device=str(device),
                        weights=tmp_weights,
                    )
                    val_result = mata.val(
                        task=self.task,
                        adapter=adapter,
                        data=val_dataset,
                    )

                if val_result is not None:
                    map50 = _extract_val_metric(val_result)
                    logger.info(
                        "Epoch %d validation — mAP50: %.4f", epoch + 1, map50
                    )
                    return {"map50": map50}

            except Exception as val_err:  # noqa: BLE001
                logger.debug(
                    "mata.val() integration unavailable (%s); "
                    "falling back to loss-based validation.",
                    val_err,
                )

            # ── Fallback: validation loss ──────────────────────────────────────
            from torch.utils.data import DataLoader

            from mata.training.datasets.collators import detection_collate_fn

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=detection_collate_fn,
                num_workers=self.config.num_workers,
                pin_memory=(device.type == "cuda"),
            )

            # Torchvision models only produce loss dicts in .train() mode.
            # Switch to train() momentarily to compute validation loss,
            # then restore eval() afterwards.
            model.train()
            total_loss = 0.0
            num_batches = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    targets = _move_targets_to_device(targets, device)
                    loss_dict = model(images, targets)
                    batch_loss = sum(loss_dict.values()).item()
                    total_loss += batch_loss
                    num_batches += 1

            avg_val_loss = total_loss / max(num_batches, 1)
            logger.info(
                "Epoch %d validation loss: %.4f", epoch + 1, avg_val_loss
            )
            return {"val_loss": avg_val_loss}

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Validation failed at epoch %d: %s — skipping.", epoch + 1, exc
            )
            return None
        finally:
            model.eval()

    # ── Main training loop ─────────────────────────────────────────────────────

    def train(
        self,
        train_dataset: Any,
        val_dataset: Any | None = None,
    ) -> TrainingResult:
        """Run the full training loop.

        Per-epoch workflow:

        1. ``model.train()``
        2. For each batch:

           * Move images and targets to device.
           * Forward: ``losses = model(images, targets)``
           * Sum losses: ``total_loss = sum(losses.values())``
           * Backward + optimizer step (with AMP on CUDA).
        3. Validate if ``val_every`` interval and *val_dataset* provided.
        4. Save periodic checkpoint if ``save_every`` interval.
        5. Save best checkpoint if validation improves.
        6. Early-stopping check.
        7. LR scheduler step.

        After the loop, the *last* checkpoint is always saved.

        Args:
            train_dataset: A ``torch.utils.data.Dataset`` returning
                ``(image_tensor, target_dict)`` tuples.  Target dicts must
                follow the torchvision detection format:
                ``{"boxes": Tensor[N,4], "labels": Tensor[N], ...}``.
            val_dataset: Optional validation dataset with the same format.
                When provided and ``config.val_every > 0``, validation runs
                every N epochs.

        Returns:
            :class:`~mata.training.result.TrainingResult` with populated
            ``history``, ``best_checkpoint``, ``last_checkpoint``, and
            optionally ``best_metrics`` / ``final_metrics``.

        Raises:
            TrainingError: If data loading fails or the training loop
                encounters an unrecoverable error.
        """
        import torch
        from torch.utils.data import DataLoader

        from mata.training.datasets.collators import detection_collate_fn

        tqdm_cls = _get_tqdm()
        device = self._resolve_device()
        use_amp = self.config.amp and device.type == "cuda"

        # ── Load model ────────────────────────────────────────────────────────
        model = self._load_model_for_training()

        # ── Head replacement ──────────────────────────────────────────────────
        num_classes: int | None = None
        if hasattr(train_dataset, "num_classes"):
            num_classes = int(train_dataset.num_classes)
        elif hasattr(train_dataset, "class_names"):
            num_classes = len(train_dataset.class_names)

        if num_classes is not None:
            self._modify_head(model, num_classes)

        # ── Freeze layers ─────────────────────────────────────────────────────
        if self.config.freeze_backbone:
            self._freeze_backbone(model)
        self._apply_freeze_layers(model)

        # ── Move to device ────────────────────────────────────────────────────
        model = model.to(device)

        # ── Resume from checkpoint ────────────────────────────────────────────
        start_epoch = 0
        best_metric_val = float("-inf")
        history: dict[str, list[float]] = {"train_loss": [], "lr": []}

        resume_opt_state: dict[str, Any] = {}
        if self.config.resume:
            try:
                resume_data = self.ckpt_manager.load(self.config.resume)
                model.load_state_dict(resume_data["model_state"])
                start_epoch = (
                    resume_data.get("training_state", {}).get("epoch", 0) + 1
                )
                best_metric_val = (
                    resume_data.get("training_state", {}).get("best_metric")
                    or float("-inf")
                )
                history = resume_data.get("training_state", {}).get(
                    "history", history
                )
                # Ensure mandatory keys survive resume from checkpoints that
                # saved an empty history dict.
                history.setdefault("train_loss", [])
                history.setdefault("lr", [])
                resume_opt_state = resume_data.get("optimizer_state", {})
                logger.info(
                    "Resumed from checkpoint '%s' at epoch %d.",
                    self.config.resume,
                    start_epoch,
                )
            except Exception as exc:
                raise TrainingError(
                    f"Failed to resume from checkpoint '{self.config.resume}': {exc}"
                ) from exc

        # ── Optimizer & scheduler ─────────────────────────────────────────────
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)

        if resume_opt_state:
            if "optimizer" in resume_opt_state:
                optimizer.load_state_dict(resume_opt_state["optimizer"])
            if "scheduler" in resume_opt_state:
                scheduler.load_state_dict(resume_opt_state["scheduler"])

        # ── AMP scaler ────────────────────────────────────────────────────────
        scaler: Any = None
        if use_amp:
            scaler = torch.amp.GradScaler("cuda")

        # ── DataLoader ────────────────────────────────────────────────────────
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=detection_collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # ── Checkpoint save metadata — include engine/model_source so that
        # mata.load() can recognise this checkpoint as a torchvision model.
        import dataclasses as _dc
        _ckpt_meta: dict[str, Any] = _dc.asdict(self.config)
        _ckpt_meta["engine"] = "torchvision"
        _ckpt_meta["model_source"] = self.model_name

        # ── Output directory ──────────────────────────────────────────────────
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # ── State tracking ────────────────────────────────────────────────────
        best_checkpoint = ""
        last_checkpoint = ""
        best_metrics: Any = None
        final_metrics: Any = None
        no_improve_count = 0
        epochs_completed = start_epoch
        last_val_metrics: Any = None

        logger.info(
            "TorchTrainingEngine starting: model=%s, epochs=%d, device=%s, amp=%s",
            self.model_key,
            self.config.epochs,
            device,
            use_amp,
        )

        # ── Training loop ─────────────────────────────────────────────────────
        for epoch in range(start_epoch, self.config.epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            # Optionally wrap loader with tqdm
            if tqdm_cls is not None and self.config.verbose:
                loader_iter = tqdm_cls(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{self.config.epochs}",
                    leave=False,
                )
            else:
                loader_iter = train_loader

            for images, targets in loader_iter:
                # Move data to device
                images = [img.to(device) for img in images]
                targets = _move_targets_to_device(targets, device)

                optimizer.zero_grad()

                if use_amp and scaler is not None:
                    with torch.amp.autocast("cuda"):
                        loss_dict: dict[str, Any] = model(images, targets)
                        total_loss = sum(loss_dict.values())
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_dict = model(images, targets)
                    total_loss = sum(loss_dict.values())
                    total_loss.backward()
                    optimizer.step()

                batch_loss = float(total_loss.item())
                epoch_loss += batch_loss
                num_batches += 1

                # Update tqdm postfix if available
                if tqdm_cls is not None and hasattr(loader_iter, "set_postfix"):
                    loader_iter.set_postfix(loss=f"{batch_loss:.4f}")

            avg_loss = epoch_loss / max(num_batches, 1)
            current_lr = float(optimizer.param_groups[0]["lr"])

            history["train_loss"].append(avg_loss)
            history["lr"].append(current_lr)

            if self.config.verbose:
                logger.info(
                    "Epoch %d/%d — train_loss: %.4f, lr: %.2e",
                    epoch + 1,
                    self.config.epochs,
                    avg_loss,
                    current_lr,
                )

            # ── Validation ────────────────────────────────────────────────────
            val_metrics: Any = None
            if val_dataset is not None and (epoch + 1) % self.config.val_every == 0:
                val_metrics = self._validate(model, val_dataset, device, epoch)
                last_val_metrics = val_metrics

                if val_metrics is not None:
                    # Track val_map50 or val_loss in history
                    if "map50" in val_metrics:
                        history.setdefault("val_map50", []).append(
                            float(val_metrics["map50"])
                        )
                    if "val_loss" in val_metrics:
                        history.setdefault("val_loss", []).append(
                            float(val_metrics["val_loss"])
                        )

                    metric_val = _extract_val_metric(val_metrics)

                    if metric_val > best_metric_val:
                        best_metric_val = metric_val
                        best_metrics = val_metrics
                        best_ckpt_path = save_dir / "best"
                        self.ckpt_manager.save(
                            model, optimizer, scheduler, epoch,
                            val_metrics, _ckpt_meta, best_ckpt_path,
                        )
                        best_checkpoint = str(best_ckpt_path)
                        no_improve_count = 0
                        logger.info(
                            "New best metric: %.4f — checkpoint saved to '%s'.",
                            metric_val,
                            best_ckpt_path,
                        )
                    else:
                        no_improve_count += 1

            # ── Periodic checkpoint ───────────────────────────────────────────
            if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                periodic_path = save_dir / f"epoch{epoch + 1}"
                self.ckpt_manager.save(
                    model, optimizer, scheduler, epoch,
                    val_metrics, _ckpt_meta, periodic_path,
                )
                logger.debug("Periodic checkpoint saved to '%s'.", periodic_path)

            # ── LR scheduler step ─────────────────────────────────────────────
            scheduler.step()

            epochs_completed = epoch + 1

            # ── Early stopping ────────────────────────────────────────────────
            if self.config.patience > 0 and no_improve_count >= self.config.patience:
                logger.info(
                    "Early stopping triggered at epoch %d "
                    "(no improvement for %d epochs).",
                    epoch + 1,
                    self.config.patience,
                )
                break

        # ── Save last checkpoint ──────────────────────────────────────────────
        last_ckpt_path = save_dir / "last"
        self.ckpt_manager.save(
            model, optimizer, scheduler,
            epochs_completed - 1,
            last_val_metrics, _ckpt_meta, last_ckpt_path,
        )
        last_checkpoint = str(last_ckpt_path)

        # If validation was never run, best == last
        if not best_checkpoint:
            best_checkpoint = last_checkpoint

        final_metrics = last_val_metrics

        return TrainingResult(
            best_metrics=best_metrics,
            final_metrics=final_metrics,
            best_checkpoint=best_checkpoint,
            last_checkpoint=last_checkpoint,
            history=history,
            config=self.config,
            epochs_completed=epochs_completed,
        )
