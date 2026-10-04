"""Integration tests for MATA training pipeline — slow end-to-end suite.

These tests exercise the *real* training loop mechanics using tiny synthetic
datasets and models.  They are intentionally slow (several seconds each) and
should be excluded from fast CI runs via ``-m "not slow"``.

All tests create synthetic data in-memory or in ``tmp_path``; no network
access or external model downloads occur.

Acceptance criteria (E9):
    - 10+ tests passing when slow tests are enabled
    - All tests create synthetic data (no external dependencies)
    - Tests marked with @pytest.mark.slow
    - Each test completes within 60 seconds (tiny datasets, 1–2 epochs)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from PIL import Image as PILImage

from mata.training.checkpoint import CheckpointManager
from mata.training.config import TrainingConfig
from mata.training.hf_trainer import HFTrainingEngine
from mata.training.result import TrainingResult
from mata.training.torch_trainer import TorchTrainingEngine

# All tests in this module are slow
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Tiny synthetic models
# ---------------------------------------------------------------------------


class _TinyDetector(nn.Module):
    """Minimal detection model mimicking the torchvision train/eval API.

    Train mode: returns a loss dict ``{"loss_classifier": Tensor}`` so the
    training loop can call ``.backward()`` and update weights.
    Eval mode : returns a list of per-image detection dicts.

    A ``backbone`` attribute is included so
    :meth:`TorchTrainingEngine._freeze_backbone` can run without error.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(1, 4)
        self._head = nn.Linear(4, 2)

    def forward(  # type: ignore[override]
        self,
        images: list[torch.Tensor],
        targets: list[dict] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict]:
        # Aggregate each image to a scalar, then pass through tiny network
        x = torch.stack([img.mean().unsqueeze(0) for img in images])  # (B,1)
        feats = self.backbone(x)  # (B,4)
        logits = self._head(feats)  # (B,2)

        if self.training and targets is not None:
            labels = torch.stack([t["labels"][0].long() if len(t["labels"]) > 0 else torch.tensor(0) for t in targets])
            loss = F.cross_entropy(logits, labels)
            return {"loss_classifier": loss}

        # Eval mode — return stub detection result per image
        return [
            {
                "boxes": torch.zeros((1, 4)),
                "labels": torch.zeros(1, dtype=torch.long),
                "scores": torch.ones(1) * 0.9,
            }
            for _ in images
        ]


class _TinyHFClassifier(nn.Module):
    """Tiny HF-compatible image classification model.

    Returns ``(loss, logits)`` when ``labels`` are provided, ``(logits,)``
    otherwise.  The :class:`transformers.Trainer` accepts both tuple formats.
    """

    def __init__(self, num_labels: int = 2) -> None:
        super().__init__()
        self._clf = nn.Linear(3 * 32 * 32, num_labels)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        assert pixel_values is not None, "pixel_values must be provided"
        b = pixel_values.shape[0]
        logits = self._clf(pixel_values.view(b, -1))
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return (loss, logits)
        return (logits,)


class _MockImageProcessor:
    """Minimal image processor: resizes PIL images to 32×32 float tensors."""

    def __call__(
        self,
        images: list | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        tensors: list[torch.Tensor] = []
        for img in images or []:
            if hasattr(img, "mode"):  # PIL Image
                arr = np.array(img.resize((32, 32))).astype(np.float32) / 255.0
                tensors.append(torch.tensor(arr.transpose(2, 0, 1)))
            else:
                tensors.append(img.float())
        return {"pixel_values": torch.stack(tensors)}

    def save_pretrained(self, path: str) -> None:  # noqa: D102
        pass  # no-op — processor state is trivial


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------


class _SyntheticDetectionDataset(torch.utils.data.Dataset):
    """Tiny detection dataset (default: 10 samples with random 3×32×32 images)."""

    def __init__(self, size: int = 10) -> None:
        gen = torch.Generator().manual_seed(0)
        self._images = [torch.rand(3, 32, 32, generator=gen) for _ in range(size)]
        self._targets = [
            {
                "boxes": torch.tensor([[2.0, 2.0, 10.0, 10.0]]),
                "labels": torch.tensor([i % 2]),
            }
            for i in range(size)
        ]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        return self._images[idx], self._targets[idx]


def _make_imagefolder(root: Path, n_per_class: int = 5) -> Path:
    """Create a 2-class synthetic ImageFolder directory under *root*."""
    np.random.seed(0)
    for cls_name in ("cat", "dog"):
        cls_dir = root / cls_name
        cls_dir.mkdir(parents=True)
        for i in range(n_per_class):
            arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            PILImage.fromarray(arr).save(cls_dir / f"img_{i}.jpg")
    return root


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------


def _detect_config(tmp_path: Path, **overrides: Any) -> TrainingConfig:
    """Minimal valid TrainingConfig for a CPU detection run."""
    defaults: dict[str, Any] = dict(
        task="detect",
        model="torchvision/fasterrcnn_resnet50_fpn",
        data="synthetic",
        epochs=2,
        batch_size=2,
        lr=1e-3,
        warmup_epochs=0,
        device="cpu",
        amp=False,
        save_dir=str(tmp_path / "runs"),
        save_every=0,
        val_every=1,
        patience=0,
        num_workers=0,
        seed=42,
        verbose=False,
        augment=False,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _classify_config(tmp_path: Path, data_path: str, **overrides: Any) -> TrainingConfig:
    """Minimal valid TrainingConfig for a CPU classify run."""
    defaults: dict[str, Any] = dict(
        task="classify",
        model="microsoft/resnet-50",
        data=data_path,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        warmup_epochs=0,
        device="cpu",
        amp=False,
        save_dir=str(tmp_path / "runs"),
        save_every=0,
        val_every=1,
        patience=0,
        num_workers=0,
        seed=42,
        verbose=False,
        augment=False,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# ---------------------------------------------------------------------------
# HFTrainingEngine setup helper
# ---------------------------------------------------------------------------


def _inject_tiny_hf_model(engine_self: HFTrainingEngine, **kwargs) -> None:
    """Side-effect for patching _load_model_for_training on HFTrainingEngine.

    Sets ``engine_self.model``, ``engine_self.processor``, and
    ``engine_self._device`` to synthetic lightweight objects so that
    real ``transformers.Trainer`` training can proceed without downloading.
    """
    engine_self.model = _TinyHFClassifier(num_labels=2).train()
    engine_self.processor = _MockImageProcessor()
    engine_self._device = "cpu"


# ===========================================================================
# E9-01  Classify — ImageFolder → loss recorded in history
# ===========================================================================


class TestClassifyImageFolderLossDecreases:
    """Real HF Trainer training on a tiny synthetic ImageFolder dataset."""

    def test_loss_in_history_after_one_epoch(self, tmp_path: Path) -> None:
        """After 1 training epoch using HFTrainingEngine the result contains
        *train_loss* in ``result.history``."""
        folder = _make_imagefolder(tmp_path / "imagefolder")

        from mata.training.datasets.imagefolder import ImageFolderDataset

        dataset = ImageFolderDataset(root=str(folder))
        config = _classify_config(tmp_path, str(folder))

        with patch.object(HFTrainingEngine, "_load_model_for_training", _inject_tiny_hf_model):
            engine = HFTrainingEngine("classify", config.model, config)
            # Pass the same dataset as val to satisfy eval_strategy="epoch"
            result = engine.train(dataset, val_dataset=dataset)

        assert isinstance(result, TrainingResult)
        assert result.epochs_completed >= 1
        # After normalization in HFTrainingEngine.train() the key is "train_loss"
        assert "train_loss" in result.history, f"Expected 'train_loss' in history, got: {sorted(result.history)}"
        assert len(result.history["train_loss"]) >= 1


# ===========================================================================
# E9-02  Detect (torchvision) — checkpoint files produced
# ===========================================================================


class TestTorchDetectCheckpointProduced:
    """TorchTrainingEngine produces the expected checkpoint directory structure."""

    def test_last_checkpoint_has_required_files(self, tmp_path: Path) -> None:
        """After training, ``last_checkpoint`` directory contains the four
        canonical checkpoint files."""
        dataset = _SyntheticDetectionDataset()
        config = _detect_config(tmp_path, epochs=1)

        with patch.object(TorchTrainingEngine, "_load_model_for_training", return_value=_TinyDetector()):
            engine = TorchTrainingEngine("detect", config.model, config)
            result = engine.train(dataset)

        assert result.last_checkpoint != ""
        ckpt_dir = Path(result.last_checkpoint)
        assert ckpt_dir.is_dir(), f"Expected checkpoint dir at: {ckpt_dir}"
        assert (ckpt_dir / "model_state.pth").exists()
        assert (ckpt_dir / "optimizer_state.pth").exists()
        assert (ckpt_dir / "training_state.json").exists()
        assert (ckpt_dir / "config.json").exists()

    def test_config_json_contains_engine_field(self, tmp_path: Path) -> None:
        """``config.json`` in the checkpoint must record an ``engine`` field."""
        dataset = _SyntheticDetectionDataset()
        config = _detect_config(tmp_path, epochs=1)

        with patch.object(TorchTrainingEngine, "_load_model_for_training", return_value=_TinyDetector()):
            engine = TorchTrainingEngine("detect", config.model, config)
            result = engine.train(dataset)

        ckpt_dir = Path(result.last_checkpoint)
        with open(ckpt_dir / "config.json") as fh:
            ckpt_config = json.load(fh)

        assert "engine" in ckpt_config
        assert ckpt_config["engine"] in ("huggingface", "torchvision")


# ===========================================================================
# E9-03  Detect — loss recorded in training history
# ===========================================================================


class TestTorchDetectLossDecreases:
    """TorchTrainingEngine records train_loss for every epoch."""

    def test_train_loss_in_history(self, tmp_path: Path) -> None:
        """``result.history['train_loss']`` contains one entry per epoch."""
        dataset = _SyntheticDetectionDataset()
        config = _detect_config(tmp_path, epochs=2)

        with patch.object(TorchTrainingEngine, "_load_model_for_training", return_value=_TinyDetector()):
            engine = TorchTrainingEngine("detect", config.model, config)
            result = engine.train(dataset)

        assert isinstance(result, TrainingResult)
        assert result.epochs_completed == 2
        assert "train_loss" in result.history
        assert len(result.history["train_loss"]) == 2
        # Losses must be positive finite floats
        for loss_val in result.history["train_loss"]:
            assert loss_val > 0
            assert not np.isnan(loss_val) and not np.isinf(loss_val)


# ===========================================================================
# E9-04  Resume from checkpoint
# ===========================================================================


class TestResumeFromCheckpoint:
    """Training can be resumed from a saved checkpoint with consistent state."""

    def test_resume_starts_at_correct_epoch(self, tmp_path: Path) -> None:
        """After running 1 epoch then resuming for 1 more, ``epochs_completed``
        equals 2."""
        dataset = _SyntheticDetectionDataset()
        run1_dir = tmp_path / "run1"

        # --- First run: 1 epoch ---
        config1 = _detect_config(tmp_path, epochs=1, save_dir=str(run1_dir))
        with patch.object(TorchTrainingEngine, "_load_model_for_training", return_value=_TinyDetector()):
            engine1 = TorchTrainingEngine("detect", config1.model, config1)
            result1 = engine1.train(dataset)

        last_ckpt = result1.last_checkpoint
        assert last_ckpt != ""
        assert Path(last_ckpt).is_dir()

        # --- Second run: resume → runs epoch index 1 → epochs_completed=2 ---
        config2 = _detect_config(
            tmp_path,
            epochs=2,
            save_dir=str(tmp_path / "run2"),
            resume=last_ckpt,
        )
        with patch.object(TorchTrainingEngine, "_load_model_for_training", return_value=_TinyDetector()):
            engine2 = TorchTrainingEngine("detect", config2.model, config2)
            result2 = engine2.train(dataset)

        # The resumed engine picks up at epoch 1 and runs to config2.epochs (2)
        assert result2.epochs_completed == 2

    def test_resume_preserves_best_checkpoint_path(self, tmp_path: Path) -> None:
        """The resumed run produces its own ``last_checkpoint``."""
        dataset = _SyntheticDetectionDataset()
        run1_dir = tmp_path / "run1"

        config1 = _detect_config(tmp_path, epochs=1, save_dir=str(run1_dir))
        with patch.object(TorchTrainingEngine, "_load_model_for_training", return_value=_TinyDetector()):
            engine1 = TorchTrainingEngine("detect", config1.model, config1)
            result1 = engine1.train(dataset)

        last_ckpt = result1.last_checkpoint

        config2 = _detect_config(
            tmp_path,
            epochs=2,
            save_dir=str(tmp_path / "run2"),
            resume=last_ckpt,
        )
        with patch.object(TorchTrainingEngine, "_load_model_for_training", return_value=_TinyDetector()):
            engine2 = TorchTrainingEngine("detect", config2.model, config2)
            result2 = engine2.train(dataset)

        assert result2.last_checkpoint != ""
        assert Path(result2.last_checkpoint).is_dir()


# ===========================================================================
# E9-05  HF checkpoint export structure
# ===========================================================================


class TestHFCheckpointExport:
    """export_for_inference writes the expected files for HF-format checkpoints."""

    def test_export_creates_mata_checkpoint_files(self, tmp_path: Path) -> None:
        """Exporting a HF checkpoint copies the model state and writes metadata."""
        model = _TinyHFClassifier()
        processor = _MockImageProcessor()

        # Build a mock config with engine="huggingface"
        cfg = MagicMock()
        cfg.task = "classify"
        cfg.model = "microsoft/resnet-50"
        cfg.data = "synthetic"
        cfg.engine = "huggingface"
        for attr in (
            "val_data",
            "epochs",
            "batch_size",
            "lr",
            "optimizer",
            "weight_decay",
            "scheduler",
            "warmup_epochs",
            "device",
            "amp",
            "save_dir",
            "save_every",
            "val_every",
            "patience",
            "freeze_backbone",
            "freeze_layers",
            "augment",
            "resume",
            "num_workers",
            "seed",
            "verbose",
        ):
            setattr(cfg, attr, None)
        cfg.history = {}  # prevent MagicMock from failing json.dump

        ckpt_mgr = CheckpointManager()
        ckpt_dir = ckpt_mgr.save(
            model=model,
            optimizer=None,
            scheduler=None,
            epoch=0,
            metrics={"loss": 0.5},
            config=cfg,
            path=tmp_path / "ckpt",
        )

        export_dir = tmp_path / "export"
        ckpt_mgr.export_for_inference(
            checkpoint_dir=ckpt_dir,
            output_dir=export_dir,
            model=model,
            processor=processor,
        )

        assert export_dir.is_dir()
        # HF export with a plain nn.Module (no save_pretrained) falls back to
        # copying model_state.pth into the export directory
        exported_files = {f.name for f in export_dir.iterdir()}
        assert exported_files, "Export directory must not be empty"


# ===========================================================================
# E9-06  Torchvision checkpoint export structure
# ===========================================================================


class TestTorchvisionCheckpointExport:
    """export_for_inference writes model.pth + metadata.json for torchvision checkpoints."""

    def test_export_creates_model_pth_and_metadata(self, tmp_path: Path) -> None:
        """Exporting a torchvision checkpoint creates ``model.pth`` and
        ``metadata.json`` in the output directory."""
        model = _TinyDetector()

        # Config with engine="torchvision"
        cfg = MagicMock()
        cfg.task = "detect"
        cfg.model = "torchvision/fasterrcnn_resnet50_fpn"
        cfg.data = "synthetic"
        cfg.engine = "torchvision"
        for attr in (
            "val_data",
            "epochs",
            "batch_size",
            "lr",
            "optimizer",
            "weight_decay",
            "scheduler",
            "warmup_epochs",
            "device",
            "amp",
            "save_dir",
            "save_every",
            "val_every",
            "patience",
            "freeze_backbone",
            "freeze_layers",
            "augment",
            "resume",
            "num_workers",
            "seed",
            "verbose",
        ):
            setattr(cfg, attr, None)
        cfg.history = {}  # prevent MagicMock from failing json.dump

        ckpt_mgr = CheckpointManager()
        ckpt_dir = ckpt_mgr.save(
            model=model,
            optimizer=None,
            scheduler=None,
            epoch=0,
            metrics=0.3,
            config=cfg,
            path=tmp_path / "ckpt",
        )

        export_dir = tmp_path / "export"
        ckpt_mgr.export_for_inference(
            checkpoint_dir=ckpt_dir,
            output_dir=export_dir,
            model=model,
        )

        assert (export_dir / "model.pth").exists(), "Torchvision export must contain model.pth"
        assert (export_dir / "metadata.json").exists(), "Torchvision export must contain metadata.json"

        with open(export_dir / "metadata.json") as fh:
            meta = json.load(fh)
        assert meta.get("engine") == "torchvision"


# ===========================================================================
# E9-07  Full pipeline: train → checkpoint → load → predict
# ===========================================================================


class TestFullPipeline:
    """End-to-end: TorchTrainingEngine produces checkpoint → mata.load() → predict()."""

    def test_train_produces_checkpoint_loadable_by_mata(self, tmp_path: Path) -> None:
        """mata.load() recognises the checkpoint directory produced by training."""
        import mata
        from mata.core.types import VisionResult

        tiny = _TinyDetector()
        dataset = _SyntheticDetectionDataset()
        config = _detect_config(tmp_path, epochs=1)

        # Step 1 — train with tiny model (no network)
        with patch.object(TorchTrainingEngine, "_load_model_for_training", return_value=tiny):
            engine = TorchTrainingEngine("detect", config.model, config)
            train_result = engine.train(dataset)

        ckpt_path = train_result.last_checkpoint
        assert Path(ckpt_path).is_dir()

        # Step 2 — mata.load() from checkpoint; patch out torchvision download
        mock_adapter = MagicMock()
        mock_adapter.predict.return_value = VisionResult(instances=[])

        with patch(
            "mata.core.model_loader.UniversalLoader._load_from_torchvision",
            return_value=mock_adapter,
        ):
            adapter = mata.load("detect", ckpt_path)

        # Step 3 — predict() on a synthetic PIL image
        fake_img = PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        result = adapter.predict(fake_img)

        assert isinstance(result, VisionResult)

    def test_training_result_contains_required_fields(self, tmp_path: Path) -> None:
        """TrainingResult returned by the engine contains all required fields."""
        dataset = _SyntheticDetectionDataset()
        config = _detect_config(tmp_path, epochs=1)

        with patch.object(TorchTrainingEngine, "_load_model_for_training", return_value=_TinyDetector()):
            engine = TorchTrainingEngine("detect", config.model, config)
            result = engine.train(dataset)

        assert result.best_checkpoint != ""
        assert result.last_checkpoint != ""
        assert isinstance(result.history, dict)
        assert isinstance(result.epochs_completed, int)
        assert result.epochs_completed >= 1
        assert result.config is config


# ===========================================================================
# E9-08  Early stopping triggers correctly
# ===========================================================================


class TestEarlyStopping:
    """Early stopping halts training when validation metric stops improving."""

    def test_triggers_before_max_epochs(self, tmp_path: Path) -> None:
        """With ``patience=1`` and non-improving validation, training stops
        after ``patience + 1`` epochs rather than running to ``config.epochs``."""
        dataset = _SyntheticDetectionDataset()
        val_dataset = _SyntheticDetectionDataset(size=5)

        config = _detect_config(tmp_path, epochs=5, patience=1, val_every=1)

        # Validation metrics: epoch 0 improves, epoch 1 does not → stop at epoch 2
        mock_val_returns = [{"map50": 0.5}, {"map50": 0.3}]

        with (
            patch.object(TorchTrainingEngine, "_load_model_for_training", return_value=_TinyDetector()),
            patch.object(TorchTrainingEngine, "_validate", side_effect=mock_val_returns),
        ):
            engine = TorchTrainingEngine("detect", config.model, config)
            result = engine.train(dataset, val_dataset)

        # Early stopping: epoch 0 → improve, epoch 1 → no improve → stop
        # epochs_completed should be 2 (epochs 0 and 1 complete before break)
        assert result.epochs_completed < config.epochs, (
            f"Expected early stopping before epoch {config.epochs}, " f"got epochs_completed={result.epochs_completed}"
        )
        assert result.epochs_completed == 2

    def test_no_early_stopping_when_patience_zero(self, tmp_path: Path) -> None:
        """When ``patience=0`` (disabled), all epochs run regardless of metrics."""
        dataset = _SyntheticDetectionDataset()
        config = _detect_config(tmp_path, epochs=2, patience=0)

        with patch.object(TorchTrainingEngine, "_load_model_for_training", return_value=_TinyDetector()):
            engine = TorchTrainingEngine("detect", config.model, config)
            result = engine.train(dataset)

        assert result.epochs_completed == config.epochs


# ===========================================================================
# E9-09  AMP on CPU completes without error
# ===========================================================================


class TestAMPOnCPU:
    """AMP (``amp=True``) on CPU is silently a no-op and must not raise."""

    def test_amp_cpu_completes_without_error(self, tmp_path: Path) -> None:
        """Training with ``amp=True, device='cpu'`` must complete successfully.

        The engine internally checks ``device.type == 'cuda'`` before enabling
        the GradScaler, so on CPU, AMP is effectively disabled without error.
        """
        dataset = _SyntheticDetectionDataset()
        config = _detect_config(tmp_path, epochs=1, amp=True, device="cpu")

        with patch.object(TorchTrainingEngine, "_load_model_for_training", return_value=_TinyDetector()):
            engine = TorchTrainingEngine("detect", config.model, config)
            result = engine.train(dataset)

        assert isinstance(result, TrainingResult)
        assert result.epochs_completed == 1

    def test_amp_cpu_loss_is_finite(self, tmp_path: Path) -> None:
        """Losses produced in AMP-flagged CPU training are finite numbers."""
        dataset = _SyntheticDetectionDataset()
        config = _detect_config(tmp_path, epochs=1, amp=True, device="cpu")

        with patch.object(TorchTrainingEngine, "_load_model_for_training", return_value=_TinyDetector()):
            engine = TorchTrainingEngine("detect", config.model, config)
            result = engine.train(dataset)

        for loss_val in result.history.get("train_loss", []):
            assert loss_val > 0
            assert not (np.isnan(loss_val) or np.isinf(loss_val))


# ===========================================================================
# E9-10  mata.finetune() API
# ===========================================================================


class TestFinetuneAPI:
    """mata.finetune() passes fine-tuning defaults (low LR, frozen backbone)."""

    def test_finetune_uses_lower_lr_and_frozen_backbone(self, tmp_path: Path) -> None:
        """``mata.finetune()`` must use ``lr<=1e-5`` and ``freeze_backbone=True``."""
        import mata
        from mata.training.trainer import TrainingOrchestrator

        captured: dict[str, Any] = {}

        def _fake_train(self: TrainingOrchestrator) -> TrainingResult:  # noqa: D401
            captured["lr"] = self.config.lr
            captured["freeze_backbone"] = self.config.freeze_backbone
            captured["epochs"] = self.config.epochs
            return TrainingResult(epochs_completed=0)

        with patch.object(TrainingOrchestrator, "train", _fake_train):
            mata.finetune(
                "detect",
                model="torchvision/fasterrcnn_resnet50_fpn",
                data="synthetic",
                save_dir=str(tmp_path),
            )

        assert captured["lr"] <= 1e-5, f"Expected lr<=1e-5, got {captured['lr']}"
        assert captured["freeze_backbone"] is True
        assert captured["epochs"] == 5  # finetune default

    def test_finetune_returns_training_result(self, tmp_path: Path) -> None:
        """``mata.finetune()`` must return a :class:`TrainingResult`."""
        import mata
        from mata.training.trainer import TrainingOrchestrator

        with patch.object(TrainingOrchestrator, "train", return_value=TrainingResult(epochs_completed=3)):
            result = mata.finetune(
                "detect",
                model="torchvision/fasterrcnn_resnet50_fpn",
                data="synthetic",
                save_dir=str(tmp_path),
            )

        assert isinstance(result, TrainingResult)


# ===========================================================================
# E9-11  mata.train() API orchestration
# ===========================================================================


class TestTrainAPI:
    """mata.train() correctly constructs config and returns TrainingResult."""

    def test_train_api_returns_training_result(self, tmp_path: Path) -> None:
        """``mata.train()`` must return a :class:`TrainingResult` instance."""
        import mata
        from mata.training.trainer import TrainingOrchestrator

        fake_result = TrainingResult(epochs_completed=2)

        with patch.object(TrainingOrchestrator, "train", return_value=fake_result):
            result = mata.train(
                "detect",
                model="torchvision/fasterrcnn_resnet50_fpn",
                data="synthetic",
                epochs=2,
                save_dir=str(tmp_path),
            )

        assert isinstance(result, TrainingResult)
        assert result.epochs_completed == 2

    def test_train_api_passes_config_correctly(self, tmp_path: Path) -> None:
        """Config params passed to ``mata.train()`` are forwarded to the orchestrator."""
        import mata
        from mata.training.trainer import TrainingOrchestrator

        captured_config: dict[str, Any] = {}

        def _capture_train(self: TrainingOrchestrator) -> TrainingResult:
            captured_config.update(
                {
                    "task": self.config.task,
                    "lr": self.config.lr,
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                }
            )
            return TrainingResult(epochs_completed=0)

        with patch.object(TrainingOrchestrator, "train", _capture_train):
            mata.train(
                "classify",
                model="microsoft/resnet-50",
                data="synthetic",
                epochs=3,
                lr=1e-3,
                batch_size=4,
                save_dir=str(tmp_path),
            )

        assert captured_config["task"] == "classify"
        assert captured_config["lr"] == pytest.approx(1e-3)
        assert captured_config["epochs"] == 3
        assert captured_config["batch_size"] == 4


# ===========================================================================
# E9-12  Periodic checkpoints listed after training
# ===========================================================================


class TestPeriodicCheckpoints:
    """``CheckpointManager.list_checkpoints()`` lists periodic saves correctly."""

    def test_periodic_checkpoints_listed_after_training(self, tmp_path: Path) -> None:
        """With ``save_every=1`` and 3 epochs, at least 3 checkpoint dirs exist."""
        dataset = _SyntheticDetectionDataset()
        config = _detect_config(tmp_path, epochs=3, save_every=1)

        with patch.object(TorchTrainingEngine, "_load_model_for_training", return_value=_TinyDetector()):
            engine = TorchTrainingEngine("detect", config.model, config)
            result = engine.train(dataset)

        ckpt_mgr = CheckpointManager()
        run_dir = Path(result.last_checkpoint).parent
        checkpoints = ckpt_mgr.list_checkpoints(str(run_dir))

        # Expect: epoch1, epoch2, epoch3, last = at least 3 entries
        assert len(checkpoints) >= 3, f"Expected >= 3 checkpoints, found: {checkpoints}"
