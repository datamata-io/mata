"""Tests for mata.training.hf_trainer.HFTrainingEngine.

All tests use mocked HuggingFace models and Trainer — no real model downloads occur.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from mata.core.exceptions import TrainingError
from mata.training.config import TrainingConfig
from mata.training.result import TrainingResult

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> TrainingConfig:
    """Return a minimal valid TrainingConfig, overridable via kwargs."""
    defaults = dict(
        task="detect",
        model="facebook/detr-resnet-50",
        data="coco.yaml",
        epochs=2,
        batch_size=2,
        lr=1e-4,
        warmup_epochs=0,
        save_dir="runs/test_hf_trainer",
        num_workers=0,
        patience=0,
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def _make_simple_model() -> nn.Module:
    """Return a tiny nn.Module with named sub-modules for freeze testing."""

    class _TinyDetect(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.class_labels_classifier = nn.Linear(4, 2)
            self.bbox_predictor = nn.Linear(4, 4)
            self.pixel_level_module = nn.Linear(4, 4)
            self.class_predictor = nn.Linear(4, 2)
            self.mask_embedder = nn.Linear(4, 4)
            self.classifier = nn.Linear(4, 2)

        def forward(self, x):
            return x

    return _TinyDetect()


def _make_mock_trainer_state(epoch: float = 2.0, log_history=None):
    """Create a mock TrainerState."""
    state = Mock()
    state.epoch = epoch
    state.log_history = log_history or [{"loss": 0.5, "epoch": 1.0}, {"loss": 0.3, "epoch": 2.0}]
    return state


def _make_mock_trainer(state=None):
    """Return a mock Trainer instance."""
    trainer = Mock()
    trainer.state = state or _make_mock_trainer_state()
    trainer.optimizer = Mock()
    trainer.train = Mock(return_value=None)
    return trainer


def _make_mock_transformers(trainer_instance=None):
    """Build a dict of mocked transformers classes."""
    mock_model = _make_simple_model()
    mock_processor = Mock()

    if trainer_instance is None:
        trainer_instance = _make_mock_trainer()

    AutoImageProcessor = Mock()
    AutoImageProcessor.from_pretrained = Mock(return_value=mock_processor)

    AutoModelForObjectDetection = Mock()
    AutoModelForObjectDetection.from_pretrained = Mock(return_value=mock_model)

    AutoModelForImageClassification = Mock()
    AutoModelForImageClassification.from_pretrained = Mock(return_value=mock_model)

    mock_training_args = Mock()
    TrainingArguments = Mock(return_value=mock_training_args)

    TrainerCls = Mock(return_value=trainer_instance)

    class _FakeTrainerCallback:
        pass

    return {
        "AutoConfig": Mock(),
        "AutoImageProcessor": AutoImageProcessor,
        "AutoModelForObjectDetection": AutoModelForObjectDetection,
        "AutoModelForImageClassification": AutoModelForImageClassification,
        "Mask2FormerForUniversalSegmentation": Mock(),
        "TrainingArguments": TrainingArguments,
        "Trainer": TrainerCls,
        "TrainerCallback": _FakeTrainerCallback,
        # references for inspection
        "_mock_model": mock_model,
        "_mock_processor": mock_processor,
        "_mock_training_args": mock_training_args,
        "_mock_trainer_instance": trainer_instance,
    }


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestHFTrainingEngineInit:
    """Tests for HFTrainingEngine.__init__."""

    def test_valid_detect_task(self):
        with patch("mata.training.hf_trainer._ensure_transformers", return_value={}):
            from mata.training.hf_trainer import HFTrainingEngine

            engine = HFTrainingEngine("detect", "facebook/detr-resnet-50", _make_config())
            assert engine.task == "detect"
            assert engine.model_id == "facebook/detr-resnet-50"

    def test_valid_classify_task(self):
        with patch("mata.training.hf_trainer._ensure_transformers", return_value={}):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(task="classify", model="microsoft/resnet-50")
            engine = HFTrainingEngine("classify", "microsoft/resnet-50", cfg)
            assert engine.task == "classify"

    def test_valid_segment_task(self):
        with patch("mata.training.hf_trainer._ensure_transformers", return_value={}):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(task="segment", model="facebook/mask2former-swin-tiny-coco-instance")
            engine = HFTrainingEngine("segment", "facebook/mask2former-swin-tiny-coco-instance", cfg)
            assert engine.task == "segment"

    def test_invalid_task_raises_value_error(self):
        with patch("mata.training.hf_trainer._ensure_transformers", return_value={}):
            from mata.training.hf_trainer import HFTrainingEngine

            with pytest.raises(ValueError, match="unsupported task"):
                HFTrainingEngine("ocr", "some-model", _make_config())

    def test_missing_transformers_raises_import_error(self):
        """If transformers is absent, __init__ should raise ImportError immediately."""
        from mata.training.hf_trainer import HFTrainingEngine

        with patch(
            "mata.training.hf_trainer._ensure_transformers",
            side_effect=ImportError("transformers not installed"),
        ):
            with pytest.raises(ImportError, match="transformers"):
                HFTrainingEngine("detect", "facebook/detr-resnet-50", _make_config())

    def test_model_and_processor_initially_none(self):
        with patch("mata.training.hf_trainer._ensure_transformers", return_value={}):
            from mata.training.hf_trainer import HFTrainingEngine

            engine = HFTrainingEngine("detect", "facebook/detr-resnet-50", _make_config())
            assert engine.model is None
            assert engine.processor is None

    def test_config_stored(self):
        with patch("mata.training.hf_trainer._ensure_transformers", return_value={}):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(epochs=5)
            engine = HFTrainingEngine("detect", "facebook/detr-resnet-50", cfg)
            assert engine.config is cfg


# ---------------------------------------------------------------------------
# _load_model_for_training tests
# ---------------------------------------------------------------------------


class TestLoadModelForTraining:
    """Tests for HFTrainingEngine._load_model_for_training."""

    def _make_engine(self, task="detect", **cfg_kwargs):
        tf = _make_mock_transformers()
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(task=task, **cfg_kwargs)
            engine = HFTrainingEngine(task, "facebook/detr-resnet-50", cfg)
        return engine, tf

    def test_detect_loads_object_detection_model(self):
        engine, tf = self._make_engine(task="detect")
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._load_model_for_training()
        tf["AutoModelForObjectDetection"].from_pretrained.assert_called_once_with("facebook/detr-resnet-50")
        assert engine.model is not None

    def test_classify_loads_classification_model(self):
        engine, tf = self._make_engine(task="classify")
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._load_model_for_training()
        tf["AutoModelForImageClassification"].from_pretrained.assert_called_once()
        assert engine.model is not None

    def test_processor_loaded_for_detect(self):
        engine, tf = self._make_engine(task="detect")
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._load_model_for_training()
        tf["AutoImageProcessor"].from_pretrained.assert_called_once_with("facebook/detr-resnet-50", use_fast=True)
        assert engine.processor is not None

    def test_model_not_in_eval_mode_after_load(self):
        """Model must remain in training mode — NOT eval mode."""
        engine, tf = self._make_engine(task="classify")
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._load_model_for_training()
        # The mock model is a real nn.Module; training mode is True by default
        assert engine.model.training is True

    def test_model_gradients_not_disabled(self):
        """After load, parameters should have requires_grad=True (no no_grad applied)."""
        engine, tf = self._make_engine(task="classify")
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._load_model_for_training()
        # All params in the model should be grad-enabled by default
        for param in engine.model.parameters():
            assert param.requires_grad is True

    def test_segment_uses_mask2former(self):
        """Segment task should use Mask2FormerForUniversalSegmentation from tf dict."""
        mock_mask2former_cls = Mock()
        mock_mask2former_cls.from_pretrained = Mock(return_value=_make_simple_model())
        tf = _make_mock_transformers()
        tf["Mask2FormerForUniversalSegmentation"] = mock_mask2former_cls

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(task="segment")
            engine = HFTrainingEngine("segment", cfg.model, cfg)

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._load_model_for_training()

        mock_mask2former_cls.from_pretrained.assert_called_once()

    def test_load_failure_raises_training_error(self):
        engine, tf = self._make_engine(task="detect")
        tf["AutoModelForObjectDetection"].from_pretrained.side_effect = RuntimeError("network error")
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            with pytest.raises(TrainingError, match="Failed to load model"):
                engine._load_model_for_training()

    def test_auto_device_cpu_when_no_cuda(self):
        engine, tf = self._make_engine(task="detect", device="auto")
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            with patch("torch.cuda.is_available", return_value=False):
                engine._load_model_for_training()
        assert engine._device == "cpu"

    def test_explicit_device_respected(self):
        engine, tf = self._make_engine(task="detect", device="cpu")
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._load_model_for_training()
        assert engine._device == "cpu"


# ---------------------------------------------------------------------------
# _build_training_args tests
# ---------------------------------------------------------------------------


class TestBuildTrainingArgs:
    """Tests for HFTrainingEngine._build_training_args."""

    def _engine_with_tf(self, task="detect", **cfg_kwargs):
        tf = _make_mock_transformers()
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(task=task, **cfg_kwargs)
            engine = HFTrainingEngine(task, cfg.model, cfg)
        return engine, tf

    def test_training_args_created(self):
        engine, tf = self._engine_with_tf()
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            args = engine._build_training_args()
        tf["TrainingArguments"].assert_called_once()
        assert args is tf["_mock_training_args"]

    def test_lr_passed_to_training_args(self):
        engine, tf = self._engine_with_tf(lr=5e-5)
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._build_training_args()
        call_kwargs = tf["TrainingArguments"].call_args[1]
        assert call_kwargs["learning_rate"] == 5e-5

    def test_epochs_passed_to_training_args(self):
        engine, tf = self._engine_with_tf(epochs=3)
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._build_training_args()
        call_kwargs = tf["TrainingArguments"].call_args[1]
        assert call_kwargs["num_train_epochs"] == 3

    def test_batch_size_passed(self):
        engine, tf = self._engine_with_tf(batch_size=4)
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._build_training_args()
        call_kwargs = tf["TrainingArguments"].call_args[1]
        assert call_kwargs["per_device_train_batch_size"] == 4

    def test_weight_decay_passed(self):
        engine, tf = self._engine_with_tf(weight_decay=0.05)
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._build_training_args()
        call_kwargs = tf["TrainingArguments"].call_args[1]
        assert call_kwargs["weight_decay"] == 0.05

    def test_adamw_optimizer_mapped(self):
        engine, tf = self._engine_with_tf(optimizer="adamw")
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._build_training_args()
        call_kwargs = tf["TrainingArguments"].call_args[1]
        assert call_kwargs["optim"] == "adamw_torch"

    def test_sgd_optimizer_mapped(self):
        engine, tf = self._engine_with_tf(optimizer="sgd")
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._build_training_args()
        call_kwargs = tf["TrainingArguments"].call_args[1]
        assert call_kwargs["optim"] == "sgd"

    def test_cosine_scheduler_mapped(self):
        engine, tf = self._engine_with_tf(scheduler="cosine")
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._build_training_args()
        call_kwargs = tf["TrainingArguments"].call_args[1]
        assert call_kwargs["lr_scheduler_type"] == "cosine"

    def test_linear_scheduler_mapped(self):
        engine, tf = self._engine_with_tf(scheduler="linear")
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._build_training_args()
        call_kwargs = tf["TrainingArguments"].call_args[1]
        assert call_kwargs["lr_scheduler_type"] == "linear"

    def test_none_scheduler_mapped(self):
        engine, tf = self._engine_with_tf(scheduler="none")
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._build_training_args()
        call_kwargs = tf["TrainingArguments"].call_args[1]
        assert call_kwargs["lr_scheduler_type"] == "constant"

    def test_report_to_none(self):
        engine, tf = self._engine_with_tf()
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._build_training_args()
        call_kwargs = tf["TrainingArguments"].call_args[1]
        assert call_kwargs["report_to"] == "none"

    def test_remove_unused_columns_false(self):
        """Must be False so custom dataset columns are preserved."""
        engine, tf = self._engine_with_tf()
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            engine._build_training_args()
        call_kwargs = tf["TrainingArguments"].call_args[1]
        assert call_kwargs["remove_unused_columns"] is False


# ---------------------------------------------------------------------------
# _freeze_backbone tests
# ---------------------------------------------------------------------------


class TestFreezeBackbone:
    """Tests for HFTrainingEngine._freeze_backbone."""

    def _make_engine(self, task="detect"):
        with patch("mata.training.hf_trainer._ensure_transformers", return_value={}):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(task=task)
            return HFTrainingEngine(task, cfg.model, cfg)

    def test_detect_freezes_all_except_head(self):
        engine = self._make_engine(task="detect")
        model = _make_simple_model()
        engine._freeze_backbone(model)

        # Head parameters should be trainable
        assert model.class_labels_classifier.weight.requires_grad is True
        assert model.bbox_predictor.weight.requires_grad is True

        # Backbone should be frozen
        assert model.backbone.weight.requires_grad is False

    def test_classify_freezes_all_except_classifier(self):
        engine = self._make_engine(task="classify")
        model = _make_simple_model()
        engine._freeze_backbone(model)

        assert model.classifier.weight.requires_grad is True
        assert model.backbone.weight.requires_grad is False

    def test_segment_freezes_all_except_segment_head(self):
        engine = self._make_engine(task="segment")
        model = _make_simple_model()
        engine._freeze_backbone(model)

        assert model.class_predictor.weight.requires_grad is True
        assert model.mask_embedder.weight.requires_grad is True
        assert model.pixel_level_module.weight.requires_grad is False

    def test_no_trainable_params_if_no_matching_head(self):
        """Unknown task has empty head patterns → all frozen."""
        with patch("mata.training.hf_trainer._ensure_transformers", return_value={}):
            from mata.training.hf_trainer import _TASK_HEAD_PATTERNS, HFTrainingEngine

            cfg = _make_config(task="detect")
            engine = HFTrainingEngine("detect", cfg.model, cfg)
            # Temporarily clear detect patterns
            original = _TASK_HEAD_PATTERNS.pop("detect")
            model = _make_simple_model()
            engine._freeze_backbone(model)
            _TASK_HEAD_PATTERNS["detect"] = original  # restore

        for param in model.parameters():
            assert param.requires_grad is False


# ---------------------------------------------------------------------------
# _freeze_layers tests
# ---------------------------------------------------------------------------


class TestFreezeLayers:
    """Tests for HFTrainingEngine._freeze_layers."""

    def _make_engine(self):
        with patch("mata.training.hf_trainer._ensure_transformers", return_value={}):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config()
            return HFTrainingEngine("detect", cfg.model, cfg)

    def test_freeze_by_name_pattern(self):
        engine = self._make_engine()
        model = _make_simple_model()
        engine._freeze_layers(model, ["backbone"])
        assert model.backbone.weight.requires_grad is False
        assert model.classifier.weight.requires_grad is True

    def test_freeze_multiple_patterns(self):
        engine = self._make_engine()
        model = _make_simple_model()
        engine._freeze_layers(model, ["backbone", "classifier"])
        assert model.backbone.weight.requires_grad is False
        assert model.classifier.weight.requires_grad is False
        assert model.bbox_predictor.weight.requires_grad is True

    def test_nothing_frozen_if_pattern_not_matched(self):
        engine = self._make_engine()
        model = _make_simple_model()
        engine._freeze_layers(model, ["nonexistent_layer"])
        for param in model.parameters():
            assert param.requires_grad is True

    def test_empty_pattern_list_freezes_nothing(self):
        engine = self._make_engine()
        model = _make_simple_model()
        engine._freeze_layers(model, [])
        for param in model.parameters():
            assert param.requires_grad is True


# ---------------------------------------------------------------------------
# _build_data_collator tests
# ---------------------------------------------------------------------------


class TestBuildDataCollator:
    """Tests for HFTrainingEngine._build_data_collator."""

    def _make_engine(self, task):
        with patch("mata.training.hf_trainer._ensure_transformers", return_value={}):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(task=task)
            engine = HFTrainingEngine(task, cfg.model, cfg)
            engine.processor = Mock()
            engine.model = _make_simple_model()
            return engine

    def test_classify_returns_classification_collator(self):
        from mata.training.hf_trainer import _ClassificationCollator

        engine = self._make_engine("classify")
        collator = engine._build_data_collator()
        assert isinstance(collator, _ClassificationCollator)

    def test_detect_returns_detection_collator(self):
        from mata.training.hf_trainer import _DetectionCollator

        engine = self._make_engine("detect")
        collator = engine._build_data_collator()
        assert isinstance(collator, _DetectionCollator)

    def test_segment_returns_detection_collator(self):
        from mata.training.hf_trainer import _DetectionCollator

        engine = self._make_engine("segment")
        collator = engine._build_data_collator()
        assert isinstance(collator, _DetectionCollator)

    def test_detection_collator_has_correct_task(self):

        engine = self._make_engine("segment")
        collator = engine._build_data_collator()
        assert collator.task == "segment"


# ---------------------------------------------------------------------------
# _build_compute_metrics tests
# ---------------------------------------------------------------------------


class TestBuildComputeMetrics:
    """Tests for HFTrainingEngine._build_compute_metrics."""

    def _make_engine(self, task):
        with patch("mata.training.hf_trainer._ensure_transformers", return_value={}):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(task=task)
            return HFTrainingEngine(task, cfg.model, cfg)

    def test_classify_returns_callable(self):
        engine = self._make_engine("classify")
        fn = engine._build_compute_metrics()
        assert callable(fn)

    def test_classify_compute_metrics_accuracy(self):
        import numpy as np

        engine = self._make_engine("classify")
        fn = engine._build_compute_metrics()
        logits = np.array([[0.1, 0.9], [0.8, 0.2]])  # pred [1, 0]
        labels = np.array([1, 0])
        result = fn((logits, labels))
        assert result["accuracy"] == pytest.approx(1.0)

    def test_classify_compute_metrics_partial_accuracy(self):
        import numpy as np

        engine = self._make_engine("classify")
        fn = engine._build_compute_metrics()
        logits = np.array([[0.1, 0.9], [0.8, 0.2]])  # pred [1, 0]
        labels = np.array([0, 0])  # first is wrong
        result = fn((logits, labels))
        assert result["accuracy"] == pytest.approx(0.5)

    def test_detect_returns_none(self):
        engine = self._make_engine("detect")
        assert engine._build_compute_metrics() is None

    def test_segment_returns_none(self):
        engine = self._make_engine("segment")
        assert engine._build_compute_metrics() is None


# ---------------------------------------------------------------------------
# train() integration tests
# ---------------------------------------------------------------------------


class TestTrain:
    """Integration tests for HFTrainingEngine.train()."""

    def _run_train(
        self,
        task="detect",
        trainer_instance=None,
        val_ok=False,
        **cfg_kwargs,
    ):
        """Helper: run train() with fully mocked transformers and datasets."""
        tf = _make_mock_transformers(trainer_instance=trainer_instance)

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(task=task, **cfg_kwargs)
            engine = HFTrainingEngine(task, cfg.model, cfg)

        dummy_dataset = [
            (torch.zeros(3, 32, 32), {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)})
        ]

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            with patch("mata.training.checkpoint.CheckpointManager.save", return_value=Path("runs/last")):
                with patch(
                    "mata.training.checkpoint.CheckpointManager.export_for_inference",
                    return_value=Path("runs/best"),
                ):
                    result = engine.train(
                        train_dataset=dummy_dataset,
                        val_dataset=dummy_dataset if val_ok else None,
                    )
        return result, tf

    def test_train_returns_training_result(self):
        result, _ = self._run_train()
        assert isinstance(result, TrainingResult)

    def test_train_calls_trainer_train(self):
        mock_trainer = _make_mock_trainer()
        result, tf = self._run_train(trainer_instance=mock_trainer)
        mock_trainer.train.assert_called_once()

    def test_train_with_resume(self):
        mock_trainer = _make_mock_trainer()
        result, tf = self._run_train(trainer_instance=mock_trainer, resume="/some/ckpt")
        call_kwargs = mock_trainer.train.call_args[1]
        assert call_kwargs["resume_from_checkpoint"] == "/some/ckpt"

    def test_train_without_resume_passes_none(self):
        mock_trainer = _make_mock_trainer()
        result, tf = self._run_train(trainer_instance=mock_trainer)
        call_kwargs = mock_trainer.train.call_args[1]
        assert call_kwargs["resume_from_checkpoint"] is None

    def test_train_history_populated_from_logs(self):
        state = _make_mock_trainer_state(log_history=[{"loss": 0.8, "epoch": 1.0}, {"loss": 0.5, "epoch": 2.0}])
        mock_trainer = _make_mock_trainer(state=state)
        result, _ = self._run_train(trainer_instance=mock_trainer)
        assert "train_loss" in result.history
        assert result.history["train_loss"] == pytest.approx([0.8, 0.5])

    def test_train_epochs_completed(self):
        state = _make_mock_trainer_state(epoch=3.0)
        result, _ = self._run_train(trainer_instance=_make_mock_trainer(state=state))
        assert result.epochs_completed == 3

    def test_train_with_val_dataset_passes_eval_dataset(self):
        mock_trainer_cls = Mock()
        mock_trainer_instance = _make_mock_trainer()
        mock_trainer_cls.return_value = mock_trainer_instance

        tf = _make_mock_transformers()
        tf["Trainer"] = mock_trainer_cls

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config()
            engine = HFTrainingEngine("detect", cfg.model, cfg)

        dummy = [(torch.zeros(3, 32, 32), {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)})]
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            with patch("mata.training.checkpoint.CheckpointManager.save", return_value=Path("runs/last")):
                with patch(
                    "mata.training.checkpoint.CheckpointManager.export_for_inference", return_value=Path("runs/best")
                ):
                    engine.train(train_dataset=dummy, val_dataset=dummy)

        trainer_kwargs = mock_trainer_cls.call_args[1]
        assert "eval_dataset" in trainer_kwargs

    def test_train_without_val_no_eval_dataset(self):
        mock_trainer_cls = Mock()
        mock_trainer_instance = _make_mock_trainer()
        mock_trainer_cls.return_value = mock_trainer_instance

        tf = _make_mock_transformers()
        tf["Trainer"] = mock_trainer_cls

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config()
            engine = HFTrainingEngine("detect", cfg.model, cfg)

        dummy = [(torch.zeros(3, 32, 32), {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)})]
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            with patch("mata.training.checkpoint.CheckpointManager.save", return_value=Path("runs/last")):
                with patch(
                    "mata.training.checkpoint.CheckpointManager.export_for_inference", return_value=Path("runs/best")
                ):
                    engine.train(train_dataset=dummy, val_dataset=None)

        trainer_kwargs = mock_trainer_cls.call_args[1]
        assert "eval_dataset" not in trainer_kwargs

    def test_trainer_failure_raises_training_error(self):
        mock_trainer = _make_mock_trainer()
        mock_trainer.train.side_effect = RuntimeError("CUDA OOM")

        tf = _make_mock_transformers(trainer_instance=mock_trainer)

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config()
            engine = HFTrainingEngine("detect", cfg.model, cfg)

        dummy = [(torch.zeros(3, 32, 32), {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)})]
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            with pytest.raises(TrainingError, match="Training failed"):
                engine.train(train_dataset=dummy)

    def test_freeze_backbone_applied_when_configured(self):
        tf = _make_mock_transformers()
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(freeze_backbone=True)
            engine = HFTrainingEngine("detect", cfg.model, cfg)

        dummy = [(torch.zeros(3, 32, 32), {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)})]
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            with patch.object(engine.__class__, "_freeze_backbone") as mock_freeze:
                with patch("mata.training.checkpoint.CheckpointManager.save", return_value=Path("runs/last")):
                    with patch(
                        "mata.training.checkpoint.CheckpointManager.export_for_inference",
                        return_value=Path("runs/best"),
                    ):
                        engine.train(train_dataset=dummy)
        mock_freeze.assert_called_once()

    def test_freeze_not_applied_when_not_configured(self):
        tf = _make_mock_transformers()
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(freeze_backbone=False)
            engine = HFTrainingEngine("detect", cfg.model, cfg)

        dummy = [(torch.zeros(3, 32, 32), {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)})]
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            with patch.object(engine.__class__, "_freeze_backbone") as mock_freeze:
                with patch("mata.training.checkpoint.CheckpointManager.save", return_value=Path("runs/last")):
                    with patch(
                        "mata.training.checkpoint.CheckpointManager.export_for_inference",
                        return_value=Path("runs/best"),
                    ):
                        engine.train(train_dataset=dummy)
        mock_freeze.assert_not_called()

    def test_result_config_is_training_config(self):
        result, _ = self._run_train()
        assert isinstance(result.config, TrainingConfig)

    def test_result_history_type(self):
        result, _ = self._run_train()
        assert isinstance(result.history, dict)

    def test_history_key_normalisation_eval_to_val(self):
        """eval_loss → val_loss, eval_accuracy → val_accuracy."""
        state = _make_mock_trainer_state(
            log_history=[{"loss": 0.8, "eval_loss": 1.0, "eval_accuracy": 0.6, "epoch": 1.0}]
        )
        mock_trainer = _make_mock_trainer(state=state)
        result, _ = self._run_train(trainer_instance=mock_trainer)
        assert "val_loss" in result.history
        assert "val_accuracy" in result.history
        assert "train_loss" in result.history

    def test_classify_train_returns_result(self):
        result, _ = self._run_train(task="classify")
        assert isinstance(result, TrainingResult)

    def test_segment_train_returns_result(self):
        mock_mask2former_cls = Mock()
        mock_mask2former_cls.from_pretrained = Mock(return_value=_make_simple_model())
        tf = _make_mock_transformers()
        tf["Mask2FormerForUniversalSegmentation"] = mock_mask2former_cls

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(task="segment")
            engine = HFTrainingEngine("segment", cfg.model, cfg)

        dummy = [(torch.zeros(3, 32, 32), {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)})]
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            with patch("mata.training.checkpoint.CheckpointManager.save", return_value=Path("runs/last")):
                with patch(
                    "mata.training.checkpoint.CheckpointManager.export_for_inference",
                    return_value=Path("runs/best"),
                ):
                    result = engine.train(train_dataset=dummy)
        assert isinstance(result, TrainingResult)


# ---------------------------------------------------------------------------
# Collator unit tests
# ---------------------------------------------------------------------------


class TestClassificationCollator:
    """Unit tests for _ClassificationCollator."""

    def test_returns_pixel_values_and_labels(self):
        from mata.training.hf_trainer import _ClassificationCollator

        processor = Mock()
        processor.return_value = {"pixel_values": torch.zeros(2, 3, 32, 32)}
        collator = _ClassificationCollator(processor=processor)

        img1 = Mock()
        img1.mode = "RGB"
        img2 = Mock()
        img2.mode = "RGB"
        batch = [
            (img1, {"label": 0}),
            (img2, {"label": 1}),
        ]
        output = collator(batch)
        assert "labels" in output
        assert output["labels"].tolist() == [0, 1]

    def test_labels_dtype_is_long(self):
        from mata.training.hf_trainer import _ClassificationCollator

        processor = Mock()
        processor.return_value = {"pixel_values": torch.zeros(1, 3, 32, 32)}
        collator = _ClassificationCollator(processor=processor)

        img = Mock()
        img.mode = "RGB"
        output = collator([(img, {"label": 2})])
        assert output["labels"].dtype == torch.long


class TestDetectionCollator:
    """Unit tests for _DetectionCollator."""

    def test_labels_included_in_output(self):
        from mata.training.hf_trainer import _DetectionCollator

        processor = Mock()
        processor.return_value = {"pixel_values": torch.zeros(1, 3, 32, 32)}
        collator = _DetectionCollator(processor=processor, task="detect")

        img = Mock()
        img.mode = "RGB"
        target = {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1]),
        }
        output = collator([(img, target)])
        assert "labels" in output
        assert len(output["labels"]) == 1

    def test_empty_target_yields_empty_tensors(self):
        from mata.training.hf_trainer import _DetectionCollator

        processor = Mock()
        processor.return_value = {"pixel_values": torch.zeros(1, 3, 32, 32)}
        collator = _DetectionCollator(processor=processor, task="detect")

        img = Mock()
        img.mode = "RGB"
        output = collator([(img, {})])
        label = output["labels"][0]
        assert label["class_labels"].numel() == 0
        assert label["boxes"].numel() == 0


# ---------------------------------------------------------------------------
# _HistoryCallback tests
# ---------------------------------------------------------------------------


class TestHistoryCallback:
    """Unit tests for _HistoryCallback."""

    def test_collects_numeric_logs(self):
        from mata.training.hf_trainer import _HistoryCallback

        cb = _HistoryCallback()
        cb.on_log(None, None, None, logs={"loss": 0.5, "lr": 1e-4})
        assert cb.history["loss"] == [0.5]
        assert cb.history["lr"] == pytest.approx([1e-4])

    def test_ignores_non_numeric(self):
        from mata.training.hf_trainer import _HistoryCallback

        cb = _HistoryCallback()
        cb.on_log(None, None, None, logs={"epoch": 1, "note": "hello"})
        assert "note" not in cb.history

    def test_appends_multiple_calls(self):
        from mata.training.hf_trainer import _HistoryCallback

        cb = _HistoryCallback()
        cb.on_log(None, None, None, logs={"loss": 1.0})
        cb.on_log(None, None, None, logs={"loss": 0.5})
        assert cb.history["loss"] == [1.0, 0.5]

    def test_none_logs_ignored(self):
        from mata.training.hf_trainer import _HistoryCallback

        cb = _HistoryCallback()
        cb.on_log(None, None, None, logs=None)  # should not raise
        assert cb.history == {}


# ---------------------------------------------------------------------------
# Coverage gap tests — _ensure_transformers (lines 36-72)
# ---------------------------------------------------------------------------


class TestEnsureTransformers:
    """Tests for _ensure_transformers() covering the function branches."""

    def test_returns_cached_value_immediately(self):
        """When _transformers is already populated, returns it without re-importing."""
        import mata.training.hf_trainer as hf_mod

        sentinel = {"Trainer": object(), "AutoImageProcessor": object()}
        original = hf_mod._transformers
        hf_mod._transformers = sentinel
        try:
            result = hf_mod._ensure_transformers()
        finally:
            hf_mod._transformers = original

        assert result is sentinel

    def _make_fake_transformers_module(self, *, include_mask2former: bool = True):
        """Build a types.ModuleType that emulates transformers for import."""

        mod = types.ModuleType("transformers")
        mod.AutoConfig = Mock()
        mod.AutoImageProcessor = Mock()
        mod.AutoModelForObjectDetection = Mock()
        mod.AutoModelForImageClassification = Mock()
        mod.AutoModelForSemanticSegmentation = Mock()
        mod.Trainer = Mock()
        mod.TrainerCallback = Mock()
        mod.TrainingArguments = Mock()
        mod.default_data_collator = Mock()
        mod.EarlyStoppingCallback = Mock()
        if include_mask2former:
            mod.Mask2FormerForUniversalSegmentation = Mock()
        # (omit Mask2Former when include_mask2former=False to trigger the except ImportError path)
        return mod

    def test_caches_result_after_first_call(self):
        """Two successive calls return the same dict object (cache populated)."""
        import sys

        import mata.training.hf_trainer as hf_mod

        fake_tf = self._make_fake_transformers_module()
        original = hf_mod._transformers
        hf_mod._transformers = None
        try:
            with patch.dict(sys.modules, {"transformers": fake_tf}):
                first = hf_mod._ensure_transformers()
                second = hf_mod._ensure_transformers()
        finally:
            hf_mod._transformers = original

        assert first is second
        assert "AutoImageProcessor" in first

    def test_raises_import_error_when_transformers_missing(self):
        """ImportError raised with helpful message when transformers not installed."""
        import sys

        import mata.training.hf_trainer as hf_mod

        original = hf_mod._transformers
        hf_mod._transformers = None
        try:
            with patch.dict(sys.modules, {"transformers": None}):
                with pytest.raises(ImportError, match="transformers"):
                    hf_mod._ensure_transformers()
        finally:
            hf_mod._transformers = original

    def test_mask2former_none_when_attr_absent(self):
        """Mask2FormerForUniversalSegmentation is set to None when absent on transformers."""
        import sys

        import mata.training.hf_trainer as hf_mod

        # Module without Mask2Former → triggers the inner except ImportError
        fake_tf = self._make_fake_transformers_module(include_mask2former=False)

        original = hf_mod._transformers
        hf_mod._transformers = None
        try:
            with patch.dict(sys.modules, {"transformers": fake_tf}):
                result = hf_mod._ensure_transformers()
        finally:
            hf_mod._transformers = original

        assert "Mask2FormerForUniversalSegmentation" in result
        assert result["Mask2FormerForUniversalSegmentation"] is None


# ---------------------------------------------------------------------------
# Coverage gap tests — _ClassificationCollator with tensor images (line 149)
# ---------------------------------------------------------------------------


class TestClassificationCollatorTensorInput:
    """Tests for _ClassificationCollator with pre-converted tensor inputs."""

    def test_tensor_images_stacked_without_processor(self):
        """When images lack .mode (not PIL), processor is bypassed (line 149).

        Note: torch.Tensor *has* a .mode method, so we use a custom wrapper
        that is not a PIL Image and not a bare tensor to trigger the else branch.
        torch.stack is patched to return a real tensor while accepting the wrappers.
        """
        from mata.training.hf_trainer import _ClassificationCollator

        processor = Mock()
        collator = _ClassificationCollator(processor=processor)

        # Objects without .mode → else branch triggers (lines 149-150)
        class _RawTensor:
            pass

        img1, img2 = _RawTensor(), _RawTensor()

        stacked = torch.zeros(2, 3, 32, 32)
        with patch("torch.stack", return_value=stacked):
            batch = [(img1, 0), (img2, 1)]
            output = collator(batch)

        assert "pixel_values" in output
        assert output["pixel_values"].shape == (2, 3, 32, 32)
        # Processor should NOT be called — tensor path does not use it
        processor.assert_not_called()

    def test_int_labels_converted_to_long_tensor(self):
        from mata.training.hf_trainer import _ClassificationCollator

        processor = Mock()
        collator = _ClassificationCollator(processor=processor)

        class _RawTensor:
            pass

        img = _RawTensor()
        stacked = torch.zeros(1, 3, 16, 16)
        with patch("torch.stack", return_value=stacked):
            output = collator([(img, 7)])

        assert output["labels"][0].item() == 7
        assert output["labels"].dtype == torch.long


# ---------------------------------------------------------------------------
# Coverage gap tests — _DetectionCollator with tensor images + segment targets
# (lines 195-209, 218-223, 247-250)
# ---------------------------------------------------------------------------


class TestDetectionCollatorCoverageGaps:
    """Tests targeting uncovered branches in _DetectionCollator.__call__."""

    # Reusable helper: image without .mode (triggers the non-PIL else branch)
    class _RawTensor:
        """Non-PIL, non-mode image object for triggering the else branch."""

        pass

    def test_tensor_images_use_processor_encoding(self):
        """Non-PIL tensor images (no .mode) → processor called via else branch (lines 195-209)."""
        from mata.training.hf_trainer import _DetectionCollator

        processor = Mock()
        processor.return_value = {"pixel_values": torch.zeros(1, 3, 32, 32)}
        collator = _DetectionCollator(processor=processor, task="detect")

        img = self._RawTensor()  # no .mode → else branch
        target = {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1]),
        }
        output = collator([(img, target)])

        processor.assert_called()
        assert "labels" in output

    def test_tensor_images_fallback_when_processor_fails(self):
        """When processor raises for non-PIL input, falls back to torch.stack (lines 249-250)."""
        from mata.training.hf_trainer import _DetectionCollator

        processor = Mock()
        processor.side_effect = Exception("processor error")
        collator = _DetectionCollator(processor=processor, task="detect")

        img = self._RawTensor()  # no .mode → else branch
        target = {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1]),
        }
        stacked = torch.zeros(1, 3, 32, 32)
        with patch("torch.stack", return_value=stacked):
            output = collator([(img, target)])

        # Fallback: pixel_values from torch.stack
        assert "pixel_values" in output
        assert output["pixel_values"].shape == (1, 3, 32, 32)

    def test_segment_task_non_dict_target_yields_zero_masks(self):
        """segment task with non-dict target → empty mask_labels (lines 218-223)."""
        from mata.training.hf_trainer import _DetectionCollator

        processor = Mock()
        processor.return_value = {"pixel_values": torch.zeros(1, 3, 32, 32)}
        collator = _DetectionCollator(processor=processor, task="segment")

        img = Mock()
        img.mode = "RGB"
        # Non-dict target (e.g., plain int class label)
        batch = [(img, 0)]
        output = collator(batch)

        label = output["labels"][0]
        assert "class_labels" in label
        assert "mask_labels" in label
        assert label["class_labels"].numel() == 0

    def test_detect_task_non_dict_target_yields_zero_boxes(self):
        """detect task with non-dict target → empty boxes (lines 218-223 detect path)."""
        from mata.training.hf_trainer import _DetectionCollator

        processor = Mock()
        processor.return_value = {"pixel_values": torch.zeros(1, 3, 32, 32)}
        collator = _DetectionCollator(processor=processor, task="detect")

        img = Mock()
        img.mode = "RGB"
        batch = [(img, "not_a_dict")]  # non-dict target
        output = collator(batch)
        label = output["labels"][0]
        assert label["boxes"].numel() == 0

    def test_segment_task_with_masks_in_target(self):
        """segment task with actual mask tensors in target (lines 228-234)."""
        from mata.training.hf_trainer import _DetectionCollator

        processor = Mock()
        processor.return_value = {"pixel_values": torch.zeros(1, 3, 32, 32)}
        collator = _DetectionCollator(processor=processor, task="segment")

        img = Mock()
        img.mode = "RGB"
        target = {
            "labels": torch.tensor([1, 2]),
            "masks": torch.zeros(2, 32, 32, dtype=torch.uint8),
        }
        output = collator([(img, target)])
        label = output["labels"][0]
        assert "mask_labels" in label
        assert label["mask_labels"].shape[0] == 2

    def test_detect_target_missing_boxes_yields_empty(self):
        """detect target with labels but boxes=None → empty boxes (lines 247-250)."""
        from mata.training.hf_trainer import _DetectionCollator

        processor = Mock()
        processor.return_value = {"pixel_values": torch.zeros(1, 3, 32, 32)}
        collator = _DetectionCollator(processor=processor, task="detect")

        img = Mock()
        img.mode = "RGB"
        target = {"labels": torch.tensor([1]), "boxes": None}
        output = collator([(img, target)])
        label = output["labels"][0]
        assert label["boxes"].numel() == 0


# ---------------------------------------------------------------------------
# Coverage gap tests — _load_model_for_training classify with id2label
# (lines 385-392)
# ---------------------------------------------------------------------------


class TestLoadModelForTrainingClassifyWithId2Label:
    """Tests for the id2label branch in _load_model_for_training (classify)."""

    def test_classify_with_id2label_passes_kwargs_to_from_pretrained(self):
        """When id2label provided, model loaded with num_labels + label maps (lines 385-392)."""
        tf = _make_mock_transformers()

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(task="classify")
            engine = HFTrainingEngine("classify", cfg.model, cfg)

        id2label = {0: "cat", 1: "dog"}
        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            with patch("torch.cuda.is_available", return_value=False):
                engine._load_model_for_training(id2label=id2label)

        # from_pretrained should have been called with num_labels, id2label, label2id
        call_kwargs = tf["AutoModelForImageClassification"].from_pretrained.call_args[1]
        assert call_kwargs.get("num_labels") == 2
        assert call_kwargs.get("id2label") == {0: "cat", 1: "dog"}
        assert call_kwargs.get("ignore_mismatched_sizes") is True

    def test_classify_without_id2label_skips_kwargs(self):
        """When id2label is None, from_pretrained called with no extra kwargs."""
        tf = _make_mock_transformers()

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(task="classify")
            engine = HFTrainingEngine("classify", cfg.model, cfg)

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            with patch("torch.cuda.is_available", return_value=False):
                engine._load_model_for_training(id2label=None)

        call_kwargs = tf["AutoModelForImageClassification"].from_pretrained.call_args[1]
        assert "num_labels" not in call_kwargs


# ---------------------------------------------------------------------------
# Coverage gap tests — _load_model_for_training segment Mask2Former=None
# (lines 406-407, 454-455)
# ---------------------------------------------------------------------------


class TestLoadModelForTrainingSegmentMask2FormerNone:
    """segment task raises ImportError when Mask2Former class is None."""

    def test_segment_raises_import_error_when_mask2former_none(self):
        tf = _make_mock_transformers()
        tf["Mask2FormerForUniversalSegmentation"] = None  # Simulate absent class

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(task="segment")
            engine = HFTrainingEngine("segment", cfg.model, cfg)

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            with pytest.raises(ImportError, match="Mask2Former"):
                engine._load_model_for_training()

    def test_segment_succeeds_when_mask2former_available(self):
        """segment succeeds when Mask2Former class is present."""
        mock_m2f = Mock()
        mock_m2f.from_pretrained = Mock(return_value=_make_simple_model())

        tf = _make_mock_transformers()
        tf["Mask2FormerForUniversalSegmentation"] = mock_m2f

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            from mata.training.hf_trainer import HFTrainingEngine

            cfg = _make_config(task="segment")
            engine = HFTrainingEngine("segment", cfg.model, cfg)

        with patch("mata.training.hf_trainer._ensure_transformers", return_value=tf):
            with patch("torch.cuda.is_available", return_value=False):
                engine._load_model_for_training()

        mock_m2f.from_pretrained.assert_called_once()
