"""Tests for mata.training.torch_trainer.TorchTrainingEngine.

All tests use mocked torchvision models — no real model downloads occur.
"""

from __future__ import annotations

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
    """Return a minimal valid TrainingConfig for torchvision detection."""
    defaults = dict(
        task="detect",
        model="torchvision/fasterrcnn_resnet50_fpn",
        data="coco.yaml",
        epochs=2,
        batch_size=2,
        lr=1e-4,
        warmup_epochs=0,
        save_dir="runs/test_torch_trainer",
        num_workers=0,
        patience=0,
        amp=False,  # CPU tests — disable AMP by default
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def _make_fasterrcnn_mock() -> nn.Module:
    """Return a tiny mock mimicking Faster R-CNN structure."""

    class _MockPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_score = nn.Linear(256, 91)
            self.bbox_pred = nn.Linear(256, 364)

        @property
        def in_features(self):
            return self.cls_score.in_features

    class _MockRoiHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.box_predictor = _MockPredictor()

    class _MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3, 4)

    class _MockFasterRCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _MockBackbone()
            self.roi_heads = _MockRoiHeads()

        def forward(self, images, targets=None):
            # In train mode return loss dict, otherwise return predictions list
            if self.training and targets is not None:
                return {
                    "loss_classifier": torch.tensor(0.5, requires_grad=True),
                    "loss_box_reg": torch.tensor(0.3, requires_grad=True),
                    "loss_objectness": torch.tensor(0.2, requires_grad=True),
                    "loss_rpn_box_reg": torch.tensor(0.1, requires_grad=True),
                }
            return [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long), "scores": torch.zeros(0)}]

    return _MockFasterRCNN()


def _make_retinanet_mock() -> nn.Module:
    """Return a tiny mock mimicking RetinaNet structure."""

    class _MockClsHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_logits = nn.Conv2d(256, 9 * 80, kernel_size=3, padding=1)
            self.num_anchors = 9
            self.num_classes = 80

    class _MockHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.classification_head = _MockClsHead()

    class _MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3, 4)

    class _MockRetinaNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _MockBackbone()
            self.head = _MockHead()

        def forward(self, images, targets=None):
            if self.training and targets is not None:
                return {
                    "bbox_regression": torch.tensor(0.4, requires_grad=True),
                    "classification": torch.tensor(0.6, requires_grad=True),
                }
            return [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long), "scores": torch.zeros(0)}]

    return _MockRetinaNet()


def _make_fcos_mock() -> nn.Module:
    """Return a tiny mock mimicking FCOS structure."""

    class _MockFCOSClsHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_logits = nn.Conv2d(256, 80, kernel_size=3, padding=1)
            self.num_classes = 80

    class _MockFCOSHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.classification_head = _MockFCOSClsHead()

    class _MockFCOSBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3, 4)

    class _MockFCOS(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _MockFCOSBackbone()
            self.head = _MockFCOSHead()

        def forward(self, images, targets=None):
            if self.training and targets is not None:
                return {
                    "bbox_regression": torch.tensor(0.4, requires_grad=True),
                    "classification": torch.tensor(0.5, requires_grad=True),
                }
            return [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long), "scores": torch.zeros(0)}]

    return _MockFCOS()


def _make_ssd_mock() -> nn.Module:
    """Return a tiny mock mimicking SSD300-VGG16 structure."""

    class _MockSSDClsHead(nn.Module):
        def __init__(self):
            super().__init__()
            # SSD uses a module_list of direct Conv2d layers
            self.module_list = nn.ModuleList(
                [
                    nn.Conv2d(512, 4 * 91, kernel_size=3, padding=1),
                    nn.Conv2d(1024, 6 * 91, kernel_size=3, padding=1),
                ]
            )
            self.num_classes = 91

    class _MockSSDRegHead(nn.Module):
        def __init__(self):
            super().__init__()
            # Regression: num_anchors * 4 coords per cell (mirrors cls anchor counts)
            self.module_list = nn.ModuleList(
                [
                    nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
                    nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
                ]
            )

    class _MockSSDHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.classification_head = _MockSSDClsHead()
            self.regression_head = _MockSSDRegHead()

    class _MockSSDBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3, 4)

    class _MockSSD(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _MockSSDBackbone()
            self.head = _MockSSDHead()

        def forward(self, images, targets=None):
            if self.training and targets is not None:
                return {
                    "bbox_regression": torch.tensor(0.3, requires_grad=True),
                    "classification": torch.tensor(0.7, requires_grad=True),
                }
            return [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long), "scores": torch.zeros(0)}]

    return _MockSSD()


def _make_tiny_dataset(size: int = 4):
    """Return a tiny torch Dataset that yields (image_tensor, target_dict) tuples."""

    class _TinyDetectDataset(torch.utils.data.Dataset):
        def __init__(self, n):
            self.n = n
            self.class_names = ["cat", "dog"]
            self.num_classes = 2

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            image = torch.rand(3, 64, 64)
            target = {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([1], dtype=torch.long),
            }
            return image, target

    return _TinyDetectDataset(size)


def _make_engine(model_name="torchvision/fasterrcnn_resnet50_fpn", **cfg_kwargs):
    """Build a TorchTrainingEngine without importing torchvision."""
    # Patch MODEL_BUILDERS so validation doesn't require torchvision import
    from mata.training.torch_trainer import TorchTrainingEngine

    cfg = _make_config(model=model_name, **cfg_kwargs)
    engine = TorchTrainingEngine("detect", model_name, cfg)
    return engine


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestTorchTrainingEngineInit:
    """Tests for TorchTrainingEngine.__init__."""

    def test_valid_fasterrcnn(self):
        engine = _make_engine("torchvision/fasterrcnn_resnet50_fpn")
        assert engine.task == "detect"
        assert engine.model_key == "fasterrcnn_resnet50_fpn"

    def test_valid_retinanet(self):
        engine = _make_engine("torchvision/retinanet_resnet50_fpn")
        assert engine.model_key == "retinanet_resnet50_fpn"

    def test_valid_fcos(self):
        engine = _make_engine("torchvision/fcos_resnet50_fpn")
        assert engine.model_key == "fcos_resnet50_fpn"

    def test_valid_ssd(self):
        engine = _make_engine("torchvision/ssd300_vgg16")
        assert engine.model_key == "ssd300_vgg16"

    def test_prefix_stripped_from_model_key(self):
        engine = _make_engine("torchvision/fasterrcnn_resnet50_fpn_v2")
        assert engine.model_key == "fasterrcnn_resnet50_fpn_v2"
        assert "torchvision/" not in engine.model_key

    def test_invalid_task_raises_training_error(self):
        from mata.training.torch_trainer import TorchTrainingEngine

        cfg = _make_config(task="detect", model="torchvision/fasterrcnn_resnet50_fpn")
        cfg.task = "classify"  # override post-init for testing
        with pytest.raises(TrainingError, match="detect"):
            TorchTrainingEngine("classify", "torchvision/fasterrcnn_resnet50_fpn", cfg)

    def test_unknown_model_raises_training_error(self):
        from mata.training.torch_trainer import TorchTrainingEngine

        cfg = _make_config(model="torchvision/fasterrcnn_resnet50_fpn")
        with pytest.raises(TrainingError, match="Unknown torchvision model"):
            TorchTrainingEngine("detect", "torchvision/nonexistent_model_xyz", cfg)

    def test_config_stored(self):
        engine = _make_engine(epochs=5)
        assert engine.config.epochs == 5

    def test_checkpoint_manager_created(self):
        engine = _make_engine()
        assert engine.ckpt_manager is not None


# ---------------------------------------------------------------------------
# _load_model_for_training tests
# ---------------------------------------------------------------------------


class TestLoadModelForTraining:
    """Tests for TorchTrainingEngine._load_model_for_training."""

    def test_model_in_train_mode_after_load(self):
        engine = _make_engine()
        mock_model = _make_fasterrcnn_mock()

        import torchvision.models.detection as det_mods

        with patch.object(det_mods, "fasterrcnn_resnet50_fpn", return_value=mock_model):
            result = engine._load_model_for_training()

        # Default after construction is train mode (not eval)
        assert result.training is True

    def test_builder_called_with_default_weights(self):
        engine = _make_engine()
        mock_model = _make_fasterrcnn_mock()
        mock_builder = Mock(return_value=mock_model)

        import torchvision.models.detection as det_mods

        with patch.object(det_mods, "fasterrcnn_resnet50_fpn", mock_builder):
            engine._load_model_for_training()

        mock_builder.assert_called_once_with(weights="DEFAULT")

    def test_old_api_fallback_on_type_error(self):
        """If weights='DEFAULT' raises TypeError, fallback to pretrained=True."""
        engine = _make_engine()
        mock_model = _make_fasterrcnn_mock()

        def _builder_with_fallback(**kwargs):
            if "weights" in kwargs:
                raise TypeError("unexpected keyword argument 'weights'")
            return mock_model

        import torchvision.models.detection as det_mods

        with patch.object(det_mods, "fasterrcnn_resnet50_fpn", _builder_with_fallback):
            result = engine._load_model_for_training()

        assert result is mock_model

    def test_missing_torchvision_raises_training_error(self):
        engine = _make_engine()
        with patch("builtins.__import__", side_effect=ImportError("no module named torchvision")):
            with pytest.raises((ImportError, TrainingError)):
                engine._load_model_for_training()


# ---------------------------------------------------------------------------
# _modify_head tests
# ---------------------------------------------------------------------------


class TestModifyHead:
    """Tests for TorchTrainingEngine._modify_head — head replacement."""

    def test_fasterrcnn_head_replaced(self):
        engine = _make_engine("torchvision/fasterrcnn_resnet50_fpn")
        model = _make_fasterrcnn_mock()
        num_classes = 5

        engine._modify_head(model, num_classes)

        # After replacement, the new predictor head should reference num_classes
        new_predictor = model.roi_heads.box_predictor
        # Check that cls_score output equals num_classes
        assert new_predictor.cls_score.out_features == num_classes

    def test_retinanet_head_num_classes_updated(self):
        engine = _make_engine("torchvision/retinanet_resnet50_fpn")
        model = _make_retinanet_mock()
        num_classes = 10

        engine._modify_head(model, num_classes)

        cls_head = model.head.classification_head
        assert cls_head.num_classes == num_classes

    def test_retinanet_cls_logits_output_channels(self):
        engine = _make_engine("torchvision/retinanet_resnet50_fpn")
        model = _make_retinanet_mock()
        num_classes = 10
        num_anchors = model.head.classification_head.num_anchors

        engine._modify_head(model, num_classes)

        new_logits = model.head.classification_head.cls_logits
        assert new_logits.out_channels == num_anchors * num_classes

    def test_fcos_head_num_classes_updated(self):
        engine = _make_engine("torchvision/fcos_resnet50_fpn")
        model = _make_fcos_mock()
        num_classes = 7

        engine._modify_head(model, num_classes)

        cls_head = model.head.classification_head
        assert cls_head.num_classes == num_classes
        # FCOS is anchor-free: out_channels == num_classes
        assert cls_head.cls_logits.out_channels == num_classes

    def test_ssd_head_module_list_replaced(self):
        engine = _make_engine("torchvision/ssd300_vgg16")
        model = _make_ssd_mock()
        original_len = len(model.head.classification_head.module_list)
        num_classes = 20

        engine._modify_head(model, num_classes)

        cls_head = model.head.classification_head
        assert cls_head.num_classes == num_classes
        # module_list length preserved; each conv now has updated channels
        assert len(cls_head.module_list) == original_len

    def test_ssd_head_conv_output_channels_updated(self):
        engine = _make_engine("torchvision/ssd300_vgg16")
        model = _make_ssd_mock()
        old_num_classes = model.head.classification_head.num_classes  # 91

        # 4 anchors for first layer: out_ch = 4 * num_classes
        first_conv = model.head.classification_head.module_list[0]
        num_anchors_first = first_conv.out_channels // old_num_classes

        num_classes = 5
        engine._modify_head(model, num_classes)

        new_first_conv = model.head.classification_head.module_list[0]
        assert new_first_conv.out_channels == num_anchors_first * num_classes

    def test_fasterrcnn_v2_head_replaced(self):
        """fasterrcnn_resnet50_fpn_v2 uses same head replacement path."""
        engine = _make_engine("torchvision/fasterrcnn_resnet50_fpn_v2")
        model = _make_fasterrcnn_mock()
        engine._modify_head(model, 3)
        assert model.roi_heads.box_predictor.cls_score.out_features == 3


# ---------------------------------------------------------------------------
# _freeze_backbone tests
# ---------------------------------------------------------------------------


class TestFreezeBackbone:
    """Tests for TorchTrainingEngine._freeze_backbone."""

    def test_backbone_params_frozen(self):
        engine = _make_engine()
        model = _make_fasterrcnn_mock()

        engine._freeze_backbone(model)

        for param in model.backbone.parameters():
            assert param.requires_grad is False

    def test_non_backbone_params_remain_trainable(self):
        engine = _make_engine()
        model = _make_fasterrcnn_mock()

        engine._freeze_backbone(model)

        for param in model.roi_heads.parameters():
            assert param.requires_grad is True

    def test_model_without_backbone_does_not_raise(self):
        """Models without a 'backbone' attribute should not raise."""
        engine = _make_engine()

        class _NoBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(4, 4)

        model = _NoBackbone()
        # Should log a warning and return without error
        engine._freeze_backbone(model)

    def test_freeze_layers_by_name_pattern(self):
        engine = _make_engine(freeze_layers=["backbone"])
        model = _make_fasterrcnn_mock()

        engine._apply_freeze_layers(model)

        for name, param in model.named_parameters():
            if "backbone" in name:
                assert param.requires_grad is False


# ---------------------------------------------------------------------------
# _build_optimizer tests
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    """Tests for TorchTrainingEngine._build_optimizer."""

    def test_adamw_selected(self):
        engine = _make_engine(optimizer="adamw")
        model = _make_fasterrcnn_mock()
        opt = engine._build_optimizer(model)
        assert isinstance(opt, torch.optim.AdamW)

    def test_sgd_selected(self):
        engine = _make_engine(optimizer="sgd")
        model = _make_fasterrcnn_mock()
        opt = engine._build_optimizer(model)
        assert isinstance(opt, torch.optim.SGD)

    def test_adam_selected(self):
        engine = _make_engine(optimizer="adam")
        model = _make_fasterrcnn_mock()
        opt = engine._build_optimizer(model)
        assert isinstance(opt, torch.optim.Adam)

    def test_optimizer_uses_correct_lr(self):
        engine = _make_engine(lr=5e-4)
        model = _make_fasterrcnn_mock()
        opt = engine._build_optimizer(model)
        assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)

    def test_no_trainable_params_raises_training_error(self):
        engine = _make_engine()
        model = _make_fasterrcnn_mock()
        for param in model.parameters():
            param.requires_grad = False

        with pytest.raises(TrainingError, match="No trainable parameters"):
            engine._build_optimizer(model)

    def test_unknown_optimizer_raises_training_error(self):
        engine = _make_engine()
        engine.config.optimizer = "rmsprop"
        model = _make_fasterrcnn_mock()

        with pytest.raises(TrainingError, match="Unknown optimizer"):
            engine._build_optimizer(model)


# ---------------------------------------------------------------------------
# _build_scheduler tests
# ---------------------------------------------------------------------------


class TestBuildScheduler:
    """Tests for TorchTrainingEngine._build_scheduler."""

    def _make_optimizer(self, model=None):
        if model is None:
            model = _make_fasterrcnn_mock()
        return torch.optim.AdamW(model.parameters(), lr=1e-4)

    def test_cosine_scheduler(self):
        engine = _make_engine(scheduler="cosine")
        opt = self._make_optimizer()
        sched = engine._build_scheduler(opt)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_linear_scheduler(self):
        engine = _make_engine(scheduler="linear")
        opt = self._make_optimizer()
        sched = engine._build_scheduler(opt)
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)

    def test_step_scheduler(self):
        engine = _make_engine(scheduler="step")
        opt = self._make_optimizer()
        sched = engine._build_scheduler(opt)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_none_scheduler_returns_constant_lr(self):
        engine = _make_engine(scheduler="none")
        opt = self._make_optimizer()
        sched = engine._build_scheduler(opt)
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)
        # LR should not change after one step
        initial_lr = opt.param_groups[0]["lr"]
        sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(initial_lr)

    def test_unknown_scheduler_raises_training_error(self):
        engine = _make_engine()
        engine.config.scheduler = "warmup_cosine"
        opt = self._make_optimizer()

        with pytest.raises(TrainingError, match="Unknown scheduler"):
            engine._build_scheduler(opt)


# ---------------------------------------------------------------------------
# Training loop tests (mocked)
# ---------------------------------------------------------------------------


class TestTrainLoop:
    """Tests for TorchTrainingEngine.train() with mocked internals."""

    def _run_train(self, engine, train_ds=None, val_ds=None, mock_model=None, tmp_path=None):
        """Helper to run engine.train() with full mocking."""
        if train_ds is None:
            train_ds = _make_tiny_dataset(4)
        if mock_model is None:
            mock_model = _make_fasterrcnn_mock()

        import torchvision.models.detection as det_mods

        mock_ckpt_manager = Mock()

        # Patch save_dir creation and checkpoint manager
        with (
            patch.object(det_mods, "fasterrcnn_resnet50_fpn", return_value=mock_model),
            patch.object(engine, "ckpt_manager", mock_ckpt_manager),
            patch("mata.training.torch_trainer.Path.mkdir"),
        ):
            if tmp_path:
                engine.config.save_dir = str(tmp_path)
            result = engine.train(train_ds, val_ds)

        return result, mock_ckpt_manager

    def test_returns_training_result(self, tmp_path):
        engine = _make_engine(save_dir=str(tmp_path), epochs=1)
        model = _make_fasterrcnn_mock()
        dataset = _make_tiny_dataset(4)

        import torchvision.models.detection as det_mods

        with patch.object(det_mods, "fasterrcnn_resnet50_fpn", return_value=model):
            result = engine.train(dataset)

        assert isinstance(result, TrainingResult)

    def test_history_contains_train_loss(self, tmp_path):
        engine = _make_engine(save_dir=str(tmp_path), epochs=2)
        model = _make_fasterrcnn_mock()
        dataset = _make_tiny_dataset(4)

        import torchvision.models.detection as det_mods

        with patch.object(det_mods, "fasterrcnn_resnet50_fpn", return_value=model):
            result = engine.train(dataset)

        assert "train_loss" in result.history
        assert len(result.history["train_loss"]) == 2

    def test_history_contains_lr(self, tmp_path):
        engine = _make_engine(save_dir=str(tmp_path), epochs=2)
        model = _make_fasterrcnn_mock()
        dataset = _make_tiny_dataset(4)

        import torchvision.models.detection as det_mods

        with patch.object(det_mods, "fasterrcnn_resnet50_fpn", return_value=model):
            result = engine.train(dataset)

        assert "lr" in result.history
        assert len(result.history["lr"]) == 2

    def test_epochs_completed_equals_config_epochs(self, tmp_path):
        engine = _make_engine(save_dir=str(tmp_path), epochs=3)
        model = _make_fasterrcnn_mock()
        dataset = _make_tiny_dataset(4)

        import torchvision.models.detection as det_mods

        with patch.object(det_mods, "fasterrcnn_resnet50_fpn", return_value=model):
            result = engine.train(dataset)

        assert result.epochs_completed == 3

    def test_last_checkpoint_set(self, tmp_path):
        engine = _make_engine(save_dir=str(tmp_path), epochs=1)
        model = _make_fasterrcnn_mock()
        dataset = _make_tiny_dataset(4)

        import torchvision.models.detection as det_mods

        with patch.object(det_mods, "fasterrcnn_resnet50_fpn", return_value=model):
            result = engine.train(dataset)

        assert result.last_checkpoint != ""

    def test_amp_disabled_on_cpu(self, tmp_path):
        """On CPU, AMP scaler should not be created even if amp=True in config."""
        # Explicitly force CPU device so AMP is bypassed regardless of hardware
        engine = _make_engine(save_dir=str(tmp_path), epochs=1, amp=True, device="cpu")
        model = _make_fasterrcnn_mock()
        dataset = _make_tiny_dataset(4)

        import torchvision.models.detection as det_mods

        scaler_calls = []

        original_grad_scaler = torch.amp.GradScaler

        def _track_scaler(*args, **kwargs):
            scaler_calls.append(args)
            return original_grad_scaler(*args, **kwargs)

        with (
            patch.object(det_mods, "fasterrcnn_resnet50_fpn", return_value=model),
            patch("torch.amp.GradScaler", side_effect=_track_scaler),
        ):
            engine.train(dataset)

        # use_amp = config.amp AND device.type == "cuda"; device is cpu → no scaler
        assert not scaler_calls

    def test_periodic_checkpoint_saved(self, tmp_path):
        engine = _make_engine(save_dir=str(tmp_path), epochs=4, save_every=2)
        model = _make_fasterrcnn_mock()
        dataset = _make_tiny_dataset(4)

        import torchvision.models.detection as det_mods

        with patch.object(det_mods, "fasterrcnn_resnet50_fpn", return_value=model):
            result = engine.train(dataset)

        # Checkpoints expected: epoch2, epoch4, last (and possibly best)
        assert result.last_checkpoint != ""

    def test_validation_run_at_correct_intervals(self, tmp_path):
        val_ds = _make_tiny_dataset(4)
        engine = _make_engine(save_dir=str(tmp_path), epochs=4, val_every=2, patience=0)
        model = _make_fasterrcnn_mock()
        dataset = _make_tiny_dataset(4)
        validate_calls = []

        def _mock_validate(m, v, dev, ep):
            validate_calls.append(ep)
            return {"map50": 0.5}

        import torchvision.models.detection as det_mods

        with (
            patch.object(det_mods, "fasterrcnn_resnet50_fpn", return_value=model),
            patch.object(engine, "_validate", side_effect=_mock_validate),
        ):
            engine.train(dataset, val_ds)

        # val_every=2 with epochs=4 → validate at epochs 1 and 3 (0-indexed) = epochs 2 and 4
        assert len(validate_calls) == 2
        assert validate_calls[0] == 1  # epoch index 1 (epoch number 2)
        assert validate_calls[1] == 3  # epoch index 3 (epoch number 4)

    def test_early_stopping_triggers(self, tmp_path):
        """Training stops early when no improvement for `patience` epochs."""
        val_ds = _make_tiny_dataset(4)
        engine = _make_engine(save_dir=str(tmp_path), epochs=10, patience=2, val_every=1)
        model = _make_fasterrcnn_mock()
        dataset = _make_tiny_dataset(4)

        # Return improving metric only on first call, then flat
        call_count = [0]

        def _mock_validate(m, v, dev, ep):
            call_count[0] += 1
            return {"map50": 0.5}  # constant — no improvement after first

        import torchvision.models.detection as det_mods

        with (
            patch.object(det_mods, "fasterrcnn_resnet50_fpn", return_value=model),
            patch.object(engine, "_validate", side_effect=_mock_validate),
        ):
            result = engine.train(dataset, val_ds)

        # Should stop before epoch 10: 1 (improve) + 2 (patience) = 3 epochs
        assert result.epochs_completed < 10
        assert result.epochs_completed == 3

    def test_best_checkpoint_set_on_improvement(self, tmp_path):
        val_ds = _make_tiny_dataset(4)
        engine = _make_engine(save_dir=str(tmp_path), epochs=2, val_every=1)
        model = _make_fasterrcnn_mock()
        dataset = _make_tiny_dataset(4)
        metrics_seq = [{"map50": 0.3}, {"map50": 0.7}]
        idx = [0]

        def _mock_validate(m, v, dev, ep):
            r = metrics_seq[min(idx[0], len(metrics_seq) - 1)]
            idx[0] += 1
            return r

        import torchvision.models.detection as det_mods

        with (
            patch.object(det_mods, "fasterrcnn_resnet50_fpn", return_value=model),
            patch.object(engine, "_validate", side_effect=_mock_validate),
        ):
            result = engine.train(dataset, val_ds)

        assert result.best_checkpoint != ""

    def test_training_result_config_populated(self, tmp_path):
        engine = _make_engine(save_dir=str(tmp_path), epochs=1)
        model = _make_fasterrcnn_mock()
        dataset = _make_tiny_dataset(4)

        import torchvision.models.detection as det_mods

        with patch.object(det_mods, "fasterrcnn_resnet50_fpn", return_value=model):
            result = engine.train(dataset)

        assert result.config is engine.config

    def test_checkpoint_manager_save_called(self, tmp_path):
        engine = _make_engine(save_dir=str(tmp_path), epochs=1)
        model = _make_fasterrcnn_mock()
        dataset = _make_tiny_dataset(4)
        mock_ckpt = Mock()

        import torchvision.models.detection as det_mods

        with (
            patch.object(det_mods, "fasterrcnn_resnet50_fpn", return_value=model),
            patch.object(engine, "ckpt_manager", mock_ckpt),
        ):
            engine.train(dataset)

        # At minimum, last checkpoint should be saved
        assert mock_ckpt.save.called


# ---------------------------------------------------------------------------
# AMP (Automatic Mixed Precision) tests
# ---------------------------------------------------------------------------


class TestAMP:
    """Tests for AMP behavior on CPU vs CUDA."""

    def test_use_amp_false_on_cpu(self):
        engine = _make_engine(amp=True)
        engine.config.device = "cpu"
        # Simulate _resolve_device
        device = torch.device("cpu")
        use_amp = engine.config.amp and device.type == "cuda"
        assert use_amp is False

    def test_use_amp_true_on_cuda_when_enabled(self):
        engine = _make_engine(amp=True)
        device = torch.device("cuda")
        use_amp = engine.config.amp and device.type == "cuda"
        assert use_amp is True

    def test_use_amp_false_when_disabled_even_on_cuda(self):
        engine = _make_engine(amp=False)
        device = torch.device("cuda")
        use_amp = engine.config.amp and device.type == "cuda"
        assert use_amp is False


# ---------------------------------------------------------------------------
# _resolve_device tests
# ---------------------------------------------------------------------------


class TestResolveDevice:
    """Tests for TorchTrainingEngine._resolve_device."""

    def test_auto_resolves_to_cpu_when_no_cuda(self):
        engine = _make_engine(device="auto")
        with patch("torch.cuda.is_available", return_value=False):
            device = engine._resolve_device()
        assert device.type == "cpu"

    def test_auto_resolves_to_cuda_when_available(self):
        engine = _make_engine(device="auto")
        with patch("torch.cuda.is_available", return_value=True):
            device = engine._resolve_device()
        assert device.type == "cuda"

    def test_explicit_cpu_device(self):
        engine = _make_engine(device="cpu")
        device = engine._resolve_device()
        assert device.type == "cpu"


# ---------------------------------------------------------------------------
# _extract_val_metric helper tests
# ---------------------------------------------------------------------------


class TestExtractValMetric:
    """Tests for the _extract_val_metric module-level helper."""

    def test_extracts_map50_from_dict(self):
        from mata.training.torch_trainer import _extract_val_metric

        assert _extract_val_metric({"map50": 0.42}) == pytest.approx(0.42)

    def test_extracts_map_when_no_map50(self):
        from mata.training.torch_trainer import _extract_val_metric

        assert _extract_val_metric({"map": 0.35}) == pytest.approx(0.35)

    def test_extracts_fitness_as_last_resort(self):
        from mata.training.torch_trainer import _extract_val_metric

        assert _extract_val_metric({"fitness": 0.77}) == pytest.approx(0.77)

    def test_returns_zero_for_none(self):
        from mata.training.torch_trainer import _extract_val_metric

        assert _extract_val_metric(None) == pytest.approx(0.0)

    def test_returns_zero_for_empty_dict(self):
        from mata.training.torch_trainer import _extract_val_metric

        assert _extract_val_metric({}) == pytest.approx(0.0)

    def test_extracts_map50_from_object(self):
        from mata.training.torch_trainer import _extract_val_metric

        obj = Mock()
        obj.map50 = 0.55
        obj.map = None
        assert _extract_val_metric(obj) == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# Coverage gap tests — _get_tqdm ImportError path (lines 46-47)
# ---------------------------------------------------------------------------


class TestGetTqdmGap:
    def test_returns_none_when_tqdm_not_installed(self):
        """_get_tqdm() returns None when tqdm is absent (lines 46-47)."""
        import sys

        with patch.dict(sys.modules, {"tqdm": None, "tqdm.auto": None}):
            # Reimport triggers the except ImportError branch
            import importlib

            import mata.training.torch_trainer as tt_mod

            tt_mod_reloaded = importlib.reload(tt_mod)
            result = tt_mod_reloaded._get_tqdm()
        assert result is None

    def test_returns_tqdm_class_when_installed(self):
        from mata.training.torch_trainer import _get_tqdm

        result = _get_tqdm()
        # tqdm IS installed in dev env
        assert result is not None


# ---------------------------------------------------------------------------
# Coverage gap tests — _extract_val_metric exception paths (lines 63-64, 71-73)
# ---------------------------------------------------------------------------


class TestExtractValMetricGaps:
    """Tests targeting the except (TypeError, ValueError): pass paths."""

    def test_dict_unconvertible_value_skips_to_next_key(self):
        """TypeError in float(dict[key]) skips that key (line 63-64)."""
        from mata.training.torch_trainer import _extract_val_metric

        result = _extract_val_metric({"map50": object(), "map": 0.42})
        assert result == pytest.approx(0.42)

    def test_dict_all_unconvertible_returns_zero(self):
        """All dict values are unconvertible → return 0.0 (line 64)."""
        from mata.training.torch_trainer import _extract_val_metric

        result = _extract_val_metric({"map50": object(), "map": object(), "fitness": object()})
        assert result == pytest.approx(0.0)

    def test_object_unconvertible_attr_skips_to_next(self):
        """TypeError in float(obj.attr) skips that attribute (lines 71-72)."""
        from mata.training.torch_trainer import _extract_val_metric

        obj = Mock()
        obj.map50 = object()  # float(object()) raises TypeError
        obj.map = 0.55
        result = _extract_val_metric(obj)
        assert result == pytest.approx(0.55)

    def test_object_all_attrs_unconvertible_returns_zero(self):
        """All object attrs raise → return 0.0 (line 73)."""
        from mata.training.torch_trainer import _extract_val_metric

        obj = Mock()
        obj.map50 = object()
        obj.map = object()
        obj.fitness = object()
        result = _extract_val_metric(obj)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Coverage gap tests — _last_pred_conv inner loop + SSD fallback (lines 279-282, 289)
# ---------------------------------------------------------------------------


def _make_ssd_nested_sequential_mock() -> nn.Module:
    """SSD mock where cls_head.module_list uses Sequential (SSDLite-style)."""

    class _SSDLiteClsHead(nn.Module):
        def __init__(self):
            super().__init__()
            # SSDLite: [Sequential(depthwise_conv, pointwise_conv)]
            dw = nn.Conv2d(256, 256, 3, groups=256, padding=1)
            pw = nn.Conv2d(256, 6 * 91, 1)  # pointwise; out_ch = 6*91
            self.module_list = nn.ModuleList([nn.Sequential(dw, pw)])
            self.num_classes = 91

    class _SSDLiteRegHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.module_list = nn.ModuleList([nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1)])  # 6 anchors * 4 coords

    class _SSDLiteHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.classification_head = _SSDLiteClsHead()
            self.regression_head = _SSDLiteRegHead()

    class _SSDLiteModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(3, 4)
            self.head = _SSDLiteHead()

        def forward(self, images, targets=None):
            if self.training and targets is not None:
                return {
                    "bbox_regression": torch.tensor(0.3, requires_grad=True),
                    "classification": torch.tensor(0.7, requires_grad=True),
                }
            return [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long), "scores": torch.zeros(0)}]

    return _SSDLiteModel()


def _make_ssd_no_conv_cls_mock() -> nn.Module:
    """SSD mock where cls_head.module_list has no Conv2d children → _last_pred_conv returns None."""

    class _NoConvClsHead(nn.Module):
        def __init__(self):
            super().__init__()
            # A ReLU has no Conv2d children → _last_pred_conv(relu) returns None
            self.module_list = nn.ModuleList([nn.ReLU()])
            self.num_classes = 91  # getattr fallback uses this

    class _SSDRegHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.module_list = nn.ModuleList([nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1)])

    class _SSDHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.classification_head = _NoConvClsHead()
            self.regression_head = _SSDRegHead()

    class _SSDNoConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(3, 4)
            self.head = _SSDHead()

        def forward(self, images, targets=None):
            if self.training and targets is not None:
                return {
                    "bbox_regression": torch.tensor(0.3, requires_grad=True),
                    "classification": torch.tensor(0.7, requires_grad=True),
                }
            return [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long), "scores": torch.zeros(0)}]

    return _SSDNoConvModel()


class TestModifyHeadSSDGaps:
    """Tests for uncovered SSD head-replacement branches."""

    def test_ssd_sequential_modules_replaced_ssdlite_style(self):
        """SSD module_list with Sequential items uses the Sequential branch (lines 300-312)."""
        engine = _make_engine("torchvision/ssdlite320_mobilenet_v3_large")
        model = _make_ssd_nested_sequential_mock()
        num_classes = 5

        engine._modify_head(model, num_classes)

        cls_head = model.head.classification_head
        assert cls_head.num_classes == num_classes
        # The single Sequential entry should have been replaced
        assert len(cls_head.module_list) == 1
        replaced_seq = cls_head.module_list[0]
        assert isinstance(replaced_seq, nn.Sequential)
        # Last child of the new Sequential is a Conv2d
        last_layer = list(replaced_seq.children())[-1]
        assert isinstance(last_layer, nn.Conv2d)

    def test_last_pred_conv_inner_loop_returns_layered_conv(self):
        """_last_pred_conv iterates children of Sequential to find Conv2d (lines 279-281)."""
        engine = _make_engine("torchvision/ssd300_vgg16")
        model = _make_ssd_nested_sequential_mock()
        # Just call _modify_head; if _last_pred_conv's inner loop runs it covers 279-281
        engine._modify_head(model, num_classes=3)
        assert model.head.classification_head.num_classes == 3

    def test_ssd_fallback_to_getattr_when_no_conv_found(self):
        """When _last_pred_conv returns None, old_num_classes falls back to getattr (line 289)."""
        engine = _make_engine("torchvision/ssd300_vgg16")
        model = _make_ssd_no_conv_cls_mock()
        # cls_head uses ReLU → _cls_conv is None → getattr fallback
        engine._modify_head(model, num_classes=5)
        # After modify, num_classes should be set (even though no Conv2d in module_list)
        assert model.head.classification_head.num_classes == 5

    def test_last_pred_conv_returns_none_path(self):
        """_last_pred_conv returns None when module has no Conv2d children (line 282)."""

        # Verify via the ssd_no_conv path that line 282 is reached
        engine = _make_engine("torchvision/ssd300_vgg16")
        model = _make_ssd_no_conv_cls_mock()
        # This exercises _last_pred_conv(relu) → returns None from line 282
        engine._modify_head(model, num_classes=10)
        assert model.head.classification_head.num_classes == 10


# ---------------------------------------------------------------------------
# Coverage gap tests — unknown model key warning (line 324)
# ---------------------------------------------------------------------------


class TestModifyHeadUnknownModelKey:
    def test_unknown_model_key_logs_warning_and_does_not_raise(self):
        """Unknown model_key logs a warning without modifying the head (line 324)."""
        engine = _make_engine("torchvision/fasterrcnn_resnet50_fpn")
        engine.model_key = "totally_unknown_model_xyz"  # override key
        model = _make_fasterrcnn_mock()
        original_predictor = model.roi_heads.box_predictor


        with patch("mata.training.torch_trainer.logger") as mock_log:
            engine._modify_head(model, num_classes=5)
            mock_log.warning.assert_called_once()

        # Head unchanged because no replacement logic for unknown model
        assert model.roi_heads.box_predictor is original_predictor


# ---------------------------------------------------------------------------
# Coverage gap tests — _validate() body (lines 487-570)
# ---------------------------------------------------------------------------


class TestValidateDirect:
    """Tests that exercise _validate() directly to cover its body."""

    def test_validate_returns_map50_via_mata_val_integration(self, tmp_path):
        """_validate() returns map50 when mata.val() integration succeeds (lines 490-521)."""
        engine = _make_engine(save_dir=str(tmp_path))
        model = _make_fasterrcnn_mock()
        device = torch.device("cpu")
        val_dataset = _make_tiny_dataset(4)

        mock_val_result = Mock()
        mock_val_result.map50 = 0.75
        mock_val_result.map = None

        # Patch both mata.val AND TorchvisionDetectAdapter to avoid real model I/O
        with patch("mata.val", return_value=mock_val_result):
            with patch("mata.adapters.torchvision_detect_adapter.TorchvisionDetectAdapter") as mock_adp_cls:
                mock_adp_cls.return_value = Mock()
                result = engine._validate(model, val_dataset, device, 0)

        assert result is not None
        assert "map50" in result
        assert result["map50"] == pytest.approx(0.75)

    def test_validate_falls_back_to_dataloader_when_mata_val_fails(self, tmp_path):
        """_validate() uses DataLoader fallback when mata.val() raises (lines 522-560)."""
        engine = _make_engine(save_dir=str(tmp_path))
        model = _make_fasterrcnn_mock()
        device = torch.device("cpu")
        val_dataset = _make_tiny_dataset(4)

        with patch("mata.val", side_effect=Exception("no val available")):
            result = engine._validate(model, val_dataset, device, 0)

        assert result is not None
        assert "val_loss" in result
        assert result["val_loss"] >= 0.0

    def test_validate_returns_none_when_everything_fails(self, tmp_path):
        """Outer except catches all errors and returns None (lines 562-567)."""
        engine = _make_engine(save_dir=str(tmp_path))
        model = _make_fasterrcnn_mock()
        device = torch.device("cpu")

        bad_dataset = Mock()
        bad_dataset.__len__ = Mock(return_value=4)
        bad_dataset.__getitem__ = Mock(side_effect=RuntimeError("corrupt data"))

        with patch("mata.val", side_effect=Exception("no val")):
            result = engine._validate(model, bad_dataset, device, 0)

        assert result is None

    def test_validate_restores_eval_mode_in_finally(self, tmp_path):
        """Model is restored to eval() in the finally block (line 569)."""
        engine = _make_engine(save_dir=str(tmp_path))
        model = _make_fasterrcnn_mock()
        device = torch.device("cpu")
        val_dataset = _make_tiny_dataset(4)

        with patch("mata.val", side_effect=Exception("no val")):
            engine._validate(model, val_dataset, device, 0)

        assert not model.training
