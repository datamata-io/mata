"""Tests for mata.train(), mata.finetune(), and checkpoint loading via mata.load().

Covers Task E7: Public API & Integration Tests.
All heavy I/O (engine.train, dataset loading, model inference) is mocked —
no real model downloads or disk writes occur beyond tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from mata.training.trainer import TrainingOrchestrator

import mata
from mata.core.exceptions import ConfigurationError
from mata.training.config import TrainingConfig
from mata.training.result import TrainingResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(**kwargs) -> TrainingResult:
    defaults = dict(epochs_completed=2)
    defaults.update(kwargs)
    return TrainingResult(**defaults)


def _minimal_train_kwargs(**overrides):
    """Return minimal keyword args for mata.train()."""
    defaults = dict(
        model="facebook/detr-resnet-50",
        data="coco.yaml",
        epochs=2,
        batch_size=2,
        lr=1e-4,
        warmup_epochs=0,
        num_workers=0,
        patience=0,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


class TestImportability:
    def test_train_importable_from_mata(self):
        from mata import train  # noqa: F401

    def test_finetune_importable_from_mata(self):
        from mata import finetune  # noqa: F401

    def test_train_in_mata_dunder_all(self):
        assert "train" in mata.__all__

    def test_finetune_in_mata_dunder_all(self):
        assert "finetune" in mata.__all__

    def test_mata_train_is_callable(self):
        assert callable(mata.train)

    def test_mata_finetune_is_callable(self):
        assert callable(mata.finetune)


# ---------------------------------------------------------------------------
# mata.train() — basic dispatch
# ---------------------------------------------------------------------------


class TestTrainAPI:
    """Tests that mata.train() correctly builds config and dispatches."""

    def _call_train(self, task="detect", **overrides):
        """Call mata.train() with mocked orchestrator, return (result, mock_orch)."""
        kwargs = _minimal_train_kwargs(**overrides)
        mock_result = _make_result()

        with patch("mata.api.TrainingOrchestrator") as MockOrch:
            MockOrch.return_value.train.return_value = mock_result
            with patch("mata.api.TrainingConfig") as MockConfig:
                instance = MagicMock()
                instance.validate.return_value = None
                MockConfig.return_value = instance
                result = mata.train(task, **kwargs)

        return result, MockOrch, MockConfig

    def test_train_returns_training_result(self, tmp_path):
        kwargs = _minimal_train_kwargs(save_dir=str(tmp_path / "runs"))
        mock_result = _make_result()

        with (
            patch("mata.training.TrainingOrchestrator") as MockOrch,
            patch("mata.training.TrainingConfig") as MockCfg,
        ):
            mock_cfg = MagicMock()
            mock_cfg.validate.return_value = None
            MockCfg.return_value = mock_cfg
            MockOrch.return_value.train.return_value = mock_result
            result = mata.train("detect", **kwargs)

        assert result is mock_result

    def test_train_constructs_training_config(self, tmp_path):
        kwargs = _minimal_train_kwargs(save_dir=str(tmp_path / "runs"))

        with (
            patch("mata.training.TrainingOrchestrator") as MockOrch,
            patch("mata.training.TrainingConfig") as MockCfg,
        ):
            mock_cfg = MagicMock()
            mock_cfg.validate.return_value = None
            MockCfg.return_value = mock_cfg
            MockOrch.return_value.train.return_value = _make_result()
            mata.train("detect", **kwargs)

        MockCfg.assert_called_once()
        call_kwargs = MockCfg.call_args[1]
        assert call_kwargs["task"] == "detect"
        assert call_kwargs["model"] == "facebook/detr-resnet-50"
        assert call_kwargs["data"] == "coco.yaml"

    def test_train_calls_validate(self, tmp_path):
        kwargs = _minimal_train_kwargs(save_dir=str(tmp_path / "runs"))

        with (
            patch("mata.training.TrainingOrchestrator") as MockOrch,
            patch("mata.training.TrainingConfig") as MockCfg,
        ):
            mock_cfg = MagicMock()
            MockCfg.return_value = mock_cfg
            MockOrch.return_value.train.return_value = _make_result()
            mata.train("detect", **kwargs)

        mock_cfg.validate.assert_called_once()

    def test_train_dispatches_to_orchestrator(self, tmp_path):
        kwargs = _minimal_train_kwargs(save_dir=str(tmp_path / "runs"))

        with (
            patch("mata.training.TrainingOrchestrator") as MockOrch,
            patch("mata.training.TrainingConfig") as MockCfg,
        ):
            mock_cfg = MagicMock()
            mock_cfg.validate.return_value = None
            MockCfg.return_value = mock_cfg
            MockOrch.return_value.train.return_value = _make_result()
            mata.train("detect", **kwargs)

        MockOrch.assert_called_once_with(mock_cfg)
        MockOrch.return_value.train.assert_called_once()

    def test_train_invalid_task_raises_configuration_error(self):
        """Invalid task → TrainingConfig.validate() raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="task"):
            mata.train(
                "invalid_task",
                model="facebook/detr-resnet-50",
                data="coco.yaml",
                warmup_epochs=0,
                num_workers=0,
                patience=0,
            )

    def test_train_missing_model_raises_type_error(self):
        """model is required keyword-only — omitting it must raise TypeError."""
        with pytest.raises(TypeError):
            mata.train("detect", data="coco.yaml")  # type: ignore[call-arg]

    def test_train_missing_data_raises_type_error(self):
        """data is required keyword-only — omitting it must raise TypeError."""
        with pytest.raises(TypeError):
            mata.train("detect", model="facebook/detr-resnet-50")  # type: ignore[call-arg]

    def test_train_default_values_passed_to_config(self, tmp_path):
        """Default hyperparameters are forwarded correctly to TrainingConfig."""
        with (
            patch("mata.training.TrainingOrchestrator") as MockOrch,
            patch("mata.training.TrainingConfig") as MockCfg,
        ):
            mock_cfg = MagicMock()
            mock_cfg.validate.return_value = None
            MockCfg.return_value = mock_cfg
            MockOrch.return_value.train.return_value = _make_result()
            mata.train(
                "detect",
                model="facebook/detr-resnet-50",
                data="coco.yaml",
                warmup_epochs=0,
                num_workers=0,
                patience=0,
            )

        call_kwargs = MockCfg.call_args[1]
        # Verify that default values inherited from the function signature are passed
        assert call_kwargs["epochs"] == 10
        assert call_kwargs["batch_size"] == 8
        assert call_kwargs["lr"] == pytest.approx(1e-4)
        assert call_kwargs["optimizer"] == "adamw"
        assert call_kwargs["scheduler"] == "cosine"
        assert call_kwargs["freeze_backbone"] is False


# ---------------------------------------------------------------------------
# mata.finetune() — defaults & delegation
# ---------------------------------------------------------------------------


class TestFinetuneAPI:
    """Tests that mata.finetune() enforces fine-tuning defaults and delegates."""

    def test_finetune_returns_training_result(self, tmp_path):
        mock_result = _make_result()
        with (
            patch("mata.training.TrainingOrchestrator") as MockOrch,
            patch("mata.training.TrainingConfig") as MockCfg,
        ):
            mock_cfg = MagicMock()
            mock_cfg.validate.return_value = None
            MockCfg.return_value = mock_cfg
            MockOrch.return_value.train.return_value = mock_result
            result = mata.finetune(
                "classify",
                model="microsoft/resnet-50",
                data="/data/flowers/",
                warmup_epochs=0,
                num_workers=0,
            )
        assert result is mock_result

    def test_finetune_default_lr_is_lower(self):
        """finetune() default lr must be 1e-5 (lower than train()'s 1e-4)."""
        with (
            patch("mata.training.TrainingOrchestrator") as MockOrch,
            patch("mata.training.TrainingConfig") as MockCfg,
        ):
            mock_cfg = MagicMock()
            mock_cfg.validate.return_value = None
            MockCfg.return_value = mock_cfg
            MockOrch.return_value.train.return_value = _make_result()
            mata.finetune(
                "classify",
                model="microsoft/resnet-50",
                data="data/",
                warmup_epochs=0,
                num_workers=0,
            )

        call_kwargs = MockCfg.call_args[1]
        assert call_kwargs["lr"] == pytest.approx(1e-5)

    def test_finetune_default_epochs_is_5(self):
        with (
            patch("mata.training.TrainingOrchestrator") as MockOrch,
            patch("mata.training.TrainingConfig") as MockCfg,
        ):
            mock_cfg = MagicMock()
            mock_cfg.validate.return_value = None
            MockCfg.return_value = mock_cfg
            MockOrch.return_value.train.return_value = _make_result()
            mata.finetune(
                "classify",
                model="microsoft/resnet-50",
                data="data/",
                warmup_epochs=0,
                num_workers=0,
            )

        call_kwargs = MockCfg.call_args[1]
        assert call_kwargs["epochs"] == 5

    def test_finetune_default_freeze_backbone_true(self):
        with (
            patch("mata.training.TrainingOrchestrator") as MockOrch,
            patch("mata.training.TrainingConfig") as MockCfg,
        ):
            mock_cfg = MagicMock()
            mock_cfg.validate.return_value = None
            MockCfg.return_value = mock_cfg
            MockOrch.return_value.train.return_value = _make_result()
            mata.finetune(
                "classify",
                model="microsoft/resnet-50",
                data="data/",
                warmup_epochs=0,
                num_workers=0,
            )

        call_kwargs = MockCfg.call_args[1]
        assert call_kwargs["freeze_backbone"] is True

    def test_finetune_default_batch_size_is_16(self):
        with (
            patch("mata.training.TrainingOrchestrator") as MockOrch,
            patch("mata.training.TrainingConfig") as MockCfg,
        ):
            mock_cfg = MagicMock()
            mock_cfg.validate.return_value = None
            MockCfg.return_value = mock_cfg
            MockOrch.return_value.train.return_value = _make_result()
            mata.finetune(
                "classify",
                model="microsoft/resnet-50",
                data="data/",
                warmup_epochs=0,
                num_workers=0,
            )

        call_kwargs = MockCfg.call_args[1]
        assert call_kwargs["batch_size"] == 16

    def test_finetune_kwargs_forwarded_to_train(self):
        """Extra kwargs passed to finetune() are forwarded to train()."""
        with (
            patch("mata.training.TrainingOrchestrator") as MockOrch,
            patch("mata.training.TrainingConfig") as MockCfg,
        ):
            mock_cfg = MagicMock()
            mock_cfg.validate.return_value = None
            MockCfg.return_value = mock_cfg
            MockOrch.return_value.train.return_value = _make_result()
            mata.finetune(
                "classify",
                model="microsoft/resnet-50",
                data="data/",
                warmup_epochs=0,
                num_workers=0,
                seed=99,  # extra kwarg
            )

        call_kwargs = MockCfg.call_args[1]
        assert call_kwargs["seed"] == 99

    def test_finetune_override_lr(self):
        """Explicit lr in finetune() overrides the default 1e-5."""
        with (
            patch("mata.training.TrainingOrchestrator") as MockOrch,
            patch("mata.training.TrainingConfig") as MockCfg,
        ):
            mock_cfg = MagicMock()
            mock_cfg.validate.return_value = None
            MockCfg.return_value = mock_cfg
            MockOrch.return_value.train.return_value = _make_result()
            mata.finetune(
                "classify",
                model="microsoft/resnet-50",
                data="data/",
                lr=3e-5,
                warmup_epochs=0,
                num_workers=0,
            )

        call_kwargs = MockCfg.call_args[1]
        assert call_kwargs["lr"] == pytest.approx(3e-5)

    def test_finetune_invalid_task_raises_configuration_error(self):
        with pytest.raises(ConfigurationError, match="task"):
            mata.finetune(
                "invalid_task",
                model="microsoft/resnet-50",
                data="data/",
                warmup_epochs=0,
                num_workers=0,
            )

    def test_finetune_missing_model_raises_type_error(self):
        with pytest.raises(TypeError):
            mata.finetune("classify", data="data/")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TrainingOrchestrator engine detection (via real orchestrator — no mocking)
# ---------------------------------------------------------------------------


class TestOrchestratorEngineDetection:
    """TrainingOrchestrator._detect_engine() — HF vs torchvision routing."""

    def _make_orch(self, model: str) -> TrainingOrchestrator:
        from mata.training.trainer import TrainingOrchestrator

        cfg = TrainingConfig(
            task="detect",
            model=model,
            data="coco.yaml",
            epochs=1,
            batch_size=2,
            lr=1e-4,
            warmup_epochs=0,
            num_workers=0,
            patience=0,
        )
        return TrainingOrchestrator(cfg)

    def test_hf_org_model_detected_as_huggingface(self):

        orch = self._make_orch("facebook/detr-resnet-50")
        assert orch._detect_engine("facebook/detr-resnet-50") == "huggingface"

    def test_torchvision_prefix_detected_as_torchvision(self):
        orch = self._make_orch("torchvision/fasterrcnn_resnet50_fpn")
        assert orch._detect_engine("torchvision/fasterrcnn_resnet50_fpn") == "torchvision"

    def test_torchvision_retinanet_detected(self):
        orch = self._make_orch("torchvision/retinanet_resnet50_fpn")
        assert orch._detect_engine("torchvision/retinanet_resnet50_fpn") == "torchvision"

    def test_another_hf_org_detected(self):
        orch = self._make_orch("microsoft/resnet-50")
        assert orch._detect_engine("microsoft/resnet-50") == "huggingface"


# ---------------------------------------------------------------------------
# Config alias resolution in TrainingOrchestrator
# ---------------------------------------------------------------------------


class TestOrchestratorAliasResolution:
    def test_alias_resolves_to_hf_engine(self):
        from mata.training.trainer import TrainingOrchestrator

        cfg = TrainingConfig(
            task="detect",
            model="my-detector",
            data="coco.yaml",
            epochs=1,
            batch_size=2,
            lr=1e-4,
            warmup_epochs=0,
            num_workers=0,
            patience=0,
        )
        orch = TrainingOrchestrator(cfg)

        with patch("mata.core.model_registry.ModelRegistry") as MockReg:
            registry = MockReg.return_value
            registry.has_alias.side_effect = lambda task, src: src == "my-detector"
            registry.get_config.return_value = {"source": "facebook/detr-resnet-50"}
            engine = orch._detect_engine("my-detector")

        assert engine == "huggingface"

    def test_alias_resolves_to_torchvision_engine(self):
        from mata.training.trainer import TrainingOrchestrator

        cfg = TrainingConfig(
            task="detect",
            model="my-fast-detector",
            data="coco.yaml",
            epochs=1,
            batch_size=2,
            lr=1e-4,
            warmup_epochs=0,
            num_workers=0,
            patience=0,
        )
        orch = TrainingOrchestrator(cfg)

        with patch("mata.core.model_registry.ModelRegistry") as MockReg:
            registry = MockReg.return_value
            registry.has_alias.side_effect = lambda task, src: src == "my-fast-detector"
            registry.get_config.return_value = {"source": "torchvision/fasterrcnn_resnet50_fpn"}
            engine = orch._detect_engine("my-fast-detector")

        assert engine == "torchvision"


# ---------------------------------------------------------------------------
# mata.load() checkpoint loading
# ---------------------------------------------------------------------------


class TestCheckpointLoading:
    """Tests for mata.load("detect", "<checkpoint_dir>") with trained models."""

    def _make_hf_checkpoint(self, tmp_path: Path, task: str = "detect") -> Path:
        """Create a minimal HuggingFace-style checkpoint directory."""
        ckpt = tmp_path / "best"
        ckpt.mkdir(parents=True)
        (ckpt / "config.json").write_text(
            json.dumps({"engine": "huggingface", "model_source": "facebook/detr-resnet-50"})
        )
        (ckpt / "model.safetensors").write_bytes(b"\x00" * 8)  # non-empty placeholder
        return ckpt

    def _make_torchvision_checkpoint(self, tmp_path: Path, task: str = "detect") -> Path:
        """Create a minimal torchvision-style checkpoint directory."""
        import torch

        ckpt = tmp_path / "best_tv"
        ckpt.mkdir(parents=True)
        (ckpt / "config.json").write_text(
            json.dumps(
                {
                    "engine": "torchvision",
                    "model_source": "torchvision/fasterrcnn_resnet50_fpn",
                }
            )
        )
        # Minimal state dict
        state = {"dummy": torch.tensor([1.0])}
        torch.save(state, ckpt / "model_state.pth")
        return ckpt

    def test_hf_checkpoint_detected_as_trained_checkpoint(self, tmp_path):
        """_is_checkpoint_dir() returns True for valid HF checkpoint dir."""
        from mata.core.model_loader import UniversalLoader

        ckpt = self._make_hf_checkpoint(tmp_path)
        loader = UniversalLoader()
        assert loader._is_checkpoint_dir(str(ckpt))

    def test_torchvision_checkpoint_detected(self, tmp_path):
        """_is_checkpoint_dir() returns True for torchvision checkpoint."""
        from mata.core.model_loader import UniversalLoader

        ckpt = self._make_torchvision_checkpoint(tmp_path)
        loader = UniversalLoader()
        assert loader._is_checkpoint_dir(str(ckpt))

    def test_plain_directory_not_checkpoint(self, tmp_path):
        """An ordinary directory (no config.json/weights) is not a checkpoint."""
        from mata.core.model_loader import UniversalLoader

        plain_dir = tmp_path / "not_a_checkpoint"
        plain_dir.mkdir()
        loader = UniversalLoader()
        assert not loader._is_checkpoint_dir(str(plain_dir))

    def test_dir_with_only_config_json_not_checkpoint(self, tmp_path):
        """config.json alone (no weights) is NOT a checkpoint."""
        from mata.core.model_loader import UniversalLoader

        d = tmp_path / "partial"
        d.mkdir()
        (d / "config.json").write_text("{}")
        loader = UniversalLoader()
        assert not loader._is_checkpoint_dir(str(d))

    def test_mata_load_routes_hf_checkpoint(self, tmp_path):
        """mata.load("detect", <hf_ckpt_dir>) calls _load_from_checkpoint."""
        ckpt = self._make_hf_checkpoint(tmp_path)

        mock_adapter = MagicMock()
        with patch("mata.core.model_loader.UniversalLoader._load_from_checkpoint") as mock_load:
            mock_load.return_value = mock_adapter
            adapter = mata.load("detect", str(ckpt))

        mock_load.assert_called_once_with("detect", str(ckpt))
        assert adapter is mock_adapter

    def test_mata_load_routes_torchvision_checkpoint(self, tmp_path):
        """mata.load("detect", <tv_ckpt_dir>) calls _load_from_checkpoint."""
        ckpt = self._make_torchvision_checkpoint(tmp_path)

        mock_adapter = MagicMock()
        with patch("mata.core.model_loader.UniversalLoader._load_from_checkpoint") as mock_load:
            mock_load.return_value = mock_adapter
            adapter = mata.load("detect", str(ckpt))

        mock_load.assert_called_once_with("detect", str(ckpt))
        assert adapter is mock_adapter

    def test_loaded_hf_checkpoint_inference(self, tmp_path):
        """Adapter loaded from checkpoint can make predictions."""
        import numpy as np

        from mata.core.types import VisionResult

        ckpt = self._make_hf_checkpoint(tmp_path)
        mock_adapter = MagicMock()
        dummy_result = VisionResult(instances=[])
        mock_adapter.predict.return_value = dummy_result

        with patch("mata.core.model_loader.UniversalLoader._load_from_checkpoint") as mock_load:
            mock_load.return_value = mock_adapter
            adapter = mata.load("detect", str(ckpt))

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = adapter.predict(image)
        mock_adapter.predict.assert_called_once_with(image)
        assert isinstance(result, VisionResult)

    def test_source_type_for_checkpoint_is_trained_checkpoint(self, tmp_path):
        """_detect_source_type() returns 'trained_checkpoint' for ckpt dirs."""
        from mata.core.model_loader import UniversalLoader

        ckpt = self._make_hf_checkpoint(tmp_path)
        loader = UniversalLoader()
        source_type, _ = loader._detect_source_type("detect", str(ckpt))
        assert source_type == "trained_checkpoint"


# ---------------------------------------------------------------------------
# Checkpoint round-trip: train → save → load → predict
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    """Verify the full train → checkpoint → load → predict lifecycle (mocked)."""

    def test_train_result_has_best_checkpoint(self, tmp_path):
        """After training, result.best_checkpoint is set."""
        ckpt_dir = str(tmp_path / "best")
        mock_result = TrainingResult(
            epochs_completed=2,
            best_checkpoint=ckpt_dir,
            last_checkpoint=ckpt_dir,
        )

        with (
            patch("mata.training.TrainingOrchestrator") as MockOrch,
            patch("mata.training.TrainingConfig") as MockCfg,
        ):
            mock_cfg = MagicMock()
            mock_cfg.validate.return_value = None
            MockCfg.return_value = mock_cfg
            MockOrch.return_value.train.return_value = mock_result
            result = mata.train(
                "detect",
                **_minimal_train_kwargs(save_dir=str(tmp_path / "runs")),
            )

        assert result.best_checkpoint == ckpt_dir

    def test_load_from_checkpoint_produces_predictions(self, tmp_path):
        """Checkpoint loaded via mata.load() produces VisionResult predictions."""
        import numpy as np

        from mata.core.types import VisionResult

        # Build a fake checkpoint directory
        ckpt = tmp_path / "best"
        ckpt.mkdir()
        (ckpt / "config.json").write_text(
            json.dumps({"engine": "huggingface", "model_source": "facebook/detr-resnet-50"})
        )
        (ckpt / "model.safetensors").write_bytes(b"\x00" * 8)

        mock_adapter = MagicMock()
        expected = VisionResult(instances=[])
        mock_adapter.predict.return_value = expected

        with patch("mata.core.model_loader.UniversalLoader._load_from_checkpoint") as mock_load:
            mock_load.return_value = mock_adapter
            adapter = mata.load("detect", str(ckpt))

        result = adapter.predict(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result is expected

    def test_full_round_trip_train_save_load_predict(self, tmp_path):
        """Simulate train() → checkpoint dir → mata.load() → predict() lifecycle."""
        import numpy as np

        from mata.core.types import VisionResult

        # Step 1: "train" produces a result with a checkpoint path
        ckpt = tmp_path / "runs" / "detect" / "best"
        ckpt.mkdir(parents=True)
        (ckpt / "config.json").write_text(
            json.dumps({"engine": "huggingface", "model_source": "facebook/detr-resnet-50"})
        )
        (ckpt / "model.safetensors").write_bytes(b"\x00" * 8)

        mock_train_result = TrainingResult(
            epochs_completed=2,
            best_checkpoint=str(ckpt),
            last_checkpoint=str(ckpt),
        )

        with (
            patch("mata.training.TrainingOrchestrator") as MockOrch,
            patch("mata.training.TrainingConfig") as MockCfg,
        ):
            mock_cfg = MagicMock()
            mock_cfg.validate.return_value = None
            MockCfg.return_value = mock_cfg
            MockOrch.return_value.train.return_value = mock_train_result
            train_result = mata.train(
                "detect",
                **_minimal_train_kwargs(save_dir=str(tmp_path / "runs")),
            )

        # Step 2: Load the checkpoint
        mock_adapter = MagicMock()
        mock_adapter.predict.return_value = VisionResult(instances=[])

        with patch("mata.core.model_loader.UniversalLoader._load_from_checkpoint") as mock_load:
            mock_load.return_value = mock_adapter
            adapter = mata.load("detect", train_result.best_checkpoint)

        # Step 3: Run inference
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = adapter.predict(image)

        assert isinstance(result, VisionResult)
        mock_adapter.predict.assert_called_once_with(image)


# ---------------------------------------------------------------------------
# Unsupported task error path validation (Task B1)
# ---------------------------------------------------------------------------


class TestUnsupportedTrainingTasks:
    """Verify that non-trainable tasks raise ConfigurationError with clear messages."""

    @pytest.mark.parametrize("task", ["depth", "ocr", "vlm", "track", "pose"])
    def test_train_unsupported_task_raises_configuration_error(self, task):
        with pytest.raises(ConfigurationError, match=r"Must be one of"):
            mata.train(task, model="dummy/model", data="dummy/path")

    @pytest.mark.parametrize("task", ["depth", "ocr", "vlm", "track", "pose"])
    def test_finetune_unsupported_task_raises_configuration_error(self, task):
        with pytest.raises(ConfigurationError, match=r"Must be one of"):
            mata.finetune(task, model="dummy/model", data="dummy/path")

    @pytest.mark.parametrize("task", ["depth", "ocr", "vlm", "track", "pose"])
    def test_error_message_lists_supported_tasks(self, task):
        with pytest.raises(ConfigurationError) as exc_info:
            mata.train(task, model="dummy/model", data="dummy/path")
        msg = str(exc_info.value)
        for supported in ("classify", "detect", "segment"):
            assert supported in msg
