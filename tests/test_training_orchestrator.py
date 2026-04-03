"""Tests for mata.training.trainer.TrainingOrchestrator.

All heavy lifting (engine.train, dataset building) is mocked — no real
model downloads or data loading occurs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mata.core.exceptions import TrainingError
from mata.training.config import TrainingConfig
from mata.training.result import TrainingResult
from mata.training.trainer import TrainingOrchestrator, _auto_save_dir

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
        save_dir="runs/test_orchestrator",
        num_workers=0,
        patience=0,
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def _make_result() -> TrainingResult:
    return TrainingResult(epochs_completed=2)


# ---------------------------------------------------------------------------
# _auto_save_dir
# ---------------------------------------------------------------------------


class TestAutoSaveDir:
    def test_returns_base_task_path_when_not_exists(self, tmp_path):
        result = _auto_save_dir(str(tmp_path / "runs"), "detect")
        assert result == tmp_path / "runs" / "detect"

    def test_increments_when_base_exists(self, tmp_path):
        base = tmp_path / "runs"
        (base / "detect").mkdir(parents=True)
        result = _auto_save_dir(str(base), "detect")
        assert result == base / "detect2"

    def test_increments_multiple_times(self, tmp_path):
        base = tmp_path / "runs"
        (base / "detect").mkdir(parents=True)
        (base / "detect2").mkdir(parents=True)
        (base / "detect3").mkdir(parents=True)
        result = _auto_save_dir(str(base), "detect")
        assert result == base / "detect4"

    def test_uses_task_name_in_path(self, tmp_path):
        result = _auto_save_dir(str(tmp_path), "classify")
        assert "classify" in str(result)


# ---------------------------------------------------------------------------
# Engine detection — _detect_engine
# ---------------------------------------------------------------------------


class TestDetectEngine:
    def test_huggingface_org_slash_model(self):
        orch = TrainingOrchestrator(_make_config(model="facebook/detr-resnet-50"))
        assert orch._detect_engine("facebook/detr-resnet-50") == "huggingface"

    def test_huggingface_other_org(self):
        orch = TrainingOrchestrator(_make_config(model="microsoft/resnet-50"))
        assert orch._detect_engine("microsoft/resnet-50") == "huggingface"

    def test_torchvision_prefix(self):
        orch = TrainingOrchestrator(_make_config(model="torchvision/fasterrcnn_resnet50_fpn"))
        assert orch._detect_engine("torchvision/fasterrcnn_resnet50_fpn") == "torchvision"

    def test_torchvision_other_model(self):
        orch = TrainingOrchestrator(_make_config(model="torchvision/retinanet_resnet50_fpn"))
        assert orch._detect_engine("torchvision/retinanet_resnet50_fpn") == "torchvision"

    def test_plain_file_raises(self):
        orch = TrainingOrchestrator(_make_config(model="model.pth"))
        with pytest.raises(TrainingError, match="Cannot train from a plain weight file"):
            orch._detect_engine("model.pth")

    def test_unknown_bare_string_raises(self):
        orch = TrainingOrchestrator(_make_config(model="unknownmodel"))
        with pytest.raises(TrainingError):
            orch._detect_engine("unknownmodel")


# ---------------------------------------------------------------------------
# Alias resolution — _resolve_alias
# ---------------------------------------------------------------------------


class TestResolveAlias:
    def test_non_alias_returned_unchanged(self):
        orch = TrainingOrchestrator(_make_config())
        with patch("mata.core.model_registry.ModelRegistry") as MockReg:
            MockReg.return_value.has_alias.return_value = False
            result = orch._resolve_alias("facebook/detr-resnet-50")
        assert result == "facebook/detr-resnet-50"

    def test_alias_resolved_to_underlying_source(self):
        orch = TrainingOrchestrator(_make_config(model="my-detector"))
        with patch("mata.core.model_registry.ModelRegistry") as MockReg:
            registry = MockReg.return_value
            # "my-detector" → "facebook/detr-resnet-50"
            registry.has_alias.side_effect = lambda task, src: src == "my-detector"
            registry.get_config.return_value = {"source": "facebook/detr-resnet-50"}
            result = orch._resolve_alias("my-detector")
        assert result == "facebook/detr-resnet-50"

    def test_alias_resolves_to_correct_engine(self):
        """Config alias that points to a torchvision model → 'torchvision' engine."""
        orch = TrainingOrchestrator(_make_config(model="my-tv-model"))
        with patch("mata.core.model_registry.ModelRegistry") as MockReg:
            registry = MockReg.return_value
            registry.has_alias.side_effect = lambda task, src: src == "my-tv-model"
            registry.get_config.return_value = {"source": "torchvision/fasterrcnn_resnet50_fpn"}
            engine = orch._detect_engine("my-tv-model")
        assert engine == "torchvision"

    def test_alias_chain_circular_stops(self):
        """Circular alias chain should not loop forever."""
        orch = TrainingOrchestrator(_make_config(model="a"))
        with patch("mata.core.model_registry.ModelRegistry") as MockReg:
            registry = MockReg.return_value
            registry.has_alias.return_value = True
            # Alias always points to itself
            registry.get_config.return_value = {"source": "a"}
            result = orch._resolve_alias("a")
        assert result == "a"


# ---------------------------------------------------------------------------
# Checkpoint directory detection
# ---------------------------------------------------------------------------


class TestCheckpointDetection:
    def test_hf_engine_from_checkpoint_config_json(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text(json.dumps({"engine": "huggingface", "model": "facebook/detr-resnet-50"}))
        orch = TrainingOrchestrator(_make_config())
        assert orch._engine_from_checkpoint(ckpt) == "huggingface"

    def test_torchvision_engine_from_checkpoint_config_json(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text(
            json.dumps({"engine": "torchvision", "model": "torchvision/fasterrcnn_resnet50_fpn"})
        )
        orch = TrainingOrchestrator(_make_config())
        assert orch._engine_from_checkpoint(ckpt) == "torchvision"

    def test_fallback_via_safetensors(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")  # no engine key
        (ckpt / "model.safetensors").write_bytes(b"")
        orch = TrainingOrchestrator(_make_config())
        assert orch._engine_from_checkpoint(ckpt) == "huggingface"

    def test_detect_engine_routes_checkpoint_dir(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text(json.dumps({"engine": "huggingface"}))
        orch = TrainingOrchestrator(_make_config())
        assert orch._detect_engine(str(ckpt)) == "huggingface"


# ---------------------------------------------------------------------------
# _write_config
# ---------------------------------------------------------------------------


class TestWriteConfig:
    def test_writes_yaml_with_all_fields(self, tmp_path):
        import yaml

        cfg = _make_config(task="classify", epochs=5)
        orch = TrainingOrchestrator(cfg)
        orch._write_config(tmp_path)

        config_file = tmp_path / "config.yaml"
        assert config_file.exists()
        with config_file.open() as fh:
            data = yaml.safe_load(fh)

        assert data["task"] == "classify"
        assert data["epochs"] == 5

    def test_config_yaml_uses_safe_dump(self, tmp_path):
        """Ensure no Python-specific YAML tags are written."""
        cfg = _make_config()
        orch = TrainingOrchestrator(cfg)
        orch._write_config(tmp_path)

        raw = (tmp_path / "config.yaml").read_text()
        assert "!!" not in raw  # no Python tags from yaml.dump


# ---------------------------------------------------------------------------
# _set_seeds
# ---------------------------------------------------------------------------


class TestSetSeeds:
    def test_sets_torch_seed(self):
        import torch

        TrainingOrchestrator._set_seeds(123)
        # Just verify it doesn't raise and torch manualSeed was applied
        assert torch.initial_seed() == 123

    def test_sets_random_seed(self):
        import random

        TrainingOrchestrator._set_seeds(42)
        val1 = random.random()
        TrainingOrchestrator._set_seeds(42)
        val2 = random.random()
        assert val1 == val2

    def test_sets_numpy_seed(self):
        import numpy as np

        TrainingOrchestrator._set_seeds(7)
        arr1 = np.random.rand(4)
        TrainingOrchestrator._set_seeds(7)
        arr2 = np.random.rand(4)
        assert (arr1 == arr2).all()


# ---------------------------------------------------------------------------
# train() — full orchestration (engines mocked)
# ---------------------------------------------------------------------------


class TestOrchestratorTrain:
    def _patch_all(self):
        """Context manager stack for mocking heavy I/O."""
        return [
            patch("mata.training.trainer.DatasetFactory") if False else None,
        ]

    def test_train_dispatches_to_hf_engine(self, tmp_path):
        cfg = _make_config(
            model="facebook/detr-resnet-50",
            data="coco.yaml",
            save_dir=str(tmp_path / "runs"),
        )
        orch = TrainingOrchestrator(cfg)
        mock_result = _make_result()

        with (
            patch.object(orch, "_build_datasets", return_value=(Mock(), None, Mock())),
            patch("mata.training.hf_trainer.HFTrainingEngine") as MockHF,
        ):
            MockHF.return_value.train.return_value = mock_result
            result = orch.train()

        MockHF.assert_called_once()
        assert result is mock_result

    def test_train_dispatches_to_torch_engine(self, tmp_path):
        cfg = _make_config(
            model="torchvision/fasterrcnn_resnet50_fpn",
            data="coco.yaml",
            save_dir=str(tmp_path / "runs"),
        )
        orch = TrainingOrchestrator(cfg)
        mock_result = _make_result()

        with (
            patch.object(orch, "_build_datasets", return_value=(Mock(), None, Mock())),
            patch("mata.training.torch_trainer.TorchTrainingEngine") as MockTV,
        ):
            MockTV.return_value.train.return_value = mock_result
            result = orch.train()

        MockTV.assert_called_once()
        assert result is mock_result

    def test_train_creates_save_dir(self, tmp_path):
        cfg = _make_config(
            model="facebook/detr-resnet-50",
            save_dir=str(tmp_path / "runs"),
        )
        orch = TrainingOrchestrator(cfg)

        with (
            patch.object(orch, "_build_datasets", return_value=(Mock(), None, Mock())),
            patch("mata.training.hf_trainer.HFTrainingEngine") as MockHF,
        ):
            MockHF.return_value.train.return_value = _make_result()
            orch.train()

        # Some directory under tmp_path/runs/detect must exist
        assert any((tmp_path / "runs").iterdir())

    def test_train_writes_config_yaml(self, tmp_path):
        cfg = _make_config(
            model="facebook/detr-resnet-50",
            save_dir=str(tmp_path / "runs"),
        )
        orch = TrainingOrchestrator(cfg)

        with (
            patch.object(orch, "_build_datasets", return_value=(Mock(), None, Mock())),
            patch("mata.training.hf_trainer.HFTrainingEngine") as MockHF,
        ):
            MockHF.return_value.train.return_value = _make_result()
            orch.train()

        config_files = list((tmp_path / "runs").rglob("config.yaml"))
        assert len(config_files) == 1

    def test_train_save_dir_auto_increments(self, tmp_path):
        base = tmp_path / "runs"
        cfg1 = _make_config(model="facebook/detr-resnet-50", save_dir=str(base))
        cfg2 = _make_config(model="facebook/detr-resnet-50", save_dir=str(base))

        for cfg in (cfg1, cfg2):
            orch = TrainingOrchestrator(cfg)
            with (
                patch.object(orch, "_build_datasets", return_value=(Mock(), None, Mock())),
                patch("mata.training.hf_trainer.HFTrainingEngine") as MockHF,
            ):
                MockHF.return_value.train.return_value = _make_result()
                orch.train()

        dirs = sorted(p.name for p in base.iterdir() if p.is_dir())
        assert "detect" in dirs
        assert "detect2" in dirs

    def test_train_sets_seeds(self, tmp_path):
        cfg = _make_config(model="facebook/detr-resnet-50", save_dir=str(tmp_path / "r"), seed=99)
        orch = TrainingOrchestrator(cfg)

        with (
            patch.object(orch, "_set_seeds") as mock_seeds,
            patch.object(orch, "_build_datasets", return_value=(Mock(), None, Mock())),
            patch("mata.training.hf_trainer.HFTrainingEngine") as MockHF,
        ):
            MockHF.return_value.train.return_value = _make_result()
            orch.train()

        mock_seeds.assert_called_once_with(99)

    def test_train_patches_save_dir_on_config(self, tmp_path):
        """The config.save_dir must be updated to the resolved directory."""
        cfg = _make_config(model="facebook/detr-resnet-50", save_dir=str(tmp_path / "runs"))
        orch = TrainingOrchestrator(cfg)

        with (
            patch.object(orch, "_build_datasets", return_value=(Mock(), None, Mock())),
            patch("mata.training.hf_trainer.HFTrainingEngine") as MockHF,
        ):
            MockHF.return_value.train.return_value = _make_result()
            orch.train()

        # After train(), config.save_dir should point to the concrete directory
        assert Path(cfg.save_dir).exists()


# ---------------------------------------------------------------------------
# finetune() — sets freeze_backbone=True
# ---------------------------------------------------------------------------


class TestOrchestratorFinetune:
    def test_finetune_sets_freeze_backbone(self, tmp_path):
        cfg = _make_config(model="facebook/detr-resnet-50", save_dir=str(tmp_path / "r"))
        assert cfg.freeze_backbone is False  # default

        orch = TrainingOrchestrator(cfg)
        with (
            patch.object(orch, "_build_datasets", return_value=(Mock(), None, Mock())),
            patch("mata.training.hf_trainer.HFTrainingEngine") as MockHF,
        ):
            MockHF.return_value.train.return_value = _make_result()
            orch.finetune()

        assert cfg.freeze_backbone is True

    def test_finetune_delegates_to_train(self, tmp_path):
        cfg = _make_config(model="facebook/detr-resnet-50", save_dir=str(tmp_path / "r"))
        orch = TrainingOrchestrator(cfg)

        with patch.object(orch, "train", return_value=_make_result()) as mock_train:
            orch.finetune()

        mock_train.assert_called_once()

    def test_finetune_returns_training_result(self, tmp_path):
        cfg = _make_config(model="facebook/detr-resnet-50", save_dir=str(tmp_path / "r"))
        orch = TrainingOrchestrator(cfg)

        with (
            patch.object(orch, "_build_datasets", return_value=(Mock(), None, Mock())),
            patch("mata.training.hf_trainer.HFTrainingEngine") as MockHF,
        ):
            MockHF.return_value.train.return_value = _make_result()
            result = orch.finetune()

        assert isinstance(result, TrainingResult)


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


class TestImportability:
    def test_import_from_training_package(self):
        from mata.training import TrainingOrchestrator  # noqa: F401

    def test_import_from_trainer_module(self):
        from mata.training.trainer import TrainingOrchestrator  # noqa: F401

    def test_in_all(self):
        import mata.training as training_pkg

        assert "TrainingOrchestrator" in training_pkg.__all__


# ---------------------------------------------------------------------------
# _resolve_alias — ImportError path (lines 100-101)
# ---------------------------------------------------------------------------


class TestResolveAliasImportError:
    def test_returns_source_unchanged_when_model_registry_import_fails(self):
        """If mata.core.model_registry cannot be imported, source is returned as-is."""
        orch = TrainingOrchestrator(_make_config(model="facebook/detr-resnet-50"))
        with patch.dict("sys.modules", {"mata.core.model_registry": None}):
            result = orch._resolve_alias("facebook/detr-resnet-50")
        assert result == "facebook/detr-resnet-50"

    def test_non_alias_unchanged_with_import_error(self):
        orch = TrainingOrchestrator(_make_config(model="torchvision/fasterrcnn_resnet50_fpn"))
        with patch.dict("sys.modules", {"mata.core.model_registry": None}):
            result = orch._resolve_alias("torchvision/fasterrcnn_resnet50_fpn")
        assert result == "torchvision/fasterrcnn_resnet50_fpn"


# ---------------------------------------------------------------------------
# _engine_from_checkpoint — additional fallback paths
# ---------------------------------------------------------------------------


class TestEngineFromCheckpointFallbacks:
    def test_pytorch_model_bin_fallback(self, tmp_path):
        """Checkpoint with pytorch_model.bin but no 'engine' in config.json → huggingface."""
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")
        (ckpt / "pytorch_model.bin").write_bytes(b"")
        orch = TrainingOrchestrator(_make_config())
        assert orch._engine_from_checkpoint(ckpt) == "huggingface"

    def test_json_decode_error_falls_back_to_safetensors(self, tmp_path):
        """Malformed config.json: falls back to file-based detection."""
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("NOT VALID JSON !!!")
        (ckpt / "model.safetensors").write_bytes(b"")
        orch = TrainingOrchestrator(_make_config())
        # Malformed JSON → logs warning; safetensors present → huggingface
        assert orch._engine_from_checkpoint(ckpt) == "huggingface"

    def test_json_decode_error_no_hf_files_returns_torchvision(self, tmp_path):
        """Malformed config.json and no HF files → torchvision (last resort)."""
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("NOT VALID JSON !!!")
        orch = TrainingOrchestrator(_make_config())
        assert orch._engine_from_checkpoint(ckpt) == "torchvision"

    def test_unknown_engine_in_config_json_uses_model_source(self, tmp_path):
        """config.json has unknown engine value → falls back to model source detection."""
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text(
            json.dumps({"engine": "unknown_engine", "model_source": "facebook/detr-resnet-50"})
        )
        orch = TrainingOrchestrator(_make_config())
        # model_source contains '/' → huggingface
        assert orch._engine_from_checkpoint(ckpt) == "huggingface"

    def test_unknown_engine_torchvision_model_source(self, tmp_path):
        """config.json with unknown engine and torchvision model_source → torchvision."""
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text(json.dumps({"engine": "X", "model": "torchvision/fasterrcnn_resnet50_fpn"}))
        orch = TrainingOrchestrator(_make_config())
        assert orch._engine_from_checkpoint(ckpt) == "torchvision"


# ---------------------------------------------------------------------------
# _build_datasets — direct tests (lines 232-275)
# ---------------------------------------------------------------------------


class TestBuildDatasets:
    """Direct tests for TrainingOrchestrator._build_datasets()."""

    def test_build_datasets_returns_train_val_and_collate(self, tmp_path):
        cfg = _make_config(
            model="facebook/detr-resnet-50",
            data="coco.yaml",
            save_dir=str(tmp_path),
        )
        orch = TrainingOrchestrator(cfg)
        mock_train_ds = Mock()
        mock_val_ds = Mock()
        mock_collate = Mock()

        with (
            patch("mata.training.datasets.factory.DatasetFactory.create") as mock_create,
            patch("mata.training.augmentations.factory.AugmentationFactory.create", return_value=None),
        ):
            mock_create.side_effect = [
                (mock_train_ds, mock_collate),  # train split
                (mock_val_ds, mock_collate),  # val split
            ]
            train_ds, val_ds, collate_fn = orch._build_datasets(tmp_path)

        assert train_ds is mock_train_ds
        assert val_ds is mock_val_ds
        assert collate_fn is mock_collate

    def test_build_datasets_no_augmentation(self, tmp_path):
        cfg = _make_config(augment=False)
        orch = TrainingOrchestrator(cfg)
        mock_ds = Mock()
        mock_collate = Mock()

        with (
            patch("mata.training.datasets.factory.DatasetFactory.create") as mock_create,
            patch("mata.training.augmentations.factory.AugmentationFactory.create") as mock_aug,
        ):
            mock_create.side_effect = [
                (mock_ds, mock_collate),
                (mock_ds, mock_collate),
            ]
            orch._build_datasets(tmp_path)

        # AugmentationFactory.create should NOT be called when augment=False
        mock_aug.assert_not_called()

    def test_build_datasets_with_augmentation(self, tmp_path):
        cfg = _make_config(augment=True)
        orch = TrainingOrchestrator(cfg)
        mock_ds = Mock()
        mock_collate = Mock()
        mock_aug = Mock()

        with (
            patch("mata.training.datasets.factory.DatasetFactory.create") as mock_create,
            patch(
                "mata.training.augmentations.factory.AugmentationFactory.create", return_value=mock_aug
            ) as mock_aug_factory,
        ):
            mock_create.side_effect = [
                (mock_ds, mock_collate),
                (mock_ds, mock_collate),
            ]
            orch._build_datasets(tmp_path)

        # AugmentationFactory called for both train and val when augment=True
        assert mock_aug_factory.call_count == 2

    def test_build_datasets_val_failure_returns_none(self, tmp_path):
        """If building val dataset raises, returns None gracefully (lines 267-274)."""
        cfg = _make_config(data="coco.yaml")
        orch = TrainingOrchestrator(cfg)
        mock_train_ds = Mock()
        mock_collate = Mock()

        def _create_side_effect(task, source, split, transforms=None):
            if split == "train":
                return (mock_train_ds, mock_collate)
            raise RuntimeError("Val data not found")

        with (
            patch("mata.training.datasets.factory.DatasetFactory.create", side_effect=_create_side_effect),
            patch("mata.training.augmentations.factory.AugmentationFactory.create", return_value=None),
        ):
            train_ds, val_ds, collate_fn = orch._build_datasets(tmp_path)

        assert train_ds is mock_train_ds
        assert val_ds is None  # graceful fallback

    def test_build_datasets_uses_val_data_config(self, tmp_path):
        """When config.val_data is set, it is used for val split (not config.data)."""
        cfg = _make_config(data="coco_train.yaml", val_data="coco_val.yaml")
        orch = TrainingOrchestrator(cfg)
        mock_ds = Mock()
        create_calls = []

        def _create_side_effect(task, source, split, transforms=None):
            create_calls.append((source, split))
            return (mock_ds, Mock())

        with (
            patch("mata.training.datasets.factory.DatasetFactory.create", side_effect=_create_side_effect),
            patch("mata.training.augmentations.factory.AugmentationFactory.create", return_value=None),
        ):
            orch._build_datasets(tmp_path)

        sources = [s for s, _ in create_calls]
        assert "coco_val.yaml" in sources

    def test_dispatch_unknown_engine_raises_training_error(self, tmp_path):
        """_dispatch with an unrecognised engine name raises TrainingError."""
        from mata.core.exceptions import TrainingError

        cfg = _make_config(model="facebook/detr-resnet-50", save_dir=str(tmp_path))
        orch = TrainingOrchestrator(cfg)

        with pytest.raises(TrainingError, match="Unknown engine"):
            orch._dispatch("unknown_engine", Mock(), None)
