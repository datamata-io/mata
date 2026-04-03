"""Tests for mata.training.checkpoint — CheckpointManager save/load/export flow."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from mata.core.exceptions import TrainingError
from mata.training.checkpoint import (
    CheckpointManager,
    _config_to_dict,
    _extract_scalar_metric,
)
from mata.training.result import TrainingResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(in_features: int = 4, out_features: int = 2) -> nn.Module:
    """Return a tiny deterministic linear model for tests."""
    return nn.Linear(in_features, out_features)


def _make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


def _make_scheduler(optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)


def _make_config(**kwargs) -> MagicMock:
    defaults = {
        "task": "detect",
        "model": "facebook/detr-resnet-50",
        "data": "coco.yaml",
        "val_data": None,
        "epochs": 10,
        "batch_size": 8,
        "lr": 1e-4,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "warmup_epochs": 1,
        "device": "cpu",
        "amp": False,
        "save_dir": "runs/train",
        "save_every": 0,
        "val_every": 1,
        "patience": 0,
        "freeze_backbone": False,
        "freeze_layers": None,
        "augment": True,
        "resume": None,
        "num_workers": 0,
        "seed": 42,
        "verbose": True,
        "engine": "huggingface",
        # history must be a real dict so json.dump succeeds (MagicMock auto-attrs
        # would return another MagicMock which is not JSON-serialisable)
        "history": {},
    }
    defaults.update(kwargs)
    cfg = MagicMock()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


# =============================================================================
# save()
# =============================================================================


class TestCheckpointManagerSave:
    def test_creates_four_expected_files(self, tmp_path):
        mgr = CheckpointManager()
        model = _make_model()
        opt = _make_optimizer(model)
        sched = _make_scheduler(opt)
        cfg = _make_config()

        ckpt_dir = mgr.save(model, opt, sched, epoch=1, metrics=0.5, config=cfg, path=tmp_path / "ckpt")

        assert (ckpt_dir / "model_state.pth").exists()
        assert (ckpt_dir / "optimizer_state.pth").exists()
        assert (ckpt_dir / "training_state.json").exists()
        assert (ckpt_dir / "config.json").exists()

    def test_returns_resolved_path(self, tmp_path):
        mgr = CheckpointManager()
        model = _make_model()
        opt = _make_optimizer(model)
        cfg = _make_config()

        result = mgr.save(model, opt, None, epoch=0, metrics=None, config=cfg, path=tmp_path / "ckpt1")

        assert isinstance(result, Path)
        assert result.is_dir()

    def test_creates_directory_if_missing(self, tmp_path):
        mgr = CheckpointManager()
        model = _make_model()
        cfg = _make_config()
        deep_path = tmp_path / "a" / "b" / "c"

        mgr.save(model, None, None, epoch=0, metrics=None, config=cfg, path=deep_path)

        assert deep_path.is_dir()

    def test_training_state_json_content(self, tmp_path):
        mgr = CheckpointManager()
        model = _make_model()
        cfg = _make_config()

        ckpt_dir = mgr.save(model, None, None, epoch=5, metrics=0.75, config=cfg, path=tmp_path / "ckpt")

        with open(ckpt_dir / "training_state.json") as f:
            state = json.load(f)

        assert state["epoch"] == 5
        assert pytest.approx(state["best_metric"]) == 0.75

    def test_config_json_has_task_and_model(self, tmp_path):
        mgr = CheckpointManager()
        model = _make_model()
        cfg = _make_config(task="classify", model="microsoft/resnet-50", engine="huggingface")

        ckpt_dir = mgr.save(model, None, None, epoch=0, metrics=None, config=cfg, path=tmp_path / "ckpt")

        with open(ckpt_dir / "config.json") as f:
            data = json.load(f)

        assert data["task"] == "classify"
        assert data["engine"] == "huggingface"

    def test_save_with_none_optimizer_creates_empty_opt_state(self, tmp_path):
        mgr = CheckpointManager()
        model = _make_model()
        cfg = _make_config()

        ckpt_dir = mgr.save(model, None, None, epoch=0, metrics=None, config=cfg, path=tmp_path / "ckpt")

        # File should still be created (empty dict)
        assert (ckpt_dir / "optimizer_state.pth").exists()
        loaded = torch.load(ckpt_dir / "optimizer_state.pth", weights_only=False)
        assert loaded == {}

    def test_save_state_dict_matches_model_weights(self, tmp_path):
        mgr = CheckpointManager()
        model = _make_model()
        orig_state = {k: v.clone() for k, v in model.state_dict().items()}
        cfg = _make_config()

        ckpt_dir = mgr.save(model, None, None, epoch=0, metrics=None, config=cfg, path=tmp_path / "ckpt")

        loaded_state = torch.load(ckpt_dir / "model_state.pth", weights_only=True)
        for key in orig_state:
            assert torch.equal(orig_state[key], loaded_state[key])


# =============================================================================
# load()
# =============================================================================


class TestCheckpointManagerLoad:
    def _save_checkpoint(self, tmp_path, model, opt=None, sched=None, epoch=1, metrics=0.5, **cfg_kwargs):
        mgr = CheckpointManager()
        cfg = _make_config(**cfg_kwargs)
        return mgr.save(model, opt, sched, epoch=epoch, metrics=metrics, config=cfg, path=tmp_path / "ckpt")

    def test_load_returns_dict_with_required_keys(self, tmp_path):
        model = _make_model()
        ckpt_dir = self._save_checkpoint(tmp_path, model)
        mgr = CheckpointManager()

        result = mgr.load(ckpt_dir)

        assert "model_state" in result
        assert "optimizer_state" in result
        assert "training_state" in result
        assert "config" in result

    def test_model_state_round_trip(self, tmp_path):
        """Save a model then load its state_dict — weights must match."""
        orig_model = _make_model()
        ckpt_dir = self._save_checkpoint(tmp_path, orig_model)

        mgr = CheckpointManager()
        loaded = mgr.load(ckpt_dir)

        new_model = _make_model()
        new_model.load_state_dict(loaded["model_state"])

        for key in orig_model.state_dict():
            assert torch.equal(orig_model.state_dict()[key], new_model.state_dict()[key])

    def test_optimizer_state_round_trip(self, tmp_path):
        model = _make_model()
        opt = _make_optimizer(model)
        # Simulate a training step so optimizer has non-trivial state
        loss = model(torch.zeros(1, 4)).sum()
        loss.backward()
        opt.step()

        ckpt_dir = self._save_checkpoint(tmp_path, model, opt=opt)
        mgr = CheckpointManager()
        loaded = mgr.load(ckpt_dir)

        assert "optimizer" in loaded["optimizer_state"]
        opt_state = loaded["optimizer_state"]["optimizer"]
        # Verify state group lr matches
        assert opt_state["param_groups"][0]["lr"] == pytest.approx(1e-3)

    def test_scheduler_state_round_trip(self, tmp_path):
        model = _make_model()
        opt = _make_optimizer(model)
        sched = _make_scheduler(opt)
        sched.step()  # advance once so last_epoch != 0

        ckpt_dir = self._save_checkpoint(tmp_path, model, opt=opt, sched=sched)
        mgr = CheckpointManager()
        loaded = mgr.load(ckpt_dir)

        assert "scheduler" in loaded["optimizer_state"]
        assert loaded["optimizer_state"]["scheduler"]["last_epoch"] == 1

    def test_training_state_restored(self, tmp_path):
        model = _make_model()
        ckpt_dir = self._save_checkpoint(tmp_path, model, epoch=7, metrics=0.88)
        mgr = CheckpointManager()
        loaded = mgr.load(ckpt_dir)

        ts = loaded["training_state"]
        assert ts["epoch"] == 7
        assert pytest.approx(ts["best_metric"]) == 0.88

    def test_raises_training_error_if_directory_missing(self, tmp_path):
        mgr = CheckpointManager()
        with pytest.raises(TrainingError, match="not found"):
            mgr.load(tmp_path / "nonexistent")

    def test_raises_training_error_if_model_state_missing(self, tmp_path):
        model = _make_model()
        ckpt_dir = self._save_checkpoint(tmp_path, model)
        (ckpt_dir / "model_state.pth").unlink()

        mgr = CheckpointManager()
        with pytest.raises(TrainingError, match="model_state.pth not found"):
            mgr.load(ckpt_dir)

    def test_raises_training_error_if_training_state_json_missing(self, tmp_path):
        model = _make_model()
        ckpt_dir = self._save_checkpoint(tmp_path, model)
        (ckpt_dir / "training_state.json").unlink()

        mgr = CheckpointManager()
        with pytest.raises(TrainingError, match="training_state.json not found"):
            mgr.load(ckpt_dir)

    def test_raises_training_error_on_corrupt_model_state(self, tmp_path):
        model = _make_model()
        ckpt_dir = self._save_checkpoint(tmp_path, model)
        # Overwrite with corrupt data
        (ckpt_dir / "model_state.pth").write_bytes(b"\x00\x01\x02corrupt")

        mgr = CheckpointManager()
        with pytest.raises(TrainingError, match="Failed to load model_state.pth"):
            mgr.load(ckpt_dir)

    def test_raises_training_error_on_corrupt_training_state_json(self, tmp_path):
        model = _make_model()
        ckpt_dir = self._save_checkpoint(tmp_path, model)
        (ckpt_dir / "training_state.json").write_text("{ not valid json }", encoding="utf-8")

        mgr = CheckpointManager()
        with pytest.raises(TrainingError, match="Failed to parse training_state.json"):
            mgr.load(ckpt_dir)

    def test_config_absent_returns_empty_dict(self, tmp_path):
        """Older checkpoints may not have config.json — should not crash."""
        model = _make_model()
        ckpt_dir = self._save_checkpoint(tmp_path, model)
        (ckpt_dir / "config.json").unlink()

        mgr = CheckpointManager()
        loaded = mgr.load(ckpt_dir)

        assert loaded["config"] == {}

    def test_weights_only_true_used_for_model_state(self, tmp_path):
        """Verify torch.load is called with weights_only=True for model weights."""
        model = _make_model()
        ckpt_dir = self._save_checkpoint(tmp_path, model)
        mgr = CheckpointManager()

        real_torch_load = torch.load
        calls_kwargs = []

        def spy_load(*args, **kwargs):
            calls_kwargs.append(kwargs)
            return real_torch_load(*args, **kwargs)

        with patch("torch.load", side_effect=spy_load):
            mgr.load(ckpt_dir)

        # First call (model_state.pth) must use weights_only=True
        assert calls_kwargs[0].get("weights_only") is True


# =============================================================================
# export_for_inference()
# =============================================================================


class TestCheckpointManagerExport:
    def _make_ckpt(self, tmp_path, engine="huggingface", **cfg_overrides):
        mgr = CheckpointManager()
        model = _make_model()
        cfg = _make_config(engine=engine, **cfg_overrides)
        ckpt_dir = mgr.save(model, None, None, epoch=1, metrics=0.5, config=cfg, path=tmp_path / "ckpt")
        return ckpt_dir, model

    def test_hf_export_calls_save_pretrained(self, tmp_path):
        ckpt_dir, _ = self._make_ckpt(tmp_path, engine="huggingface")
        hf_model = MagicMock()
        hf_model.save_pretrained = MagicMock()

        mgr = CheckpointManager()
        out_dir = tmp_path / "export"
        mgr.export_for_inference(ckpt_dir, out_dir, model=hf_model)

        hf_model.save_pretrained.assert_called_once_with(out_dir)

    def test_hf_export_with_processor_calls_processor_save_pretrained(self, tmp_path):
        ckpt_dir, _ = self._make_ckpt(tmp_path, engine="huggingface")
        hf_model = MagicMock()
        processor = MagicMock()

        mgr = CheckpointManager()
        out_dir = tmp_path / "export"
        mgr.export_for_inference(ckpt_dir, out_dir, model=hf_model, processor=processor)

        processor.save_pretrained.assert_called_once_with(out_dir)

    def test_hf_export_without_model_copies_state_dict(self, tmp_path):
        ckpt_dir, _ = self._make_ckpt(tmp_path, engine="huggingface")
        out_dir = tmp_path / "export"

        mgr = CheckpointManager()
        mgr.export_for_inference(ckpt_dir, out_dir, model=None)

        # Raw state dict should be copied over
        assert (out_dir / "model_state.pth").exists()

    def test_torchvision_export_creates_model_pth(self, tmp_path):
        ckpt_dir, model = self._make_ckpt(tmp_path, engine="torchvision", model="torchvision/fasterrcnn_resnet50_fpn")

        out_dir = tmp_path / "export"
        mgr = CheckpointManager()
        mgr.export_for_inference(ckpt_dir, out_dir, model=model)

        assert (out_dir / "model.pth").exists()

    def test_torchvision_export_creates_metadata_json(self, tmp_path):
        ckpt_dir, model = self._make_ckpt(tmp_path, engine="torchvision", model="torchvision/fasterrcnn_resnet50_fpn")

        out_dir = tmp_path / "export"
        mgr = CheckpointManager()
        mgr.export_for_inference(ckpt_dir, out_dir, model=model)

        assert (out_dir / "metadata.json").exists()
        with open(out_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["engine"] == "torchvision"

    def test_torchvision_export_without_model_copies_pth(self, tmp_path):
        ckpt_dir, _ = self._make_ckpt(tmp_path, engine="torchvision")
        out_dir = tmp_path / "export"

        mgr = CheckpointManager()
        mgr.export_for_inference(ckpt_dir, out_dir, model=None)

        assert (out_dir / "model.pth").exists()

    def test_export_returns_output_path(self, tmp_path):
        ckpt_dir, model = self._make_ckpt(tmp_path, engine="huggingface")
        hf_model = MagicMock()
        out_dir = tmp_path / "export"

        mgr = CheckpointManager()
        result = mgr.export_for_inference(ckpt_dir, out_dir, model=hf_model)

        assert isinstance(result, Path)
        assert result == out_dir.resolve() or result == out_dir

    def test_export_creates_output_directory(self, tmp_path):
        ckpt_dir, _ = self._make_ckpt(tmp_path, engine="torchvision")
        out_dir = tmp_path / "deep" / "export"
        assert not out_dir.exists()

        mgr = CheckpointManager()
        mgr.export_for_inference(ckpt_dir, out_dir, model=None)

        assert out_dir.is_dir()


# =============================================================================
# list_checkpoints()
# =============================================================================


class TestCheckpointManagerListCheckpoints:
    def test_returns_sorted_list(self, tmp_path):
        mgr = CheckpointManager()
        model = _make_model()
        cfg = _make_config()

        paths = []
        for name in ("epoch3", "epoch1", "epoch2"):
            ckpt_dir = mgr.save(model, None, None, epoch=0, metrics=None, config=cfg, path=tmp_path / name)
            paths.append(str(ckpt_dir))

        result = mgr.list_checkpoints(tmp_path)

        assert result == sorted(paths)

    def test_returns_empty_list_for_missing_dir(self, tmp_path):
        mgr = CheckpointManager()
        result = mgr.list_checkpoints(tmp_path / "nonexistent")
        assert result == []

    def test_ignores_dirs_without_model_state_pth(self, tmp_path):
        mgr = CheckpointManager()
        model = _make_model()
        cfg = _make_config()

        # Valid checkpoint
        ckpt_dir = mgr.save(model, None, None, epoch=0, metrics=None, config=cfg, path=tmp_path / "valid")
        # Non-checkpoint directory (no model_state.pth)
        bad_dir = tmp_path / "not_a_checkpoint"
        bad_dir.mkdir()
        (bad_dir / "some_file.txt").write_text("hello")

        result = mgr.list_checkpoints(tmp_path)

        assert len(result) == 1
        assert str(ckpt_dir) in result[0] or result[0].endswith("valid")

    def test_returns_only_checkpoint_directories(self, tmp_path):
        mgr = CheckpointManager()
        model = _make_model()
        cfg = _make_config()

        # Save 3 checkpoints
        for i in range(3):
            mgr.save(model, None, None, epoch=i, metrics=None, config=cfg, path=tmp_path / f"ckpt{i}")

        result = mgr.list_checkpoints(tmp_path)

        assert len(result) == 3
        for p in result:
            assert (Path(p) / "model_state.pth").exists()


# =============================================================================
# Resume: save → load → state consistent
# =============================================================================


class TestCheckpointResume:
    def test_resume_state_consistent(self, tmp_path):
        """Save at epoch 5, load back, verify epoch and metric are intact."""
        mgr = CheckpointManager()
        model = _make_model()
        opt = _make_optimizer(model)
        cfg = _make_config()

        ckpt_dir = mgr.save(model, opt, None, epoch=5, metrics=0.65, config=cfg, path=tmp_path / "resume_ckpt")
        loaded = mgr.load(ckpt_dir)

        assert loaded["training_state"]["epoch"] == 5
        assert pytest.approx(loaded["training_state"]["best_metric"]) == 0.65

        # Restore model weights and verify
        new_model = _make_model()
        new_model.load_state_dict(loaded["model_state"])
        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], new_model.state_dict()[key])

        # Restore optimizer and verify
        new_opt = _make_optimizer(new_model)
        new_opt.load_state_dict(loaded["optimizer_state"]["optimizer"])
        assert new_opt.state_dict()["param_groups"][0]["lr"] == pytest.approx(1e-3)

    def test_multiple_save_load_cycles(self, tmp_path):
        """Simulate a multi-epoch training loop with periodic checkpointing."""
        mgr = CheckpointManager()
        model = _make_model()
        opt = _make_optimizer(model)
        cfg = _make_config()

        for epoch in range(1, 4):
            # Simulate weight update
            loss = model(torch.zeros(1, 4)).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

            mgr.save(
                model, opt, None, epoch=epoch, metrics=epoch * 0.1, config=cfg, path=tmp_path / f"epoch{epoch}"
            )

        # Load the last checkpoint and verify
        last_ckpt = tmp_path / "epoch3"
        loaded = mgr.load(last_ckpt)
        assert loaded["training_state"]["epoch"] == 3
        assert pytest.approx(loaded["training_state"]["best_metric"]) == 0.3


# =============================================================================
# TrainingResult dataclass
# =============================================================================


class TestTrainingResult:
    def test_default_construction(self):
        result = TrainingResult()
        assert result.best_metrics is None
        assert result.final_metrics is None
        assert result.best_checkpoint == ""
        assert result.last_checkpoint == ""
        assert result.history == {}
        assert result.config is None
        assert result.epochs_completed == 0

    def test_history_is_mutable_list_dict(self):
        result = TrainingResult()
        result.history["train_loss"] = [0.8, 0.6, 0.4]
        assert result.history["train_loss"] == [0.8, 0.6, 0.4]

    def test_summary_returns_string(self):
        cfg = MagicMock()
        cfg.task = "detect"
        cfg.model = "facebook/detr-resnet-50"
        result = TrainingResult(
            epochs_completed=5,
            best_checkpoint="runs/train/detect/best",
            last_checkpoint="runs/train/detect/last",
            history={"train_loss": [0.8, 0.6, 0.4, 0.3, 0.25]},
            config=cfg,
        )
        summary = result.summary()
        assert isinstance(summary, str)
        assert "5" in summary
        assert "detect" in summary
        assert "runs/train/detect/best" in summary

    def test_summary_works_without_matplotlib(self):
        """summary() must never require matplotlib."""
        result = TrainingResult(epochs_completed=3)
        # Must not raise ImportError even on headless CI
        out = result.summary()
        assert "3" in out

    def test_best_checkpoint_and_last_checkpoint_fields(self):
        result = TrainingResult(
            best_checkpoint="runs/best",
            last_checkpoint="runs/last",
        )
        assert result.best_checkpoint == "runs/best"
        assert result.last_checkpoint == "runs/last"

    def test_epochs_completed_field(self):
        result = TrainingResult(epochs_completed=10)
        assert result.epochs_completed == 10

    def test_plot_loss_raises_import_error_if_matplotlib_missing(self):
        result = TrainingResult(history={"train_loss": [0.5, 0.4]})
        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
            with pytest.raises(ImportError, match="matplotlib"):
                result.plot_loss()

    def test_history_train_loss_accessible(self):
        """result.history['train_loss'] returns list of per-epoch losses."""
        losses = [0.9, 0.7, 0.5]
        result = TrainingResult(history={"train_loss": losses})
        assert result.history["train_loss"] == losses


# =============================================================================
# _extract_scalar_metric helper
# =============================================================================


class TestExtractScalarMetric:
    def test_returns_none_for_none(self):
        assert _extract_scalar_metric(None) is None

    def test_returns_float_from_float(self):
        assert _extract_scalar_metric(0.75) == pytest.approx(0.75)

    def test_returns_float_from_int(self):
        assert _extract_scalar_metric(1) == pytest.approx(1.0)

    def test_extracts_map50_from_dict(self):
        assert _extract_scalar_metric({"map50": 0.6, "map": 0.4}) == pytest.approx(0.6)

    def test_extracts_top1_from_dict(self):
        assert _extract_scalar_metric({"top1": 0.92}) == pytest.approx(0.92)

    def test_returns_none_for_unrecognised_dict(self):
        assert _extract_scalar_metric({"unknown_key": 0.5}) is None

    def test_extracts_map50_from_object(self):
        metrics = MagicMock()
        metrics.map50 = 0.55
        assert _extract_scalar_metric(metrics) == pytest.approx(0.55)


# =============================================================================
# _config_to_dict helper
# =============================================================================


class TestConfigToDict:
    def test_passes_through_plain_dict(self):
        cfg = {"task": "detect", "epochs": 10, "model": "some/model"}
        result = _config_to_dict(cfg)
        assert result["task"] == "detect"
        assert result["epochs"] == 10

    def test_reads_attributes_from_object(self):
        cfg = MagicMock()
        cfg.task = "classify"
        cfg.model = "microsoft/resnet-50"
        cfg.engine = "huggingface"
        result = _config_to_dict(cfg)
        assert result["task"] == "classify"
        assert result["engine"] == "huggingface"

    def test_model_source_alias_set(self):
        # model_source aliasing happens on the object path (not the plain dict path)
        cfg = MagicMock()
        cfg.task = "detect"
        cfg.model = "facebook/detr-resnet-50"
        cfg.engine = "huggingface"
        # ensure model_source is absent so the alias logic runs
        del cfg.model_source
        result = _config_to_dict(cfg)
        assert result.get("model_source") == "facebook/detr-resnet-50"

    def test_non_serialisable_values_excluded(self):
        """Non-JSON-serialisable values (e.g. tensors) should be dropped."""
        cfg = {"task": "detect", "model": _make_model(), "epochs": 5}
        result = _config_to_dict(cfg)
        assert "task" in result
        assert "epochs" in result
        assert "model" not in result
