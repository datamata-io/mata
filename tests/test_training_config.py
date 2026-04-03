"""Tests for mata.training.config.TrainingConfig — validation, from_yaml, defaults."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from mata.core.exceptions import ConfigurationError
from mata.training.config import TrainingConfig


# =============================================================================
# Default construction
# =============================================================================


class TestTrainingConfigDefaults:
    """All fields have correct default values after bare construction."""

    def test_default_task(self):
        cfg = TrainingConfig()
        assert cfg.task == ""

    def test_default_model(self):
        cfg = TrainingConfig()
        assert cfg.model == ""

    def test_default_data(self):
        cfg = TrainingConfig()
        assert cfg.data == ""

    def test_default_val_data(self):
        assert TrainingConfig().val_data is None

    def test_default_epochs(self):
        assert TrainingConfig().epochs == 10

    def test_default_batch_size(self):
        assert TrainingConfig().batch_size == 8

    def test_default_lr(self):
        assert TrainingConfig().lr == pytest.approx(1e-4)

    def test_default_optimizer(self):
        assert TrainingConfig().optimizer == "adamw"

    def test_default_weight_decay(self):
        assert TrainingConfig().weight_decay == pytest.approx(0.01)

    def test_default_scheduler(self):
        assert TrainingConfig().scheduler == "cosine"

    def test_default_warmup_epochs(self):
        assert TrainingConfig().warmup_epochs == 1

    def test_default_device(self):
        assert TrainingConfig().device == "auto"

    def test_default_amp(self):
        assert TrainingConfig().amp is True

    def test_default_save_dir(self):
        assert TrainingConfig().save_dir == "runs/train"

    def test_default_save_every(self):
        assert TrainingConfig().save_every == 0

    def test_default_val_every(self):
        assert TrainingConfig().val_every == 1

    def test_default_patience(self):
        assert TrainingConfig().patience == 0

    def test_default_freeze_backbone(self):
        assert TrainingConfig().freeze_backbone is False

    def test_default_freeze_layers(self):
        assert TrainingConfig().freeze_layers is None

    def test_default_augment(self):
        assert TrainingConfig().augment is True

    def test_default_augment_config(self):
        assert TrainingConfig().augment_config is None

    def test_default_resume(self):
        assert TrainingConfig().resume is None

    def test_default_num_workers(self):
        # On Windows, __post_init__ forces num_workers=0 to avoid multiprocessing issues.
        expected = 0 if sys.platform == "win32" else 4
        assert TrainingConfig().num_workers == expected

    def test_default_seed(self):
        assert TrainingConfig().seed == 42

    def test_default_verbose(self):
        assert TrainingConfig().verbose is True


# =============================================================================
# Valid configurations — do NOT raise on validate()
# =============================================================================


class TestTrainingConfigValidConfigurations:
    """Full valid configs for each task / model combination pass validate()."""

    def _minimal(self, task: str, model: str) -> TrainingConfig:
        return TrainingConfig(task=task, model=model, data="data.yaml", epochs=5)

    def test_detect_huggingface(self):
        cfg = self._minimal("detect", "facebook/detr-resnet-50")
        cfg.validate()  # must not raise

    def test_classify_huggingface(self):
        cfg = self._minimal("classify", "microsoft/resnet-50")
        cfg.validate()

    def test_segment_huggingface(self):
        cfg = self._minimal("segment", "facebook/mask2former-swin-small-coco-instance")
        cfg.validate()

    def test_detect_torchvision(self):
        cfg = self._minimal("detect", "torchvision/fasterrcnn_resnet50_fpn")
        cfg.validate()

    def test_all_non_default_valid_fields(self):
        cfg = TrainingConfig(
            task="detect",
            model="facebook/detr-resnet-50",
            data="coco.yaml",
            val_data="val.yaml",
            epochs=20,
            batch_size=16,
            lr=5e-5,
            optimizer="sgd",
            weight_decay=0.05,
            scheduler="step",
            warmup_epochs=2,
            device="cuda:0",
            amp=False,
            save_dir="runs/custom",
            save_every=5,
            val_every=2,
            patience=5,
            freeze_backbone=True,
            freeze_layers=["encoder"],
            augment=False,
            num_workers=0,
            seed=0,
            verbose=False,
        )
        cfg.validate()

    def test_valid_device_cpu(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", device="cpu", epochs=2)
        cfg.validate()

    def test_valid_device_cuda(self):
        cfg = TrainingConfig(task="classify", model="m", data="d", device="cuda", epochs=2)
        cfg.validate()

    def test_valid_device_cuda_indexed(self):
        cfg = TrainingConfig(task="classify", model="m", data="d", device="cuda:1", epochs=2)
        cfg.validate()

    def test_valid_optimizer_adam(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", optimizer="adam", epochs=2)
        cfg.validate()

    def test_valid_scheduler_linear(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", scheduler="linear", epochs=2)
        cfg.validate()

    def test_valid_scheduler_none(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", scheduler="none", epochs=2)
        cfg.validate()

    def test_valid_patience_zero_disabled(self):
        # patience=0 means disabled — should not raise
        cfg = TrainingConfig(task="segment", model="m", data="d", patience=0, epochs=5)
        cfg.validate()

    def test_valid_warmup_epochs_zero(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", warmup_epochs=0, epochs=2)
        cfg.validate()


# =============================================================================
# Invalid configurations — validate() must raise ConfigurationError
# =============================================================================


class TestTrainingConfigInvalidTask:
    def test_invalid_task_raises(self):
        cfg = TrainingConfig(task="invalid", model="m", data="d", epochs=2)
        with pytest.raises(ConfigurationError, match="Invalid task"):
            cfg.validate()

    def test_empty_task_raises(self):
        cfg = TrainingConfig(task="", model="m", data="d", epochs=2)
        with pytest.raises(ConfigurationError, match="Invalid task"):
            cfg.validate()

    def test_track_task_raises(self):
        cfg = TrainingConfig(task="track", model="m", data="d", epochs=2)
        with pytest.raises(ConfigurationError, match="Invalid task"):
            cfg.validate()

    def test_error_message_lists_valid_tasks(self):
        cfg = TrainingConfig(task="depth", model="m", data="d", epochs=2)
        with pytest.raises(ConfigurationError) as exc_info:
            cfg.validate()
        msg = str(exc_info.value)
        assert "classify" in msg
        assert "detect" in msg
        assert "segment" in msg


class TestTrainingConfigInvalidOptimizer:
    def test_invalid_optimizer_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", optimizer="nadam", epochs=2)
        with pytest.raises(ConfigurationError, match="Invalid optimizer"):
            cfg.validate()

    def test_error_message_lists_valid_optimizers(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", optimizer="rmsprop", epochs=2)
        with pytest.raises(ConfigurationError) as exc_info:
            cfg.validate()
        msg = str(exc_info.value)
        assert "adamw" in msg


class TestTrainingConfigInvalidScheduler:
    def test_invalid_scheduler_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", scheduler="onecycle", epochs=2)
        with pytest.raises(ConfigurationError, match="Invalid scheduler"):
            cfg.validate()

    def test_error_message_lists_valid_schedulers(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", scheduler="warmup", epochs=2)
        with pytest.raises(ConfigurationError) as exc_info:
            cfg.validate()
        msg = str(exc_info.value)
        assert "cosine" in msg


class TestTrainingConfigInvalidDevice:
    def test_invalid_device_tpu_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", device="tpu", epochs=2)
        with pytest.raises(ConfigurationError, match="Invalid device"):
            cfg.validate()

    def test_invalid_device_mps_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", device="mps", epochs=2)
        with pytest.raises(ConfigurationError, match="Invalid device"):
            cfg.validate()

    def test_invalid_device_cuda_negative_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", device="cuda:-1", epochs=2)
        with pytest.raises(ConfigurationError, match="Invalid device"):
            cfg.validate()


class TestTrainingConfigInvalidNumericRanges:
    def test_epochs_zero_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", epochs=0)
        with pytest.raises(ConfigurationError, match="epochs"):
            cfg.validate()

    def test_epochs_negative_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", epochs=-1)
        with pytest.raises(ConfigurationError, match="epochs"):
            cfg.validate()

    def test_batch_size_zero_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", epochs=2, batch_size=0)
        with pytest.raises(ConfigurationError, match="batch_size"):
            cfg.validate()

    def test_batch_size_negative_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", epochs=2, batch_size=-4)
        with pytest.raises(ConfigurationError, match="batch_size"):
            cfg.validate()

    def test_lr_zero_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", epochs=2, lr=0.0)
        with pytest.raises(ConfigurationError, match="lr"):
            cfg.validate()

    def test_lr_negative_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", epochs=2, lr=-1e-4)
        with pytest.raises(ConfigurationError, match="lr"):
            cfg.validate()

    def test_weight_decay_negative_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", epochs=2, weight_decay=-0.01)
        with pytest.raises(ConfigurationError, match="weight_decay"):
            cfg.validate()

    def test_val_every_zero_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", epochs=2, val_every=0)
        with pytest.raises(ConfigurationError, match="val_every"):
            cfg.validate()

    def test_patience_negative_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", epochs=5, patience=-1)
        with pytest.raises(ConfigurationError, match="patience"):
            cfg.validate()

    def test_warmup_epochs_negative_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", epochs=5, warmup_epochs=-1)
        with pytest.raises(ConfigurationError, match="warmup_epochs"):
            cfg.validate()

    def test_num_workers_negative_raises(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", epochs=2, num_workers=-1)
        with pytest.raises(ConfigurationError, match="num_workers"):
            cfg.validate()

    def test_warmup_epochs_exceeds_epochs_raises(self):
        # warmup_epochs >= epochs should raise
        cfg = TrainingConfig(task="detect", model="m", data="d", epochs=3, warmup_epochs=3)
        with pytest.raises(ConfigurationError, match="warmup_epochs"):
            cfg.validate()


# =============================================================================
# validate(): resume path must exist
# =============================================================================


class TestTrainingConfigResumeValidation:
    def test_resume_nonexistent_path_raises(self, tmp_path):
        missing = str(tmp_path / "no_such_checkpoint")
        cfg = TrainingConfig(
            task="detect", model="m", data="d", epochs=2, resume=missing
        )
        with pytest.raises(ConfigurationError, match="resume"):
            cfg.validate()

    def test_resume_existing_path_passes(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoint"
        ckpt_dir.mkdir()
        cfg = TrainingConfig(
            task="detect", model="m", data="d", epochs=2, resume=str(ckpt_dir)
        )
        cfg.validate()  # must not raise

    def test_resume_none_passes(self):
        cfg = TrainingConfig(task="detect", model="m", data="d", epochs=2, resume=None)
        cfg.validate()  # must not raise

    def test_resume_error_message_contains_path(self, tmp_path):
        bad_path = str(tmp_path / "ghost_checkpoint")
        cfg = TrainingConfig(task="detect", model="m", data="d", epochs=2, resume=bad_path)
        with pytest.raises(ConfigurationError) as exc_info:
            cfg.validate()
        assert bad_path in str(exc_info.value)


# =============================================================================
# from_yaml()
# =============================================================================


class TestTrainingConfigFromYaml:
    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    def test_from_yaml_loads_all_basic_fields(self, tmp_path):
        p = self._write_yaml(
            tmp_path,
            """\
            task: detect
            model: facebook/detr-resnet-50
            data: coco.yaml
            epochs: 15
            batch_size: 16
            lr: 0.0005
            optimizer: sgd
            scheduler: step
            device: cpu
            verbose: false
            """,
        )
        cfg = TrainingConfig.from_yaml(p)
        assert cfg.task == "detect"
        assert cfg.model == "facebook/detr-resnet-50"
        assert cfg.data == "coco.yaml"
        assert cfg.epochs == 15
        assert cfg.batch_size == 16
        assert cfg.lr == pytest.approx(0.0005)
        assert cfg.optimizer == "sgd"
        assert cfg.scheduler == "step"
        assert cfg.device == "cpu"
        assert cfg.verbose is False

    def test_from_yaml_round_trip(self, tmp_path):
        """from_yaml followed by accessing fields mirrors what was written."""
        p = self._write_yaml(
            tmp_path,
            """\
            task: classify
            model: microsoft/resnet-50
            data: /data/flowers/
            val_data: /data/flowers_val/
            epochs: 5
            lr: 0.00001
            freeze_backbone: true
            seed: 0
            """,
        )
        cfg = TrainingConfig.from_yaml(p)
        assert cfg.task == "classify"
        assert cfg.val_data == "/data/flowers_val/"
        assert cfg.freeze_backbone is True
        assert cfg.seed == 0
        assert cfg.lr == pytest.approx(1e-5)

    def test_from_yaml_unknown_keys_ignored(self, tmp_path):
        """Extra keys in YAML are silently ignored."""
        p = self._write_yaml(
            tmp_path,
            """\
            task: detect
            model: m
            data: d
            epochs: 2
            totally_unknown_key: whatever
            """,
        )
        cfg = TrainingConfig.from_yaml(p)  # must not raise
        assert cfg.task == "detect"
        assert not hasattr(cfg, "totally_unknown_key")

    def test_from_yaml_missing_keys_use_defaults(self, tmp_path):
        """Keys absent from YAML file fall back to dataclass defaults."""
        p = self._write_yaml(tmp_path, "task: segment\nmodel: m\ndata: d\nepochs: 3\n")
        cfg = TrainingConfig.from_yaml(p)
        # unspecified fields keep defaults
        assert cfg.optimizer == "adamw"
        assert cfg.batch_size == 8
        assert cfg.amp is True

    def test_from_yaml_file_not_found_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(ConfigurationError, match="not found"):
            TrainingConfig.from_yaml(missing)

    def test_from_yaml_invalid_yaml_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(": invalid: [unclosed")
        with pytest.raises(ConfigurationError, match="parse"):
            TrainingConfig.from_yaml(p)

    def test_from_yaml_non_mapping_raises(self, tmp_path):
        """A YAML file that is a list rather than a mapping raises ConfigurationError."""
        p = tmp_path / "list.yaml"
        p.write_text("- detect\n- classify\n")
        with pytest.raises(ConfigurationError, match="mapping"):
            TrainingConfig.from_yaml(p)

    def test_from_yaml_accepts_path_object(self, tmp_path):
        p = self._write_yaml(tmp_path, "task: detect\nmodel: m\ndata: d\nepochs: 2\n")
        cfg = TrainingConfig.from_yaml(Path(p))  # pass Path, not str
        assert cfg.task == "detect"

    def test_from_yaml_string_path(self, tmp_path):
        p = self._write_yaml(tmp_path, "task: classify\nmodel: m\ndata: d\nepochs: 2\n")
        cfg = TrainingConfig.from_yaml(str(p))  # pass str
        assert cfg.task == "classify"

    def test_from_yaml_list_field(self, tmp_path):
        """freeze_layers can be a YAML list."""
        p = self._write_yaml(
            tmp_path,
            """\
            task: detect
            model: m
            data: d
            epochs: 2
            freeze_layers:
              - encoder
              - backbone.layer4
            """,
        )
        cfg = TrainingConfig.from_yaml(p)
        assert cfg.freeze_layers == ["encoder", "backbone.layer4"]


# =============================================================================
# Serialization: write to JSON and read back
# =============================================================================


class TestTrainingConfigSerialization:
    def test_fields_serializable_to_json(self, tmp_path):
        """TrainingConfig fields can be dumped to JSON without error."""
        import dataclasses

        cfg = TrainingConfig(
            task="detect",
            model="facebook/detr-resnet-50",
            data="coco.yaml",
            epochs=5,
            freeze_layers=["backbone"],
        )
        d = dataclasses.asdict(cfg)
        out = tmp_path / "config.json"
        out.write_text(json.dumps(d))
        loaded = json.loads(out.read_text())
        assert loaded["task"] == "detect"
        assert loaded["epochs"] == 5
        assert loaded["freeze_layers"] == ["backbone"]

    def test_json_round_trip_reconstruct(self, tmp_path):
        """Round-trip through JSON preserves all primitive fields."""
        import dataclasses

        original = TrainingConfig(
            task="classify",
            model="microsoft/resnet-50",
            data="/data/flowers/",
            epochs=8,
            batch_size=32,
            lr=3e-4,
            optimizer="adam",
            scheduler="linear",
            seed=7,
            verbose=False,
        )
        d = dataclasses.asdict(original)
        reconstructed = TrainingConfig(**d)
        assert reconstructed.task == original.task
        assert reconstructed.model == original.model
        assert reconstructed.lr == pytest.approx(original.lr)
        assert reconstructed.seed == original.seed
        assert reconstructed.verbose == original.verbose
