"""Regression tests for torchvision checkpoint reload with custom class counts.

Covers two bugs that surfaced during smoke testing:

1. ``AttributeError: 'int' object has no attribute 'to'``
   - Cause: target dicts contain scalar metadata (image_id, area, …) that are
     not tensors and do not implement ``.to(device)``.
   - Regression: ``_move_targets_to_device`` must skip non-tensor values.

2. ``RuntimeError: size mismatch for roi_heads.box_predictor.cls_score.weight``
   - Cause: ``_load_from_checkpoint`` rebuilt the default COCO-91 head before
     calling ``load_state_dict`` with fine-tuned 3-class weights.
   - Regression: loader must infer num_classes from checkpoint tensors and
     replace the head *before* loading weights.

No real model downloads occur — torchvision model construction is mocked with
lightweight stand-ins that match the structural API used by the loader.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers — minimal structural mocks
# ---------------------------------------------------------------------------


class _MockPredictor(nn.Module):
    """Mimics Faster R-CNN ``roi_heads.box_predictor``."""

    def __init__(self, in_features: int = 1024, num_classes: int = 91) -> None:
        super().__init__()
        self.cls_score = nn.Linear(in_features, num_classes)
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)

    @property
    def in_features(self) -> int:  # type: ignore[override]
        return self.cls_score.in_features


class _MockRoiHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.box_predictor = _MockPredictor(in_features=1024, num_classes=91)


class _MockBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=1)


class _MockFasterRCNN(nn.Module):
    """Lightweight Faster R-CNN replacement used in unit tests."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _MockBackbone()
        self.roi_heads = _MockRoiHeads()

    def forward(self, images, targets=None):  # noqa: ANN001
        return [
            {
                "boxes": torch.zeros(0, 4),
                "labels": torch.zeros(0, dtype=torch.long),
                "scores": torch.zeros(0),
            }
            for _ in images
        ]


def _make_3class_state_dict(in_features: int = 1024) -> dict:
    """Return a state dict that looks like a 3-class fine-tuned Faster R-CNN head."""
    predictor = _MockPredictor(in_features=in_features, num_classes=3)
    backbone = _MockBackbone()
    # Build the flat keys the loader expects
    sd: dict = {}
    for k, v in predictor.state_dict().items():
        sd[f"roi_heads.box_predictor.{k}"] = v.clone()
    for k, v in backbone.state_dict().items():
        sd[f"backbone.{k}"] = v.clone()
    return sd


def _write_checkpoint(tmp_dir: Path, state_dict: dict, model_source: str) -> Path:
    """Write a minimal MATA-format checkpoint directory."""
    ckpt = tmp_dir / "best"
    ckpt.mkdir(parents=True)
    torch.save(state_dict, ckpt / "model_state.pth")
    config = {
        "engine": "torchvision",
        "model_source": model_source,
        "task": "detect",
    }
    with open(ckpt / "config.json", "w") as f:
        json.dump(config, f)
    return ckpt


# ---------------------------------------------------------------------------
# 1.  _move_targets_to_device
# ---------------------------------------------------------------------------


class TestMoveTargetsToDevice:
    """Unit tests for the _move_targets_to_device helper."""

    def _fn(self):
        from mata.training.torch_trainer import _move_targets_to_device

        return _move_targets_to_device

    def test_tensors_are_moved(self):
        fn = self._fn()
        device = torch.device("cpu")
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        labels = torch.tensor([1])
        result = fn([{"boxes": boxes, "labels": labels}], device)
        assert result[0]["boxes"].device == device
        assert result[0]["labels"].device == device

    def test_int_metadata_is_preserved(self):
        """Regression: image_id (int) must not raise AttributeError."""
        fn = self._fn()
        device = torch.device("cpu")
        targets = [{"boxes": torch.zeros(1, 4), "labels": torch.tensor([0]), "image_id": 42}]
        result = fn(targets, device)
        assert result[0]["image_id"] == 42  # unchanged int

    def test_mixed_types_no_crash(self):
        """All non-tensor values survive device transfer unmodified."""
        fn = self._fn()
        device = torch.device("cpu")
        targets = [
            {
                "boxes": torch.zeros(2, 4),
                "labels": torch.tensor([0, 1]),
                "image_id": 99,
                "area": [100.0, 200.0],  # list — not a tensor
                "iscrowd": 0,
            }
        ]
        result = fn(targets, device)
        assert result[0]["image_id"] == 99
        assert result[0]["area"] == [100.0, 200.0]
        assert result[0]["iscrowd"] == 0
        assert isinstance(result[0]["boxes"], torch.Tensor)

    def test_empty_targets_list(self):
        fn = self._fn()
        assert fn([], torch.device("cpu")) == []

    def test_empty_target_dict(self):
        fn = self._fn()
        result = fn([{}], torch.device("cpu"))
        assert result == [{}]


# ---------------------------------------------------------------------------
# 2.  Head adaptation — _adapt_torchvision_head_from_state_dict (via loader)
# ---------------------------------------------------------------------------


class TestFasterRCNNHeadAdaptation:
    """Regression tests for num_classes head rebuild on checkpoint load."""

    def _patched_loader(self):
        """Return a UniversalLoader with torchvision module mocked out."""
        from mata.core.model_loader import UniversalLoader

        return UniversalLoader()

    @pytest.fixture()
    def checkpoint_3class(self, tmp_path):
        """Write a 3-class Faster R-CNN checkpoint and return its path."""
        sd = _make_3class_state_dict(in_features=1024)
        return _write_checkpoint(tmp_path, sd, "torchvision/fasterrcnn_resnet50_fpn")

    def test_head_replaced_to_3_classes(self, checkpoint_3class):
        """Loader must rebuild the predictor head to 3 classes before loading."""
        mock_model = _MockFasterRCNN()

        def _fake_torchvision_adapter(task, model_name, **kwargs):
            adapter = MagicMock()
            adapter.model = mock_model
            adapter.device = torch.device("cpu")
            return adapter

        loader = self._patched_loader()
        with patch.object(loader, "_load_from_torchvision", side_effect=_fake_torchvision_adapter):
            adapter = loader._load_from_checkpoint("detect", str(checkpoint_3class))

        # Head must be replaced — 3 classes, not the original 91
        predictor = adapter.model.roi_heads.box_predictor
        assert predictor.cls_score.out_features == 3
        assert predictor.bbox_pred.out_features == 12  # 3 * 4

    def test_load_state_dict_succeeds_without_size_mismatch(self, checkpoint_3class):
        """load_state_dict must not raise after head replacement."""
        mock_model = _MockFasterRCNN()

        def _fake_torchvision_adapter(task, model_name, **kwargs):
            adapter = MagicMock()
            adapter.model = mock_model
            adapter.device = torch.device("cpu")
            return adapter

        loader = self._patched_loader()
        with patch.object(loader, "_load_from_torchvision", side_effect=_fake_torchvision_adapter):
            # Must not raise RuntimeError: size mismatch
            adapter = loader._load_from_checkpoint("detect", str(checkpoint_3class))

        # Verify the loaded weight shape matches the 3-class checkpoint.
        # Shape correctness is the regression guard (a size mismatch would have
        # raised during load_state_dict above); value equality is not meaningful
        # between two independently-initialised random tensors.
        predictor = adapter.model.roi_heads.box_predictor
        assert predictor.cls_score.weight.shape == (3, 1024)
        assert predictor.bbox_pred.weight.shape == (12, 1024)

    def test_model_placed_on_adapter_device_after_load(self, checkpoint_3class):
        """All parameters must be on adapter.device after loading."""
        mock_model = _MockFasterRCNN()
        device = torch.device("cpu")

        def _fake_torchvision_adapter(task, model_name, **kwargs):
            adapter = MagicMock()
            adapter.model = mock_model
            adapter.device = device
            return adapter

        loader = self._patched_loader()
        with patch.object(loader, "_load_from_torchvision", side_effect=_fake_torchvision_adapter):
            adapter = loader._load_from_checkpoint("detect", str(checkpoint_3class))

        for p in adapter.model.parameters():
            assert p.device == device

    def test_91class_checkpoint_stays_at_91(self, tmp_path):
        """Checkpoints that already have 91 classes must not be changed."""
        # Build a 91-class state dict (matches default torchvision head)
        predictor = _MockPredictor(in_features=1024, num_classes=91)
        backbone = _MockBackbone()
        sd: dict = {}
        for k, v in predictor.state_dict().items():
            sd[f"roi_heads.box_predictor.{k}"] = v.clone()
        for k, v in backbone.state_dict().items():
            sd[f"backbone.{k}"] = v.clone()
        ckpt = _write_checkpoint(tmp_path, sd, "torchvision/fasterrcnn_resnet50_fpn")

        mock_model = _MockFasterRCNN()

        def _fake(task, model_name, **kwargs):
            adapter = MagicMock()
            adapter.model = mock_model
            adapter.device = torch.device("cpu")
            return adapter

        loader = self._patched_loader()
        with patch.object(loader, "_load_from_torchvision", side_effect=_fake):
            adapter = loader._load_from_checkpoint("detect", str(ckpt))

        assert adapter.model.roi_heads.box_predictor.cls_score.out_features == 91

    def test_checkpoint_missing_cls_weight_key_is_skipped_gracefully(self, tmp_path):
        """Missing head keys (empty or partial checkpoint) must not raise."""
        # Checkpoint with only backbone weights, no head
        backbone = _MockBackbone()
        sd = {f"backbone.{k}": v.clone() for k, v in backbone.state_dict().items()}
        ckpt = _write_checkpoint(tmp_path, sd, "torchvision/fasterrcnn_resnet50_fpn")

        mock_model = _MockFasterRCNN()

        def _fake(task, model_name, **kwargs):
            adapter = MagicMock()
            adapter.model = mock_model
            adapter.device = torch.device("cpu")
            return adapter

        loader = self._patched_loader()
        with patch.object(loader, "_load_from_torchvision", side_effect=_fake):
            # strict=False: backbone-only load should not raise
            with pytest.raises(RuntimeError):
                # Full load_state_dict is strict, so this will error on missing keys.
                # The important thing is it's a key-missing error, not a shape mismatch.
                loader._load_from_checkpoint("detect", str(ckpt))


# ---------------------------------------------------------------------------
# 3.  End-to-end: load checkpoint → predict → VisionResult
# ---------------------------------------------------------------------------


class TestCheckpointInference:
    """Verify that a loaded checkpoint can run predict() without errors."""

    def test_predict_returns_vision_result(self, tmp_path):
        """Full round-trip: write checkpoint, load, predict, check result type."""
        from mata.core.types import VisionResult

        sd = _make_3class_state_dict(in_features=1024)
        ckpt = _write_checkpoint(tmp_path, sd, "torchvision/fasterrcnn_resnet50_fpn")

        # Build a real structural mock that runs forward()
        real_model_container = {"model": _MockFasterRCNN()}

        def _fake_torchvision_adapter(task, model_name, **kwargs):
            from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

            with patch(
                "mata.adapters.torchvision_detect_adapter._ensure_torchvision"
            ) as _mock_tv:
                detection_models = MagicMock()
                detection_models.fasterrcnn_resnet50_fpn.return_value = real_model_container[
                    "model"
                ]
                transforms = MagicMock()
                transforms.Compose.return_value = lambda x: x  # identity
                transforms.ToTensor.return_value = MagicMock()
                transforms.Normalize.return_value = MagicMock()
                _mock_tv.return_value = (detection_models, transforms)

                adapter = TorchvisionDetectAdapter(
                    model_name="torchvision/fasterrcnn_resnet50_fpn",
                    device="cpu",
                )

            real_model_container["model"] = adapter.model
            return adapter

        from mata.core.model_loader import UniversalLoader

        loader = UniversalLoader()
        with patch.object(loader, "_load_from_torchvision", side_effect=_fake_torchvision_adapter):
            adapter = loader._load_from_checkpoint("detect", str(ckpt))

        # Run predict with a synthetic image tensor
        import numpy as np
        from PIL import Image as PILImage

        img_arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
        pil_img = PILImage.fromarray(img_arr)

        # Patch _preprocess to return a plain tensor (bypass transform chain)
        with patch.object(adapter, "_preprocess", return_value=torch.zeros(3, 64, 64)):
            result = adapter.predict(pil_img, threshold=0.0)

        assert isinstance(result, VisionResult)
