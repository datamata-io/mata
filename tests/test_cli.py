"""Unit tests for the mata CLI (mata.cli).

Uses argparse parsing and mocked mata.* API calls — no model downloads.
Run independently: pytest tests/test_cli.py -v
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import mata.cli as cli_module
from mata.cli import main, _build_parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args: list[str]) -> int:
    """Invoke main() with explicit argv; return exit code."""
    return main(args)


def _capture(args: list[str]) -> tuple[int, str, str]:
    """Return (exit_code, stdout_text, stderr_text)."""
    out = StringIO()
    err = StringIO()
    with patch("sys.stdout", out), patch("sys.stderr", err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# TestParser
# ---------------------------------------------------------------------------

class TestParser:
    def test_run_subcommand_parsed(self):
        p = _build_parser()
        ns = p.parse_args(["run", "detect", "image.jpg"])
        assert ns.command == "run"
        assert ns.task == "detect"
        assert ns.input == "image.jpg"

    def test_run_defaults(self):
        p = _build_parser()
        ns = p.parse_args(["run", "detect", "img.jpg"])
        assert ns.model is None
        assert ns.conf is None
        assert ns.device is None
        assert not ns.save
        assert not ns.output_json

    def test_run_model_flag(self):
        p = _build_parser()
        ns = p.parse_args(["run", "detect", "img.jpg", "--model", "facebook/detr-resnet-50"])
        assert ns.model == "facebook/detr-resnet-50"

    def test_run_short_model_flag(self):
        p = _build_parser()
        ns = p.parse_args(["run", "detect", "img.jpg", "-m", "my-model"])
        assert ns.model == "my-model"

    def test_run_conf_flag(self):
        p = _build_parser()
        ns = p.parse_args(["run", "detect", "img.jpg", "--conf", "0.5"])
        assert ns.conf == pytest.approx(0.5)

    def test_run_text_flag(self):
        p = _build_parser()
        ns = p.parse_args(["run", "classify", "img.jpg", "--text", "cat,dog"])
        assert ns.text == "cat,dog"

    def test_run_prompt_flag(self):
        p = _build_parser()
        ns = p.parse_args(["run", "vlm", "img.jpg", "--prompt", "Describe this"])
        assert ns.prompt == "Describe this"

    def test_run_save_flag(self):
        p = _build_parser()
        ns = p.parse_args(["run", "detect", "img.jpg", "--save"])
        assert ns.save

    def test_run_json_flag(self):
        p = _build_parser()
        ns = p.parse_args(["run", "detect", "img.jpg", "--json"])
        assert ns.output_json

    def test_track_subcommand_parsed(self):
        p = _build_parser()
        ns = p.parse_args(["track", "video.mp4"])
        assert ns.command == "track"
        assert ns.source == "video.mp4"

    def test_track_defaults(self):
        p = _build_parser()
        ns = p.parse_args(["track", "video.mp4"])
        assert ns.tracker == "botsort"
        assert ns.conf == pytest.approx(0.25)
        assert ns.iou == pytest.approx(0.7)

    def test_track_tracker_flag(self):
        p = _build_parser()
        ns = p.parse_args(["track", "video.mp4", "--tracker", "bytetrack"])
        assert ns.tracker == "bytetrack"

    def test_track_reid_model_flag(self):
        p = _build_parser()
        ns = p.parse_args(["track", "video.mp4", "--reid-model", "openai/clip"])
        assert ns.reid_model == "openai/clip"

    def test_val_subcommand_parsed(self):
        p = _build_parser()
        ns = p.parse_args(["val", "detect", "--data", "coco.yaml"])
        assert ns.command == "val"
        assert ns.task == "detect"
        assert ns.data == "coco.yaml"

    def test_val_defaults(self):
        p = _build_parser()
        ns = p.parse_args(["val", "detect", "--data", "coco.yaml"])
        assert ns.conf == pytest.approx(0.001)
        assert ns.iou == pytest.approx(0.5)
        assert ns.split == "val"

    def test_val_data_required(self):
        p = _build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["val", "detect"])  # missing --data

    def test_export_subcommand_parsed(self):
        p = _build_parser()
        ns = p.parse_args(["export", "detect", "model.pt", "--format", "onnx"])
        assert ns.command == "export"
        assert ns.task == "detect"
        assert ns.model == "model.pt"
        assert ns.format == "onnx"

    def test_export_format_choices(self):
        p = _build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["export", "detect", "model.pt", "--format", "invalid"])

    def test_no_command_exits(self):
        p = _build_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_version_flag(self):
        p = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            p.parse_args(["--version"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# TestCmdRun
# ---------------------------------------------------------------------------

class TestCmdRun:
    def _mock_detect_result(self):
        inst = MagicMock()
        inst.label_name = "cat"
        inst.score = 0.95
        inst.bbox = [10.0, 20.0, 100.0, 200.0]
        result = MagicMock()
        result.instances = [inst]
        result.to_json.return_value = json.dumps({"instances": []})
        return result

    def test_run_detect_calls_mata_run(self):
        result = self._mock_detect_result()
        with patch("mata.run", return_value=result) as mock_run:
            code = _run(["run", "detect", "img.jpg", "--model", "my-model"])
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs[0][0] == "detect"
        assert call_kwargs[0][1] == "img.jpg"

    def test_run_detect_exits_0_on_success(self):
        result = self._mock_detect_result()
        with patch("mata.run", return_value=result):
            code = _run(["run", "detect", "img.jpg"])
        assert code == 0

    def test_run_json_flag_emits_json(self):
        result = self._mock_detect_result()
        with patch("mata.run", return_value=result):
            code, stdout, _ = _capture(["run", "detect", "img.jpg", "--json"])
        # Should produce valid JSON on stdout
        parsed = json.loads(stdout)
        assert parsed is not None

    def test_run_conf_forwarded_as_threshold(self):
        result = self._mock_detect_result()
        with patch("mata.run", return_value=result) as mock_run:
            _run(["run", "detect", "img.jpg", "--conf", "0.5"])
        kwargs = mock_run.call_args[1]
        assert kwargs.get("threshold") == pytest.approx(0.5)

    def test_run_device_forwarded(self):
        result = self._mock_detect_result()
        with patch("mata.run", return_value=result) as mock_run:
            _run(["run", "detect", "img.jpg", "--device", "cpu"])
        kwargs = mock_run.call_args[1]
        assert kwargs.get("device") == "cpu"

    def test_run_text_split_by_comma(self):
        result = self._mock_detect_result()
        with patch("mata.run", return_value=result) as mock_run:
            _run(["run", "classify", "img.jpg", "--text", "cat,dog,bird"])
        kwargs = mock_run.call_args[1]
        assert kwargs.get("text_prompts") == ["cat", "dog", "bird"]

    def test_run_embed_ndarray_output(self):
        arr = np.zeros((1, 512), dtype=np.float32)
        with patch("mata.run", return_value=arr):
            code, stdout, _ = _capture(["run", "embed", "img.jpg"])
        assert code == 0
        assert "512" in stdout

    def test_run_embed_json_flag(self):
        arr = np.zeros((1, 4), dtype=np.float32)
        with patch("mata.run", return_value=arr):
            code, stdout, _ = _capture(["run", "embed", "img.jpg", "--json"])
        parsed = json.loads(stdout)
        assert "embeddings" in parsed

    def test_run_error_exits_1(self):
        with patch("mata.run", side_effect=RuntimeError("model not found")):
            code, _, stderr = _capture(["run", "detect", "img.jpg"])
        assert code == 1
        assert "ERROR" in stderr

    def test_run_classify_output(self):
        clf = MagicMock()
        clf.label_name = "tabby_cat"
        clf.score = 0.88
        result = MagicMock()
        result.classifications = [clf]
        result.to_json.return_value = "{}"
        with patch("mata.run", return_value=result):
            code, stdout, _ = _capture(["run", "classify", "img.jpg"])
        assert code == 0
        assert "tabby_cat" in stdout


# ---------------------------------------------------------------------------
# TestCmdTrack
# ---------------------------------------------------------------------------

class TestCmdTrack:
    def _mock_track_result(self, n_instances: int = 2):
        inst = MagicMock()
        result = MagicMock()
        result.instances = [inst] * n_instances
        result.to_dict.return_value = {"instances": []}
        return [result]

    def test_track_calls_mata_track(self):
        with patch("mata.track", return_value=iter(self._mock_track_result())) as mock_track:
            code = _run(["track", "video.mp4", "--model", "my-model"])
        mock_track.assert_called_once()

    def test_track_exits_0(self):
        with patch("mata.track", return_value=iter(self._mock_track_result())):
            code = _run(["track", "video.mp4"])
        assert code == 0

    def test_track_tracker_forwarded(self):
        with patch("mata.track", return_value=iter(self._mock_track_result())) as mock_track:
            _run(["track", "video.mp4", "--tracker", "bytetrack"])
        kwargs = mock_track.call_args[1]
        assert kwargs.get("tracker") == "bytetrack"

    def test_track_conf_forwarded(self):
        with patch("mata.track", return_value=iter(self._mock_track_result())) as mock_track:
            _run(["track", "video.mp4", "--conf", "0.4"])
        kwargs = mock_track.call_args[1]
        assert kwargs.get("conf") == pytest.approx(0.4)

    def test_track_error_exits_1(self):
        with patch("mata.track", side_effect=FileNotFoundError("video not found")):
            code, _, stderr = _capture(["track", "video.mp4"])
        assert code == 1

    def test_track_json_output(self):
        with patch("mata.track", return_value=iter(self._mock_track_result())):
            code, stdout, _ = _capture(["track", "video.mp4", "--json"])
        assert code == 0
        # Should output JSON
        parsed = json.loads(stdout)
        assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# TestCmdVal
# ---------------------------------------------------------------------------

class TestCmdVal:
    def test_val_calls_mata_val(self):
        metrics = MagicMock()
        metrics.to_dict.return_value = {"mAP": 0.5}
        with patch("mata.val", return_value=metrics) as mock_val:
            code = _run(["val", "detect", "--data", "coco.yaml"])
        mock_val.assert_called_once()
        assert code == 0

    def test_val_passes_data_arg(self):
        metrics = MagicMock()
        with patch("mata.val", return_value=metrics) as mock_val:
            _run(["val", "detect", "--data", "my_data.yaml"])
        kwargs = mock_val.call_args[1]
        assert kwargs.get("data") == "my_data.yaml"

    def test_val_json_flag_emits_json(self):
        metrics = MagicMock()
        metrics.to_dict.return_value = {"mAP50": 0.6}
        with patch("mata.val", return_value=metrics):
            code, stdout, _ = _capture(["val", "detect", "--data", "data.yaml", "--json"])
        assert code == 0
        parsed = json.loads(stdout)
        assert "mAP50" in parsed

    def test_val_error_exits_1(self):
        with patch("mata.val", side_effect=ValueError("bad dataset")):
            code, _, stderr = _capture(["val", "detect", "--data", "data.yaml"])
        assert code == 1

    def test_val_iou_forwarded(self):
        metrics = MagicMock()
        with patch("mata.val", return_value=metrics) as mock_val:
            _run(["val", "detect", "--data", "d.yaml", "--iou", "0.75"])
        kwargs = mock_val.call_args[1]
        assert kwargs.get("iou") == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# TestCmdExport
# ---------------------------------------------------------------------------

class TestCmdExport:
    def test_export_exits_0(self):
        code, stdout, _ = _capture(["export", "detect", "model.pt", "--format", "onnx"])
        assert code == 0

    def test_export_prints_coming_v2(self):
        _, stdout, _ = _capture(["export", "detect", "model.pt", "--format", "onnx"])
        assert "2.0" in stdout

    def test_export_shows_task_and_model(self):
        _, stdout, _ = _capture(["export", "detect", "my_model.pt", "--format", "onnx"])
        assert "detect" in stdout
        assert "my_model.pt" in stdout


# ---------------------------------------------------------------------------
# TestVersionAndVerbosity
# ---------------------------------------------------------------------------

class TestVersionAndVerbosity:
    def test_version_contains_mata(self):
        ver = cli_module._get_version()
        assert "mata" in ver.lower()

    def test_main_module_importable(self):
        import mata.__main__ as mm
        assert mm is not None
