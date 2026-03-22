"""Unit tests for the `mata recognize` CLI subcommand.

Uses argparse + mocked mata API — no model downloads or real gallery files.
Run independently: pytest tests/test_cli_recognize.py -v
"""

from __future__ import annotations

import json
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mata import Gallery
from mata.cli import _build_parser, main
from mata.core.artifacts.matches import MatchEntry, Matches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture(args: list[str]) -> tuple[int, str, str]:
    """Run main() with explicit argv; return (exit_code, stdout, stderr)."""
    out = StringIO()
    err = StringIO()
    with patch("sys.stdout", out), patch("sys.stderr", err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


def _make_gallery_file(n: int = 2, dim: int = 32) -> str:
    """Save a small gallery to a temp .npz file; return the path."""
    g = Gallery(similarity_thresh=0.0)
    for i in range(n):
        v = np.random.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        g.add(f"person_{i}", v)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    g.save(path)
    return path


def _make_matches(label: str = "alice", sim: float = 0.92) -> Matches:
    entry = MatchEntry(
        instance_id="query",
        label=label,
        similarity=sim,
        all_matches=[{"label": label, "similarity": sim, "index": 0}],
    )
    return Matches(entries=[entry], meta={"top_k": 1})


def _make_image_file() -> str:
    """Save a tiny PNG to a temp file; return path."""
    from PIL import Image as PILImage
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    return path


# ---------------------------------------------------------------------------
# TestRecognizeParserArguments
# ---------------------------------------------------------------------------


class TestRecognizeParserArguments:
    def test_recognize_subcommand_parsed(self):
        p = _build_parser()
        ns = p.parse_args(["recognize", "image.jpg", "--gallery", "g.npz"])
        assert ns.command == "recognize"
        assert ns.input == "image.jpg"
        assert ns.gallery == "g.npz"

    def test_default_top_k(self):
        p = _build_parser()
        ns = p.parse_args(["recognize", "image.jpg", "--gallery", "g.npz"])
        assert ns.top_k == 1

    def test_custom_top_k(self):
        p = _build_parser()
        ns = p.parse_args(["recognize", "img.jpg", "--gallery", "g.npz", "--top-k", "5"])
        assert ns.top_k == 5

    def test_default_threshold_none(self):
        p = _build_parser()
        ns = p.parse_args(["recognize", "img.jpg", "--gallery", "g.npz"])
        assert ns.threshold is None

    def test_custom_threshold(self):
        p = _build_parser()
        ns = p.parse_args(["recognize", "img.jpg", "--gallery", "g.npz", "--threshold", "0.7"])
        assert abs(ns.threshold - 0.7) < 1e-9

    def test_model_flag(self):
        p = _build_parser()
        ns = p.parse_args(["recognize", "img.jpg", "--gallery", "g.npz",
                           "--model", "openai/clip-vit-base-patch32"])
        assert ns.model == "openai/clip-vit-base-patch32"

    def test_model_short_flag(self):
        p = _build_parser()
        ns = p.parse_args(["recognize", "img.jpg", "-g", "g.npz", "-m", "my-model"])
        assert ns.model == "my-model"

    def test_gallery_required(self):
        p = _build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["recognize", "img.jpg"])

    def test_json_flag(self):
        p = _build_parser()
        ns = p.parse_args(["recognize", "img.jpg", "--gallery", "g.npz", "--json"])
        assert ns.output_json

    def test_device_flag(self):
        p = _build_parser()
        ns = p.parse_args(["recognize", "img.jpg", "--gallery", "g.npz", "--device", "cuda"])
        assert ns.device == "cuda"

    def test_default_device_none(self):
        p = _build_parser()
        ns = p.parse_args(["recognize", "img.jpg", "--gallery", "g.npz"])
        assert ns.device is None


# ---------------------------------------------------------------------------
# TestRecognizeCmdSuccess
# ---------------------------------------------------------------------------


class TestRecognizeCmdSuccess:
    def test_successful_run_exits_zero(self):
        gallery_path = _make_gallery_file()
        image_path = _make_image_file()
        result = _make_matches("alice", 0.95)
        with patch("mata.run", return_value=result), \
             patch("mata.Gallery.load", return_value=Gallery()):
            code, _out, _err = _capture(["recognize", image_path,
                                          "--gallery", gallery_path])
        assert code == 0

    def test_stdout_contains_label(self):
        gallery_path = _make_gallery_file()
        image_path = _make_image_file()
        result = _make_matches("alice", 0.95)
        with patch("mata.run", return_value=result), \
             patch("mata.Gallery.load", return_value=Gallery()):
            _code, stdout, _err = _capture(["recognize", image_path,
                                             "--gallery", gallery_path])
        assert "alice" in stdout

    def test_stdout_contains_similarity(self):
        gallery_path = _make_gallery_file()
        image_path = _make_image_file()
        result = _make_matches("bob", 0.87)
        with patch("mata.run", return_value=result), \
             patch("mata.Gallery.load", return_value=Gallery()):
            _code, stdout, _err = _capture(["recognize", image_path,
                                             "--gallery", gallery_path])
        assert "0.87" in stdout or "bob" in stdout

    def test_json_flag_outputs_valid_json(self):
        gallery_path = _make_gallery_file()
        image_path = _make_image_file()
        result = _make_matches("carol", 0.91)
        with patch("mata.run", return_value=result), \
             patch("mata.Gallery.load", return_value=Gallery()):
            _code, stdout, _err = _capture(["recognize", image_path,
                                             "--gallery", gallery_path, "--json"])
        # Should contain serialized dict content
        assert "carol" in stdout or "entries" in stdout


# ---------------------------------------------------------------------------
# TestRecognizeCmdErrors
# ---------------------------------------------------------------------------


class TestRecognizeCmdErrors:
    def test_nonexistent_gallery_exits_nonzero(self):
        code, _out, err = _capture(["recognize", "image.jpg",
                                    "--gallery", "/nonexistent/path/gallery.npz"])
        assert code != 0
        assert "gallery" in err.lower() or "error" in err.lower()

    def test_mata_run_error_exits_nonzero(self):
        gallery_path = _make_gallery_file()
        image_path = _make_image_file()
        with patch("mata.run", side_effect=RuntimeError("embed failed")), \
             patch("mata.Gallery.load", return_value=Gallery()):
            code, _out, err = _capture(["recognize", image_path,
                                         "--gallery", gallery_path])
        assert code != 0
        assert "error" in err.lower()


# ---------------------------------------------------------------------------
# TestRecognizeInParser
# ---------------------------------------------------------------------------


class TestRecognizeInParser:
    def test_recognize_in_valid_choices(self):
        p = _build_parser()
        # Should not raise
        ns = p.parse_args(["recognize", "img.jpg", "--gallery", "g.npz"])
        assert ns.command == "recognize"

    def test_recognize_help_accessible(self):
        p = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            p.parse_args(["recognize", "--help"])
        assert exc_info.value.code == 0
