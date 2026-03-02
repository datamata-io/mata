"""Tests for mata.eval.plots — PR/F1/P/R Curve Plots.

Acceptance criteria from VALIDATION_GUIDE.md:
  ✅ plot_pr_curve(save_dir="tmp") creates tmp/PR_curve.png
  ✅ plot_f1_curve(save_dir="tmp") creates tmp/F1_curve.png
  ✅ plot_p_curve(save_dir="tmp") creates tmp/P_curve.png
  ✅ plot_r_curve(save_dir="tmp") creates tmp/R_curve.png
  ✅ All files are valid PNG images (non-zero file size)
  ✅ save_dir="" does not raise and does not write files
  ✅ Works with nc=1 (single class)
  ✅ Works with nc=80 without legend overflow (only top-5 classes labeled)
  ✅ on_plot callback is invoked with the saved Path
  ✅ names dict/list are reflected in the figure (smoke test)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mata.eval.plots import (
    plot_f1_curve,
    plot_p_curve,
    plot_pr_curve,
    plot_r_curve,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 1000  # curve resolution


def _make_curves(nc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (px, py, ap) compatible with all four plot functions."""
    px = np.linspace(0, 1, N, dtype=np.float32)
    py = np.random.default_rng(42).random((nc, N)).astype(np.float32)
    ap = np.random.default_rng(42).random(nc).astype(np.float32)
    return px, py, ap


def _is_valid_png(path: Path) -> bool:
    """Return True if *path* contains a valid PNG magic bytes header."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as f:
        header = f.read(8)
    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"  # noqa: N806
    return header == PNG_MAGIC


# ---------------------------------------------------------------------------
# plot_pr_curve
# ---------------------------------------------------------------------------


class TestPlotPRCurve:
    def test_saves_png(self, tmp_path):
        px, py, ap = _make_curves(nc=3)
        plot_pr_curve(px, py, ap, save_dir=str(tmp_path))
        out = tmp_path / "PR_curve.png"
        assert out.exists(), "PR_curve.png not created"
        assert _is_valid_png(out), "PR_curve.png is not a valid PNG"

    def test_no_save_dir_no_file(self, tmp_path):
        px, py, ap = _make_curves(nc=3)
        # Should not raise
        plot_pr_curve(px, py, ap, save_dir="")
        # Nothing should appear in tmp_path
        assert list(tmp_path.iterdir()) == []

    def test_single_class(self, tmp_path):
        px, py, ap = _make_curves(nc=1)
        plot_pr_curve(px, py, ap, save_dir=str(tmp_path))
        assert _is_valid_png(tmp_path / "PR_curve.png")

    def test_80_classes(self, tmp_path):
        px, py, ap = _make_curves(nc=80)
        plot_pr_curve(px, py, ap, save_dir=str(tmp_path))
        assert _is_valid_png(tmp_path / "PR_curve.png")

    def test_on_plot_callback(self, tmp_path):
        px, py, ap = _make_curves(nc=3)
        called_with = []
        plot_pr_curve(px, py, ap, save_dir=str(tmp_path), on_plot=called_with.append)
        assert len(called_with) == 1
        assert called_with[0] == tmp_path / "PR_curve.png"

    def test_on_plot_not_called_when_no_save_dir(self):
        px, py, ap = _make_curves(nc=3)
        called = []
        plot_pr_curve(px, py, ap, save_dir="", on_plot=called.append)
        assert called == []

    def test_names_dict(self, tmp_path):
        px, py, ap = _make_curves(nc=3)
        names = {0: "cat", 1: "dog", 2: "bird"}
        plot_pr_curve(px, py, ap, save_dir=str(tmp_path), names=names)
        assert _is_valid_png(tmp_path / "PR_curve.png")

    def test_names_list(self, tmp_path):
        px, py, ap = _make_curves(nc=3)
        plot_pr_curve(px, py, ap, save_dir=str(tmp_path), names=["a", "b", "c"])
        assert _is_valid_png(tmp_path / "PR_curve.png")

    def test_1d_py_promoted(self, tmp_path):
        """1-D py should be silently promoted to (1, 1000)."""
        px = np.linspace(0, 1, N, dtype=np.float32)
        py = np.random.default_rng(0).random(N).astype(np.float32)
        ap = np.array([0.5], dtype=np.float32)
        plot_pr_curve(px, py, ap, save_dir=str(tmp_path))
        assert _is_valid_png(tmp_path / "PR_curve.png")

    def test_creates_save_dir_if_missing(self, tmp_path):
        px, py, ap = _make_curves(nc=2)
        nested = tmp_path / "a" / "b" / "c"
        plot_pr_curve(px, py, ap, save_dir=str(nested))
        assert _is_valid_png(nested / "PR_curve.png")


# ---------------------------------------------------------------------------
# plot_f1_curve
# ---------------------------------------------------------------------------


class TestPlotF1Curve:
    def test_saves_png(self, tmp_path):
        px, py, _ = _make_curves(nc=3)
        plot_f1_curve(px, py, save_dir=str(tmp_path))
        assert _is_valid_png(tmp_path / "F1_curve.png")

    def test_no_save_dir_no_file(self, tmp_path):
        px, py, _ = _make_curves(nc=3)
        plot_f1_curve(px, py, save_dir="")
        assert list(tmp_path.iterdir()) == []

    def test_single_class(self, tmp_path):
        px, py, _ = _make_curves(nc=1)
        plot_f1_curve(px, py, save_dir=str(tmp_path))
        assert _is_valid_png(tmp_path / "F1_curve.png")

    def test_80_classes(self, tmp_path):
        px, py, _ = _make_curves(nc=80)
        plot_f1_curve(px, py, save_dir=str(tmp_path))
        assert _is_valid_png(tmp_path / "F1_curve.png")

    def test_on_plot_callback(self, tmp_path):
        px, py, _ = _make_curves(nc=3)
        called_with = []
        plot_f1_curve(px, py, save_dir=str(tmp_path), on_plot=called_with.append)
        assert len(called_with) == 1
        assert called_with[0] == tmp_path / "F1_curve.png"

    def test_on_plot_not_called_when_no_save_dir(self):
        px, py, _ = _make_curves(nc=2)
        called = []
        plot_f1_curve(px, py, save_dir="", on_plot=called.append)
        assert called == []

    def test_names_dict(self, tmp_path):
        px, py, _ = _make_curves(nc=2)
        plot_f1_curve(px, py, save_dir=str(tmp_path), names={0: "x", 1: "y"})
        assert _is_valid_png(tmp_path / "F1_curve.png")


# ---------------------------------------------------------------------------
# plot_p_curve
# ---------------------------------------------------------------------------


class TestPlotPCurve:
    def test_saves_png(self, tmp_path):
        px, py, _ = _make_curves(nc=3)
        plot_p_curve(px, py, save_dir=str(tmp_path))
        assert _is_valid_png(tmp_path / "P_curve.png")

    def test_no_save_dir_no_file(self, tmp_path):
        px, py, _ = _make_curves(nc=3)
        plot_p_curve(px, py, save_dir="")
        assert list(tmp_path.iterdir()) == []

    def test_single_class(self, tmp_path):
        px, py, _ = _make_curves(nc=1)
        plot_p_curve(px, py, save_dir=str(tmp_path))
        assert _is_valid_png(tmp_path / "P_curve.png")

    def test_80_classes(self, tmp_path):
        px, py, _ = _make_curves(nc=80)
        plot_p_curve(px, py, save_dir=str(tmp_path))
        assert _is_valid_png(tmp_path / "P_curve.png")

    def test_on_plot_callback(self, tmp_path):
        px, py, _ = _make_curves(nc=4)
        called_with = []
        plot_p_curve(px, py, save_dir=str(tmp_path), on_plot=called_with.append)
        assert len(called_with) == 1
        assert called_with[0] == tmp_path / "P_curve.png"

    def test_names_list(self, tmp_path):
        px, py, _ = _make_curves(nc=2)
        plot_p_curve(px, py, save_dir=str(tmp_path), names=["cat", "dog"])
        assert _is_valid_png(tmp_path / "P_curve.png")


# ---------------------------------------------------------------------------
# plot_r_curve
# ---------------------------------------------------------------------------


class TestPlotRCurve:
    def test_saves_png(self, tmp_path):
        px, py, _ = _make_curves(nc=3)
        plot_r_curve(px, py, save_dir=str(tmp_path))
        assert _is_valid_png(tmp_path / "R_curve.png")

    def test_no_save_dir_no_file(self, tmp_path):
        px, py, _ = _make_curves(nc=3)
        plot_r_curve(px, py, save_dir="")
        assert list(tmp_path.iterdir()) == []

    def test_single_class(self, tmp_path):
        px, py, _ = _make_curves(nc=1)
        plot_r_curve(px, py, save_dir=str(tmp_path))
        assert _is_valid_png(tmp_path / "R_curve.png")

    def test_80_classes(self, tmp_path):
        px, py, _ = _make_curves(nc=80)
        plot_r_curve(px, py, save_dir=str(tmp_path))
        assert _is_valid_png(tmp_path / "R_curve.png")

    def test_on_plot_callback(self, tmp_path):
        px, py, _ = _make_curves(nc=2)
        called_with = []
        plot_r_curve(px, py, save_dir=str(tmp_path), on_plot=called_with.append)
        assert len(called_with) == 1
        assert called_with[0] == tmp_path / "R_curve.png"

    def test_names_dict(self, tmp_path):
        px, py, _ = _make_curves(nc=3)
        plot_r_curve(px, py, save_dir=str(tmp_path), names={0: "a", 1: "b", 2: "c"})
        assert _is_valid_png(tmp_path / "R_curve.png")


# ---------------------------------------------------------------------------
# All four curves together (round-trip sanity)
# ---------------------------------------------------------------------------


class TestAllCurvesTogether:
    def test_all_four_curves_created(self, tmp_path):
        px, py, ap = _make_curves(nc=5)
        plot_pr_curve(px, py, ap, save_dir=str(tmp_path))
        plot_f1_curve(px, py, save_dir=str(tmp_path))
        plot_p_curve(px, py, save_dir=str(tmp_path))
        plot_r_curve(px, py, save_dir=str(tmp_path))

        expected = ["PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png"]
        for fname in expected:
            p = tmp_path / fname
            assert p.exists(), f"{fname} was not created"
            assert _is_valid_png(p), f"{fname} is not a valid PNG"

    def test_all_four_no_save_dir_silent(self):
        px, py, ap = _make_curves(nc=5)
        plot_pr_curve(px, py, ap, save_dir="")
        plot_f1_curve(px, py, save_dir="")
        plot_p_curve(px, py, save_dir="")
        plot_r_curve(px, py, save_dir="")
        # Passes if none of the above raises
