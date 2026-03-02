"""Curve plot utilities for MATA evaluation metrics (Task E2).

Four matplotlib-based functions mirror YOLO's output style:
  - plot_pr_curve  → PR_curve.png
  - plot_f1_curve  → F1_curve.png
  - plot_p_curve   → P_curve.png
  - plot_r_curve   → R_curve.png

Visual style (mirrors Ultralytics YOLO):
  - Thin gray lines for individual classes (top-5 by AP/F1/P/R if nc > 10)
  - Bold blue line for the mean curve
  - Legend shows class names with AP50 (PR) or max-value (other curves)
  - save_dir="" → no-op (nothing written, no error raised)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib

import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAX_LEGEND_CLASSES = 10  # show individual-class lines only if nc ≤ this
_TOP_K = 5  # show top-K classes when nc > _MAX_LEGEND_CLASSES


def _save_figure(fig: matplotlib.figure.Figure, save_dir: str, filename: str) -> Path | None:
    """Save *fig* to ``save_dir/filename`` and return the Path, or None."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if not save_dir:
        plt.close(fig)
        return None

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _class_names(names: dict[int, str] | list[str] | None, nc: int) -> list[str]:
    """Return a list of *nc* class-name strings."""
    if names is None:
        return [f"class {i}" for i in range(nc)]
    if isinstance(names, dict):
        return [names.get(i, f"class {i}") for i in range(nc)]
    return list(names)[:nc]


def _select_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-*k* values (descending)."""
    if len(scores) <= k:
        return np.arange(len(scores))
    return np.argsort(scores)[::-1][:k]


def _make_figure(xlabel: str, ylabel: str, title: str):
    """Create and return ``(fig, ax)`` with common styling."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def _add_legend(ax, handles, labels, nc: int) -> None:
    """Add a compact legend; suppress it entirely for very many classes."""
    if not handles:
        return
    ax.legend(handles, labels, loc="lower left", fontsize=8 if nc > 5 else 10)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------


def plot_pr_curve(
    px: np.ndarray,
    py: np.ndarray,
    ap: np.ndarray,
    save_dir: str = "",
    names: dict[int, str] | list[str] | None = None,
    on_plot: Callable | None = None,
    # Legacy alias kept for backward compat with old stub callers
    save_path: str | Path | None = None,
) -> None:
    """Save a Precision-Recall curve as ``PR_curve.png``.

    Args:
        px:        (1000,) recall values on the x-axis (confidence sweep).
        py:        (nc, 1000) precision values per class.
        ap:        (nc,) AP@0.50 per class — used in legend labels.
        save_dir:  Directory to write ``PR_curve.png``.  ``""`` → no-op.
        names:     Class names as ``{id: name}`` dict or ordered list.
        on_plot:   Optional callback ``fn(path: Path)`` called after save.
        save_path: **Deprecated** — use *save_dir* instead.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    py = np.asarray(py, dtype=np.float32)
    ap = np.asarray(ap, dtype=np.float32)
    px = np.asarray(px, dtype=np.float32)

    if py.ndim == 1:
        py = py[np.newaxis, :]  # (1, 1000)
    nc = py.shape[0]

    class_names = _class_names(names, nc)

    if not save_dir and not save_path:
        return

    fig, ax = _make_figure("Recall", "Precision", "Precision-Recall Curve")

    handles, labels = [], []

    if nc <= _MAX_LEGEND_CLASSES:
        show_idx = np.arange(nc)
    else:
        show_idx = _select_top_k(ap, _TOP_K)

    for i in show_idx:
        lbl = f"{class_names[i]} {ap[i]:.3f}" if nc > 1 else class_names[i]
        (line,) = ax.plot(px, py[i], linewidth=0.5, color="grey")
        handles.append(line)
        labels.append(lbl)

    # Mean curve (bold blue)
    mean_py = py.mean(axis=0)
    mean_ap = ap.mean() if len(ap) > 0 else 0.0
    mean_lbl = f"all classes {mean_ap:.3f} mAP@0.5" if nc > 1 else class_names[0]
    (mean_line,) = ax.plot(px, mean_py, linewidth=3, color="#1f77b4", label=mean_lbl)
    handles.insert(0, mean_line)
    labels.insert(0, mean_lbl)

    _add_legend(ax, handles, labels, nc)

    out_path = _save_figure(fig, save_dir, "PR_curve.png")

    # Legacy save_path fallback
    if out_path is None and save_path:
        fig, ax = _make_figure("Recall", "Precision", "Precision-Recall Curve")
        for i in show_idx:
            ax.plot(px, py[i], linewidth=0.5, color="grey")
        ax.plot(px, mean_py, linewidth=3, color="#1f77b4")
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_path = sp

    if on_plot is not None and out_path is not None:
        on_plot(out_path)


def plot_f1_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: str = "",
    names: dict[int, str] | list[str] | None = None,
    on_plot: Callable | None = None,
    # Legacy alias kept for backward compat with old stub callers
    f1: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Save an F1-Confidence curve as ``F1_curve.png``.

    Args:
        px:       (1000,) confidence values on the x-axis.
        py:       (nc, 1000) F1 values per class *or* positional alias for
                  the old ``f1`` parameter.
        save_dir: Directory to write ``F1_curve.png``.  ``""`` → no-op.
        names:    Class names as ``{id: name}`` dict or ordered list.
        on_plot:  Optional callback ``fn(path: Path)`` called after save.
        f1:       **Deprecated** positional alias for *py*.
        save_path: **Deprecated** — use *save_dir* instead.
    """
    # Handle legacy positional call: plot_f1_curve(px, f1, save_path, names)
    if f1 is not None:
        py = f1

    import matplotlib.pyplot as plt  # noqa: PLC0415

    py = np.asarray(py, dtype=np.float32)
    px = np.asarray(px, dtype=np.float32)

    if py.ndim == 1:
        py = py[np.newaxis, :]
    nc = py.shape[0]

    class_names = _class_names(names, nc)

    if not save_dir and not save_path:
        return

    fig, ax = _make_figure("Confidence", "F1", "F1-Confidence Curve")

    # Max F1 per class — used for legend labels and top-K selection
    max_f1 = py.max(axis=1)  # (nc,)

    if nc <= _MAX_LEGEND_CLASSES:
        show_idx = np.arange(nc)
    else:
        show_idx = _select_top_k(max_f1, _TOP_K)

    handles, labels = [], []
    for i in show_idx:
        lbl = f"{class_names[i]} {max_f1[i]:.3f}" if nc > 1 else class_names[i]
        (line,) = ax.plot(px, py[i], linewidth=0.5, color="grey")
        handles.append(line)
        labels.append(lbl)

    mean_py = py.mean(axis=0)
    mean_max = float(mean_py.max())
    mean_lbl = f"all classes {mean_max:.3f}" if nc > 1 else class_names[0]
    (mean_line,) = ax.plot(px, mean_py, linewidth=3, color="#1f77b4", label=mean_lbl)
    handles.insert(0, mean_line)
    labels.insert(0, mean_lbl)

    _add_legend(ax, handles, labels, nc)

    out_path = _save_figure(fig, save_dir, "F1_curve.png")

    if out_path is None and save_path:
        fig, ax = _make_figure("Confidence", "F1", "F1-Confidence Curve")
        for i in show_idx:
            ax.plot(px, py[i], linewidth=0.5, color="grey")
        ax.plot(px, mean_py, linewidth=3, color="#1f77b4")
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_path = sp

    if on_plot is not None and out_path is not None:
        on_plot(out_path)


def plot_p_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: str = "",
    names: dict[int, str] | list[str] | None = None,
    on_plot: Callable | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Save a Precision-Confidence curve as ``P_curve.png``.

    Args:
        px:       (1000,) confidence values on the x-axis.
        py:       (nc, 1000) precision values per class.
        save_dir: Directory to write ``P_curve.png``.  ``""`` → no-op.
        names:    Class names as ``{id: name}`` dict or ordered list.
        on_plot:  Optional callback ``fn(path: Path)`` called after save.
        save_path: **Deprecated** — use *save_dir* instead.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    py = np.asarray(py, dtype=np.float32)
    px = np.asarray(px, dtype=np.float32)

    if py.ndim == 1:
        py = py[np.newaxis, :]
    nc = py.shape[0]

    class_names = _class_names(names, nc)

    if not save_dir and not save_path:
        return

    fig, ax = _make_figure("Confidence", "Precision", "Precision-Confidence Curve")

    max_p = py.max(axis=1)

    if nc <= _MAX_LEGEND_CLASSES:
        show_idx = np.arange(nc)
    else:
        show_idx = _select_top_k(max_p, _TOP_K)

    handles, labels = [], []
    for i in show_idx:
        lbl = f"{class_names[i]} {max_p[i]:.3f}" if nc > 1 else class_names[i]
        (line,) = ax.plot(px, py[i], linewidth=0.5, color="grey")
        handles.append(line)
        labels.append(lbl)

    mean_py = py.mean(axis=0)
    mean_max = float(mean_py.max())
    mean_lbl = f"all classes {mean_max:.3f}" if nc > 1 else class_names[0]
    (mean_line,) = ax.plot(px, mean_py, linewidth=3, color="#1f77b4", label=mean_lbl)
    handles.insert(0, mean_line)
    labels.insert(0, mean_lbl)

    _add_legend(ax, handles, labels, nc)

    out_path = _save_figure(fig, save_dir, "P_curve.png")

    if out_path is None and save_path:
        fig, ax = _make_figure("Confidence", "Precision", "Precision-Confidence Curve")
        for i in show_idx:
            ax.plot(px, py[i], linewidth=0.5, color="grey")
        ax.plot(px, mean_py, linewidth=3, color="#1f77b4")
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_path = sp

    if on_plot is not None and out_path is not None:
        on_plot(out_path)


def plot_r_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: str = "",
    names: dict[int, str] | list[str] | None = None,
    on_plot: Callable | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Save a Recall-Confidence curve as ``R_curve.png``.

    Args:
        px:       (1000,) confidence values on the x-axis.
        py:       (nc, 1000) recall values per class.
        save_dir: Directory to write ``R_curve.png``.  ``""`` → no-op.
        names:    Class names as ``{id: name}`` dict or ordered list.
        on_plot:  Optional callback ``fn(path: Path)`` called after save.
        save_path: **Deprecated** — use *save_dir* instead.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    py = np.asarray(py, dtype=np.float32)
    px = np.asarray(px, dtype=np.float32)

    if py.ndim == 1:
        py = py[np.newaxis, :]
    nc = py.shape[0]

    class_names = _class_names(names, nc)

    if not save_dir and not save_path:
        return

    fig, ax = _make_figure("Confidence", "Recall", "Recall-Confidence Curve")

    max_r = py.max(axis=1)

    if nc <= _MAX_LEGEND_CLASSES:
        show_idx = np.arange(nc)
    else:
        show_idx = _select_top_k(max_r, _TOP_K)

    handles, labels = [], []
    for i in show_idx:
        lbl = f"{class_names[i]} {max_r[i]:.3f}" if nc > 1 else class_names[i]
        (line,) = ax.plot(px, py[i], linewidth=0.5, color="grey")
        handles.append(line)
        labels.append(lbl)

    mean_py = py.mean(axis=0)
    mean_max = float(mean_py.max())
    mean_lbl = f"all classes {mean_max:.3f}" if nc > 1 else class_names[0]
    (mean_line,) = ax.plot(px, mean_py, linewidth=3, color="#1f77b4", label=mean_lbl)
    handles.insert(0, mean_line)
    labels.insert(0, mean_lbl)

    _add_legend(ax, handles, labels, nc)

    out_path = _save_figure(fig, save_dir, "R_curve.png")

    if out_path is None and save_path:
        fig, ax = _make_figure("Confidence", "Recall", "Recall-Confidence Curve")
        for i in show_idx:
            ax.plot(px, py[i], linewidth=0.5, color="grey")
        ax.plot(px, mean_py, linewidth=3, color="#1f77b4")
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_path = sp

    if on_plot is not None and out_path is not None:
        on_plot(out_path)
