"""Rich display rendering for Jupyter Notebook / JupyterLab.

Provides render functions that produce HTML strings or PNG bytes for MATA result
types. These are called by _repr_html_() / _repr_png_() on result dataclasses.

All IPython/matplotlib imports are guarded — this module is importable without
Jupyter installed (functions return None when optional deps are missing).
"""
from __future__ import annotations

import base64
import html
import io
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mata.core.artifacts.embeddings import Embeddings
    from mata.core.types import BarcodeResult, ClassifyResult, DepthResult, OCRResult, VisionResult

_MAX_TABLE_ROWS = 20
_TRUNCATION_THRESHOLD = 100

_TABLE_STYLE = (
    "border-collapse:collapse;font-family:monospace,monospace;"
    "font-size:13px;width:100%"
)
_TH_STYLE = (
    "background:#f0f0f0;border:1px solid #ccc;padding:4px 8px;"
    "text-align:left;font-weight:bold"
)
_TD_STYLE = "border:1px solid #ddd;padding:4px 8px"
_TD_NUM_STYLE = "border:1px solid #ddd;padding:4px 8px;text-align:right"


def _th(text: str) -> str:
    return f'<th style="{_TH_STYLE}">{html.escape(text)}</th>'


def _td(text: str, numeric: bool = False) -> str:
    style = _TD_NUM_STYLE if numeric else _TD_STYLE
    return f'<td style="{style}">{html.escape(str(text))}</td>'


def _bbox_str(bbox: tuple[float, ...] | None) -> str:
    if bbox is None:
        return "—"
    return f"({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})"


def _table_wrap(header_html: str, rows_html: str, note: str = "") -> str:
    note_html = f'<p style="font-size:12px;color:#888;margin:4px 0">{html.escape(note)}</p>' if note else ""
    return (
        f'<div style="overflow-x:auto;margin:4px 0">'
        f'<table style="{_TABLE_STYLE}"><thead><tr>{header_html}</tr></thead>'
        f"<tbody>{rows_html}</tbody></table>{note_html}</div>"
    )


def render_vision_html(result: VisionResult) -> str | None:
    """Render VisionResult as an HTML table with optional inline image overlay."""
    try:
        instances = result.instances
        n = len(instances)

        # --- Optional image overlay ---
        img_html = ""
        input_path = result.meta.get("input_path")
        if input_path:
            try:
                import os

                from mata.core.exporters import export_image

                if os.path.isfile(str(input_path)):
                    buf = io.BytesIO()
                    export_image(result, buf, image=str(input_path), format="image")
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    img_html = (
                        f'<img src="data:image/png;base64,{b64}" '
                        f'style="max-width:100%;height:auto;display:block;margin-bottom:6px" />'
                    )
            except Exception:
                pass

        # --- Table ---
        has_track = any(inst.track_id is not None for inst in instances)
        headers = [_th("Label"), _th("Score"), _th("BBox")]
        if has_track:
            headers.append(_th("Track ID"))

        rows = []
        display_instances = instances[:_MAX_TABLE_ROWS] if n > _TRUNCATION_THRESHOLD else instances
        for inst in display_instances:
            label = inst.label_name or str(inst.label)
            cells = [
                _td(label),
                _td(f"{inst.score:.3f}", numeric=True),
                _td(_bbox_str(inst.bbox)),
            ]
            if has_track:
                cells.append(_td(str(inst.track_id) if inst.track_id is not None else "—"))
            rows.append(f'<tr>{"".join(cells)}</tr>')

        note = f"…and {n - _MAX_TABLE_ROWS} more" if n > _TRUNCATION_THRESHOLD else ""
        header_html = "".join(headers)
        rows_html = "".join(rows)

        title = f'<p style="font-weight:bold;margin:4px 0">VisionResult — {n} instance{"s" if n != 1 else ""}</p>'
        text_html = ""
        if result.text:
            text_html = (
                f'<pre style="background:#f5f5f5;border:1px solid #ddd;'
                f'padding:6px;font-size:12px;overflow-x:auto">{html.escape(result.text)}</pre>'
            )
        table_html = _table_wrap(header_html, rows_html, note)
        return f"<div>{title}{img_html}{text_html}{table_html}</div>"

    except Exception:
        return None


def render_classify_html(result: ClassifyResult) -> str | None:
    """Render ClassifyResult as a horizontal SVG bar chart + score table."""
    try:
        preds = result.predictions
        n = len(preds)
        if n == 0:
            title = '<p style="font-weight:bold;margin:4px 0">ClassifyResult — 0 predictions</p>'
            return f"<div>{title}<p style='color:#888;font-size:12px'>No predictions.</p></div>"

        top5 = preds[:5]
        max_score = max(p.score for p in top5) or 1.0

        # SVG bar chart
        bar_h = 22
        bar_gap = 4
        label_w = 160
        bar_max_w = 260
        svg_w = label_w + bar_max_w + 60
        svg_h = len(top5) * (bar_h + bar_gap) + 8

        bars = []
        for i, pred in enumerate(top5):
            y = i * (bar_h + bar_gap) + 4
            label = html.escape(pred.label_name or str(pred.label))
            fill_w = int((pred.score / max_score) * bar_max_w)
            score_str = f"{pred.score:.3f}"
            bars.append(
                f'<text x="{label_w - 4}" y="{y + bar_h - 6}" '
                f'text-anchor="end" font-size="12" font-family="monospace">{label}</text>'
                f'<rect x="{label_w}" y="{y}" width="{fill_w}" height="{bar_h}" '
                f'fill="#4a90d9" rx="2"/>'
                f'<text x="{label_w + fill_w + 4}" y="{y + bar_h - 6}" '
                f'font-size="12" font-family="monospace" fill="#555">{score_str}</text>'
            )

        svg = (
            f'<svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="display:block;margin-bottom:6px">'
            f'{"".join(bars)}</svg>'
        )

        # Table
        header_html = _th("Label") + _th("Score")
        rows = "".join(
            f'<tr>{_td(p.label_name or str(p.label))}{_td(f"{p.score:.4f}", numeric=True)}</tr>'
            for p in preds[:_MAX_TABLE_ROWS]
        )
        note = f"…and {n - _MAX_TABLE_ROWS} more" if n > _TRUNCATION_THRESHOLD else ""
        table_html = _table_wrap(header_html, rows, note)

        title = f'<p style="font-weight:bold;margin:4px 0">ClassifyResult — {n} prediction{"s" if n != 1 else ""}</p>'
        return f"<div>{title}{svg}{table_html}</div>"

    except Exception:
        return None


def render_depth_png(result: DepthResult) -> bytes | None:
    """Render DepthResult as colormap PNG bytes (magma colormap)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        depth = result.normalized if result.normalized is not None else result.depth
        depth_arr = np.asarray(depth, dtype=np.float32)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        im = ax.imshow(depth_arr, cmap="magma")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Depth Map", fontsize=11)
        ax.axis("off")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception:
        return None


def render_ocr_html(result: OCRResult) -> str | None:
    """Render OCRResult as an HTML text region table."""
    try:
        regions = result.regions
        n = len(regions)

        header_html = _th("Text") + _th("Score") + _th("BBox") + _th("Label")
        display = regions[:_MAX_TABLE_ROWS] if n > _TRUNCATION_THRESHOLD else regions
        rows = "".join(
            f"<tr>"
            f"{_td(r.text)}"
            f'{_td(f"{r.score:.3f}", numeric=True)}'
            f"{_td(_bbox_str(r.bbox))}"
            f"{_td(r.label or '—')}"
            f"</tr>"
            for r in display
        )
        note = f"…and {n - _MAX_TABLE_ROWS} more" if n > _TRUNCATION_THRESHOLD else ""
        table_html = _table_wrap(header_html, rows, note)
        title = f'<p style="font-weight:bold;margin:4px 0">OCRResult — {n} region{"s" if n != 1 else ""}</p>'
        return f"<div>{title}{table_html}</div>"

    except Exception:
        return None


def render_barcode_html(result: BarcodeResult) -> str | None:
    """Render BarcodeResult as an HTML decoded barcode table."""
    try:
        barcodes = result.barcodes
        n = len(barcodes)

        header_html = _th("Data") + _th("Type") + _th("Score") + _th("BBox")
        display = barcodes[:_MAX_TABLE_ROWS] if n > _TRUNCATION_THRESHOLD else barcodes
        rows = "".join(
            f"<tr>"
            f"{_td(b.data)}"
            f"{_td(b.type)}"
            f'{_td(f"{b.score:.3f}", numeric=True)}'
            f"{_td(_bbox_str(b.bbox))}"
            f"</tr>"
            for b in display
        )
        note = f"…and {n - _MAX_TABLE_ROWS} more" if n > _TRUNCATION_THRESHOLD else ""
        table_html = _table_wrap(header_html, rows, note)
        title = f'<p style="font-weight:bold;margin:4px 0">BarcodeResult — {n} barcode{"s" if n != 1 else ""}</p>'
        return f"<div>{title}{table_html}</div>"

    except Exception:
        return None


def render_embeddings_html(result: Embeddings) -> str | None:
    """Render Embeddings artifact as an HTML summary table."""
    try:
        n, d = result.vectors.shape
        normalized_str = "Yes" if result.normalized else "No"
        dtype_str = str(result.vectors.dtype)

        summary_rows = (
            f"<tr>{_td('Shape')}{_td(f'({n}, {d})')}</tr>"
            f"<tr>{_td('Dimensions')}{_td(str(d), numeric=True)}</tr>"
            f"<tr>{_td('Count')}{_td(str(n), numeric=True)}</tr>"
            f"<tr>{_td('dtype')}{_td(dtype_str)}</tr>"
            f"<tr>{_td('Normalized')}{_td(normalized_str)}</tr>"
        )
        summary_table = _table_wrap(_th("Property") + _th("Value"), summary_rows)

        ids_html = ""
        if result.instance_ids:
            preview = list(result.instance_ids[:5])
            if len(result.instance_ids) > 5:
                preview.append("…")
            ids_str = ", ".join(html.escape(str(i)) for i in preview)
            ids_html = (
                f'<p style="font-size:12px;color:#555;margin:4px 0">'
                f"<b>Instance IDs (first 5):</b> {ids_str}</p>"
            )

        title = f'<p style="font-weight:bold;margin:4px 0">Embeddings — {n} vectors × {d} dims</p>'
        return f"<div>{title}{summary_table}{ids_html}</div>"

    except Exception:
        return None


def show(result: Any, image: str | None = None, **kwargs: Any) -> None:
    """Display a MATA result object in the current Jupyter notebook cell.

    Args:
        result: Any MATA result type (VisionResult, ClassifyResult, etc.)
        image: Optional path to source image for overlay rendering.
        **kwargs: Additional arguments passed to the underlying render function.
    """
    try:
        from IPython.display import HTML, Image, display
    except ImportError as exc:
        raise ImportError(
            "Notebook display requires IPython. Install with: pip install datamata[notebook]"
        ) from exc

    # Inject image path into meta so render functions can use it
    if image is not None and hasattr(result, "meta"):
        try:
            from mata.core.types import VisionResult

            if isinstance(result, VisionResult):
                meta_with_image = {**result.meta, "input_path": image}
                from dataclasses import replace

                result = replace(result, meta=meta_with_image)
        except Exception:
            pass

    # Try _repr_png_() first (DepthResult), then _repr_html_()
    if hasattr(result, "_repr_png_"):
        png_bytes = result._repr_png_()
        if png_bytes is not None:
            display(Image(data=png_bytes))
            return

    if hasattr(result, "_repr_html_"):
        html_str = result._repr_html_()
        if html_str is not None:
            display(HTML(html_str))
            return

    # Fallback: plain repr
    from IPython.display import display as ipython_display

    ipython_display(result)
