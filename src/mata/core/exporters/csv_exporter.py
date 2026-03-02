"""CSV exporter for MATA result types.

Exports detection, classification, segmentation, and tracking results to CSV
format for spreadsheet analysis and MOT evaluation tool compatibility.
Supports VisionResult (detections/masks), DetectResult, SegmentResult,
ClassifyResult, and list[VisionResult] (tracking sequences).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from mata.core.logging import get_logger

if TYPE_CHECKING:
    from mata.core.types import ClassifyResult, DetectResult, OCRResult, SegmentResult, VisionResult

logger = get_logger(__name__)


def export_csv(
    result: VisionResult | DetectResult | SegmentResult | ClassifyResult | list,
    output_path: str | Path,
    **kwargs,
) -> None:
    """Export result to CSV file.

    Format depends on result type:
    - VisionResult/DetectResult: id, label, label_name, score, x1, y1, x2, y2, area
    - SegmentResult: id, label, label_name, score, area, has_mask
    - ClassifyResult: rank, label, label_name, score
    - list[VisionResult]: frame_id, track_id, label, label_name, score, x1, y1, x2, y2, area

    Args:
        result: Result object to export. Pass a list[VisionResult] for tracking
            sequences (one VisionResult per frame with Instance.track_id populated).
        output_path: Path to save CSV file
        **kwargs: Additional options:
            - include_header (bool): Whether to write a header row (default True).
              Only used for tracking list exports.

    Raises:
        IOError: If file write fails
        ValueError: If result type has no CSV representation

    Examples:
        >>> result = mata.run("detect", "image.jpg")
        >>> export_csv(result, "detections.csv")
        >>>
        >>> # Classification result
        >>> result = mata.run("classify", "image.jpg")
        >>> export_csv(result, "classifications.csv")
        >>>
        >>> # Tracking sequence
        >>> results = mata.track("video.mp4", model="facebook/detr-resnet-50")
        >>> export_csv(results, "tracks.csv")
    """
    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # list[VisionResult] → tracking CSV
    if isinstance(result, list):
        include_header = kwargs.get("include_header", True)
        try:
            _export_tracks_csv(result, output_path, include_header=include_header)
        except Exception as e:
            logger.error(f"Failed to write tracking CSV to {output_path}: {e}")
            raise OSError(f"Failed to export tracking CSV: {e}") from e
        return

    # Determine result type and export accordingly
    result_type = type(result).__name__

    try:
        if result_type in ("VisionResult", "DetectResult"):
            _export_detections_csv(result, output_path)
        elif result_type == "SegmentResult":
            _export_segments_csv(result, output_path)
        elif result_type == "ClassifyResult":
            _export_classifications_csv(result, output_path)
        elif result_type == "OCRResult":
            _export_ocr_csv(result, output_path)
        else:
            raise ValueError(f"Unsupported result type for CSV export: {result_type}")
    except Exception as e:
        logger.error(f"Failed to write CSV to {output_path}: {e}")
        raise OSError(f"Failed to export CSV: {e}") from e


def export_tracks_csv(
    results: list[VisionResult],
    output_path: str | Path,
    include_header: bool = True,
) -> None:
    """Export tracking results to CSV with per-frame track data.

    This is the public convenience wrapper for exporting a list of
    per-frame VisionResult objects (as returned by ``mata.track()``)
    to a CSV file compatible with MOT evaluation tools.

    Columns: frame_id, track_id, label, label_name, score,
             x1, y1, x2, y2, area

    Args:
        results: List of VisionResult (one per frame). Instances must have
            ``track_id`` populated (use ``mata.track()`` to produce these).
        output_path: Output CSV file path.
        include_header: Whether to write a column-header row (default True).

    Raises:
        IOError: If the file cannot be written.

    Examples:
        >>> results = mata.track("video.mp4", model="facebook/detr-resnet-50")
        >>> export_tracks_csv(results, "tracks.csv")
        >>>
        >>> # No header — e.g. for tools that expect raw MOT data
        >>> export_tracks_csv(results, "tracks_raw.csv", include_header=False)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _export_tracks_csv(results, output_path, include_header=include_header)
    except Exception as e:
        logger.error(f"Failed to write tracking CSV to {output_path}: {e}")
        raise OSError(f"Failed to export tracking CSV: {e}") from e


def _export_detections_csv(result: VisionResult | DetectResult, output_path: Path) -> None:
    """Export detections to CSV.

    CSV format: id, label, label_name, score, x1, y1, x2, y2, area
    """
    # Get instances with bboxes
    if hasattr(result, "detections"):
        instances = result.detections
    elif hasattr(result, "instances"):
        instances = [inst for inst in result.instances if inst.bbox is not None]
    else:
        instances = []

    rows: list[dict] = []
    for idx, inst in enumerate(instances):
        row = {
            "id": idx,
            "label": inst.label,
            "label_name": inst.label_name or f"class_{inst.label}",
            "score": f"{inst.score:.4f}",
        }

        if inst.bbox:
            x1, y1, x2, y2 = inst.bbox
            row.update(
                {
                    "x1": f"{x1:.2f}",
                    "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}",
                    "y2": f"{y2:.2f}",
                }
            )
        else:
            row.update({"x1": "", "y1": "", "x2": "", "y2": ""})

        row["area"] = str(getattr(inst, "area", "")) if getattr(inst, "area", None) is not None else ""
        rows.append(row)

    # Write CSV
    fieldnames = ["id", "label", "label_name", "score", "x1", "y1", "x2", "y2", "area"]
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Exported {len(rows)} detections to {output_path}")


def _export_segments_csv(result: SegmentResult, output_path: Path) -> None:
    """Export segmentation masks to CSV.

    CSV format: id, label, label_name, score, area, has_mask, x1, y1, x2, y2
    """
    instances = result.masks if hasattr(result, "masks") else []

    rows: list[dict] = []
    for idx, inst in enumerate(instances):
        row = {
            "id": idx,
            "label": inst.label,
            "label_name": inst.label_name or f"class_{inst.label}",
            "score": f"{inst.score:.4f}",
            "area": str(inst.area) if inst.area is not None else "",
            "has_mask": "yes" if inst.mask is not None else "no",
        }

        # Add bbox if available
        if inst.bbox:
            x1, y1, x2, y2 = inst.bbox
            row.update(
                {
                    "x1": f"{x1:.2f}",
                    "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}",
                    "y2": f"{y2:.2f}",
                }
            )
        else:
            row.update({"x1": "", "y1": "", "x2": "", "y2": ""})

        rows.append(row)

    # Write CSV
    fieldnames = ["id", "label", "label_name", "score", "area", "has_mask", "x1", "y1", "x2", "y2"]
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Exported {len(rows)} segments to {output_path}")


def _export_classifications_csv(result: ClassifyResult, output_path: Path) -> None:
    """Export classifications to CSV.

    CSV format: rank, label, label_name, score
    """
    predictions = result.predictions if hasattr(result, "predictions") else []

    rows: list[dict] = []
    for rank, pred in enumerate(predictions, start=1):
        rows.append(
            {
                "rank": rank,
                "label": pred.label,
                "label_name": pred.label_name or f"class_{pred.label}",
                "score": f"{pred.score:.4f}",
            }
        )

    # Write CSV
    fieldnames = ["rank", "label", "label_name", "score"]
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Exported {len(rows)} classifications to {output_path}")


def export_ocr_csv(result: OCRResult, output_path: str | Path, **kwargs) -> None:
    """Export OCRResult to CSV with one row per text region.

    Columns: region_index, text, score, bbox_x1, bbox_y1, bbox_x2, bbox_y2, label

    Args:
        result: OCRResult to export
        output_path: Path to save CSV file
        **kwargs: Reserved for future options

    Raises:
        IOError: If file write fails

    Examples:
        >>> result = mata.run("ocr", "image.jpg", model="easyocr")
        >>> export_ocr_csv(result, "ocr_results.csv")
        >>> # Or via result.save():
        >>> result.save("ocr_results.csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _export_ocr_csv(result, output_path)
    except Exception as e:
        logger.error(f"Failed to write OCR CSV to {output_path}: {e}")
        raise OSError(f"Failed to export OCR CSV: {e}") from e


def _export_ocr_csv(result: OCRResult, output_path: Path) -> None:
    """Export OCRResult to CSV.

    CSV format: region_index, text, score, bbox_x1, bbox_y1, bbox_x2, bbox_y2, label
    """
    regions = result.regions if hasattr(result, "regions") else []

    fieldnames = ["region_index", "text", "score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "label"]
    rows: list[dict] = []
    for i, region in enumerate(regions):
        if region.bbox:
            x1, y1, x2, y2 = region.bbox
            bbox_x1 = f"{x1:.2f}"
            bbox_y1 = f"{y1:.2f}"
            bbox_x2 = f"{x2:.2f}"
            bbox_y2 = f"{y2:.2f}"
        else:
            bbox_x1 = bbox_y1 = bbox_x2 = bbox_y2 = ""
        rows.append(
            {
                "region_index": i,
                "text": region.text,
                "score": f"{region.score:.4f}",
                "bbox_x1": bbox_x1,
                "bbox_y1": bbox_y1,
                "bbox_x2": bbox_x2,
                "bbox_y2": bbox_y2,
                "label": region.label or "",
            }
        )

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Exported {len(rows)} OCR regions to {output_path}")


def _export_tracks_csv(
    results: list[VisionResult],
    output_path: Path,
    include_header: bool = True,
) -> None:
    """Export tracking results to CSV with per-frame track data.

    Columns: frame_id, track_id, label, label_name, score, x1, y1, x2, y2, area

    One row per tracked instance per frame.  Column order matches the YOLO
    MOT-compatible export format so downstream evaluation tools can consume
    the file without any conversion.

    Args:
        results: List of VisionResult (one per frame).  ``result.meta`` may
            contain ``"frame_idx"`` (set by ``mata.track()``) which is used
            as ``frame_id``; otherwise the list index is used.
        output_path: Resolved Path to write the CSV file to.
        include_header: Whether to write the column-header row.
    """
    fieldnames = [
        "frame_id",
        "track_id",
        "label",
        "label_name",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "area",
    ]

    rows: list[dict] = []

    for list_idx, result in enumerate(results):
        # Prefer the frame index stored by mata.track(); fall back to list position.
        frame_id = result.meta.get("frame_idx", list_idx) if hasattr(result, "meta") and result.meta else list_idx

        instances = result.instances if hasattr(result, "instances") else []
        for inst in instances:
            # Skip mask-only instances that carry no bounding box.
            if inst.bbox is None:
                continue

            x1, y1, x2, y2 = inst.bbox

            # Compute area: prefer the stored value, fall back to bbox area.
            if inst.area is not None:
                area = inst.area
            else:
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                area = w * h

            rows.append(
                {
                    "frame_id": frame_id,
                    "track_id": inst.track_id if inst.track_id is not None else -1,
                    "label": inst.label,
                    "label_name": inst.label_name or f"class_{inst.label}",
                    "score": f"{inst.score:.4f}",
                    "x1": f"{x1:.2f}",
                    "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}",
                    "y2": f"{y2:.2f}",
                    "area": f"{area:.2f}",
                }
            )

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if include_header:
            writer.writeheader()
        writer.writerows(rows)

    total_frames = len(results)
    logger.info(f"Exported {len(rows)} tracked instances across {total_frames} frame(s) to {output_path}")
