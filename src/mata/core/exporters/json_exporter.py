"""JSON exporter for MATA result types.

Exports VisionResult, DetectResult, SegmentResult, ClassifyResult, and DepthResult to JSON format.
Uses the existing to_json() methods with optional pretty-printing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mata.core.logging import get_logger

if TYPE_CHECKING:
    from mata.core.types import ClassifyResult, DepthResult, DetectResult, SegmentResult, VisionResult

logger = get_logger(__name__)


def export_json(
    result: VisionResult | DetectResult | SegmentResult | ClassifyResult | DepthResult,
    output_path: str | Path,
    indent: int = 2,
    **kwargs,
) -> None:
    """Export result to JSON file.

    Uses the result's built-in to_json() method with pretty-printing.

    Args:
        result: Result object to export
        output_path: Path to save JSON file
        indent: Indentation level for pretty-printing (default: 2)
        **kwargs: Additional json.dumps() parameters

    Raises:
        IOError: If file write fails

    Examples:
        >>> result = mata.run("detect", "image.jpg")
        >>> export_json(result, "detections.json")
        >>>
        >>> # Compact format (no indentation)
        >>> export_json(result, "compact.json", indent=None)
    """
    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate JSON string using result's method
    json_str = result.to_json(indent=indent, **kwargs)

    # Write to file
    try:
        output_path.write_text(json_str, encoding="utf-8")
        logger.info(f"Exported JSON to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write JSON to {output_path}: {e}")
        raise OSError(f"Failed to export JSON: {e}") from e


def export_tracking_json(
    results: list[VisionResult],
    output_path: str | Path,
    include_meta: bool = True,
    indent: int = 2,
) -> None:
    """Export tracking results as structured JSON with frame-level indexing.

    Each element in *results* corresponds to one frame.  The output document
    groups instances by frame and appends summary metadata so the file can
    be used directly by MOT evaluation tooling or downstream analytics.

    Output format::

        {
            "frames": [
                {
                    "frame_id": 0,
                    "instances": [
                        {
                            "track_id": 1,
                            "label": 0,
                            "label_name": "person",
                            "bbox": [x1, y1, x2, y2],
                            "score": 0.95
                        },
                        ...
                    ]
                },
                ...
            ],
            "meta": {
                "num_frames": 100,
                "unique_tracks": 15,
                "tracker": "botsort"
            }
        }

    Args:
        results: Sequence of ``VisionResult`` objects, one per video frame.
            Instances are expected to carry a populated ``track_id`` field;
            untracked instances (``track_id=None``) are included with
            ``track_id`` set to ``null``.
        output_path: Destination ``.json`` file path.  Parent directories are
            created automatically.
        include_meta: When *True* (default), append the top-level ``"meta"``
            block with aggregate statistics.  Set to *False* to emit a
            frames-only document.
        indent: JSON indentation level (default: 2).  Pass ``None`` for
            compact (single-line) output.

    Raises:
        OSError: If the output file cannot be written.

    Examples:
        >>> results = mata.track("video.mp4", model="facebook/detr-resnet-50")
        >>> export_tracking_json(results, "tracks.json")
        >>>
        >>> # Without summary metadata
        >>> export_tracking_json(results, "tracks.json", include_meta=False)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    unique_track_ids: set[int] = set()

    for frame_idx, result in enumerate(results):
        # Prefer an explicit frame index embedded by mata.track(), fall back to
        # the enumeration index so the function also works with plain lists.
        frame_id: int = result.meta.get("frame_idx", frame_idx)

        instances_out: list[dict[str, Any]] = []
        for inst in result.instances:
            inst_dict: dict[str, Any] = {
                "track_id": inst.track_id,
                "label": inst.label,
                "label_name": inst.label_name,
                "bbox": list(inst.bbox) if inst.bbox is not None else None,
                "score": inst.score,
            }
            instances_out.append(inst_dict)
            if inst.track_id is not None:
                unique_track_ids.add(inst.track_id)

        frames.append({"frame_id": frame_id, "instances": instances_out})

    document: dict[str, Any] = {"frames": frames}

    if include_meta:
        # Collect tracker name from the first result that carries it.
        tracker_name: str | None = None
        for result in results:
            tracker_name = result.meta.get("tracker")
            if tracker_name is not None:
                break

        document["meta"] = {
            "num_frames": len(results),
            "unique_tracks": len(unique_track_ids),
            "tracker": tracker_name,
        }

    json_str = json.dumps(document, indent=indent)

    try:
        output_path.write_text(json_str, encoding="utf-8")
        logger.info(f"Exported tracking JSON ({len(results)} frames) to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write tracking JSON to {output_path}: {e}")
        raise OSError(f"Failed to export tracking JSON: {e}") from e
