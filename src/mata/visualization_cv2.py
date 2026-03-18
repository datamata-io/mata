"""OpenCV-based visualization helpers for real-time annotation.

Low-level drawing functions operating on BGR numpy arrays.
Used by AnnotateRT node and available for direct use.

All functions operate in-place on the provided frame and return the same
array to allow chaining.  ``cv2`` is imported lazily so that importing this
module does NOT raise ``ImportError`` in environments without OpenCV.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mata.core.artifacts.cross_matches import CrossMatches


def track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color from track ID via MD5 hash.

    The color generation algorithm matches the ``_track_color()`` helper used
    in ``camera_agent.py`` and ``api.py``.

    Args:
        track_id: Integer track identifier.

    Returns:
        A ``(B, G, R)`` tuple with values in ``[0, 255]``.
    """
    h = hashlib.md5(str(track_id).encode()).hexdigest()
    r = int(h[4:6], 16)
    g = int(h[8:10], 16)
    b = int(h[12:14], 16)
    # Boost saturation: ensure at least one channel is bright
    if max(r, g, b) < 128:
        r = min(r + 128, 255)
    return (b, g, r)  # BGR


def draw_boxes(
    frame: np.ndarray,
    instances: list[Any],
    *,
    show_labels: bool = True,
    show_scores: bool = True,
    show_track_ids: bool = True,
    line_width: int = 2,
    cross_matches: CrossMatches | None = None,
) -> np.ndarray:
    """Draw bounding boxes, labels, scores, and cross-camera highlights.

    Accepts any objects that expose ``.bbox`` (4-element xyxy sequence) and
    optionally ``.track_id``, ``.score``, and ``.label`` / ``.label_name``
    attributes (duck-typed — works with ``Track``, ``Instance``, or custom
    objects).

    Cross-camera matches are highlighted with a double yellow border around
    the box.  The label text gains a ``~camId#trackId`` suffix.

    Args:
        frame: HWC uint8 BGR NumPy array.  Modified in-place.
        instances: List of detection/track objects to draw.
        show_labels: Include class label in the overlay text.
        show_scores: Include confidence score in the overlay text.
        show_track_ids: Prepend ``#id`` to the overlay text.
        line_width: Thickness of the bounding box border in pixels.
        cross_matches: Optional ``CrossMatches`` artifact.  When provided,
            matched tracks receive a yellow double-border highlight.

    Returns:
        The same ``frame`` array (for chaining).
    """
    try:
        import cv2
    except ImportError:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Build a quick look-up of matched local track IDs → CrossMatch objects
    matched: dict[int, Any] = {}
    if cross_matches is not None:
        for m in cross_matches.matches:
            matched[m.local_track_id] = m

    for inst in instances:
        bbox = inst.bbox
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        tid = getattr(inst, "track_id", None)
        score = getattr(inst, "score", None)
        label_name = getattr(inst, "label_name", None) or str(
            getattr(inst, "label", "")
        )

        color = track_color(tid) if tid is not None else (0, 255, 0)

        # Cross-camera highlight: yellow double border
        is_xcam = tid is not None and tid in matched
        if is_xcam:
            cv2.rectangle(
                frame,
                (x1 - 3, y1 - 3),
                (x2 + 3, y2 + 3),
                (0, 255, 255),
                line_width + 1,
            )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)

        # Build label text
        parts: list[str] = []
        if show_track_ids and tid is not None:
            parts.append(f"#{tid}")
        if show_labels and label_name:
            parts.append(label_name)
        if show_scores and score is not None:
            parts.append(f"{score:.2f}")
        if is_xcam:
            m = matched[tid]  # type: ignore[index]
            parts.append(f"~{m.remote_camera_id}#{m.remote_track_id}")
        text = " ".join(parts)

        if not text:
            continue

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        ly = max(y1 - 4, th + baseline)
        cv2.rectangle(
            frame,
            (x1, ly - th - baseline),
            (x1 + tw, ly + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            text,
            (x1, ly),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    return frame


def draw_trails(
    frame: np.ndarray,
    trail_history: dict[int, list[tuple[int, int]]],
    trail_length: int = 30,
) -> np.ndarray:
    """Draw trajectory trails as fading polylines.

    Trails become progressively thicker and brighter towards the current
    position (alpha-fade effect via varying line thickness).

    Args:
        frame: HWC uint8 BGR NumPy array.  Modified in-place.
        trail_history: Mutable mapping of ``track_id`` → list of ``(cx, cy)``
            centre points in chronological order.
        trail_length: Maximum number of points retained per trail (informational
            here — callers manage trimming).

    Returns:
        The same ``frame`` array (for chaining).
    """
    try:
        import cv2
    except ImportError:
        return frame

    for tid, pts in trail_history.items():
        if len(pts) < 2:
            continue
        color = track_color(tid)
        n = len(pts)
        for i in range(1, n):
            alpha = i / n
            thickness = max(1, int(2 * alpha))
            cv2.line(frame, pts[i - 1], pts[i], color, thickness)

    return frame


def draw_camera_label(
    frame: np.ndarray,
    label: str,
    color: tuple[int, int, int] = (255, 100, 60),
) -> np.ndarray:
    """Draw a colored camera label bar in the top-left corner.

    Renders a filled rectangle with the label text overlaid in black,
    matching the style used in ``camera_agent.py``.

    Args:
        frame: HWC uint8 BGR NumPy array.  Modified in-place.
        label: Text to display (e.g. ``" CAM 1 "``).
        color: BGR fill color for the bar.  Defaults to the first palette
            entry from ``camera_agent.py``.

    Returns:
        The same ``frame`` array (for chaining).
    """
    try:
        import cv2
    except ImportError:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (lw, lh), lb = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.rectangle(frame, (0, 0), (lw + 8, lh + lb + 8), color, cv2.FILLED)
    cv2.putText(frame, label, (4, lh + 4), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    return frame
