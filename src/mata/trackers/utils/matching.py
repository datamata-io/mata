"""Detection-to-track association utilities.

Provides IoU distance computation, linear assignment solving,
and score fusion for multi-object tracking.

Ported from Ultralytics tracker utils (MIT-compatible).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_xyxy(track: object) -> np.ndarray:
    """Extract xyxy bounding box from a track or detection object.

    Tries ``track.xyxy`` first, then converts from ``track.tlwh``.

    Args:
        track: Object with ``.xyxy`` (N,4) or ``.tlwh`` (N,4) attribute.

    Returns:
        1-D array ``[x1, y1, x2, y2]``.

    Raises:
        AttributeError: If neither attribute is present.
    """
    if hasattr(track, "xyxy"):
        arr = np.asarray(track.xyxy, dtype=np.float32).ravel()
        if arr.shape == (4,):
            return arr
    if hasattr(track, "tlwh"):
        t = np.asarray(track.tlwh, dtype=np.float32).ravel()
        return np.array([t[0], t[1], t[0] + t[2], t[1] + t[3]], dtype=np.float32)
    raise AttributeError(
        f"Object {track!r} has neither 'xyxy' nor 'tlwh' attribute. "
        "Track/detection objects must expose at least one of these."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Vectorized IoU computation between two sets of bounding boxes.

    Uses numpy broadcasting — no Python loops over boxes.

    Args:
        bboxes1: ``(N, 4)`` array in **xyxy** format (x1, y1, x2, y2).
        bboxes2: ``(M, 4)`` array in **xyxy** format.

    Returns:
        ``(N, M)`` float32 IoU matrix.  Non-overlapping or zero-area pairs
        yield 0.0.
    """
    bboxes1 = np.asarray(bboxes1, dtype=np.float64)
    bboxes2 = np.asarray(bboxes2, dtype=np.float64)

    if bboxes1.ndim == 1:
        bboxes1 = bboxes1[np.newaxis, :]
    if bboxes2.ndim == 1:
        bboxes2 = bboxes2[np.newaxis, :]

    if bboxes1.shape[0] == 0 or bboxes2.shape[0] == 0:
        return np.zeros((bboxes1.shape[0], bboxes2.shape[0]), dtype=np.float32)

    # Expand dims for broadcasting: (N,1,4) vs (1,M,4)
    b1 = bboxes1[:, np.newaxis, :]  # (N, 1, 4)
    b2 = bboxes2[np.newaxis, :, :]  # (1, M, 4)

    # Intersection rectangle
    inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = np.minimum(b1[..., 3], b2[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h  # (N, M)

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])  # (N,)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])  # (M,)

    union_area = area1[:, np.newaxis] + area2[np.newaxis, :] - inter_area  # (N, M)

    iou = np.where(union_area > 0.0, inter_area / union_area, 0.0)
    return iou.astype(np.float32)


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """Compute cost matrix based on IoU distance between tracks/detections.

    Args:
        atracks: List of track objects with ``.tlwh`` or ``.xyxy`` property.
        btracks: List of track/detection objects with ``.tlwh`` or ``.xyxy``
            property.

    Returns:
        Cost matrix of shape ``(len(atracks), len(btracks))`` where
        ``cost = 1 – IoU``.  Empty inputs produce a zero-size array.
    """
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)

    a_xyxy = np.stack([_get_xyxy(t) for t in atracks])  # (N, 4)
    b_xyxy = np.stack([_get_xyxy(t) for t in btracks])  # (M, 4)

    ious = iou_batch(a_xyxy, b_xyxy)  # (N, M)
    return (1.0 - ious).astype(np.float32)


def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve linear assignment problem with threshold gating.

    Attempts to use ``scipy.optimize.linear_sum_assignment`` (Hungarian
    algorithm) and falls back to a greedy O(N×M) approach when scipy is
    unavailable.

    Args:
        cost_matrix: ``(N, M)`` cost matrix.  Lower cost = better match.
        thresh: Maximum cost threshold.  Pairs with cost > thresh are
            rejected and their indices appear in the *unmatched* outputs.

    Returns:
        matches: ``(K, 2)`` int array of matched ``(row, col)`` index pairs.
        unmatched_a: 1-D int array of unmatched row indices.
        unmatched_b: 1-D int array of unmatched column indices.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0]),
            np.arange(cost_matrix.shape[1]),
        )

    # --- Try scipy Hungarian algorithm ---
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        valid = cost_matrix[row_ind, col_ind] <= thresh
        matches = np.stack([row_ind[valid], col_ind[valid]], axis=1)
        matched_rows = set(row_ind[valid].tolist())
        matched_cols = set(col_ind[valid].tolist())
    except ImportError:
        # --- Greedy fallback ---
        matches_list: list[list[int]] = []
        matched_rows = set()
        matched_cols = set()

        # Flatten cost matrix and sort by cost ascending
        flat_idx = np.argsort(cost_matrix, axis=None)
        rows, cols = np.unravel_index(flat_idx, cost_matrix.shape)

        for r, c in zip(rows.tolist(), cols.tolist()):
            if cost_matrix[r, c] > thresh:
                break  # Remaining costs are higher; early exit
            if r in matched_rows or c in matched_cols:
                continue
            matches_list.append([r, c])
            matched_rows.add(r)
            matched_cols.add(c)

        matches = np.array(matches_list, dtype=int).reshape(-1, 2)

    unmatched_a = np.array([i for i in range(cost_matrix.shape[0]) if i not in matched_rows], dtype=int)
    unmatched_b = np.array([j for j in range(cost_matrix.shape[1]) if j not in matched_cols], dtype=int)
    return matches, unmatched_a, unmatched_b


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """Fuse detection confidence scores with IoU cost matrix.

    Lowers cost for high-confidence detections, making them more likely
    to match existing tracks.  Mirrors the Ultralytics implementation::

        iou_sim   = 1 - cost_matrix
        fuse_sim  = iou_sim * det_scores   # broadcast over rows
        fuse_cost = 1 - fuse_sim

    Which simplifies to: ``fuse_cost = 1 – (1 – cost) * score``.

    Args:
        cost_matrix: IoU distance matrix of shape ``(N, M)``.
        detections: List of detection objects with a ``.score`` float
            attribute (confidence in ``[0, 1]``).

    Returns:
        Fused cost matrix of the same shape as *cost_matrix*.
    """
    if cost_matrix.size == 0:
        return cost_matrix

    det_scores = np.array([d.score for d in detections], dtype=np.float32)  # (M,)
    # Broadcast scores over the N track rows
    iou_sim = 1.0 - cost_matrix  # (N, M)
    fuse_sim = iou_sim * det_scores[np.newaxis, :]  # (N, M)
    return (1.0 - fuse_sim).astype(np.float32)


def embedding_distance(tracks: list, detections: list) -> np.ndarray:
    """Compute distance matrix based on appearance embeddings.

    .. note::
        **Stub for ReID** — returns an all-infinity distance matrix when
        embeddings are unavailable.  Will be replaced with cosine similarity
        computation in the ReID phase.

    Args:
        tracks: List of track objects with a ``.smooth_feat`` attribute
            (``np.ndarray | None``).
        detections: List of detection objects with a ``.curr_feat``
            attribute (``np.ndarray | None``).

    Returns:
        Cosine distance matrix of shape ``(len(tracks), len(detections))``.
        Currently returns ``np.inf`` for all pairs; once ReID encoders are
        integrated, real cosine distances will be computed.
    """
    n_tracks = len(tracks)
    n_dets = len(detections)

    if n_tracks == 0 or n_dets == 0:
        return np.full((n_tracks, n_dets), np.inf, dtype=np.float32)

    # Attempt real embedding distance when features are present
    track_feats = [getattr(t, "smooth_feat", None) for t in tracks]
    det_feats = [getattr(d, "curr_feat", None) for d in detections]

    if all(f is not None for f in track_feats) and all(f is not None for f in det_feats):
        # L2-normalised cosine distance: 1 - (a · b)
        t_mat = np.stack(track_feats).astype(np.float32)  # (N, D)
        d_mat = np.stack(det_feats).astype(np.float32)  # (M, D)

        # Normalise rows
        t_norm = np.linalg.norm(t_mat, axis=1, keepdims=True)
        d_norm = np.linalg.norm(d_mat, axis=1, keepdims=True)
        t_mat = np.where(t_norm > 0, t_mat / t_norm, t_mat)
        d_mat = np.where(d_norm > 0, d_mat / d_norm, d_mat)

        cosine_sim = t_mat @ d_mat.T  # (N, M)
        return (1.0 - cosine_sim).astype(np.float32)

    # ReID not yet available — return infinity
    return np.full((n_tracks, n_dets), np.inf, dtype=np.float32)
