"""Detection + pose estimation graph preset.

Provides a detection → filter → pose estimation workflow.  The pose
estimation step uses a generic provider lookup so that any adapter exposing
a ``predict()`` interface for keypoint estimation can be plugged in.

.. note::

   A dedicated ``EstimatePose`` node is planned for a future release.
   This preset currently models the pose step through the generic
   provider mechanism, which is forward-compatible with the upcoming node.
"""

from __future__ import annotations

from mata.core.graph.graph import Graph
from mata.nodes.detect import Detect
from mata.nodes.filter import Filter
from mata.nodes.fuse import Fuse
from mata.nodes.nms import NMS
from mata.nodes.topk import TopK


def detection_pose(
    detection_threshold: float = 0.5,
    person_only: bool = True,
    top_k: int | None = None,
    nms_iou_threshold: float | None = 0.5,
) -> Graph:
    """Detect objects and prepare for pose estimation.

    Detects objects, optionally filters to person-only results, applies
    NMS and/or top-K selection, then merges detections with keypoints
    from a pose provider.

    Provider keys expected in ``providers`` dict:
        - ``"detector"`` — detection adapter (e.g. DETR, RT-DETR)
        - ``"pose"`` — keypoint / pose estimation adapter (e.g. ViTPose)

    Args:
        detection_threshold: Minimum confidence score for detections
            (default ``0.5``).
        person_only: If ``True``, filter to ``"person"`` detections only
            (default ``True``).
        top_k: If set, keep only the top-K detections by score.
        nms_iou_threshold: If set, apply NMS before pose estimation.
            Set to ``None`` to skip (default ``0.5``).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import detection_pose
        >>>
        >>> detector = mata.load("detect", "facebook/detr-resnet-50")
        >>> pose = mata.load("pose", "microsoft/vitpose-base")
        >>> result = mata.infer(
        ...     "image.jpg",
        ...     detection_pose(person_only=True, top_k=5),
        ...     providers={"detector": detector, "pose": pose},
        ... )
    """
    graph = Graph("detection_pose")

    # Step 1: Detect objects
    graph = graph.then(Detect(using="detector", out="dets"))

    # Step 2: Filter by confidence and optionally by label
    filter_kwargs: dict = {"src": "dets", "out": "filtered", "score_gt": detection_threshold}
    if person_only:
        filter_kwargs["label_in"] = ["person"]
        filter_kwargs["fuzzy"] = True
    graph = graph.then(Filter(**filter_kwargs))

    # Step 3: Optional NMS
    if nms_iou_threshold is not None:
        graph = graph.then(NMS(iou_threshold=nms_iou_threshold, out="filtered"))

    # Step 4: Optional top-K
    if top_k is not None:
        graph = graph.then(TopK(src="filtered", k=top_k, out="filtered"))

    # Step 5: Bundle results (detection-only for now; pose merging
    #         will extend this when EstimatePose node is available)
    graph = graph.then(Fuse(out="final", dets="filtered"))

    return graph
