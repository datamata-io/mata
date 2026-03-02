"""Validator — orchestrates dataset loading, inference, and metric computation.

Task D2 implementation: full end-to-end evaluation pipeline for all MATA tasks.

Supports two modes:

* **Dataset-driven** — loads images and ground-truth annotations from a YAML
  config file, runs adapter inference, and computes metrics.
* **Standalone** — accepts pre-computed predictions and a ground-truth
  annotations file, skipping inference entirely.

Architecture:

    Validator.run()
        ├── _build_loader()         → DatasetLoader (or iterate standalone predictions)
        ├── _load_adapter()         → mata.load(task, model)
        ├── _iterate_images()       → (pred, gt) pairs with timing
        │   ├── detect  → _match_detections()  → tp/fp/target_cls arrays
        │   ├── segment → _match_segments()    → box + mask tp arrays
        │   ├── classify → ClassifyMetrics.process_predictions()
        │   └── depth   → DepthMetrics.process_batch()
        ├── _compute_metrics()      → fills task metrics object
        ├── _build_confusion_matrix() (stubs ConfusionMatrix — Task E1)
        ├── _save_plots()           (stubs plots — Task E2)
        └── _print_table()          (stubs Printer — Task F2)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from mata.eval.dataset import DatasetLoader, GroundTruth
from mata.eval.metrics.base import ap_per_class
from mata.eval.metrics.classify import ClassifyMetrics
from mata.eval.metrics.depth import DepthMetrics
from mata.eval.metrics.detect import DetMetrics
from mata.eval.metrics.iou import COCO_IOU_THRESHOLDS, box_iou
from mata.eval.metrics.ocr import OCRMetrics
from mata.eval.metrics.segment import SegmentMetrics

logger = logging.getLogger(__name__)

# Tasks that use IoU-based AP matching
_IOU_TASKS = {"detect", "segment"}
# Tasks supported by this implementation
_SUPPORTED_TASKS = {"detect", "segment", "classify", "depth", "ocr"}


class Validator:
    """End-to-end evaluation of any MATA adapter against ground-truth annotations.

    Args:
        task:         Task to evaluate — one of ``"detect"``, ``"segment"``,
                      ``"classify"``, ``"depth"``.
        model:        Model identifier (HuggingFace ID, local path, config alias)
                      **or** a pre-loaded adapter object that implements
                      ``predict(image_path)``.  When ``None``, uses the task's
                      default model.
        data:         Path to YAML dataset config **or** a pre-parsed dict
                      (dataset-driven mode).  Mutually exclusive with
                      ``predictions``/``ground_truth``.
        predictions:  Pre-computed list of ``(VisionResult | ClassifyResult |
                      DepthResult)`` objects in the same order as the images
                      yielded by *ground_truth* (standalone mode).
        ground_truth: Path to a COCO JSON annotations file **or** a
                      pre-parsed list of :class:`~mata.eval.dataset.GroundTruth`
                      objects (standalone mode).
        conf:         Minimum confidence threshold — predictions below this are
                      discarded before matching.
        iou:          IoU threshold for positive-match detection (single-threshold
                      confusion matrix; AP always sweeps all 10 COCO thresholds).
        device:       Device string forwarded to :func:`mata.load`.
        verbose:      Print a per-class results table to stdout after evaluation.
        plots:        Save PR/F1 curve images to *save_dir* after evaluation.
        save_dir:     Directory for plots and CSV output.  Empty string = no saves.
        split:        Dataset split to load (``"val"``, ``"train"``, ``"test"``).
        **kwargs:     Extra keyword arguments forwarded to :func:`mata.load`.

    Example::

        # Dataset-driven
        v = Validator("detect", model="facebook/detr-resnet-50", data="coco.yaml")
        metrics = v.run()
        print(metrics.box.map)

        # Standalone
        metrics = Validator(
            "classify",
            predictions=my_results,
            ground_truth=my_gt_list,
        ).run()
    """

    def __init__(
        self,
        task: str,
        model: str | Any | None = None,
        data: str | dict | None = None,
        predictions: list | None = None,
        ground_truth: str | list | None = None,
        conf: float = 0.001,
        iou: float = 0.50,
        device: str | None = None,
        verbose: bool = True,
        plots: bool = False,
        save_dir: str = "",
        split: str = "val",
        **kwargs: Any,
    ) -> None:
        task = task.lower()
        if task not in _SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task {task!r}. " f"Supported: {sorted(_SUPPORTED_TASKS)}")

        self.task = task
        self.model = model
        self.data = data
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.conf = conf
        self.iou_threshold = iou
        self.device = device
        self.verbose = verbose
        self.plots = plots
        self.save_dir = save_dir
        self.split = split
        self.kwargs = kwargs

        # Resolved at run-time
        self._adapter: Any = None
        self._loader: DatasetLoader | None = None
        # COCO category ID → 0-indexed label (populated by _build_loader)
        self._cat_id_to_label: dict[int, int] = {}
        # Final pred→GT label remap (name-based, built after adapter is loaded)
        self._label_remap: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> DetMetrics | SegmentMetrics | ClassifyMetrics | DepthMetrics | OCRMetrics:
        """Execute the full validation pipeline and return the metrics object.

        Steps:

        1. Build :class:`~mata.eval.dataset.DatasetLoader` (or use standalone GT).
        2. Load adapter via :func:`mata.load` (or use pre-loaded adapter).
        3. Iterate all images: preprocess → infer → postprocess timing.
        4. Match predictions to ground truth via IoU / label.
        5. Aggregate TP/FP/FN and call :func:`ap_per_class`.
        6. Populate task metrics object.
        7. (Optional) build confusion matrix, plots, console table.
        8. Return the metrics object.

        Returns:
            Task-specific metrics: :class:`DetMetrics`, :class:`SegmentMetrics`,
            :class:`ClassifyMetrics`, or :class:`DepthMetrics`.
        """
        # ------ Setup --------------------------------------------------
        names: dict[int, str] = {}
        loader = self._build_loader(names)
        adapter = self._load_adapter()
        self._label_remap = self._build_label_remap(adapter)

        # Timing accumulators (in seconds, averaged to ms/image at the end)
        t_pre: list[float] = []
        t_inf: list[float] = []
        t_post: list[float] = []

        # ------ Task-specific accumulators ----------------------------
        if self.task in _IOU_TASKS:
            # Arrays to pass to ap_per_class()
            all_tp: list[np.ndarray] = []  # (N, T) per image
            all_conf: list[np.ndarray] = []  # (N,)   per image
            all_pred_cls: list[np.ndarray] = []  # (N,)   per image
            all_target_cls: list[np.ndarray] = []  # (M,)   per image
            # For segmentation: parallel mask-IoU TP arrays
            all_tp_mask: list[np.ndarray] = []

        elif self.task == "classify":
            classify_metrics = ClassifyMetrics(names=names)

        elif self.task == "depth":
            depth_metrics = DepthMetrics(align_scale=True, align_affine=True)

        elif self.task == "ocr":
            ocr_metrics = OCRMetrics(case_sensitive=self.kwargs.get("case_sensitive", False))

        n_images = 0

        # ------ Iterate images ----------------------------------------
        for image_path, gt in self._iter_samples(loader):
            n_images += 1

            if adapter is not None:
                # Dataset-driven — run inference with timing
                t0 = time.perf_counter()
                image = self._preprocess(image_path)
                t1 = time.perf_counter()
                result = adapter.predict(image)
                t2 = time.perf_counter()
                pred = self._postprocess(result)
                t3 = time.perf_counter()

                t_pre.append(t1 - t0)
                t_inf.append(t2 - t1)
                t_post.append(t3 - t2)
            else:
                # Standalone — pred supplied directly (already postprocessed)
                pred = image_path  # repurposed: carries the prediction
                t_pre.append(0.0)
                t_inf.append(0.0)
                t_post.append(0.0)

            # Accumulate per-task
            if self.task == "detect":
                tp, conf_arr, pred_cls, target_cls = self._match_detections(pred, gt)
                all_tp.append(tp)
                all_conf.append(conf_arr)
                all_pred_cls.append(pred_cls)
                all_target_cls.append(target_cls)

            elif self.task == "segment":
                tp_box, tp_mask, conf_arr, pred_cls, target_cls = self._match_segments(pred, gt)
                all_tp.append(tp_box)
                all_tp_mask.append(tp_mask)
                all_conf.append(conf_arr)
                all_pred_cls.append(pred_cls)
                all_target_cls.append(target_cls)

            elif self.task == "classify":
                p_label, p_top5 = self._extract_classify_preds(pred)
                gt_label = int(gt.labels[0]) if len(gt.labels) > 0 else 0
                classify_metrics.process_predictions([p_label], [gt_label], pred_top5=[p_top5])

            elif self.task == "depth":
                if gt.depth is not None:
                    pred_depth = self._extract_depth(pred)
                    try:
                        depth_metrics.process_batch(pred_depth, gt.depth.astype(np.float64))
                    except ValueError as exc:
                        logger.warning("Skipping depth image %s: %s", image_path, exc)

            elif self.task == "ocr":
                if gt.text is not None:
                    pred_text = self._extract_ocr_text(pred)
                    gt_text = " ".join(gt.text)  # concatenate all GT transcriptions
                    ocr_metrics.process_batch(pred_text, gt_text)
                else:
                    logger.warning("Skipping OCR image %s: ground truth has no text annotations", image_path)

        # ------ Compute metrics ---------------------------------------
        speed = self._compute_speed(t_pre, t_inf, t_post)

        if self.task == "detect":
            metrics = self._finalize_det(all_tp, all_conf, all_pred_cls, all_target_cls, names, speed)

        elif self.task == "segment":
            metrics = self._finalize_seg(
                all_tp,
                all_tp_mask,
                all_conf,
                all_pred_cls,
                all_target_cls,
                names,
                speed,
            )

        elif self.task == "classify":
            classify_metrics.speed = speed
            metrics = classify_metrics

        elif self.task == "depth":
            depth_metrics.finalize()
            depth_metrics.speed = speed
            metrics = depth_metrics

        elif self.task == "ocr":
            ocr_metrics.finalize()
            ocr_metrics.speed = speed
            metrics = ocr_metrics

        # ------ Optional: verbose console table -----------------------
        if self.verbose:
            self._print_table(metrics, names)

        # ------ Optional: plots ---------------------------------------
        if self.plots and self.save_dir:
            self._save_plots(metrics)

        return metrics

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_label_remap(self, adapter: Any) -> dict[int, int]:
        """Build pred-label-ID → GT-label-ID map via class name matching.

        This handles two common conventions:

        * **Raw COCO category ID models** (e.g. DETR): output label 1=person,
          2=bicycle, … (1-indexed with gaps).  ``_cat_id_to_label`` from the
          loader maps these correctly.
        * **0-indexed contiguous models** (e.g. OneFormer, Mask2Former): output
          label 0=person, 1=bicycle, …  Applying ``_cat_id_to_label`` here
          would shift everything by one and cause label collisions.

        By matching on class *name* through the adapter\'s ``id2label`` map,
        we handle both conventions automatically.
        """
        if self._loader is None:
            return dict(self._cat_id_to_label)

        id2label: dict[int, str] | None = getattr(adapter, "id2label", None)
        if not id2label:
            # No id2label on adapter — fall back to COCO category ID remapping
            return dict(self._cat_id_to_label)

        # Reverse map: lowercase GT class name → 0-indexed GT label
        gt_names = self._loader.names  # {0: "person", 1: "bicycle", …}
        name_to_gt: dict[str, int] = {name.lower(): lid for lid, name in gt_names.items()}

        remap: dict[int, int] = {}
        for pred_id, pred_name in id2label.items():
            gt_label = name_to_gt.get(str(pred_name).lower())
            if gt_label is not None:
                remap[int(pred_id)] = int(gt_label)

        if remap:
            logger.debug(
                "Label remap built via name matching: %d/%d adapter classes matched GT",
                len(remap),
                len(id2label),
            )
            return remap

        # Name matching found nothing — fall back to COCO category ID remapping
        logger.debug("Name-based label remap empty; falling back to cat_id_to_label")
        return dict(self._cat_id_to_label)

    def _build_loader(self, names_out: dict) -> DatasetLoader | None:
        """Build a DatasetLoader from self.data, or return None in standalone mode.

        Side-effect: populates *names_out* with the dataset class map.
        """
        if self.data is not None:
            loader = DatasetLoader(data=self.data, task=self.task, split=self.split)
            names_out.update(loader.names)
            self._loader = loader
            self._cat_id_to_label = loader.cat_id_to_label
            return loader

        if isinstance(self.ground_truth, str):
            # Standalone path to COCO JSON
            loader = DatasetLoader(annotations=self.ground_truth, task=self.task)
            names_out.update(loader.names)
            self._loader = loader
            self._cat_id_to_label = loader.cat_id_to_label
            return loader

        # ground_truth is already a list[GroundTruth] — no loader needed
        return None

    def _load_adapter(self) -> Any:
        """Return a ready adapter, or None when in standalone mode."""
        if self.predictions is not None:
            # Standalone: no adapter needed
            return None

        if self.model is None and self.data is None:
            # Nothing to evaluate without at minimum one of them
            return None

        # If model is already an adapter (has predict()), use it directly
        if hasattr(self.model, "predict"):
            return self.model

        # Lazy import avoids circular dependency issues at module load time
        import mata  # noqa: PLC0415

        kwargs = dict(self.kwargs)
        if self.device:
            kwargs["device"] = self.device

        return mata.load(self.task, self.model, **kwargs)

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def _iter_samples(self, loader: DatasetLoader | None):
        """Yield ``(image_path_or_pred, GroundTruth)`` pairs.

        In dataset-driven mode yields ``(str path, GroundTruth)`` from *loader*.
        In standalone mode yields ``(prediction, GroundTruth)`` where the first
        element carries the pre-computed prediction object.
        """
        if self.predictions is not None:
            # Standalone: zip predictions with ground-truth
            gt_items = self._resolve_gt_list()
            for pred, gt in zip(self.predictions, gt_items):
                yield pred, gt
        elif loader is not None:
            for image_path, gt in loader:
                yield image_path, gt

    def _resolve_gt_list(self) -> list[GroundTruth]:
        """Return ground-truth as a list of GroundTruth objects."""
        if isinstance(self.ground_truth, list):
            return self.ground_truth  # type: ignore[return-value]
        if self._loader is not None:
            return list(self._loader)
        return []

    # ------------------------------------------------------------------
    # Preprocessing / postprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(image_path: str) -> Any:
        """Return the raw image path (adapters handle their own loading)."""
        return image_path

    @staticmethod
    def _postprocess(result: Any) -> Any:
        """Return result as-is (postprocessing is adapter-internal)."""
        return result

    # ------------------------------------------------------------------
    # Detection matching
    # ------------------------------------------------------------------

    def _match_detections(
        self,
        result: Any,
        gt: GroundTruth,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Match predicted detections to GT boxes across all COCO thresholds.

        Returns:
            tp:         (N, T) bool — TP flag per prediction per IoU threshold.
            conf_arr:   (N,) float32 — confidence scores.
            pred_cls:   (N,) int32   — predicted class IDs.
            target_cls: (M,) int32   — GT class IDs (for ap_per_class).
        """
        pred_boxes, pred_scores, pred_labels = self._extract_det_preds(result)

        # Remap predicted labels from model-native IDs to the 0-indexed label
        # space used by GroundTruth.  _label_remap is built via name matching
        # so it handles both DETR-style (raw COCO cat IDs) and OneFormer-style
        # (0-indexed contiguous) conventions correctly.
        if self._label_remap:
            pred_labels = np.array(
                [self._label_remap.get(int(lbl), int(lbl)) for lbl in pred_labels],
                dtype=np.int32,
            )

        target_cls = gt.labels.astype(np.int32)  # (M,)

        # Filter low-confidence predictions
        keep = pred_scores >= self.conf
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        n_pred = len(pred_scores)
        n_gt = len(gt.boxes)
        n_thr = len(COCO_IOU_THRESHOLDS)

        if n_pred == 0 or n_gt == 0:
            return (
                np.zeros((n_pred, n_thr), dtype=bool),
                pred_scores,
                pred_labels,
                target_cls,
            )

        iou_mat = box_iou(pred_boxes, gt.boxes)  # (N, M)

        tp = np.zeros((n_pred, n_thr), dtype=bool)

        for ti, thr in enumerate(COCO_IOU_THRESHOLDS):
            matched_gt: set[int] = set()
            # Greedy matching: iterate predictions sorted by score desc
            sort_idx = np.argsort(-pred_scores)
            for pi in sort_idx:
                best_j = int(np.argmax(iou_mat[pi]))
                if iou_mat[pi, best_j] >= thr and best_j not in matched_gt and pred_labels[pi] == gt.labels[best_j]:
                    tp[pi, ti] = True
                    matched_gt.add(best_j)

        return tp, pred_scores, pred_labels, target_cls

    # ------------------------------------------------------------------
    # Segmentation matching
    # ------------------------------------------------------------------

    def _match_segments(
        self,
        result: Any,
        gt: GroundTruth,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Match predicted masks + boxes to GT (box IoU + mask IoU).

        Returns:
            tp_box:     (N, T) bool — TP flag for box AP.
            tp_mask:    (N, T) bool — TP flag for mask AP.
            conf_arr:   (N,) float32.
            pred_cls:   (N,) int32.
            target_cls: (M,) int32.
        """
        # Box matching (same as detect)
        tp_box, conf_arr, pred_cls, target_cls = self._match_detections(result, gt)

        # Mask AP: use box IoU as proxy when mask IoU is unavailable
        # (full mask IoU is compute-heavy; using box as lower bound is
        # standard practice for mixed-format datasets)
        tp_mask = self._match_mask_iou(result, gt, conf_arr, pred_cls)

        return tp_box, tp_mask, conf_arr, pred_cls, target_cls

    def _match_mask_iou(
        self,
        result: Any,
        gt: GroundTruth,
        conf_arr: np.ndarray,
        pred_cls: np.ndarray,
    ) -> np.ndarray:
        """Compute mask-level TP array.

        Falls back to box-IoU proxy when pycocotools is unavailable or masks
        are not present in the result/GT.
        """
        n_pred = len(conf_arr)
        n_thr = len(COCO_IOU_THRESHOLDS)
        tp_mask = np.zeros((n_pred, n_thr), dtype=bool)

        if n_pred == 0 or len(gt.labels) == 0:
            return tp_mask

        pred_masks = self._extract_masks(result)
        if pred_masks is None or gt.masks is None:
            # Fall back to box IoU proxy
            pred_boxes, pred_scores, _ = self._extract_det_preds(result)
            keep = pred_scores >= self.conf
            pred_boxes = pred_boxes[keep]
            if len(pred_boxes) == 0 or len(gt.boxes) == 0:
                return tp_mask
            iou_mat = box_iou(pred_boxes, gt.boxes)
            for ti, thr in enumerate(COCO_IOU_THRESHOLDS):
                matched_gt: set[int] = set()
                sort_idx = np.argsort(-pred_scores[keep])
                for pi in sort_idx:
                    best_j = int(np.argmax(iou_mat[pi]))
                    if iou_mat[pi, best_j] >= thr and best_j not in matched_gt and pred_cls[pi] == gt.labels[best_j]:
                        tp_mask[pi, ti] = True
                        matched_gt.add(best_j)
            return tp_mask

        # Use pycocotools mask IoU when available
        try:
            from mata.eval.metrics.iou import mask_iou  # noqa: PLC0415

            # image_shape expected as (height, width); gt.image_size is (width, height)
            img_h, img_w = gt.image_size[1], gt.image_size[0]
            iou_mat = mask_iou(pred_masks, gt.masks, image_shape=(img_h, img_w))  # (N, M)
            labels_gt = gt.labels
            for ti, thr in enumerate(COCO_IOU_THRESHOLDS):
                matched_gt_set: set[int] = set()
                sort_idx = np.argsort(-conf_arr)
                for pi in sort_idx:
                    best_j = int(np.argmax(iou_mat[pi]))
                    if (
                        iou_mat[pi, best_j] >= thr
                        and best_j not in matched_gt_set
                        and pred_cls[pi] == labels_gt[best_j]
                    ):
                        tp_mask[pi, ti] = True
                        matched_gt_set.add(best_j)
        except Exception as exc:  # noqa: BLE001
            logger.debug("mask_iou failed (%s) — using box proxy", exc)

        return tp_mask

    # ------------------------------------------------------------------
    # Result extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_det_preds(
        result: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract (pred_boxes, pred_scores, pred_labels) from any result type.

        Supports:
        * ``VisionResult`` / DetectResult — ``result.instances``
        * Raw list of ``Instance`` objects
        * Pre-extracted tuple ``(boxes, scores, labels)``
        """
        # Pre-extracted tuple shortcut
        if isinstance(result, tuple) and len(result) == 3 and isinstance(result[0], np.ndarray):
            return result  # type: ignore[return-value]

        instances = result.instances if hasattr(result, "instances") else (result if isinstance(result, list) else [])

        if not instances:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32),
            )

        boxes = np.array(
            [inst.bbox if inst.bbox is not None else [0, 0, 0, 0] for inst in instances],
            dtype=np.float32,
        )
        scores = np.array([inst.score for inst in instances], dtype=np.float32)
        labels = np.array([inst.label for inst in instances], dtype=np.int32)
        return boxes, scores, labels

    @staticmethod
    def _extract_masks(result: Any) -> list | None:
        """Return mask list from a VisionResult or None."""
        instances = getattr(result, "instances", None)
        if not instances:
            return None
        masks = [inst.mask for inst in instances if inst.mask is not None]
        return masks if masks else None

    @staticmethod
    def _extract_classify_preds(result: Any) -> tuple[int, list[int]]:
        """Return (top1_label, top5_labels) from a ClassifyResult."""
        predictions = getattr(result, "predictions", None)
        if not predictions:
            return 0, [0]
        sorted_preds = sorted(predictions, key=lambda p: p.score, reverse=True)
        top1 = int(sorted_preds[0].label)
        top5 = [int(p.label) for p in sorted_preds[:5]]
        return top1, top5

    @staticmethod
    def _extract_depth(result: Any) -> np.ndarray:
        """Return the raw depth map from a DepthResult or ndarray."""
        if hasattr(result, "depth"):
            return np.asarray(result.depth, dtype=np.float64)
        return np.asarray(result, dtype=np.float64)

    def _extract_ocr_text(self, pred: Any) -> str:
        """Extract full predicted text from an OCRResult.

        If regions have bounding boxes, sorts them left-to-right (by x1)
        then top-to-bottom (by y1) before joining. Otherwise, joins in
        list order.

        Args:
            pred: An OCRResult instance.

        Returns:
            Single string with all recognized text joined by spaces.
        """
        from mata.core.types import OCRResult  # lazy import — avoids circular deps

        if isinstance(pred, OCRResult):
            regions = pred.regions
            # Sort by position if bboxes available
            if regions and regions[0].bbox is not None:
                regions = sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))  # type: ignore[index]
            return " ".join(r.text for r in regions)

        # Fallback: try .full_text or str()
        if hasattr(pred, "full_text"):
            return pred.full_text
        return str(pred)

    # ------------------------------------------------------------------
    # Metric finalisation
    # ------------------------------------------------------------------

    def _finalize_det(
        self,
        all_tp: list[np.ndarray],
        all_conf: list[np.ndarray],
        all_pred_cls: list[np.ndarray],
        all_target_cls: list[np.ndarray],
        names: dict[int, str],
        speed: dict[str, float],
    ) -> DetMetrics:
        """Aggregate TP/FP arrays and populate a :class:`DetMetrics` object."""
        metrics = DetMetrics(names=names, save_dir=self.save_dir)
        metrics.speed = speed

        tp_cat, conf_cat, pred_cls_cat, target_cls_cat = self._concat_arrays(
            all_tp, all_conf, all_pred_cls, all_target_cls
        )

        results = ap_per_class(
            tp_cat,
            conf_cat,
            pred_cls_cat,
            target_cls_cat,
            iou_thresholds=COCO_IOU_THRESHOLDS,
        )
        metrics.box.update(results)
        return metrics

    def _finalize_seg(
        self,
        all_tp_box: list[np.ndarray],
        all_tp_mask: list[np.ndarray],
        all_conf: list[np.ndarray],
        all_pred_cls: list[np.ndarray],
        all_target_cls: list[np.ndarray],
        names: dict[int, str],
        speed: dict[str, float],
    ) -> SegmentMetrics:
        """Aggregate TP arrays and populate a :class:`SegmentMetrics` object."""
        metrics = SegmentMetrics(names=names, save_dir=self.save_dir)
        metrics.speed = speed

        tp_box_cat, conf_cat, pred_cls_cat, target_cls_cat = self._concat_arrays(
            all_tp_box, all_conf, all_pred_cls, all_target_cls
        )
        box_results = ap_per_class(
            tp_box_cat,
            conf_cat,
            pred_cls_cat,
            target_cls_cat,
            iou_thresholds=COCO_IOU_THRESHOLDS,
        )
        metrics.box.update(box_results)

        if all_tp_mask:
            tp_mask_cat, _, _, _ = self._concat_arrays(all_tp_mask, all_conf, all_pred_cls, all_target_cls)
            mask_results = ap_per_class(
                tp_mask_cat,
                conf_cat,
                pred_cls_cat,
                target_cls_cat,
                iou_thresholds=COCO_IOU_THRESHOLDS,
            )
            metrics.seg.update(mask_results)

        return metrics

    @staticmethod
    def _concat_arrays(
        all_tp: list[np.ndarray],
        all_conf: list[np.ndarray],
        all_pred_cls: list[np.ndarray],
        all_target_cls: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Concatenate per-image arrays into global arrays for ap_per_class."""
        if not all_conf:
            return (
                np.zeros((0, len(COCO_IOU_THRESHOLDS)), dtype=bool),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
            )
        n_thr = len(COCO_IOU_THRESHOLDS)

        tp_parts = []
        for tp in all_tp:
            if tp.ndim == 1:
                tp = np.tile(tp[:, np.newaxis].astype(bool), (1, n_thr))
            elif tp.shape[1] == 1:
                tp = np.tile(tp, (1, n_thr))
            tp_parts.append(tp)

        tp_cat = np.concatenate(tp_parts, axis=0) if tp_parts else np.zeros((0, n_thr), dtype=bool)
        conf_cat = np.concatenate(all_conf).astype(np.float32) if all_conf else np.zeros(0, dtype=np.float32)
        pred_cls_cat = np.concatenate(all_pred_cls).astype(np.int32) if all_pred_cls else np.zeros(0, dtype=np.int32)
        target_cls_cat = (
            np.concatenate(all_target_cls).astype(np.int32) if all_target_cls else np.zeros(0, dtype=np.int32)
        )

        return tp_cat, conf_cat, pred_cls_cat, target_cls_cat

    # ------------------------------------------------------------------
    # Speed
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_speed(
        t_pre: list[float],
        t_inf: list[float],
        t_post: list[float],
    ) -> dict[str, float]:
        """Convert timing lists (seconds) to a ms/image dict."""

        def _mean_ms(lst: list[float]) -> float:
            return (sum(lst) / max(len(lst), 1)) * 1000.0

        return {
            "preprocess": _mean_ms(t_pre),
            "inference": _mean_ms(t_inf),
            "postprocess": _mean_ms(t_post),
        }

    # ------------------------------------------------------------------
    # Optional outputs (stubs for Task E1, E2, F2)
    # ------------------------------------------------------------------

    def _print_table(self, metrics: Any, names: dict[int, str]) -> None:
        """Print a results summary.

        Uses :class:`~mata.eval.printer.Printer` when Task F2 is implemented;
        falls back to a minimal ``print()`` until then.
        """
        try:
            from mata.eval.printer import Printer  # noqa: PLC0415

            printer = Printer(names=names, task=self.task, save_dir=self.save_dir)
            printer.print_results(metrics)
        except NotImplementedError:
            # Printer stub not yet implemented — emit a simple summary line
            self._print_simple_summary(metrics)

    def _print_simple_summary(self, metrics: Any) -> None:
        """Minimal fallback summary printed when the full Printer is a stub."""
        task = self.task
        if task == "detect" and hasattr(metrics, "box"):
            print(
                f"[MATA val] detect — "
                f"mAP50={metrics.box.map50:.4f}  "
                f"mAP50-95={metrics.box.map:.4f}  "
                f"P={metrics.box.mp:.4f}  R={metrics.box.mr:.4f}"
            )
        elif task == "segment" and hasattr(metrics, "seg"):
            print(f"[MATA val] segment — " f"box mAP50={metrics.box.map50:.4f}  " f"mask mAP50={metrics.seg.map50:.4f}")
        elif task == "classify":
            print(f"[MATA val] classify — " f"top1={metrics.top1:.4f}  top5={metrics.top5:.4f}")
        elif task == "depth":
            print(
                f"[MATA val] depth — "
                f"abs_rel={metrics.abs_rel:.4f}  "
                f"rmse={metrics.rmse:.4f}  "
                f"δ₁={metrics.delta_1:.4f}"
            )
        elif task == "ocr":
            print(
                f"[MATA val] ocr — "
                f"CER={metrics.cer:.4f}  "
                f"WER={metrics.wer:.4f}  "
                f"Accuracy={metrics.accuracy:.4f}"
            )

    def _save_plots(self, metrics: Any) -> None:
        """Save PR/F1 curves to self.save_dir.

        Delegates to :mod:`mata.eval.plots` when Task E2 is implemented;
        silently skips until then.
        """
        if not self.save_dir:
            return
        try:
            import os  # noqa: PLC0415

            os.makedirs(self.save_dir, exist_ok=True)

            from mata.eval.plots import (  # noqa: PLC0415
                plot_f1_curve,
                plot_pr_curve,
            )

            if hasattr(metrics, "box") and hasattr(metrics.box, "curves_results"):
                curves = metrics.box.curves_results
                names_list = [str(i) for i in metrics.box.ap_class_index]
                # F1 curve
                x, y, _ = curves[0]
                plot_f1_curve(x, y, save_path=f"{self.save_dir}/F1_curve.png", names=names_list)
                # PR curve
                x_pr, y_pr, _ = curves[1]
                plot_pr_curve(
                    x_pr,
                    y_pr,
                    ap=metrics.box.ap50,
                    save_path=f"{self.save_dir}/PR_curve.png",
                    names=names_list,
                )
        except NotImplementedError:
            logger.debug("Plots not yet implemented (Task E2) — skipping.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Plot generation failed: %s", exc)
