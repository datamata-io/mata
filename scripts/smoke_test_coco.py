"""COCO val2017 stability smoke test for MATA.

Loops over COCO val2017 images, runs all configured tasks (detect, classify,
segment, depth) for each image, logs per-image results alongside ground-truth
annotations, and prints a summary table at the end.

Purpose: **stability** (no crashes, graceful error handling) — not accuracy
evaluation.  Each task is judged pass/fail purely on whether it runs without
exception; label-set overlap with GT annotations is logged as an informational
signal only.

Usage:
    python examples/smoke_test_coco.py \\
        --images-dir /data/coco/val2017 \\
        --annotations /data/coco/annotations/instances_val2017.json \\
        --limit 50 \\
        --tasks detect,classify,segment,depth \\
        --threshold 0.5 \\
        --log-dir ./smoke_logs

Download COCO val2017:
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip val2017.zip && unzip annotations_trainval2017.zip
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"smoke_{timestamp}.log"

    root = logging.getLogger("smoke")
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(ch)
    root.info("Log file: %s", log_file)
    return root


# ---------------------------------------------------------------------------
# Default model configuration
# ---------------------------------------------------------------------------

DEFAULT_MODELS: dict[str, str] = {
    "detect":           "facebook/detr-resnet-101",
    "classify":         "microsoft/resnet-101",
    "segment":          "shi-labs/oneformer_coco_swin_large",
    "depth":            "depth-anything/Depth-Anything-V2-Small-hf",
    # Zero-shot variants (use GT category names as text prompts)
    "detect_zeroshot":  "IDEA-Research/grounding-dino-tiny",
    "classify_zeroshot":"openai/clip-vit-base-patch32",
    "segment_zeroshot": "CIDAS/clipseg-rd64-refined",
    "segment_sam3":     "facebook/sam3",
}

SUPPORTED_TASKS = list(DEFAULT_MODELS.keys())

# Maps smoke-test task name → actual mata task string used with mata.load()
TASK_TO_MATA_TASK: dict[str, str] = {
    "detect":           "detect",
    "classify":         "classify",
    "segment":          "segment",
    "depth":            "depth",
    "detect_zeroshot":  "detect",
    "classify_zeroshot":"classify",
    "segment_zeroshot": "segment",
    "segment_sam3":     "segment",
}


# ---------------------------------------------------------------------------
# Per-task statistics accumulator
# ---------------------------------------------------------------------------

@dataclass
class TaskStats:
    task: str
    total: int = 0
    successes: int = 0
    errors: int = 0
    total_time: float = 0.0
    total_pred_count: int = 0
    total_overlap_ratio: float = 0.0
    failed_images: list[dict[str, Any]] = field(default_factory=list)

    @property
    def avg_time(self) -> float:
        return self.total_time / self.successes if self.successes else 0.0

    @property
    def avg_pred_count(self) -> float:
        return self.total_pred_count / self.successes if self.successes else 0.0

    @property
    def avg_overlap_ratio(self) -> float:
        return self.total_overlap_ratio / self.successes if self.successes else 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# Environment information
# ---------------------------------------------------------------------------

_KEY_PACKAGES = [
    "torch",
    "torchvision",
    "transformers",
    "onnxruntime",
    "onnxruntime_gpu",
    "PIL",
    "Pillow",
    "numpy",
    "opencv-python",
    "cv2",
    "pycocotools",
    "mata",
]


def _log_environment(logger: logging.Logger) -> None:
    """Log Python version, OS, hardware, and key library versions."""
    import importlib
    import importlib.metadata
    import os
    import platform
    import sys

    sep = "-" * 60
    logger.info(sep)
    logger.info("ENVIRONMENT")
    logger.info(sep)

    # Python
    logger.info("Python       : %s", sys.version.replace("\n", " "))
    logger.info("Executable   : %s", sys.executable)

    # OS / platform
    logger.info("OS           : %s", platform.platform())
    logger.info("Machine      : %s  %s", platform.machine(), platform.processor())

    # CPU count
    cpu_count = os.cpu_count()
    logger.info("CPU cores    : %s", cpu_count)

    # PyTorch + CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        device_names: list[str] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        logger.info("PyTorch      : %s  (CUDA available: %s, version: %s)", torch.__version__, cuda_available, cuda_version)
        if device_names:
            for i, name in enumerate(device_names):
                mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                logger.info("  GPU[%d]     : %s  (%.1f GB)", i, name, mem)
        else:
            logger.info("  GPU        : none")
    except ImportError:
        logger.info("PyTorch      : not installed")

    # Key packages
    checked: set[str] = set()
    for pkg in _KEY_PACKAGES:
        if pkg in ("PIL", "cv2"):  # import name differs from dist name
            import_name = pkg
            try:
                mod = importlib.import_module(import_name)
                version = getattr(mod, "__version__", "unknown")
                logger.info("%-13s: %s", pkg, version)
            except ImportError:
                pass  # silently skip aliases that failed
            continue
        if pkg in checked:
            continue
        checked.add(pkg)
        if pkg == "torch":  # already printed above
            continue
        try:
            version = importlib.metadata.version(pkg)
            logger.info("%-13s: %s", pkg, version)
        except importlib.metadata.PackageNotFoundError:
            pass  # skip packages that are not installed

    logger.info(sep)


# ---------------------------------------------------------------------------
# Model loading (once per task, before the main loop)
# ---------------------------------------------------------------------------

def _load_models(
    tasks: list[str],
    models: dict[str, str],
    threshold: float,
    skip_load_errors: bool,
    logger: logging.Logger,
    hf_token: Optional[str] = None,
) -> dict[str, Any]:
    """Load all task adapters.  Returns {task: adapter} for successful loads."""
    import mata
    from mata.core.logging import suppress_third_party_logs

    adapters: dict[str, Any] = {}
    for task in tasks:
        model_id = models[task]
        mata_task = TASK_TO_MATA_TASK.get(task, task)
        logger.info("Loading model for task=%s  mata_task=%s  model=%s", task, mata_task, model_id)
        t0 = time.perf_counter()
        try:
            with suppress_third_party_logs():
                kwargs: dict[str, Any] = {}
                if mata_task in ("detect", "segment") and task not in ("detect_zeroshot", "classify_zeroshot", "segment_zeroshot", "segment_sam3"):
                    kwargs["threshold"] = threshold
                if task == "segment_sam3" and hf_token:
                    kwargs["token"] = hf_token
                adapter = mata.load(mata_task, model_id, **kwargs)
            elapsed = time.perf_counter() - t0
            logger.info(
                "Loaded task=%s in %.2fs  adapter=%s",
                task, elapsed, type(adapter).__name__,
            )
            adapters[task] = adapter
        except Exception:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            msg = traceback.format_exc()
            if skip_load_errors:
                logger.error(
                    "Failed to load task=%s model=%s after %.2fs — skipping task.\n%s",
                    task, model_id, elapsed, msg,
                )
            else:
                logger.critical(
                    "Failed to load task=%s model=%s after %.2fs.\n%s",
                    task, model_id, elapsed, msg,
                )
                raise
    return adapters


# ---------------------------------------------------------------------------
# Per-image, per-task inference + comparison
# ---------------------------------------------------------------------------

def _run_detect(
    adapter: Any,
    image_path: Path,
    gt: dict[str, Any],
    threshold: float,
    logger: logging.Logger,
) -> tuple[int, float, float]:
    """Run detection inference and compare with GT.

    Returns: (pred_count, overlap_ratio, elapsed_seconds)
    """
    from coco_utils import compute_label_overlap

    t0 = time.perf_counter()
    result = adapter.predict(str(image_path), threshold=threshold)
    elapsed = time.perf_counter() - t0

    detections = result.detections
    pred_names = {
        d.label_name for d in detections if d.label_name
    }
    gt_names: set[str] = gt["category_names"]

    overlap = compute_label_overlap(pred_names, gt_names)
    pred_count = len(detections)

    logger.info(
        "detect  image_id=%-8d pred=%3d gt=%3d overlap=%.2f  matched=%s  missed=%s  extra=%s  %.3fs",
        gt["image_id"], pred_count, gt["instance_count"],
        overlap["overlap_ratio"],
        sorted(overlap["matched"])[:5],
        sorted(overlap["missed"])[:5],
        sorted(overlap["extra"])[:5],
        elapsed,
    )
    if pred_count == 0 and gt["instance_count"] > 0:
        logger.warning(
            "detect  image_id=%d: zero predictions but GT has %d instances",
            gt["image_id"], gt["instance_count"],
        )

    return pred_count, overlap["overlap_ratio"], elapsed


def _run_classify(
    adapter: Any,
    image_path: Path,
    gt: dict[str, Any],
    logger: logging.Logger,
) -> tuple[int, float, float]:
    """Run classification inference and compare with GT.

    Returns: (pred_count, overlap_ratio, elapsed_seconds)
    """
    from coco_utils import compute_label_overlap

    t0 = time.perf_counter()
    result = adapter.predict(str(image_path))
    elapsed = time.perf_counter() - t0

    preds = result.predictions
    pred_names = {p.label_name for p in preds if p.label_name}
    gt_names: set[str] = gt["category_names"]

    overlap = compute_label_overlap(pred_names, gt_names)
    pred_count = len(preds)

    top1_name = preds[0].label_name if preds else "—"
    logger.info(
        "classify image_id=%-8d top1=%-20s gt_labels=%s  overlap=%.2f  %.3fs",
        gt["image_id"], top1_name,
        sorted(gt_names)[:5],
        overlap["overlap_ratio"],
        elapsed,
    )
    return pred_count, overlap["overlap_ratio"], elapsed


def _run_segment(
    adapter: Any,
    image_path: Path,
    gt: dict[str, Any],
    threshold: float,
    logger: logging.Logger,
) -> tuple[int, float, float]:
    """Run segmentation inference and compare with GT.

    Returns: (pred_count, overlap_ratio, elapsed_seconds)
    """
    from coco_utils import compute_label_overlap

    t0 = time.perf_counter()
    result = adapter.predict(str(image_path), threshold=threshold)
    elapsed = time.perf_counter() - t0

    instances = result.instances
    pred_names = {i.label_name for i in instances if i.label_name}
    gt_names: set[str] = gt["category_names"]

    overlap = compute_label_overlap(pred_names, gt_names)
    pred_count = len(instances)

    logger.info(
        "segment image_id=%-8d pred=%3d gt=%3d overlap=%.2f  matched=%s  %.3fs",
        gt["image_id"], pred_count, gt["instance_count"],
        overlap["overlap_ratio"],
        sorted(overlap["matched"])[:5],
        elapsed,
    )
    return pred_count, overlap["overlap_ratio"], elapsed


def _run_depth(
    adapter: Any,
    image_path: Path,
    gt: dict[str, Any],
    logger: logging.Logger,
) -> tuple[int, float, float]:
    """Run depth estimation.  GT comparison: verify output shape matches image.

    Returns: (1 if shape ok else 0, 1.0, elapsed_seconds)
    """
    t0 = time.perf_counter()
    result = adapter.predict(str(image_path), normalize=True)
    elapsed = time.perf_counter() - t0

    depth = result.depth
    expected_h, expected_w = gt["height"], gt["width"]
    shape_ok = depth.ndim == 2 and depth.shape[0] > 0 and depth.shape[1] > 0

    logger.info(
        "depth   image_id=%-8d shape=%s expected_hw=(%d,%d)  shape_ok=%s  %.3fs",
        gt["image_id"], depth.shape, expected_h, expected_w, shape_ok, elapsed,
    )
    if not shape_ok:
        logger.warning(
            "depth   image_id=%d: unexpected depth shape %s",
            gt["image_id"], depth.shape,
        )

    return 1, 1.0, elapsed


def _run_detect_zeroshot(
    adapter: Any,
    image_path: Path,
    gt: dict[str, Any],
    threshold: float,
    logger: logging.Logger,
) -> tuple[int, float, float]:
    """Zero-shot detection (GroundingDINO) using GT category names as text prompts.

    Returns: (pred_count, overlap_ratio, elapsed_seconds)
    """
    from coco_utils import compute_label_overlap

    gt_names: set[str] = gt["category_names"]
    if not gt_names:
        logger.warning("detect_zeroshot image_id=%d: no GT categories — skipping", gt["image_id"])
        return 0, 0.0, 0.0

    # GroundingDINO expects " . "-separated string
    text_prompts = " . ".join(sorted(gt_names))

    t0 = time.perf_counter()
    result = adapter.predict(str(image_path), text_prompts=text_prompts, threshold=threshold)
    elapsed = time.perf_counter() - t0

    detections = result.detections
    pred_names = {d.label_name for d in detections if d.label_name}
    overlap = compute_label_overlap(pred_names, gt_names)
    pred_count = len(detections)

    logger.info(
        "detect_zs image_id=%-8d pred=%3d gt=%3d overlap=%.2f  prompts=%s  matched=%s  %.3fs",
        gt["image_id"], pred_count, gt["instance_count"],
        overlap["overlap_ratio"],
        sorted(gt_names)[:5],
        sorted(overlap["matched"])[:5],
        elapsed,
    )
    return pred_count, overlap["overlap_ratio"], elapsed


def _run_classify_zeroshot(
    adapter: Any,
    image_path: Path,
    gt: dict[str, Any],
    logger: logging.Logger,
) -> tuple[int, float, float]:
    """Zero-shot classification (CLIP) using GT category names as text prompts.

    Returns: (pred_count, overlap_ratio, elapsed_seconds)
    """
    from coco_utils import compute_label_overlap

    gt_names: set[str] = gt["category_names"]
    if not gt_names:
        logger.warning("classify_zeroshot image_id=%d: no GT categories — skipping", gt["image_id"])
        return 0, 0.0, 0.0

    text_prompts = sorted(gt_names)

    t0 = time.perf_counter()
    result = adapter.predict(str(image_path), text_prompts=text_prompts)
    elapsed = time.perf_counter() - t0

    preds = result.predictions
    pred_names = {p.label_name for p in preds if p.label_name}
    overlap = compute_label_overlap(pred_names, gt_names)
    pred_count = len(preds)

    top1_name = preds[0].label_name if preds else "—"
    logger.info(
        "classify_zs image_id=%-8d top1=%-20s gt_labels=%s  overlap=%.2f  %.3fs",
        gt["image_id"], top1_name,
        sorted(gt_names)[:5],
        overlap["overlap_ratio"],
        elapsed,
    )
    return pred_count, overlap["overlap_ratio"], elapsed


def _run_segment_zeroshot(
    adapter: Any,
    image_path: Path,
    gt: dict[str, Any],
    threshold: float,
    logger: logging.Logger,
) -> tuple[int, float, float]:
    """Zero-shot segmentation (CLIPSeg) using GT category names as text prompts.

    Returns: (pred_count, overlap_ratio, elapsed_seconds)
    """
    from coco_utils import compute_label_overlap

    gt_names: set[str] = gt["category_names"]
    if not gt_names:
        logger.warning("segment_zeroshot image_id=%d: no GT categories — skipping", gt["image_id"])
        return 0, 0.0, 0.0

    text_prompts = sorted(gt_names)

    t0 = time.perf_counter()
    result = adapter.predict(str(image_path), text_prompts=text_prompts, threshold=threshold)
    elapsed = time.perf_counter() - t0

    instances = result.instances
    pred_names = {i.label_name for i in instances if i.label_name}
    overlap = compute_label_overlap(pred_names, gt_names)
    pred_count = len(instances)

    logger.info(
        "segment_zs image_id=%-8d pred=%3d gt=%3d overlap=%.2f  prompts=%s  matched=%s  %.3fs",
        gt["image_id"], pred_count, gt["instance_count"],
        overlap["overlap_ratio"],
        sorted(gt_names)[:5],
        sorted(overlap["matched"])[:5],
        elapsed,
    )
    return pred_count, overlap["overlap_ratio"], elapsed


def _run_segment_sam3(
    adapter: Any,
    image_path: Path,
    gt: dict[str, Any],
    threshold: float,
    logger: logging.Logger,
) -> tuple[int, float, float]:
    """SAM3 zero-shot segmentation using GT category names as text prompts.

    Returns: (pred_count, overlap_ratio, elapsed_seconds)
    """
    from coco_utils import compute_label_overlap

    gt_names: set[str] = gt["category_names"]
    if not gt_names:
        logger.warning("segment_sam3 image_id=%d: no GT categories — skipping", gt["image_id"])
        return 0, 0.0, 0.0

    text_prompts = sorted(gt_names)

    t0 = time.perf_counter()
    result = adapter.predict(str(image_path), text_prompts=text_prompts, threshold=threshold)
    elapsed = time.perf_counter() - t0

    instances = result.instances
    pred_names = {i.label_name for i in instances if i.label_name}
    overlap = compute_label_overlap(pred_names, gt_names)
    pred_count = len(instances)

    logger.info(
        "sam3      image_id=%-8d pred=%3d gt=%3d overlap=%.2f  prompts=%s  matched=%s  %.3fs",
        gt["image_id"], pred_count, gt["instance_count"],
        overlap["overlap_ratio"],
        sorted(gt_names)[:5],
        sorted(overlap["matched"])[:5],
        elapsed,
    )
    return pred_count, overlap["overlap_ratio"], elapsed


# ---------------------------------------------------------------------------
# Image processing dispatcher
# ---------------------------------------------------------------------------

_TASK_RUNNERS = {
    "detect":           _run_detect,
    "classify":         _run_classify,
    "segment":          _run_segment,
    "depth":            _run_depth,
    "detect_zeroshot":  _run_detect_zeroshot,
    "classify_zeroshot":_run_classify_zeroshot,
    "segment_zeroshot": _run_segment_zeroshot,
    "segment_sam3":     _run_segment_sam3,
}


def _process_image(
    image_idx: int,
    total_images: Optional[int],
    gt: dict[str, Any],
    adapters: dict[str, Any],
    threshold: float,
    stats: dict[str, TaskStats],
    logger: logging.Logger,
) -> None:
    progress = f"[{image_idx}/{total_images or '?'}]" if total_images else f"[{image_idx}]"
    logger.info("%s image_id=%d  file=%s", progress, gt["image_id"], gt["image_path"].name)

    for task, adapter in adapters.items():
        s = stats[task]
        s.total += 1
        try:
            if task == "detect":
                pred_count, overlap, elapsed = _run_detect(adapter, gt["image_path"], gt, threshold, logger)
            elif task == "classify":
                pred_count, overlap, elapsed = _run_classify(adapter, gt["image_path"], gt, logger)
            elif task == "segment":
                pred_count, overlap, elapsed = _run_segment(adapter, gt["image_path"], gt, threshold, logger)
            elif task == "depth":
                pred_count, overlap, elapsed = _run_depth(adapter, gt["image_path"], gt, logger)
            elif task == "detect_zeroshot":
                pred_count, overlap, elapsed = _run_detect_zeroshot(adapter, gt["image_path"], gt, threshold, logger)
            elif task == "classify_zeroshot":
                pred_count, overlap, elapsed = _run_classify_zeroshot(adapter, gt["image_path"], gt, logger)
            elif task == "segment_zeroshot":
                pred_count, overlap, elapsed = _run_segment_zeroshot(adapter, gt["image_path"], gt, threshold, logger)
            elif task == "segment_sam3":
                pred_count, overlap, elapsed = _run_segment_sam3(adapter, gt["image_path"], gt, threshold, logger)
            else:
                raise ValueError(f"Unknown task: {task}")

            s.successes += 1
            s.total_time += elapsed
            s.total_pred_count += pred_count
            s.total_overlap_ratio += overlap

        except Exception:  # noqa: BLE001
            s.errors += 1
            tb = traceback.format_exc()
            logger.error(
                "ERROR task=%s image_id=%d file=%s\n%s",
                task, gt["image_id"], gt["image_path"].name, tb,
            )
            s.failed_images.append({
                "image_id": gt["image_id"],
                "file": gt["image_path"].name,
                "error": tb.strip().splitlines()[-1],
            })


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

_COL = {
    "task":      14,
    "total":      7,
    "ok":         7,
    "err":        7,
    "err_rate":  10,
    "avg_time":  10,
    "avg_preds": 10,
    "avg_ovlp":  10,
}
_HDR = (
    f"{'Task':<{_COL['task']}} {'Total':>{_COL['total']}} {'OK':>{_COL['ok']}} "
    f"{'Err':>{_COL['err']}} {'Err%':>{_COL['err_rate']}} "
    f"{'AvgTime':>{_COL['avg_time']}} {'AvgPreds':>{_COL['avg_preds']}} "
    f"{'AvgOverlap':>{_COL['avg_ovlp']}}"
)
_SEP = "-" * len(_HDR)


def _print_summary(
    stats: dict[str, TaskStats],
    wall_time: float,
    pass_threshold: float,
    logger: logging.Logger,
) -> bool:
    logger.info("")
    logger.info("=" * len(_HDR))
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * len(_HDR))
    logger.info(_HDR)
    logger.info(_SEP)

    all_pass = True
    for task, s in stats.items():
        row = (
            f"{task:<{_COL['task']}} "
            f"{s.total:>{_COL['total']}} "
            f"{s.successes:>{_COL['ok']}} "
            f"{s.errors:>{_COL['err']}} "
            f"{s.error_rate:>{_COL['err_rate']}.1%} "
            f"{s.avg_time:>{_COL['avg_time']}.3f}s "
            f"{s.avg_pred_count:>{_COL['avg_preds']}.1f} "
            f"{s.avg_overlap_ratio:>{_COL['avg_ovlp']}.2f}"
        )
        logger.info(row)
        if s.error_rate > pass_threshold:
            all_pass = False

    logger.info(_SEP)
    logger.info("Wall time: %.1fs", wall_time)
    logger.info("")

    # Failed image details
    for task, s in stats.items():
        if s.failed_images:
            logger.info("Failed images for task=%s:", task)
            for fi in s.failed_images[:20]:  # cap to 20 for readability
                logger.info("  image_id=%-8d file=%-30s %s", fi["image_id"], fi["file"], fi["error"])
            if len(s.failed_images) > 20:
                logger.info("  ... and %d more", len(s.failed_images) - 20)
            logger.info("")

    verdict = "PASS" if all_pass else "FAIL"
    logger.info(
        "Overall: %s  (pass threshold: error_rate < %.0f%% per task)",
        verdict, pass_threshold * 100,
    )
    return all_pass


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MATA COCO val2017 stability smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--images-dir", required=True, type=Path,
        help="Path to the COCO val2017 images directory (contains *.jpg files)",
    )
    p.add_argument(
        "--annotations", required=True, type=Path,
        help="Path to instances_val2017.json",
    )
    p.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process at most N images (default: all ~5000)",
    )
    p.add_argument(
        "--tasks", default=",".join(SUPPORTED_TASKS),
        help=f"Comma-separated tasks to run (default: {','.join(SUPPORTED_TASKS)})",
    )
    p.add_argument(
        "--models", type=json.loads, default=None, metavar="JSON",
        help=(
            'JSON object overriding model IDs per task, e.g. '
            '\'{"detect": "facebook/detr-resnet-50"}\''
        ),
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Detection/segmentation confidence threshold (default: 0.5)",
    )
    p.add_argument(
        "--log-dir", type=Path, default=Path("./smoke_logs"),
        help="Directory for log files (default: ./smoke_logs)",
    )
    p.add_argument(
        "--shuffle", action="store_true",
        help="Randomise image iteration order",
    )
    p.add_argument(
        "--skip-load-errors", action="store_true",
        help="Skip tasks whose model fails to load instead of aborting",
    )
    p.add_argument(
        "--pass-threshold", type=float, default=0.05, metavar="RATE",
        help="Max allowed error rate per task to count as PASS (default: 0.05 = 5%%)",
    )
    p.add_argument(
        "--hf-token", default=None, metavar="TOKEN",
        help="HuggingFace token for gated models (e.g. facebook/sam3). Can also be set via HF_TOKEN env var.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import os
    args = _parse_args(argv)

    logger = _setup_logging(args.log_dir)
    _log_environment(logger)

    # Resolve HF token (CLI arg takes priority over env var)
    hf_token: Optional[str] = args.hf_token or os.getenv("HF_TOKEN") or None

    # Resolve tasks
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unsupported = [t for t in tasks if t not in SUPPORTED_TASKS]
    if unsupported:
        logger.error("Unsupported tasks: %s.  Supported: %s", unsupported, SUPPORTED_TASKS)
        return 1

    # Resolve models
    models: dict[str, str] = dict(DEFAULT_MODELS)
    if args.models:
        models.update(args.models)

    logger.info("Tasks: %s", tasks)
    logger.info("Models: %s", {t: models[t] for t in tasks})
    logger.info("Threshold: %.2f", args.threshold)
    logger.info("Image limit: %s", args.limit or "all")

    # Add examples/ to sys.path so coco_utils can be imported
    examples_dir = Path(__file__).parent
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))

    # Load COCO annotations
    from coco_utils import load_coco, iter_images
    try:
        coco = load_coco(args.annotations)
    except (FileNotFoundError, ImportError) as exc:
        logger.critical("Cannot load annotations: %s", exc)
        return 1

    # Load models
    adapters = _load_models(tasks, models, args.threshold, args.skip_load_errors, logger, hf_token=hf_token)
    if not adapters:
        logger.critical("No adapters loaded — aborting.")
        return 1

    # Initialise stats
    stats: dict[str, TaskStats] = {task: TaskStats(task) for task in adapters}

    # Determine total for progress display
    total_available = len(coco.imgs)
    total_to_process = min(args.limit, total_available) if args.limit else total_available

    logger.info("Starting smoke test: %d images × %d tasks", total_to_process, len(adapters))
    wall_start = time.perf_counter()

    image_idx = 0
    for gt in iter_images(coco, args.images_dir, limit=args.limit, shuffle=args.shuffle):
        image_idx += 1
        _process_image(
            image_idx=image_idx,
            total_images=total_to_process,
            gt=gt,
            adapters=adapters,
            threshold=args.threshold,
            stats=stats,
            logger=logger,
        )

    wall_time = time.perf_counter() - wall_start
    all_pass = _print_summary(stats, wall_time, args.pass_threshold, logger)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())


# python examples/smoke_test_coco.py --images-dir /home/mtp/code/MATA/data/coco/val2017 --annotations /home/mtp/code/MATA/data/coco/annotations_trainval2017/annotations/instances_val2017.json --limit 50 --tasks detect,depth --threshold 0.5 --log-dir ./smoke_logs

# ALL TASKS:
# python examples/smoke_test_coco.py --images-dir /home/mtp/code/MATA/data/coco/val2017 --annotations /home/mtp/code/MATA/data/coco/annotations_trainval2017/annotations/instances_val2017.json --limit 150 --threshold 0.5 --log-dir ./smoke_logs

# ZERO-SHOT TASKS ONLY:
# python examples/smoke_test_coco.py --images-dir /home/mtp/code/MATA/data/coco/val2017 --annotations /home/mtp/code/MATA/data/coco/annotations_trainval2017/annotations/instances_val2017.json --limit 50 --tasks detect_zeroshot,classify_zeroshot,segment_zeroshot --threshold 0.5 --log-dir ./smoke_logs

# WITH SAM3 (requires HF token for gated model):
# python examples/smoke_test_coco.py --images-dir /home/mtp/code/MATA/data/coco/val2017 --annotations /home/mtp/code/MATA/data/coco/annotations_trainval2017/annotations/instances_val2017.json --limit 20 --tasks segment_sam3 --threshold 0.5 --hf-token $HF_TOKEN --log-dir ./smoke_logs

# ALL TASKS INCLUDING ZERO-SHOT:
# You will need at least 8GB GPU memory to run all tasks including the zero-shot variants, which load multiple models.  Adjust --limit and/or --tasks as needed to fit your hardware.
# python examples/smoke_test_coco.py --images-dir /home/mtp/code/MATA/data/coco/val2017 --annotations /home/mtp/code/MATA/data/coco/annotations_trainval2017/annotations/instances_val2017.json --limit 50 --tasks detect,classify,segment,depth,detect_zeroshot,classify_zeroshot,segment_zeroshot --threshold 0.5 --log-dir ./smoke_logs