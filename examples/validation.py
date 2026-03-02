"""MATA Validation Examples — Task H1.

End-to-end examples for all four supported tasks:
  1. Detection      (COCO)
  2. Segmentation   (COCO)
  3. Classification (ImageNet)
  4. Depth          (DIODE / NYU)
  5. Standalone     (pre-run predictions → metrics without re-running inference)

Dataset path setup
------------------
Before running, create the relevant YAML configs under examples/configs/ or
point the `data=` argument at your own YAML file.  Each YAML must define at
minimum::

    path: /absolute/or/relative/path/to/dataset
    val:  images/val2017          # sub-directory that holds images
    anno: annotations/instances_val2017.json   # COCO-format annotations

Set the environment variables (or edit the constants below) to point at your
local dataset directories:

    COCO_YAML=examples/configs/coco.yaml
    IMAGENET_YAML=examples/configs/imagenet.yaml
    DIODE_YAML=examples/configs/diode.yaml

Running
-------
    # activate your virtual environment first
    source venv/bin/activate

    # run all sections (requires dataset paths to be configured)
    python examples/validation.py

    # run a single section
    python examples/validation.py --task detect
    python examples/validation.py --task segment
    python examples/validation.py --task classify
    python examples/validation.py --task depth
    python examples/validation.py --task standalone
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mata

# ---------------------------------------------------------------------------
# Dataset YAML paths — override via environment variables or edit directly.
# ---------------------------------------------------------------------------
COCO_YAML = os.environ.get("COCO_YAML", "examples/configs/coco.yaml")
IMAGENET_YAML = os.environ.get("IMAGENET_YAML", "examples/configs/imagenet.yaml")
DIODE_YAML = os.environ.get("DIODE_YAML", "examples/configs/diode.yaml")

# Sample images for the standalone section (uses the bundled example images)
_EXAMPLE_IMAGES_DIR = Path(__file__).parent / "images"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _save_metrics_json(metrics: object, save_dir: str, filename: str = "metrics.json") -> None:
    """Serialise *metrics* to ``<save_dir>/<filename>`` via its ``to_json()`` method."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(metrics.to_json(), encoding="utf-8")
    print(f"Metrics saved → {out_path}")


# ---------------------------------------------------------------------------
# 1. Detection validation (COCO dataset)
# ---------------------------------------------------------------------------
def run_detect() -> None:
    """Evaluate a detection model on the COCO val2017 split.

    Reported metrics:
        * mAP50      — AP at IoU=0.50
        * mAP50-95   — AP averaged over IoU thresholds 0.50:0.05:0.95
        * per-class  — mAP50-95 for every COCO category
    """
    print("\n" + "=" * 60)
    print("Detection Validation (COCO)")
    print("=" * 60)
    print(f"Dataset config : {COCO_YAML}")

    metrics = mata.val(
        "detect",
        model="facebook/detr-resnet-50",
        data=COCO_YAML,
        iou=0.50,
        conf=0.001,
        verbose=True,
        plots=True,
        save_dir="runs/val/detect",
    )

    print(f"\nmAP50    : {metrics.box.map50:.3f}")
    print(f"mAP50-95 : {metrics.box.map:.3f}")
    print(f"mAP75    : {metrics.box.map75:.3f}")
    print(f"Per-class mAP50-95 (first 5): {[round(v, 3) for v in metrics.box.maps[:5]]}")
    _save_metrics_json(metrics, "runs/val/detect")


# ---------------------------------------------------------------------------
# 2. Segmentation validation (COCO dataset)
# ---------------------------------------------------------------------------
def run_segment() -> None:
    """Evaluate a panoptic/instance segmentation model on COCO val2017.

    Reported metrics:
        * Box mAP50  — bounding-box AP at IoU=0.50
        * Mask mAP50 — mask AP at IoU=0.50
    """
    print("\n" + "=" * 60)
    print("Segmentation Validation (COCO)")
    print("=" * 60)
    print(f"Dataset config : {COCO_YAML}")

    metrics = mata.val(
        "segment",
        model="shi-labs/oneformer_coco_swin_large",
        data=COCO_YAML,
        iou=0.50,
        conf=0.001,
        verbose=True,
        plots=True,
        save_dir="runs/val/segment",
    )

    print(f"\nBox  mAP50    : {metrics.box.map50:.3f}")
    print(f"Box  mAP50-95 : {metrics.box.map:.3f}")
    print(f"Mask mAP50    : {metrics.seg.map50:.3f}")
    print(f"Mask mAP50-95 : {metrics.seg.map:.3f}")
    _save_metrics_json(metrics, "runs/val/segment")


# ---------------------------------------------------------------------------
# 3. Classification validation (ImageNet)
# ---------------------------------------------------------------------------
def run_classify() -> None:
    """Evaluate a classification model on an ImageNet-style val split.

    Reported metrics:
        * Top-1 accuracy
        * Top-5 accuracy
    """
    print("\n" + "=" * 60)
    print("Classification Validation (ImageNet)")
    print("=" * 60)
    print(f"Dataset config : {IMAGENET_YAML}")

    metrics = mata.val(
        "classify",
        model="microsoft/resnet-101",
        data=IMAGENET_YAML,
        verbose=True,
        save_dir="runs/val/classify",
    )

    print(f"\nTop-1 Accuracy : {metrics.top1:.1%}")
    print(f"Top-5 Accuracy : {metrics.top5:.1%}")
    _save_metrics_json(metrics, "runs/val/classify")


# ---------------------------------------------------------------------------
# 4. Depth validation (DIODE / NYU)
# ---------------------------------------------------------------------------
def run_depth() -> None:
    """Evaluate a monocular depth model on the DIODE validation set.

    Reported metrics:
        * AbsRel     — absolute relative error (lower is better)
        * SqRel      — squared relative error (lower is better)
        * RMSE       — root mean squared error (lower is better)
        * log-RMSE   — log-scale RMSE (lower is better)
        * δ < 1.25   — threshold accuracy (higher is better)
        * δ < 1.25²  — threshold accuracy (higher is better)
        * δ < 1.25³  — threshold accuracy (higher is better)
    """
    print("\n" + "=" * 60)
    print("Depth Estimation Validation (DIODE)")
    print("=" * 60)
    print(f"Dataset config : {DIODE_YAML}")

    metrics = mata.val(
        "depth",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        data=DIODE_YAML,
        verbose=True,
        save_dir="runs/val/depth",
    )

    print(f"\nAbsRel      : {metrics.abs_rel:.4f}")
    print(f"SqRel       : {metrics.sq_rel:.4f}")
    print(f"RMSE        : {metrics.rmse:.4f}")
    print(f"log-RMSE    : {metrics.log_rmse:.4f}")
    print(f"δ < 1.25    : {metrics.delta_1:.1%}")
    print(f"δ < 1.25²   : {metrics.delta_2:.1%}")
    print(f"δ < 1.25³   : {metrics.delta_3:.1%}")
    _save_metrics_json(metrics, "runs/val/depth")


# ---------------------------------------------------------------------------
# 5. Standalone validation (pre-run predictions)
# ---------------------------------------------------------------------------
def run_standalone() -> None:
    """Compute detection metrics from pre-run predictions.

    This workflow is useful when:
    - Inference was done in a separate step (e.g. on GPU servers).
    - You want to compare different post-processing thresholds without
        re-running the model.

    Steps:
        1. Collect images.
        2. Run inference to produce predictions.
        3. Call ``mata.val()`` with ``predictions=`` and ``ground_truth=``.
    """
    print("\n" + "=" * 60)
    print("Standalone Validation (pre-run predictions)")
    print("=" * 60)

    # --- Collect images -------------------------------------------------------
    images = sorted(_EXAMPLE_IMAGES_DIR.glob("*.jpg"))
    if not images:
        print(
            f"[SKIP] No images found in {_EXAMPLE_IMAGES_DIR}.  "
            "Add .jpg files to examples/images/ to run this section."
        )
        return

    print(f"Found {len(images)} image(s) in {_EXAMPLE_IMAGES_DIR}")

    # --- Run inference --------------------------------------------------------
    model_id = "facebook/detr-resnet-50"
    print(f"Running inference with {model_id} ...")

    predictions = [
        mata.run("detect", str(img), model=model_id, threshold=0.5)
        for img in images
    ]
    print(f"Collected {len(predictions)} prediction result(s).")

    # --- Validate against ground-truth annotations ----------------------------
    # Replace ANNOTATIONS_JSON with the path to your COCO-format JSON file.
    ANNOTATIONS_JSON = os.environ.get(
        "STANDALONE_ANNO", str(_EXAMPLE_IMAGES_DIR.parent / "annotations.json")
    )

    if not Path(ANNOTATIONS_JSON).exists():
        print(
            f"\n[INFO] No ground-truth file found at '{ANNOTATIONS_JSON}'.\n"
            "       Set the STANDALONE_ANNO environment variable or create the\n"
            "       file to enable metric computation.\n"
            "       Predictions were collected successfully — skipping scoring."
        )
        return

    metrics = mata.val(
        "detect",
        predictions=predictions,
        ground_truth=ANNOTATIONS_JSON,
        conf=0.001,
        iou=0.50,
        verbose=True,
        save_dir="runs/val/standalone",
    )

    print(f"\nmAP50    : {metrics.box.map50:.3f}")
    print(f"mAP50-95 : {metrics.box.map:.3f}")
    _save_metrics_json(metrics, "runs/val/standalone")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
_TASK_MAP = {
    "detect": run_detect,
    "segment": run_segment,
    "classify": run_classify,
    "depth": run_depth,
    "standalone": run_standalone,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MATA validation examples for all supported tasks."
    )
    parser.add_argument(
        "--task",
        choices=list(_TASK_MAP.keys()),
        default=None,
        help=(
            "Which validation section to run.  "
            "Omit to run all sections sequentially."
        ),
    )
    args = parser.parse_args()

    tasks = [args.task] if args.task else list(_TASK_MAP.keys())

    for task in tasks:
        try:
            _TASK_MAP[task]()
        except Exception as exc:  # noqa: BLE001
            print(f"\n[ERROR] {task} validation failed: {exc}")
            print("        Check dataset paths and model availability.\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
