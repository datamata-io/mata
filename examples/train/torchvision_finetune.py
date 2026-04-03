"""Fine-Tune Faster R-CNN (Torchvision) for Object Detection — MATA Framework

Demonstrates the full training lifecycle with a torchvision-native model:
  1. Dataset Setup    — COCO-format annotation YAML
  2. Train            — mata.train() with the torchvision/ prefix
  3. Evaluate         — mata.val() using the best checkpoint
  4. Export           — CheckpointManager exports weights + metadata
  5. Reload & Predict — mata.load() from checkpoint → predict on new images

Requirements:
    pip install datamata[training] torchvision

Torchvision model keys use the "torchvision/" prefix, e.g.:
  - torchvision/fasterrcnn_resnet50_fpn      ← Faster R-CNN ResNet-50 + FPN
  - torchvision/fasterrcnn_mobilenet_v3_large  ← Faster R-CNN MobileNet (faster)
  - torchvision/retinanet_resnet50_fpn        ← RetinaNet one-stage detector
  - torchvision/fcos_resnet50_fpn             ← FCOS anchor-free detector
  - torchvision/ssd300_vgg16                  ← SSD300 (legacy, very fast)
  - torchvision/ssdlite320_mobilenet_v3_large ← SSDLite (mobile-friendly)
  - torchvision/maskrcnn_resnet50_fpn         ← Mask R-CNN (detect + segment)

Dataset expected layout (COCO format):
    data/
      coco_custom/
        images/
          train2017/
          val2017/
        annotations/
          instances_train2017.json
          instances_val2017.json

Run:
    python examples/train/torchvision_finetune.py

See also:
    examples/detect/torchvision_detection.py  — torchvision inference examples
    examples/configs/training_detect.yaml     — full YAML config reference
    docs/TRAINING_GUIDE.md                    — comprehensive training guide
"""

from __future__ import annotations

from pathlib import Path

import mata

# ---------------------------------------------------------------------------
# Configuration — adjust these paths to match your dataset
# ---------------------------------------------------------------------------

# Torchvision model key — head is automatically replaced for your num_classes
MODEL = "torchvision/fasterrcnn_resnet50_fpn"

# Path to a COCO-format YAML config
TRAIN_DATA_YAML = "examples/configs/coco_custom_train.yaml"

# Optional: separate val YAML/path. None → use 'val' key in TRAIN_DATA_YAML.
VAL_DATA_YAML: str | None = None

# Number of classes in your dataset (background is NOT counted).
# Set to 3 for coco_custom_train.yaml (person, car, dog).
# Update this to match your own dataset's category count.
NUM_CLASSES = 3  # coco_mini has 3 foreground classes: person, car, dog

# Output directory — auto-incremented to avoid overwriting previous runs
SAVE_DIR = "runs/train/detect_tv"

# Image to use for the final prediction smoke test
SAMPLE_IMAGE = "examples/images/000000039769.jpg"


def main() -> None:
    # -----------------------------------------------------------------------
    # Step 1: Verify dataset config exists
    # -----------------------------------------------------------------------
    data_path = Path(TRAIN_DATA_YAML)
    if not data_path.exists():
        print(
            f"[info] Dataset config not found at '{TRAIN_DATA_YAML}'.\n"
            "       Create a YAML file with 'train' and 'val' keys pointing\n"
            "       to your COCO annotation JSONs, then rerun this script.\n\n"
            "       Example YAML (examples/configs/coco_custom_train.yaml):\n"
            "         train: data/coco_custom/annotations/instances_train2017.json\n"
            "         val:   data/coco_custom/annotations/instances_val2017.json\n"
            "         images: data/coco_custom/images\n"
            f"         nc: {NUM_CLASSES}\n"
        )
        return

    # -----------------------------------------------------------------------
    # Step 2: Train Faster R-CNN with a custom detection head
    #
    # MATA automatically replaces the final classification head to match
    # `num_classes`.  Pass num_classes via kwargs so the trainer can resize
    # the box predictor before fine-tuning begins.
    #
    # Key differences from HuggingFace training:
    #   - mata.train() (not mata.finetune()) — torchvision models benefit from
    #     a slightly higher LR since the head is initialised from scratch.
    #   - optimizer="sgd" with momentum is the Torchvision Detection tutorial default.
    #   - freeze_backbone=True still works — freezes model.backbone parameters.
    # -----------------------------------------------------------------------
    print(f"[train] Training {MODEL} …")
    result = mata.train(
        "detect",
        model=MODEL,
        data=TRAIN_DATA_YAML,
        val_data=VAL_DATA_YAML,
        epochs=20,
        batch_size=4,           # detection batches are memory-heavy; tune as needed
        lr=5e-3,                # SGD with momentum works well at this LR
        optimizer="sgd",        # matches torchvision tutorial recommendation
        weight_decay=5e-4,
        scheduler="cosine",
        warmup_epochs=2,        # warm up LR during first 2 epochs
        freeze_backbone=False,  # full fine-tune — set True for faster convergence
        save_dir=SAVE_DIR,
        save_every=5,           # save a checkpoint every 5 epochs
        val_every=2,            # run validation every 2 epochs
        patience=8,             # stop if no improvement for 8 consecutive epochs
        augment=True,
        device="auto",
        amp=True,               # AMP disabled automatically on CPU
        num_classes=NUM_CLASSES,  # forwarded to the head replacement logic
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Step 3: Inspect training results
    # -----------------------------------------------------------------------
    print(result.summary())

    if result.history.get("train_loss"):
        losses = result.history["train_loss"]
        print(f"[train] Loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
              f"  ({result.epochs_completed} epochs)")

    if result.history.get("val_map50"):
        best_map50 = max(result.history["val_map50"])
        print(f"[train] Best mAP50: {best_map50:.3f}")

    # -----------------------------------------------------------------------
    # Step 4: Evaluate best checkpoint with mata.val()
    # -----------------------------------------------------------------------
    if result.best_checkpoint:
        print(f"\n[eval] Evaluating best checkpoint: {result.best_checkpoint}")
        eval_result = mata.val(
            "detect",
            model=result.best_checkpoint,
            data=VAL_DATA_YAML or TRAIN_DATA_YAML,
            batch_size=4,
            device="auto",
            verbose=True,
        )
        if hasattr(eval_result, "box"):
            print(f"[eval] mAP50   : {eval_result.box.map50:.3f}")
            print(f"[eval] mAP50-95: {eval_result.box.map:.3f}")

    # -----------------------------------------------------------------------
    # Step 5: Export the best checkpoint for inference deployment
    #
    # Torchvision checkpoints are exported as model.pth + metadata.json.
    # Reload them with mata.load("detect", checkpoint_dir) — the loader reads
    # metadata.json to detect the torchvision engine and restores weights.
    # -----------------------------------------------------------------------
    if result.best_checkpoint:
        from mata.training import CheckpointManager

        ckpt = CheckpointManager(result.best_checkpoint)
        export_path = Path(result.best_checkpoint) / "export"
        ckpt.export_for_inference(output_dir=str(export_path), engine="torchvision")
        print(f"[export] Model exported to {export_path}")
        print(f"         -> model.pth + metadata.json written to {export_path}")

    # -----------------------------------------------------------------------
    # Step 6: Reload the trained checkpoint and run a prediction
    #
    # mata.load() inspects config.json / metadata.json in the checkpoint
    # directory, detects the "torchvision" engine, and reconstructs the exact
    # same architecture before loading the saved weights.
    # -----------------------------------------------------------------------
    checkpoint_to_load = result.best_checkpoint or result.last_checkpoint
    if not checkpoint_to_load:
        print("[predict] No checkpoint found — skipping prediction demo.")
        return

    print(f"\n[predict] Loading trained model from {checkpoint_to_load}")
    detector = mata.load("detect", checkpoint_to_load, threshold=0.5)

    sample = Path(SAMPLE_IMAGE)
    if not sample.exists():
        print(f"[predict] Sample image not found at '{SAMPLE_IMAGE}'. Skipping.")
        return

    predict_result = detector.predict(str(sample))
    print(f"[predict] Detected {len(predict_result.detections)} objects in {sample.name}")
    for det in predict_result.detections[:5]:
        print(f"  {det.label_name:<20} score={det.score:.1%}  box={det.bbox}")


if __name__ == "__main__":
    main()
