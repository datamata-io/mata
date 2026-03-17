"""Fine-Tune DETR for Object Detection — MATA Framework

Demonstrates the full fine-tuning lifecycle for an object detection model:
  1. Dataset Setup    — point to a COCO-format dataset (train + val splits)
  2. Fine-Tune        — mata.finetune() with backbone frozen, low LR
  3. Evaluate         — mata.val() on the validation split
  4. Export           — export best checkpoint for deployment
  5. Reload & Predict — mata.load() from checkpoint → predict on new images

Requirements:
    pip install datamata[training]

Dataset expected layout (COCO format):
    data/
      coco_custom/
        images/
          train2017/  ← training images
          val2017/    ← validation images
        annotations/
          instances_train2017.json
          instances_val2017.json

    OR point `TRAIN_DATA_YAML` to an existing .mata/models.yaml-style YAML:
        train: data/coco_custom/annotations/instances_train2017.json
        val:   data/coco_custom/annotations/instances_val2017.json
        images: data/coco_custom/images

Run:
    python examples/train/finetune_detection.py

See also:
    examples/detect/basic_detection.py   — inference examples
    examples/configs/training_detect.yaml — full YAML config reference
    docs/TRAINING_GUIDE.md               — comprehensive training guide
"""

from __future__ import annotations

from pathlib import Path

import mata

# ---------------------------------------------------------------------------
# Configuration — adjust these paths to match your dataset
# ---------------------------------------------------------------------------

MODEL = "facebook/detr-resnet-50"  # HuggingFace model ID

# Path to a COCO-format YAML or annotation directory.
# If the path doesn't exist, the script prints instructions and exits.
TRAIN_DATA_YAML = "examples/configs/coco_custom_train.yaml"

# Optional: separate val split path. None → use 'val' key in TRAIN_DATA_YAML.
VAL_DATA_YAML: str | None = None

# Output directory — auto-incremented (detect, detect2, ...) to avoid overwrite
SAVE_DIR = "runs/train/detect"

# Image to use for final prediction smoke test
SAMPLE_IMAGE = "examples/images/000000039769.jpg"


def main() -> None:
    # -----------------------------------------------------------------------
    # Step 1: Verify dataset exists (skip with mock path for demo)
    # -----------------------------------------------------------------------
    data_path = Path(TRAIN_DATA_YAML)
    if not data_path.exists():
        print(
            f"[info] Dataset config not found at '{TRAIN_DATA_YAML}'.\n"
            "       Create a YAML file with 'train:' and 'val:' keys pointing\n"
            "       to your COCO annotation JSONs, then rerun this script.\n\n"
            "       Example YAML (examples/configs/coco_custom_train.yaml):\n"
            "         train: data/coco_custom/annotations/instances_train2017.json\n"
            "         val:   data/coco_custom/annotations/instances_val2017.json\n"
            "         images: data/coco_custom/images\n"
        )
        return

    # -----------------------------------------------------------------------
    # Step 2: Fine-tune DETR on your dataset
    #
    # mata.finetune() uses fine-tuning defaults:
    #   - freeze_backbone=True  (only the detection head is updated)
    #   - lr=1e-5               (conservative learning rate)
    #   - epochs=5              (few epochs — weights are already pre-trained)
    # -----------------------------------------------------------------------
    print(f"[finetune] Fine-tuning {MODEL} …")
    result = mata.finetune(
        "detect",
        model=MODEL,
        data=TRAIN_DATA_YAML,
        val_data=VAL_DATA_YAML,
        epochs=10,
        batch_size=4,                  # reduce if GPU memory is limited
        lr=1e-5,
        save_dir=SAVE_DIR,
        save_every=5,                  # keep a checkpoint every 5 epochs
        val_every=2,                   # validate every 2 epochs
        patience=5,                    # stop early if no improvement for 5 epochs
        augment=True,
        device="auto",                 # uses CUDA if available, otherwise CPU
        gradient_checkpointing=True,   # recompute activations to save VRAM
        max_grad_norm=0.1,             # DETR paper value; prevents inf grad_norm
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Step 3: Inspect training results
    # -----------------------------------------------------------------------
    print(result.summary())

    if result.history.get("train_loss"):
        initial = result.history["train_loss"][0]
        final = result.history["train_loss"][-1]
        print(f"[finetune] Loss: {initial:.4f} -> {final:.4f}")

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
    # Step 5: Export the best checkpoint for deployment
    # -----------------------------------------------------------------------
    if result.best_checkpoint:
        from mata.training import CheckpointManager

        ckpt = CheckpointManager(result.best_checkpoint)
        export_path = Path(result.best_checkpoint) / "export"
        ckpt.export_for_inference(output_dir=str(export_path), engine="huggingface")
        print(f"[export] Model exported to {export_path}")

    # -----------------------------------------------------------------------
    # Step 6: Reload the trained model and run a prediction
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
