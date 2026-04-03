"""Fine-Tune ResNet-50 for Image Classification — MATA Framework

Demonstrates the full fine-tuning lifecycle for an image classification model:
  1. Dataset Setup    — ImageFolder directory layout (one sub-folder per class)
  2. Fine-Tune        — mata.finetune() with frozen backbone for fast convergence
  3. Evaluate         — mata.val() on the validation split
  4. Export           — export best checkpoint for deployment
  5. Reload & Predict — mata.load() from checkpoint → classify new images

Requirements:
    pip install datamata[training]

Dataset expected layout (ImageFolder format):
    data/
      flowers/
        train/
          roses/       ← training images for class "roses"
          sunflowers/
          tulips/
          dandelion/
          daisy/
        val/
          roses/       ← validation images for class "roses"
          sunflowers/
          tulips/
          dandelion/
          daisy/

    The class names are inferred automatically from sub-folder names.

Run:
    python examples/train/finetune_classification.py

See also:
    examples/classify/basic_classification.py  — inference examples
    examples/configs/training_classify.yaml    — full YAML config reference
    docs/TRAINING_GUIDE.md                     — comprehensive training guide
"""

from __future__ import annotations

from pathlib import Path

import mata

# ---------------------------------------------------------------------------
# Configuration — adjust these paths to match your dataset
# ---------------------------------------------------------------------------

MODEL = "microsoft/resnet-50"  # HuggingFace model ID

# Root directories for training and validation splits (ImageFolder layout).
# Uses the auto-generated classify_mini dataset (3 classes: circle/square/triangle).
# Generate with:  python scripts/generate_classify_mini.py
TRAIN_DIR = "data/classify_mini/train"
VAL_DIR = "data/classify_mini/val"

# Output directory — auto-incremented to avoid overwriting previous runs
SAVE_DIR = "runs/train/classify"

# Image to use for the final prediction smoke test
SAMPLE_IMAGE = "examples/images/000000039769.jpg"


def main() -> None:
    # -----------------------------------------------------------------------
    # Step 1: Verify dataset exists
    # -----------------------------------------------------------------------
    train_path = Path(TRAIN_DIR)
    if not train_path.exists():
        print(
            f"[info] Training directory not found at '{TRAIN_DIR}'.\n"
            "       Generate the default dataset with:\n"
            "         python scripts/generate_classify_mini.py\n\n"
            "       Or create your own ImageFolder-style directory and update\n"
            "       TRAIN_DIR / VAL_DIR at the top of this script.\n\n"
            "       Example structure:\n"
            "         data/classify_mini/train/circle/\n"
            "         data/classify_mini/train/square/\n"
            "         data/classify_mini/val/circle/\n"
            "         data/classify_mini/val/square/\n"
        )
        return

    # -----------------------------------------------------------------------
    # Step 2: Fine-tune ResNet-50 on your image folder
    #
    # mata.finetune() defaults are tuned for transfer learning:
    #   - freeze_backbone=True  → only the final classification head is updated
    #   - lr=1e-5               → conservative LR prevents catastrophic forgetting
    #   - epochs=5              → few epochs; backbone features are already strong
    # -----------------------------------------------------------------------
    print(f"[finetune] Fine-tuning {MODEL} on {TRAIN_DIR} …")
    result = mata.finetune(
        "classify",
        model=MODEL,
        data=TRAIN_DIR,           # training split (ImageFolder directory)
        val_data=VAL_DIR,         # validation split
        epochs=10,
        batch_size=32,            # classification handles larger batches
        lr=1e-5,
        weight_decay=1e-4,
        save_dir=SAVE_DIR,
        save_every=0,             # save only best + last checkpoint
        val_every=1,              # validate every epoch
        patience=5,               # stop early if Top-1 stagnates for 5 epochs
        augment=True,
        device="auto",
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Step 3: Inspect training results
    # -----------------------------------------------------------------------
    print(result.summary())

    if result.history.get("train_loss"):
        epochs_done = len(result.history["train_loss"])
        print(f"[finetune] Completed {epochs_done} epoch(s).")
        print(
            f"[finetune] Final train loss : {result.history['train_loss'][-1]:.4f}"
        )
    if result.history.get("val_top1"):
        print(f"[finetune] Best Top-1 accuracy: {max(result.history['val_top1']):.1%}")

    # -----------------------------------------------------------------------
    # Step 4: Report validation metrics from training history
    #
    # NOTE: mata.val() for classification requires COCO-format annotations,
    # which ImageFolder datasets do not provide out of the box.  Training
    # already computes val accuracy each epoch; use those values here.
    # For full re-evaluation after training, provide a COCO-format labels JSON
    # (see docs/TRAINING_GUIDE.md — "Classification Evaluation" section).
    # -----------------------------------------------------------------------
    if result.history.get("val_top1"):
        best_top1 = max(result.history["val_top1"])
        print(f"\n[eval] Best val Top-1 (from training history): {best_top1:.1%}")
    elif result.history.get("val_loss"):
        best_val_loss = min(result.history["val_loss"])
        print(f"\n[eval] Best val loss (from training history): {best_val_loss:.4f}")
    else:
        print("\n[eval] Validation metrics not available in training history.")

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
    # Step 6: Reload the trained model and classify a new image
    # -----------------------------------------------------------------------
    checkpoint_to_load = result.best_checkpoint or result.last_checkpoint
    if not checkpoint_to_load:
        print("[predict] No checkpoint found — skipping prediction demo.")
        return

    print(f"\n[predict] Loading trained model from {checkpoint_to_load}")
    classifier = mata.load("classify", checkpoint_to_load, top_k=5)

    sample = Path(SAMPLE_IMAGE)
    if not sample.exists():
        print(f"[predict] Sample image not found at '{SAMPLE_IMAGE}'. Skipping.")
        return

    classify_result = classifier.predict(str(sample))
    top1 = classify_result.get_top1()
    print(f"[predict] Top prediction: {top1.label_name} ({top1.score:.1%})")
    print("[predict] Top-5 predictions:")
    for rank, pred in enumerate(classify_result.predictions[:5], 1):
        print(f"  {rank}. {pred.label_name:<25} {pred.score:.1%}")


if __name__ == "__main__":
    main()
