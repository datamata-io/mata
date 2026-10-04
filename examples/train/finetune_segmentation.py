"""Fine-Tune Mask2Former for Instance Segmentation — MATA Framework

Demonstrates the full fine-tuning lifecycle for a segmentation model:
  1. Dataset Setup    — COCO-format instance segmentation annotations
  2. Fine-Tune        — mata.finetune() with backbone frozen
  3. Evaluate         — mata.val() on the validation split (mask mAP)
  4. Export           — export best checkpoint for deployment
  5. Reload & Predict — mata.load() from checkpoint → segment new images

Requirements:
    pip install datamata[training]

Dataset expected layout (COCO instance segmentation format):
    data/
      coco_seg/
        images/
          train2017/  ← training images
          val2017/    ← validation images
        annotations/
          instances_train2017.json   ← must include 'segmentation' RLE/polygon fields
          instances_val2017.json

    Each annotation object in the JSON should include:
      "segmentation": [[x1,y1, x2,y2, ...]]   # polygon, OR
      "segmentation": {"counts": "...", "size": [H, W]}  # RLE

Run:
    python examples/train/finetune_segmentation.py

See also:
    examples/segment/           — inference examples
    docs/TRAINING_GUIDE.md      — comprehensive training guide
"""

from __future__ import annotations

from pathlib import Path

import mata

# ---------------------------------------------------------------------------
# Configuration — adjust these paths to match your dataset
# ---------------------------------------------------------------------------

# Use the swin-tiny variant — lower VRAM footprint (~8 GB with gradient checkpointing).
# Switch to mask2former-swin-small-coco-instance for higher accuracy on ≥16 GB GPUs.
MODEL = "facebook/mask2former-swin-tiny-coco-instance"  # HuggingFace model ID

# Path to a COCO-format YAML with 'train' and 'val' annotation paths.
TRAIN_DATA_YAML = "examples/configs/coco_seg_train.yaml"

# Optional: separate val YAML/path. None → use 'val' key in TRAIN_DATA_YAML.
VAL_DATA_YAML: str | None = None

# Output directory — auto-incremented to avoid overwriting previous runs
SAVE_DIR = "runs/train/segment"

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
            "       Create a YAML file pointing to COCO segmentation annotations.\n\n"
            "       Example YAML (examples/configs/coco_seg_train.yaml):\n"
            "         train: data/coco_seg/annotations/instances_train2017.json\n"
            "         val:   data/coco_seg/annotations/instances_val2017.json\n"
            "         images: data/coco_seg/images\n"
        )
        return

    # -----------------------------------------------------------------------
    # Step 2: Fine-tune Mask2Former on your segmentation dataset
    #
    # Mask2Former is a large model — fine-tuning the full network is expensive.
    # With freeze_backbone=True only the mask decoder is updated, which is
    # 5–10× faster and works well when your domain is close to COCO.
    # -----------------------------------------------------------------------
    print(f"[finetune] Fine-tuning {MODEL} …")
    result = mata.finetune(
        "segment",
        model=MODEL,
        data=TRAIN_DATA_YAML,
        val_data=VAL_DATA_YAML,
        epochs=10,
        batch_size=2,                  # Mask2Former is memory-intensive; use small batch
        lr=1e-5,
        weight_decay=0.05,             # Mask2Former default from the paper
        scheduler="cosine",
        warmup_epochs=1,
        save_dir=SAVE_DIR,
        save_every=0,                  # save only best + last
        val_every=2,                   # validation is slow — run every 2 epochs
        patience=5,
        augment=True,
        device="auto",
        amp=True,                      # mixed precision is important for large models
        gradient_checkpointing=True,   # recompute activations to save VRAM
        max_grad_norm=0.1,             # prevents gradient explosion common in Mask2Former
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Step 3: Inspect training results
    # -----------------------------------------------------------------------
    print(result.summary())

    if result.history.get("train_loss"):
        print(
            f"[finetune] Loss: "
            f"{result.history['train_loss'][0]:.4f} -> "
            f"{result.history['train_loss'][-1]:.4f}"
        )

    # -----------------------------------------------------------------------
    # Step 4: Evaluate best checkpoint with mata.val()
    #
    # Reports PQ (panoptic quality), mAP for instance segmentation, etc.
    # -----------------------------------------------------------------------
    if result.best_checkpoint:
        print(f"\n[eval] Evaluating best checkpoint: {result.best_checkpoint}")
        eval_result = mata.val(
            "segment",
            model=result.best_checkpoint,
            data=VAL_DATA_YAML or TRAIN_DATA_YAML,
            batch_size=2,
            device="auto",
            verbose=True,
        )
        # Segmentation metrics surface varies by model type (panoptic vs instance)
        if hasattr(eval_result, "box"):
            print(f"[eval] Mask mAP50   : {eval_result.box.map50:.3f}")
            print(f"[eval] Mask mAP50-95: {eval_result.box.map:.3f}")
        elif hasattr(eval_result, "pq"):
            print(f"[eval] Panoptic Quality (PQ): {eval_result.pq:.3f}")

    # -----------------------------------------------------------------------
    # Step 5: Export best checkpoint for inference deployment
    # -----------------------------------------------------------------------
    if result.best_checkpoint:
        from mata.training import CheckpointManager

        ckpt = CheckpointManager(result.best_checkpoint)
        export_path = Path(result.best_checkpoint) / "export"
        ckpt.export_for_inference(output_dir=str(export_path), engine="huggingface")
        print(f"[export] Model exported to {export_path}")

    # -----------------------------------------------------------------------
    # Step 6: Reload the trained model and segment a new image
    # -----------------------------------------------------------------------
    checkpoint_to_load = result.best_checkpoint or result.last_checkpoint
    if not checkpoint_to_load:
        print("[predict] No checkpoint found — skipping prediction demo.")
        return

    print(f"\n[predict] Loading trained model from {checkpoint_to_load}")
    segmentor = mata.load("segment", checkpoint_to_load, threshold=0.5)

    sample = Path(SAMPLE_IMAGE)
    if not sample.exists():
        print(f"[predict] Sample image not found at '{SAMPLE_IMAGE}'. Skipping.")
        return

    segment_result = segmentor.predict(str(sample))
    instances = segment_result.get_instances()
    print(f"[predict] Found {len(instances)} instances in {sample.name}")
    for inst in instances[:5]:
        label = getattr(inst, "label_name", getattr(inst, "label", "?"))
        score = getattr(inst, "score", 0.0)
        has_mask = inst.mask is not None
        print(f"  {label:<20} score={score:.1%}  mask={'yes' if has_mask else 'no'}")


if __name__ == "__main__":
    main()
