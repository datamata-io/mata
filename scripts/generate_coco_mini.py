"""Generate a synthetic COCO-mini dataset for smoke testing.

Creates ``data/coco_mini/`` with tiny JPEG images and valid COCO JSON
annotations so that mata.train() exercises the full training pipeline
without needing the real COCO download.

Usage::

    python scripts/generate_coco_mini.py           # default: 16 train / 4 val
    python scripts/generate_coco_mini.py --train 8 --val 2 --size 128

Images are 320×320 synthetic RGB frames with coloured rectangles acting as
"objects".  Three COCO-compatible categories are used so that fine-tuning a
DETR checkpoint (which has 91 COCO classes pre-trained) works without any
label-space mismatch.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Subset of real COCO category ids & names so that pre-trained checkpoints
# don't need a new classification head during the smoke test.
CATEGORIES = [
    {"id": 1,  "name": "person",     "supercategory": "person"},
    {"id": 3,  "name": "car",        "supercategory": "vehicle"},
    {"id": 17, "name": "dog",        "supercategory": "animal"},
]
CAT_IDS = [c["id"] for c in CATEGORIES]

# Vivid fill colours (RGB) for the synthetic "objects"
_COLOURS = [
    (220, 60,  60),   # red
    (60,  180, 60),   # green
    (60,  100, 220),  # blue
    (220, 200, 50),   # yellow
    (180, 60,  220),  # purple
]

_BG_COLOURS = [
    (240, 232, 210),
    (210, 232, 240),
    (232, 240, 210),
    (240, 210, 232),
]


# ---------------------------------------------------------------------------
# Image + annotation generation
# ---------------------------------------------------------------------------

def _make_image(width: int, height: int, n_objects: int, rng: random.Random):
    """Return (PIL.Image, list[bbox_xywh]) for one synthetic frame."""
    from PIL import Image, ImageDraw  # pillow is a mata dependency

    bg = rng.choice(_BG_COLOURS)
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    boxes_xywh: list[list[float]] = []

    min_side = max(20, min(width, height) // 6)
    max_side = min(width, height) // 2

    for _ in range(n_objects):
        w = rng.randint(min_side, max_side)
        h = rng.randint(min_side, max_side)
        x = rng.randint(0, width  - w)
        y = rng.randint(0, height - h)
        colour = rng.choice(_COLOURS)
        draw.rectangle([x, y, x + w, y + h], fill=colour, outline=(0, 0, 0), width=2)
        boxes_xywh.append([float(x), float(y), float(w), float(h)])

    return img, boxes_xywh


def _build_split(
    out_dir: Path,
    split: str,
    n_images: int,
    image_size: int,
    seed: int,
    force: bool = False,
) -> None:
    """Generate images + COCO JSON for one split (``train`` or ``val``)."""
    rng = random.Random(seed)

    images_dir = out_dir / f"{split}2017"
    ann_dir    = out_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    ann_path = ann_dir / f"instances_{split}2017.json"
    if ann_path.exists():
        # Check if the number of images matches – if so, skip regeneration.
        existing = json.loads(ann_path.read_text())
        if len(existing.get("images", [])) == n_images and not force:
            print(f"[coco_mini] {split}: already exists with {n_images} images – skipping.")
            return

    images_meta: list[dict] = []
    annotations:  list[dict] = []
    ann_id = 1

    for idx in range(1, n_images + 1):
        n_objs = rng.randint(2, 5)
        img, boxes = _make_image(image_size, image_size, n_objs, rng)

        file_name = f"{idx:06d}.jpg"
        img.save(images_dir / file_name, "JPEG", quality=85)

        images_meta.append({
            "id":        idx,
            "file_name": file_name,
            "width":     image_size,
            "height":    image_size,
        })

        for bbox in boxes:
            x, y, w, h = bbox
            cat_id = rng.choice(CAT_IDS)
            # Polygon = rectangle corners (clockwise), matching the bounding box.
            # This gives COCOSegmentationDataset a valid non-empty segmentation
            # field so mask decoding produces a filled rectangle mask rather
            # than an all-zero mask.
            polygon = [x, y,  x + w, y,  x + w, y + h,  x, y + h]
            annotations.append({
                "id":          ann_id,
                "image_id":    idx,
                "category_id": cat_id,
                "bbox":        [x, y, w, h],
                "area":        w * h,
                "iscrowd":     0,
                "segmentation": [polygon],
            })
            ann_id += 1

    coco_json = {
        "info":        {"description": "MATA coco-mini synthetic dataset"},
        "licenses":    [],
        "images":      images_meta,
        "annotations": annotations,
        "categories":  CATEGORIES,
    }
    ann_path.write_text(json.dumps(coco_json, indent=2), encoding="utf-8")
    print(
        f"[coco_mini] {split}: {n_images} images, {len(annotations)} annotations → {ann_path}"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic COCO-mini dataset.")
    parser.add_argument("--out",   default="data/coco_mini", help="Output root directory")
    parser.add_argument("--train", type=int, default=16,     help="Number of training images")
    parser.add_argument("--val",   type=int, default=4,      help="Number of validation images")
    parser.add_argument("--size",  type=int, default=320,    help="Square image size in pixels")
    parser.add_argument("--seed",  type=int, default=42,     help="Random seed")
    parser.add_argument("--force", action="store_true",      help="Overwrite existing dataset")
    args = parser.parse_args()

    out = Path(args.out)
    _build_split(out, "train", args.train, args.size, seed=args.seed, force=args.force)
    _build_split(out, "val",   args.val,   args.size, seed=args.seed + 1, force=args.force)
    print(f"[coco_mini] Done. Dataset root: {out.resolve()}")


if __name__ == "__main__":
    main()
