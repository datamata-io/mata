"""Generate a synthetic ImageFolder mini-dataset for classification smoke testing.

Creates ``data/classify_mini/`` with tiny JPEG images in class sub-directories
so that mata.train("classify", ...) exercises the full pipeline without needing
any real image download.

Usage::

    python scripts/generate_classify_mini.py              # default: 3 classes, 20 train / 5 val each
    python scripts/generate_classify_mini.py --per-class-train 8 --per-class-val 2 --size 64

Directory structure produced::

    data/classify_mini/
    ├── train/
    │   ├── circle/
    │   ├── square/
    │   └── triangle/
    └── val/
        ├── circle/
        ├── square/
        └── triangle/
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Synthetic "classes" — simple geometric shapes on coloured backgrounds
# ---------------------------------------------------------------------------

CLASSES = ["circle", "square", "triangle"]

_BG_COLOURS = [
    (230, 230, 200),
    (200, 230, 230),
    (230, 200, 230),
    (240, 240, 240),
]

_SHAPE_COLOURS = [
    (220, 60,  60),
    (60,  180, 60),
    (60,  100, 220),
    (220, 180, 50),
]


def _make_image(cls: str, size: int, rng: random.Random):
    """Return a PIL.Image showing a synthetic shape for the given class."""
    from PIL import Image, ImageDraw

    bg = rng.choice(_BG_COLOURS)
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    colour = rng.choice(_SHAPE_COLOURS)

    margin = size // 6
    x0 = rng.randint(margin, size // 2)
    y0 = rng.randint(margin, size // 2)
    x1 = rng.randint(size // 2, size - margin)
    y1 = rng.randint(size // 2, size - margin)
    # Ensure minimum size
    x1 = max(x1, x0 + margin)
    y1 = max(y1, y0 + margin)

    if cls == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=colour, outline=(0, 0, 0), width=2)
    elif cls == "square":
        draw.rectangle([x0, y0, x1, y1], fill=colour, outline=(0, 0, 0), width=2)
    else:  # triangle
        cx = (x0 + x1) // 2
        points = [(cx, y0), (x0, y1), (x1, y1)]
        draw.polygon(points, fill=colour, outline=(0, 0, 0))

    return img


def _build_split(
    out_dir: Path,
    split: str,
    per_class: int,
    size: int,
    seed: int,
    force: bool = False,
) -> None:
    split_dir = out_dir / split
    # Check existing
    if not force and split_dir.exists():
        existing = sum(1 for _ in split_dir.rglob("*.jpg"))
        expected = per_class * len(CLASSES)
        if existing == expected:
            print(f"[classify_mini] {split}: already exists with {existing} images – skipping.")
            return

    rng = random.Random(seed)
    total = 0
    for cls in CLASSES:
        cls_dir = split_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = _make_image(cls, size, rng)
            img.save(cls_dir / f"{i:04d}.jpg", "JPEG", quality=85)
            total += 1

    print(
        f"[classify_mini] {split}: {total} images across {len(CLASSES)} classes → {split_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic classification mini-dataset.")
    parser.add_argument("--out",              default="data/classify_mini", help="Output root directory")
    parser.add_argument("--per-class-train",  type=int, default=20,         help="Training images per class")
    parser.add_argument("--per-class-val",    type=int, default=5,          help="Validation images per class")
    parser.add_argument("--size",             type=int, default=64,         help="Square image size in pixels")
    parser.add_argument("--seed",             type=int, default=42,         help="Random seed")
    parser.add_argument("--force",            action="store_true",          help="Overwrite existing dataset")
    args = parser.parse_args()

    out = Path(args.out)
    _build_split(out, "train", args.per_class_train, args.size, seed=args.seed,     force=args.force)
    _build_split(out, "val",   args.per_class_val,   args.size, seed=args.seed + 1, force=args.force)
    print(f"[classify_mini] Done. Dataset root: {out.resolve()}")


if __name__ == "__main__":
    main()
