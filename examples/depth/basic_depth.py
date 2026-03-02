"""Depth Estimation Examples — MATA Framework

Demonstrates Depth Anything V1 and V2 with common output patterns.

Run:
  python examples/depth/basic_depth.py

Requirements:
  pip install transformers torch
"""

from pathlib import Path

import mata

# ── paths ─────────────────────────────────────────────────────────────────────
IMAGE_PATH = Path(__file__).parent.parent / "images" / "000000039769.jpg"
OUTPUT_DIR = Path("runs") / "depth"


# === Section 1: One-Shot Depth (Depth Anything V1) ===

def example_depth_v1():
    """Run depth estimation with Depth Anything V1 (small variant)."""
    print("\n=== Depth Anything V1 ===")

    result = mata.run(
        "depth",
        str(IMAGE_PATH),
        model="LiheYoung/depth-anything-small-hf",
        normalize=True,
    )

    print(f"Depth map shape: {result.depth_map.shape}")
    print(f"Min depth: {result.depth_map.min():.4f}  Max: {result.depth_map.max():.4f}")
    print(f"Model: {result.meta['model_id']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT_DIR / "depth_v1.png", colormap="magma")
    result.save(OUTPUT_DIR / "depth_v1.json")
    print(f"Saved → {OUTPUT_DIR}/depth_v1.png and depth_v1.json")


# === Section 2: One-Shot Depth (Depth Anything V2) ===

def example_depth_v2():
    """Run depth estimation with Depth Anything V2 (small variant)."""
    print("\n=== Depth Anything V2 ===")

    result = mata.run(
        "depth",
        str(IMAGE_PATH),
        model="depth-anything/Depth-Anything-V2-Small-hf",
        normalize=True,
    )

    print(f"Depth map shape: {result.depth_map.shape}")
    print(f"Min depth: {result.depth_map.min():.4f}  Max: {result.depth_map.max():.4f}")
    print(f"Model: {result.meta['model_id']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT_DIR / "depth_v2.png", colormap="magma")
    result.save(OUTPUT_DIR / "depth_v2.json")
    print(f"Saved → {OUTPUT_DIR}/depth_v2.png and depth_v2.json")


# === Section 3: Load Once, Predict Many ===

def example_load_once():
    """Load model once and run on multiple images for efficiency."""
    print("\n=== Load-once, predict many ===")

    depth_model = mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")

    images = [IMAGE_PATH]  # Add more paths here as needed
    for img_path in images:
        if not img_path.exists():
            print(f"  [skip] not found: {img_path}")
            continue
        result = depth_model.predict(str(img_path), normalize=True)
        print(f"  {img_path.name}: shape={result.depth_map.shape}")


def main():
    print("MATA — Depth Estimation Examples")
    print("=" * 40)

    if not IMAGE_PATH.exists():
        print(f"[warn] Test image not found: {IMAGE_PATH}")
        print("       Place an image at examples/images/000000039769.jpg to run examples.")
        return

    try:
        example_depth_v1()
    except Exception as exc:
        print(f"  [error] V1: {exc}")

    try:
        example_depth_v2()
    except Exception as exc:
        print(f"  [error] V2: {exc}")

    try:
        example_load_once()
    except Exception as exc:
        print(f"  [error] load-once: {exc}")


if __name__ == "__main__":
    main()
