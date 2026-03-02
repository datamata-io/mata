"""Build DIODE depth ground-truth files for mata.val().

Generates two artefacts from the DIODE validation split:
  1. depth_gt_flat/ — flattened (H, W) float32 .npy arrays (invalid pixels zeroed)
  2. diode_val_depth.json — {relative_rgb_path: relative_depth_npy_path} mapping

Usage (from the MATA project root):

    python scripts/build_diode_depth_gt.py \\
        --diode-root /data/diode/val \\
        --output diode_val_depth.json
"""
import argparse
import json
import numpy as np
from pathlib import Path


def build(diode_val_dir: Path, output_json: Path) -> None:
    depth_gt_dir = output_json.parent / "depth_gt_flat"
    depth_gt_dir.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, str] = {}
    n = 0
    for png in sorted(diode_val_dir.rglob("*.png")):
        stem = png.stem
        depth_npy = png.parent / f"{stem}_depth.npy"
        mask_npy = png.parent / f"{stem}_depth_mask.npy"
        if not depth_npy.exists():
            continue

        depth = np.load(str(depth_npy)).astype(np.float32)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)  # (H, W, 1) → (H, W)

        if mask_npy.exists():
            mask = np.load(str(mask_npy)).astype(bool)
            depth[~mask] = 0.0  # 0 = invalid; auto-mask in process_batch excludes gt<=0

        flat = depth_gt_dir / f"{stem}.npy"
        flat.unlink(missing_ok=True)  # remove old file if present
        np.save(str(flat), depth)

        rel_rgb = str(png.relative_to(diode_val_dir.parent))
        rel_depth = str(flat.relative_to(output_json.parent))
        mapping[rel_rgb] = rel_depth

        n += 1
        if n % 100 == 0:
            print(f"  {n} done...")

    output_json.write_text(json.dumps(mapping, indent=2))
    print(f"Saved {n} depth arrays to {depth_gt_dir}")
    print(f"Wrote annotation mapping to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DIODE depth GT files for mata.val()")
    parser.add_argument(
        "--diode-root",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "diode" / "val",
        help="Path to the DIODE val/ directory (default: data/diode/val)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the JSON annotation file (default: <diode-root>/../diode_val_depth.json)",
    )
    args = parser.parse_args()

    val_dir: Path = args.diode_root.resolve()
    output_json: Path = (
        args.output.resolve()
        if args.output
        else val_dir.parent / "diode_val_depth.json"
    )

    if not val_dir.exists():
        raise SystemExit(f"DIODE val directory not found: {val_dir}")

    print(f"Building depth GT from: {val_dir}")
    print(f"Output JSON: {output_json}")
    build(val_dir, output_json)
