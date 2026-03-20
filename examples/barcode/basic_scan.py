"""Barcode & QR Code Scanning Examples — MATA Framework

Progressive examples from simplest one-liner to advanced patterns:
    1. One-Shot Scan       — mata.run() for quick single-image barcode reading
    2. Load Once, Predict Many — reuse an adapter across multiple images
    3. Switch Engines      — pyzbar vs zxing-cpp
    4. Work with Results   — iterate barcodes, access data/type/bbox, filter
    5. Export Results      — JSON string, .json file, .csv file
    6. ROI Pipeline        — Detect regions first, then scan crops

Requirements:
    pip install datamata[barcode]          # pyzbar (default engine)
    pip install datamata[barcode-zxing]    # zxing-cpp (alternative engine)
    pip install datamata[barcode-all]      # both engines

Run:
    python examples/barcode/basic_scan.py
    python examples/barcode/basic_scan.py path/to/image_with_barcode.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import mata

# ── paths ─────────────────────────────────────────────────────────────────────
IMAGE_DIR = Path(__file__).parent.parent / "images"

# Use a command-line argument if provided, otherwise fall back to a sample image
_cli_image = sys.argv[1] if len(sys.argv) > 1 else None
BARCODE_IMAGE = Path(_cli_image) if _cli_image else IMAGE_DIR / "sample_barcode.jpg"
QR_IMAGE = Path(_cli_image) if _cli_image else IMAGE_DIR / "sample_qr.jpg"
MULTI_IMAGE = Path(_cli_image) if _cli_image else IMAGE_DIR / "sample_graph_barcode.png"


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def _check_image(path: Path) -> bool:
    if not path.exists():
        print(f"  [skip] image not found: {path}")
        print("         Pass an image path as argument: python basic_scan.py myimage.jpg")
        return False
    return True


# === Section 1: One-Shot Scan ===


def section_one_shot():
    """Single call — no setup required."""
    print("\n=== Section 1: One-Shot Scan ===")
    if not _check_image(BARCODE_IMAGE):
        return

    result = mata.run("barcode", str(BARCODE_IMAGE), model="pyzbar")
    print(f"Found {len(result.barcodes)} barcode(s):")
    for bc in result.barcodes:
        print(f"  [{bc.type}] {bc.data!r}  (confidence: {bc.score:.2f})")
        if bc.bbox:
            x1, y1, x2, y2 = bc.bbox
            print(f"    bbox: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")


# === Section 2: Load Once, Predict Many ===


def section_load_reuse():
    """Load the adapter once, then call .predict() for each image."""
    print("\n=== Section 2: Load Once, Predict Many ===")

    adapter = mata.load("barcode", "pyzbar")
    for img in [BARCODE_IMAGE, QR_IMAGE]:  # replace with your own list
        if not _check_image(img):
            continue
        result = adapter.predict(str(img))
        print(f"  {img.name}: {len(result.barcodes)} barcode(s)")


# === Section 3: Switch Engines ===


def section_switch_engines():
    """Compare pyzbar and zxing-cpp on the same image."""
    print("\n=== Section 3: Switch Engines ===")
    if not _check_image(BARCODE_IMAGE):
        return

    engines = {
        "pyzbar":  "pip install datamata[barcode]",
        "zxing":   "pip install datamata[barcode-zxing]",
    }
    for engine, install_hint in engines.items():
        try:
            result = mata.run("barcode", str(BARCODE_IMAGE), model=engine)
            print(f"  [{engine}] {len(result.barcodes)} barcode(s) — engine: {result.meta.get('engine', engine)}")
        except ImportError as exc:
            print(f"  [{engine}] not installed — {install_hint}")
            print(f"    ({exc})")


# === Section 4: Work with Results ===


def section_work_with_results():
    """Iterate barcodes, inspect fields, use filter_by_type()."""
    print("\n=== Section 4: Work with Results ===")

    # 1D barcodes from BARCODE_IMAGE
    if _check_image(BARCODE_IMAGE):
        result = mata.run("barcode", str(BARCODE_IMAGE), model="pyzbar")
        if not result.barcodes:
            print("  No barcodes found in BARCODE_IMAGE.")
        else:
            print(f"BARCODE_IMAGE — {len(result)} barcode(s):")
            for i, bc in enumerate(result, start=1):
                print(f"  {i}. type={bc.type!r}  data={bc.data!r}  score={bc.score:.2f}")
                if bc.raw_bytes:
                    print(f"     raw_bytes (hex): {bc.raw_bytes.hex()[:32]}...")

            # Filter to only 1D barcodes
            oned = result.filter_by_type("EAN_13", "EAN_8", "CODE_128", "CODE_39", "UPC_A", "UPC_E")
            print(f"  1D barcodes: {len(oned)}")

    # QR codes from QR_IMAGE
    if _check_image(QR_IMAGE):
        qr_result = mata.run("barcode", str(QR_IMAGE), model="pyzbar")
        if not qr_result.barcodes:
            print("  No barcodes found in QR_IMAGE.")
        else:
            print(f"\nQR_IMAGE — {len(qr_result)} barcode(s):")
            for i, bc in enumerate(qr_result, start=1):
                print(f"  {i}. type={bc.type!r}  data={bc.data!r}  score={bc.score:.2f}")

            qr_only = qr_result.filter_by_type("QR_CODE")
            print(f"  QR codes: {len(qr_only)}")


# === Section 5: Export Results ===


def section_export():
    """Save results as JSON or CSV."""
    print("\n=== Section 5: Export Results ===")
    if not _check_image(BARCODE_IMAGE):
        return

    result = mata.run("barcode", str(BARCODE_IMAGE), model="pyzbar")

    # JSON string
    json_str = result.to_json(indent=2)
    print(f"JSON preview (first 200 chars):\n{json_str[:200]}...")

    # Save to .json file
    json_path = OUTPUT_DIR / "barcodes.json"
    result.save(str(json_path))
    print(f"\nSaved JSON: {json_path}")

    # Save to .csv (data, type, score, bbox columns)
    csv_path = OUTPUT_DIR / "barcodes.csv"
    result.save(str(csv_path))
    print(f"Saved CSV:  {csv_path}")


# === Section 6: ROI Pipeline (Detect → Scan Crops) ===


def section_roi_pipeline():
    """Use the Barcode graph node after zero-shot detection to scan only relevant crops.

    This pattern is useful when:
    - You know barcodes appear on specific objects (e.g. product labels)
    - You want to correlate barcode data with detected instances
    - You need to limit scanning to bounding-box crops for speed

    Grounding DINO is used as the detector because it supports open-vocabulary
    text prompts (e.g. "barcode . qr code"), unlike COCO-trained models which
    have no barcode class.

    The graph: Detect(grounding-dino) >> ExtractROIs >> Barcode >> Fuse
    """
    print("\n=== Section 6: ROI Pipeline ===")
    if not _check_image(MULTI_IMAGE):
        return

    try:
        from mata.nodes import Barcode, Detect, ExtractROIs, Fuse
    except ImportError:
        print("  [skip] graph nodes not available in this environment")
        return

    graph = [
        Detect(using="gdino", out="dets", threshold=0.3, text_prompts="barcode . qr code"),
        ExtractROIs(src_dets="dets", out="rois"),
        Barcode(using="pyzbar", src="rois", out="codes"),
        Fuse(),
    ]

    detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
    barcode_adapter = mata.load("barcode", "pyzbar")
    try:
        result = mata.infer(
            str(MULTI_IMAGE),
            graph,
            providers={
                "detect": {"gdino": detector},
                "barcode": {"pyzbar": barcode_adapter},
            },
            text_prompts="barcode . qr code",
        )

        json_path = OUTPUT_DIR / "barcodes_graph.json"
        # avoid result.save() here because it tries to save all channels, including "rois" which is huge
        # Exclude "rois" — it contains raw pixel arrays and inflates JSON to gigabytes.
        # Save only the meaningful channels: detections and decoded barcodes.
        import json
        slim = {
            "channels": {k: v for k, v in result.to_dict()["channels"].items() if k != "rois"},
            "provenance": result.provenance,
            "metrics": result.metrics,
        }
        json_path.write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
        print(f"  Saved JSON: {json_path}")

        print(f"  Graph result type: {type(result).__name__}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [graph] {exc}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("MATA Barcode Scanning Examples")
    print("=" * 40)

    section_one_shot()
    section_load_reuse()
    section_switch_engines()
    section_work_with_results()
    section_export()
    section_roi_pipeline()

    print("\nDone.")


if __name__ == "__main__":
    main()
