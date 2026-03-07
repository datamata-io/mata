"""OCR Examples — MATA Framework

Demonstrates four OCR backends and common result workflows:
    - EasyOCR         (80+ languages, polygon bboxes)
    - PaddleOCR       (multilingual, strong on non-Latin scripts)
    - Tesseract       (classic open-source engine)
    - HuggingFace GOT-OCR2 / TrOCR (transformer-based)

Plus utilities: load-once, export, confidence filtering, config aliases.

Run a specific backend:
    python examples/vlm/ocr.py easyocr
    python examples/vlm/ocr.py paddleocr
    python examples/vlm/ocr.py tesseract
    python examples/vlm/ocr.py trocr
    python examples/vlm/ocr.py           # runs all sections

Requirements (install the backends you want to test):
    pip install easyocr
    pip install paddlepaddle paddleocr
    pip install pytesseract              # also needs Tesseract binary on PATH
    pip install transformers torch       # for HuggingFace models
"""

from __future__ import annotations

import sys
from pathlib import Path

import mata

# ── paths ─────────────────────────────────────────────────────────────────────
IMAGE_DIR = Path(__file__).parent.parent / "images"
STOCKS_IMAGE = IMAGE_DIR / "ocr" / "stocks.png"
RECEIPT_IMAGE = IMAGE_DIR / "ocr" / "receipt.png"
LINE_CROP_IMAGE = IMAGE_DIR / "ocr" / "a01-122-02.jpg"
ROI_TEST_IMAGE = IMAGE_DIR / "license_plate" / "inst_0001.png"


def _check_image(path: Path) -> bool:
    if not path.exists():
        print(f"  [skip] image not found: {path}")
        return False
    return True


# === Section 1: EasyOCR ===

def demo_easyocr():
    """EasyOCR — multi-language, returns polygon bboxes."""
    print("\n=== EasyOCR ===")
    if not _check_image(RECEIPT_IMAGE):
        return

    adapter = mata.load("ocr", "easyocr", languages=[ "en"])
    result = adapter.predict(RECEIPT_IMAGE)

    print(f"Full text:\n{result.full_text}\n")
    print(f"Detected {len(result.regions)} region(s):")
    for region in result.regions:
        loc = f" @ {region.bbox}" if region.bbox else ""
        print(f"  {region.text!r}  (conf={region.score:.2f}){loc}")


# === Section 2: PaddleOCR ===

def demo_paddleocr():
    """PaddleOCR — great for Chinese, Japanese, Korean and many other scripts."""
    print("\n=== PaddleOCR ===")
    if not _check_image(STOCKS_IMAGE):
        return

    result = mata.run("ocr", STOCKS_IMAGE, model="paddleocr", lang="en")
    print(f"Full text (EN):\n{result.full_text}\n")

    # For non-Latin scripts, pass lang="zh", "ja", "ko", etc.


# === Section 3: Tesseract ===

def demo_tesseract():
    """Tesseract — battle-tested open-source OCR engine."""
    print("\n=== Tesseract ===")
    if not _check_image(STOCKS_IMAGE):
        return

    result = mata.run("ocr", STOCKS_IMAGE, model="tesseract", lang="eng")
    print(f"Full text:\n{result.full_text}\n")
    print(f"Regions: {len(result.regions)}")


# === Section 4: HuggingFace GOT-OCR2 ===

def demo_got_ocr2():
    """GOT-OCR2 — state-of-the-art end-to-end document OCR."""
    print("\n=== HuggingFace GOT-OCR2 ===")
    print("  [skip] GOT-OCR2 hallucinates with current transformers version — skipping.")
    print("  See: https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf")


# === Section 5: HuggingFace TrOCR ===

def demo_trocr():
    """TrOCR — transformer-based OCR, excellent for single line crops."""
    print("\n=== HuggingFace TrOCR (printed) ===")
    if not _check_image(LINE_CROP_IMAGE):
        return

    result = mata.run("ocr", LINE_CROP_IMAGE, model="microsoft/trocr-base-printed")
    print(f"Recognized: {result.full_text!r}")

    result_hw = mata.run("ocr", LINE_CROP_IMAGE, model="microsoft/trocr-base-handwritten")
    print(f"Handwritten: {result_hw.full_text!r}")


# === Section 6: Load Once, Predict Many ===

def demo_load_once():
    """Load an adapter once, then call predict() on multiple images."""
    print("\n=== Load-once adapter pattern ===")

    adapter = mata.load("ocr", "easyocr")
    images = [STOCKS_IMAGE, LINE_CROP_IMAGE]
    for img_path in images:
        if not _check_image(img_path):
            continue
        result = adapter.predict(img_path)
        print(f"  [{img_path.name}] > {result.full_text[:80]!r}")


# === Section 7: Export Results ===

def demo_export():
    """Save OCR results in multiple formats: txt, csv, json, and image overlay."""
    print("\n=== Export results ===")
    if not _check_image(STOCKS_IMAGE):
        return

    result = mata.run("ocr", STOCKS_IMAGE, model="easyocr")

    out_dir = Path("runs") / "ocr"
    out_dir.mkdir(parents=True, exist_ok=True)

    result.save(str(out_dir / "output.txt"))                             # plain text
    result.save(str(out_dir / "output.csv"))                             # CSV rows
    result.save(str(out_dir / "output.json"))                            # structured JSON
    result.save(str(out_dir / "overlay.png"), source_image=STOCKS_IMAGE) # annotated image

    print(f"Saved to {out_dir}/")
    print(f"  output.txt  — {len(result.full_text)} chars")
    print(f"  output.csv  — {len(result.regions)} rows")
    print(f"  overlay.png — bbox annotations")


# === Section 8: Filter by Confidence ===

def demo_filter():
    """Keep only high-confidence text regions."""
    print("\n=== Filter by confidence ===")
    if not _check_image(STOCKS_IMAGE):
        return

    result = mata.run("ocr", STOCKS_IMAGE, model="easyocr")
    high_conf = result.filter_by_score(0.85)
    very_high = result.filter_by_score(0.95)

    print(f"All regions:  {len(result.regions)}")
    print(f"Score ≥ 0.85: {len(high_conf.regions)}")
    print(f"Score ≥ 0.95: {len(very_high.regions)}")
    print(f"\nHigh-confidence text:\n{high_conf.full_text}")


# === Section 9: Config Alias ===

def demo_config_alias():
    """Load OCR models by a named alias from .mata/models.yaml.

    Example .mata/models.yaml entry:
        models:
          ocr:
            doc-scanner:
              source: "easyocr"
              languages: ["en"]
    """
    print("\n=== Config alias ===")
    print("  (requires .mata/models.yaml with 'doc-scanner' alias)")
    try:
        adapter = mata.load("ocr", "doc-scanner")
        if _check_image(STOCKS_IMAGE):
            result = adapter.predict(STOCKS_IMAGE)
            print(f"  {result.full_text[:120]!r}")
    except Exception as exc:
        print(f"  [skip] {exc}")


# ── entry point ───────────────────────────────────────────────────────────────

_DEMOS = {
    "easyocr":   demo_easyocr,
    "paddleocr": demo_paddleocr,
    "tesseract": demo_tesseract,
    "got-ocr2":  demo_got_ocr2,
    "trocr":     demo_trocr,
    "load-once": demo_load_once,
    "export":    demo_export,
    "filter":    demo_filter,
    "config":    demo_config_alias,
}

if __name__ == "__main__":
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if target == "all":
        for fn in _DEMOS.values():
            try:
                fn()
            except Exception as exc:
                print(f"  [error] {exc}")
    elif target in _DEMOS:
        _DEMOS[target]()
    else:
        print(f"Unknown demo {target!r}. Available: {', '.join(_DEMOS)}")
        sys.exit(1)
