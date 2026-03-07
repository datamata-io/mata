"""Basic Detection Examples — MATA Framework

Progressive examples from simplest one-liner to advanced patterns:
  1. One-Shot Detection    — mata.run() for quick single-image detection
  2. Load Once, Predict Many — reuse a detector across multiple images
  3. Switch Models         — DETR, RT-DETR, Conditional DETR
  4. Work with Results     — iterate instances, access bbox/score/label, filter
  5. Export Results        — JSON string, .json file, annotated image
  6. Config Aliases        — register named shortcuts with mata.register_model()

Run:
    python examples/detect/basic_detection.py
"""

from pathlib import Path

import mata

IMG = Path(__file__).parent.parent / "images" / "000000039769.jpg"


# === Section 1: One-Shot Detection ===


def section_one_shot():
    """Single call — no setup required."""
    result = mata.run("detect", str(IMG), model="facebook/detr-resnet-50", threshold=0.8)
    print(f"[one-shot] {len(result.detections)} detections")
    for det in result.detections[:3]:
        print(f"  {det.label_name:<20} {det.score:.1%}")


# === Section 2: Load Once, Predict Many ===


def section_load_reuse():
    """Load the detector once, then call .predict() for each image."""
    detector = mata.load("detect", "facebook/detr-resnet-50", threshold=0.8)
    images = [IMG, IMG]  # replace with your own list of paths
    for img in images:
        result = detector.predict(str(img))
        print(f"[load/reuse] {img.name}: {len(result.detections)} detections")


# === Section 3: Switch Models ===


def section_switch_models():
    """Load different architecture families from HuggingFace Hub."""
    models = {
        "DETR":         "facebook/detr-resnet-50",
        "RT-DETR":      "PekingU/rtdetr_r50vd",
        "Cond-DETR":    "microsoft/conditional-detr-resnet-50",
    }
    for name, model_id in models.items():
        detector = mata.load("detect", model_id, threshold=0.6)
        print(f"[switch] {name:<12} loaded {detector.__class__.__name__}")


# === Section 4: Work with Results ===


def section_results():
    """Access bbox, score, label; filter by confidence threshold."""
    result = mata.run("detect", str(IMG), model="facebook/detr-resnet-50", threshold=0.5)
    print(f"[results] {len(result.detections)} total detections")

    high_conf = [d for d in result.detections if d.score >= 0.9]
    print(f"[results] {len(high_conf)} above 90% confidence")

    for det in high_conf[:5]:
        x1, y1, x2, y2 = det.bbox
        print(f"  {det.label_name:<20} {det.score:.1%}  bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")


# === Section 5: Export Results ===


def section_export(output_dir: Path):
    """Save results as a JSON file and as an annotated image."""
    result = mata.run("detect", str(IMG), model="facebook/detr-resnet-50", threshold=0.8)

    # JSON string (e.g. for an API response)
    json_str = result.to_json()
    print(f"[export] JSON length: {len(json_str)} chars")

    # Save to .json file
    json_path = output_dir / "detections.json"
    result.save(str(json_path))
    print(f"[export] Saved JSON  to {json_path}")

    # Save annotated image (overlay bboxes on the source image)
    img_path = output_dir / "detections_overlay.jpg"
    result.save(str(img_path))
    print(f"[export] Saved image to {img_path}")


# === Section 6: Config Aliases ===


def section_config_aliases():
    """Register human-readable aliases and load by name."""
    mata.register_model("detect", "my-detr",   "facebook/detr-resnet-50", threshold=0.8)
    mata.register_model("detect", "my-rtdetr",  "PekingU/rtdetr_r50vd",   threshold=0.6)

    detector = mata.load("detect", "my-detr")
    print(f"[alias] Loaded 'my-detr' to {detector.__class__.__name__}")

    # Config-file aliases work the same way — set them in .mata/models.yaml
    # and load by name without calling register_model() in code.


def main():
    output_dir = Path("runs/detect_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Section 1: One-Shot Detection ===")
    section_one_shot()

    print("\n=== Section 2: Load Once, Predict Many ===")
    section_load_reuse()

    print("\n=== Section 3: Switch Models ===")
    section_switch_models()

    print("\n=== Section 4: Work with Results ===")
    section_results()

    print("\n=== Section 5: Export Results ===")
    section_export(output_dir)

    print("\n=== Section 6: Config Aliases ===")
    section_config_aliases()


if __name__ == "__main__":
    main()
