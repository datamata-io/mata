"""
Torchvision CNN Detection — MATA Framework

Demonstrates using Apache 2.0-licensed CNN detectors from torchvision,
including RetinaNet, Faster R-CNN, and comparison with transformer models.

Models supported:
- RetinaNet (torchvision/retinanet_resnet50_fpn): Fast single-stage detector
- Faster R-CNN (torchvision/fasterrcnn_resnet50_fpn): Classic two-stage detector
- Faster R-CNN v2 (torchvision/fasterrcnn_resnet50_fpn_v2): Improved variant

Usage:
    python examples/detect/torchvision_detection.py
"""

from pathlib import Path

import mata


def main():
    # === Example 1: Quick detection with default model ===
    print("=== Example 1: RetinaNet Detection ===")
    result = mata.run(
        "detect",
        "examples/images/000000039769.jpg",
        model="torchvision/retinanet_resnet50_fpn",
        threshold=0.4,
    )
    print(f"Detected {len(result.instances)} objects")
    for inst in result.instances:
        print(f"  - {inst.label_name}: {inst.score:.2f}")

    # === Example 2: Load once, predict many ===
    print("\n=== Example 2: Batch Inference ===")
    detector = mata.load("detect", "torchvision/fasterrcnn_resnet50_fpn")

    images = list(Path("examples/images").glob("*.jpg"))
    for img_path in images:
        result = detector.predict(str(img_path), threshold=0.5)
        print(f"{img_path.name}: {len(result.instances)} detections")

    # === Example 3: Config alias ===
    print("\n=== Example 3: Config Alias ===")
    # Assumes ~/.mata/models.yaml defines a "cnn-fast" alias
    # See examples/tools/config_aliases.py for setup instructions
    try:
        detector = mata.load("detect", "cnn-fast")
        result = detector.predict("examples/images/000000039769.jpg")
        result.save("output/detections.json")
        result.save("output/visualized.jpg")
        print(f"Detected {len(result.instances)} objects via alias")
    except Exception as e:
        print(f"Skipping alias example (alias not configured): {e}")

    # === Example 4: Compare CNN vs transformer models ===
    print("\n=== Example 4: Model Comparison ===")
    models = [
        ("RetinaNet", "torchvision/retinanet_resnet50_fpn"),
        ("Faster R-CNN", "torchvision/fasterrcnn_resnet50_fpn_v2"),
        ("RT-DETR", "PekingU/rtdetr_r18vd"),
    ]

    for name, model_id in models:
        try:
            detector = mata.load("detect", model_id)
            result = detector.predict("examples/images/000000039769.jpg", threshold=0.4)
            print(f"{name}: {len(result.instances)} detections")
        except Exception as e:
            print(f"{name}: skipped ({e})")


if __name__ == "__main__":
    main()
