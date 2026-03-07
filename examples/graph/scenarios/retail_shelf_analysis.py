#!/usr/bin/env python3
"""Retail: Shelf Product Analysis — Detection & Classification.

Real-World Problem:
    Retailers need automated shelf monitoring to track product placement,
    identify brands, detect out-of-stock items, and ensure planogram
    compliance. Manual shelf audits are time-consuming and error-prone.

Solution:
    Standard object detector (Faster R-CNN) finds all products on shelves,
    applies NMS to remove duplicate detections, extracts each product as
    an ROI crop, then uses CLIP zero-shot classification to identify the
    brand or category of each product.

Models:
    - Detector: torchvision/fasterrcnn_resnet50_fpn_v2 (production-ready)
    - Classifier: openai/clip-vit-base-patch32 (zero-shot brand matching)

Graph Flow:
    Detect > Filter > NMS > ExtractROIs > Classify > Fuse

Use Cases:
    - Planogram compliance verification
    - Out-of-stock detection
    - Brand placement audits
    - Competitor product tracking
    - Automated shelf replenishment

Usage:
    python retail_shelf_analysis.py                    # Mock mode
    python retail_shelf_analysis.py --real shelf.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "shelf.jpg")

    if USE_REAL:
        import mata
        from mata.presets import shelf_product_analysis

        detector = mata.load("detect", "torchvision/fasterrcnn_resnet50_fpn_v2")
        classifier = mata.load("classify", "openai/clip-vit-base-patch32")

        result = mata.infer(
            image_path,
            shelf_product_analysis(
                detection_threshold=0.5,
                nms_iou_threshold=0.5,
                classification_labels=[
                    "coca_cola",
                    "pepsi",
                    "sprite",
                    "fanta",
                    "mountain_dew",
                    "dr_pepper",
                    "energy_drink",
                    "water_bottle",
                    "juice",
                    "other_beverage",
                ],
                roi_padding=5,
            ),
            providers={"detector": detector, "classifier": classifier},
        )

        print(f"=== Retail Shelf Analysis Results ===")
        print(f"\nDetected {len(result['final'].instances)} products on shelf:")
        print()

        # Group by brand
        from collections import defaultdict
        brand_counts = defaultdict(int)

        for inst in result["final"].instances:
            brand_name = inst.label_name
            brand_counts[brand_name] += 1
            print(f"  • {brand_name}: confidence {inst.score:.2f} at {inst.bbox}")

        print(f"\n=== Brand Summary ===")
        for brand, count in sorted(brand_counts.items(), key=lambda x: -x[1]):
            print(f"  {brand}: {count} units")

        print(f"\nTotal unique brands: {len(brand_counts)}")

    else:
        print("=== Retail: Shelf Product Analysis (Mock Mode) ===")
        print()
        print("Real-World Problem:")
        print("  Retailers need automated shelf monitoring for product placement,")
        print("  brand identification, out-of-stock detection, and planogram")
        print("  compliance. Manual audits are time-consuming and error-prone.")
        print()
        print("Solution:")
        print("  1. Faster R-CNN detects all products on shelves")
        print("  2. NMS removes duplicate/overlapping detections")
        print("  3. Each product is cropped as an ROI")
        print("  4. CLIP classifies each crop into brand/category")
        print()
        print("Graph Flow:")
        print("  Detect > Filter > NMS > ExtractROIs > Classify > Fuse")
        print()
        print("Models:")
        print("  - Detector: torchvision/fasterrcnn_resnet50_fpn_v2")
        print("  - Classifier: openai/clip-vit-base-patch32")
        print()
        print("Expected output structure:")
        print("  result['final'].instances > list of product instances")
        print("    - Each instance has: bbox, label_name (brand), score")
        print("  result['rois'] > cropped images of each product")
        print("  result['classes'] > classification results per crop")
        print()

        # Verify preset construction
        from mata.presets import shelf_product_analysis
        graph = shelf_product_analysis()
        print(f"✓ Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"  Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <shelf_image.jpg> for actual inference.")
        print()


if __name__ == "__main__":
    main()
