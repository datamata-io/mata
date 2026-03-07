#!/usr/bin/env python3
"""Retail: Product Search & Segmentation — Zero-Shot Detection.

Real-World Problem:
    Retailers need to quickly locate specific products in store photos
    or warehouse images. Traditional methods require training custom models
    for each product type. Natural language search would enable flexible,
    on-demand product finding without retraining.

Solution:
    GroundingDINO enables zero-shot text-prompted detection — just describe
    the product in plain English (e.g., "red can", "blue bottle") and it
    finds matching items. SAM then segments each detected product for
    precise boundary extraction, useful for calculating shelf space or
    picking robots.

Models:
    - Detector: IDEA-Research/grounding-dino-tiny (zero-shot text prompts)
    - Segmenter: facebook/sam-vit-base (segment-anything)

Graph Flow:
    Detect(text_prompts) > Filter > PromptBoxes(SAM) > RefineMask > Fuse

Use Cases:
    - Visual product search in inventory photos
    - Automated picking guidance for warehouse robots
    - Shelf space allocation measurement
    - Promotional display verification
    - Competitor product monitoring

Demonstrates:
    Reusing the `grounding_dino_sam()` preset with retail-specific text
    prompts — no new preset needed, just customize the text prompts at
    inference time.

Usage:
    python retail_product_search.py                      # Mock mode
    python retail_product_search.py --real shelf.jpg     # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "shelf.jpg")

    if USE_REAL:
        import mata
        from mata.presets import grounding_dino_sam

        detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
        segmenter = mata.load("segment", "facebook/sam-vit-base")

        # Custom retail product search prompts
        search_prompts = "red can . blue bottle . cereal box . milk carton . orange juice"

        result = mata.infer(
            image_path,
            grounding_dino_sam(
                detection_threshold=0.25,
                nms_iou_threshold=0.5,
                refine_method="morph_close",
                refine_radius=3,
            ),
            providers={
                "detector": detector,
                "segmenter": segmenter,
            },
            # Text prompts passed at inference time
            text_prompts=search_prompts,
        )

        print(f"=== Retail Product Search Results ===")
        print(f"Search query: \"{search_prompts}\"")
        print()
        print(f"Found {len(result['final'].instances)} matching products:")
        print()

        for i, inst in enumerate(result["final"].instances, 1):
            has_mask = inst.mask is not None
            mask_pixels = inst.mask.sum() if has_mask else 0
            print(f"{i}. {inst.label_name}")
            print(f"   Confidence: {inst.score:.2f}")
            print(f"   Location: {inst.bbox}")
            if has_mask:
                print(f"   Mask area: {mask_pixels} pixels")
            print()

        print("Tip: Use result['final'].instances[i].mask for shelf space calculation")

    else:
        print("=== Retail: Product Search & Segmentation (Mock Mode) ===")
        print()
        print("Real-World Problem:")
        print("  Find specific products in store photos using natural language")
        print("  descriptions without training custom detection models.")
        print()
        print("Solution:")
        print("  GroundingDINO accepts text prompts like 'red can . blue bottle'")
        print("  and detects matching products. SAM then segments each product")
        print("  for precise boundary extraction.")
        print()
        print("Graph Flow:")
        print("  Detect(text_prompts) > Filter > PromptBoxes(SAM) > RefineMask > Fuse")
        print()
        print("Models:")
        print("  - Detector: IDEA-Research/grounding-dino-tiny (zero-shot)")
        print("  - Segmenter: facebook/sam-vit-base")
        print()
        print("Example search prompts:")
        print('  "red can . blue bottle . cereal box . milk carton"')
        print('  "coca_cola . pepsi . sprite . energy_drink"')
        print('  "organic product . gluten_free label . sale tag"')
        print()
        print("Expected output structure:")
        print("  result['final'].instances > list of detected products")
        print("    - Each instance has: bbox, mask (segmentation), score, label_name")
        print("  Mask area can be used for shelf space allocation calculations")
        print()

        # Verify preset construction
        from mata.presets import grounding_dino_sam
        graph = grounding_dino_sam()
        print(f"✓ Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"  Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Key Insight:")
        print("  This example reuses the `grounding_dino_sam()` preset — no new")
        print("  preset needed! Just customize the text_prompts at inference time.")
        print()
        print("Run with --real <shelf_image.jpg> for actual inference.")
        print()


if __name__ == "__main__":
    main()
