#!/usr/bin/env python3
"""Retail: Stock Level Assessment — Multi-Modal Inventory Analysis.

Real-World Problem:
    Retail store managers and automated inventory systems need to assess
    shelf stock levels to trigger restocking, prevent out-of-stock events,
    and optimize supply chain operations. Purely counting products misses
    context like shelf capacity, product arrangement quality, and visual
    merchandising compliance.

Solution:
    Three-pronged parallel analysis provides comprehensive stock assessment:
    1. VLM (Vision-Language Model) gives semantic stock description
       (e.g., "Bottom shelf nearly empty", "Top shelf fully stocked")
    2. Object detector counts individual products
    3. CLIP classifier categorizes overall stock level

    Combining all three provides both quantitative (product count) and
    qualitative (VLM assessment, stock level category) insights.

Models:
    - VLM: Qwen/Qwen3-VL-2B-Instruct (semantic scene understanding)
    - Detector: facebook/detr-resnet-50 (product counting)
    - Classifier: openai/clip-vit-base-patch32 (stock level categorization)

Graph Flow:
    Parallel(VLMDescribe, Detect, Classify) → Filter → Fuse

Use Cases:
    - Automated restocking alerts
    - Inventory optimization
    - Planogram compliance verification
    - Supply chain demand forecasting
    - Store audit automation

Usage:
    python retail_stock_level.py                    # Mock mode
    python retail_stock_level.py --real shelf.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "shelf.jpg")

    if USE_REAL:
        import mata
        from mata.presets import stock_level_analysis

        vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")
        detector = mata.load("detect", "facebook/detr-resnet-50")
        classifier = mata.load("classify", "openai/clip-vit-base-patch32")

        result = mata.infer(
            image_path,
            stock_level_analysis(
                vlm_prompt=(
                    "Analyze the stock levels on each shelf. "
                    "Describe which areas are fully stocked, "
                    "which have low inventory, and note any empty spaces. "
                    "Comment on product arrangement and facing quality."
                ),
                detection_threshold=0.4,
                classification_labels=[
                    "fully_stocked",
                    "mostly_stocked",
                    "partially_stocked",
                    "low_stock",
                    "critical_low_stock",
                    "empty_shelf",
                ],
            ),
            providers={
                "vlm": vlm,
                "detector": detector,
                "classifier": classifier,
            },
        )

        print("=== Retail Stock Level Assessment Results ===")
        print()

        # VLM semantic assessment
        print("--- VLM Semantic Assessment ---")
        vlm_desc = result["final"].meta.get("vlm_description", "N/A")
        print(vlm_desc)
        print()

        # Product count from detector
        product_count = len(result["final"].instances)
        print(f"--- Product Count ---")
        print(f"Detected products: {product_count}")
        if product_count > 0:
            print(f"Average confidence: {sum(i.score for i in result['final'].instances) / product_count:.2f}")
        print()

        # Stock level classification
        print("--- Stock Level Category ---")
        if "classifications" in result["final"].meta:
            classifications = result["final"].meta["classifications"]
            if hasattr(classifications, "classifications"):
                top_class = classifications.classifications[0]
                print(f"Classification: {top_class.label} (confidence: {top_class.score:.2f})")
                print()
                print("Top 3 stock level predictions:")
                for i, c in enumerate(classifications.classifications[:3], 1):
                    print(f"  {i}. {c.label}: {c.score:.2f}")
            else:
                print("Classification data available but in unexpected format")
        else:
            print("No classification data in results")
        print()

        # Restocking recommendation
        print("--- Recommendation ---")
        if product_count < 10:
            print("⚠️  RESTOCK NEEDED — Low product count detected")
        elif product_count < 20:
            print("⚡ Monitor closely — Stock levels moderate")
        else:
            print("✓ Stock levels appear adequate")
        print()

    else:
        print("=== Retail: Stock Level Assessment (Mock Mode) ===")
        print()
        print("Real-World Problem:")
        print("  Retailers need comprehensive stock assessment that goes beyond")
        print("  simple product counting. Context matters: shelf capacity, product")
        print("  arrangement, visual merchandising compliance, and qualitative")
        print("  assessment of stock levels.")
        print()
        print("Solution:")
        print("  Three-pronged parallel analysis:")
        print("  1. VLM provides semantic stock description")
        print("     (e.g., 'Bottom shelf nearly empty, top shelf full')")
        print("  2. Object detector counts individual products")
        print("  3. CLIP categories stock level (fully/partially/low/empty)")
        print()
        print("Graph Flow:")
        print("  Parallel(VLMDescribe, Detect, Classify) → Filter → Fuse")
        print()
        print("Models:")
        print("  - VLM: Qwen/Qwen3-VL-2B-Instruct")
        print("  - Detector: facebook/detr-resnet-50")
        print("  - Classifier: openai/clip-vit-base-patch32")
        print()
        print("Expected output structure:")
        print("  result['final'].meta['vlm_description'] → semantic assessment")
        print("  result['final'].instances → detected products (count)")
        print("  result['final'].meta['classifications'] → stock level category")
        print()
        print("Benefits:")
        print("  • Quantitative: Exact product count from detector")
        print("  • Qualitative: Contextual assessment from VLM")
        print("  • Categorical: Stock level classification for automation")
        print()

        # Verify preset construction
        from mata.presets import stock_level_analysis
        graph = stock_level_analysis()
        print(f"✓ Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"  Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Key Feature:")
        print("  Parallel execution ensures fast multi-modal analysis — all three")
        print("  models run simultaneously instead of sequentially.")
        print()
        print("Run with --real <shelf_image.jpg> for actual inference.")
        print()


if __name__ == "__main__":
    main()
