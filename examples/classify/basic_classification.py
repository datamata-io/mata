"""Basic Classification Examples — MATA Framework

Progressive examples: one-shot → load/reuse → model comparison → filtering.
Run: python examples/classify/basic_classification.py
"""

from pathlib import Path

import mata

IMAGE = Path("examples/images/000000039769.jpg")
FALLBACK = "https://images.unsplash.com/photo-1574158622682-e40e69881006"


def get_image() -> str:
    return str(IMAGE) if IMAGE.exists() else FALLBACK


# === Section 1: One-Shot Classification ===
def one_shot():
    print("\n=== Section 1: One-Shot Classification ===")
    result = mata.run("classify", get_image(), model="microsoft/resnet-50")
    top1 = result.get_top1()
    print(f"Top prediction: {top1.label_name} ({top1.score * 100:.2f}%)")


# === Section 2: Load Once, Classify Many ===
def load_and_reuse():
    print("\n=== Section 2: Load Once, Classify Many ===")
    classifier = mata.load("classify", "microsoft/resnet-50", top_k=5)
    for _ in range(2):
        result = classifier.predict(get_image())
        top1 = result.get_top1()
        print(f"  → {top1.label_name}: {top1.score * 100:.2f}%")


# === Section 3: Access Results (.get_top1, top-5 predictions) ===
def access_results():
    print("\n=== Section 3: Access Results ===")
    classifier = mata.load("classify", "microsoft/resnet-50", top_k=5)
    result = classifier.predict(get_image())

    top1 = result.get_top1()
    print(f"Top-1: {top1.label_name} ({top1.score * 100:.2f}%)")

    print("Top-5:")
    for i, pred in enumerate(result.predictions[:5], 1):
        print(f"  {i}. {pred.label_name}: {pred.score * 100:.2f}%")


# === Section 4: Model Comparison (ResNet vs ViT) ===
def compare_models():
    print("\n=== Section 4: Model Comparison ===")
    models = [
        ("ResNet-50", "microsoft/resnet-50"),
        ("ViT-Base", "google/vit-base-patch16-224"),
    ]
    for name, model_id in models:
        try:
            classifier = mata.load("classify", model_id, top_k=1)
            result = classifier.predict(get_image())
            top1 = result.get_top1()
            print(f"  {name}: {top1.label_name} ({top1.score * 100:.2f}%)")
        except Exception as e:
            print(f"  {name}: failed — {e}")


# === Section 5: Confidence Filtering ===
def confidence_filtering():
    print("\n=== Section 5: Confidence Filtering ===")
    classifier = mata.load("classify", "microsoft/resnet-50", top_k=10)
    result = classifier.predict(get_image())

    print(f"All predictions: {len(result.predictions)}")
    high_conf = result.filter_by_score(0.05)
    print(f"Above 5% confidence: {len(high_conf.predictions)}")
    for pred in high_conf.predictions:
        print(f"  {pred.label_name}: {pred.score * 100:.2f}%")


def main():
    print("=" * 50)
    print("MATA Basic Classification Examples")
    print("=" * 50)

    for fn in [one_shot, load_and_reuse, access_results, compare_models, confidence_filtering]:
        try:
            fn()
        except Exception as e:
            print(f"  [skipped] {fn.__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
