"""VLM (Vision-Language Model) Examples — MATA Framework

Core patterns for working with vision-language models:
  1. Basic image description
  2. Visual question answering (VQA)
  3. Custom system prompts for domain-specific tasks
  4. Load-once, predict many (batch efficiency)
  5. Accessing result metadata
  6. Structured output parsing

Run:
  python examples/vlm/basic_vlm.py

Requirements:
  pip install transformers torch
"""

from pathlib import Path

import mata

# ── paths ─────────────────────────────────────────────────────────────────────
IMAGE_DIR = Path(__file__).parent.parent / "images"
IMAGE_1 = IMAGE_DIR / "000000039769.jpg"
IMAGE_2 = IMAGE_DIR / "hanvin-cheong-tuR2XRPdtYI-unsplash.jpg"

MODEL = "Qwen/Qwen3-VL-2B-Instruct"


# === Section 1: Basic Image Description ===

def example_basic_description():
    """The simplest use case — describe what's in an image."""
    print("\n=== 1. Basic Image Description ===")

    result = mata.run(
        "vlm",
        str(IMAGE_1),
        model=MODEL,
        prompt="Describe this image in detail.",
        max_new_tokens=300,
    )

    print(f"Response:\n{result.text}")


# === Section 2: Visual Question Answering (VQA) ===

def example_vqa():
    """Ask specific questions about image content."""
    print("\n=== 2. Visual Question Answering ===")

    questions = [
        "How many cats are in this image?",
        "What color is the remote control?",
        "What are the cats doing?",
    ]

    for question in questions:
        result = mata.run(
            "vlm",
            str(IMAGE_1),
            model=MODEL,
            prompt=question,
            max_new_tokens=150,
        )
        print(f"Q: {question}")
        print(f"A: {result.text}\n")


# === Section 3: Custom System Prompts ===

def example_system_prompts():
    """Use system prompts to guide model behaviour for domain-specific tasks."""
    print("\n=== 3. Custom System Prompts ===")

    system_prompt = (
        "You are a veterinary assistant AI. "
        "Analyze images for pet health and behaviour. "
        "Be observant and note any unusual signs."
    )

    result = mata.run(
        "vlm",
        str(IMAGE_1),
        model=MODEL,
        prompt="Analyze the health and condition of the animals in this image.",
        system_prompt=system_prompt,
        max_new_tokens=300,
    )

    print(f"System prompt: {system_prompt}")
    print(f"Response:\n{result.text}")


# === Section 4: Load Once, Predict Many ===

def example_load_once():
    """Load the model once and reuse it for efficient batch processing."""
    print("\n=== 4. Load-Once, Predict Many ===")

    vlm = mata.load("vlm", MODEL)

    images = [p for p in [IMAGE_1, IMAGE_2] if p.exists()]
    prompt = "What is in this image? Describe in one sentence."

    for img_path in images:
        result = vlm.predict(str(img_path), prompt=prompt, max_new_tokens=150)
        print(f"  {img_path.name}: {result.text}")


# === Section 5: Accessing Result Metadata ===

def example_metadata():
    """VLMResult objects expose rich inference metadata."""
    print("\n=== 5. Accessing Metadata ===")

    result = mata.run(
        "vlm",
        str(IMAGE_1),
        model=MODEL,
        prompt="What objects can you see in this image?",
        max_new_tokens=200,
    )

    print(f"Response: {result.text}")
    print("\nMetadata:")
    for key in ("model_id", "device", "backend", "max_new_tokens", "tokens_generated"):
        print(f"  {key}: {result.meta.get(key)}")


# === Section 6: Structured Output Parsing ===

def example_structured_output():
    """Request JSON output and parse it into Entity objects (v1.5.4+)."""
    print("\n=== 6. Structured Output Parsing ===")

    result = mata.run(
        "vlm",
        str(IMAGE_1),
        model=MODEL,
        prompt="List all objects you can identify in this image.",
        output_mode="detect",
        max_new_tokens=300,
    )

    print(f"Raw response:\n{result.text}")
    print(f"\nParsed entities: {len(result.entities)}")

    if result.entities:
        for entity in result.entities[:5]:
            print(f"  [{entity.label}] score={entity.score:.2f}")
    else:
        print("  (No entities parsed — graceful fallback to raw text.)")


def main():
    print("MATA — VLM Examples")
    print("=" * 40)

    if not IMAGE_1.exists():
        print(f"[warn] Test image not found: {IMAGE_1}")
        print("       Place an image at examples/images/000000039769.jpg to run examples.")
        return

    for fn in [
        example_basic_description,
        example_vqa,
        example_system_prompts,
        example_load_once,
        example_metadata,
        example_structured_output,
    ]:
        try:
            fn()
        except Exception as exc:
            print(f"  [error] {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
