"""VLM (Vision-Language Model) Examples — MATA Framework

Core patterns for working with vision-language models:
  1. Basic image description
  2. Visual question answering (VQA)
  3. Custom system prompts for domain-specific tasks
  4. Load-once, predict many (batch efficiency)
  5. Accessing result metadata
  6. Structured output parsing
  7. Medical imaging with MedGemma (dtype='bfloat16')
  8. Lightweight VLM with LFM2.5 (dtype='bfloat16')
  9. Florence-2 grounding/captioning (community model)
  10. PaliGemma 2 document understanding (gated)

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
IMAGE_2 = IMAGE_DIR / "000000015338.jpg"

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


# === Section 7: Medical Imaging with MedGemma ===

def example_medgemma():
    """Use MedGemma for medical image analysis (requires dtype='bfloat16')."""
    print("\n=== 7. Medical Imaging with MedGemma ===")

    vlm = mata.load("vlm", "google/medgemma-1.5-4b-it", dtype="bfloat16")
    # Medical imaging test image (chest X-ray, CC0)
    XRAY_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    result = vlm.predict(XRAY_URL, prompt="Describe this X-ray image.")
    print(f"Response:\n{result.text}")


# === Section 8: Lightweight VLM with LFM2.5 ===

def example_lfm2():
    """Use LFM2.5-VL for lightweight inference (requires dtype='bfloat16')."""
    print("\n=== 8. Lightweight VLM with LFM2.5 ===")

    vlm = mata.load("vlm", "LiquidAI/LFM2.5-VL-1.6B", dtype="bfloat16")
    result = vlm.predict(str(IMAGE_1), prompt="What is in this image?")
    print(f"Response:\n{result.text}")


# === Section 9: Florence-2 (Grounding / Captioning) ===

def example_florence2():
    """Load Florence-2 for dense captioning and grounding.

    The canonical loading ID is florence-community/Florence-2-large, which is the
    official transformers-5.x port.  MATA transparently redirects the legacy
    microsoft/Florence-2-large ID to the community model automatically, so both
    forms work — but loading the community ID directly avoids the redirect overhead.
    """
    print("\n=== 9. Florence-2 (Grounding / Captioning) ===")

    # Direct community model (recommended — no trust_remote_code required)
    vlm = mata.load("vlm", "florence-community/Florence-2-large")
    result = vlm.predict(str(IMAGE_1), prompt="Describe this image in detail.")
    print(f"Response:\n{result.text}")

    # The legacy microsoft/Florence-2-large ID also works — MATA auto-redirects it
    # vlm = mata.load("vlm", "microsoft/Florence-2-large", trust_remote_code=True)


# === Section 10: PaliGemma 2 (Document Understanding) ===

def example_paligemma2():
    """Use PaliGemma 2 for document understanding and fine-grained recognition.

    PaliGemma models are gated on HuggingFace — run `huggingface-cli login` first.
    """
    print("\n=== 10. PaliGemma 2 (Document Understanding) ===")

    vlm = mata.load("vlm", "google/paligemma2-3b-pt-224", dtype="bfloat16")
    result = vlm.predict(str(IMAGE_1), prompt="Describe this image.")
    print(f"Response:\n{result.text}")


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
        example_medgemma,
        example_lfm2,
        example_florence2,
        example_paligemma2,
    ]:
        try:
            fn()
        except Exception as exc:
            print(f"  [error] {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
