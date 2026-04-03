"""Multimodal Embedding & Retrieval with Qwen3-VL-Embedding.

Demonstrates text, image, video, and mixed-modal embeddings in a shared
vector space using Qwen3-VL-Embedding-2B (or 8B). All modalities are
embedded into the same dimensional space, enabling cross-modal similarity
search and retrieval.

Requirements:
    pip install datamata
    pip install datamata[qwen3-embedding]   # optional: enhanced preprocessing

    # GPU with at least 8 GB VRAM recommended for 2B model
    # GPU with at least 20 GB VRAM needed for 8B model

Model IDs:
    2B (recommended) : "Qwen/Qwen3-VL-Embedding-2B"    ~8 GB VRAM   512-dim default
    8B (high quality): "Qwen/Qwen3-VL-Embedding-8B"    ~20 GB VRAM  4096-dim default

Usage:
    # Basic run (uses examples/images/000000015338.jpg and examples/videos/cup.mp4)
    python examples/embed/qwen3_vl_multimodal.py

    # Specify your own image and video
    python examples/embed/qwen3_vl_multimodal.py --image path/to/photo.jpg --video path/to/clip.mp4

    # Use the 8B model (requires more VRAM)
    python examples/embed/qwen3_vl_multimodal.py --model Qwen/Qwen3-VL-Embedding-8B

    # Test Matryoshka dimension truncation
    python examples/embed/qwen3_vl_multimodal.py --embed-dim 256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure workspace src takes priority over any globally-installed mata
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mata
from mata.recognition import Gallery

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_IMAGE = str(Path(__file__).parent.parent / "images" / "000000015338.jpg")
_DEFAULT_VIDEO = str(Path(__file__).parent.parent / "videos" / "cup.mp4")
_DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two 1-D vectors."""
    a = a.ravel()
    b = b.ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Section 1 — Load adapter
# ---------------------------------------------------------------------------

def section_load_adapter(model_id: str, embed_dim: int | None, dtype: str) -> None:
    """Show how to load a Qwen3-VL-Embedding adapter with mata.load()."""
    print_section("1. Loading the Adapter")
    print(f"  Model   : {model_id}")
    print(f"  dtype   : {dtype}")
    if embed_dim:
        print(f"  embed_dim: {embed_dim}  (Matryoshka truncation enabled)")
    print()

    # Standard load — returns an EmbedAdapter wrapping Qwen3VLEmbeddingAdapter
    adapter = mata.load("embed", model_id, dtype=dtype,
                        **({"embed_dim": embed_dim} if embed_dim else {}))
    
    info = adapter.info()
    print(f"  Adapter type       : {info.get('encoder_type', info.get('type'))}")
    print(f"  qwen-vl-utils      : {info.get('qwen_vl_utils_available', 'n/a')}")
    print(f"  Native embed dim   : {info.get('native_dim', 'available after first call')}")
    return adapter


# ---------------------------------------------------------------------------
# Section 2 — Image embedding
# ---------------------------------------------------------------------------

def section_image_embedding(adapter, image_path: str) -> np.ndarray:
    """Embed a single image file."""
    print_section("2. Image Embedding")
    print(f"  Input: {image_path}")

    # Option A — via mata.run() (one-liner, loads a fresh adapter internally)
    #   result = mata.run("embed", image_path, model=model_id, dtype=dtype)

    # Option B — reuse an existing adapter (more efficient in loops)
    from mata.core.artifacts.image import Image as ImageArtifact
    img_artifact = ImageArtifact.from_path(image_path)
    result = adapter.embed(img_artifact)  # EmbedResult

    emb = result.embedding if hasattr(result, "embedding") else result
    if hasattr(emb, "embeddings"):
        emb = emb.embeddings[0]
    emb = np.array(emb).ravel()

    print(f"  Embedding shape : {emb.shape}")
    print(f"  dtype           : {emb.dtype}")
    print(f"  L2 norm         : {np.linalg.norm(emb):.6f}  (should be ~1.0)")
    return emb


# ---------------------------------------------------------------------------
# Section 3 — Text embedding
# ---------------------------------------------------------------------------

def section_text_embedding(model_id: str, dtype: str, embed_dim: int | None) -> dict[str, np.ndarray]:
    """Embed several text descriptions using mata.run()."""
    print_section("3. Text Embedding")

    queries = [
        "a dog playing on a beach",
        "a cat sitting on a sofa",
        "a red sports car on a highway",
        "people eating food at a restaurant",
    ]

    text_embeddings: dict[str, np.ndarray] = {}
    for q in queries:
        # Pass text= kwarg — signals text-only input path in api.run()
        result = mata.run(
            "embed",
            None,
            text=q,
            model=model_id,
            dtype=dtype,
            **({"embed_dim": embed_dim} if embed_dim else {}),
        )
        emb = np.array(result).ravel() if not hasattr(result, "embedding") else np.array(result.embedding).ravel()
        text_embeddings[q] = emb
        print(f"  [{emb.shape[0]}D] '{q}'  norm={np.linalg.norm(emb):.4f}")

    return text_embeddings


# ---------------------------------------------------------------------------
# Section 4 — Video embedding
# ---------------------------------------------------------------------------

def section_video_embedding(model_id: str, dtype: str, video_path: str,
                             embed_dim: int | None) -> np.ndarray:
    """Embed an entire short video as a single vector."""
    print_section("4. Video Embedding")
    print(f"  Input: {video_path}")

    # fps=1.0 → sample 1 frame per second (sufficient for concept-level retrieval)
    result = mata.run(
        "embed",
        video_path,
        model=model_id,
        dtype=dtype,
        fps=1.0,
        max_frames=32,
        **({"embed_dim": embed_dim} if embed_dim else {}),
    )
    emb = np.array(result).ravel() if not hasattr(result, "embedding") else np.array(result.embedding).ravel()

    print(f"  Embedding shape : {emb.shape}")
    print(f"  L2 norm         : {np.linalg.norm(emb):.6f}  (should be ~1.0)")
    return emb


# ---------------------------------------------------------------------------
# Section 5 — Mixed-modal embedding
# ---------------------------------------------------------------------------

def section_mixed_modal(model_id: str, dtype: str, image_path: str,
                         embed_dim: int | None) -> np.ndarray:
    """Embed an image together with a text instruction."""
    print_section("5. Mixed-Modal Embedding (Image + Text)")
    print(f"  Image : {image_path}")
    caption_prompt = "Describe what you see in this image in detail."
    print(f"  Text  : '{caption_prompt}'")

    # Passing both input= (image) and text= produces a mixed-modal embedding.
    # The model attends to both modalities before extracting the EOS vector.
    result = mata.run(
        "embed",
        image_path,
        text=caption_prompt,
        model=model_id,
        dtype=dtype,
        **({"embed_dim": embed_dim} if embed_dim else {}),
    )
    emb = np.array(result).ravel() if not hasattr(result, "embedding") else np.array(result.embedding).ravel()

    print(f"  Embedding shape : {emb.shape}")
    print(f"  L2 norm         : {np.linalg.norm(emb):.6f}")
    return emb


# ---------------------------------------------------------------------------
# Section 6 — Cosine similarity matrix
# ---------------------------------------------------------------------------

def section_similarity_matrix(image_emb: np.ndarray,
                                video_emb: np.ndarray,
                                mixed_emb: np.ndarray,
                                text_embeddings: dict[str, np.ndarray]) -> None:
    """Print a cross-modal cosine-similarity table."""
    print_section("6. Cross-Modal Cosine Similarity")

    # Abbreviate text keys so they fit in fixed-width columns
    text_keys = list(text_embeddings.keys())
    abbrevs = {k: f"T{i + 1}" for i, k in enumerate(text_keys)}

    modalities = {
        "Image":    image_emb,
        "Video":    video_emb,
        "Img+Text": mixed_emb,
        **{abbrevs[k]: v for k, v in text_embeddings.items()},
    }
    keys = list(modalities.keys())

    col_w = 10
    lbl_w = max(len(k) for k in keys) + 2

    header = " " * lbl_w + "".join(f"{k:>{col_w}}" for k in keys)
    sep    = "-" * len(header)
    print(f"\n{header}\n{sep}")

    for row_k in keys:
        row = f"  {row_k:<{lbl_w - 2}}"
        for col_k in keys:
            sim = cosine_similarity(modalities[row_k], modalities[col_k])
            row += f"{sim:>{col_w}.4f}"
        print(row)

    # Legend
    print("\n  Legend:")
    for k, abbrev in abbrevs.items():
        print(f"    {abbrev} = \"{k}\"")


# ---------------------------------------------------------------------------
# Section 7 — Gallery integration
# ---------------------------------------------------------------------------

def section_gallery(model_id: str, dtype: str, image_path: str,
                    embed_dim: int | None) -> None:
    """Build a small visual Gallery and search it with a text query."""
    print_section("7. Gallery Integration")

    # Build a gallery of labelled image embeddings
    gallery = Gallery(similarity_thresh=0.3)

    # Enroll reference images (in a real use case these would be distinct photos)
    reference_descriptions = {
        "dog_beach":       "a dog playing on a beach",
        "cat_sofa":        "a cat sitting on a sofa",
        "sample_image":    image_path,   # actual image from disk
    }

    print("  Enrolling references into gallery ...")
    for label, source in reference_descriptions.items():
        if source.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            # Image file
            result = mata.run("embed", source, model=model_id, dtype=dtype,
                              **({"embed_dim": embed_dim} if embed_dim else {}))
        else:
            # Text description used as a proxy embedding
            result = mata.run("embed", None, text=source, model=model_id, dtype=dtype,
                              **({"embed_dim": embed_dim} if embed_dim else {}))
        emb = np.array(result).ravel() if not hasattr(result, "embedding") else np.array(result.embedding).ravel()
        gallery.add(label, emb)
        print(f"    enrolled '{label}'  ({emb.shape[0]}D)")

    print(f"\n  Gallery size: {len(gallery)} entries")

    # Search with text queries
    print("\n  Searching gallery with text queries ...")
    search_queries = [
        "a dog at the shore",
        "a cat resting indoors",
        "a food scene",
    ]
    for query in search_queries:
        q_emb = mata.run("embed", None, text=query, model=model_id, dtype=dtype,
                         **({"embed_dim": embed_dim} if embed_dim else {}))
        q_vec = np.array(q_emb).ravel() if not hasattr(q_emb, "embedding") else np.array(q_emb.embedding).ravel()
        matches = gallery.search(q_vec, top_k=2)
        top = matches[0] if matches else None
        if top:
            print(f"    '{query}' → '{top.label}'  sim={top.similarity:.4f}")
        else:
            print(f"    '{query}' → no match above threshold")


# ---------------------------------------------------------------------------
# Section 8 — Matryoshka dimension comparison (optional)
# ---------------------------------------------------------------------------

def section_matryoshka(model_id: str, dtype: str, image_path: str) -> None:
    """Compare retrieval quality at different Matryoshka embedding dimensions."""
    print_section("8. Matryoshka Dimension Comparison (optional)")
    print("  Comparing full dim vs. truncated dims for an image–text pair.\n")

    text_query = "a dog playing on a beach"

    dims_to_test = [None, 512, 256, 128, 64]  # None → full native dim

    sims = []
    for dim in dims_to_test:
        label = f"dim={dim if dim else 'full'}"
        img_result = mata.run("embed", image_path, model=model_id, dtype=dtype,
                              **({"embed_dim": dim} if dim else {}))
        txt_result = mata.run("embed", None, text=text_query, model=model_id, dtype=dtype,
                              **({"embed_dim": dim} if dim else {}))

        img_emb = np.array(img_result).ravel() if not hasattr(img_result, "embedding") else np.array(img_result.embedding).ravel()
        txt_emb = np.array(txt_result).ravel() if not hasattr(txt_result, "embedding") else np.array(txt_result.embedding).ravel()

        sim = cosine_similarity(img_emb, txt_emb)
        actual_dim = img_emb.shape[0]
        sims.append((label, actual_dim, sim))
        print(f"  {label:<14}  actual_dim={actual_dim:>5}  cosine_sim={sim:.4f}")

    print(
        "\n  Note: Matryoshka embeddings are designed so that truncation retains\n"
        "  most of the retrieval quality. Smaller dims save memory and speed up\n"
        "  indexing with only a small accuracy cost."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multimodal embedding & retrieval with Qwen3-VL-Embedding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default=_DEFAULT_MODEL,
                   help=f"HuggingFace model ID (default: {_DEFAULT_MODEL}). "
                        "Use 'Qwen/Qwen3-VL-Embedding-8B' for higher quality (~20 GB VRAM).")
    p.add_argument("--image", default=_DEFAULT_IMAGE,
                   help="Path to a source image (JPEG/PNG).")
    p.add_argument("--video", default=_DEFAULT_VIDEO,
                   help="Path to a short video clip (mp4/avi/mov).")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Model weight dtype. bfloat16 recommended for CUDA (default).")
    p.add_argument("--embed-dim", type=int, default=None,
                   metavar="DIM",
                   help="Matryoshka truncation: output dimension. "
                        "Must be <= native model dim (2048 for 2B, 4096 for 8B). "
                        "None = full native dim.")
    p.add_argument("--skip-matryoshka", action="store_true",
                   help="Skip the Matryoshka comparison section (faster).")
    p.add_argument("--skip-video", action="store_true",
                   help="Skip video embedding (no OpenCV / no video file).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("\nQwen3-VL-Embedding — Multimodal Embedding & Retrieval Demo")
    print("=" * 60)
    print(f"  Model  : {args.model}")
    print(f"  Image  : {args.image}")
    if not args.skip_video:
        print(f"  Video  : {args.video}")
    print(f"  dtype  : {args.dtype}")
    if args.embed_dim:
        print(f"  Matryoshka embed_dim: {args.embed_dim}")
    print()
    print("  Note: First run downloads the model (~5-16 GB).")
    print("  Install enhanced preprocessing: pip install datamata[qwen3-embedding]")
    print()

    # 1 — Load adapter once
    adapter = section_load_adapter(args.model, args.embed_dim, args.dtype)

    # 2 — Image
    image_emb = section_image_embedding(adapter, args.image)

    # 3 — Text
    text_embeddings = section_text_embedding(args.model, args.dtype, args.embed_dim)

    # 4 — Video
    if not args.skip_video and Path(args.video).exists():
        video_emb = section_video_embedding(args.model, args.dtype, args.video, args.embed_dim)
    else:
        print_section("4. Video Embedding")
        print("  [skipped — use --video <path> to specify a video file]")
        video_emb = image_emb.copy()   # placeholder for similarity table

    # 5 — Mixed-modal
    mixed_emb = section_mixed_modal(args.model, args.dtype, args.image, args.embed_dim)

    # 6 — Similarity matrix
    section_similarity_matrix(image_emb, video_emb, mixed_emb, text_embeddings)

    # 7 — Gallery
    section_gallery(args.model, args.dtype, args.image, args.embed_dim)

    # 8 — Matryoshka (optional)
    if not args.skip_matryoshka:
        section_matryoshka(args.model, args.dtype, args.image)
    else:
        print_section("8. Matryoshka Dimension Comparison")
        print("  [skipped — remove --skip-matryoshka to enable]")

    print("\nDone.")


if __name__ == "__main__":
    main()
