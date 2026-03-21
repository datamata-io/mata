"""GGUF Embedding Example — MATA Framework

Demonstrates extracting L2-normalized feature embeddings from quantized CLIP
GGUF files via llama-cpp-python. Suitable for similarity search, image
retrieval, and zero-shot classification pipelines on CPU-constrained hardware.

Download a CLIP GGUF from HuggingFace Hub, for example:
  - CLIP ViT-B/32 Q8: https://huggingface.co/mys/ggml_clip
    -> Download: ggml-model-q8_0.gguf

Requirements:
  pip install datamata[gguf]

Run:
  python examples/inference/gguf_embed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import mata

# ---------------------------------------------------------------------------
# Configuration — update these paths to point to your downloaded GGUF files
# ---------------------------------------------------------------------------

CLIP_GGUF_PATH = "clip.gguf"    # Path to CLIP GGUF model

IMAGE_PATH = str(Path(__file__).parent.parent / "images" / "000000039769.jpg")


def check_prerequisites():
    """Verify model file exists before proceeding."""
    if not Path(CLIP_GGUF_PATH).exists():
        print(f"Error: GGUF model not found at '{CLIP_GGUF_PATH}'.")
        print("Download a CLIP GGUF file first — see the docstring at the top.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 1. One-liner embedding
# ---------------------------------------------------------------------------

def example_run_embed():
    """mata.run() — simplest usage, returns (N, D) ndarray."""
    print("\n=== 1. One-liner Embedding ===")

    embeddings = mata.run("embed", IMAGE_PATH, model=CLIP_GGUF_PATH)
    print(f"Shape:  {embeddings.shape}")     # (1, D)
    print(f"Dtype:  {embeddings.dtype}")     # float32
    print(f"Norm:   {np.linalg.norm(embeddings[0]):.6f}")  # ~1.0 (L2 normalised)


# ---------------------------------------------------------------------------
# 2. Load once, embed many images
# ---------------------------------------------------------------------------

def example_batch_embed():
    """Load the encoder once and embed several images."""
    print("\n=== 2. Batch Embedding ===")

    # CPU-only; for GPU: n_gpu_layers=-1
    encoder = mata.load("embed", CLIP_GGUF_PATH, n_gpu_layers=0)

    image_paths = [IMAGE_PATH, IMAGE_PATH]   # Replace with your own images
    all_embeddings = []
    for path in image_paths:
        emb = encoder.predict(path)           # (1, D)
        all_embeddings.append(emb)

    stacked = np.vstack(all_embeddings)       # (N, D)
    print(f"Stacked shape: {stacked.shape}")


# ---------------------------------------------------------------------------
# 3. Cosine similarity between two images
# ---------------------------------------------------------------------------

def example_cosine_similarity():
    """Compute cosine similarity between two image embeddings."""
    print("\n=== 3. Cosine Similarity ===")

    encoder = mata.load("embed", CLIP_GGUF_PATH)

    emb_a = encoder.predict(IMAGE_PATH)[0]    # (D,)
    emb_b = encoder.predict(IMAGE_PATH)[0]    # (D,) — same image → similarity=1.0

    similarity = float(np.dot(emb_a, emb_b))  # L2-normalised → dot = cosine sim
    print(f"Cosine similarity (same image): {similarity:.4f}")   # ~1.0


# ---------------------------------------------------------------------------
# 4. GGUF embed inside a graph pipeline
# ---------------------------------------------------------------------------

def example_graph_pipeline():
    """GGUF embed encoder as a drop-in provider in a graph pipeline."""
    print("\n=== 4. Graph Pipeline ===")

    # Real usage (requires all models to be available):
    #
    # from mata.core.graph import Graph
    # from mata.nodes import Detect, Filter, ExtractROIs, Embed
    #
    # graph = (
    #     Graph("gguf_embed_pipeline")
    #     .then(Detect(using="detector", out="dets"))
    #     .then(Filter(src="dets", score_gt=0.5, out="filtered"))
    #     .then(ExtractROIs(src_dets="filtered", out="rois"))
    #     .then(Embed(using="encoder", src="rois", out="embeddings"))
    # )
    #
    # result = mata.infer(graph, image="photo.jpg", providers={
    #     "detector": mata.load("detect", "facebook/detr-resnet-50"),
    #     "encoder":  mata.load("embed", CLIP_GGUF_PATH),   # ← GGUF encoder
    # })
    # embs = result["embeddings"]
    # print(embs.vectors.shape)   # (N, D)

    print("  (See commented code above for graph pipeline usage)")


if __name__ == "__main__":
    check_prerequisites()
    example_run_embed()
    example_batch_embed()
    example_cosine_similarity()
    example_graph_pipeline()
