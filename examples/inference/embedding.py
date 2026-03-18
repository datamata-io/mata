"""Feature Embedding Example — mata v1.9.2b2

Demonstrates whole-image embedding extraction and graph pipeline usage.
Runs in mock mode (no real model downloads required).
"""

from __future__ import annotations

import sys
import os

# Ensure the workspace src is first on sys.path so the correct
# mata installation is loaded (not any globally-installed version).
_WORKSPACE_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _WORKSPACE_SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_WORKSPACE_SRC))

from unittest.mock import MagicMock

import numpy as np


# ---------------------------------------------------------------------------
# 1. One-liner whole-image embedding (real usage)
# ---------------------------------------------------------------------------

def example_run_embed_real():
    """mata.run("embed", ...) — real usage (requires model download)."""
    # import mata
    # emb = mata.run("embed", "photo.jpg", model="openai/clip-vit-base-patch32")
    # print(f"shape={emb.shape}, dtype={emb.dtype}")   # (1, 512) float32
    pass


# ---------------------------------------------------------------------------
# 2. EmbedAdapter mock demo
# ---------------------------------------------------------------------------

def example_embed_adapter_mock():
    """Simulate adapter usage with a mock encoder."""
    from mata.adapters.embed_adapter import EmbedAdapter
    from mata.core.artifacts.image import Image

    # Mock inner encoder — no model download
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = np.random.randn(1, 512).astype(np.float32)
    mock_encoder.embedding_dim = 512
    mock_encoder.info.return_value = {"model": "clip", "dim": 512}

    adapter = EmbedAdapter(encoder=mock_encoder)
    print(f"[adapter] type={type(adapter).__name__}")
    print(f"[adapter] embedding_dim={adapter.embedding_dim}")
    info = adapter.info()
    print(f"[adapter] info type={info['type']}")  # embed

    # Real one-liner usage:
    # import mata
    # adapter = mata.load("embed", "openai/clip-vit-base-patch32")
    # adapter = mata.load("embed", "./osnet_x0_25.onnx")  # ONNX
    print("[adapter] OK")


# ---------------------------------------------------------------------------
# 3. Embeddings artifact — direct construction
# ---------------------------------------------------------------------------

def example_embeddings_artifact():
    """Embeddings artifact — creation and access."""
    from mata.core.artifacts.embeddings import Embeddings

    embs = Embeddings(
        vectors=np.random.randn(5, 512).astype(np.float32),
        normalized=True,
        meta={"model": "clip"},
    )

    print(f"[artifact] len={len(embs)}, dim={embs.embedding_dim}")
    print(f"[artifact] ids={embs.instance_ids[:2]}...")
    print(f"[artifact] first vector norm: {np.linalg.norm(embs[0]):.4f}")

    # 1-D input auto-reshaped to (1, D)
    single = Embeddings(vectors=np.random.randn(512).astype(np.float32))
    assert single.vectors.shape == (1, 512)
    print(f"[artifact] 1-D reshape: {single.vectors.shape}")


# ---------------------------------------------------------------------------
# 4. Graph pipeline: Detect → ExtractROIs → Embed (simulated)
# ---------------------------------------------------------------------------

def example_graph_pipeline():
    """Simulate the graph pipeline result."""
    from mata.core.artifacts.embeddings import Embeddings

    # Simulate graph output artifact
    n_detections = 3
    fake_embeddings = Embeddings(
        vectors=np.random.randn(n_detections, 512).astype(np.float32),
        instance_ids=tuple(f"det_{i:04d}" for i in range(n_detections)),
        normalized=True,
        meta={"model": "clip"},
    )

    print(f"[graph] embeddings.vectors.shape = {fake_embeddings.vectors.shape}")
    print(f"[graph] instance_ids = {fake_embeddings.instance_ids}")
    for i, iid in enumerate(fake_embeddings.instance_ids):
        vec = fake_embeddings[i]
        print(f"  [{iid}] shape={vec.shape}")

    # Real graph usage (requires model downloads):
    #
    # import mata
    # from mata.core.graph import Graph
    # from mata.nodes import Detect, Filter, ExtractROIs, Embed
    #
    # graph = (
    #     Graph("embed_pipeline")
    #     .then(Detect(using="detector", out="dets"))
    #     .then(Filter(src="dets", score_gt=0.5, out="filtered"))
    #     .then(ExtractROIs(src_dets="filtered", out="rois"))
    #     .then(Embed(using="encoder", src="rois", out="embeddings"))
    # )
    #
    # result = mata.infer(graph, image="photo.jpg", providers={
    #     "detector": mata.load("detect", "facebook/detr-resnet-50"),
    #     "encoder":  mata.load("embed", "openai/clip-vit-base-patch32"),
    # })
    # embs = result["embeddings"]
    # print(embs.vectors.shape)   # (N, 512)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Feature Embedding Examples (mock mode) ===\n")

    print("--- 1. EmbedAdapter mock ---")
    example_embed_adapter_mock()

    print("\n--- 2. Embeddings artifact ---")
    example_embeddings_artifact()

    print("\n--- 3. Graph pipeline (simulated) ---")
    example_graph_pipeline()

    print("\nAll examples completed successfully.")

