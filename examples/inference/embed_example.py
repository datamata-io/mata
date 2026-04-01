#!/usr/bin/env python3
"""Embedding Extraction Example — mata embed task.

Demonstrates:
    1. Single image embedding with mata.run()
    2. Batch crop embeddings with mata.run()
    3. Pre-loaded adapter with mata.load() for reuse across many images
    4. Saving and loading EmbedResult to/from disk

Runs in mock mode by default (no real model downloads required).
Use --real to load models from HuggingFace.

Usage:
    python examples/inference/embed_example.py
    python examples/inference/embed_example.py --real
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure workspace src takes priority over any globally-installed mata
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Mock adapter — no model download required
# ---------------------------------------------------------------------------

def _make_mock_adapter(dim: int = 512):
    """Return a mock EmbedAdapter for demonstration."""
    from unittest.mock import MagicMock

    from mata.core.types import EmbedResult

    mock_encoder = MagicMock()
    mock_encoder.embedding_dim = dim

    def _predict(input_, **kw):
        # Return a (1, dim) result for Image inputs, or (N, dim) for lists
        n = 1
        if hasattr(input_, "__len__") and not isinstance(input_, np.ndarray):
            n = len(input_)
        vecs = np.random.randn(n, dim).astype(np.float32)
        # L2-normalise
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-8)
        return EmbedResult(embeddings=vecs)

    mock_encoder.predict.side_effect = _predict

    from mata.adapters.embed_adapter import EmbedAdapter
    return EmbedAdapter(encoder=mock_encoder)


# ---------------------------------------------------------------------------
# Example 1: Single image embedding — mata.run("embed", ...)
# ---------------------------------------------------------------------------

def example_single_image(use_real: bool = False):
    """Extract a single embedding from an image."""
    print("\n--- 1. Single image embedding ---")

    if use_real:
        import mata
        from mata.core.types import EmbedResult
        _raw = mata.run("embed", "examples/images/000000039769.jpg",
                        model="openai/clip-vit-base-patch32")  # returns np.ndarray
        result = EmbedResult(embeddings=_raw)
    else:
        from mata.core.types import EmbedResult

        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        result = EmbedResult(embeddings=vec.reshape(1, -1))

    print(f"  result type   : {type(result).__name__}")
    print(f"  embeddings    : shape={result.embeddings.shape}, dtype={result.embeddings.dtype}")
    print(f"  .embedding    : shape={result.embedding.shape}  (first row convenience)")
    print(f"  .dim          : {result.dim}")
    norm = float(np.linalg.norm(result.embedding))
    print(f"  L2 norm       : {norm:.6f}  (should be ~1.0 — L2-normalised)")


# ---------------------------------------------------------------------------
# Example 2: Batch crops — mata.run("embed", list_of_arrays, ...)
# ---------------------------------------------------------------------------

def example_batch_crops(use_real: bool = False):
    """Extract embeddings from a batch of numpy crops."""
    print("\n--- 2. Batch crop embeddings ---")

    crops = [np.random.randint(0, 255, (64, 32, 3), dtype=np.uint8) for _ in range(5)]

    if use_real:
        import mata
        result = mata.run("embed", crops, model="openai/clip-vit-base-patch32")
    else:
        from mata.core.types import EmbedResult

        vecs = np.random.randn(len(crops), 512).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= np.maximum(norms, 1e-8)
        result = EmbedResult(embeddings=vecs)

    print(f"  input crops   : {len(crops)} arrays of shape (64, 32, 3)")
    print(f"  embeddings    : shape={result.embeddings.shape}")   # (5, 512)
    print(f"  .dim          : {result.dim}")
    norms = np.linalg.norm(result.embeddings, axis=1)
    print(f"  norms         : min={norms.min():.4f}, max={norms.max():.4f}  (all ~1.0)")


# ---------------------------------------------------------------------------
# Example 3: Pre-loaded adapter — mata.load("embed", ...)
# ---------------------------------------------------------------------------

def example_preloaded_adapter(use_real: bool = False):
    """Load an adapter once and call predict() many times."""
    print("\n--- 3. Pre-loaded adapter ---")

    if use_real:
        import mata
        embedder = mata.load("embed", "openai/clip-vit-base-patch32")
    else:
        embedder = _make_mock_adapter()

    image_paths = ["photo_1.jpg", "photo_2.jpg", "photo_3.jpg"]
    all_embeddings = []

    for path in image_paths:
        if use_real:
            from mata.core.artifacts.image import Image as ImageArtifact
            from mata.core.types import EmbedResult
            _raw = embedder.embed(ImageArtifact.from_path(path))  # (1, D) ndarray
            result = EmbedResult(embeddings=_raw)
        else:
            from mata.core.types import EmbedResult
            vec = np.random.randn(512).astype(np.float32)
            vec /= np.linalg.norm(vec)
            result = EmbedResult(embeddings=vec.reshape(1, -1))
        all_embeddings.append(result.embedding)
        print(f"  {path}: shape={result.embedding.shape}, norm={np.linalg.norm(result.embedding):.4f}")

    # Stack into (N, D) matrix for downstream use
    matrix = np.stack(all_embeddings)
    print(f"  stacked matrix: {matrix.shape}")


# ---------------------------------------------------------------------------
# Example 4: Save and load EmbedResult
# ---------------------------------------------------------------------------

def example_save_load(tmp_dir: Path):
    """Serialize EmbedResult to JSON and NPZ."""
    print("\n--- 4. Save / load EmbedResult ---")

    from mata.core.types import EmbedResult

    vecs = np.random.randn(3, 512).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.maximum(norms, 1e-8)
    result = EmbedResult(embeddings=vecs, labels=["alice", "bob", "carol"])

    # Save as JSON
    json_path = tmp_dir / "embeddings.json"
    result.save(str(json_path))
    print(f"  Saved JSON  : {json_path}")

    # Save as NPZ
    npz_path = tmp_dir / "embeddings.npz"
    result.save(str(npz_path))
    print(f"  Saved NPZ   : {npz_path}")

    # Round-trip from JSON
    loaded = EmbedResult.from_json(json_path.read_text())
    print(f"  Loaded JSON : shape={loaded.embeddings.shape}, labels={loaded.labels}")

    # Round-trip from NPZ
    data = np.load(str(npz_path), allow_pickle=False)
    print(f"  Loaded NPZ  : keys={list(data.keys())}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    use_real = "--real" in sys.argv

    if use_real:
        print("=== Embedding Extraction Examples (real model mode) ===")
    else:
        print("=== Embedding Extraction Examples (mock mode — no downloads) ===")
        print("    Pass --real to use actual HuggingFace models.\n")

    example_single_image(use_real)
    example_batch_crops(use_real)
    example_preloaded_adapter(use_real)
    example_save_load(Path("."))

    print("\nAll examples completed successfully.")
