#!/usr/bin/env python3
"""Gallery Creation and Matching Example — mata recognition.

Demonstrates:
    1. Building a Gallery from reference embeddings
    2. Searching a gallery with cosine similarity
    3. One-liner recognition with mata.run("recognize", ...)
    4. Gallery persistence: save and load from .npz
    5. Batch search and threshold filtering

Runs in mock mode by default (no real model downloads required).
Use --real to load models from HuggingFace.

Usage:
    python examples/inference/gallery_match_example.py
    python examples/inference/gallery_match_example.py --real
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure workspace src takes priority over any globally-installed mata
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mata.recognition import Gallery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2(vec: np.ndarray) -> np.ndarray:
    """Return L2-normalised copy of vec."""
    return vec / (np.linalg.norm(vec) + 1e-8)


def _make_reference_embeddings(n_identities: int = 4, dim: int = 512):
    """Return (labels, embeddings) pairs of L2-normalised reference vectors."""
    rng = np.random.default_rng(42)
    labels = [f"person_{i:02d}" for i in range(n_identities)]
    matrix = rng.standard_normal((n_identities, dim)).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)
    return labels, matrix


def _make_query_embedding(reference: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
    """Return a noisy variant of a reference embedding (same identity)."""
    rng = np.random.default_rng(0)
    noisy = reference + rng.standard_normal(reference.shape).astype(np.float32) * noise_scale
    return _l2(noisy)


# ---------------------------------------------------------------------------
# Example 1: Build a gallery
# ---------------------------------------------------------------------------

def example_build_gallery():
    """Create a Gallery and enroll reference identities."""
    print("\n--- 1. Building a gallery ---")

    gallery = Gallery(similarity_thresh=0.5)

    labels, matrix = _make_reference_embeddings(n_identities=4)

    for label, vec in zip(labels, matrix):
        idx = gallery.add(label, vec)
        print(f"  enrolled: {label!r}  at index={idx}  norm={np.linalg.norm(vec):.4f}")

    print(f"\n  gallery size    : {len(gallery)}")
    print(f"  unique labels   : {gallery.unique_labels}")
    return gallery, labels, matrix


# ---------------------------------------------------------------------------
# Example 2: Search the gallery
# ---------------------------------------------------------------------------

def example_search(gallery: Gallery, labels: list, matrix: np.ndarray):
    """Search for a noisy query — should recover the correct identity."""
    print("\n--- 2. Gallery search ---")

    # Build a query close to person_00
    query = _make_query_embedding(matrix[0], noise_scale=0.05)
    print(f"  query norm     : {np.linalg.norm(query):.4f}")

    matches = gallery.search(query, top_k=3)
    print(f"  top-3 matches  :")
    for m in matches:
        print(f"    {m.label!r:14s}  similarity={m.similarity:.4f}  index={m.index}")

    # Verify correct identity is top-1
    if matches:
        print(f"\n  Predicted: {matches[0].label!r}  (expected: {labels[0]!r})")
        assert matches[0].label == labels[0], "Top-1 should match the source identity!"
        print("  ✓ Correct identity recovered")


# ---------------------------------------------------------------------------
# Example 3: One-liner recognition — mata.run("recognize", ...)
# ---------------------------------------------------------------------------

def example_recognize_api(gallery: Gallery, matrix: np.ndarray, use_real: bool = False):
    """Use the convenience mata.run('recognize', ...) API."""
    print("\n--- 3. One-liner recognition ---")

    if use_real:
        import mata
        result = mata.run(
            "recognize", "examples/images/000000039769.jpg",
            gallery=gallery,
            model="openai/clip-vit-base-patch32",
            top_k=3,
        )
        entry = result.entries[0]  # MatchEntry for the single query image
        for m in entry.all_matches:  # list[dict] with 'label', 'similarity', 'index'
            print(f"  {m['label']!r}: {m['similarity']:.4f}")
    else:
        # Simulate using the gallery directly
        query = _make_query_embedding(matrix[1], noise_scale=0.05)
        matches = gallery.search(query, top_k=3)
        print("  (mock) top-3 matches:")
        for m in matches:
            print(f"    {m.label!r:14s}  similarity={m.similarity:.4f}")


# ---------------------------------------------------------------------------
# Example 4: Gallery persistence
# ---------------------------------------------------------------------------

def example_persistence(gallery: Gallery, tmp_dir: Path):
    """Save the gallery to .npz and reload it."""
    print("\n--- 4. Gallery save / load ---")

    path = tmp_dir / "persons.npz"
    gallery.save(str(path))
    print(f"  Saved gallery to {path} ({path.stat().st_size} bytes)")

    gallery2 = Gallery.load(str(path))
    print(f"  Loaded gallery  : size={len(gallery2)}, thresh={gallery2.similarity_thresh}")
    print(f"  Labels match    : {gallery.labels == gallery2.labels}")


# ---------------------------------------------------------------------------
# Example 5: Batch search and threshold filtering
# ---------------------------------------------------------------------------

def example_batch_search(gallery: Gallery, matrix: np.ndarray):
    """Run batch cosine search and filter by threshold."""
    print("\n--- 5. Batch search + threshold filtering ---")

    # Build 4 queries: 3 near known identities, 1 random (unknown)
    rng = np.random.default_rng(7)
    queries = np.stack([
        _make_query_embedding(matrix[0], noise_scale=0.05),
        _make_query_embedding(matrix[1], noise_scale=0.05),
        _make_query_embedding(matrix[2], noise_scale=0.05),
        _l2(rng.standard_normal(matrix.shape[1]).astype(np.float32)),  # unknown
    ])

    batch_results = gallery.search_batch(queries, top_k=1)
    print("  Query results (threshold=0.5):")
    for i, matches in enumerate(batch_results):
        if matches:
            m = matches[0]
            status = "✓ known" if m.similarity >= gallery.similarity_thresh else "? below threshold"
            print(f"    query[{i}]  →  {m.label!r:14s}  sim={m.similarity:.4f}  {status}")
        else:
            print(f"    query[{i}]  →  (no match above threshold)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    use_real = "--real" in sys.argv

    if use_real:
        print("=== Gallery Matching Examples (real model mode) ===")
    else:
        print("=== Gallery Matching Examples (mock mode — no downloads) ===")
        print("    Pass --real to use actual HuggingFace models.\n")

    gallery, labels, matrix = example_build_gallery()
    example_search(gallery, labels, matrix)
    example_recognize_api(gallery, matrix, use_real)
    example_persistence(gallery, tmp_dir=Path("."))
    example_batch_search(gallery, matrix)

    print("\nAll examples completed successfully.")
