#!/usr/bin/env python3
"""Graph-Based Recognition Pipeline — Detect >> ExtractROIs >> Embed >> GalleryMatchNode.

Demonstrates:
    1. Building a reference gallery from known-identity enrollment embeddings
    2. Running a Detect → ExtractROIs → Embed → GalleryMatchNode graph pipeline
    3. Accessing per-instance match results from the graph output
    4. Extending to a multi-image batch recognition loop

Runs in mock mode by default (no real model downloads required).
Use --real to load models from HuggingFace.

Usage:
    # Mock mode (no real models needed)
    python examples/graph/recognition_pipeline.py

    # Real mode (downloads HuggingFace models)
    python examples/graph/recognition_pipeline.py --real

Graph topology:
    Detect → ExtractROIs → Embed → GalleryMatchNode
               (bbox crop)   (CLIP)    (cosine search)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root without installing the package
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

IMAGE_DIR = Path(__file__).parent.parent / "images"
IMAGE_PATH = IMAGE_DIR / "000000039769.jpg"

# ---------------------------------------------------------------------------
# Gallery helpers
# ---------------------------------------------------------------------------

def _l2(v: np.ndarray) -> np.ndarray:
    """Return L2-normalised copy of *v*."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def build_gallery():
    """Build a small reference gallery with four enrolled identities.

    Returns:
        Populated Gallery ready for recognition.
    """
    from mata.recognition import Gallery

    rng = np.random.default_rng(42)
    gallery = Gallery(similarity_thresh=0.5)

    identities = ["alice", "bob", "carol", "dave"]
    for name in identities:
        # Two enrollment embeddings per identity for a more robust gallery
        emb_a = _l2(rng.standard_normal(512).astype(np.float32))
        emb_b = _l2(emb_a + rng.standard_normal(512).astype(np.float32) * 0.05)
        gallery.add(name, emb_a)
        gallery.add(name, _l2(emb_b))

    print(f"Gallery built: {gallery.size} vectors, {len(gallery.unique_labels)} identities")
    return gallery


# ---------------------------------------------------------------------------
# Mock providers (default — no HuggingFace downloads)
# ---------------------------------------------------------------------------

def create_mock_providers(gallery):
    """Return mock detector and embedder that produce plausible artifacts.

    The fake embedder returns embeddings close to the first gallery vector
    for "alice" so that the match assertion passes.

    Args:
        gallery: Pre-populated Gallery (used to anchor mock embeddings).
    """
    from unittest.mock import Mock

    from mata.core.artifacts.detections import Detections
    from mata.core.artifacts.embeddings import Embeddings
    from mata.core.artifacts.rois import ROIs
    from mata.core.types import Instance, VisionResult

    # -- Fake detector: two confident instances --
    mock_detector = Mock()

    def _fake_predict(image, **_kw):
        return VisionResult(
            instances=[
                Instance(bbox=(50, 30, 220, 300), label=0, score=0.91, label_name="person"),
                Instance(bbox=(280, 60, 450, 350), label=0, score=0.85, label_name="person"),
            ],
            meta={"model": "mock-detr"},
        )

    mock_detector.predict = Mock(side_effect=_fake_predict)

    # -- Fake embedder: returns embeddings near "alice" --
    alice_vec = gallery._vectors[0]  # first enrolled alice vector
    rng = np.random.default_rng(0)

    def _fake_embed(rois, **_kw):
        n = len(rois) if hasattr(rois, "__len__") else 2
        vecs = np.stack([
            _l2(alice_vec + rng.standard_normal(512).astype(np.float32) * 0.02)
            for _ in range(n)
        ])
        from mata.core.types import EmbedResult
        return EmbedResult(embeddings=vecs)

    mock_embedder = Mock()
    mock_embedder.predict = Mock(side_effect=_fake_embed)

    return {"detector": mock_detector, "embedder": mock_embedder}


# ---------------------------------------------------------------------------
# Real providers (--real flag)
# ---------------------------------------------------------------------------

def create_real_providers():
    """Load real HuggingFace detector and CLIP embedder.

    Returns:
        Provider dict for mata.infer().
    """
    import mata

    detector = mata.load("detect", "facebook/detr-resnet-50")
    embedder = mata.load("embed", "openai/clip-vit-base-patch32")
    return {"detector": detector, "embedder": embedder}


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(gallery, providers, image_path: str | Path):
    """Execute the Detect → ExtractROIs → Embed → GalleryMatch graph.

    Args:
        gallery: Populated Gallery for identity matching.
        providers: Dict mapping provider names to adapter instances.
        image_path: Path to the query image.

    Returns:
        MultiResult from mata.infer().
    """
    import mata
    from mata.nodes import Detect, ExtractROIs, Embed, GalleryMatchNode

    result = mata.infer(
        image=str(image_path),
        graph=[
            # Step 1: Detect persons / objects of interest
            Detect(using="detector", out="dets"),
            # Step 2: Crop bounding-box regions from the image
            ExtractROIs(src_dets="dets", out="rois"),
            # Step 3: Extract feature embeddings from each crop
            Embed(using="embedder", src="rois", out="embeddings"),
            # Step 4: Match embeddings against the reference gallery
            GalleryMatchNode(gallery=gallery, src="embeddings", out="matches", top_k=3),
        ],
        providers=providers,
    )
    return result


def print_results(result):
    """Display recognition matches from the graph output."""
    matches_artifact = result["matches"]
    all_matches = matches_artifact.all_matches  # list[list[GalleryMatch]]

    print(f"\nRecognition results: {len(all_matches)} instance(s) matched")
    for idx, candidates in enumerate(all_matches):
        if not candidates:
            print(f"  Instance {idx}: no match above threshold")
        else:
            top = candidates[0]
            print(f"  Instance {idx}: best match = '{top.label}' "
                  f"(sim={top.similarity:.4f}, dist={top.distance:.4f})")
            if len(candidates) > 1:
                others = ", ".join(
                    f"{m.label}@{m.similarity:.3f}" for m in candidates[1:]
                )
                print(f"              other candidates: {others}")


# ---------------------------------------------------------------------------
# Batch extension
# ---------------------------------------------------------------------------

def run_batch(gallery, providers, image_paths: list[str | Path]):
    """Run recognition on multiple images sequentially.

    Args:
        gallery: Shared Gallery for all queries.
        providers: Shared provider dict.
        image_paths: List of image paths.
    """
    for i, path in enumerate(image_paths):
        print(f"\n--- Image {i + 1}: {Path(path).name} ---")
        result = run_pipeline(gallery, providers, path)
        print_results(result)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Run the recognition pipeline demo."""
    use_real = "--real" in sys.argv

    print("=== Graph-Based Recognition Pipeline ===")
    print(f"Mode: {'real (HuggingFace)' if use_real else 'mock (no model downloads)'}\n")

    # 1. Build the reference gallery
    gallery = build_gallery()

    # 2. Select providers
    if use_real:
        print("Loading real models from HuggingFace...")
        providers = create_real_providers()
    else:
        print("Using mock providers")
        providers = create_mock_providers(gallery)

    # 3. Run on a single image
    print("\n--- Single image recognition ---")
    result = run_pipeline(gallery, providers, IMAGE_PATH)
    print_results(result)

    # Verify top match in mock mode
    if not use_real:
        matches_artifact = result["matches"]
        all_matches = matches_artifact.all_matches
        if all_matches and all_matches[0]:
            top_label = all_matches[0][0].label
            assert top_label == "alice", (
                f"Expected 'alice' as top match in mock mode, got '{top_label}'"
            )
            print("\n[OK] Mock assertion passed: top match is 'alice'")

    # 4. Batch mode demo (same image repeated for illustration)
    print("\n--- Batch recognition (3x same image) ---")
    run_batch(gallery, providers, [IMAGE_PATH] * 3)

    print("\nDone.")


if __name__ == "__main__":
    main()
