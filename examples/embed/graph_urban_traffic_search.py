"""Urban Traffic Safety Analysis with Qwen3-VL-Embedding.

Indexes a dashcam / street-scene video at 1 fps and retrieves the most
relevant timestamps for text queries — from simple ("red bus") to complex
("person dangerously jaywalking").

Qwen3-VL-Embedding's VLM backbone understands rich scene descriptions that
CLIP-style models often miss, making it well suited for safety-critical
surveillance queries.  Individual frames are indexed (not fixed-length clips),
giving finer temporal resolution than the X-CLIP sliding-window approach.

Demo video  : examples/videos/dashcam_bus.mp4
                "Streets of London at Night"
                (by George Morina — https://www.pexels.com/video/streets-of-london-at-night-5823504/)

Requirements:
    pip install datamata
    GPU with 8+ GB VRAM recommended (bfloat16)

Usage:
    # Index dashcam_bus.mp4 and run the default query tiers
    python examples/embed/graph_urban_traffic_search.py

    # Point at your own footage, run custom queries
    python examples/embed/graph_urban_traffic_search.py \\
        --video path/to/footage.mp4 \\
        --queries "red bus" "cyclist weaving through traffic"

    python examples/embed/graph_urban_traffic_search.py --video examples/videos/dashcam_video.mp4 --queries "car aggressively cutting and bumping other vehicles"

    # Save top-matching frames as annotated JPEGs
    python examples/embed/graph_urban_traffic_search.py --save-frames output/traffic/

    # Faster indexing (lower temporal resolution)
    python examples/embed/graph_urban_traffic_search.py --sample-fps 0.5

    # CPU-safe dtype
    python examples/embed/graph_urban_traffic_search.py --dtype float32

Default query tiers:
    Tier 1 — Simple:      "red bus", "traffic lights"
    Tier 2 — Descriptive: "vehicles merging at a busy junction",
                            "wet road surface reflecting street lights"
    Tier 3 — Complex:     "person dangerously jaywalking between moving vehicles",
                            "cyclist weaving through fast-moving traffic"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mata
from mata.core.graph.graph import Graph
from mata.nodes import EmbeddingSearch, IndexVideo

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_VIDEO  = str(Path(__file__).parent.parent / "videos" / "dashcam_bus.mp4")
_DEFAULT_MODEL  = "Qwen/Qwen3-VL-Embedding-2B"
_DEFAULT_DTYPE  = "bfloat16"
_DEFAULT_TOP_K  = 3
# Qwen3 cosine scores are typically in a tighter range than CLIP—start low.
_DEFAULT_THRESH = 0.18

# Three query tiers, ordered simple → complex.
# The model's VLM backbone gives it an edge on Tier 3 queries that pure
# vision-language contrastive models (CLIP/X-CLIP) struggle with.
_DEFAULT_QUERIES: list[tuple[str, str]] = [
    # (tier_label, natural-language query)
    ("Tier 1 — Simple",       "red double-decker bus"),
    ("Tier 1 — Simple",       "traffic lights at night"),
    ("Tier 1 — Simple",       "a parked taxi"),
    ("Tier 2 — Descriptive",  "vehicles merging at a busy junction"),
    ("Tier 2 — Descriptive",  "wet road surface reflecting street lights"),
    ("Tier 2 — Descriptive",  "pedestrian waiting to cross the road"),
    ("Tier 3 — Complex",      "person dangerously jaywalking between moving vehicles"),
    ("Tier 3 — Complex",      "cyclist weaving through fast-moving traffic at night"),
    ("Tier 3 — Complex",      "vehicle making an abrupt lane change near pedestrians"),
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Urban traffic safety analysis with Qwen3-VL-Embedding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--video", default=_DEFAULT_VIDEO,
        help="Path to a dashcam or street-scene video file.",
    )
    p.add_argument(
        "--model", default=_DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {_DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--dtype", default=_DEFAULT_DTYPE,
        choices=["bfloat16", "float16", "float32"],
        help="Model weight dtype. bfloat16 recommended on CUDA. "
                "Use float32 for CPU.",
    )
    p.add_argument(
        "--queries", nargs="+", metavar="QUERY",
        help="Override the default query tiers with one or more quoted strings.",
    )
    p.add_argument(
        "--sample-fps", type=float, default=1.0, metavar="FPS",
        help="Frames to index per second (default: 1.0). "
                "Lower = faster, coarser resolution.",
    )
    p.add_argument(
        "--top-k", type=int, default=_DEFAULT_TOP_K,
        help=f"Top-K results per query (default: {_DEFAULT_TOP_K}).",
    )
    p.add_argument(
        "--threshold", type=float, default=_DEFAULT_THRESH,
        help=f"Minimum cosine similarity to show (default: {_DEFAULT_THRESH}). "
                "Qwen3 scores are tighter than CLIP; start around 0.15–0.25.",
    )
    p.add_argument(
        "--embed-dim", type=int, default=None, metavar="DIM",
        help="Matryoshka truncation dim (None = full native dim).",
    )
    p.add_argument(
        "--save-frames", metavar="DIR", default=None,
        help="Save annotated top-matching frames as JPEGs under DIR "
                "(requires opencv-python).",
    )
    p.add_argument(
        "--index-only", action="store_true",
        help="Index the video and exit without searching.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("\nUrban Traffic Safety Analysis — Qwen3-VL-Embedding (Graph API)")
    print("=" * 70)
    print(f"  Model : {args.model}  |  dtype : {args.dtype}")
    print(f"  Video : {args.video}")
    print(f"  top-k : {args.top_k}  |  threshold : {args.threshold}")
    print()

    # ==================================================================
    embedder = mata.load(
        "embed", args.model, dtype=args.dtype,
        **({"embed_dim": args.embed_dim} if args.embed_dim else {}),
    )
    queries = args.queries if args.queries else [q for _, q in _DEFAULT_QUERIES]

    result = (
        Graph("urban_traffic_search")
        .then(IndexVideo(using="embedder", mode="frame", sample_fps=args.sample_fps))
        .then(EmbeddingSearch(using="embedder", text=queries,
                            top_k=args.top_k, threshold=args.threshold))
    ).run(video=args.video, providers={"embedder": embedder})

    print("=" * 70 + "\n  Search Results\n" + "=" * 70)
    for qr in result["search_results"].results:
        print(f'\n  Query: "{qr.query}"')
        if not qr.matches:
            print(f"    (no matches above threshold={args.threshold:.2f}"
                " — try lowering --threshold)")
            continue
        for rank, m in enumerate(qr.matches, 1):
            mm, ss = int(m.start_s) // 60, int(m.start_s) % 60
            print(f"    #{rank}  sim={m.similarity:.4f}  @ {mm:02d}m{ss:02d}s  [{m.label}]")
    # ==================================================================

    print("\nDone.")


if __name__ == "__main__":
    main()
