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
    python examples/embed/urban_traffic_search.py

    # Point at your own footage, run custom queries
    python examples/embed/urban_traffic_search.py \\
        --video path/to/footage.mp4 \\
        --queries "red bus" "cyclist weaving through traffic"

    python examples/embed/urban_traffic_search.py --video examples/videos/dashcam_video.mp4 --queries "car aggressively cutting and bumping other vehicles"

    # Save top-matching frames as annotated JPEGs
    python examples/embed/urban_traffic_search.py --save-frames output/traffic/

    # Faster indexing (lower temporal resolution)
    python examples/embed/urban_traffic_search.py --sample-fps 0.5

    # CPU-safe dtype
    python examples/embed/urban_traffic_search.py --dtype float32

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
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mata
from mata.core.video_io import get_video_info
from mata.recognition import VideoIndex, index_video

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
# Step 2 — Search
# ---------------------------------------------------------------------------

def run_queries(
    video_index: VideoIndex,
    queries: list[tuple[str, str]],
    model: str,
    dtype: str,
    embed_dim: int | None,
    top_k: int,
    threshold: float,
    video_path: str,
    save_dir: str | None,
) -> None:
    """Embed each text query and print the top-K matching timestamps."""
    print("=" * 70)
    print("  Search Results")
    print("=" * 70)

    current_tier: str | None = None

    for tier, query in queries:
        if tier != current_tier:
            current_tier = tier
            print(f"\n  ── {tier} ──")

        result = mata.run(
            "embed", None,
            text=query,
            model=model,
            dtype=dtype,
            **({"embed_dim": embed_dim} if embed_dim else {}),
        )
        q_vec = np.array(result).ravel()
        matches = video_index.search(q_vec, top_k=top_k, threshold=threshold)

        print(f"\n  Query: \"{query}\"")
        if not matches:
            print(f"    (no matches above threshold={threshold:.2f}  "
                    f"— try lowering --threshold)")
            continue

        for rank, m in enumerate(matches, 1):
            mm, ss = int(m.start_s) // 60, int(m.start_s) % 60
            bar = _sim_bar(m.similarity)
            print(f"    #{rank}  {bar}  sim={m.similarity:.4f}  "
                    f"@ {mm:02d}m{ss:02d}s  [{m.label}]")

            if save_dir:
                _save_frame(
                    video_path, m.label, m.start_s, video_index.native_fps,
                    save_dir, query, rank,
                )


def _sim_bar(sim: float, width: int = 8) -> str:
    """Tiny ASCII bar for quick visual comparison."""
    filled = max(0, min(width, round(sim * width)))
    return f"[{'#' * filled}{'.' * (width - filled)}]"


# ---------------------------------------------------------------------------
# Optional: save annotated frames to disk
# ---------------------------------------------------------------------------

def _save_frame(
    video_path: str,
    chunk_id: str,
    timestamp_s: float,
    native_fps: float,
    save_dir: str,
    query: str,
    rank: int,
) -> None:
    """Extract and save the matching frame as an annotated JPEG."""
    try:
        import cv2
    except ImportError:
        print("      [skip save — opencv-python not installed]")
        return

    frame_idx = int(timestamp_s * native_fps)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mm, ss = int(timestamp_s) // 60, int(timestamp_s) % 60
    label = f"{mm:02d}m{ss:02d}s | rank#{rank} | {query[:55]}"
    cv2.putText(frame, label, (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (0, 230, 80), 2, cv2.LINE_AA)

    safe_q = query[:40].replace(" ", "_").replace("/", "-")
    fname = out_dir / f"{chunk_id}_r{rank}_{safe_q}.jpg"
    cv2.imwrite(str(fname), frame)
    print(f"      → saved {fname.name}")


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

    print("\nUrban Traffic Safety Analysis — Qwen3-VL-Embedding")
    print("=" * 70)
    print(f"  Model    : {args.model}")
    print(f"  Video    : {args.video}")
    print(f"  dtype    : {args.dtype}")
    print(f"  top-k    : {args.top_k}  |  threshold: {args.threshold}")
    if args.embed_dim:
        print(f"  Matryoshka dim: {args.embed_dim}")
    print()
    print("  Note: First run downloads the model (~5-16 GB).")
    print()

    # ── Step 1: Index ──────────────────────────────────────────────────
    print("Step 1 / 2 — Indexing video frames ...")
    info = get_video_info(args.video)
    duration_s = info["frame_count"] / info["fps"]
    frame_stride = max(1, int(info["fps"] / args.sample_fps))
    print(f"  Video      : {Path(args.video).name}")
    print(f"  Duration   : {duration_s:.1f} s  ({info['frame_count']} frames @ {info['fps']:.1f} fps)")
    print(f"  Sample rate: {args.sample_fps:.1f} fps  (stride = every {frame_stride} frames)")
    print(f"  Est. index : ~{int(info['frame_count'] / frame_stride)} embeddings")
    print()

    started_at = time.time()

    def _progress(current: int, total: int) -> None:
        if current % 20 != 0:
            return
        elapsed = time.time() - started_at
        rate = current / max(elapsed, 1e-6)
        remaining = max(total - current, 0) / max(rate, 1e-6)
        print(
            f"  {current} frames indexed  ({elapsed:.0f}s elapsed, ~{remaining:.0f}s left)   ",
            end="\r",
        )

    video_index = index_video(
        args.video,
        model=args.model,
        mode="frame",
        sample_fps=args.sample_fps,
        progress=_progress,
        dtype=args.dtype,
        **({"embed_dim": args.embed_dim} if args.embed_dim else {}),
    )
    elapsed = time.time() - started_at
    print(
        f"  {video_index.indexed_count} frames indexed in {elapsed:.1f} s  "
        f"({video_index.indexed_count / max(elapsed, 1e-6):.1f} emb/s)          "
    )
    print()
    print(f"  Gallery: {len(video_index.gallery)} frame embeddings ready.\n")

    if args.index_only:
        print("--index-only set. Exiting.")
        return

    # ── Step 2: Search ─────────────────────────────────────────────────
    if args.queries:
        queries = [("Custom", q) for q in args.queries]
    else:
        queries = _DEFAULT_QUERIES

    print(f"Step 2 / 2 — Running {len(queries)} queries ...\n")
    run_queries(
        video_index=video_index,
        queries=queries,
        model=args.model,
        dtype=args.dtype,
        embed_dim=args.embed_dim,
        top_k=args.top_k,
        threshold=args.threshold,
        video_path=args.video,
        save_dir=args.save_frames,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
