"""Image-to-video search using mata.run("recognize") + X-CLIP.

Index a video, then search for visually similar segments using a query *image*
instead of a text string.  Both the gallery chunks and the query frame are
embedded in the same X-CLIP vector space, so cosine similarity finds clips
that look like the query image.

(Video by George Morina: https://www.pexels.com/video/streets-of-london-at-night-5823504/)

Workflow:
    1. Index the video  →  Gallery of chunk embeddings
    2. (optional) Dump individual frames so you can pick a query image
    3. Search the gallery with a query image via mata.run("recognize", ...)

Usage:
    # Index + search in one shot
    python examples/embed/video_search_by_image.py examples/videos/dashcam_bus.mp4  examples/embed/frame_001239.jpg

    # Dump frames first to pick a good query image
    python examples/embed/video_search_by_image.py \\
        examples/videos/dashcam_bus.mp4 --dump-frames output/frames

    # Then search with the chosen frame
    python examples/embed/video_search_by_image.py \\
        examples/videos/dashcam_bus.mp4 output/frames/frame_001239.jpg
"""
import sys
import argparse
import numpy as np
import mata
from mata.core.video_io import iter_frames, get_video_info
from mata.recognition import Gallery

# Tune these for your video / target action duration.
# Run recommend_chunk_params() below to get suggested values.
CHUNK_FRAMES = 15   # frames sampled per chunk
CHUNK_STRIDE = 89   # stride in frames (~3 s at 30 fps)


def index_video(video_path: str, model: str) -> tuple[Gallery, dict]:
    """Embed every CHUNK_STRIDE-frame window and store in a Gallery."""
    adapter = mata.load("embed", model)
    gallery = Gallery()
    frame_map: dict[str, int] = {}

    frames_buf: list = []
    chunk_start = 0

    for frame_idx, bgr_frame in iter_frames(video_path):
        frames_buf.append(bgr_frame)

        if len(frames_buf) == CHUNK_STRIDE:
            sampled = [frames_buf[i] for i in
                    np.linspace(0, CHUNK_STRIDE - 1, CHUNK_FRAMES, dtype=int)]
            emb = adapter.embed(sampled)        # (1, D)
            chunk_id = f"chunk_{chunk_start:06d}"
            gallery.add(chunk_id, emb[0])
            frame_map[chunk_id] = chunk_start
            chunk_start = frame_idx + 1
            frames_buf = []
            print(f"Indexed up to frame {frame_idx}...", end="\r")

    # Flush tail
    if frames_buf:
        n = len(frames_buf)
        indices = np.linspace(0, n - 1, min(CHUNK_FRAMES, n), dtype=int)
        sampled = [frames_buf[i] for i in indices]
        emb = adapter.embed(sampled)
        chunk_id = f"chunk_{chunk_start:06d}"
        gallery.add(chunk_id, emb[0])
        frame_map[chunk_id] = chunk_start

    return gallery, frame_map


def search_by_image(frame_path: str, gallery: Gallery, frame_map: dict,
                    model: str, fps: float, top_k: int = 5) -> None:
    """Find video chunks visually similar to a query image frame.

    Uses mata.run("recognize") which embeds the image through X-CLIP's video
    encoder (as a 1-frame clip) and performs cosine similarity search against
    the indexed gallery.
    """

    result = mata.run("recognize", frame_path,
                        gallery=gallery,
                        model=model,
                        top_k=top_k)
    
    entry = result.entries[0]   # one MatchEntry for the single query image
    print(f"\nImage-to-video results for: '{frame_path}'")
    for i, m in enumerate(entry.all_matches, 1):
        chunk_id = m["label"]
        if chunk_id in frame_map:
            start_s = frame_map[chunk_id] / fps
            end_s = start_s + CHUNK_STRIDE / fps
            sm, ss = int(start_s) // 60, int(start_s) % 60
            em, es = int(end_s) // 60, int(end_s) % 60
            ts = f"@ {sm}m{ss:02d}s – {em}m{es:02d}s"
        else:
            ts = "(not in frame map)"
        print(f"  #{i} [{m['similarity']:.3f}]  {chunk_id}  {ts}")


def dump_frames(video_path: str, out_dir: str, fps: float) -> None:
    """Save every frame as a JPEG with a timestamp label burned in."""
    import cv2, os
    os.makedirs(out_dir, exist_ok=True)
    for frame_idx, bgr in iter_frames(video_path):
        ts = frame_idx / fps
        label = f"{int(ts // 60):02d}m{ts % 60:05.2f}s  f{frame_idx}"
        cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg"), bgr)
    print(f"Saved frames to {out_dir}")


def recommend_chunk_params(fps: int = 30, action_duration_sec: float = 3.0,
                        event_speed: str = "fast", model_native_frames: int = 8) -> None:
    density_map = {"slow": 2, "normal": 3, "fast": 5}
    chunk_stride = int(action_duration_sec * fps)
    chunk_frames = int(action_duration_sec * density_map[event_speed])
    chunk_frames = max(model_native_frames, min(chunk_frames, 16))
    print(f"CHUNK_STRIDE = {chunk_stride} frames ({action_duration_sec}s)")
    print(f"CHUNK_FRAMES = {chunk_frames}")
    print(f"Density      = {chunk_frames / action_duration_sec:.1f} fps sampled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image-to-video search with X-CLIP")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("query_image", nargs="?",
                        help="Path to the query image frame (omit with --dump-frames)")
    parser.add_argument("--dump-frames", metavar="OUT_DIR",
                        help="Dump all frames as JPEGs to OUT_DIR, then exit")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to return (default: 5)")
    args = parser.parse_args()

    model = "microsoft/xclip-base-patch32"
    info = get_video_info(args.video)
    fps = info["fps"]

    if args.dump_frames:
        print(f"Dumping frames from {args.video} to {args.dump_frames} ...")
        dump_frames(args.video, args.dump_frames, fps)
        sys.exit(0)

    if not args.query_image:
        parser.error("query_image is required unless --dump-frames is used")

    print(f"Indexing {args.video} ({info['frame_count']} frames @ {fps:.1f} fps)...")
    gallery, frame_map = index_video(args.video, model)
    print(f"Indexed {len(gallery)} chunks.\n")

    search_by_image(args.query_image, gallery, frame_map, model,
                    fps=fps, top_k=args.top_k)
