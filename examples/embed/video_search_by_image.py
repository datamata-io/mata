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
import mata
from mata.core.video_io import iter_frames, get_video_info
from mata.recognition import index_video

# Tune these for your video / target action duration.
# Run recommend_chunk_params() below to get suggested values.
CHUNK_FRAMES = 15   # frames sampled per chunk
CHUNK_STRIDE = 89   # stride in frames (~3 s at 30 fps)


def search_by_image(frame_path: str, video_index, model: str, top_k: int = 5) -> None:
    """Find video chunks visually similar to a query image frame.

    Uses mata.run("recognize") which embeds the image through X-CLIP's video
    encoder (as a 1-frame clip) and performs cosine similarity search against
    the indexed gallery.
    """

    result = mata.run("recognize", frame_path,
                        gallery=video_index.gallery,
                        model=model,
                        top_k=top_k)
    
    entry = result.entries[0]   # one MatchEntry for the single query image
    print(f"\nImage-to-video results for: '{frame_path}'")
    for i, m in enumerate(entry.all_matches, 1):
        chunk_id = m["label"]
        if chunk_id in video_index.frame_map:
            start_s = video_index.frame_map[chunk_id]
            end_s = video_index.end_map[chunk_id]
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
    video_index = index_video(
        args.video,
        model=model,
        mode="chunk",
        chunk_stride=CHUNK_STRIDE,
        chunk_frames=CHUNK_FRAMES,
    )
    print(f"Indexed {len(video_index.gallery)} chunks.\n")

    search_by_image(args.query_image, video_index, model, top_k=args.top_k)
