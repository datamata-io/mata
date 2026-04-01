"""Semantic video search with X-CLIP.

Index a video file and search for clips matching a text description.
Pass multiple queries to compare them against the same indexed video.

Usage:
    # Single query
    python examples/embed/video_semantic_search.py examples/videos/eating_spaghetti.mp4 "eating spaghetti"

    # Multi-query (Video by George Morina: https://www.pexels.com/video/streets-of-london-at-night-5823504/)
    python examples/embed/video_semantic_search.py examples/videos/dashcam_bus.mp4 "red bus" "a white car at traffic light" "traffic congestion"

"""
import sys
import numpy as np
import mata
from mata.core.video_io import iter_frames, get_video_info
from mata.recognition import Gallery

# RUN recommend_chunk_params() to get recommended CHUNK_STRIDE and CHUNK_FRAMES values for your video and target actions, then set them here: (See function recommend_chunk_params for details)
CHUNK_FRAMES = 15   # frames to sample
CHUNK_STRIDE = 89   # 3-second stride at 30 fps → 90 frames 

# Used frames [0, 6, 12, 19, 25, 32, 38, 44, 51, 57, 64, 70, 76, 83, 89] (used linspace not range)

def index_video(video_path: str, model: str) -> tuple[Gallery, dict]:
    adapter = mata.load("embed", model)
    gallery = Gallery()
    frame_map = {}  # chunk_id → start_frame_idx

    frames_buf = []
    chunk_start = 0

    for frame_idx, bgr_frame in iter_frames(video_path):
        frames_buf.append(bgr_frame)

        if len(frames_buf) == CHUNK_STRIDE:
            # Sample CHUNK_FRAMES from this chunk
            sampled = [frames_buf[i] for i in
                    np.linspace(0, CHUNK_STRIDE - 1, CHUNK_FRAMES, dtype=int)]
            emb = adapter.embed(sampled)  # (1, 512)
            chunk_id = f"chunk_{chunk_start:06d}"
            gallery.add(chunk_id, emb[0])
            frame_map[chunk_id] = chunk_start
            chunk_start = frame_idx + 1  # next chunk starts at the NEXT frame
            frames_buf = []
            print(f"Chunked at frame {frame_idx}...", end="\r")

    # Flush remaining frames (handles videos shorter than CHUNK_STRIDE)
    if frames_buf:
        n = len(frames_buf)
        indices = np.linspace(0, n - 1, min(CHUNK_FRAMES, n), dtype=int)
        sampled = [frames_buf[i] for i in indices]
        emb = adapter.embed(sampled)  # (1, D)
        chunk_id = f"chunk_{chunk_start:06d}"
        gallery.add(chunk_id, emb[0])
        frame_map[chunk_id] = chunk_start

    return gallery, frame_map


def search(gallery: Gallery, frame_map: dict, query: str | list[str],
        model: str, fps: float = 30.0) -> None:
    adapter = mata.load("embed", model)
    queries = [query] if isinstance(query, str) else query

    # Embed all queries at once → (N, D)
    text_embs = np.vstack([adapter.embed(q) for q in queries])
    top_k = 10 # retrieve more than needed to show how threshold filtering works
    all_matches = gallery.search_batch(text_embs, top_k=top_k, threshold=0.22) # adjust threshold as needed to balance precision vs recall

    for q, matches in zip(queries, all_matches):
        print(f"\nResults for: '{q}'")
        for i, m in enumerate(matches, 1):
            start_s = frame_map[m.label] / fps
            end_s = start_s + CHUNK_STRIDE / fps
            sm, ss = int(start_s) // 60, int(start_s) % 60
            em, es = int(end_s) // 60, int(end_s) % 60
            print(f"  #{i} [{m.similarity:.3f}]  {m.label}  "
                f"@ {sm}m{ss:02d}s – {em}m{es:02d}s")
            
def dump_frames(video_path: str, out_dir: str, fps: float) -> None:
    """Save every frame as a JPEG with the timestamp burned in."""
    import cv2, os
    os.makedirs(out_dir, exist_ok=True)
    for frame_idx, bgr in iter_frames(video_path):
        ts = frame_idx / fps
        label = f"{int(ts//60):02d}m{ts%60:05.2f}s  f{frame_idx}"
        cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg"), bgr)
    print(f"Saved frames to {out_dir}")

def recommend_chunk_params(
    fps: int = 30,
    action_duration_sec: float = 3.0,
    event_speed: str = "fast",   # "slow", "normal", "fast"
    model_native_frames: int = 8
):
    # microsoft/xclip-base-patch32 was pretrained on 8 frames per clip (from Kinetics-400/600 dataset).
    # action_duration_sec is a rough guess of how long the target actions typically last — it doesn't have to be exact, just a ballpark. 
    # The event_speed is a subjective estimate of how densely the relevant actions occur in the video. 
    # Adjusting these will change how many chunks are generated and how well they capture the target actions.
    # for example, a car hit by another accelerating car might be a 3-second "fast" event, while a person walking around a store might be a "slow" 10-second event.

    density_map = {"slow": 2, "normal": 3, "fast": 5}
    target_density = density_map[event_speed]

    chunk_stride = int(action_duration_sec * fps)
    chunk_frames = int(action_duration_sec * target_density)

    # Clamp to model's sweet spot
    chunk_frames = max(model_native_frames, min(chunk_frames, 16))

    actual_density = chunk_frames / action_duration_sec

    print(f"RECOMMENDED_CHUNK_STRIDE  = {chunk_stride} frames ({action_duration_sec}s)")
    print(f"RECOMMENDED_CHUNK_FRAMES  = {chunk_frames}")
    print(f"Actual density = {actual_density:.1f} fps sampled")
    print(f"Ratio (stride/frames) = {chunk_stride/chunk_frames:.1f}x compression")
    exit(0)

if __name__ == "__main__":
    video_path = sys.argv[1]
    queries = sys.argv[2:]  # one or more query strings
    if not queries:
        print("Usage: video_semantic_search.py <video> <query> [<query2> ...]")
        sys.exit(1)
    model = "microsoft/xclip-base-patch32"

    info = get_video_info(video_path)
    # Uncomment the next line to get recommended CHUNK_STRIDE and CHUNK_FRAMES values for your video and target actions, then set them above.
    # recommend_chunk_params(fps=info['fps'], action_duration_sec=3.0, event_speed="fast")

    print(f"Indexing {video_path} ({info['frame_count']} frames @ {info['fps']:.1f} fps)")

    gallery, frame_map = index_video(video_path, model)
    print(f"Indexed {len(gallery)} chunks.")

    # Pass a single string or a list — both are handled
    search(gallery, frame_map, queries if len(queries) > 1 else queries[0],
        model, fps=info["fps"])

    # For image-to-video search, see examples/embed/video_search_by_image.py
