#!/usr/bin/env python3
"""Single-camera Graph pipeline: Detect → Track → Embed → ReID.

Demonstrates how to wire the full cross-camera re-identification pipeline
using MATA's Graph API.  The graph nodes mirror a production video-analytics
stack where each camera runs its own pipeline and shares embeddings via
Valkey so that the same physical identity is assigned a consistent global ID
across cameras.

.. note::
    **Work in progress.**  The pipeline runs end-to-end and works well for a
    single camera feed.  Multi-camera tuning (similarity thresholds, TTL
    values, ReID model choice) is highly dataset-dependent and has not been
    systematically optimised.  Results on dense multi-camera datasets such as
    WildTrack may require parameter tuning before the cross-camera IDs are
    reliable.  Contributions are very welcome!

Graph topology::

    Detect → Filter → Track → ExtractROIs → Embed → ReID → AnnotateRT

Key components:
- ``Detect``      — run a detection adapter on each frame
- ``Filter``      — drop low-confidence detections
- ``Track``       — BotSort / ByteTrack per-frame state update
- ``ExtractROIs`` — crop the detected regions for the encoder
- ``Embed``       — extract appearance embeddings (ReID encoder)
- ``ReID``        — publish embeddings to Valkey, query for cross-camera matches
- ``AnnotateRT``  — draw bounding boxes, track IDs and trail overlays

Usage (mock mode — no models or Valkey required):
    python examples/graph/graph_reid_pipeline.py

Usage (real video + detector + ReID encoder):
    python examples/graph/graph_reid_pipeline.py \\
        --video path/to/cam1.mp4 \\
        --model facebook/detr-resnet-50 \\
        --reid-model openai/clip-vit-base-patch32

Usage (with cross-camera Valkey store):
    python examples/graph/graph_reid_pipeline.py \\
        --video path/to/cam1.mp4 \\
        --model facebook/detr-resnet-50 \\
        --reid-model openai/clip-vit-base-patch32 \\
        --valkey valkey://localhost:6379 \\
        --camera-id cam-1

Requirements:
    pip install datamata
    # For cross-camera ReID:
    pip install datamata[valkey]
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# GlobalIDRegistry — maps (camera_id, local_track_id) → stable global ID
# ---------------------------------------------------------------------------

class GlobalIDRegistry:
    """Map ``(camera_id, local_track_id)`` pairs to stable cross-camera IDs.

    A TTL mechanism evicts any entry not seen for ``ttl_frames`` consecutive
    frames so that re-appearing objects start a fresh identity assignment.
    """

    def __init__(self, ttl_frames: int = 30) -> None:
        self._map: dict[tuple[str, int], int] = {}
        self._last_seen: dict[tuple[str, int], int] = {}
        self._next_id = 1
        self.ttl_frames = ttl_frames

    def tick(self, frame_idx: int, active_keys: set[tuple[str, int]]) -> None:
        """Update last-seen timestamps and evict stale entries."""
        for key in active_keys:
            self._last_seen[key] = frame_idx
        expired = [
            k for k, last_f in self._last_seen.items()
            if frame_idx - last_f > self.ttl_frames
        ]
        for k in expired:
            self._map.pop(k, None)
            del self._last_seen[k]

    def resolve(
        self,
        cam_id: str,
        local_tid: int,
        matched_cam_id: str,
        matched_tid: int,
    ) -> int:
        """Return a stable global ID for a matched pair, or ``-1`` on conflict."""
        key_a = (cam_id, local_tid)
        key_b = (matched_cam_id, matched_tid)
        gid_a = self._map.get(key_a)
        gid_b = self._map.get(key_b)
        if gid_a is not None and gid_b is not None:
            if gid_a == gid_b:
                return gid_a
            return -1  # conflicting assignments — likely a false-positive match
        elif gid_a is not None:
            self._map[key_b] = gid_a
            return gid_a
        elif gid_b is not None:
            self._map[key_a] = gid_b
            return gid_b
        else:
            gid = self._next_id
            self._next_id += 1
            self._map[key_a] = gid
            self._map[key_b] = gid
            return gid

    @property
    def num_global_ids(self) -> int:
        return self._next_id - 1


# ---------------------------------------------------------------------------
# Mock helpers (used when --model / --reid-model are not supplied)
# ---------------------------------------------------------------------------

def _make_mock_detector():
    from unittest.mock import Mock
    from mata.core.types import Instance, VisionResult

    call_count = {"n": 0}

    def mock_predict(image, **kwargs):
        n = call_count["n"]
        call_count["n"] += 1
        x = 80 + (n % 40) * 4
        return VisionResult(
            instances=[
                Instance(bbox=(x, 50, x + 80, 260), label=0,
                         score=0.91, label_name="person"),
                Instance(bbox=(350, 110, 480, 265), label=0,
                         score=0.85, label_name="person"),
            ],
            meta={"frame_idx": n},
        )

    det = Mock()
    det.predict = mock_predict
    det.id2label = {0: "person"}
    return det


def _make_mock_encoder(embedding_dim: int = 128):
    import numpy as np
    from unittest.mock import Mock

    def mock_predict(crops, **kwargs):
        if not crops:
            return np.empty((0, 0), dtype=np.float32)
        n = len(crops)
        raw = np.random.randn(n, embedding_dim).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / np.where(norms == 0, 1.0, norms)

    enc = Mock()
    enc.predict = mock_predict
    return enc


def _make_mock_tracker():
    """Return a SimpleIOUTracker (built-in, no external deps)."""
    from mata.nodes.track import SimpleIOUTracker
    return SimpleIOUTracker()


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_graph(
    name: str,
    conf: float,
    has_encoder: bool,
    cam_label: str,
    cam_color: tuple[int, int, int],
    show_trails: bool,
    trail_length: int,
) -> Any:
    from mata.core.graph import Graph
    from mata.nodes import AnnotateRT, Detect, Embed, ExtractROIs, Filter, ReID
    from mata.nodes.track import Track

    g = (
        Graph(name)
        .then(Detect(using="detector", out="dets"))
        .then(Filter(src="dets", score_gt=conf, out="filtered"))
        .add(
            Track(using="tracker", out="tracks"),
            inputs={"detections": "Filter.filtered"},
        )
        .add(
            ExtractROIs(src_image="image", src_dets="filtered", out="rois", padding=4),
            inputs={"image": "input.image", "detections": "Filter.filtered"},
        )
    )

    if has_encoder:
        g = g.add(
            Embed(using="encoder", src="rois", out="embeddings", normalize=True),
            inputs={"rois": "ExtractROIs.rois"},
        )
        g = g.add(
            ReID(using="bridge", out="cross_matches"),
            inputs={"tracks": "Track.tracks", "embeddings": "Embed.embeddings"},
        )

    annotate_inputs: dict[str, str] = {
        "image": "input.image",
        "detections": "Track.tracks",
        "tracks": "Track.tracks",
    }
    if has_encoder:
        annotate_inputs["cross_matches"] = "ReID.cross_matches"

    g = g.add(
        AnnotateRT(
            show_track_ids=True,
            show_trails=show_trails,
            trail_length=trail_length,
            camera_label=cam_label,
            camera_color=cam_color,
            out="annotated",
            tracks_src="tracks",
            cross_matches_src="cross_matches" if has_encoder else None,
        ),
        inputs=annotate_inputs,
    )
    return g


# ---------------------------------------------------------------------------
# ReIDBridge initialisation
# ---------------------------------------------------------------------------

def _init_reid_bridge(
    cam_id: str,
    valkey_url: str,
    similarity_thresh: float,
    ttl: int,
) -> Any | None:
    try:
        from mata.trackers import ReIDBridge
        return ReIDBridge(
            valkey_url,
            camera_id=cam_id,
            ttl=ttl,
            similarity_thresh=similarity_thresh,
        )
    except Exception as exc:
        print(f"[warn] ReIDBridge unavailable ({exc}). Skipping cross-camera ReID.")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Graph-based single-camera tracking + cross-camera ReID",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--video", default=None, metavar="PATH",
        help="Path to a video file. Omit to run in mock (synthetic) mode.",
    )
    p.add_argument(
        "--model", default=None, metavar="MODEL",
        help=(
            "Detection model — HuggingFace ID, .onnx/.pt path, or config alias. "
            "Example: 'facebook/detr-resnet-50'. Omit for mock mode."
        ),
    )
    p.add_argument(
        "--reid-model", default=None, metavar="MODEL",
        help=(
            "ReID / appearance encoder — HuggingFace ID or local .onnx path. "
            "Example: 'openai/clip-vit-base-patch32'. "
            "Omit to run tracking without cross-camera ReID."
        ),
    )
    p.add_argument("--tracker", default="botsort", choices=["botsort", "bytetrack"])
    p.add_argument("--conf", type=float, default=0.5,
                   help="Detection confidence threshold.")
    p.add_argument(
        "--camera-id", default="cam-1", metavar="ID",
        help="Logical camera identifier used when publishing embeddings to Valkey.",
    )
    p.add_argument(
        "--valkey", default=None, metavar="URL",
        help=(
            "Valkey / Redis URL for cross-camera embedding sharing. "
            "Example: 'valkey://localhost:6379'. "
            "Omit to disable cross-camera ReID (single-camera mode)."
        ),
    )
    p.add_argument("--reid-thresh", type=float, default=0.65,
                   help="Cosine-similarity threshold for a positive cross-camera match.")
    p.add_argument("--reid-ttl", type=int, default=10,
                   help="Embedding TTL in seconds for Valkey store.")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Process every Nth frame (1 = all frames).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many frames (useful for quick demos).")
    p.add_argument("--trails", dest="trails", default=True, action="store_true",
                   help="Draw track trail overlays.")
    p.add_argument("--no-trails", dest="trails", action="store_false")
    p.add_argument("--trail-length", type=int, default=30)
    p.add_argument("--cell-size", default="640x360", metavar="WxH",
                   help="Output frame dimensions (width x height).")
    p.add_argument("--headless", action="store_true", default=False,
                   help="Suppress the live preview window.")
    p.add_argument("--save", default=None, metavar="PATH",
                   help="Write annotated output to this .mp4 path.")
    p.add_argument(
        "--save-crops", default=None, metavar="DIR",
        help=(
            "Save cropped images for each cross-camera ReID match into "
            "DIR/id_NNNN/ subfolders (requires --reid-model)."
        ),
    )
    p.add_argument("--max-crops-per-id", type=int, default=100,
                   help="Maximum crop images saved per global identity (0 = unlimited).")
    p.add_argument("--id-ttl-frames", type=int, default=30,
                   help="Evict a (camera, track_id) → global_id mapping after N frames of absence.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    mock_mode = args.model is None or args.video is None

    if args.save_crops and not args.reid_model:
        print(
            "[WARN] --save-crops has no effect without --reid-model "
            "(no cross-camera matches to crop).",
            file=sys.stderr,
        )

    if mock_mode:
        print(
            "Running in mock mode (synthetic detections, no real models).\n"
            "Pass --video <path> --model <id> to use real inference.\n"
        )

    # -----------------------------------------------------------------------
    # Load or mock models
    # -----------------------------------------------------------------------
    if mock_mode:
        detector = _make_mock_detector()
        encoder  = _make_mock_encoder() if args.reid_model else None
    else:
        import mata
        print(f"Loading detector: {args.model}")
        detector = mata.load("detect", args.model)
        encoder  = None
        if args.reid_model:
            print(f"Loading ReID encoder: {args.reid_model}")
            encoder = mata.load("embed", args.reid_model)

    # -----------------------------------------------------------------------
    # Build tracker
    # -----------------------------------------------------------------------
    if args.tracker == "botsort":
        from mata.nodes.track import BotSortWrapper
        tracker = BotSortWrapper(track_buffer=30, frame_rate=30,
                                 track_thresh=args.conf)
    else:
        from mata.nodes.track import ByteTrackWrapper
        tracker = ByteTrackWrapper(track_buffer=30, frame_rate=30,
                                   track_thresh=args.conf)

    # -----------------------------------------------------------------------
    # Build ReIDBridge (cross-camera, optional)
    # -----------------------------------------------------------------------
    bridge = None
    if encoder is not None and args.valkey:
        bridge = _init_reid_bridge(
            args.camera_id, args.valkey, args.reid_thresh, args.reid_ttl,
        )
    elif encoder is not None:
        print(
            "[info] No --valkey URL supplied. ReID embeddings will not be "
            "shared across cameras (single-camera appearance mode only)."
        )

    # -----------------------------------------------------------------------
    # Assemble providers
    # -----------------------------------------------------------------------
    providers: dict[str, Any] = {"detector": detector, "tracker": tracker}
    if encoder is not None:
        providers["encoder"] = encoder
    if bridge is not None:
        providers["bridge"] = bridge

    # -----------------------------------------------------------------------
    # Compile graph
    # -----------------------------------------------------------------------
    try:
        cell_w, cell_h = [int(x) for x in args.cell_size.lower().split("x")]
    except ValueError:
        cell_w, cell_h = 640, 360

    cam_label = f" {args.camera_id.upper()} "
    cam_color  = (60, 100, 255)  # blue — override per camera if needed

    graph = _build_graph(
        name="reid_pipeline",
        conf=args.conf,
        has_encoder=(encoder is not None),
        cam_label=cam_label,
        cam_color=cam_color,
        show_trails=args.trails,
        trail_length=args.trail_length,
    )
    print(f"Graph compiled ({len(graph._nodes)} nodes)")

    # -----------------------------------------------------------------------
    # Mock-mode: run a short synthetic loop and exit
    # -----------------------------------------------------------------------
    if mock_mode:
        _run_mock_loop(graph, providers, args)
        return

    # -----------------------------------------------------------------------
    # Real-mode: process video file
    # -----------------------------------------------------------------------
    import cv2
    from mata.core.graph.temporal import FramePolicyEveryN

    vid_path = Path(args.video)
    if not vid_path.exists():
        print(f"[error] Video not found: {vid_path}", file=sys.stderr)
        sys.exit(1)

    _cap_probe = cv2.VideoCapture(str(vid_path))
    if not _cap_probe.isOpened():
        print(f"[error] Cannot open: {vid_path}", file=sys.stderr)
        sys.exit(1)
    fps = _cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    _cap_probe.release()

    writer: Any = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (cell_w, cell_h))
        print(f"Saving output to: {args.save}")

    stop = threading.Event()
    frame_count: list[int] = [0]
    crops_saved: list[int] = [0]
    crops_per_id: dict[int, int] = {}
    global_id_registry = GlobalIDRegistry(ttl_frames=args.id_ttl_frames)

    def _on_frame(result: Any, frame_num: int, frame_bgr: Any) -> None:
        frame_count[0] = frame_num
        annotated_art = result.channels.get("annotated")
        frame_out = cv2.resize(
            annotated_art.to_numpy() if annotated_art is not None else frame_bgr,
            (cell_w, cell_h),
        )

        tracks_art = result.channels.get("tracks")
        n_active = (
            len(tracks_art.get_active_tracks().tracks) if tracks_art else 0
        )
        cross_art = result.channels.get("cross_matches")
        n_xcam = len(cross_art) if cross_art is not None else 0
        print(
            f"\r  frame={frame_num:5d}  active={n_active}  "
            f"xcam={n_xcam}  crops={crops_saved[0]}   ",
            end="", flush=True,
        )

        # Save identity crops for cross-camera ReID matches
        if args.save_crops and cross_art is not None:
            fh, fw = frame_bgr.shape[:2]
            track_bbox: dict[int, tuple] = {}
            if tracks_art is not None:
                for t in tracks_art.get_active_tracks().tracks:
                    if t.track_id is not None and t.bbox is not None:
                        track_bbox[t.track_id] = t.bbox

            active_keys: set[tuple[str, int]] = set()
            for match in cross_art.matches:
                local_tid = match.local_track_id
                active_keys.add((args.camera_id, local_tid))
                bbox = track_bbox.get(local_tid)
                if bbox is None:
                    continue
                gid = global_id_registry.resolve(
                    args.camera_id, local_tid,
                    match.remote_camera_id, match.remote_track_id,
                )
                if gid == -1:
                    continue
                limit = args.max_crops_per_id
                if limit > 0 and crops_per_id.get(gid, 0) >= limit:
                    continue
                x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
                x2, y2 = min(fw, int(bbox[2])), min(fh, int(bbox[3]))
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_dir = Path(args.save_crops) / f"id_{gid:04d}"
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"{args.camera_id}_frame{frame_num:05d}_track{local_tid}.jpg"
                    cv2.imwrite(str(crop_dir / fname), crop)
                    crops_saved[0] += 1
                    crops_per_id[gid] = crops_per_id.get(gid, 0) + 1
            global_id_registry.tick(frame_num, active_keys)

        if writer is not None:
            writer.write(frame_out)

        if not args.headless:
            cv2.imshow(f"MATA Graph ReID — {args.camera_id}", frame_out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop.set()

    try:
        graph.run(
            str(vid_path),
            providers=providers,
            frame_policy=FramePolicyEveryN(n=args.frame_stride),
            max_frames=args.max_frames,
            callback=_on_frame,
            stop_event=stop,
        )
    finally:
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    summary = f"\nDone. ({frame_count[0]} frames processed)"
    if args.save_crops and crops_saved[0] > 0:
        n_ids = global_id_registry.num_global_ids
        summary += f"  |  crops saved: {crops_saved[0]} ({n_ids} unique identities)"
    print(summary)


# ---------------------------------------------------------------------------
# Mock loop (no video / model required)
# ---------------------------------------------------------------------------

def _run_mock_loop(graph: Any, providers: dict[str, Any], args: argparse.Namespace) -> None:
    """Run a short synthetic loop to demonstrate the graph structure."""
    import numpy as np
    from mata.core.graph.temporal import FramePolicyEveryN

    print("\n=== Mock Graph ReID Pipeline ===")
    print(f"Graph nodes: {[n.__class__.__name__ for n in graph._nodes]}\n")

    # Synthesise a few blank frames and pass them through the graph
    num_frames = 5
    for i in range(num_frames):
        frame_bgr = np.zeros((360, 640, 3), dtype=np.uint8)
        # Draw a moving rectangle to simulate a person
        x = 80 + i * 30
        frame_bgr[50:260, x:x+80] = (200, 150, 100)

        try:
            result = graph.infer(frame_bgr, providers=providers)
            tracks_art = result.channels.get("tracks")
            n_tracks = (
                len(tracks_art.get_active_tracks().tracks) if tracks_art else 0
            )
        except Exception:
            n_tracks = "N/A"

        print(f"  Frame {i+1}/{num_frames}: active_tracks={n_tracks}")

    print("\nMock run complete.")
    print(
        "\nTo run with a real video:\n"
        "  python examples/graph/graph_reid_pipeline.py \\\n"
        "      --video path/to/video.mp4 \\\n"
        "      --model facebook/detr-resnet-50 \\\n"
        "      --reid-model openai/clip-vit-base-patch32\n"
        "\nTo enable cross-camera ReID (requires a Valkey/Redis server):\n"
        "  python examples/graph/graph_reid_pipeline.py \\\n"
        "      --video path/to/cam1.mp4 \\\n"
        "      --model facebook/detr-resnet-50 \\\n"
        "      --reid-model openai/clip-vit-base-patch32 \\\n"
        "      --valkey valkey://localhost:6379 \\\n"
        "      --camera-id cam-1\n"
    )


if __name__ == "__main__":
    main()
