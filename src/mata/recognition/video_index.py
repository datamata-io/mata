"""Video indexing helpers built on top of embed adapters and Gallery.

Provides a reusable way to sample a video, extract embeddings, and store them
in a Gallery with timestamp metadata for later search.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mata.core.artifacts.image import Image
from mata.core.video_io import get_video_info, iter_frames

from .gallery import Gallery, GalleryMatch

IndexMode = Literal["auto", "frame", "chunk"]
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class VideoMatch:
    """Single video-aware match result with resolved timing metadata."""

    label: str
    similarity: float
    index: int
    start_s: float
    end_s: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "label": self.label,
            "similarity": float(self.similarity),
            "index": self.index,
            "start_s": float(self.start_s),
            "end_s": float(self.end_s),
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)


@dataclass
class VideoIndex:
    """A Gallery plus timing metadata for video retrieval workflows."""

    gallery: Gallery
    frame_map: dict[str, float]
    end_map: dict[str, float]
    native_fps: float
    total_frames: int
    indexed_count: int
    mode: Literal["frame", "chunk"]

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[VideoMatch]:
        """Search the index and resolve timestamps for each match."""
        matches = self.gallery.search(query, top_k=top_k, threshold=threshold)
        return [self.resolve_match(match) for match in matches]

    def resolve_match(self, match: GalleryMatch) -> VideoMatch:
        """Attach timing metadata to a raw Gallery match."""
        return VideoMatch(
            label=match.label,
            similarity=match.similarity,
            index=match.index,
            start_s=float(self.frame_map[match.label]),
            end_s=float(self.end_map.get(match.label, self.frame_map[match.label])),
        )

    def save(self, path: str) -> None:
        """Persist the index to JSON or NPZ."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if out_path.suffix.lower() == ".npz":
            np.savez_compressed(
                out_path,
                payload_json=np.array([json.dumps(payload)], dtype=str),
            )
            return
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> VideoIndex:
        """Load an index from JSON or NPZ."""
        in_path = Path(path)
        if in_path.suffix.lower() == ".npz":
            data = np.load(in_path, allow_pickle=False)
            payload = json.loads(str(data["payload_json"][0]))
            return cls.from_dict(payload)
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "gallery": self.gallery.to_dict(),
            "frame_map": {key: float(value) for key, value in self.frame_map.items()},
            "end_map": {key: float(value) for key, value in self.end_map.items()},
            "native_fps": float(self.native_fps),
            "total_frames": int(self.total_frames),
            "indexed_count": int(self.indexed_count),
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoIndex:
        """Restore from :meth:`to_dict` output."""
        return cls(
            gallery=Gallery.from_dict(data["gallery"]),
            frame_map={key: float(value) for key, value in data["frame_map"].items()},
            end_map={key: float(value) for key, value in data.get("end_map", {}).items()},
            native_fps=float(data["native_fps"]),
            total_frames=int(data["total_frames"]),
            indexed_count=int(data["indexed_count"]),
            mode=data["mode"],
        )


def index_video(
    video_path: str,
    adapter: Any | None = None,
    *,
    model: str | None = None,
    mode: IndexMode = "auto",
    sample_fps: float = 1.0,
    chunk_stride: int = 89,
    chunk_frames: int = 15,
    progress: ProgressCallback | None = None,
    index: VideoIndex | None = None,
    gallery: Gallery | None = None,
    frame_map: dict[str, float] | None = None,
    end_map: dict[str, float] | None = None,
    id_prefix: str | None = None,
    **load_kwargs: Any,
) -> VideoIndex:
    """Index a video into a Gallery using frame or chunk sampling.

    Args:
        video_path: Path to a video file readable by ``iter_frames``.
        adapter: Pre-loaded ``mata.load("embed", ...)`` adapter.
        model: Model ID or alias. Used when ``adapter`` is omitted.
        mode: ``frame`` for image-per-sample indexing, ``chunk`` for sampled
            clip indexing, or ``auto`` to infer from adapter capabilities.
        sample_fps: Sampling rate for frame mode.
        chunk_stride: Window size in frames for chunk mode.
        chunk_frames: Number of frames sampled from each chunk window.
        progress: Optional callback receiving ``(indexed_count, estimated_total)``.
        index: Existing index to append to.
        gallery: Existing gallery to append to when ``index`` is not provided.
        frame_map: Existing label -> start time map for incremental indexing.
        end_map: Existing label -> end time map for incremental indexing.
        id_prefix: Optional label prefix. Defaults to ``frame`` or ``chunk``.
        **load_kwargs: Forwarded to ``mata.load("embed", ...)``.

    Returns:
        VideoIndex with populated gallery and timing metadata.
    """

    if adapter is None:
        if model is None:
            raise ValueError("index_video() requires either adapter= or model=")
        from mata import load

        adapter = load("embed", model, **load_kwargs)

    resolved_mode = _resolve_mode(adapter, mode)
    info = get_video_info(video_path)
    native_fps = float(info["fps"])
    total_frames = int(info["frame_count"])
    est_total = _estimate_total(total_frames, native_fps, resolved_mode, sample_fps, chunk_stride)

    if index is not None:
        target_gallery = index.gallery
        target_frame_map = dict(index.frame_map)
        target_end_map = dict(index.end_map)
    else:
        target_gallery = gallery if gallery is not None else Gallery()
        target_frame_map = dict(frame_map or {})
        target_end_map = dict(end_map or {})

    indexed_count = 0
    if resolved_mode == "frame":
        indexed_count = _index_video_frames(
            video_path=video_path,
            adapter=adapter,
            gallery=target_gallery,
            frame_map=target_frame_map,
            end_map=target_end_map,
            native_fps=native_fps,
            sample_fps=sample_fps,
            progress=progress,
            estimated_total=est_total,
            id_prefix=id_prefix or "frame",
        )
    else:
        indexed_count = _index_video_chunks(
            video_path=video_path,
            adapter=adapter,
            gallery=target_gallery,
            frame_map=target_frame_map,
            end_map=target_end_map,
            native_fps=native_fps,
            chunk_stride=chunk_stride,
            chunk_frames=chunk_frames,
            progress=progress,
            estimated_total=est_total,
            id_prefix=id_prefix or "chunk",
        )

    existing_count = index.indexed_count if index is not None else 0
    return VideoIndex(
        gallery=target_gallery,
        frame_map=target_frame_map,
        end_map=target_end_map,
        native_fps=native_fps,
        total_frames=total_frames,
        indexed_count=existing_count + indexed_count,
        mode=resolved_mode,
    )


def _resolve_mode(adapter: Any, mode: IndexMode) -> Literal["frame", "chunk"]:
    if mode == "frame":
        return "frame"
    if mode == "chunk":
        return "chunk"
    encoder = getattr(adapter, "_encoder", None)
    if encoder is not None and hasattr(encoder, "predict_video"):
        return "chunk"
    return "frame"


def _estimate_total(
    total_frames: int,
    native_fps: float,
    mode: Literal["frame", "chunk"],
    sample_fps: float,
    chunk_stride: int,
) -> int:
    if total_frames <= 0:
        return 0
    if mode == "frame":
        stride = max(1, int(native_fps / sample_fps))
        return int(np.ceil(total_frames / stride))
    return int(np.ceil(total_frames / max(1, chunk_stride)))


def _index_video_frames(
    *,
    video_path: str,
    adapter: Any,
    gallery: Gallery,
    frame_map: dict[str, float],
    end_map: dict[str, float],
    native_fps: float,
    sample_fps: float,
    progress: ProgressCallback | None,
    estimated_total: int,
    id_prefix: str,
) -> int:
    frame_stride = max(1, int(native_fps / sample_fps))
    indexed_count = 0

    for frame_idx, bgr_frame in iter_frames(video_path):
        if frame_idx % frame_stride != 0:
            continue

        emb = np.asarray(adapter.embed(Image.from_numpy(bgr_frame)), dtype=np.float32).ravel()
        label = _make_unique_label(frame_map, f"{id_prefix}_{frame_idx:06d}")
        timestamp_s = frame_idx / native_fps
        gallery.add(label, emb)
        frame_map[label] = timestamp_s
        end_map[label] = timestamp_s
        indexed_count += 1
        if progress is not None:
            progress(indexed_count, estimated_total)

    return indexed_count


def _index_video_chunks(
    *,
    video_path: str,
    adapter: Any,
    gallery: Gallery,
    frame_map: dict[str, float],
    end_map: dict[str, float],
    native_fps: float,
    chunk_stride: int,
    chunk_frames: int,
    progress: ProgressCallback | None,
    estimated_total: int,
    id_prefix: str,
) -> int:
    if chunk_stride <= 0:
        raise ValueError("chunk_stride must be > 0")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be > 0")

    frames_buf: list[np.ndarray] = []
    chunk_start = 0
    indexed_count = 0

    for frame_idx, bgr_frame in iter_frames(video_path):
        frames_buf.append(bgr_frame)
        if len(frames_buf) < chunk_stride:
            continue

        _add_chunk(
            adapter=adapter,
            gallery=gallery,
            frame_map=frame_map,
            end_map=end_map,
            frames=frames_buf,
            chunk_start=chunk_start,
            native_fps=native_fps,
            chunk_frames=chunk_frames,
            id_prefix=id_prefix,
        )
        indexed_count += 1
        if progress is not None:
            progress(indexed_count, estimated_total)
        chunk_start = frame_idx + 1
        frames_buf = []

    if frames_buf:
        _add_chunk(
            adapter=adapter,
            gallery=gallery,
            frame_map=frame_map,
            end_map=end_map,
            frames=frames_buf,
            chunk_start=chunk_start,
            native_fps=native_fps,
            chunk_frames=chunk_frames,
            id_prefix=id_prefix,
        )
        indexed_count += 1
        if progress is not None:
            progress(indexed_count, estimated_total)

    return indexed_count


def _add_chunk(
    *,
    adapter: Any,
    gallery: Gallery,
    frame_map: dict[str, float],
    end_map: dict[str, float],
    frames: list[np.ndarray],
    chunk_start: int,
    native_fps: float,
    chunk_frames: int,
    id_prefix: str,
) -> None:
    sample_indices = np.linspace(0, len(frames) - 1, min(chunk_frames, len(frames)), dtype=int)
    sampled = [frames[i] for i in sample_indices]
    emb = np.asarray(adapter.embed(sampled), dtype=np.float32).ravel()
    label = _make_unique_label(frame_map, f"{id_prefix}_{chunk_start:06d}")
    start_s = chunk_start / native_fps
    end_s = (chunk_start + max(0, len(frames) - 1)) / native_fps
    gallery.add(label, emb)
    frame_map[label] = start_s
    end_map[label] = end_s


def _make_unique_label(existing: dict[str, float], base_label: str) -> str:
    if base_label not in existing:
        return base_label
    suffix = 1
    while f"{base_label}_{suffix}" in existing:
        suffix += 1
    return f"{base_label}_{suffix}"
