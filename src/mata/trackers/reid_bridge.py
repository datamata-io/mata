"""Cross-camera ReID bridge via Valkey.

Publishes track embeddings to a shared Valkey store so that independent
tracker instances (different cameras/processes) can resolve identities
across feeds.

Data model:
    Key:   reid:{camera_id}:{track_id}
    Value: msgpack({
        "track_id": int,
        "camera_id": str,
        "embedding": list[float],     # smooth_feat (L2-normalised)
        "bbox": [x1, y1, x2, y2],
        "timestamp": float,
        "label": int
    })
    TTL:   configurable (default 300s)

Query:
    For an unmatched detection, fetch all active embeddings from other
    cameras, compute cosine similarity, return best match above threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mata.core.logging import get_logger

logger = get_logger(__name__)


class ReIDBridge:
    """Cross-camera ReID embedding store backed by Valkey.

    Args:
        url: Valkey connection URI (e.g., "valkey://localhost:6379/0").
        camera_id: Unique identifier for this camera/tracker instance.
        ttl: Time-to-live for published embeddings in seconds.
        similarity_thresh: Minimum cosine similarity for cross-camera match.
    """

    def __init__(
        self,
        url: str,
        camera_id: str = "default",
        ttl: int = 300,
        similarity_thresh: float = 0.25,
    ) -> None:
        from mata.core.exporters.valkey_exporter import _get_valkey_client

        self._client = _get_valkey_client(url)
        self._camera_id = camera_id
        self._ttl = ttl
        self._similarity_thresh = similarity_thresh
        self._prefix = "reid"

    def publish(
        self,
        track_id: int,
        embedding: np.ndarray,
        bbox: tuple[float, ...] | None = None,
        label: int = 0,
        timestamp: float | None = None,
    ) -> None:
        """Publish a track's embedding to Valkey.

        Args:
            track_id: Unique track identifier within this camera.
            embedding: L2-normalised feature vector (1-D float32 array).
            bbox: Optional bounding box (x1, y1, x2, y2) in pixel coords.
            label: Class label index (default 0).
            timestamp: Unix timestamp; defaults to ``time.time()``.
        """
        import time

        import msgpack

        key = f"{self._prefix}:{self._camera_id}:{track_id}"
        data = msgpack.packb(
            {
                "track_id": track_id,
                "camera_id": self._camera_id,
                "embedding": embedding.tolist(),
                "bbox": list(bbox) if bbox is not None else None,
                "label": label,
                "timestamp": timestamp if timestamp is not None else time.time(),
            }
        )
        try:
            self._client.set(key, data, ex=self._ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ReIDBridge.publish failed (camera=%s, track=%d): %s", self._camera_id, track_id, exc)

    def query(
        self,
        embedding: np.ndarray,
        exclude_camera: str | None = None,
        top_k: int = 1,
    ) -> list[dict[str, Any]]:
        """Find nearest embeddings from other cameras.

        Args:
            embedding: Query L2-normalised feature vector.
            exclude_camera: Camera ID to exclude from results (typically
                ``self.camera_id`` to prevent self-matching).
            top_k: Maximum number of results to return.

        Returns:
            List of matches sorted by similarity (descending), each a dict with
            keys: ``track_id``, ``camera_id``, ``similarity``, ``bbox``, ``label``.
        """
        import msgpack

        pattern = f"{self._prefix}:*"
        matches: list[dict[str, Any]] = []
        try:
            for key in self._client.scan_iter(match=pattern, count=100):
                raw = self._client.get(key)
                if raw is None:
                    continue
                entry = msgpack.unpackb(raw, raw=False)
                if exclude_camera and entry.get("camera_id") == exclude_camera:
                    continue
                stored_emb = np.array(entry["embedding"], dtype=np.float32)
                sim = float(np.dot(embedding, stored_emb))
                if sim >= self._similarity_thresh:
                    matches.append(
                        {
                            "track_id": entry["track_id"],
                            "camera_id": entry["camera_id"],
                            "similarity": sim,
                            "bbox": entry.get("bbox"),
                            "label": entry.get("label"),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ReIDBridge.query failed (camera=%s): %s", self._camera_id, exc)
            return []

        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return matches[:top_k]

    def clear(self, camera_id: str | None = None) -> int:
        """Remove published embeddings for a camera (or all cameras).

        Args:
            camera_id: Camera whose keys to delete.  Pass ``None`` to
                delete embeddings for **all** cameras.

        Returns:
            Number of keys deleted.
        """
        pattern = f"{self._prefix}:{camera_id if camera_id is not None else '*'}:*"
        count = 0
        try:
            for key in self._client.scan_iter(match=pattern, count=100):
                self._client.delete(key)
                count += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("ReIDBridge.clear failed (camera=%s): %s", camera_id, exc)
        return count

    @property
    def camera_id(self) -> str:
        """The camera ID associated with this bridge instance."""
        return self._camera_id

    @property
    def ttl(self) -> int:
        """TTL (seconds) applied to each published embedding key."""
        return self._ttl

    @property
    def similarity_thresh(self) -> float:
        """Minimum cosine similarity required for a match to be returned."""
        return self._similarity_thresh
