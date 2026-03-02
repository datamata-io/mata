"""MultiResult artifact for graph system.

Provides unified result bundle with channel-based access and provenance tracking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mata.core.artifacts.base import Artifact


@dataclass(frozen=True)
class MultiResult(Artifact):
    """Channel-based multi-task result bundle.

    MultiResult serves as the unified output container for graph execution,
    providing:
    - Channel-based artifact storage (detections, masks, keypoints, etc.)
    - Dynamic attribute access (result.detections, result.masks)
    - Provenance tracking (models, graph config, versions, timestamps)
    - Metrics aggregation (per-node latency, memory usage)
    - Instance cross-referencing across channels

    Attributes:
        channels: Dictionary mapping channel names to artifacts
        provenance: Provenance metadata (model hashes, graph config, versions)
        metrics: Execution metrics (per-node timing, memory usage)
        meta: Optional additional metadata

    Examples:
        >>> # Create MultiResult with channels
        >>> from mata.core.artifacts import Detections, Masks, Image
        >>> result = MultiResult(
        ...     channels={
        ...         "image": image_artifact,
        ...         "detections": detections_artifact,
        ...         "masks": masks_artifact,
        ...     },
        ...     provenance={
        ...         "models": {"detector": "detr-resnet-50", "segmenter": "sam"},
        ...         "graph_hash": "abc123",
        ...         "timestamp": "2026-02-12T10:30:00",
        ...     },
        ...     metrics={
        ...         "detect_node": {"latency_ms": 45.2, "memory_mb": 512},
        ...         "segment_node": {"latency_ms": 120.5, "memory_mb": 1024},
        ...     }
        ... )
        >>>
        >>> # Access channels via attributes
        >>> dets = result.detections  # Same as result.channels["detections"]
        >>> masks = result.masks
        >>>
        >>> # Check channel existence
        >>> if result.has_channel("keypoints"):
        ...     kpts = result.keypoints
        >>>
        >>> # Get all artifacts for a specific instance
        >>> instance_data = result.get_instance_artifacts("inst_0000")
        >>> # Returns: {"detections": Instance(...), "masks": Instance(...)}
        >>>
        >>> # Serialize to JSON
        >>> json_str = result.to_json()
        >>>
        >>> # Deserialize from dict
        >>> result_copy = MultiResult.from_dict(result.to_dict())
    """

    channels: dict[str, Artifact] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Artifact:
        """Enable dynamic attribute access for channels.

        Allows accessing channels as attributes: result.detections instead of
        result.channels["detections"].

        Args:
            name: Channel name to retrieve

        Returns:
            Artifact from the specified channel

        Raises:
            AttributeError: If channel does not exist

        Examples:
            >>> result.detections  # Returns channels["detections"]
            >>> result.masks  # Returns channels["masks"]
            >>> result.nonexistent  # Raises AttributeError
        """
        # Prevent infinite recursion - only lookup in channels if it exists
        if "channels" in self.__dict__:
            channels = self.__dict__["channels"]
            if name in channels:
                return channels[name]

        raise AttributeError(
            f"No channel '{name}' in MultiResult. " f"Available channels: {list(self.channels.keys())}"
        )

    def has_channel(self, name: str) -> bool:
        """Check if a channel exists.

        Args:
            name: Channel name to check

        Returns:
            True if channel exists, False otherwise

        Examples:
            >>> result.has_channel("detections")  # True
            >>> result.has_channel("nonexistent")  # False
        """
        return name in self.channels

    def get_channel(self, name: str, default: Artifact | None = None) -> Artifact | None:
        """Get channel artifact with optional default.

        Args:
            name: Channel name to retrieve
            default: Default value if channel doesn't exist

        Returns:
            Artifact from channel or default value

        Examples:
            >>> dets = result.get_channel("detections")
            >>> masks = result.get_channel("masks", default=None)
        """
        return self.channels.get(name, default)

    def get_instance_artifacts(self, instance_id: str) -> dict[str, Any]:
        """Get all artifacts for a specific instance ID across channels.

        This method searches through all channels that contain instance-based
        artifacts (Detections, Masks, Keypoints) and collects data for the
        specified instance ID.

        Args:
            instance_id: Instance ID to search for

        Returns:
            Dictionary mapping channel names to instance data for that ID.
            Empty dict if instance ID not found in any channel.

        Examples:
            >>> # Get all data for instance "inst_0000"
            >>> data = result.get_instance_artifacts("inst_0000")
            >>> # Returns: {
            >>> #     "detections": Instance(bbox=..., label="cat"),
            >>> #     "masks": Instance(mask=..., label="cat"),
            >>> # }
        """
        instance_data = {}

        for channel_name, artifact in self.channels.items():
            # Check if artifact has instances and instance_ids
            if hasattr(artifact, "instances") and hasattr(artifact, "instance_ids"):
                try:
                    # Find index of instance_id
                    idx = artifact.instance_ids.index(instance_id)
                    # Get corresponding instance
                    instance_data[channel_name] = artifact.instances[idx]
                except (ValueError, IndexError):
                    # Instance ID not found in this channel, skip
                    continue

        return instance_data

    def list_instance_ids(self) -> set[str]:
        """Get all unique instance IDs across all channels.

        Returns:
            Set of all instance IDs found in any channel

        Examples:
            >>> ids = result.list_instance_ids()
            >>> # Returns: {"inst_0000", "inst_0001", "inst_0002"}
        """
        all_ids = set()

        for artifact in self.channels.values():
            if hasattr(artifact, "instance_ids"):
                all_ids.update(artifact.instance_ids)

        return all_ids

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all MultiResult data

        Note:
            Channel artifacts are serialized using their to_dict() methods.
        """
        return {
            "channels": {
                name: {
                    "type": artifact.__class__.__name__,
                    "data": artifact.to_dict(),
                }
                for name, artifact in self.channels.items()
            },
            "provenance": self.provenance,
            "metrics": self.metrics,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiResult:
        """Construct MultiResult from dictionary representation.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Reconstructed MultiResult instance

        Note:
            This requires artifact types to be registered and importable.
            For full deserialization, use the artifact registry to reconstruct
            each artifact from its type name and data.
        """
        from mata.core.artifacts.base import ArtifactTypeRegistry

        # Reconstruct channels
        channels = {}
        registry = ArtifactTypeRegistry()

        for name, channel_data in data.get("channels", {}).items():
            artifact_type_name = channel_data["type"]
            artifact_data = channel_data["data"]

            # Get artifact type from registry
            try:
                artifact_type = registry.get(artifact_type_name)
                # Reconstruct artifact using its from_dict method
                channels[name] = artifact_type.from_dict(artifact_data)
            except (KeyError, AttributeError):
                # If artifact type not found or doesn't have from_dict,
                # store raw data with warning
                import warnings

                warnings.warn(
                    f"Cannot deserialize artifact type '{artifact_type_name}' "
                    f"in channel '{name}'. Storing raw data.",
                    UserWarning,
                )
                channels[name] = artifact_data

        return cls(
            channels=channels,
            provenance=data.get("provenance", {}),
            metrics=data.get("metrics", {}),
            meta=data.get("meta", {}),
        )

    def to_json(self, indent: int | None = 2) -> str:
        """Convert to JSON string.

        Args:
            indent: JSON indentation (None for compact, int for pretty-print)

        Returns:
            JSON string representation

        Examples:
            >>> json_str = result.to_json()
            >>> json_compact = result.to_json(indent=None)
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> MultiResult:
        """Construct MultiResult from JSON string.

        Args:
            json_str: JSON string from to_json()

        Returns:
            Reconstructed MultiResult instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate(self) -> None:
        """Validate MultiResult structure.

        Raises:
            ValueError: If validation fails
        """
        # Validate channels are Artifacts (or skip if raw data during deserialization)
        for name in self.channels:
            artifact = self.channels[name]
            if not isinstance(artifact, (Artifact, dict)):
                raise ValueError(
                    f"Channel '{name}' contains invalid type {type(artifact)}. " f"Expected Artifact or dict."
                )

        # Validate provenance is dict
        if not isinstance(self.provenance, dict):
            raise ValueError(f"Provenance must be dict, got {type(self.provenance)}")

        # Validate metrics is dict
        if not isinstance(self.metrics, dict):
            raise ValueError(f"Metrics must be dict, got {type(self.metrics)}")

        # Validate meta is dict
        if not isinstance(self.meta, dict):
            raise ValueError(f"Meta must be dict, got {type(self.meta)}")

    def __repr__(self) -> str:
        """String representation."""
        channels_str = ", ".join(self.channels.keys())
        return (
            f"MultiResult("
            f"channels=[{channels_str}], "
            f"provenance={len(self.provenance)} keys, "
            f"metrics={len(self.metrics)} keys"
            f")"
        )
