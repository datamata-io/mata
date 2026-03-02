"""Tracking adapter — composes detection + multi-object tracking.

Wraps any detection adapter with a vendored ByteTrack or BotSort tracker.
Returns VisionResult with Instance.track_id populated.

Usage::

    from mata.adapters.tracking_adapter import TrackingAdapter, TrackerConfig

    # Default BotSort config
    adapter = TrackingAdapter(detector)

    # ByteTrack by name
    adapter = TrackingAdapter(detector, tracker_config="bytetrack")

    # Custom config dict
    cfg = TrackerConfig(tracker_type="bytetrack", track_high_thresh=0.6)
    adapter = TrackingAdapter(detector, tracker_config=cfg)

    # Per-frame update (video loop)
    for frame in frames:
        result = adapter.update(frame)          # VisionResult with track_ids
    adapter.reset()                             # Start fresh sequence
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from mata.core.logging import get_logger
from mata.core.types import Instance, VisionResult
from mata.trackers.byte_tracker import DetectionResults  # re-export for convenience

logger = get_logger(__name__)

# Built-in config directory (same package: src/mata/trackers/configs/)
_BUILTIN_CONFIGS_DIR = Path(__file__).parent.parent / "trackers" / "configs"

# Mapping of short names → built-in YAML filenames
_BUILTIN_CONFIG_NAMES: dict[str, str] = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}


# ---------------------------------------------------------------------------
# TrackerConfig
# ---------------------------------------------------------------------------


@dataclass
class TrackerConfig:
    """Tracker configuration matching Ultralytics YAML format.

    Provides a type-safe container for all ByteTrack / BotSort parameters,
    plus factory class-methods to load from YAML files or plain dicts.

    Attributes:
        tracker_type: ``'bytetrack'`` or ``'botsort'``.
        track_high_thresh: High-confidence threshold — first-stage association.
        track_low_thresh: Low-confidence threshold — second-stage association.
        new_track_thresh: Minimum confidence to initialise a new track.
        track_buffer: Number of frames a lost track is kept before removal.
        match_thresh: IoU threshold for valid assignments.
        fuse_score: Fuse detection confidence into IoU cost matrix.
        gmc_method: Global motion compensation method (BotSort only).
            ``'sparseOptFlow'`` or ``None`` (disabled).
        proximity_thresh: Minimum IoU for ReID candidate set (BotSort only).
        appearance_thresh: Minimum cosine similarity for ReID (BotSort only).
        with_reid: Whether ReID matching is enabled.  ``False`` in v1.8.
    """

    tracker_type: str = "botsort"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True
    gmc_method: str | None = "sparseOptFlow"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    with_reid: bool = False

    # ------------------------------------------------------------------ #
    # Factory helpers                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: str) -> TrackerConfig:
        """Load configuration from a YAML file.

        Accepts:
        - A short name: ``"bytetrack"`` or ``"botsort"`` — resolves to
          the built-in config bundled with MATA.
        - An absolute or relative file path ending in ``.yaml`` /
          ``.yml``.

        Args:
            path: Built-in config name or path to a YAML file.

        Returns:
            :class:`TrackerConfig` populated from the file.

        Raises:
            FileNotFoundError: If the YAML file cannot be located.
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyYAML is required to load tracker configs. " "Install it with: pip install pyyaml"
            ) from exc

        # 1. Check short name aliases first.
        key = path.lower().removesuffix(".yaml").removesuffix(".yml")
        if key in _BUILTIN_CONFIG_NAMES:
            resolved = _BUILTIN_CONFIGS_DIR / _BUILTIN_CONFIG_NAMES[key]
        else:
            # 2. Treat as a raw file path.
            resolved = Path(path)

        if not resolved.is_file():
            builtin_names = list(_BUILTIN_CONFIG_NAMES.keys())
            raise FileNotFoundError(
                f"Tracker config not found: {path!r}.\n"
                f"Built-in names: {builtin_names}.\n"
                f"Resolved path: {resolved}"
            )

        with open(resolved) as fh:
            data = yaml.safe_load(fh) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict) -> TrackerConfig:
        """Create a :class:`TrackerConfig` from a plain dictionary.

        Unknown keys are silently ignored so that custom YAML files with
        extra annotations do not raise errors.

        Args:
            d: Dictionary of tracker parameters.

        Returns:
            :class:`TrackerConfig` with values from *d* (defaults for
            any missing keys).
        """
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# _resolve_config — internal helper
# ---------------------------------------------------------------------------


def _resolve_config(tracker_config: TrackerConfig | str | dict | None) -> TrackerConfig:
    """Normalise the *tracker_config* argument to a :class:`TrackerConfig`."""
    if tracker_config is None:
        return TrackerConfig()  # default (BotSort)
    if isinstance(tracker_config, TrackerConfig):
        return tracker_config
    if isinstance(tracker_config, str):
        return TrackerConfig.from_yaml(tracker_config)
    if isinstance(tracker_config, dict):
        return TrackerConfig.from_dict(tracker_config)
    raise TypeError(
        f"tracker_config must be TrackerConfig, str, dict, or None; " f"got {type(tracker_config).__name__}"
    )


# ---------------------------------------------------------------------------
# TrackingAdapter
# ---------------------------------------------------------------------------


class TrackingAdapter:
    """Stateful adapter that composes any detection adapter with a tracker.

    Accepts any object that exposes a ``predict(image, **kwargs) ->
    VisionResult`` method and wraps it with either ByteTrack or BotSort, both
    vendored in :mod:`mata.trackers`.

    The adapter is *stateful*: calling :meth:`update` multiple times on
    successive video frames maintains track continuity.  Call :meth:`reset`
    when starting a new video sequence.

    Args:
        detector: Detection adapter with a ``predict()`` method.
        tracker_config: One of:
            - :class:`TrackerConfig` instance,
            - ``str`` — built-in name ``"bytetrack"`` / ``"botsort"`` or
              path to a YAML file,
            - ``dict`` — passed to :meth:`TrackerConfig.from_dict`,
            - ``None`` — uses default BotSort configuration.
        frame_rate: Nominal video frame rate used to derive
            ``max_time_lost = frame_rate / 30 * track_buffer``.

    Example::

        adapter = TrackingAdapter(my_detector, tracker_config="bytetrack")
        for frame in video_frames:
            result = adapter.update(frame)
            for inst in result.instances:
                print(inst.track_id, inst.bbox)
        adapter.reset()
    """

    def __init__(
        self,
        detector: Any,
        tracker_config: TrackerConfig | str | dict | None = None,
        frame_rate: int = 30,
    ) -> None:
        self._detector = detector
        self._config: TrackerConfig = _resolve_config(tracker_config)
        self._frame_rate: int = int(frame_rate)
        self._tracker: Any = self._build_tracker()

        logger.debug(
            "TrackingAdapter initialised: tracker_type=%s, frame_rate=%d",
            self._config.tracker_type,
            self._frame_rate,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_tracker(self) -> Any:
        """Instantiate the underlying ByteTrack or BotSort tracker."""
        tracker_type = self._config.tracker_type.lower()
        if tracker_type == "bytetrack":
            from mata.trackers.byte_tracker import BYTETracker

            return BYTETracker(self._config, frame_rate=self._frame_rate)
        elif tracker_type == "botsort":
            from mata.trackers.bot_sort import BOTSORT

            return BOTSORT(self._config, frame_rate=self._frame_rate)
        else:
            raise ValueError(
                f"Unsupported tracker_type: {self._config.tracker_type!r}. " f"Choose from: 'bytetrack', 'botsort'."
            )

    @staticmethod
    def _to_numpy_image(image: Any) -> np.ndarray | None:
        """Convert an image to a uint8 numpy HWC array, or return None.

        Used to pass a frame to GMC in BotSort.  Returns *None* if conversion
        is not possible (e.g. image is a URL string), which causes the tracker
        to skip GMC gracefully.
        """
        try:
            if isinstance(image, np.ndarray):
                return image
            from PIL import Image as PILImage

            if isinstance(image, PILImage.Image):
                return np.array(image.convert("RGB"))
        except Exception:  # pragma: no cover
            pass
        return None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(
        self,
        image: Any,
        persist: bool = True,
        conf: float | None = None,
        iou: float | None = None,
        classes: list[int] | None = None,
        **kwargs: Any,
    ) -> VisionResult:
        """Run detection + tracking on a single frame.

        Pipeline: detect → (class filter) → convert → track → build result.

        Args:
            image: Input image — PIL ``Image``, numpy array, file path, or URL.
            persist: When ``True`` (default), tracker state is maintained
                across calls (use for video sequences).  When ``False``,
                :meth:`reset` is called *before* running detection so every
                call is independent.
            conf: Override detection confidence threshold.  Passed to
                ``detector.predict()`` if the adapter supports it.
            iou: Override NMS IoU threshold.  Passed to ``detector.predict()``
                if supported.
            classes: Filter detections to only these class IDs before feeding
                into the tracker.  ``None`` keeps all classes.
            **kwargs: Extra keyword arguments forwarded to
                ``detector.predict()``.

        Returns:
            :class:`~mata.core.types.VisionResult` whose ``instances`` list
            contains :class:`~mata.core.types.Instance` objects with
            ``track_id`` populated for all confirmed tracks.
        """
        if not persist:
            self.reset()

        # ---- 1. Build detect kwargs ----------------------------------------
        detect_kwargs: dict[str, Any] = dict(kwargs)
        if conf is not None:
            detect_kwargs["conf"] = conf
        if iou is not None:
            detect_kwargs["iou"] = iou

        # ---- 2. Run detector -----------------------------------------------
        vision_result: VisionResult = self._detector.predict(image, **detect_kwargs)

        # ---- 3. Optional class filter --------------------------------------
        if classes is not None and vision_result.instances:
            class_set = set(classes)
            filtered = [inst for inst in vision_result.instances if inst.label in class_set]
            vision_result = VisionResult(
                instances=filtered,
                meta=dict(vision_result.meta),
            )

        # ---- 4. Convert to tracker input format ----------------------------
        det_results = DetectionResults.from_vision_result(vision_result)

        # ---- 5. Optional numpy image for GMC (BotSort) ---------------------
        np_image = self._to_numpy_image(image)

        # ---- 6. Run tracker ------------------------------------------------
        tracked: np.ndarray = self._tracker.update(det_results, img=np_image)

        # ---- 7. Build output VisionResult ----------------------------------
        id2label = getattr(self._detector, "id2label", None)
        result = self._convert_tracker_output(tracked, id2label)

        return result

    def _convert_tracker_output(
        self,
        tracked: np.ndarray,
        id2label: dict[int, str] | None,
    ) -> VisionResult:
        """Convert tracker output array to a :class:`VisionResult`.

        Args:
            tracked: ``(N, 8)`` float array output from
                :meth:`BYTETracker.update` / :meth:`BOTSORT.update`, where
                each row is ``[x1, y1, x2, y2, track_id, score, cls, idx]``.
            id2label: Optional ``{class_id: label_name}`` mapping sourced
                from the wrapped detector.

        Returns:
            :class:`VisionResult` with one :class:`Instance` per tracked
            object, each carrying ``track_id``.
        """
        if tracked is None or len(tracked) == 0:
            return VisionResult(instances=[], meta={"source": "tracking_adapter"})

        instances: list[Instance] = []
        for row in tracked:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            track_id = int(row[4])
            score = float(row[5])
            cls_id = int(row[6])

            # Resolve label name from detector's id2label or fallback.
            if id2label is not None:
                label_name: str | None = id2label.get(cls_id, f"class_{cls_id}")
            else:
                label_name = f"class_{cls_id}"

            inst = Instance(
                bbox=(x1, y1, x2, y2),
                score=score,
                label=cls_id,
                label_name=label_name,
                track_id=track_id,
            )
            instances.append(inst)

        return VisionResult(
            instances=instances,
            meta={"source": "tracking_adapter"},
        )

    def reset(self) -> None:
        """Reset tracker state for a new video sequence.

        Clears all track history and resets the global track-ID counter so
        track IDs restart from 1 for the next sequence.
        """
        self._tracker.reset()
        logger.debug("TrackingAdapter reset: tracker state cleared.")

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def tracker_type(self) -> str:
        """Tracker algorithm name: ``'bytetrack'`` or ``'botsort'``."""
        return self._config.tracker_type

    @property
    def id2label(self) -> dict[int, str] | None:
        """Class ID → label mapping from the wrapped detector (or ``None``)."""
        return getattr(self._detector, "id2label", None)

    @property
    def config(self) -> TrackerConfig:
        """The resolved :class:`TrackerConfig` used by this adapter."""
        return self._config

    def __repr__(self) -> str:
        return (
            f"TrackingAdapter("
            f"detector={type(self._detector).__name__}, "
            f"tracker={self._config.tracker_type}, "
            f"frame_rate={self._frame_rate})"
        )
