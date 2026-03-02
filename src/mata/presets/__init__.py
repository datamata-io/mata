"""Pre-built graph presets for common multi-task workflows.

Provides factory functions that return configured :class:`~mata.core.graph.graph.Graph`
objects ready for compilation and execution via :func:`mata.infer`.

Traditional CV presets:
    - :func:`grounding_dino_sam` — Detection + SAM segmentation
    - :func:`detection_pose` — Detection + pose estimation
    - :func:`full_scene_analysis` — Parallel detection + classification + depth
    - :func:`detect_and_track` — Detection + BYTETrack tracking
    - :func:`segment_and_refine` — Segmentation + morphological refinement

General-purpose presets:
    - :func:`ensemble_detection` — Parallel dual-detector ensemble with NMS

Manufacturing presets:
    - :func:`defect_detect_classify` — Zero-shot defect detection + classification
    - :func:`assembly_verification` — VLM assembly inspection + component detection
    - :func:`component_inspection` — Detect components + VLM per-component analysis

Retail presets:
    - :func:`shelf_product_analysis` — Shelf product detection + brand/category classification
    - :func:`stock_level_analysis` — Parallel VLM + detection + classification for stock assessment

Autonomous driving presets:
    - :func:`vehicle_distance_estimation` — Parallel detection + depth for distance estimation
    - :func:`road_scene_analysis` — Complete road scene analysis with 4 parallel tasks
    - :func:`traffic_tracking` — Detection + tracking for traffic monitoring video
    - :func:`traffic_tracking_botsort` — Detection + BotSort tracking (GMC, moving cameras)

Security/Surveillance presets:
    - :func:`crowd_monitoring` — Person detection + tracking for crowd monitoring
    - :func:`crowd_monitoring_botsort` — Person detection + BotSort tracking (GMC, panning cameras)
    - :func:`suspicious_object_detection` — Zero-shot detection + SAM + VLM reasoning

Agriculture presets:
    - :func:`aerial_crop_analysis` — Parallel segmentation + depth for aerial crop surveying

VLM presets:
    - :func:`vlm_grounded_detection` — VLM semantic detection + spatial grounding
    - :func:`vlm_scene_understanding` — Parallel VLM description + detection + depth
    - :func:`vlm_multi_image_comparison` — Multi-image VLM comparison

Example:
    >>> import mata
    >>> from mata.presets import grounding_dino_sam
    >>>
    >>> detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
    >>> segmenter = mata.load("segment", "facebook/sam-vit-base")
    >>> result = mata.infer(
    ...     "image.jpg",
    ...     grounding_dino_sam(),
    ...     providers={"detector": detector, "segmenter": segmenter},
    ... )
"""

from __future__ import annotations

# Agriculture presets
from mata.presets.agriculture import aerial_crop_analysis
from mata.presets.detection_pose import detection_pose

# Traditional CV presets
from mata.presets.detection_segmentation import (
    grounding_dino_sam,
    segment_and_refine,
)

# Autonomous driving presets
from mata.presets.driving import (
    road_scene_analysis,
    traffic_tracking,
    traffic_tracking_botsort,
    vehicle_distance_estimation,
)
from mata.presets.full_scene import (
    detect_and_track,
    full_scene_analysis,
)

# General-purpose presets
from mata.presets.general import ensemble_detection

# Manufacturing presets
from mata.presets.manufacturing import (
    assembly_verification,
    component_inspection,
    defect_detect_classify,
)

# Retail presets
from mata.presets.retail import (
    shelf_product_analysis,
    stock_level_analysis,
)

# Security/Surveillance presets
from mata.presets.surveillance import (
    crowd_monitoring,
    crowd_monitoring_botsort,
    suspicious_object_detection,
)

# VLM presets
from mata.presets.vlm import (
    vlm_grounded_detection,
    vlm_multi_image_comparison,
    vlm_scene_understanding,
)

__all__ = [
    # Traditional CV presets
    "grounding_dino_sam",
    "segment_and_refine",
    "detection_pose",
    "full_scene_analysis",
    "detect_and_track",
    # General-purpose presets
    "ensemble_detection",
    # Manufacturing presets
    "defect_detect_classify",
    "assembly_verification",
    "component_inspection",
    # Retail presets
    "shelf_product_analysis",
    "stock_level_analysis",
    # Autonomous driving presets
    "vehicle_distance_estimation",
    "road_scene_analysis",
    "traffic_tracking",
    "traffic_tracking_botsort",
    # Agriculture presets
    "aerial_crop_analysis",
    # Security/Surveillance presets
    "crowd_monitoring",
    "crowd_monitoring_botsort",
    "suspicious_object_detection",
    # VLM presets
    "vlm_grounded_detection",
    "vlm_scene_understanding",
    "vlm_multi_image_comparison",
]
