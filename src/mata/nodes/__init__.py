"""Built-in nodes for MATA graph system.

Provides task execution nodes, data transformation nodes, and fusion nodes
for building computer vision processing graphs.
"""

from __future__ import annotations

# Visualization & Analysis nodes (Task 5.8)
from mata.nodes.annotate import Annotate
from mata.nodes.classify import Classify
from mata.nodes.depth import EstimateDepth

# Core task nodes (Task 5.1)
from mata.nodes.detect import Detect
from mata.nodes.expand_boxes import ExpandBoxes

# Data transformation nodes (Task 5.2)
from mata.nodes.filter import Filter

# Fusion nodes (Task 5.6)
from mata.nodes.fuse import Fuse
from mata.nodes.keep_best_mask import KeepBestMask
from mata.nodes.mask_to_box import MaskToBox
from mata.nodes.merge import Merge
from mata.nodes.nms import NMS
from mata.nodes.ocr import OCR
from mata.nodes.promote_entities import PromoteEntities
from mata.nodes.prompt_boxes import PromptBoxes
from mata.nodes.prompt_points import PromptPoints

# Mask refinement nodes (Task 5.4)
from mata.nodes.refine_mask import RefineMask
from mata.nodes.roi import ExtractROIs
from mata.nodes.segment import SegmentImage
from mata.nodes.segment_everything import SegmentEverything
from mata.nodes.topk import TopK

# Tracking nodes (Task 5.5)
from mata.nodes.track import Track

# Storage nodes (v1.10.0)
from mata.nodes.valkey_load import ValkeyLoad
from mata.nodes.valkey_store import ValkeyStore

# VLM nodes (Task 5.7)
from mata.nodes.vlm_describe import VLMDescribe
from mata.nodes.vlm_detect import VLMDetect
from mata.nodes.vlm_query import VLMQuery

__all__ = [
    # Core task nodes
    "Detect",
    "Classify",
    "SegmentImage",
    "EstimateDepth",
    # Data transformation nodes
    "Filter",
    "TopK",
    "ExtractROIs",
    "ExpandBoxes",
    "PromptBoxes",
    "PromptPoints",
    "SegmentEverything",
    # Mask refinement nodes
    "RefineMask",
    "MaskToBox",
    # Tracking nodes
    "Track",
    # Fusion nodes
    "Fuse",
    "Merge",
    "KeepBestMask",
    # VLM nodes
    "VLMDescribe",
    "VLMDetect",
    "VLMQuery",
    "PromoteEntities",
    # OCR nodes
    "OCR",
    # Visualization & Analysis nodes
    "Annotate",
    "NMS",
    # Storage nodes
    "ValkeyStore",
    "ValkeyLoad",
]
