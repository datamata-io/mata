"""MATA vendored tracker implementations.

Provides ByteTrack and BotSort multi-object tracking algorithms,
ported from Ultralytics (MIT license).

Tracker hierarchy:
    BYTETracker   — two-stage IoU + Kalman tracking (Task A4 ✅)
    BOTSORT       — BYTETracker + GMC + optional ReID (Task A5 ✅)

Available (Tasks A3–A5):
    TrackState        — track lifecycle state enumeration
    BaseTrack         — abstract base with shared ID counter
    STrack            — single tracked object with KalmanFilterXYAH
    BOTrack           — STrack variant with KalmanFilterXYWH + ReID stubs
    DetectionResults  — adapter: VisionResult → tracker input format
    BYTETracker       — full two-stage ByteTrack algorithm
    BOTSORT           — BotSort: BYTETracker + GMC + optional ReID
"""

from __future__ import annotations

from mata.trackers.basetrack import BaseTrack, TrackState
from mata.trackers.bot_sort import BOTSORT, BOTrack
from mata.trackers.byte_tracker import BYTETracker, DetectionResults, STrack

__all__ = [
    "BaseTrack",
    "TrackState",
    "STrack",
    "BOTrack",
    "DetectionResults",
    "BYTETracker",
    "BOTSORT",
]
