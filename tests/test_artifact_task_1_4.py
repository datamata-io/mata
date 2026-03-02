"""Unit tests for Masks, Keypoints, Tracks, and ROIs artifacts.

Tests artifact functionality including creation, validation, conversions,
and edge case handling.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image as PILImage

from mata.core.artifacts.keypoints import Keypoints
from mata.core.artifacts.masks import Masks
from mata.core.artifacts.rois import ROIs
from mata.core.artifacts.tracks import Track, Tracks
from mata.core.types import Instance, VisionResult

# ============================================================================
# Masks Artifact Tests
# ============================================================================


class TestMasksArtifact:
    """Test Masks artifact functionality."""

    def test_masks_creation_from_vision_result(self):
        """Test creating Masks from VisionResult with RLE masks."""
        # Create RLE mask
        rle_mask = {"size": [100, 100], "counts": "test_rle_encoded_data"}

        instances = [
            Instance(bbox=(10, 20, 50, 60), mask=rle_mask, score=0.9, label=0, label_name="cat"),
            Instance(bbox=(100, 150, 200, 250), mask=rle_mask, score=0.85, label=1, label_name="dog"),
        ]

        result = VisionResult(instances=instances)
        masks = Masks.from_vision_result(result)

        assert len(masks.instances) == 2
        assert len(masks.instance_ids) == 2
        assert masks.instance_ids[0] == "mask_0000"
        assert masks.instance_ids[1] == "mask_0001"
        assert masks.instances[0].label_name == "cat"
        assert masks.instances[1].label_name == "dog"

    def test_masks_auto_id_generation(self):
        """Test automatic instance ID generation."""
        rle_mask = {"size": [100, 100], "counts": "test"}

        instances = [
            Instance(mask=rle_mask, score=0.9, label=0),
            Instance(mask=rle_mask, score=0.8, label=1),
        ]

        masks = Masks(instances=instances)

        assert len(masks.instance_ids) == 2
        assert masks.instance_ids[0] == "mask_0000"
        assert masks.instance_ids[1] == "mask_0001"

    def test_masks_validation_no_mask(self):
        """Test validation fails when instance has no mask."""
        instances = [Instance(bbox=(10, 20, 50, 60), score=0.9, label=0)]  # No mask

        with pytest.raises(ValueError, match="has no mask data"):
            Masks(instances=instances, instance_ids=["test"])

    def test_masks_validation_length_mismatch(self):
        """Test validation fails when IDs don't match instances."""
        rle_mask = {"size": [100, 100], "counts": "test"}
        instances = [Instance(mask=rle_mask, score=0.9, label=0)]

        with pytest.raises(ValueError, match="length mismatch"):
            Masks(instances=instances, instance_ids=["id1", "id2"])

    def test_masks_to_vision_result(self):
        """Test conversion back to VisionResult."""
        rle_mask = {"size": [100, 100], "counts": "test"}
        instances = [Instance(mask=rle_mask, score=0.9, label=0)]

        masks = Masks(instances=instances)
        result = masks.to_vision_result()

        assert len(result.instances) == 1
        assert "instance_ids" in result.meta
        assert result.meta["instance_ids"] == masks.instance_ids

    def test_masks_to_dict_from_dict(self):
        """Test serialization round-trip."""
        rle_mask = {"size": [100, 100], "counts": "test"}
        instances = [Instance(mask=rle_mask, score=0.9, label=0, label_name="cat")]

        masks = Masks(instances=instances, meta={"test": "data"})
        data = masks.to_dict()

        assert "instances" in data
        assert "instance_ids" in data
        assert "meta" in data
        assert data["meta"]["test"] == "data"

        # Round-trip
        masks2 = Masks.from_dict(data)
        assert len(masks2.instances) == 1
        assert masks2.instance_ids == masks.instance_ids

    def test_masks_empty_result(self):
        """Test handling of VisionResult with no masks."""
        result = VisionResult(instances=[Instance(bbox=(10, 20, 50, 60), score=0.9, label=0)])  # No mask

        with pytest.raises(ValueError, match="no instances with masks"):
            Masks.from_vision_result(result)

    def test_masks_validate_method(self):
        """Test validate() method."""
        rle_mask = {"size": [100, 100], "counts": "test"}
        instances = [Instance(mask=rle_mask, score=0.9, label=0)]

        masks = Masks(instances=instances)
        masks.validate()  # Should not raise

        # Empty masks should fail
        masks_empty = Masks(instances=[], instance_ids=[])
        with pytest.raises(ValueError, match="at least one instance"):
            masks_empty.validate()


# ============================================================================
# Keypoints Artifact Tests
# ============================================================================


class TestKeypointsArtifact:
    """Test Keypoints artifact functionality."""

    def test_keypoints_creation(self):
        """Test creating Keypoints with valid data."""
        kp1 = np.array([[100.0, 200.0, 0.9], [110.0, 220.0, 0.8], [120.0, 240.0, 0.95]], dtype=np.float32)

        kp2 = np.array([[300.0, 250.0, 0.85], [310.0, 270.0, 0.9], [320.0, 290.0, 0.75]], dtype=np.float32)

        skeleton = [(0, 1), (1, 2)]

        keypoints = Keypoints(keypoints=[kp1, kp2], skeleton=skeleton, meta={"dataset": "coco"})

        assert len(keypoints.keypoints) == 2
        assert len(keypoints.instance_ids) == 2
        assert keypoints.instance_ids[0] == "kp_0000"
        assert keypoints.instance_ids[1] == "kp_0001"
        assert keypoints.skeleton == skeleton
        assert keypoints.meta["dataset"] == "coco"

    def test_keypoints_auto_id_generation(self):
        """Test automatic instance ID generation."""
        kp = np.array([[100.0, 200.0, 0.9]], dtype=np.float32)

        keypoints = Keypoints(keypoints=[kp, kp])

        assert len(keypoints.instance_ids) == 2
        assert keypoints.instance_ids[0] == "kp_0000"
        assert keypoints.instance_ids[1] == "kp_0001"

    def test_keypoints_validation_shape(self):
        """Test validation of keypoint array shapes."""
        # Wrong number of dimensions
        with pytest.raises(ValueError, match="2D array"):
            kp_bad = np.array([100.0, 200.0, 0.9])  # 1D
            Keypoints(keypoints=[kp_bad], instance_ids=["test"])

        # Wrong number of columns
        with pytest.raises(ValueError, match="3 columns"):
            kp_bad = np.array([[100.0, 200.0]], dtype=np.float32)  # Only 2 cols
            Keypoints(keypoints=[kp_bad], instance_ids=["test"])

    def test_keypoints_validation_scores(self):
        """Test validation of keypoint scores."""
        # Scores outside [0, 1] range
        with pytest.raises(ValueError, match="scores outside"):
            kp_bad = np.array([[100.0, 200.0, 1.5]], dtype=np.float32)
            Keypoints(keypoints=[kp_bad], instance_ids=["test"])

        with pytest.raises(ValueError, match="scores outside"):
            kp_bad = np.array([[100.0, 200.0, -0.1]], dtype=np.float32)
            Keypoints(keypoints=[kp_bad], instance_ids=["test"])

    def test_keypoints_validation_skeleton(self):
        """Test validation of skeleton connections."""
        kp = np.array([[100.0, 200.0, 0.9]], dtype=np.float32)

        # Invalid skeleton index
        with pytest.raises(ValueError, match="out of range"):
            Keypoints(keypoints=[kp], skeleton=[(0, 5)], instance_ids=["test"])  # Index 5 doesn't exist

    def test_keypoints_filter_by_visibility(self):
        """Test filtering by visibility threshold."""
        kp = np.array(
            [[100.0, 200.0, 0.9], [110.0, 220.0, 0.3], [120.0, 240.0, 0.95]],  # Visible  # Below threshold  # Visible
            dtype=np.float32,
        )

        keypoints = Keypoints(keypoints=[kp])
        filtered = keypoints.filter_by_visibility(threshold=0.5)

        # Low score should be zeroed
        assert filtered.keypoints[0][1, 2] == 0.0  # Second keypoint score
        assert filtered.keypoints[0][0, 2] == 0.9  # First unchanged
        assert filtered.keypoints[0][2, 2] == 0.95  # Third unchanged

    def test_keypoints_get_visible(self):
        """Test getting visible keypoints."""
        kp = np.array([[100.0, 200.0, 0.9], [110.0, 220.0, 0.3], [120.0, 240.0, 0.95]], dtype=np.float32)

        keypoints = Keypoints(keypoints=[kp])
        visible = keypoints.get_visible_keypoints(0, threshold=0.5)

        assert len(visible) == 2  # Only 2 above threshold
        assert visible[0, 2] == 0.9
        assert visible[1, 2] == 0.95

    def test_keypoints_count_visible(self):
        """Test counting visible keypoints."""
        kp1 = np.array([[100.0, 200.0, 0.9], [110.0, 220.0, 0.3], [120.0, 240.0, 0.95]], dtype=np.float32)

        kp2 = np.array([[100.0, 200.0, 0.8], [110.0, 220.0, 0.7], [120.0, 240.0, 0.2]], dtype=np.float32)

        keypoints = Keypoints(keypoints=[kp1, kp2])
        counts = keypoints.count_visible(threshold=0.5)

        assert counts == [2, 2]  # Both instances have 2 visible

    def test_keypoints_to_dict_from_dict(self):
        """Test serialization round-trip."""
        kp = np.array([[100.0, 200.0, 0.9], [110.0, 220.0, 0.8]], dtype=np.float32)
        skeleton = [(0, 1)]

        keypoints = Keypoints(keypoints=[kp], skeleton=skeleton, meta={"test": "data"})

        data = keypoints.to_dict()
        keypoints2 = Keypoints.from_dict(data)

        assert len(keypoints2.keypoints) == 1
        assert keypoints2.keypoints[0].shape == kp.shape
        np.testing.assert_array_equal(keypoints2.keypoints[0], kp)
        assert keypoints2.skeleton == skeleton
        assert keypoints2.meta["test"] == "data"

    def test_keypoints_validate_method(self):
        """Test validate() method."""
        kp = np.array([[100.0, 200.0, 0.9]], dtype=np.float32)
        keypoints = Keypoints(keypoints=[kp])

        keypoints.validate()  # Should not raise

        # Empty should fail
        keypoints_empty = Keypoints(keypoints=[], instance_ids=[])
        with pytest.raises(ValueError, match="at least one instance"):
            keypoints_empty.validate()


# ============================================================================
# Tracks Artifact Tests
# ============================================================================


class TestTracksArtifact:
    """Test Tracks artifact functionality."""

    def test_track_creation(self):
        """Test creating a single Track."""
        track = Track(track_id=5, bbox=(100.0, 200.0, 150.0, 250.0), score=0.95, label="person", age=10, state="active")

        assert track.track_id == 5
        assert track.bbox == (100.0, 200.0, 150.0, 250.0)
        assert track.score == 0.95
        assert track.label == "person"
        assert track.age == 10
        assert track.state == "active"

    def test_track_validation_score(self):
        """Test Track score validation."""
        with pytest.raises(ValueError, match="Score must be in"):
            Track(track_id=1, bbox=(10.0, 20.0, 50.0, 60.0), score=1.5, label="cat")  # Invalid

    def test_track_validation_age(self):
        """Test Track age validation."""
        with pytest.raises(ValueError, match="Age must be"):
            Track(track_id=1, bbox=(10.0, 20.0, 50.0, 60.0), score=0.9, label="cat", age=0)  # Invalid

    def test_track_validation_state(self):
        """Test Track state validation."""
        with pytest.raises(ValueError, match="State must be"):
            Track(track_id=1, bbox=(10.0, 20.0, 50.0, 60.0), score=0.9, label="cat", state="invalid")

    def test_track_validation_bbox(self):
        """Test Track bbox validation."""
        with pytest.raises(ValueError, match="Invalid bbox"):
            Track(track_id=1, bbox=(50.0, 60.0, 10.0, 20.0), score=0.9, label="cat")  # x2 < x1

    def test_tracks_creation(self):
        """Test creating Tracks artifact."""
        track1 = Track(track_id=1, bbox=(10.0, 20.0, 50.0, 60.0), score=0.9, label="car", age=5)

        track2 = Track(track_id=2, bbox=(100.0, 150.0, 140.0, 190.0), score=0.85, label="person", age=3, state="lost")

        tracks = Tracks(tracks=[track1, track2], frame_id="frame_0042", meta={"video": "traffic.mp4"})

        assert len(tracks.tracks) == 2
        assert tracks.frame_id == "frame_0042"
        assert tracks.meta["video"] == "traffic.mp4"

    def test_tracks_validation_frame_id(self):
        """Test Tracks requires frame_id."""
        track = Track(track_id=1, bbox=(10.0, 20.0, 50.0, 60.0), score=0.9, label="cat")

        with pytest.raises(ValueError, match="frame_id must be provided"):
            Tracks(tracks=[track], frame_id="")

    def test_tracks_validation_duplicate_ids(self):
        """Test Tracks rejects duplicate track IDs."""
        track1 = Track(track_id=1, bbox=(10.0, 20.0, 50.0, 60.0), score=0.9, label="cat")

        track2 = Track(track_id=1, bbox=(100.0, 150.0, 140.0, 190.0), score=0.85, label="dog")  # Duplicate

        with pytest.raises(ValueError, match="Duplicate track IDs"):
            Tracks(tracks=[track1, track2], frame_id="frame_001")

    def test_tracks_get_active(self):
        """Test filtering active tracks."""
        track1 = Track(track_id=1, bbox=(10, 20, 50, 60), score=0.9, label="car", state="active")
        track2 = Track(track_id=2, bbox=(100, 150, 140, 190), score=0.85, label="person", state="lost")
        track3 = Track(track_id=3, bbox=(200, 250, 240, 290), score=0.8, label="bike", state="active")

        tracks = Tracks(tracks=[track1, track2, track3], frame_id="frame_001")
        active = tracks.get_active_tracks()

        assert len(active.tracks) == 2
        assert active.tracks[0].track_id == 1
        assert active.tracks[1].track_id == 3

    def test_tracks_get_lost(self):
        """Test filtering lost tracks."""
        track1 = Track(track_id=1, bbox=(10, 20, 50, 60), score=0.9, label="car", state="active")
        track2 = Track(track_id=2, bbox=(100, 150, 140, 190), score=0.85, label="person", state="lost")

        tracks = Tracks(tracks=[track1, track2], frame_id="frame_001")
        lost = tracks.get_lost_tracks()

        assert len(lost.tracks) == 1
        assert lost.tracks[0].track_id == 2

    def test_tracks_get_by_id(self):
        """Test getting track by ID."""
        track1 = Track(track_id=1, bbox=(10, 20, 50, 60), score=0.9, label="car")
        track2 = Track(track_id=2, bbox=(100, 150, 140, 190), score=0.85, label="person")

        tracks = Tracks(tracks=[track1, track2], frame_id="frame_001")

        found = tracks.get_track_by_id(2)
        assert found is not None
        assert found.track_id == 2
        assert found.label == "person"

        not_found = tracks.get_track_by_id(99)
        assert not_found is None

    def test_tracks_filter_by_label(self):
        """Test filtering tracks by label."""
        track1 = Track(track_id=1, bbox=(10, 20, 50, 60), score=0.9, label="car")
        track2 = Track(track_id=2, bbox=(100, 150, 140, 190), score=0.85, label="person")
        track3 = Track(track_id=3, bbox=(200, 250, 240, 290), score=0.8, label="car")

        tracks = Tracks(tracks=[track1, track2, track3], frame_id="frame_001")
        cars = tracks.filter_by_label(["car"])

        assert len(cars.tracks) == 2
        assert all(t.label == "car" for t in cars.tracks)

    def test_tracks_filter_by_score(self):
        """Test filtering tracks by score."""
        track1 = Track(track_id=1, bbox=(10, 20, 50, 60), score=0.9, label="car")
        track2 = Track(track_id=2, bbox=(100, 150, 140, 190), score=0.85, label="person")
        track3 = Track(track_id=3, bbox=(200, 250, 240, 290), score=0.7, label="bike")

        tracks = Tracks(tracks=[track1, track2, track3], frame_id="frame_001")
        filtered = tracks.filter_by_score(0.8)

        assert len(filtered.tracks) == 2
        assert all(t.score >= 0.8 for t in filtered.tracks)

    def test_tracks_filter_by_age(self):
        """Test filtering tracks by age."""
        track1 = Track(track_id=1, bbox=(10, 20, 50, 60), score=0.9, label="car", age=10)
        track2 = Track(track_id=2, bbox=(100, 150, 140, 190), score=0.85, label="person", age=3)
        track3 = Track(track_id=3, bbox=(200, 250, 240, 290), score=0.8, label="bike", age=5)

        tracks = Tracks(tracks=[track1, track2, track3], frame_id="frame_001")
        filtered = tracks.filter_by_age(5)

        assert len(filtered.tracks) == 2
        assert all(t.age >= 5 for t in filtered.tracks)

    def test_tracks_to_dict_from_dict(self):
        """Test serialization round-trip."""
        track = Track(
            track_id=1,
            bbox=(10.0, 20.0, 50.0, 60.0),
            score=0.9,
            label="car",
            age=5,
            state="active",
            history=[(5, 10, 45, 55), (8, 15, 48, 58)],
        )

        tracks = Tracks(tracks=[track], frame_id="frame_001", meta={"test": "data"})

        data = tracks.to_dict()
        tracks2 = Tracks.from_dict(data)

        assert len(tracks2.tracks) == 1
        assert tracks2.tracks[0].track_id == 1
        assert tracks2.tracks[0].bbox == track.bbox
        assert tracks2.tracks[0].history == track.history
        assert tracks2.frame_id == "frame_001"
        assert tracks2.meta["test"] == "data"


# ============================================================================
# ROIs Artifact Tests
# ============================================================================


class TestROIsArtifact:
    """Test ROIs artifact functionality."""

    def test_rois_creation_pil(self):
        """Test creating ROIs with PIL images."""
        roi1 = PILImage.new("RGB", (50, 50), color=(255, 0, 0))
        roi2 = PILImage.new("RGB", (60, 60), color=(0, 255, 0))

        rois = ROIs(
            roi_images=[roi1, roi2],
            source_boxes=[(10, 20, 60, 70), (100, 150, 160, 210)],
            meta={"source_width": 640, "source_height": 480},
        )

        assert len(rois.roi_images) == 2
        assert len(rois.instance_ids) == 2
        assert rois.instance_ids[0] == "roi_0000"
        assert rois.instance_ids[1] == "roi_0001"
        assert len(rois.source_boxes) == 2

    def test_rois_creation_numpy(self):
        """Test creating ROIs with numpy arrays."""
        roi1 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        roi2 = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)

        rois = ROIs(roi_images=[roi1, roi2], source_boxes=[(10, 20, 60, 70), (100, 150, 160, 210)])

        assert len(rois.roi_images) == 2
        assert isinstance(rois.roi_images[0], np.ndarray)

    def test_rois_auto_id_generation(self):
        """Test automatic instance ID generation."""
        roi = PILImage.new("RGB", (50, 50))

        rois = ROIs(roi_images=[roi, roi], source_boxes=[(10, 20, 60, 70), (100, 150, 160, 210)])

        assert len(rois.instance_ids) == 2
        assert rois.instance_ids[0] == "roi_0000"
        assert rois.instance_ids[1] == "roi_0001"

    def test_rois_validation_length_mismatch(self):
        """Test validation fails on length mismatch."""
        roi = PILImage.new("RGB", (50, 50))

        with pytest.raises(ValueError, match="length mismatch"):
            ROIs(roi_images=[roi], source_boxes=[(10, 20, 60, 70), (100, 150, 160, 210)])  # 2 boxes, 1 ROI

    def test_rois_validation_invalid_box(self):
        """Test validation fails on invalid bbox."""
        roi = PILImage.new("RGB", (50, 50))

        with pytest.raises(ValueError, match="invalid coordinates"):
            ROIs(roi_images=[roi], source_boxes=[(60, 70, 10, 20)])  # x2 < x1

    def test_rois_get_sizes(self):
        """Test getting ROI dimensions."""
        roi1 = PILImage.new("RGB", (50, 60))  # width, height
        roi2 = np.random.randint(0, 255, (80, 70, 3), dtype=np.uint8)  # H, W, C

        rois = ROIs(roi_images=[roi1, roi2], source_boxes=[(10, 20, 60, 80), (100, 150, 170, 230)])

        sizes = rois.get_roi_sizes()
        assert sizes == [(50, 60), (70, 80)]  # (width, height)

    def test_rois_get_areas(self):
        """Test getting ROI areas."""
        roi1 = PILImage.new("RGB", (50, 60))
        roi2 = PILImage.new("RGB", (70, 80))

        rois = ROIs(roi_images=[roi1, roi2], source_boxes=[(10, 20, 60, 80), (100, 150, 170, 230)])

        areas = rois.get_roi_areas()
        assert areas == [3000, 5600]  # 50*60, 70*80

    def test_rois_filter_by_size(self):
        """Test filtering ROIs by size."""
        roi1 = PILImage.new("RGB", (30, 40))  # Small
        roi2 = PILImage.new("RGB", (60, 70))  # Large
        roi3 = PILImage.new("RGB", (80, 90))  # Large

        rois = ROIs(
            roi_images=[roi1, roi2, roi3], source_boxes=[(10, 20, 40, 60), (50, 60, 110, 130), (150, 160, 230, 250)]
        )

        filtered = rois.filter_by_size(min_width=50, min_height=50)

        assert len(filtered.roi_images) == 2
        assert len(filtered.instance_ids) == 2
        assert len(filtered.source_boxes) == 2

    def test_rois_to_numpy_list(self):
        """Test converting all ROIs to numpy arrays."""
        roi1 = PILImage.new("RGB", (50, 60))
        roi2 = np.random.randint(0, 255, (70, 80, 3), dtype=np.uint8)

        rois = ROIs(roi_images=[roi1, roi2], source_boxes=[(10, 20, 60, 80), (100, 150, 170, 230)])

        arrays = rois.to_numpy_list()

        assert len(arrays) == 2
        assert all(isinstance(arr, np.ndarray) for arr in arrays)
        assert arrays[0].shape == (60, 50, 3)  # PIL converted
        assert arrays[1].shape == (70, 80, 3)  # Already numpy

    def test_rois_to_pil_list(self):
        """Test converting all ROIs to PIL Images."""
        roi1 = PILImage.new("RGB", (50, 60))
        roi2 = np.random.randint(0, 255, (70, 80, 3), dtype=np.uint8)

        rois = ROIs(roi_images=[roi1, roi2], source_boxes=[(10, 20, 60, 80), (100, 150, 170, 230)])

        images = rois.to_pil_list()

        assert len(images) == 2
        assert all(isinstance(img, PILImage.Image) for img in images)

    def test_rois_to_dict_from_dict_pil(self):
        """Test serialization round-trip with PIL images."""
        roi = PILImage.new("RGB", (50, 60))

        rois = ROIs(roi_images=[roi], source_boxes=[(10, 20, 60, 80)], meta={"test": "data"})

        data = rois.to_dict()
        rois2 = ROIs.from_dict(data)

        assert len(rois2.roi_images) == 1
        assert isinstance(rois2.roi_images[0], PILImage.Image)
        assert rois2.source_boxes == rois.source_boxes
        assert rois2.meta["test"] == "data"

    def test_rois_to_dict_from_dict_numpy(self):
        """Test serialization round-trip with numpy arrays."""
        roi = np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8)

        rois = ROIs(roi_images=[roi], source_boxes=[(10, 20, 60, 80)], meta={"test": "data"})

        data = rois.to_dict()
        rois2 = ROIs.from_dict(data)

        assert len(rois2.roi_images) == 1
        assert isinstance(rois2.roi_images[0], np.ndarray)
        assert rois2.roi_images[0].shape == roi.shape
        np.testing.assert_array_equal(rois2.roi_images[0], roi)

    def test_rois_validate_method(self):
        """Test validate() method."""
        roi = PILImage.new("RGB", (50, 50))

        rois = ROIs(roi_images=[roi], source_boxes=[(10, 20, 60, 70)])

        rois.validate()  # Should not raise

        # Empty should fail
        rois_empty = ROIs(roi_images=[], source_boxes=[], instance_ids=[])
        with pytest.raises(ValueError, match="at least one ROI"):
            rois_empty.validate()


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases across all artifacts."""

    def test_empty_artifacts(self):
        """Test creating empty artifacts."""
        # Masks - should work with empty lists
        masks = Masks(instances=[], instance_ids=[])
        assert len(masks.instances) == 0

        # Keypoints - should work with empty lists
        keypoints = Keypoints(keypoints=[], instance_ids=[])
        assert len(keypoints.keypoints) == 0

        # ROIs - should work with empty lists
        rois = ROIs(roi_images=[], instance_ids=[], source_boxes=[])
        assert len(rois.roi_images) == 0

    def test_instance_id_consistency(self):
        """Test instance_id alignment across artifacts."""
        rle_mask = {"size": [100, 100], "counts": "test"}
        instances = [Instance(mask=rle_mask, score=0.9, label=0), Instance(mask=rle_mask, score=0.8, label=1)]

        # Create from same source
        result = VisionResult(instances=instances)
        masks = Masks.from_vision_result(result)

        # IDs should be consistent
        assert len(masks.instance_ids) == len(instances)
        assert all(isinstance(id, str) for id in masks.instance_ids)
        assert masks.instance_ids[0] != masks.instance_ids[1]  # Unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
