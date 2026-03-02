"""Tests for polygon mask visualization support."""

from PIL import Image

from mata.core.types import Instance, VisionResult
from mata.visualization import visualize_segmentation


class TestPolygonMaskVisualization:
    """Test polygon mask visualization."""

    def test_polygon_mask_to_binary_conversion(self):
        """Test that polygon masks are properly converted to binary masks."""
        # Create a simple image
        image = Image.new("RGB", (100, 100), color=(255, 255, 255))

        # Create instance with polygon mask (rectangle)
        polygon = [10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]  # Square
        instance = Instance(label=1, mask=polygon, score=0.95)  # Use numeric label

        result = VisionResult(instances=[instance])

        # Visualize - should not raise warning or error
        vis_image = visualize_segmentation(result, image, show_boxes=True)

        # Verify it returned an image
        assert isinstance(vis_image, Image.Image)
        assert vis_image.size == (100, 100)

    def test_multiple_polygon_masks(self):
        """Test visualization with multiple polygon masks."""
        # Create a simple image
        image = Image.new("RGB", (200, 200), color=(255, 255, 255))

        # Create instances with polygon masks
        polygon1 = [10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]  # Square
        polygon2 = [100.0, 100.0, 150.0, 100.0, 150.0, 150.0, 100.0, 150.0]  # Another square

        instances = [
            Instance(label=1, mask=polygon1, score=0.95),  # Use numeric labels
            Instance(label=2, mask=polygon2, score=0.90),
        ]

        result = VisionResult(instances=instances)

        # Visualize - should not raise warning or error
        vis_image = visualize_segmentation(result, image, show_boxes=True)

        # Verify it returned an image
        assert isinstance(vis_image, Image.Image)
        assert vis_image.size == (200, 200)

    def test_polygon_mask_with_bbox_computation(self):
        """Test that bboxes are computed from polygon masks when not provided."""
        # Create a simple image
        image = Image.new("RGB", (100, 100), color=(255, 255, 255))

        # Create instance with polygon mask but no bbox
        polygon = [10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]  # Square
        instance = Instance(label=1, mask=polygon, score=0.95, bbox=None)  # Use numeric label  # No bbox provided

        result = VisionResult(instances=[instance])

        # Visualize with show_boxes=True - should compute bbox from polygon mask
        vis_image = visualize_segmentation(result, image, show_boxes=True)

        # Verify it returned an image without errors
        assert isinstance(vis_image, Image.Image)
        assert vis_image.size == (100, 100)

    def test_nested_polygon_list(self):
        """Test visualization with nested polygon list (multiple polygons per instance)."""
        # Create a simple image
        image = Image.new("RGB", (200, 200), color=(255, 255, 255))

        # Create instance with multiple polygons (holes, disconnected regions)
        polygons = [
            [10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0],  # First polygon
            [20.0, 20.0, 40.0, 20.0, 40.0, 40.0, 20.0, 40.0],  # Second polygon (hole)
        ]
        instance = Instance(label=1, mask=polygons, score=0.95)  # Use numeric label

        result = VisionResult(instances=[instance])

        # Visualize - should handle nested list
        vis_image = visualize_segmentation(result, image, show_boxes=True)

        # Verify it returned an image
        assert isinstance(vis_image, Image.Image)
        assert vis_image.size == (200, 200)

    def test_polygon_mask_different_image_sizes(self):
        """Test polygon visualization works with different image sizes."""
        for width, height in [(100, 100), (640, 480), (1920, 1080)]:
            image = Image.new("RGB", (width, height), color=(255, 255, 255))

            # Create polygon scaled to image size
            polygon = [
                width * 0.1,
                height * 0.1,
                width * 0.5,
                height * 0.1,
                width * 0.5,
                height * 0.5,
                width * 0.1,
                height * 0.5,
            ]

            instance = Instance(label=1, mask=polygon, score=0.95)  # Use numeric label
            result = VisionResult(instances=[instance])

            # Visualize
            vis_image = visualize_segmentation(result, image, show_boxes=True)

            # Verify correct size
            assert vis_image.size == (width, height)
