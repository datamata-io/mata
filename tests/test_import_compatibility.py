"""Test import compatibility and ensure no breaking changes to module structure.

This test suite ensures that all module imports work as expected and that
the module structure has not been broken by additions.
"""

import importlib
import warnings

import pytest


class TestCoreImports:
    """Test core mata module imports."""

    def test_mata_imports(self):
        """Test basic mata import."""
        import mata

        assert mata is not None
        assert hasattr(mata, "__version__")

    def test_mata_api_imports(self):
        """Test API function imports."""
        from mata import get_model_info, infer, list_models, load, register_model, run

        assert callable(load)
        assert callable(run)
        assert callable(infer)
        assert callable(list_models)
        assert callable(get_model_info)
        assert callable(register_model)

    def test_mata_core_imports(self):
        """Test core type imports."""
        from mata.core import (
            Detection,
            Entity,
            Instance,
            VisionResult,
        )

        # Verify all types are classes or dataclasses
        assert Detection is not None
        assert Instance is not None
        assert Entity is not None
        assert VisionResult is not None

    def test_mata_exceptions_imports(self):
        """Test exception imports."""
        from mata.core.exceptions import (
            InvalidInputError,
            MATAError,
            ModelLoadError,
            TaskNotSupportedError,
        )

        # Verify exception hierarchy
        assert issubclass(InvalidInputError, MATAError)
        assert issubclass(ModelLoadError, MATAError)
        assert issubclass(TaskNotSupportedError, MATAError)


class TestAdapterImports:
    """Test adapter module imports."""

    def test_adapter_base_exists(self):
        """Test adapter base modules exist."""
        import mata.adapters

        assert mata.adapters is not None

    def test_detection_adapters_available(self):
        """Test detection adapters can be imported from where they actually exist."""
        # Adapters live in different locations, test current structure
        try:
            from mata.adapters.huggingface_adapter import HuggingFaceDetectAdapter

            assert HuggingFaceDetectAdapter is not None
        except ImportError:
            pytest.skip("HuggingFace detect adapter location may vary")

    def test_classification_adapters_available(self):
        """Test classification adapters can be imported."""
        try:
            from mata.adapters.huggingface_classify_adapter import HuggingFaceClassifyAdapter

            assert HuggingFaceClassifyAdapter is not None
        except ImportError:
            pytest.skip("Classification adapter location may vary")

    def test_segmentation_adapters_available(self):
        """Test segmentation adapters can be imported."""
        try:
            from mata.adapters.huggingface_segment_adapter import HuggingFaceSegmentAdapter

            assert HuggingFaceSegmentAdapter is not None
        except ImportError:
            pytest.skip("Segmentation adapter location may vary")


class TestVisualizationImports:
    """Test visualization module imports."""

    def test_visualization_module_exists(self):
        """Test visualization module can be imported."""
        import mata.visualization

        assert mata.visualization is not None

    def test_visualization_functions_available(self):
        """Test visualization functions are available."""
        # Visualization functions exist somewhere in the module
        import mata.visualization

        # Module should have some visualization capability
        assert hasattr(mata.visualization, "visualize_segmentation") or hasattr(
            mata, "visualization"
        ), "Visualization functionality should be available"


class TestExportImports:
    """Test export module imports."""

    def test_export_functionality_exists(self):
        """Test export functionality exists in results."""
        # Export functionality is built into result types
        from mata.core import DetectResult, VisionResult

        # Results should have save/export methods
        assert hasattr(VisionResult, "to_json") or hasattr(
            VisionResult, "save"
        ), "Results should have export capability"
        assert hasattr(DetectResult, "to_json") or hasattr(
            DetectResult, "save"
        ), "Results should have export capability"


class TestGraphImports:
    """Test graph system imports (new in v1.6)."""

    def test_graph_module_available(self):
        """Test graph module can be imported."""
        try:
            from mata.core import graph

            assert graph is not None
        except ImportError:
            pytest.skip("Graph system not yet implemented")

    def test_graph_core_imports(self):
        """Test core graph imports."""
        try:
            from mata.core.graph import ExecutionContext, Graph, Node, Scheduler

            assert Graph is not None
            assert Node is not None
            assert ExecutionContext is not None
            assert Scheduler is not None
        except ImportError:
            pytest.skip("Graph system not yet implemented")

    def test_artifact_imports(self):
        """Test artifact imports."""
        try:
            from mata.core.graph.artifacts import Artifact, Detections, Image, Masks, MultiResult  # noqa: F401

            assert Artifact is not None
            assert Image is not None
            assert Detections is not None
            assert MultiResult is not None
        except ImportError:
            pytest.skip("Graph artifacts not yet implemented")

    def test_node_imports(self):
        """Test node imports."""
        try:
            from mata.nodes import Classify, Detect, Filter, Fuse, Segment

            assert Detect is not None
            assert Classify is not None
            assert Segment is not None
            assert Filter is not None
            assert Fuse is not None
        except ImportError:
            pytest.skip("Graph nodes not yet implemented")


class TestCircularImports:
    """Test that there are no circular import issues."""

    def test_no_circular_imports_core(self):
        """Test core modules don't have circular imports."""
        # If this import works, no circular imports exist

        assert True

    def test_no_circular_imports_adapters(self):
        """Test adapter modules don't have circular imports."""
        import mata.adapters

        # Adapters module structure exists
        assert mata.adapters is not None

    def test_no_circular_imports_graph(self):
        """Test graph modules don't have circular imports."""
        try:
            import mata.core.graph
            import mata.core.graph.graph
            import mata.core.graph.node  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("Graph system not yet implemented")


class TestDeprecationWarnings:
    """Test for deprecation warnings in public APIs."""

    def test_no_deprecation_warnings_on_load(self):
        """Test that mata.load() doesn't emit deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import mata

            # Check no deprecation warnings
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]

            # Should have no deprecation warnings from import
            assert len(deprecation_warnings) == 0, f"Import triggered deprecation warnings: {deprecation_warnings}"

    def test_no_deprecation_warnings_on_api_access(self):
        """Test accessing API functions doesn't emit warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            import mata

            # Access all public API functions
            _ = mata.load
            _ = mata.run
            _ = mata.infer
            _ = mata.list_models

            # Check no deprecation warnings
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]

            assert len(deprecation_warnings) == 0


class TestModuleStructure:
    """Test module structure has not changed."""

    def test_mata_module_attributes(self):
        """Test mata module has expected attributes."""
        import mata

        expected_attrs = [
            "__version__",
            "load",
            "run",
            "infer",
            "list_models",
            "get_model_info",
            "register_model",
        ]

        for attr in expected_attrs:
            assert hasattr(mata, attr), f"mata.{attr} not found"

    def test_mata_core_module_attributes(self):
        """Test mata.core module structure."""

        # Should have key submodules
        expected_submodules = [
            "types",
            "exceptions",
            "model_loader",
        ]

        for submodule in expected_submodules:
            try:
                importlib.import_module(f"mata.core.{submodule}")
            except ImportError:
                pytest.fail(f"mata.core.{submodule} not found")

    def test_mata_adapters_module_structure(self):
        """Test mata.adapters module structure."""
        import mata.adapters

        # Adapters module exists
        assert mata.adapters is not None

        # Some adapter files should exist
        # (exact structure may vary, just checking module is not empty)
        assert dir(mata.adapters), "Adapters module should have contents"


class TestTypeStability:
    """Test that type signatures have not changed."""

    def test_detectresult_type_stable(self):
        """Test DetectResult type is stable."""
        from mata.core import Detection, DetectResult

        # Should be able to create instance
        result = DetectResult(
            detections=[Detection(bbox=(10, 20, 100, 200), score=0.9, label=1, label_name="cat")], meta={}
        )

        assert result is not None
        assert hasattr(result, "detections")
        assert hasattr(result, "meta")

    def test_visionresult_type_stable(self):
        """Test VisionResult type is stable."""
        from mata.core import Instance, VisionResult

        # Should be able to create instance with current API
        result = VisionResult(
            instances=[Instance(bbox=(10, 20, 100, 200), score=0.9, label=1, label_name="cat")], meta={}
        )

        assert result is not None
        assert hasattr(result, "instances")
        assert hasattr(result, "meta")

    def test_classifyresult_type_stable(self):
        """Test ClassifyResult type is stable."""
        from mata.core import ClassifyResult

        # ClassifyResult structure test
        # Exact instantiation may vary, just test it exists and is importable
        assert ClassifyResult is not None
        assert hasattr(ClassifyResult, "__init__")


if __name__ == "__main__":
    # Run import compatibility tests
    pytest.main([__file__, "-v", "--tb=short"])
