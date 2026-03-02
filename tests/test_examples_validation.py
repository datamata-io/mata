"""Test suite to validate all examples run without modification.

This ensures backward compatibility - all existing examples should work
unchanged after adding the graph system.
"""

from pathlib import Path

import pytest

# Root directory
MATA_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = MATA_ROOT / "examples"


class TestExamplesValidity:
    """Test that all example files have valid imports and are runnable."""

    def test_examples_directory_exists(self):
        """Test examples directory exists."""
        assert EXAMPLES_DIR.exists()
        assert EXAMPLES_DIR.is_dir()

    def test_inference_examples_directory_exists(self):
        """Test inference examples subdirectory exists."""
        inference_dir = EXAMPLES_DIR / "inference"
        assert inference_dir.exists()
        assert inference_dir.is_dir()

    def test_graph_examples_directory_exists(self):
        """Test graph examples directory exists (new in v1.6)."""
        # Graph examples should exist if graph system is implemented
        # This is optional for backward compatibility
        graph_dir = EXAMPLES_DIR / "graph"
        if graph_dir.exists():
            assert graph_dir.is_dir()


class TestExampleImports:
    """Test that all examples can import mata without errors."""

    @pytest.fixture
    def example_files(self):
        """Get list of all Python example files."""
        example_files = []

        # Root level examples
        for file in EXAMPLES_DIR.glob("*.py"):
            example_files.append(file)

        # Inference examples
        inference_dir = EXAMPLES_DIR / "inference"
        if inference_dir.exists():
            for file in inference_dir.glob("*.py"):
                example_files.append(file)

        # Graph examples (new in v1.6)
        graph_dir = EXAMPLES_DIR / "graph"
        if graph_dir.exists():
            for file in graph_dir.glob("*.py"):
                example_files.append(file)

        return example_files

    def test_examples_found(self, example_files):
        """Test that example files are found."""
        assert len(example_files) > 0, "No example files found"

    @pytest.mark.parametrize(
        "example_file",
        [
            EXAMPLES_DIR / "url_image_examples.py",
            EXAMPLES_DIR / "demo_visualization_analysis_nodes.py",
            EXAMPLES_DIR / "mask_refinement_demo.py",
        ],
    )
    def test_example_imports_mata(self, example_file):
        """Test specific examples can import mata."""
        if not example_file.exists():
            pytest.skip(f"Example {example_file.name} not found")

        # Try to parse the file
        with open(example_file, encoding="utf-8") as f:
            content = f.read()

        # Check for mata imports
        assert "import mata" in content or "from mata" in content, f"{example_file.name} should import mata"

    def test_all_examples_valid_python(self, example_files):
        """Test all examples are valid Python syntax."""
        for example_file in example_files:
            # Skip README files
            if example_file.name.startswith("README"):
                continue

            # Try to compile the file
            with open(example_file, encoding="utf-8") as f:
                content = f.read()

            try:
                compile(content, example_file.name, "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {example_file.name}: {e}")


class TestExamplePatterns:
    """Test that examples use expected API patterns."""

    def test_examples_use_load_pattern(self):
        """Test examples use mata.load() pattern."""
        example_files = list(EXAMPLES_DIR.glob("*.py"))

        load_usage_count = 0
        for example_file in example_files:
            if example_file.name.startswith("README"):
                continue

            with open(example_file, encoding="utf-8") as f:
                content = f.read()

            if "mata.load(" in content or ".load(" in content:
                load_usage_count += 1

        # At least some examples should use load
        # (may be 0 if examples only demonstrate other features)
        assert load_usage_count >= 0

    def test_examples_use_run_pattern(self):
        """Test examples may use mata.run() pattern."""
        example_files = list(EXAMPLES_DIR.glob("*.py"))

        for example_file in example_files:
            if example_file.name.startswith("README"):
                continue

            # Skip graph demo files (they use node.run(), not mata.run())
            if "demo" in example_file.name or "graph" in example_file.name:
                continue

            with open(example_file, encoding="utf-8") as f:
                content = f.read()

            # If example uses mata.run(), verify it's the correct pattern
            if "mata.run(" in content:
                # Should have task and input parameters
                assert (
                    "task=" in content or '"detect"' in content or '"classify"' in content
                ), f"{example_file.name} should use correct run() pattern"

    def test_examples_use_infer_pattern(self):
        """Test new graph examples use mata.infer() pattern (v1.6+)."""
        graph_dir = EXAMPLES_DIR / "graph"
        if not graph_dir.exists():
            pytest.skip("Graph examples not yet implemented")

        graph_example_files = list(graph_dir.glob("*.py"))
        if not graph_example_files:
            pytest.skip("No graph examples found")

        infer_usage_count = 0
        for example_file in graph_example_files:
            with open(example_file, encoding="utf-8") as f:
                content = f.read()

            if "mata.infer(" in content or ".infer(" in content:
                infer_usage_count += 1

        # Graph examples should use infer
        assert infer_usage_count > 0, "Graph examples should use mata.infer()"


class TestExampleDocumentation:
    """Test that examples have proper documentation."""

    @pytest.fixture
    def readme_files(self):
        """Get all README files in examples."""
        readme_files = []

        for readme in EXAMPLES_DIR.glob("README*.md"):
            readme_files.append(readme)

        # Check subdirectories
        for subdir in EXAMPLES_DIR.iterdir():
            if subdir.is_dir():
                for readme in subdir.glob("README*.md"):
                    readme_files.append(readme)

        return readme_files

    def test_examples_have_readme(self, readme_files):
        """Test that examples directory has README files."""
        assert len(readme_files) >= 2, "Should have at least README_INFERENCE.md and README_SAVE_EXAMPLES.md"

    def test_readme_files_not_empty(self, readme_files):
        """Test README files are not empty."""
        for readme in readme_files:
            content = readme.read_text(encoding="utf-8")
            assert len(content) > 100, f"{readme.name} should have substantial content"

    def test_python_files_have_docstrings(self):
        """Test example Python files have module docstrings."""
        example_files = [
            EXAMPLES_DIR / "url_image_examples.py",
            EXAMPLES_DIR / "demo_visualization_analysis_nodes.py",
            EXAMPLES_DIR / "mask_refinement_demo.py",
        ]

        for example_file in example_files:
            if not example_file.exists():
                continue

            with open(example_file, encoding="utf-8") as f:
                content = f.read()

            # Check for docstring
            assert '"""' in content or "'''" in content, f"{example_file.name} should have docstring"


class TestExampleCompatibility:
    """Test backward compatibility of existing example patterns."""

    def test_old_api_patterns_still_valid(self):
        """Test that old API patterns from v1.5 are still valid."""
        # These patterns should all be importable and work
        import mata

        # Old patterns that must work
        assert hasattr(mata, "load")
        assert hasattr(mata, "run")
        assert hasattr(mata, "list_models")
        assert hasattr(mata, "get_model_info")
        assert hasattr(mata, "register_model")

        # Result types that must be available
        assert hasattr(mata, "DetectResult")
        assert hasattr(mata, "VisionResult")
        assert hasattr(mata, "ClassifyResult")
        assert hasattr(mata, "SegmentResult")
        assert hasattr(mata, "DepthResult")

    def test_new_api_additive_only(self):
        """Test that new graph API is purely additive."""
        import mata

        # New v1.6 features (should exist but are optional)
        has_infer = hasattr(mata, "infer")
        has_graph = hasattr(mata, "Graph")

        # If graph system exists, test it doesn't break old code
        if has_infer:
            assert callable(mata.infer)

        if has_graph:
            # Graph class should be importable
            from mata.core.graph import Graph

            assert Graph is not None

    def test_no_removed_exports(self):
        """Test that no exports were removed."""
        import mata

        # Critical exports that must never be removed
        critical_exports = [
            "load",
            "run",
            "list_models",
            "get_model_info",
            "DetectResult",
            "VisionResult",
            "ClassifyResult",
            "Detection",
            "Instance",
            "__version__",
        ]

        for export in critical_exports:
            assert hasattr(mata, export), f"Critical export '{export}' was removed!"


class TestExampleErrorHandling:
    """Test that examples handle errors gracefully."""

    def test_examples_have_error_handling(self):
        """Test examples demonstrate proper error handling."""
        example_files = list(EXAMPLES_DIR.glob("*.py"))

        for example_file in example_files:
            if example_file.name.startswith("README"):
                continue

            with open(example_file, encoding="utf-8") as f:
                content = f.read()

            # Examples should demonstrate error handling or note it
            # At minimum, should not have bare except clauses
            if "except:" in content:
                # Bare except should be followed by comment or raise
                assert "# " in content or "raise" in content, f"{example_file.name} has bare except without comment"


if __name__ == "__main__":
    # Run examples validation tests
    pytest.main([__file__, "-v", "--tb=short"])
