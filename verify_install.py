"""Verify MATA installation and setup.

Run this script after installation to verify that MATA is correctly installed
and all v1.8.0 features are available.
"""

import sys


def check_imports():
    """Check that all imports work."""
    print("Checking imports...")

    try:
        import mata

        print(f"  ✓ mata package imported (v{mata.__version__})")
    except ImportError as e:
        print(f"  ✗ Failed to import mata: {e}")
        return False

    try:
        from mata import load, run, track, infer, list_models

        print("  ✓ API functions imported (load, run, track, infer, list_models)")
    except ImportError as e:
        print(f"  ✗ Failed to import API: {e}")
        return False

    try:
        from mata.core import VisionResult, ClassifyResult, DepthResult, Instance

        print("  ✓ Core types imported")
    except ImportError as e:
        print(f"  ✗ Failed to import core types: {e}")
        return False

    return True


def check_api():
    """Check that core API functions work."""
    print("\nChecking API functions...")

    try:
        import mata

        # Verify load, run, track, infer, list_models exist and are callable
        assert callable(mata.load), "mata.load is not callable"
        assert callable(mata.run), "mata.run is not callable"
        assert callable(mata.track), "mata.track is not callable"
        assert callable(mata.infer), "mata.infer is not callable"
        assert callable(mata.list_models), "mata.list_models is not callable"
        print(f"  ✓ API functions available (v{mata.__version__})")

        models = mata.list_models()
        supported_tasks = list(models.keys()) if models else []
        print(f"  ✓ Supported tasks: {supported_tasks}")
        return True

    except Exception as e:
        print(f"  ✗ API check failed: {e}")
        return False


def check_result_types():
    """Check that result types are importable."""
    print("\nChecking result types...")

    try:
        from mata import VisionResult, ClassifyResult, DepthResult, Instance

        print("  ✓ VisionResult, ClassifyResult, DepthResult, Instance importable")

        # v1.8.0 tracking types
        from mata import Track, TrackResult

        print("  ✓ Track, TrackResult importable (v1.8.0 tracking)")
        return True

    except ImportError as e:
        print(f"  ✗ Failed to import result types: {e}")
        return False


def check_dependencies():
    """Check required dependencies."""
    print("\nChecking dependencies...")

    deps = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "PIL": "Pillow",
        "numpy": "NumPy",
    }

    all_ok = True
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"  ✓ {name} installed")
        except ImportError:
            print(f"  ✗ {name} not found")
            all_ok = False

    return all_ok


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("MATA Installation Verification")
    print("=" * 60 + "\n")

    checks = [
        ("Imports", check_imports),
        ("Dependencies", check_dependencies),
        ("API Functions", check_api),
        ("Result Types", check_result_types),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nUnexpected error in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:.<40} {status}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n✓ All checks passed! MATA is ready to use.")
        print("\nNext steps:")
        print("  1. See examples/ directory for usage examples")
        print("  2. Run: python examples/inference/simple_transformer_detection.py")
        print("  3. Read README.md for API documentation")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
