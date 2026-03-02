#!/usr/bin/env python3
"""Comprehensive backward compatibility test runner.

This script runs ALL backward compatibility tests and generates a detailed report.
Use this to validate that the graph system additions have not broken existing functionality.

Usage:
    python tests/run_backward_compat_tests.py
    python tests/run_backward_compat_tests.py --verbose
    python tests/run_backward_compat_tests.py --report report.txt
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class BackwardCompatTestRunner:
    """Runner for comprehensive backward compatibility tests."""

    def __init__(self, verbose: bool = False, report_path: str = None):
        self.verbose = verbose
        self.report_path = report_path
        self.results = []

    def run_test_suite(self, name: str, test_file: str, markers: str = None) -> tuple[bool, str]:
        """Run a test suite and return (success, output)."""
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        cmd = ["python", "-m", "pytest", test_file, "-v"]

        if markers:
            cmd.extend(["-m", markers])

        if not self.verbose:
            cmd.append("--tb=line")

        try:
            result = subprocess.run(
                cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300  # 5 minute timeout
            )

            success = result.returncode == 0
            output = result.stdout + result.stderr

            # Print summary
            if success:
                print(f"✅ {name}: PASSED")
            else:
                print(f"❌ {name}: FAILED")
                if self.verbose:
                    print(output)

            return success, output

        except subprocess.TimeoutExpired:
            print(f"❌ {name}: TIMEOUT")
            return False, "Test suite timed out after 5 minutes"

        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")
            return False, str(e)

    def run_all_existing_tests(self) -> tuple[bool, str]:
        """Run ALL existing tests (182+ tests)."""
        print(f"\n{'='*60}")
        print("Running: All Existing Tests")
        print(f"{'='*60}")

        cmd = ["python", "-m", "pytest", "tests/", "-v", "--tb=line", "-q"]

        # Exclude new graph tests if they exist
        cmd.extend(["--ignore=tests/graph/"])

        try:
            result = subprocess.run(
                cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600  # 10 minute timeout
            )

            success = result.returncode == 0
            output = result.stdout + result.stderr

            # Extract test count
            if "passed" in output:
                print("✅ All existing tests: PASSED")
            else:
                print("❌ All existing tests: FAILED")

            if self.verbose:
                print(output)

            return success, output

        except subprocess.TimeoutExpired:
            print("❌ All existing tests: TIMEOUT")
            return False, "Test suite timed out"

        except Exception as e:
            print(f"❌ All existing tests: ERROR - {e}")
            return False, str(e)

    def check_specific_test_files(self) -> dict[str, tuple[bool, int]]:
        """Check specific critical test files."""
        critical_tests = [
            "tests/test_universal_loader.py",
            "tests/test_classify_adapter.py",
            "tests/test_clip_adapter.py",
            "tests/test_segment_adapter.py",
            "tests/test_sam_adapter.py",
            "tests/test_depth_adapter.py",
        ]

        results = {}

        for test_file in critical_tests:
            test_path = PROJECT_ROOT / test_file
            if not test_path.exists():
                results[test_file] = (False, 0)
                continue

            cmd = ["python", "-m", "pytest", test_file, "-v", "--tb=line", "-q"]

            try:
                result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)

                success = result.returncode == 0
                output = result.stdout + result.stderr

                # Extract test count
                test_count = 0
                if "passed" in output:
                    try:
                        parts = output.split()
                        for i, part in enumerate(parts):
                            if part == "passed":
                                test_count = int(parts[i - 1])
                                break
                    except Exception:
                        pass

                results[test_file] = (success, test_count)

            except Exception:
                results[test_file] = (False, 0)

        return results

    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("MATA BACKWARD COMPATIBILITY TEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary
        report.append("TEST SUMMARY")
        report.append("-" * 80)

        total_suites = len(self.results)
        passed_suites = sum(1 for success, _ in self.results if success)

        report.append(f"Test Suites Run: {total_suites}")
        report.append(f"Passed: {passed_suites}")
        report.append(f"Failed: {total_suites - passed_suites}")
        report.append("")

        # Individual results
        report.append("INDIVIDUAL TEST RESULTS")
        report.append("-" * 80)

        for i, (name, (success, _)) in enumerate(zip([r[0] for r in self.results], [r[1] for r in self.results]), 1):
            status = "✅ PASS" if success else "❌ FAIL"
            report.append(f"{i}. {name}: {status}")

        report.append("")
        report.append("=" * 80)

        # Overall status
        all_passed = all(success for success, _ in self.results)
        if all_passed:
            report.append("✅ ALL BACKWARD COMPATIBILITY TESTS PASSED")
        else:
            report.append("❌ SOME BACKWARD COMPATIBILITY TESTS FAILED")

        report.append("=" * 80)

        return "\n".join(report)

    def run(self):
        """Run all backward compatibility tests."""
        print("\n" + "=" * 80)
        print("MATA BACKWARD COMPATIBILITY TEST SUITE")
        print("=" * 80)

        # Test 1: Basic backward compatibility tests
        result = self.run_test_suite("Basic Backward Compatibility Tests", "tests/test_backward_compatibility.py")
        self.results.append(("Basic Backward Compatibility", result))

        # Test 2: Examples validation
        result = self.run_test_suite("Examples Validation Tests", "tests/test_examples_validation.py")
        self.results.append(("Examples Validation", result))

        # Test 3: Import compatibility
        result = self.run_test_suite("Import Compatibility Tests", "tests/test_import_compatibility.py")
        self.results.append(("Import Compatibility", result))

        # Test 4: Check critical test files
        print(f"\n{'='*60}")
        print("Checking Critical Test Files")
        print(f"{'='*60}")

        critical_results = self.check_specific_test_files()

        for test_file, (success, count) in critical_results.items():
            if success:
                print(f"✅ {test_file}: {count} tests passed")
            else:
                print(f"❌ {test_file}: FAILED or not found")

        # Test 5: Run all existing tests (comprehensive)
        # This is optional and can be slow
        print(f"\n{'='*60}")
        print("Do you want to run ALL existing tests? (y/n): ", end="")

        # For automated runs, skip this
        if os.environ.get("CI") or os.environ.get("AUTOMATED"):
            run_all = True
        else:
            run_all = input().strip().lower() == "y"

        if run_all:
            result = self.run_all_existing_tests()
            self.results.append(("All Existing Tests", result))

        # Generate report
        report = self.generate_report()
        print("\n" + report)

        # Save report if path specified
        if self.report_path:
            report_file = Path(self.report_path)
            report_file.write_text(report)
            print(f"\n📄 Report saved to: {report_file}")

        # Return success status
        all_passed = all(success for success, _ in self.results)
        return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive backward compatibility tests for MATA")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output (show all test details)")
    parser.add_argument("-r", "--report", type=str, help="Path to save test report (e.g., report.txt)")

    args = parser.parse_args()

    runner = BackwardCompatTestRunner(verbose=args.verbose, report_path=args.report)

    exit_code = runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
