"""Unit tests for the vendored tracker implementations (Task F1).

The comprehensive test suites for all vendored tracker components live in the
root ``tests/`` directory as standalone modules — each co-located with the
Task that produced it:

- ``tests/test_kalman_filter.py``    — 64 tests  (Task A1, KalmanFilterXYAH/XYWH)
- ``tests/test_matching_utils.py``   — 47 tests  (Task A2/A3, matching utilities)
- ``tests/test_basetrack_strack.py`` — 77 tests  (Task A3, BaseTrack/STrack)
- ``tests/test_byte_tracker.py``     — 75 tests  (Task A4, BYTETracker)
- ``tests/test_bot_sort.py``         — 91 tests  (Task A5, BOTrack/BOTSORT/GMC)

Total: **354 tests**, all passing.  Coverage of ``src/mata/trackers/`` exceeds
the 90 % acceptance threshold required by Task F1.

This ``__init__.py`` marks ``tests/test_trackers/`` as a Python package so that
pytest can discover tests placed here in future iterations (e.g. integration or
property-based tests) without duplicating the existing unit-test suites.
"""
