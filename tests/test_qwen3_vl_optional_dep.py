"""Tests for the qwen-vl-utils lazy-load guard in Qwen3VLEmbeddingAdapter.

Validates that _try_load_qwen_vl_utils() handles all edge cases:
- Import failure (package absent)
- Import success (package present)
- Module-level caching (no re-import on repeated calls)
- Global _QWEN_VL_UTILS_AVAILABLE set correctly
- Adapter info() reports correct availability status

All tests are fully offline — no model downloads required.
Run independently: pytest tests/test_qwen3_vl_optional_dep.py -v
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_guard(module):
    """Reset guard globals to uninitialized state between tests."""
    module._QWEN_VL_UTILS_AVAILABLE = None
    module._qwen_vl_utils = None


def _make_fake_qwen_vl_utils():
    """Return a minimal fake qwen_vl_utils module."""
    fake = types.ModuleType("qwen_vl_utils")
    fake.process_vision_info = MagicMock(return_value=([], None))
    return fake


# ---------------------------------------------------------------------------
# Tests: _try_load_qwen_vl_utils() guard function
# ---------------------------------------------------------------------------


class TestTryLoadQwenVLUtils:
    """Unit tests for the _try_load_qwen_vl_utils() lazy-load guard."""

    def setup_method(self):
        """Import the module fresh and reset guard state before each test."""
        import mata.adapters.qwen3_vl_embedding_adapter as mod

        self.mod = mod
        _reset_guard(mod)

    def teardown_method(self):
        """Reset guard state after each test to prevent pollution."""
        _reset_guard(self.mod)

    # ------------------------------------------------------------------
    # Test: package not installed
    # ------------------------------------------------------------------

    def test_returns_none_when_package_absent(self):
        """_try_load_qwen_vl_utils() returns None when qwen-vl-utils is not installed."""
        with patch.dict(sys.modules, {"qwen_vl_utils": None}):
            result = self.mod._try_load_qwen_vl_utils()
        assert result is None

    def test_availability_false_when_package_absent(self):
        """_QWEN_VL_UTILS_AVAILABLE is set to False when import fails."""
        with patch.dict(sys.modules, {"qwen_vl_utils": None}):
            self.mod._try_load_qwen_vl_utils()
        assert self.mod._QWEN_VL_UTILS_AVAILABLE is False

    def test_module_global_none_when_package_absent(self):
        """_qwen_vl_utils module global remains None when package absent."""
        with patch.dict(sys.modules, {"qwen_vl_utils": None}):
            self.mod._try_load_qwen_vl_utils()
        assert self.mod._qwen_vl_utils is None

    def test_no_crash_when_package_absent(self):
        """Guard function must not raise any exception when package not installed."""
        with patch.dict(sys.modules, {"qwen_vl_utils": None}):
            try:
                self.mod._try_load_qwen_vl_utils()
            except Exception as exc:
                pytest.fail(f"Guard raised unexpectedly: {exc}")

    # ------------------------------------------------------------------
    # Test: package present
    # ------------------------------------------------------------------

    def test_returns_module_when_package_present(self):
        """_try_load_qwen_vl_utils() returns the module when qwen-vl-utils is installed."""
        fake_module = _make_fake_qwen_vl_utils()
        with patch.dict(sys.modules, {"qwen_vl_utils": fake_module}):
            result = self.mod._try_load_qwen_vl_utils()
        assert result is fake_module

    def test_availability_true_when_package_present(self):
        """_QWEN_VL_UTILS_AVAILABLE is set to True when import succeeds."""
        fake_module = _make_fake_qwen_vl_utils()
        with patch.dict(sys.modules, {"qwen_vl_utils": fake_module}):
            self.mod._try_load_qwen_vl_utils()
        assert self.mod._QWEN_VL_UTILS_AVAILABLE is True

    def test_module_global_set_when_package_present(self):
        """_qwen_vl_utils global is set to the module when import succeeds."""
        fake_module = _make_fake_qwen_vl_utils()
        with patch.dict(sys.modules, {"qwen_vl_utils": fake_module}):
            self.mod._try_load_qwen_vl_utils()
        assert self.mod._qwen_vl_utils is fake_module

    # ------------------------------------------------------------------
    # Test: caching / idempotency
    # ------------------------------------------------------------------

    def test_cached_when_absent_no_reimport(self):
        """Repeated calls when absent return None without re-attempting import."""
        call_count = 0

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def counting_import(name, *args, **kwargs):
            nonlocal call_count
            if name == "qwen_vl_utils":
                call_count += 1
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {"qwen_vl_utils": None}):
            self.mod._try_load_qwen_vl_utils()
            first_result = self.mod._QWEN_VL_UTILS_AVAILABLE
            # Second call — should use cached value, not re-attempt
            self.mod._try_load_qwen_vl_utils()
            second_result = self.mod._QWEN_VL_UTILS_AVAILABLE

        assert first_result is False
        assert second_result is False

    def test_cached_when_present_no_reimport(self):
        """Repeated calls when present return module without re-importing."""
        fake_module = _make_fake_qwen_vl_utils()
        with patch.dict(sys.modules, {"qwen_vl_utils": fake_module}):
            result1 = self.mod._try_load_qwen_vl_utils()
        # Second call — guard should skip import block since flag is already True
        result2 = self.mod._try_load_qwen_vl_utils()
        assert result1 is fake_module
        assert result2 is fake_module
        assert self.mod._QWEN_VL_UTILS_AVAILABLE is True

    def test_none_check_is_guard_condition(self):
        """Guard only runs import block when _QWEN_VL_UTILS_AVAILABLE is None."""
        # Pre-set to True to simulate cached state
        self.mod._QWEN_VL_UTILS_AVAILABLE = True
        self.mod._qwen_vl_utils = _make_fake_qwen_vl_utils()

        # Even if qwen_vl_utils is "absent" in sys.modules, cached result is returned
        with patch.dict(sys.modules, {"qwen_vl_utils": None}):
            result = self.mod._try_load_qwen_vl_utils()

        # Should return the pre-set module, not None
        assert result is self.mod._qwen_vl_utils
        assert self.mod._QWEN_VL_UTILS_AVAILABLE is True

    def test_false_check_skips_reimport(self):
        """Guard skips import block when _QWEN_VL_UTILS_AVAILABLE is already False."""
        self.mod._QWEN_VL_UTILS_AVAILABLE = False
        self.mod._qwen_vl_utils = None

        # Even if qwen_vl_utils is now present, cached False is returned
        fake_module = _make_fake_qwen_vl_utils()
        with patch.dict(sys.modules, {"qwen_vl_utils": fake_module}):
            result = self.mod._try_load_qwen_vl_utils()

        assert result is None
        assert self.mod._QWEN_VL_UTILS_AVAILABLE is False


# ---------------------------------------------------------------------------
# Tests: Adapter info() reflects guard state
# ---------------------------------------------------------------------------


class TestAdapterInfoReflectsGuard:
    """Verify Qwen3VLEmbeddingAdapter.info() reports qwen_vl_utils_available correctly.

    Uses object.__new__ to bypass __init__ (which needs transformers) so tests
    are purely unit tests of the info() method's use of the module-level guard flag.
    """

    def _make_adapter_stub(self, mod):
        """Create a Qwen3VLEmbeddingAdapter without calling __init__ (no transformers needed)."""
        import torch

        adapter = object.__new__(mod.Qwen3VLEmbeddingAdapter)
        # Inject the minimal instance attributes consumed by info()
        adapter.model_id = "Qwen/Qwen3-VL-Embedding-2B"
        adapter._native_dim = None
        adapter._embedding_dim = None
        adapter._embed_dim = None
        adapter._device = torch.device("cpu")
        adapter._dtype = torch.float32
        adapter.fps = 1.0
        adapter.max_frames = 64
        return adapter

    def test_info_reports_false_when_package_absent(self):
        """info()['qwen_vl_utils_available'] is False when qwen-vl-utils not installed."""
        import mata.adapters.qwen3_vl_embedding_adapter as mod

        _reset_guard(mod)

        with patch.dict(sys.modules, {"qwen_vl_utils": None}):
            mod._try_load_qwen_vl_utils()

        adapter = self._make_adapter_stub(mod)
        info = adapter.info()
        assert info["qwen_vl_utils_available"] is False
        _reset_guard(mod)

    def test_info_reports_true_when_package_present(self):
        """info()['qwen_vl_utils_available'] is True when qwen-vl-utils is installed."""
        import mata.adapters.qwen3_vl_embedding_adapter as mod

        _reset_guard(mod)

        fake_module = _make_fake_qwen_vl_utils()
        with patch.dict(sys.modules, {"qwen_vl_utils": fake_module}):
            mod._try_load_qwen_vl_utils()

        adapter = self._make_adapter_stub(mod)
        info = adapter.info()
        assert info["qwen_vl_utils_available"] is True
        _reset_guard(mod)

    def test_info_key_always_present(self):
        """info() dict always contains 'qwen_vl_utils_available' key regardless of state."""
        import mata.adapters.qwen3_vl_embedding_adapter as mod

        _reset_guard(mod)

        with patch.dict(sys.modules, {"qwen_vl_utils": None}):
            mod._try_load_qwen_vl_utils()

        adapter = self._make_adapter_stub(mod)
        info = adapter.info()
        assert "qwen_vl_utils_available" in info
        _reset_guard(mod)

    def test_info_availability_is_bool(self):
        """info()['qwen_vl_utils_available'] is always a bool, not None."""
        import mata.adapters.qwen3_vl_embedding_adapter as mod

        _reset_guard(mod)

        with patch.dict(sys.modules, {"qwen_vl_utils": None}):
            mod._try_load_qwen_vl_utils()

        adapter = self._make_adapter_stub(mod)
        info = adapter.info()
        assert isinstance(info["qwen_vl_utils_available"], bool)
        _reset_guard(mod)

    def test_info_message_suggests_extra(self):
        """Guard's fallback log message references the qwen3-embedding extra install command."""

        import mata.adapters.qwen3_vl_embedding_adapter as mod

        _reset_guard(mod)

        with patch.dict(sys.modules, {"qwen_vl_utils": None}):
            with patch.object(mod.logger, "info") as mock_log:
                mod._try_load_qwen_vl_utils()
                # Verify at least one info call happened with the install hint
                messages = [str(call) for call in mock_log.call_args_list]
                assert any("qwen3-embedding" in msg for msg in messages), (
                    "Guard should log a message suggesting 'pip install datamata[qwen3-embedding]' "
                    f"when qwen-vl-utils is absent. Got log messages: {messages}"
                )

        _reset_guard(mod)
