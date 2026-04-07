"""Tests for mata.core.logging — is_model_cached() and suppress_third_party_logs(allow_progress=...)."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import mata
import mata.core.logging as mata_logging
from mata.core.logging import (
    get_verbosity,
    is_model_cached,
    suppress_third_party_logs,
    verbose,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def verbosity(level: int):
    """Temporarily set MATA verbosity and restore afterward."""
    prev = get_verbosity()
    verbose(level)
    try:
        yield
    finally:
        verbose(prev)


# ---------------------------------------------------------------------------
# is_model_cached()
# ---------------------------------------------------------------------------


class TestIsModelCached:
    """Tests for is_model_cached() helper."""

    def test_returns_true_when_config_json_found(self):
        """Returns True when try_to_load_from_cache returns a path string."""
        with patch("mata.core.logging.is_model_cached") as mock:
            mock.return_value = True
            assert is_model_cached("facebook/detr-resnet-50") is True

    def test_actual_implementation_cached(self):
        """Returns True when huggingface_hub finds a cached path."""
        with patch("huggingface_hub.try_to_load_from_cache", return_value="/some/path/config.json"):
            result = is_model_cached("facebook/detr-resnet-50")
        assert result is True

    def test_actual_implementation_not_cached_none(self):
        """Returns False when try_to_load_from_cache returns None."""
        with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
            result = is_model_cached("some/uncached-model")
        assert result is False

    def test_actual_implementation_sentinel_object(self):
        """Returns False when try_to_load_from_cache returns non-string sentinel."""
        sentinel = object()  # Simulates _CACHED_NO_EXIST
        with patch("huggingface_hub.try_to_load_from_cache", return_value=sentinel):
            result = is_model_cached("does/not-exist")
        assert result is False

    def test_handles_import_error(self):
        """Returns False gracefully when huggingface_hub is not installed."""
        # Remove the cached module entry so the lazy import fails
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            result = is_model_cached("some/model")
        assert result is False

    def test_handles_entry_not_found_error(self):
        """Returns False gracefully when EntryNotFoundError is raised."""
        # Simulate entrynotfounderror from huggingface_hub
        with patch("huggingface_hub.try_to_load_from_cache", side_effect=Exception("EntryNotFoundError")):
            result = is_model_cached("private/model")
        assert result is False

    def test_handles_arbitrary_exception(self):
        """Returns False gracefully for any unexpected error."""
        with patch("huggingface_hub.try_to_load_from_cache", side_effect=RuntimeError("network")):
            result = is_model_cached("any/model")
        assert result is False

    def test_returns_bool_always(self):
        """Always returns a bool, never another type."""
        with patch("huggingface_hub.try_to_load_from_cache", return_value="/path"):
            result = is_model_cached("any/model")
        assert isinstance(result, bool)

        with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
            result = is_model_cached("any/model")
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# suppress_third_party_logs() — backward compat (no allow_progress)
# ---------------------------------------------------------------------------


class TestSuppressThirdPartyLogsDefault:
    """Ensure default behaviour (allow_progress=False) is unchanged."""

    def test_sets_hf_hub_disable_progress_bars_by_default(self):
        """Default call suppresses HF progress bars."""
        with verbosity(1):
            prev = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            try:
                with suppress_third_party_logs():
                    assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"
            finally:
                if prev is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev

    def test_sets_tqdm_disable_by_default(self):
        """Default call disables tqdm."""
        with verbosity(1):
            prev = os.environ.get("TQDM_DISABLE")
            try:
                with suppress_third_party_logs():
                    assert os.environ.get("TQDM_DISABLE") == "1"
            finally:
                if prev is None:
                    os.environ.pop("TQDM_DISABLE", None)
                else:
                    os.environ["TQDM_DISABLE"] = prev

    def test_redirects_stderr_by_default(self):
        """Default call redirects stderr to devnull."""
        with verbosity(1):
            real_stderr = sys.stderr
            with suppress_third_party_logs():
                assert sys.stderr is not real_stderr
            assert sys.stderr is real_stderr

    def test_restores_env_vars_after_exit(self):
        """Env vars are restored to their pre-call values after the block."""
        with verbosity(1):
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            os.environ.pop("TQDM_DISABLE", None)
            with suppress_third_party_logs():
                pass
            assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ
            assert "TQDM_DISABLE" not in os.environ

    def test_explicit_false_same_as_default(self):
        """Passing allow_progress=False behaves identically to the default."""
        with verbosity(1):
            prev = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            try:
                with suppress_third_party_logs(allow_progress=False):
                    assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"
            finally:
                if prev is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev


# ---------------------------------------------------------------------------
# suppress_third_party_logs(allow_progress=True)
# ---------------------------------------------------------------------------


class TestSuppressThirdPartyLogsAllowProgress:
    """Ensure allow_progress=True keeps HF progress bars alive."""

    def test_does_not_set_hf_hub_disable_progress_bars(self):
        """When allow_progress=True, HF_HUB_DISABLE_PROGRESS_BARS must not become '1'."""
        with verbosity(1):
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            with suppress_third_party_logs(allow_progress=True):
                val = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            # Must be absent or not "1"
            assert val != "1"

    def test_does_not_set_tqdm_disable(self):
        """When allow_progress=True, TQDM_DISABLE must not become '1'."""
        with verbosity(1):
            os.environ.pop("TQDM_DISABLE", None)
            with suppress_third_party_logs(allow_progress=True):
                val = os.environ.get("TQDM_DISABLE")
            assert val != "1"

    def test_does_not_redirect_stderr(self):
        """When allow_progress=True, stderr is left open."""
        with verbosity(1):
            real_stderr = sys.stderr
            with suppress_third_party_logs(allow_progress=True):
                assert sys.stderr is real_stderr
            assert sys.stderr is real_stderr

    def test_still_suppresses_noisy_loggers(self):
        """Even with allow_progress=True, third-party loggers are silenced."""
        with verbosity(1):
            hf_logger = logging.getLogger("huggingface_hub")
            prev_level = hf_logger.level
            hf_logger.setLevel(logging.DEBUG)
            try:
                with suppress_third_party_logs(allow_progress=True):
                    # Inside the block the logger should be elevated to ERROR
                    assert hf_logger.level >= logging.ERROR
            finally:
                hf_logger.setLevel(prev_level)

    def test_restores_after_exit_with_allow_progress(self):
        """Env vars are still clean after allow_progress=True block."""
        with verbosity(1):
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            with suppress_third_party_logs(allow_progress=True):
                pass
            # Should still be absent (we never set it)
            assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ


# ---------------------------------------------------------------------------
# Verbosity interactions
# ---------------------------------------------------------------------------


class TestVerbosityInteraction:
    """Verify verbosity 0 and 2 override allow_progress correctly."""

    def test_verbosity_2_skips_all_suppression(self):
        """At verbosity=2, suppress_third_party_logs() is a no-op."""
        with verbosity(2):
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            real_stderr = sys.stderr
            with suppress_third_party_logs():
                assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") is None
                assert sys.stderr is real_stderr

    def test_verbosity_2_skips_suppression_even_with_allow_progress_false(self):
        """At verbosity=2, even allow_progress=False doesn't suppress."""
        with verbosity(2):
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            with suppress_third_party_logs(allow_progress=False):
                assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") is None

    def test_verbosity_0_suppresses_everything_even_with_allow_progress_true(self):
        """At verbosity=0, allow_progress=True is ignored — total silence."""
        with verbosity(0):
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            prev = os.environ.get("TQDM_DISABLE")
            real_stderr = sys.stderr
            try:
                with suppress_third_party_logs(allow_progress=True):
                    # Progress bars must still be suppressed
                    assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"
                    assert os.environ.get("TQDM_DISABLE") == "1"
                    assert sys.stderr is not real_stderr
            finally:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                if prev is None:
                    os.environ.pop("TQDM_DISABLE", None)
                else:
                    os.environ["TQDM_DISABLE"] = prev
            assert sys.stderr is real_stderr

    def test_verbosity_1_allow_progress_true_does_not_suppress(self):
        """At verbosity=1 with allow_progress=True, progress bars are left open."""
        with verbosity(1):
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            real_stderr = sys.stderr
            with suppress_third_party_logs(allow_progress=True):
                assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") != "1"
                assert sys.stderr is real_stderr
