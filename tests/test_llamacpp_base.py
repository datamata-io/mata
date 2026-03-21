"""Unit tests for LlamaCppBaseAdapter and _ensure_llama_cpp()."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llama_cpp():
    """Return a minimal mock llama_cpp module."""
    mock_llama_cpp = MagicMock()
    mock_llama_cpp.Llama = MagicMock()
    return mock_llama_cpp


# ---------------------------------------------------------------------------
# _ensure_llama_cpp
# ---------------------------------------------------------------------------


class TestEnsureLlamaCpp:
    def test_returns_module_when_available(self, tmp_path):
        """Importing llama_cpp succeeds → module is returned."""
        import mata.adapters.llamacpp_base as base_mod

        mock_llama_cpp = _make_mock_llama_cpp()
        # Reset cached state
        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None

        with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
            result = base_mod._ensure_llama_cpp()
        assert result is mock_llama_cpp
        assert base_mod.LLAMA_CPP_AVAILABLE is True

        # Reset for other tests
        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None

    def test_raises_import_error_with_install_hint_when_not_installed(self):
        """Importing llama_cpp fails → ImportError with helpful hint."""
        import mata.adapters.llamacpp_base as base_mod

        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None

        with patch.dict(sys.modules, {"llama_cpp": None}):
            # Remove from modules so import fails
            sys.modules.pop("llama_cpp", None)
            with pytest.raises(ImportError, match="pip install datamata\\[gguf\\]"):
                base_mod._ensure_llama_cpp()

        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None

    def test_caches_module_after_first_import(self):
        """Second call reuses the cached module without re-importing."""
        import mata.adapters.llamacpp_base as base_mod

        mock_llama_cpp = _make_mock_llama_cpp()
        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None

        with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
            r1 = base_mod._ensure_llama_cpp()
            r2 = base_mod._ensure_llama_cpp()

        assert r1 is r2

        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None


# ---------------------------------------------------------------------------
# LlamaCppBaseAdapter.__init__
# ---------------------------------------------------------------------------


class TestLlamaCppBaseAdapterInit:
    @pytest.fixture(autouse=True)
    def patch_llama_cpp(self):
        """Patch llama_cpp globally so no real import occurs."""
        mock_llama_cpp = _make_mock_llama_cpp()
        import mata.adapters.llamacpp_base as base_mod

        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None

        with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
            yield mock_llama_cpp

        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None

    def test_accepts_valid_gguf_path(self, tmp_path):
        """Constructor succeeds when a .gguf file exists."""
        from mata.adapters.llamacpp_base import LlamaCppBaseAdapter

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"")
        adapter = LlamaCppBaseAdapter(model_path=str(gguf))
        assert adapter.model_path == str(gguf)

    def test_raises_file_not_found_for_missing_path(self, tmp_path):
        """Nonexistent file path → FileNotFoundError."""
        from mata.adapters.llamacpp_base import LlamaCppBaseAdapter

        with pytest.raises(FileNotFoundError, match="GGUF model not found"):
            LlamaCppBaseAdapter(model_path=str(tmp_path / "nonexistent.gguf"))

    def test_raises_value_error_for_wrong_extension(self, tmp_path):
        """A file with a .onnx extension → ValueError."""
        from mata.adapters.llamacpp_base import LlamaCppBaseAdapter

        wrong_ext = tmp_path / "model.onnx"
        wrong_ext.write_bytes(b"")
        with pytest.raises(ValueError, match=r"Expected \.gguf file"):
            LlamaCppBaseAdapter(model_path=str(wrong_ext))

    def test_default_n_gpu_layers_is_zero(self, tmp_path):
        """Default n_gpu_layers must be 0 (CPU-only)."""
        from mata.adapters.llamacpp_base import LlamaCppBaseAdapter

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"")
        adapter = LlamaCppBaseAdapter(model_path=str(gguf))
        assert adapter.n_gpu_layers == 0

    def test_custom_n_gpu_layers_stored(self, tmp_path):
        from mata.adapters.llamacpp_base import LlamaCppBaseAdapter

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"")
        adapter = LlamaCppBaseAdapter(model_path=str(gguf), n_gpu_layers=-1)
        assert adapter.n_gpu_layers == -1

    def test_default_n_ctx(self, tmp_path):
        from mata.adapters.llamacpp_base import LlamaCppBaseAdapter

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"")
        adapter = LlamaCppBaseAdapter(model_path=str(gguf))
        assert adapter.n_ctx == 2048

    def test_no_torch_import(self):
        """Verify torch is NOT imported by llamacpp_base."""
        import mata.adapters.llamacpp_base as base_mod

        source = Path(base_mod.__file__).read_text(encoding="utf-8")
        assert "import torch" not in source
        assert "from torch" not in source


# ---------------------------------------------------------------------------
# _create_llm
# ---------------------------------------------------------------------------


class TestCreateLlm:
    @pytest.fixture(autouse=True)
    def patch_llama_cpp(self, tmp_path):
        mock_llama_cpp = _make_mock_llama_cpp()
        import mata.adapters.llamacpp_base as base_mod

        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None

        with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
            yield mock_llama_cpp, tmp_path

        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None

    def test_create_llm_passes_correct_kwargs(self, patch_llama_cpp):
        mock_llama_cpp, tmp_path = patch_llama_cpp
        from mata.adapters.llamacpp_base import LlamaCppBaseAdapter

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"")
        adapter = LlamaCppBaseAdapter(model_path=str(gguf), n_gpu_layers=4, n_ctx=1024)
        adapter._create_llm()

        mock_llama_cpp.Llama.assert_called_with(
            model_path=str(gguf),
            n_gpu_layers=4,
            n_ctx=1024,
            verbose=False,
        )

    def test_create_llm_forwards_extra_kwargs(self, patch_llama_cpp):
        mock_llama_cpp, tmp_path = patch_llama_cpp
        from mata.adapters.llamacpp_base import LlamaCppBaseAdapter

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"")
        adapter = LlamaCppBaseAdapter(model_path=str(gguf))
        adapter._create_llm(embedding=True)

        mock_llama_cpp.Llama.assert_called_with(
            model_path=str(gguf),
            n_gpu_layers=0,
            n_ctx=2048,
            verbose=False,
            embedding=True,
        )


# ---------------------------------------------------------------------------
# info()
# ---------------------------------------------------------------------------


class TestInfo:
    @pytest.fixture(autouse=True)
    def patch_llama_cpp(self, tmp_path):
        mock_llama_cpp = _make_mock_llama_cpp()
        import mata.adapters.llamacpp_base as base_mod

        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None

        with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
            yield tmp_path

        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None

    def test_info_returns_expected_keys(self, patch_llama_cpp):
        from mata.adapters.llamacpp_base import LlamaCppBaseAdapter

        tmp_path = patch_llama_cpp
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"")
        adapter = LlamaCppBaseAdapter(model_path=str(gguf))
        info = adapter.info()

        assert info["backend"] == "llama-cpp-python"
        assert "model_path" in info
        assert "n_gpu_layers" in info
        assert "n_ctx" in info
        assert info["name"] == "LlamaCppBaseAdapter"
