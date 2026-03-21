"""Unit tests for LlamaCppVLMAdapter."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llama_cpp():
    m = MagicMock()
    m.Llama = MagicMock()
    m.llava_chat_handler = MagicMock()
    m.llava_chat_handler.LlavaLogitsProcessor = MagicMock(return_value=MagicMock())
    return m


def _make_adapter(tmp_path, mock_llama_cpp, **kwargs):
    """Create a LlamaCppVLMAdapter with a mock llama_cpp and a temp .gguf file."""
    gguf = tmp_path / "vlm.gguf"
    gguf.write_bytes(b"")

    # Configure mock LLM response
    fake_llm = MagicMock()
    fake_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "This is a test response."}}]
    }
    mock_llama_cpp.Llama.return_value = fake_llm

    import mata.adapters.llamacpp_base as base_mod

    base_mod._llama_cpp = None
    base_mod.LLAMA_CPP_AVAILABLE = None

    with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
        from mata.adapters.llamacpp_vlm_adapter import LlamaCppVLMAdapter

        adapter = LlamaCppVLMAdapter(model_path=str(gguf), **kwargs)

    base_mod._llama_cpp = None
    base_mod.LLAMA_CPP_AVAILABLE = None
    return adapter, fake_llm


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestLlamaCppVLMAdapterInit:
    def test_without_mmproj_no_chat_handler(self, tmp_path):
        """When mmproj=None, Llama is created without chat_handler kwarg."""
        mock_llama_cpp = _mock_llama_cpp()
        _, _ = _make_adapter(tmp_path, mock_llama_cpp)
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert "chat_handler" not in call_kwargs

    def test_with_mmproj_creates_llava_handler(self, tmp_path):
        """When mmproj is provided, LlavaLogitsProcessor is wired in."""
        mmproj = tmp_path / "proj.gguf"
        mmproj.write_bytes(b"")
        mock_llama_cpp = _mock_llama_cpp()
        _, _ = _make_adapter(tmp_path, mock_llama_cpp, mmproj=str(mmproj))
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert "chat_handler" in call_kwargs

    def test_inherits_file_not_found_from_base(self, tmp_path):
        """Missing model file → FileNotFoundError from base adapter."""
        mock_llama_cpp = _mock_llama_cpp()
        import mata.adapters.llamacpp_base as base_mod

        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None
        with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
            from mata.adapters.llamacpp_vlm_adapter import LlamaCppVLMAdapter

            with pytest.raises(FileNotFoundError):
                LlamaCppVLMAdapter(model_path=str(tmp_path / "ghost.gguf"))
        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None

    def test_default_max_tokens(self, tmp_path):
        mock_llama_cpp = _mock_llama_cpp()
        adapter, _ = _make_adapter(tmp_path, mock_llama_cpp)
        assert adapter.max_tokens == 512


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestLlamaCppVLMAdapterPredict:
    @pytest.fixture
    def adapter_and_llm(self, tmp_path):
        mock_llama_cpp = _mock_llama_cpp()
        return _make_adapter(tmp_path, mock_llama_cpp)

    def _make_pil_image(self):
        from PIL import Image

        return Image.new("RGB", (64, 64), color=(100, 150, 200))

    def test_predict_returns_vision_result_with_text(self, adapter_and_llm, tmp_path):
        """predict() returns VisionResult with .text populated."""
        from mata.core.types import VisionResult

        adapter, fake_llm = adapter_and_llm
        img_path = tmp_path / "img.png"
        self._make_pil_image().save(str(img_path))

        result = adapter.predict(str(img_path), prompt="Describe this image")
        assert isinstance(result, VisionResult)
        assert result.text == "This is a test response."

    def test_predict_text_from_choices_message_content(self, adapter_and_llm, tmp_path):
        """VisionResult.text matches choices[0].message.content."""
        adapter, fake_llm = adapter_and_llm
        fake_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "A cat sits on a mat."}}]
        }
        img_path = tmp_path / "img.png"
        self._make_pil_image().save(str(img_path))
        result = adapter.predict(str(img_path), prompt="What is this?")
        assert result.text == "A cat sits on a mat."

    def test_predict_raises_invalid_input_error_for_none_prompt(self, adapter_and_llm, tmp_path):
        """prompt=None → InvalidInputError."""
        from mata.core.exceptions import InvalidInputError

        adapter, _ = adapter_and_llm
        img_path = tmp_path / "img.png"
        self._make_pil_image().save(str(img_path))
        with pytest.raises(InvalidInputError, match="prompt is required"):
            adapter.predict(str(img_path), prompt=None)

    def test_predict_raises_invalid_input_error_for_empty_prompt(self, adapter_and_llm, tmp_path):
        """prompt='' → InvalidInputError."""
        from mata.core.exceptions import InvalidInputError

        adapter, _ = adapter_and_llm
        img_path = tmp_path / "img.png"
        self._make_pil_image().save(str(img_path))
        with pytest.raises(InvalidInputError):
            adapter.predict(str(img_path), prompt="")

    def test_predict_max_new_tokens_overrides_default(self, adapter_and_llm, tmp_path):
        """max_new_tokens kwarg is passed as max_tokens to create_chat_completion."""
        adapter, fake_llm = adapter_and_llm
        img_path = tmp_path / "img.png"
        self._make_pil_image().save(str(img_path))
        adapter.predict(str(img_path), prompt="Hello", max_new_tokens=200)
        _, call_kwargs = fake_llm.create_chat_completion.call_args
        assert call_kwargs["max_tokens"] == 200

    def test_predict_meta_contains_backend(self, adapter_and_llm, tmp_path):
        adapter, _ = adapter_and_llm
        img_path = tmp_path / "img.png"
        self._make_pil_image().save(str(img_path))
        result = adapter.predict(str(img_path), prompt="Q?")
        assert result.meta["backend"] == "llama-cpp-python"

    def test_predict_meta_contains_model_path(self, adapter_and_llm, tmp_path):
        adapter, _ = adapter_and_llm
        img_path = tmp_path / "img.png"
        self._make_pil_image().save(str(img_path))
        result = adapter.predict(str(img_path), prompt="Q?")
        assert "model_path" in result.meta

    def test_predict_instances_is_empty_list(self, adapter_and_llm, tmp_path):
        """VLM result has no detection instances."""
        adapter, _ = adapter_and_llm
        img_path = tmp_path / "img.png"
        self._make_pil_image().save(str(img_path))
        result = adapter.predict(str(img_path), prompt="Q?")
        assert result.instances == []


# ---------------------------------------------------------------------------
# VLMWrapper duck-type compatibility
# ---------------------------------------------------------------------------


class TestVLMWrapperDuckType:
    def test_vlm_wrapper_instantiates_with_llamacpp_vlm_adapter(self, tmp_path):
        """VLMWrapper(LlamaCppVLMAdapter(...)) should not raise TypeError."""
        mock_llama_cpp = _mock_llama_cpp()
        adapter, _ = _make_adapter(tmp_path, mock_llama_cpp)

        import mata.adapters.llamacpp_base as base_mod

        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None
        with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
            from mata.adapters.wrappers.vlm_wrapper import VLMWrapper

            wrapper = VLMWrapper(adapter)
        assert wrapper is not None
        base_mod._llama_cpp = None
        base_mod.LLAMA_CPP_AVAILABLE = None

    def test_llamacpp_vlm_adapter_has_predict_method(self, tmp_path):
        """Adapter must expose a callable predict() for VLMWrapper duck-type."""
        mock_llama_cpp = _mock_llama_cpp()
        adapter, _ = _make_adapter(tmp_path, mock_llama_cpp)
        assert callable(getattr(adapter, "predict", None))


# ---------------------------------------------------------------------------
# info()
# ---------------------------------------------------------------------------


class TestVLMAdapterInfo:
    def test_info_includes_task_and_mmproj(self, tmp_path):
        mock_llama_cpp = _mock_llama_cpp()
        adapter, _ = _make_adapter(tmp_path, mock_llama_cpp)
        info = adapter.info()
        assert info["task"] == "vlm"
        assert "mmproj" in info
        assert "max_tokens" in info
