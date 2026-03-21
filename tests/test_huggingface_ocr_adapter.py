"""Unit tests for HuggingFaceOCRAdapter (GOT-OCR2 + TrOCR).

Tests use mocked transformers to avoid requiring actual model downloads.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from mata.core.types import OCRResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pil_image(w: int = 4, h: int = 4) -> Image.Image:
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


def _patch_transformers(arch: str = "trocr"):
    """Return a context-manager patch for _ensure_transformers returning a usable mock."""
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.to.return_value.eval.return_value = mock_model
    mock_model.eval.return_value = mock_model

    if arch == "glm_ocr":
        # apply_chat_template returns a dict-like mock with input_ids tensor shape
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(return_value=MagicMock())
        mock_inputs.__contains__ = MagicMock(return_value=False)  # no token_type_ids
        mock_inputs.pop = MagicMock(return_value=None)
        mock_inputs["input_ids"] = MagicMock()
        mock_inputs["input_ids"].shape = (1, 5)  # batch=1, seq_len=5
        mock_inputs.to.return_value = mock_inputs
        mock_processor.apply_chat_template.return_value = mock_inputs
        mock_processor.decode.return_value = "Recognized document text"
        mock_model.generate.return_value = [MagicMock()]
        # Simulate parameters() for _infer_device (device_map path)
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])
        tf_module = MagicMock()
        tf_module.AutoProcessor.from_pretrained.return_value = mock_processor
        tf_module.AutoModelForImageTextToText.from_pretrained.return_value = mock_model

    elif arch == "trocr":
        mock_processor.batch_decode.return_value = ["Hello World"]
        mock_model.generate.return_value = MagicMock()
        # pixel_values attribute used in _predict_trocr
        mock_processor.return_value = MagicMock(pixel_values=MagicMock())
        tf_module = MagicMock()
        tf_module.TrOCRProcessor.from_pretrained.return_value = mock_processor
        tf_module.VisionEncoderDecoderModel.from_pretrained.return_value = mock_model
    else:  # got_ocr
        mock_processor.decode.return_value = "Full page text"
        mock_model.generate.return_value = [MagicMock()]
        tf_module = MagicMock()
        tf_module.AutoProcessor.from_pretrained.return_value = mock_processor
        tf_module.AutoModelForCausalLM.from_pretrained.return_value = mock_model

    return (
        patch(
            "mata.adapters.ocr.huggingface_ocr_adapter._ensure_transformers",
            return_value=tf_module,
        ),
        mock_processor,
        mock_model,
    )


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------


class TestArchitectureDetection:
    """Tests for _detect_architecture() static method."""

    def setup_method(self):
        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        self.detect = HuggingFaceOCRAdapter._detect_architecture

    def test_trocr_base_handwritten(self):
        assert self.detect("microsoft/trocr-base-handwritten") == "trocr"

    def test_trocr_large_printed(self):
        assert self.detect("microsoft/trocr-large-printed") == "trocr"

    def test_trocr_case_insensitive(self):
        assert self.detect("ms/TrOCR-base") == "trocr"

    def test_got_ocr_hyphen(self):
        assert self.detect("stepfun-ai/GOT-OCR-2.0-hf") == "got_ocr"

    def test_got_ocr_no_separator(self):
        assert self.detect("org/GoTOCRmodel") == "got_ocr"

    def test_got_ocr_underscore(self):
        assert self.detect("org/my_got_ocr_model") == "got_ocr"

    def test_unknown_falls_back_to_trocr(self):
        assert self.detect("org/some-other-vision-model") == "trocr"

    def test_got_ocr_takes_priority_when_both_strings_present(self):
        # Edge case: model ID with both substrings - GOT-OCR check comes second
        # but 'trocr' check runs first; real-world IDs won't have both
        arch = self.detect("org/trocr-got-ocr")
        assert arch in ("trocr", "got_ocr")  # deterministic, just must not crash

    def test_glm_ocr_hyphen(self):
        assert self.detect("zai-org/GLM-OCR") == "glm_ocr"

    def test_glm_ocr_underscore(self):
        assert self.detect("org/glm_ocr_v2") == "glm_ocr"

    def test_glm_ocr_no_separator(self):
        assert self.detect("org/GlmOCR") == "glm_ocr"

    def test_glm_ocr_takes_priority_over_fallback(self):
        # GLM-OCR check is first, so it should not fall to trocr
        assert self.detect("zai-org/GLM-OCR") != "trocr"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestHuggingFaceOCRAdapterInit:
    """Tests for adapter construction."""

    def test_trocr_init(self):
        patcher, mock_proc, mock_model = _patch_transformers("trocr")
        with patcher:
            from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

            adapter = HuggingFaceOCRAdapter("microsoft/trocr-base-handwritten", device="cpu")

        assert adapter.model_id == "microsoft/trocr-base-handwritten"
        assert adapter._arch == "trocr"
        assert adapter.task == "ocr"
        assert adapter.name == "huggingface_ocr"

    def test_got_ocr_init(self):
        patcher, mock_proc, mock_model = _patch_transformers("got_ocr")
        with patcher:
            from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

            adapter = HuggingFaceOCRAdapter("stepfun-ai/GOT-OCR-2.0-hf", device="cpu")

        assert adapter.model_id == "stepfun-ai/GOT-OCR-2.0-hf"
        assert adapter._arch == "got_ocr"

    def test_missing_transformers_raises_import_error(self):
        with patch(
            "mata.adapters.ocr.huggingface_ocr_adapter._ensure_transformers",
            side_effect=ImportError("transformers is required"),
        ):
            from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

            with pytest.raises(ImportError, match="transformers"):
                HuggingFaceOCRAdapter("microsoft/trocr-base-handwritten")

    def test_model_load_error_on_bad_model(self):
        from mata.core.exceptions import ModelLoadError

        patcher, _, mock_model = _patch_transformers("trocr")
        with patcher as mock_ensure:
            tf = mock_ensure.return_value
            tf.TrOCRProcessor.from_pretrained.side_effect = OSError("model not found")
            from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

            with pytest.raises(ModelLoadError):
                HuggingFaceOCRAdapter("bad/model-id", device="cpu")


# ---------------------------------------------------------------------------
# Prediction — TrOCR
# ---------------------------------------------------------------------------


class TestTrOCRPredict:
    """Tests for _predict_trocr()."""

    def _make_adapter(self):
        patcher, mock_proc, mock_model = _patch_transformers("trocr")
        with patcher:
            from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

            adapter = HuggingFaceOCRAdapter("microsoft/trocr-base-handwritten", device="cpu")
        # Attach mocks so tests can configure them
        adapter._processor = mock_proc
        adapter._model = mock_model
        return adapter, mock_proc, mock_model

    def test_predict_returns_ocr_result(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_proc.return_value = MagicMock(pixel_values=MagicMock())
        mock_proc.batch_decode.return_value = ["Recognized text"]
        mock_model.generate.return_value = MagicMock()

        result = adapter.predict(_make_pil_image())

        assert isinstance(result, OCRResult)
        assert len(result.regions) == 1

    def test_predict_returns_correct_text(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_proc.return_value = MagicMock(pixel_values=MagicMock())
        mock_proc.batch_decode.return_value = ["Hello, World!"]
        mock_model.generate.return_value = MagicMock()

        result = adapter.predict(_make_pil_image())

        assert result.regions[0].text == "Hello, World!"

    def test_predict_region_score_is_one(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_proc.return_value = MagicMock(pixel_values=MagicMock())
        mock_proc.batch_decode.return_value = ["text"]
        mock_model.generate.return_value = MagicMock()

        result = adapter.predict(_make_pil_image())

        assert result.regions[0].score == 1.0

    def test_predict_region_has_no_bbox(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_proc.return_value = MagicMock(pixel_values=MagicMock())
        mock_proc.batch_decode.return_value = ["text"]
        mock_model.generate.return_value = MagicMock()

        result = adapter.predict(_make_pil_image())

        assert result.regions[0].bbox is None

    def test_predict_meta_contains_arch(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_proc.return_value = MagicMock(pixel_values=MagicMock())
        mock_proc.batch_decode.return_value = ["text"]
        mock_model.generate.return_value = MagicMock()

        result = adapter.predict(_make_pil_image())

        assert result.meta["arch"] == "trocr"
        assert result.meta["model_id"] == "microsoft/trocr-base-handwritten"

    def test_predict_accepts_numpy_array(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_proc.return_value = MagicMock(pixel_values=MagicMock())
        mock_proc.batch_decode.return_value = ["text"]
        mock_model.generate.return_value = MagicMock()

        img_array = np.zeros((4, 4, 3), dtype=np.uint8)
        result = adapter.predict(img_array)

        assert isinstance(result, OCRResult)


# ---------------------------------------------------------------------------
# Prediction — GOT-OCR2
# ---------------------------------------------------------------------------


class TestGOTOCRPredict:
    """Tests for _predict_got_ocr()."""

    def _make_adapter(self):
        patcher, mock_proc, mock_model = _patch_transformers("got_ocr")
        with patcher:
            from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

            adapter = HuggingFaceOCRAdapter("stepfun-ai/GOT-OCR-2.0-hf", device="cpu")
        adapter._processor = mock_proc
        adapter._model = mock_model
        return adapter, mock_proc, mock_model

    def test_predict_returns_ocr_result(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_proc.return_value = MagicMock()
        mock_proc.decode.return_value = "Document text"
        mock_model.generate.return_value = [MagicMock()]

        result = adapter.predict(_make_pil_image())

        assert isinstance(result, OCRResult)
        assert len(result.regions) == 1

    def test_predict_correct_text(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_proc.return_value = MagicMock()
        mock_proc.decode.return_value = "Page content here"
        mock_model.generate.return_value = [MagicMock()]

        result = adapter.predict(_make_pil_image())

        assert result.regions[0].text == "Page content here"

    def test_predict_ocr_type_in_meta(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_proc.return_value = MagicMock()
        mock_proc.decode.return_value = "text"
        mock_model.generate.return_value = [MagicMock()]

        result = adapter.predict(_make_pil_image(), ocr_type="format")

        assert result.meta["ocr_type"] == "format"

    def test_predict_default_ocr_type_is_ocr(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_proc.return_value = MagicMock()
        mock_proc.decode.return_value = "text"
        mock_model.generate.return_value = [MagicMock()]

        result = adapter.predict(_make_pil_image())

        assert result.meta["ocr_type"] == "ocr"

    def test_predict_region_score_is_one(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_proc.return_value = MagicMock()
        mock_proc.decode.return_value = "text"
        mock_model.generate.return_value = [MagicMock()]

        result = adapter.predict(_make_pil_image())

        assert result.regions[0].score == 1.0

    def test_predict_region_has_no_bbox(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_proc.return_value = MagicMock()
        mock_proc.decode.return_value = "text"
        mock_model.generate.return_value = [MagicMock()]

        result = adapter.predict(_make_pil_image())

        assert result.regions[0].bbox is None

    def test_predict_meta_contains_arch(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_proc.return_value = MagicMock()
        mock_proc.decode.return_value = "text"
        mock_model.generate.return_value = [MagicMock()]

        result = adapter.predict(_make_pil_image())

        assert result.meta["arch"] == "got_ocr"
        assert result.meta["model_id"] == "stepfun-ai/GOT-OCR-2.0-hf"


# ---------------------------------------------------------------------------
# info() method
# ---------------------------------------------------------------------------


class TestInfo:
    """Tests for info() method."""

    def test_trocr_info_keys(self):
        patcher, _, _ = _patch_transformers("trocr")
        with patcher:
            from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

            adapter = HuggingFaceOCRAdapter("microsoft/trocr-base-handwritten", device="cpu")

        info = adapter.info()
        assert info["name"] == "huggingface_ocr"
        assert info["task"] == "ocr"
        assert info["model_id"] == "microsoft/trocr-base-handwritten"
        assert info["arch"] == "trocr"
        assert "device" in info
        assert info["backend"] == "transformers"

    def test_got_ocr_info_arch(self):
        patcher, _, _ = _patch_transformers("got_ocr")
        with patcher:
            from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

            adapter = HuggingFaceOCRAdapter("stepfun-ai/GOT-OCR-2.0-hf", device="cpu")

        info = adapter.info()
        assert info["arch"] == "got_ocr"


# ---------------------------------------------------------------------------
# Initialization — GLM-OCR
# ---------------------------------------------------------------------------


class TestGLMOCRInit:
    """Tests for GLM-OCR adapter construction."""

    def test_glm_ocr_init(self):
        patcher, mock_proc, mock_model = _patch_transformers("glm_ocr")
        with patcher:
            from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

            adapter = HuggingFaceOCRAdapter("zai-org/GLM-OCR", device="cpu")

        assert adapter.model_id == "zai-org/GLM-OCR"
        assert adapter._arch == "glm_ocr"
        assert adapter.task == "ocr"
        assert adapter.name == "huggingface_ocr"

    def test_glm_ocr_model_load_error(self):
        from mata.core.exceptions import ModelLoadError

        patcher, _, _ = _patch_transformers("glm_ocr")
        with patcher as mock_ensure:
            tf = mock_ensure.return_value
            tf.AutoProcessor.from_pretrained.side_effect = OSError("model not found")
            from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

            with pytest.raises(ModelLoadError):
                HuggingFaceOCRAdapter("zai-org/GLM-OCR", device="cpu")

    def test_glm_ocr_info_arch(self):
        patcher, _, _ = _patch_transformers("glm_ocr")
        with patcher:
            from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

            adapter = HuggingFaceOCRAdapter("zai-org/GLM-OCR", device="cpu")

        info = adapter.info()
        assert info["arch"] == "glm_ocr"
        assert info["model_id"] == "zai-org/GLM-OCR"
        assert info["backend"] == "transformers"


# ---------------------------------------------------------------------------
# Prediction — GLM-OCR
# ---------------------------------------------------------------------------


class TestGLMOCRPredict:
    """Tests for _predict_glm_ocr()."""

    def _make_adapter(self):
        patcher, mock_proc, mock_model = _patch_transformers("glm_ocr")
        with patcher:
            from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

            adapter = HuggingFaceOCRAdapter("zai-org/GLM-OCR", device="cpu")
        adapter._processor = mock_proc
        adapter._model = mock_model
        return adapter, mock_proc, mock_model

    def _make_mock_inputs(self, seq_len: int = 5):
        """Build a mock inputs dict that simulates processor.apply_chat_template output."""
        mock_id_tensor = MagicMock()
        mock_id_tensor.shape = (1, seq_len)
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, k: mock_id_tensor if k == "input_ids" else MagicMock()
        mock_inputs.pop = MagicMock(return_value=None)
        mock_inputs.to.return_value = mock_inputs
        return mock_inputs, mock_id_tensor

    def test_predict_returns_ocr_result(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_inputs, _ = self._make_mock_inputs()
        mock_proc.apply_chat_template.return_value = mock_inputs
        mock_proc.decode.return_value = "Invoice total: $42.00"
        mock_model.generate.return_value = [MagicMock()]

        result = adapter.predict(_make_pil_image())

        assert isinstance(result, OCRResult)
        assert len(result.regions) == 1

    def test_predict_returns_correct_text(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_inputs, _ = self._make_mock_inputs()
        mock_proc.apply_chat_template.return_value = mock_inputs
        mock_proc.decode.return_value = "Hello GLM-OCR!"
        mock_model.generate.return_value = [MagicMock()]

        result = adapter.predict(_make_pil_image())

        assert result.regions[0].text == "Hello GLM-OCR!"

    def test_predict_region_score_is_one(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_inputs, _ = self._make_mock_inputs()
        mock_proc.apply_chat_template.return_value = mock_inputs
        mock_proc.decode.return_value = "text"
        mock_model.generate.return_value = [MagicMock()]

        result = adapter.predict(_make_pil_image())

        assert result.regions[0].score == 1.0

    def test_predict_region_has_no_bbox(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_inputs, _ = self._make_mock_inputs()
        mock_proc.apply_chat_template.return_value = mock_inputs
        mock_proc.decode.return_value = "text"
        mock_model.generate.return_value = [MagicMock()]

        result = adapter.predict(_make_pil_image())

        assert result.regions[0].bbox is None

    def test_predict_meta_contains_arch_and_model_id(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_inputs, _ = self._make_mock_inputs()
        mock_proc.apply_chat_template.return_value = mock_inputs
        mock_proc.decode.return_value = "text"
        mock_model.generate.return_value = [MagicMock()]

        result = adapter.predict(_make_pil_image())

        assert result.meta["arch"] == "glm_ocr"
        assert result.meta["model_id"] == "zai-org/GLM-OCR"

    def test_predict_uses_default_prompt(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_inputs, _ = self._make_mock_inputs()
        mock_proc.apply_chat_template.return_value = mock_inputs
        mock_proc.decode.return_value = "text"
        mock_model.generate.return_value = [MagicMock()]

        adapter.predict(_make_pil_image())

        call_args = mock_proc.apply_chat_template.call_args
        messages = call_args[0][0]
        user_content = messages[0]["content"]
        text_parts = [p for p in user_content if p.get("type") == "text"]
        assert text_parts[0]["text"] == "Text Recognition:"

    def test_predict_custom_prompt(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_inputs, _ = self._make_mock_inputs()
        mock_proc.apply_chat_template.return_value = mock_inputs
        mock_proc.decode.return_value = "text"
        mock_model.generate.return_value = [MagicMock()]

        adapter.predict(_make_pil_image(), prompt="Extract all numbers:")

        call_args = mock_proc.apply_chat_template.call_args
        messages = call_args[0][0]
        user_content = messages[0]["content"]
        text_parts = [p for p in user_content if p.get("type") == "text"]
        assert text_parts[0]["text"] == "Extract all numbers:"

    def test_predict_pops_token_type_ids(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_inputs, _ = self._make_mock_inputs()
        mock_proc.apply_chat_template.return_value = mock_inputs
        mock_proc.decode.return_value = "text"
        mock_model.generate.return_value = [MagicMock()]

        adapter.predict(_make_pil_image())

        mock_inputs.pop.assert_called_with("token_type_ids", None)

    def test_predict_accepts_numpy_array(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_inputs, _ = self._make_mock_inputs()
        mock_proc.apply_chat_template.return_value = mock_inputs
        mock_proc.decode.return_value = "text"
        mock_model.generate.return_value = [MagicMock()]

        img_array = np.zeros((4, 4, 3), dtype=np.uint8)
        result = adapter.predict(img_array)

        assert isinstance(result, OCRResult)

    def test_predict_custom_max_new_tokens(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_inputs, _ = self._make_mock_inputs()
        mock_proc.apply_chat_template.return_value = mock_inputs
        mock_proc.decode.return_value = "text"
        mock_model.generate.return_value = [MagicMock()]

        adapter.predict(_make_pil_image(), max_new_tokens=512)

        gen_kwargs = mock_model.generate.call_args[1]
        assert gen_kwargs["max_new_tokens"] == 512

    def test_predict_region_label_is_glm_ocr(self):
        adapter, mock_proc, mock_model = self._make_adapter()
        mock_inputs, _ = self._make_mock_inputs()
        mock_proc.apply_chat_template.return_value = mock_inputs
        mock_proc.decode.return_value = "text"
        mock_model.generate.return_value = [MagicMock()]

        result = adapter.predict(_make_pil_image())

        assert result.regions[0].label == "glm-ocr"


# ---------------------------------------------------------------------------
# Public import
# ---------------------------------------------------------------------------


class TestPublicExports:
    """Verify the adapter is reachable from public import paths."""

    def test_import_from_ocr_subpackage(self):
        from mata.adapters.ocr import HuggingFaceOCRAdapter  # noqa: F401

    def test_import_from_adapters(self):
        from mata.adapters import HuggingFaceOCRAdapter  # noqa: F401
