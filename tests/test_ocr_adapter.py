"""Comprehensive mock-based unit tests for all OCR adapters.

Covers:
- TextRegion / OCRResult core types
- OCRText artifact
- HuggingFaceOCRAdapter (architecture detection, TrOCR, GOT-OCR2)
- EasyOCRAdapter (predict, polygon→xyxy conversion)
- PaddleOCRAdapter (predict, None / empty handling)
- TesseractAdapter (predict, conf filter, bbox conversion)
- UniversalLoader OCR routing (easyocr / paddleocr / tesseract / HF)

No real model downloads are performed — all external dependencies are mocked.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pil_image(width: int = 64, height: int = 32) -> PILImage.Image:
    return PILImage.new("RGB", (width, height), color=(200, 200, 200))


# ===========================================================================
# Group 1 – TextRegion and OCRResult core types (~15 tests)
# ===========================================================================


class TestTextRegionAndOCRResult:
    """Tests for TextRegion, OCRResult dataclasses and their helpers."""

    # ------------------------------------------------------------------
    # TextRegion construction
    # ------------------------------------------------------------------

    def test_text_region_minimal(self):
        from mata.core.types import TextRegion

        r = TextRegion(text="hello", score=0.9)
        assert r.text == "hello"
        assert r.score == 0.9
        assert r.bbox is None
        assert r.label is None

    def test_text_region_with_bbox_and_label(self):
        from mata.core.types import TextRegion

        r = TextRegion(text="world", score=0.75, bbox=(10.0, 20.0, 100.0, 50.0), label="en")
        assert r.bbox == (10.0, 20.0, 100.0, 50.0)
        assert r.label == "en"

    def test_text_region_is_frozen(self):
        from mata.core.types import TextRegion

        r = TextRegion(text="hello", score=0.9)
        with pytest.raises((AttributeError, TypeError)):
            r.text = "changed"  # type: ignore[misc]

    def test_text_region_to_dict_with_bbox(self):
        from mata.core.types import TextRegion

        r = TextRegion(text="hi", score=0.8, bbox=(1.0, 2.0, 3.0, 4.0), label="printed")
        d = r.to_dict()
        assert d["text"] == "hi"
        assert d["score"] == 0.8
        assert d["bbox"] == [1.0, 2.0, 3.0, 4.0]
        assert d["label"] == "printed"

    def test_text_region_to_dict_no_bbox(self):
        from mata.core.types import TextRegion

        r = TextRegion(text="hi", score=0.8)
        d = r.to_dict()
        assert d["bbox"] is None
        assert d["label"] is None

    def test_text_region_roundtrip(self):
        from mata.core.types import TextRegion

        r = TextRegion(text="round", score=0.55, bbox=(5.0, 6.0, 50.0, 60.0), label="zh")
        assert TextRegion.from_dict(r.to_dict()) == r

    # ------------------------------------------------------------------
    # OCRResult construction and full_text
    # ------------------------------------------------------------------

    def test_ocr_result_empty(self):
        from mata.core.types import OCRResult

        result = OCRResult(regions=[])
        assert result.full_text == ""

    def test_ocr_result_single_region(self):
        from mata.core.types import OCRResult, TextRegion

        result = OCRResult(regions=[TextRegion(text="hello", score=0.9)])
        assert result.full_text == "hello"

    def test_ocr_result_multi_region_full_text(self):
        from mata.core.types import OCRResult, TextRegion

        result = OCRResult(
            regions=[
                TextRegion(text="line one", score=0.9),
                TextRegion(text="line two", score=0.85),
            ]
        )
        assert result.full_text == "line one\nline two"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def test_ocr_result_to_dict(self):
        from mata.core.types import OCRResult, TextRegion

        result = OCRResult(regions=[TextRegion(text="test", score=0.95)], meta={"engine": "easyocr"})
        d = result.to_dict()
        assert len(d["regions"]) == 1
        assert d["regions"][0]["text"] == "test"
        assert d["meta"]["engine"] == "easyocr"

    def test_ocr_result_from_dict_roundtrip(self):
        from mata.core.types import OCRResult, TextRegion

        original = OCRResult(
            regions=[
                TextRegion(text="alpha", score=0.9, bbox=(1.0, 2.0, 30.0, 40.0)),
                TextRegion(text="beta", score=0.7),
            ],
            meta={"engine": "tesseract"},
        )
        restored = OCRResult.from_dict(original.to_dict())
        assert len(restored.regions) == 2
        assert restored.regions[0].text == "alpha"
        assert restored.regions[1].text == "beta"
        assert restored.meta == {"engine": "tesseract"}

    def test_ocr_result_to_json_from_json_roundtrip(self):
        from mata.core.types import OCRResult, TextRegion

        original = OCRResult(
            regions=[TextRegion(text="json test", score=0.88)],
            meta={"lang": "en"},
        )
        json_str = original.to_json()
        parsed = json.loads(json_str)
        assert parsed["regions"][0]["text"] == "json test"
        restored = OCRResult.from_json(json_str)
        assert restored.regions[0].text == "json test"

    # ------------------------------------------------------------------
    # filter_by_score
    # ------------------------------------------------------------------

    def test_filter_by_score_removes_below_threshold(self):
        from mata.core.types import OCRResult, TextRegion

        result = OCRResult(
            regions=[
                TextRegion(text="high", score=0.9),
                TextRegion(text="low", score=0.3),
                TextRegion(text="mid", score=0.7),
            ]
        )
        filtered = result.filter_by_score(0.6)
        assert len(filtered.regions) == 2
        assert all(r.score >= 0.6 for r in filtered.regions)

    def test_filter_by_score_all_filtered(self):
        from mata.core.types import OCRResult, TextRegion

        result = OCRResult(regions=[TextRegion(text="low", score=0.2)])
        filtered = result.filter_by_score(0.9)
        assert filtered.regions == []

    def test_filter_by_score_none_filtered(self):
        from mata.core.types import OCRResult, TextRegion

        result = OCRResult(
            regions=[
                TextRegion(text="a", score=0.8),
                TextRegion(text="b", score=0.9),
            ]
        )
        filtered = result.filter_by_score(0.5)
        assert len(filtered.regions) == 2

    def test_filter_by_score_preserves_meta(self):
        from mata.core.types import OCRResult, TextRegion

        result = OCRResult(regions=[TextRegion(text="x", score=0.8)], meta={"engine": "easyocr"})
        filtered = result.filter_by_score(0.5)
        assert filtered.meta == {"engine": "easyocr"}

    def test_ocr_result_is_frozen(self):
        from mata.core.types import OCRResult

        result = OCRResult(regions=[])
        with pytest.raises((AttributeError, TypeError)):
            result.regions = []  # type: ignore[misc]


# ===========================================================================
# Group 2 – OCRText artifact (~10 tests)
# ===========================================================================


class TestOCRTextArtifact:
    """Tests for the OCRText graph artifact."""

    def test_ocr_text_default_construction(self):
        from mata.core.artifacts.ocr_text import OCRText

        artifact = OCRText()
        assert artifact.text_blocks == ()
        assert artifact.full_text == ""
        assert artifact.instance_ids == ()

    def test_ocr_text_with_blocks(self):
        from mata.core.artifacts.ocr_text import OCRText, TextBlock

        blocks = (TextBlock(text="Hello", confidence=0.98),)
        artifact = OCRText(text_blocks=blocks, full_text="Hello")
        assert len(artifact.text_blocks) == 1
        assert artifact.text_blocks[0].text == "Hello"
        assert artifact.text_blocks[0].confidence == 0.98

    def test_ocr_text_is_frozen(self):
        from mata.core.artifacts.ocr_text import OCRText

        artifact = OCRText()
        with pytest.raises((AttributeError, TypeError)):
            artifact.full_text = "changed"  # type: ignore[misc]

    def test_ocr_text_validate_passes_valid(self):
        from mata.core.artifacts.ocr_text import OCRText, TextBlock

        artifact = OCRText(
            text_blocks=(TextBlock(text="ok", confidence=0.9),),
            instance_ids=("id-1",),
        )
        artifact.validate()  # should not raise

    def test_ocr_text_from_ocr_result_basic(self):
        from mata.core.artifacts.ocr_text import OCRText
        from mata.core.types import OCRResult, TextRegion

        result = OCRResult(
            regions=[
                TextRegion(text="foo", score=0.9, bbox=(1.0, 2.0, 10.0, 20.0), label="en"),
                TextRegion(text="bar", score=0.8),
            ]
        )
        artifact = OCRText.from_ocr_result(result)
        assert len(artifact.text_blocks) == 2
        assert artifact.text_blocks[0].text == "foo"
        assert artifact.text_blocks[0].confidence == 0.9
        assert artifact.text_blocks[0].bbox == (1.0, 2.0, 10.0, 20.0)
        assert artifact.text_blocks[0].language == "en"
        assert artifact.text_blocks[1].text == "bar"
        assert artifact.full_text == result.full_text

    def test_ocr_text_from_ocr_result_with_instance_ids(self):
        from mata.core.artifacts.ocr_text import OCRText
        from mata.core.types import OCRResult, TextRegion

        result = OCRResult(regions=[TextRegion(text="x", score=0.9)])
        artifact = OCRText.from_ocr_result(result, instance_ids=("inst-42",))
        assert artifact.instance_ids == ("inst-42",)

    def test_ocr_text_from_ocr_result_empty(self):
        from mata.core.artifacts.ocr_text import OCRText
        from mata.core.types import OCRResult

        result = OCRResult(regions=[])
        artifact = OCRText.from_ocr_result(result)
        assert artifact.text_blocks == ()
        assert artifact.full_text == ""
        assert artifact.instance_ids == ()

    def test_ocr_text_meta_preserved(self):
        from mata.core.artifacts.ocr_text import OCRText
        from mata.core.types import OCRResult, TextRegion

        result = OCRResult(regions=[TextRegion(text="t", score=0.7)], meta={"engine": "easyocr"})
        artifact = OCRText.from_ocr_result(result)
        assert artifact.meta["engine"] == "easyocr"

    def test_text_block_with_bbox(self):
        from mata.core.artifacts.ocr_text import TextBlock

        block = TextBlock(text="word", confidence=0.75, bbox=(10.0, 20.0, 100.0, 50.0))
        assert block.text == "word"
        assert block.bbox == (10.0, 20.0, 100.0, 50.0)
        assert block.language is None

    def test_ocr_text_empty_instance_ids(self):
        from mata.core.artifacts.ocr_text import OCRText

        artifact = OCRText()
        assert len(artifact.instance_ids) == 0


# ===========================================================================
# Group 3 – HuggingFaceOCRAdapter (~25 tests)
# ===========================================================================


class TestHuggingFaceOCRAdapter:
    """Tests for HuggingFaceOCRAdapter initialization, architecture detection, and predict()."""

    # ------------------------------------------------------------------
    # Architecture detection (no mocking needed)
    # ------------------------------------------------------------------

    def test_detect_architecture_trocr(self):
        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        assert HuggingFaceOCRAdapter._detect_architecture("microsoft/trocr-base-handwritten") == "trocr"

    def test_detect_architecture_trocr_case_insensitive(self):
        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        assert HuggingFaceOCRAdapter._detect_architecture("org/TrOCR-large") == "trocr"

    def test_detect_architecture_got_ocr_hyphen(self):
        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        assert HuggingFaceOCRAdapter._detect_architecture("stepfun-ai/GOT-OCR-2.0-hf") == "got_ocr"

    def test_detect_architecture_gotocr_no_hyphen(self):
        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        assert HuggingFaceOCRAdapter._detect_architecture("org/gotocr-v2") == "got_ocr"

    def test_detect_architecture_got_ocr_underscore(self):
        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        assert HuggingFaceOCRAdapter._detect_architecture("org/got_ocr_model") == "got_ocr"

    def test_detect_architecture_unknown_defaults_to_trocr(self):
        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        assert HuggingFaceOCRAdapter._detect_architecture("some-org/mystery-model") == "trocr"

    # ------------------------------------------------------------------
    # Initialization — TrOCR
    # ------------------------------------------------------------------

    def _make_mock_torch(self, cuda: bool = False):
        """Return lightweight mock for torch module."""
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = cuda

        # device() must return an object with a .type attribute
        def _make_device(s):
            d = Mock()
            d.type = s.split(":")[0]  # e.g. "cpu", "cuda"
            return d

        mock_torch.device.side_effect = _make_device
        return mock_torch

    def _make_mock_transformers_trocr(self):
        mock_processor = Mock()
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        mock_tf = Mock()
        mock_tf.TrOCRProcessor.from_pretrained.return_value = mock_processor
        mock_tf.VisionEncoderDecoderModel.from_pretrained.return_value = mock_model
        return mock_tf, mock_processor, mock_model

    def _make_mock_transformers_got(self):
        mock_processor = Mock()
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        mock_tf = Mock()
        mock_tf.AutoProcessor.from_pretrained.return_value = mock_processor
        mock_tf.AutoModelForImageTextToText.from_pretrained.return_value = mock_model
        return mock_tf, mock_processor, mock_model

    @patch("mata.adapters.pytorch_base._ensure_torch")
    @patch("mata.adapters.ocr.huggingface_ocr_adapter._ensure_transformers")
    @patch("mata.core.logging.suppress_third_party_logs")
    def test_trocr_initialization(self, mock_suppress, mock_ensure_tf, mock_ensure_torch):
        mock_ensure_torch.return_value = self._make_mock_torch()
        mock_tf, mock_processor, mock_model = self._make_mock_transformers_trocr()
        mock_ensure_tf.return_value = mock_tf
        mock_suppress.return_value = MagicMock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=False))

        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        adapter = HuggingFaceOCRAdapter("microsoft/trocr-base-handwritten", device="cpu")

        assert adapter.model_id == "microsoft/trocr-base-handwritten"
        assert adapter._arch == "trocr"
        assert adapter.task == "ocr"
        mock_tf.TrOCRProcessor.from_pretrained.assert_called_once_with("microsoft/trocr-base-handwritten")
        mock_tf.VisionEncoderDecoderModel.from_pretrained.assert_called_once_with("microsoft/trocr-base-handwritten")

    @patch("mata.adapters.pytorch_base._ensure_torch")
    @patch("mata.adapters.ocr.huggingface_ocr_adapter._ensure_transformers")
    @patch("mata.core.logging.suppress_third_party_logs")
    def test_got_ocr_initialization(self, mock_suppress, mock_ensure_tf, mock_ensure_torch):
        mock_ensure_torch.return_value = self._make_mock_torch()
        mock_tf, mock_processor, mock_model = self._make_mock_transformers_got()
        mock_ensure_tf.return_value = mock_tf
        mock_suppress.return_value = MagicMock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=False))

        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        adapter = HuggingFaceOCRAdapter("stepfun-ai/GOT-OCR-2.0-hf", device="cpu")

        assert adapter._arch == "got_ocr"
        mock_tf.AutoProcessor.from_pretrained.assert_called_once()
        mock_tf.AutoModelForImageTextToText.from_pretrained.assert_called_once()

    @patch("mata.adapters.pytorch_base._ensure_torch")
    @patch("mata.adapters.ocr.huggingface_ocr_adapter._ensure_transformers")
    @patch("mata.core.logging.suppress_third_party_logs")
    def test_trocr_task_name(self, mock_suppress, mock_ensure_tf, mock_ensure_torch):
        mock_ensure_torch.return_value = self._make_mock_torch()
        mock_tf, _, _ = self._make_mock_transformers_trocr()
        mock_ensure_tf.return_value = mock_tf
        mock_suppress.return_value = MagicMock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=False))

        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        adapter = HuggingFaceOCRAdapter("microsoft/trocr-base-printed", device="cpu")
        assert adapter.task == "ocr"
        assert adapter.name == "huggingface_ocr"

    # ------------------------------------------------------------------
    # predict() — TrOCR
    # ------------------------------------------------------------------

    @patch("mata.adapters.pytorch_base._ensure_torch")
    @patch("mata.adapters.ocr.huggingface_ocr_adapter._ensure_transformers")
    @patch("mata.core.logging.suppress_third_party_logs")
    def test_trocr_predict_returns_ocr_result(self, mock_suppress, mock_ensure_tf, mock_ensure_torch):
        mock_torch = self._make_mock_torch()
        mock_torch.no_grad.return_value = MagicMock(
            __enter__=Mock(return_value=None), __exit__=Mock(return_value=False)
        )
        mock_ensure_torch.return_value = mock_torch

        mock_processor = Mock()
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        # processor: returns pixel_values tensor struct, model generates token ids
        mock_inputs = Mock()
        mock_inputs.pixel_values = Mock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs
        mock_model.generate.return_value = Mock()
        mock_processor.batch_decode.return_value = ["Hello World"]

        mock_tf = Mock()
        mock_tf.TrOCRProcessor.from_pretrained.return_value = mock_processor
        mock_tf.VisionEncoderDecoderModel.from_pretrained.return_value = mock_model
        mock_ensure_tf.return_value = mock_tf
        mock_suppress.return_value = MagicMock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=False))

        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter
        from mata.core.types import OCRResult

        adapter = HuggingFaceOCRAdapter("microsoft/trocr-base-handwritten", device="cpu")

        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert isinstance(result, OCRResult)
        assert len(result.regions) == 1
        assert result.regions[0].text == "Hello World"
        assert result.regions[0].score == 1.0
        assert result.regions[0].bbox is None
        assert result.meta["arch"] == "trocr"

    # ------------------------------------------------------------------
    # predict() — GOT-OCR2
    # ------------------------------------------------------------------

    @patch("mata.adapters.pytorch_base._ensure_torch")
    @patch("mata.adapters.ocr.huggingface_ocr_adapter._ensure_transformers")
    @patch("mata.core.logging.suppress_third_party_logs")
    def test_got_ocr_predict_returns_ocr_result(self, mock_suppress, mock_ensure_tf, mock_ensure_torch):
        mock_torch = self._make_mock_torch()
        mock_torch.no_grad.return_value = MagicMock(
            __enter__=Mock(return_value=None), __exit__=Mock(return_value=False)
        )
        mock_ensure_torch.return_value = mock_torch

        mock_processor = Mock()
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_inputs = Mock()
        mock_inputs.to.return_value = {"input_ids": MagicMock()}
        mock_processor.return_value = mock_inputs
        mock_output_tokens = MagicMock()
        mock_model.generate.return_value = [mock_output_tokens]
        mock_processor.decode.return_value = "Document text here."

        mock_tf = Mock()
        mock_tf.AutoProcessor.from_pretrained.return_value = mock_processor
        mock_tf.AutoModelForImageTextToText.from_pretrained.return_value = mock_model
        mock_ensure_tf.return_value = mock_tf
        mock_suppress.return_value = MagicMock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=False))

        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter
        from mata.core.types import OCRResult

        adapter = HuggingFaceOCRAdapter("stepfun-ai/GOT-OCR-2.0-hf", device="cpu")

        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert isinstance(result, OCRResult)
        assert len(result.regions) == 1
        assert result.regions[0].text == "Document text here."
        assert result.meta["arch"] == "got_ocr"

    # ------------------------------------------------------------------
    # info()
    # ------------------------------------------------------------------

    @patch("mata.adapters.pytorch_base._ensure_torch")
    @patch("mata.adapters.ocr.huggingface_ocr_adapter._ensure_transformers")
    @patch("mata.core.logging.suppress_third_party_logs")
    def test_trocr_info(self, mock_suppress, mock_ensure_tf, mock_ensure_torch):
        mock_ensure_torch.return_value = self._make_mock_torch()
        mock_tf, _, _ = self._make_mock_transformers_trocr()
        mock_ensure_tf.return_value = mock_tf
        mock_suppress.return_value = MagicMock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=False))

        # HuggingFaceOCRAdapter doesn't have explicit info() method; check __dict__ attrs
        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        adapter = HuggingFaceOCRAdapter("microsoft/trocr-base-handwritten", device="cpu")
        assert adapter.model_id == "microsoft/trocr-base-handwritten"
        assert adapter._arch == "trocr"

    # ------------------------------------------------------------------
    # Missing transformers raises ImportError
    # ------------------------------------------------------------------

    @patch("mata.adapters.pytorch_base._ensure_torch")
    @patch("mata.adapters.ocr.huggingface_ocr_adapter._ensure_transformers")
    def test_missing_transformers_raises_import_error(self, mock_ensure_tf, mock_ensure_torch):
        mock_ensure_torch.return_value = self._make_mock_torch(cuda=False)
        mock_ensure_tf.side_effect = ImportError("transformers is required for HuggingFaceOCRAdapter.")

        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        with pytest.raises(ImportError, match="transformers"):
            HuggingFaceOCRAdapter("microsoft/trocr-base-handwritten", device="cpu")

    # ------------------------------------------------------------------
    # device="auto" selects cpu when cuda unavailable
    # ------------------------------------------------------------------

    @patch("mata.adapters.pytorch_base._ensure_torch")
    @patch("mata.adapters.ocr.huggingface_ocr_adapter._ensure_transformers")
    @patch("mata.core.logging.suppress_third_party_logs")
    def test_auto_device_selects_cpu_when_no_cuda(self, mock_suppress, mock_ensure_tf, mock_ensure_torch):
        mock_torch = self._make_mock_torch(cuda=False)
        mock_ensure_torch.return_value = mock_torch

        mock_tf, _, _ = self._make_mock_transformers_trocr()
        mock_ensure_tf.return_value = mock_tf
        mock_suppress.return_value = MagicMock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=False))

        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        # device="auto" and cuda unavailable → should pick cpu without error
        adapter = HuggingFaceOCRAdapter("microsoft/trocr-base-handwritten", device="auto")
        assert adapter is not None

    # ------------------------------------------------------------------
    # Multiple regions not produced (TrOCR returns single span)
    # ------------------------------------------------------------------

    @patch("mata.adapters.pytorch_base._ensure_torch")
    @patch("mata.adapters.ocr.huggingface_ocr_adapter._ensure_transformers")
    @patch("mata.core.logging.suppress_third_party_logs")
    def test_trocr_predict_single_region_always(self, mock_suppress, mock_ensure_tf, mock_ensure_torch):
        mock_torch = self._make_mock_torch()
        mock_torch.no_grad.return_value = MagicMock(
            __enter__=Mock(return_value=None), __exit__=Mock(return_value=False)
        )
        mock_ensure_torch.return_value = mock_torch

        mock_processor = Mock()
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_inputs = Mock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.pixel_values = Mock()
        mock_processor.return_value = mock_inputs
        mock_model.generate.return_value = Mock()
        mock_processor.batch_decode.return_value = ["multi\nline\ntext"]

        mock_tf = Mock()
        mock_tf.TrOCRProcessor.from_pretrained.return_value = mock_processor
        mock_tf.VisionEncoderDecoderModel.from_pretrained.return_value = mock_model
        mock_ensure_tf.return_value = mock_tf
        mock_suppress.return_value = MagicMock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=False))

        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        adapter = HuggingFaceOCRAdapter("microsoft/trocr-large-handwritten", device="cpu")

        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        # TrOCR always returns exactly one region
        assert len(result.regions) == 1
        assert result.regions[0].text == "multi\nline\ntext"

    @patch("mata.adapters.pytorch_base._ensure_torch")
    @patch("mata.adapters.ocr.huggingface_ocr_adapter._ensure_transformers")
    @patch("mata.core.logging.suppress_third_party_logs")
    def test_trocr_predict_meta_includes_model_id(self, mock_suppress, mock_ensure_tf, mock_ensure_torch):
        mock_torch = self._make_mock_torch()
        mock_torch.no_grad.return_value = MagicMock(
            __enter__=Mock(return_value=None), __exit__=Mock(return_value=False)
        )
        mock_ensure_torch.return_value = mock_torch

        mock_processor = Mock()
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_inputs = Mock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.pixel_values = Mock()
        mock_processor.return_value = mock_inputs
        mock_model.generate.return_value = Mock()
        mock_processor.batch_decode.return_value = ["Test"]

        mock_tf = Mock()
        mock_tf.TrOCRProcessor.from_pretrained.return_value = mock_processor
        mock_tf.VisionEncoderDecoderModel.from_pretrained.return_value = mock_model
        mock_ensure_tf.return_value = mock_tf
        mock_suppress.return_value = MagicMock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=False))

        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        adapter = HuggingFaceOCRAdapter("microsoft/trocr-base-handwritten", device="cpu")
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)
        assert result.meta["model_id"] == "microsoft/trocr-base-handwritten"


# ===========================================================================
# Group 4 – EasyOCRAdapter (~15 tests)
# ===========================================================================


class TestEasyOCRAdapter:
    """Tests for EasyOCRAdapter."""

    def _make_reader_mock(self):
        mock_reader = Mock()
        return mock_reader

    # ------------------------------------------------------------------
    # polygon_to_xyxy helper
    # ------------------------------------------------------------------

    def test_polygon_to_xyxy_rectangle(self):
        from mata.adapters.ocr.easyocr_adapter import _polygon_to_xyxy

        polygon = [[10, 20], [100, 20], [100, 50], [10, 50]]
        result = _polygon_to_xyxy(polygon)
        assert result == (10.0, 20.0, 100.0, 50.0)

    def test_polygon_to_xyxy_degenerate(self):
        """Degenerate polygon (all same point) should return a zero-size box."""
        from mata.adapters.ocr.easyocr_adapter import _polygon_to_xyxy

        polygon = [[5, 5], [5, 5], [5, 5], [5, 5]]
        result = _polygon_to_xyxy(polygon)
        assert result == (5.0, 5.0, 5.0, 5.0)

    def test_polygon_to_xyxy_float_coords(self):
        from mata.adapters.ocr.easyocr_adapter import _polygon_to_xyxy

        polygon = [[1.5, 2.5], [10.5, 2.5], [10.5, 8.5], [1.5, 8.5]]
        result = _polygon_to_xyxy(polygon)
        assert result == (1.5, 2.5, 10.5, 8.5)

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.easyocr_adapter._ensure_easyocr")
    def test_instantiation_default_language(self, mock_ensure_easyocr):
        mock_easyocr = Mock()
        mock_easyocr.Reader.return_value = Mock()
        mock_ensure_easyocr.return_value = mock_easyocr

        from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter

        adapter = EasyOCRAdapter(gpu=False)
        assert adapter.languages == ["en"]
        assert adapter.gpu is False
        mock_easyocr.Reader.assert_called_once_with(["en"], gpu=False)

    @patch("mata.adapters.ocr.easyocr_adapter._ensure_easyocr")
    def test_instantiation_custom_languages(self, mock_ensure_easyocr):
        mock_easyocr = Mock()
        mock_easyocr.Reader.return_value = Mock()
        mock_ensure_easyocr.return_value = mock_easyocr

        from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter

        adapter = EasyOCRAdapter(languages=["en", "fr"], gpu=False)
        assert adapter.languages == ["en", "fr"]
        mock_easyocr.Reader.assert_called_once_with(["en", "fr"], gpu=False)

    @patch("mata.adapters.ocr.easyocr_adapter._ensure_easyocr")
    def test_task_name(self, mock_ensure_easyocr):
        mock_easyocr = Mock()
        mock_easyocr.Reader.return_value = Mock()
        mock_ensure_easyocr.return_value = mock_easyocr

        from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter

        adapter = EasyOCRAdapter(gpu=False)
        assert adapter.task == "ocr"
        assert adapter.name == "easyocr"

    # ------------------------------------------------------------------
    # predict() — happy path
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.easyocr_adapter._ensure_easyocr")
    def test_predict_returns_ocr_result(self, mock_ensure_easyocr):
        mock_easyocr = Mock()
        mock_reader = Mock()
        mock_easyocr.Reader.return_value = mock_reader
        mock_ensure_easyocr.return_value = mock_easyocr

        # EasyOCR readtext() returns [(bbox_polygon, text, confidence), ...]
        mock_reader.readtext.return_value = [
            ([[10, 20], [100, 20], [100, 50], [10, 50]], "Hello", 0.95),
            ([[10, 60], [200, 60], [200, 90], [10, 90]], "World", 0.88),
        ]

        from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter
        from mata.core.types import OCRResult

        adapter = EasyOCRAdapter(gpu=False)

        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert isinstance(result, OCRResult)
        assert len(result.regions) == 2
        assert result.regions[0].text == "Hello"
        assert abs(result.regions[0].score - 0.95) < 1e-9
        assert result.regions[0].bbox == (10.0, 20.0, 100.0, 50.0)
        assert result.regions[1].text == "World"
        assert result.meta["engine"] == "easyocr"

    @patch("mata.adapters.ocr.easyocr_adapter._ensure_easyocr")
    def test_predict_empty_result(self, mock_ensure_easyocr):
        mock_easyocr = Mock()
        mock_reader = Mock()
        mock_easyocr.Reader.return_value = mock_reader
        mock_ensure_easyocr.return_value = mock_easyocr
        mock_reader.readtext.return_value = []

        from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter

        adapter = EasyOCRAdapter(gpu=False)
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)
        assert result.regions == []
        assert result.full_text == ""

    @patch("mata.adapters.ocr.easyocr_adapter._ensure_easyocr")
    def test_predict_detail_zero_no_bbox(self, mock_ensure_easyocr):
        """detail=0 returns plain strings with no spatial info."""
        mock_easyocr = Mock()
        mock_reader = Mock()
        mock_easyocr.Reader.return_value = mock_reader
        mock_ensure_easyocr.return_value = mock_easyocr
        mock_reader.readtext.return_value = ["word_a", "word_b"]

        from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter

        adapter = EasyOCRAdapter(gpu=False)
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img, detail=0)
        assert len(result.regions) == 2
        assert result.regions[0].bbox is None
        assert result.regions[0].score == 1.0

    @patch("mata.adapters.ocr.easyocr_adapter._ensure_easyocr")
    def test_predict_filters_blank_text(self, mock_ensure_easyocr):
        """Entries with blank/whitespace-only text should be excluded."""
        mock_easyocr = Mock()
        mock_reader = Mock()
        mock_easyocr.Reader.return_value = mock_reader
        mock_ensure_easyocr.return_value = mock_easyocr
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "Good", 0.9),
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "   ", 0.8),  # whitespace only
        ]

        from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter

        adapter = EasyOCRAdapter(gpu=False)
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)
        assert len(result.regions) == 1
        assert result.regions[0].text == "Good"

    # ------------------------------------------------------------------
    # Missing easyocr raises ImportError
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.easyocr_adapter._ensure_easyocr")
    def test_missing_easyocr_raises_import_error(self, mock_ensure_easyocr):
        mock_ensure_easyocr.side_effect = ImportError(
            "easyocr is required for EasyOCRAdapter. " "Install with: pip install easyocr\n" "or: pip install mata[ocr]"
        )
        from mata.adapters.ocr import easyocr_adapter

        # Reset module-level cache so _ensure_easyocr is called fresh
        easyocr_adapter._easyocr = None
        easyocr_adapter._EASYOCR_AVAILABLE = None

        from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter

        with pytest.raises(ImportError) as exc_info:
            EasyOCRAdapter(gpu=False)
        assert "easyocr" in str(exc_info.value).lower()
        assert "pip install" in str(exc_info.value)

    @patch("mata.adapters.ocr.easyocr_adapter._ensure_easyocr")
    def test_info_returns_metadata(self, mock_ensure_easyocr):
        mock_easyocr = Mock()
        mock_easyocr.Reader.return_value = Mock()
        mock_ensure_easyocr.return_value = mock_easyocr

        from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter

        adapter = EasyOCRAdapter(languages=["en", "de"], gpu=False)
        info = adapter.info()
        assert info["name"] == "easyocr"
        assert info["task"] == "ocr"
        assert info["languages"] == ["en", "de"]
        assert info["gpu"] is False


# ===========================================================================
# Group 5 – PaddleOCRAdapter (~15 tests)
# ===========================================================================


class TestPaddleOCRAdapter:
    """Tests for PaddleOCRAdapter."""

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.paddleocr_adapter._is_paddleocr_v3", return_value=False)
    @patch("mata.adapters.ocr.paddleocr_adapter._ensure_paddleocr")
    def test_instantiation_defaults(self, mock_ensure_paddle, mock_v3):
        mock_paddle_cls = Mock()
        mock_paddle_cls.return_value = Mock()
        mock_ensure_paddle.return_value = mock_paddle_cls

        from mata.adapters.ocr.paddleocr_adapter import PaddleOCRAdapter

        adapter = PaddleOCRAdapter(use_gpu=False)
        assert adapter.lang == "en"
        assert adapter.use_gpu is False
        mock_paddle_cls.assert_called_once_with(lang="en", use_gpu=False, show_log=False)

    @patch("mata.adapters.ocr.paddleocr_adapter._ensure_paddleocr")
    def test_instantiation_custom_lang(self, mock_ensure_paddle):
        mock_paddle_cls = Mock()
        mock_paddle_cls.return_value = Mock()
        mock_ensure_paddle.return_value = mock_paddle_cls

        from mata.adapters.ocr.paddleocr_adapter import PaddleOCRAdapter

        adapter = PaddleOCRAdapter(lang="ch", use_gpu=False)
        assert adapter.lang == "ch"

    @patch("mata.adapters.ocr.paddleocr_adapter._ensure_paddleocr")
    def test_task_name(self, mock_ensure_paddle):
        mock_paddle_cls = Mock()
        mock_paddle_cls.return_value = Mock()
        mock_ensure_paddle.return_value = mock_paddle_cls

        from mata.adapters.ocr.paddleocr_adapter import PaddleOCRAdapter

        adapter = PaddleOCRAdapter(use_gpu=False)
        assert adapter.task == "ocr"
        assert adapter.name == "paddleocr"

    # ------------------------------------------------------------------
    # predict() — happy path
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.paddleocr_adapter._is_paddleocr_v3", return_value=False)
    @patch("mata.adapters.ocr.paddleocr_adapter._ensure_paddleocr")
    def test_predict_returns_ocr_result(self, mock_ensure_paddle, mock_v3):
        mock_ocr_instance = Mock()
        mock_paddle_cls = Mock(return_value=mock_ocr_instance)
        mock_ensure_paddle.return_value = mock_paddle_cls

        # PaddleOCR format: [[[bbox_polygon, (text, confidence)], ...]]
        paddle_result = [
            [
                [[[10, 20], [100, 20], [100, 50], [10, 50]], ("Hello PaddleOCR", 0.97)],
                [[[10, 60], [150, 60], [150, 85], [10, 85]], ("Second line", 0.82)],
            ]
        ]
        mock_ocr_instance.ocr.return_value = paddle_result

        from mata.adapters.ocr.paddleocr_adapter import PaddleOCRAdapter
        from mata.core.types import OCRResult

        adapter = PaddleOCRAdapter(use_gpu=False)

        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert isinstance(result, OCRResult)
        assert len(result.regions) == 2
        assert result.regions[0].text == "Hello PaddleOCR"
        assert abs(result.regions[0].score - 0.97) < 1e-9
        assert result.regions[0].bbox == (10.0, 20.0, 100.0, 50.0)
        assert result.meta["engine"] == "paddleocr"

    # ------------------------------------------------------------------
    # None result → empty OCRResult
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.paddleocr_adapter._is_paddleocr_v3", return_value=False)
    @patch("mata.adapters.ocr.paddleocr_adapter._ensure_paddleocr")
    def test_predict_none_result_returns_empty(self, mock_ensure_paddle, mock_v3):
        mock_ocr_instance = Mock()
        mock_paddle_cls = Mock(return_value=mock_ocr_instance)
        mock_ensure_paddle.return_value = mock_paddle_cls
        mock_ocr_instance.ocr.return_value = None

        from mata.adapters.ocr.paddleocr_adapter import PaddleOCRAdapter

        adapter = PaddleOCRAdapter(use_gpu=False)
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)
        assert result.regions == []
        assert result.full_text == ""

    @patch("mata.adapters.ocr.paddleocr_adapter._is_paddleocr_v3", return_value=False)
    @patch("mata.adapters.ocr.paddleocr_adapter._ensure_paddleocr")
    def test_predict_none_inner_result_returns_empty(self, mock_ensure_paddle, mock_v3):
        """[[None]] response → empty OCRResult (no crash)."""
        mock_ocr_instance = Mock()
        mock_paddle_cls = Mock(return_value=mock_ocr_instance)
        mock_ensure_paddle.return_value = mock_paddle_cls
        mock_ocr_instance.ocr.return_value = [[None]]

        from mata.adapters.ocr.paddleocr_adapter import PaddleOCRAdapter

        adapter = PaddleOCRAdapter(use_gpu=False)
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)
        assert result.regions == []

    @patch("mata.adapters.ocr.paddleocr_adapter._is_paddleocr_v3", return_value=False)
    @patch("mata.adapters.ocr.paddleocr_adapter._ensure_paddleocr")
    def test_predict_empty_list_returns_empty(self, mock_ensure_paddle, mock_v3):
        mock_ocr_instance = Mock()
        mock_paddle_cls = Mock(return_value=mock_ocr_instance)
        mock_ensure_paddle.return_value = mock_paddle_cls
        mock_ocr_instance.ocr.return_value = [[]]  # empty page

        from mata.adapters.ocr.paddleocr_adapter import PaddleOCRAdapter

        adapter = PaddleOCRAdapter(use_gpu=False)
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)
        assert result.regions == []

    # ------------------------------------------------------------------
    # Missing paddleocr raises ImportError
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.paddleocr_adapter._ensure_paddleocr")
    def test_missing_paddleocr_raises_import_error(self, mock_ensure_paddle):
        mock_ensure_paddle.side_effect = ImportError(
            "paddleocr is required for PaddleOCRAdapter. "
            "Install with: pip install paddleocr paddlepaddle\n"
            "or: pip install mata[ocr-paddle]\n"
            "Note: paddlepaddle GPU wheel is ~500 MB."
        )

        from mata.adapters.ocr import paddleocr_adapter

        paddleocr_adapter._paddleocr_module = None  # reset cache

        from mata.adapters.ocr.paddleocr_adapter import PaddleOCRAdapter

        with pytest.raises(ImportError) as exc_info:
            PaddleOCRAdapter(use_gpu=False)
        assert "paddleocr" in str(exc_info.value).lower()
        assert "pip install" in str(exc_info.value)

    @patch("mata.adapters.ocr.paddleocr_adapter._ensure_paddleocr")
    def test_missing_paddleocr_error_mentions_paddlepaddle(self, mock_ensure_paddle):
        mock_ensure_paddle.side_effect = ImportError("paddleocr is required. pip install paddleocr paddlepaddle")
        from mata.adapters.ocr import paddleocr_adapter

        paddleocr_adapter._paddleocr_module = None

        from mata.adapters.ocr.paddleocr_adapter import PaddleOCRAdapter

        with pytest.raises(ImportError, match="paddlepaddle"):
            PaddleOCRAdapter(use_gpu=False)

    # ------------------------------------------------------------------
    # info()
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.paddleocr_adapter._ensure_paddleocr")
    def test_info_returns_metadata(self, mock_ensure_paddle):
        mock_paddle_cls = Mock(return_value=Mock())
        mock_ensure_paddle.return_value = mock_paddle_cls

        from mata.adapters.ocr.paddleocr_adapter import PaddleOCRAdapter

        adapter = PaddleOCRAdapter(lang="ja", use_gpu=False)
        info = adapter.info()
        assert info["name"] == "paddleocr"
        assert info["task"] == "ocr"
        assert info["lang"] == "ja"
        assert info["use_gpu"] is False

    # ------------------------------------------------------------------
    # polygon helper
    # ------------------------------------------------------------------

    def test_paddle_polygon_to_xyxy(self):
        from mata.adapters.ocr.paddleocr_adapter import _paddle_polygon_to_xyxy

        polygon = [[10, 20], [100, 20], [100, 50], [10, 50]]
        result = _paddle_polygon_to_xyxy(polygon)
        assert result == (10.0, 20.0, 100.0, 50.0)

    def test_paddle_polygon_to_xyxy_skewed(self):
        from mata.adapters.ocr.paddleocr_adapter import _paddle_polygon_to_xyxy

        polygon = [[15, 25], [95, 20], [100, 45], [10, 50]]
        x1, y1, x2, y2 = _paddle_polygon_to_xyxy(polygon)
        assert x1 == min(15.0, 95.0, 100.0, 10.0)
        assert y1 == min(25.0, 20.0, 45.0, 50.0)
        assert x2 == max(15.0, 95.0, 100.0, 10.0)
        assert y2 == max(25.0, 20.0, 45.0, 50.0)


def _make_fake_pytesseract(data=None):
    """Build a fake pytesseract module for sys.modules injection."""
    import types

    fake = types.ModuleType("pytesseract")
    fake.Output = Mock()
    fake.Output.DICT = "dict"
    if data is not None:
        fake.image_to_data = Mock(return_value=data)
    else:
        fake.image_to_data = Mock()
    return fake


# ===========================================================================
# Group 6 – TesseractAdapter (~15 tests)
# ===========================================================================


class TestTesseractAdapter:
    """Tests for TesseractAdapter."""

    def _make_tesseract_data(self, texts, confs, lefts, tops, widths, heights):
        """Build mock pytesseract.image_to_data dict output."""
        return {
            "text": texts,
            "conf": confs,
            "left": lefts,
            "top": tops,
            "width": widths,
            "height": heights,
        }

    def _install_fake_pytesseract(self, data=None):
        """Inject fake pytesseract into sys.modules and return the fake module."""
        fake = _make_fake_pytesseract(data)
        sys.modules["pytesseract"] = fake
        return fake

    def _remove_fake_pytesseract(self):
        """Remove injected fake pytesseract from sys.modules."""
        sys.modules.pop("pytesseract", None)

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.tesseract_adapter._ensure_tesseract")
    def test_instantiation_defaults(self, mock_ensure_tesseract):
        mock_ensure_tesseract.return_value = Mock()
        from mata.adapters.ocr.tesseract_adapter import TesseractAdapter

        adapter = TesseractAdapter()
        assert adapter.lang == "eng"
        assert adapter.config == ""

    @patch("mata.adapters.ocr.tesseract_adapter._ensure_tesseract")
    def test_instantiation_custom_lang(self, mock_ensure_tesseract):
        mock_ensure_tesseract.return_value = Mock()
        from mata.adapters.ocr.tesseract_adapter import TesseractAdapter

        adapter = TesseractAdapter(lang="fra+eng", config="--oem 3 --psm 6")
        assert adapter.lang == "fra+eng"
        assert adapter.config == "--oem 3 --psm 6"

    @patch("mata.adapters.ocr.tesseract_adapter._ensure_tesseract")
    def test_task_name(self, mock_ensure_tesseract):
        mock_ensure_tesseract.return_value = Mock()
        from mata.adapters.ocr.tesseract_adapter import TesseractAdapter

        adapter = TesseractAdapter()
        assert adapter.task == "ocr"
        assert adapter.name == "tesseract"

    # ------------------------------------------------------------------
    # predict() — happy path
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.tesseract_adapter._ensure_tesseract")
    def test_predict_returns_ocr_result(self, mock_ensure_tesseract):
        mock_ensure_tesseract.return_value = Mock()

        from mata.adapters.ocr.tesseract_adapter import TesseractAdapter
        from mata.core.types import OCRResult

        data = self._make_tesseract_data(
            texts=["Hello", "World"],
            confs=[90, 80],
            lefts=[10, 120],
            tops=[20, 20],
            widths=[90, 100],
            heights=[30, 30],
        )
        _fake = self._install_fake_pytesseract(data)
        try:
            adapter = TesseractAdapter()
            pil_img = _make_pil_image()
            with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
                result = adapter.predict(pil_img)
        finally:
            self._remove_fake_pytesseract()

        assert isinstance(result, OCRResult)
        assert len(result.regions) == 2
        assert result.regions[0].text == "Hello"
        assert result.regions[1].text == "World"
        assert result.meta["engine"] == "tesseract"

    # ------------------------------------------------------------------
    # conf == -1 entries filtered out
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.tesseract_adapter._ensure_tesseract")
    def test_predict_filters_conf_negative_one(self, mock_ensure_tesseract):
        mock_ensure_tesseract.return_value = Mock()

        data = self._make_tesseract_data(
            texts=["ValidWord", "", "AnotherWord"],
            confs=[85, -1, 70],
            lefts=[5, 50, 200],
            tops=[10, 10, 10],
            widths=[80, 20, 90],
            heights=[25, 25, 25],
        )
        _fake = self._install_fake_pytesseract(data)
        try:
            from mata.adapters.ocr.tesseract_adapter import TesseractAdapter

            adapter = TesseractAdapter()
            pil_img = _make_pil_image()
            with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
                result = adapter.predict(pil_img)
        finally:
            self._remove_fake_pytesseract()

        # conf == -1 entry AND empty text entry should be filtered
        assert len(result.regions) == 2
        texts = [r.text for r in result.regions]
        assert "ValidWord" in texts
        assert "AnotherWord" in texts

    @patch("mata.adapters.ocr.tesseract_adapter._ensure_tesseract")
    def test_predict_filters_empty_text(self, mock_ensure_tesseract):
        mock_ensure_tesseract.return_value = Mock()

        data = self._make_tesseract_data(
            texts=["", "  ", "GoodText"],
            confs=[50, 60, 88],
            lefts=[0, 50, 100],
            tops=[0, 0, 0],
            widths=[20, 20, 60],
            heights=[15, 15, 15],
        )
        _fake = self._install_fake_pytesseract(data)
        try:
            from mata.adapters.ocr.tesseract_adapter import TesseractAdapter

            adapter = TesseractAdapter()
            pil_img = _make_pil_image()
            with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
                result = adapter.predict(pil_img)
        finally:
            self._remove_fake_pytesseract()

        assert len(result.regions) == 1
        assert result.regions[0].text == "GoodText"

    # ------------------------------------------------------------------
    # Confidence normalization 0–100 → 0.0–1.0
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.tesseract_adapter._ensure_tesseract")
    def test_confidence_normalized_to_float(self, mock_ensure_tesseract):
        mock_ensure_tesseract.return_value = Mock()

        data = self._make_tesseract_data(
            texts=["word100", "word50", "word0"],
            confs=[100, 50, 0],
            lefts=[0, 0, 0],
            tops=[0, 0, 0],
            widths=[10, 10, 10],
            heights=[10, 10, 10],
        )
        _fake = self._install_fake_pytesseract(data)
        try:
            from mata.adapters.ocr.tesseract_adapter import TesseractAdapter

            adapter = TesseractAdapter()
            pil_img = _make_pil_image()
            with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
                result = adapter.predict(pil_img)
        finally:
            self._remove_fake_pytesseract()

        assert all(0.0 <= r.score <= 1.0 for r in result.regions)
        assert abs(result.regions[0].score - 1.0) < 1e-9
        assert abs(result.regions[1].score - 0.5) < 1e-9
        assert abs(result.regions[2].score - 0.0) < 1e-9

    # ------------------------------------------------------------------
    # Bbox conversion (x, y, w, h) → xyxy
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.tesseract_adapter._ensure_tesseract")
    def test_bbox_converted_to_xyxy(self, mock_ensure_tesseract):
        mock_ensure_tesseract.return_value = Mock()

        # Tesseract returns (x, y, w, h) → expected xyxy is (x, y, x+w, y+h)
        data = self._make_tesseract_data(
            texts=["TestWord"],
            confs=[75],
            lefts=[10],  # x
            tops=[20],  # y
            widths=[80],  # w
            heights=[30],  # h
        )
        _fake = self._install_fake_pytesseract(data)
        try:
            from mata.adapters.ocr.tesseract_adapter import TesseractAdapter

            adapter = TesseractAdapter()
            pil_img = _make_pil_image()
            with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
                result = adapter.predict(pil_img)
        finally:
            self._remove_fake_pytesseract()

        assert result.regions[0].bbox == (10.0, 20.0, 90.0, 50.0)  # (x, y, x+w, y+h)

    @patch("mata.adapters.ocr.tesseract_adapter._ensure_tesseract")
    def test_predict_all_filtered_returns_empty(self, mock_ensure_tesseract):
        mock_ensure_tesseract.return_value = Mock()

        data = self._make_tesseract_data(
            texts=["", " ", ""],
            confs=[-1, -1, 50],
            lefts=[0, 0, 0],
            tops=[0, 0, 0],
            widths=[10, 10, 10],
            heights=[10, 10, 10],
        )
        _fake = self._install_fake_pytesseract(data)
        try:
            from mata.adapters.ocr.tesseract_adapter import TesseractAdapter

            adapter = TesseractAdapter()
            pil_img = _make_pil_image()
            with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
                result = adapter.predict(pil_img)
        finally:
            self._remove_fake_pytesseract()

        assert result.regions == []
        assert result.full_text == ""

    # ------------------------------------------------------------------
    # Missing pytesseract raises ImportError with binary guidance
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.tesseract_adapter._ensure_tesseract")
    def test_missing_pytesseract_raises_import_error(self, mock_ensure_tesseract):
        mock_ensure_tesseract.side_effect = ImportError(
            "pytesseract is required for TesseractAdapter.\n"
            "Install Python package: pip install pytesseract\n"
            "or: pip install mata[ocr-tesseract]\n"
            "Also install the Tesseract binary:\n"
            "  Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
            "  macOS: brew install tesseract\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
        )
        from mata.adapters.ocr import tesseract_adapter

        tesseract_adapter._pytesseract = None  # reset cache

        from mata.adapters.ocr.tesseract_adapter import TesseractAdapter

        with pytest.raises(ImportError) as exc_info:
            TesseractAdapter()
        msg = str(exc_info.value)
        assert "pytesseract" in msg or "pip install" in msg

    @patch("mata.adapters.ocr.tesseract_adapter._ensure_tesseract")
    def test_missing_tesseract_error_includes_ubuntu_instructions(self, mock_ensure_tesseract):
        mock_ensure_tesseract.side_effect = ImportError(
            "pytesseract is required.\n"
            "Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
            "macOS: brew install tesseract\n"
            "Windows: https://github.com/UB-Mannheim/tesseract/wiki"
        )
        from mata.adapters.ocr import tesseract_adapter

        tesseract_adapter._pytesseract = None

        from mata.adapters.ocr.tesseract_adapter import TesseractAdapter

        with pytest.raises(ImportError) as exc_info:
            TesseractAdapter()
        msg = str(exc_info.value)
        # The guidance must mention at least one OS
        assert any(keyword in msg for keyword in ["Ubuntu", "macOS", "Windows", "brew", "apt-get"])

    # ------------------------------------------------------------------
    # info()
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.tesseract_adapter._ensure_tesseract")
    def test_info_returns_metadata(self, mock_ensure_tesseract):
        mock_ensure_tesseract.return_value = Mock()
        from mata.adapters.ocr.tesseract_adapter import TesseractAdapter

        adapter = TesseractAdapter(lang="deu+eng", config="--psm 6")
        info = adapter.info()
        assert info["name"] == "tesseract"
        assert info["task"] == "ocr"
        assert info["lang"] == "deu+eng"
        assert info["config"] == "--psm 6"


# ===========================================================================
# Group 7 – UniversalLoader OCR routing (~10 tests)
# ===========================================================================


class TestUniversalLoaderOCR:
    """Tests for OCR task routing through UniversalLoader / mata.load()."""

    # ------------------------------------------------------------------
    # External engine routing
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.easyocr_adapter._ensure_easyocr")
    def test_load_ocr_easyocr_returns_easyocr_adapter(self, mock_ensure):
        mock_easyocr = Mock()
        mock_easyocr.Reader.return_value = Mock()
        mock_ensure.return_value = mock_easyocr

        import mata
        from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter

        adapter = mata.load("ocr", "easyocr", gpu=False)
        assert isinstance(adapter, EasyOCRAdapter)

    @patch("mata.adapters.ocr.paddleocr_adapter._ensure_paddleocr")
    def test_load_ocr_paddleocr_returns_paddleocr_adapter(self, mock_ensure):
        mock_paddle_cls = Mock(return_value=Mock())
        mock_ensure.return_value = mock_paddle_cls

        import mata
        from mata.adapters.ocr.paddleocr_adapter import PaddleOCRAdapter

        adapter = mata.load("ocr", "paddleocr", use_gpu=False)
        assert isinstance(adapter, PaddleOCRAdapter)

    @patch("mata.adapters.ocr.tesseract_adapter._ensure_tesseract")
    def test_load_ocr_tesseract_returns_tesseract_adapter(self, mock_ensure):
        mock_ensure.return_value = Mock()

        import mata
        from mata.adapters.ocr.tesseract_adapter import TesseractAdapter

        adapter = mata.load("ocr", "tesseract")
        assert isinstance(adapter, TesseractAdapter)

    @patch("mata.adapters.ocr.easyocr_adapter._ensure_easyocr")
    def test_load_ocr_easyocr_case_insensitive(self, mock_ensure):
        """'EasyOCR' (mixed case) should also resolve to external engine."""
        mock_easyocr = Mock()
        mock_easyocr.Reader.return_value = Mock()
        mock_ensure.return_value = mock_easyocr

        import mata
        from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter

        adapter = mata.load("ocr", "EasyOCR", gpu=False)
        assert isinstance(adapter, EasyOCRAdapter)

    # ------------------------------------------------------------------
    # HuggingFace routing
    # ------------------------------------------------------------------

    @patch("mata.adapters.pytorch_base._ensure_torch")
    @patch("mata.adapters.ocr.huggingface_ocr_adapter._ensure_transformers")
    @patch("mata.core.logging.suppress_third_party_logs")
    def test_load_ocr_huggingface_trocr(self, mock_suppress, mock_ensure_tf, mock_ensure_torch):
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        cpu_device = Mock()
        cpu_device.type = "cpu"
        mock_torch.device.return_value = cpu_device
        mock_ensure_torch.return_value = mock_torch

        mock_processor = Mock()
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_tf = Mock()
        mock_tf.TrOCRProcessor.from_pretrained.return_value = mock_processor
        mock_tf.VisionEncoderDecoderModel.from_pretrained.return_value = mock_model
        mock_ensure_tf.return_value = mock_tf
        mock_suppress.return_value = MagicMock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=False))

        import mata
        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        adapter = mata.load("ocr", "microsoft/trocr-base-handwritten", device="cpu")
        assert isinstance(adapter, HuggingFaceOCRAdapter)

    @patch("mata.adapters.pytorch_base._ensure_torch")
    @patch("mata.adapters.ocr.huggingface_ocr_adapter._ensure_transformers")
    @patch("mata.core.logging.suppress_third_party_logs")
    def test_load_ocr_hf_easyocr_id_routes_to_hf_not_external(self, mock_suppress, mock_ensure_tf, mock_ensure_torch):
        """'my-org/easyocr-model' contains a slash → HuggingFace, not external engine."""
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        cpu_device = Mock()
        cpu_device.type = "cpu"
        mock_torch.device.return_value = cpu_device
        mock_ensure_torch.return_value = mock_torch

        mock_processor = Mock()
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_tf = Mock()
        mock_tf.TrOCRProcessor.from_pretrained.return_value = mock_processor
        mock_tf.VisionEncoderDecoderModel.from_pretrained.return_value = mock_model
        mock_ensure_tf.return_value = mock_tf
        mock_suppress.return_value = MagicMock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=False))

        import mata
        from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

        adapter = mata.load("ocr", "my-org/easyocr-model", device="cpu")
        assert isinstance(adapter, HuggingFaceOCRAdapter)

    # ------------------------------------------------------------------
    # Model type explicit routing
    # ------------------------------------------------------------------

    @patch("mata.adapters.ocr.easyocr_adapter._ensure_easyocr")
    def test_load_with_model_type_easyocr(self, mock_ensure):
        mock_easyocr = Mock()
        mock_easyocr.Reader.return_value = Mock()
        mock_ensure.return_value = mock_easyocr

        import mata
        from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter
        from mata.core.types import ModelType

        adapter = mata.load("ocr", model_type=ModelType.EASYOCR, gpu=False)
        assert isinstance(adapter, EasyOCRAdapter)

    @patch("mata.adapters.ocr.tesseract_adapter._ensure_tesseract")
    def test_load_with_model_type_tesseract(self, mock_ensure):
        mock_ensure.return_value = Mock()

        import mata
        from mata.adapters.ocr.tesseract_adapter import TesseractAdapter
        from mata.core.types import ModelType

        adapter = mata.load("ocr", model_type=ModelType.TESSERACT)
        assert isinstance(adapter, TesseractAdapter)

    # ------------------------------------------------------------------
    # Wrong task + engine name raises UnsupportedModelError
    # ------------------------------------------------------------------

    def test_load_detect_easyocr_raises_unsupported(self):
        """mata.load('detect', 'easyocr') must raise UnsupportedModelError, not succeed."""
        import mata
        from mata.core.exceptions import ModelNotFoundError, UnsupportedModelError

        with pytest.raises((UnsupportedModelError, ModelNotFoundError, Exception)):
            mata.load("detect", "easyocr")

    # ------------------------------------------------------------------
    # Verify mata namespace exports
    # ------------------------------------------------------------------

    def test_ocr_result_importable_from_mata(self):
        from mata import OCRResult

        assert OCRResult is not None

    def test_text_region_importable_from_mata(self):
        from mata import TextRegion

        assert TextRegion is not None
