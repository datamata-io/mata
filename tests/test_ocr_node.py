"""Tests for the OCR graph node, OCRWrapper, graph composition, and VLM tool integration.

Test groups:

- TestOCRWrapper   (~8 tests)  — OCRWrapper & wrap_ocr factory
- TestOCRNode      (~20 tests) — OCR node execution, ROIs, metrics, error handling
- TestOCRGraphComposition (~12 tests) — Sequential / pipeline / DSL / parallel graphs
- TestOCRVLMTool   (~10 tests) — TASK_SCHEMA_DEFAULTS, _format_provider_result, tool dispatch
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

import mata
from mata.adapters.wrappers.ocr_wrapper import OCRWrapper, wrap_ocr
from mata.core.artifacts.image import Image
from mata.core.artifacts.ocr_text import OCRText, TextBlock
from mata.core.artifacts.rois import ROIs
from mata.core.graph.context import ExecutionContext
from mata.core.tool_registry import ToolRegistry
from mata.core.tool_schema import TASK_SCHEMA_DEFAULTS, ToolCall
from mata.core.types import OCRResult, TextRegion
from mata.nodes.ocr import OCR

# ---------------------------------------------------------------------------
# Helpers / shared factories
# ---------------------------------------------------------------------------


def _make_ocr_result(*texts: str, score: float = 0.95) -> OCRResult:
    """Build a simple OCRResult from varargs text strings."""
    regions = [TextRegion(text=t, score=score) for t in texts]
    return OCRResult(regions=regions)


def _make_ocr_text(*texts: str, instance_ids: tuple[str, ...] = ()) -> OCRText:
    """Build an OCRText artifact from varargs text strings."""
    blocks = tuple(TextBlock(text=t, confidence=0.95) for t in texts)
    full_text = "\n".join(t for t in texts)
    return OCRText(text_blocks=blocks, full_text=full_text, instance_ids=instance_ids)


def _make_pil_image(width: int = 64, height: int = 32) -> PILImage.Image:
    return PILImage.new("RGB", (width, height), color=(200, 200, 200))


def _make_image_artifact(source_path: str | None = None) -> Image:
    """Return an Image artifact backed by a PIL image (no real file)."""
    pil = _make_pil_image()
    img = Image.from_pil(pil)
    if source_path:
        # Monkey-patch source_path for testing _convert_image path preference
        object.__setattr__(img, "source_path", source_path)
    return img


def _make_ctx(providers: dict[str, dict[str, Any]] | None = None) -> ExecutionContext:
    return ExecutionContext(providers=providers or {}, device="cpu")


def _make_mock_recognizer(return_value: OCRText | OCRResult | None = None) -> MagicMock:
    """Build a mock recognizer whose recognize() returns *return_value*."""
    mock = MagicMock()
    if return_value is None:
        return_value = _make_ocr_text("hello")
    mock.recognize.return_value = return_value
    return mock


def _make_rois(n: int = 3) -> ROIs:
    """Build a ROIs artifact with *n* tiny PIL crops."""
    crops = [_make_pil_image() for _ in range(n)]
    boxes = [(i * 10, 0, i * 10 + 10, 10) for i in range(n)]
    ids = [f"roi_{i:04d}" for i in range(n)]
    return ROIs(roi_images=crops, instance_ids=ids, source_boxes=boxes)


# ===========================================================================
# TestOCRWrapper
# ===========================================================================


class TestOCRWrapper:
    """Tests for the OCRWrapper class and wrap_ocr() factory."""

    def test_instantiation_with_valid_adapter(self):
        """OCRWrapper wraps any adapter with predict()."""
        adapter = MagicMock(spec=["predict"])
        wrapper = OCRWrapper(adapter)
        assert wrapper.adapter is adapter

    def test_rejects_adapter_without_predict(self):
        """OCRWrapper raises TypeError if predict() is missing."""
        bad_adapter = object()
        with pytest.raises(TypeError, match="predict"):
            OCRWrapper(bad_adapter)

    def test_recognize_calls_adapter_predict_and_returns_ocr_text(self):
        """recognize() calls adapter.predict() and wraps result as OCRText."""
        raw = _make_ocr_result("hello world")
        adapter = MagicMock()
        adapter.predict.return_value = raw
        wrapper = OCRWrapper(adapter)
        image = _make_image_artifact()

        result = wrapper.recognize(image)

        adapter.predict.assert_called_once()
        assert isinstance(result, OCRText)
        assert result.text_blocks[0].text == "hello world"

    def test_convert_image_prefers_source_path(self):
        """_convert_image() returns the source_path when present."""
        adapter = MagicMock()
        adapter.predict.return_value = _make_ocr_result("x")
        wrapper = OCRWrapper(adapter)
        image = _make_image_artifact(source_path="/tmp/doc.png")

        wrapper.recognize(image)

        # The first positional arg to predict should be the path string
        call_arg = adapter.predict.call_args[0][0]
        assert call_arg == "/tmp/doc.png"

    def test_convert_image_falls_back_to_pil(self):
        """_convert_image() returns PIL image when no source_path."""
        adapter = MagicMock()
        adapter.predict.return_value = _make_ocr_result("x")
        wrapper = OCRWrapper(adapter)
        image = _make_image_artifact()  # no source_path

        wrapper.recognize(image)

        call_arg = adapter.predict.call_args[0][0]
        assert isinstance(call_arg, PILImage.Image)

    def test_predict_with_image_artifact_delegates_to_recognize(self):
        """predict(Image) delegates to recognize() and returns OCRText."""
        raw = _make_ocr_result("text")
        adapter = MagicMock()
        adapter.predict.return_value = raw
        wrapper = OCRWrapper(adapter)
        image = _make_image_artifact()

        result = wrapper.predict(image)

        assert isinstance(result, OCRText)

    def test_predict_with_raw_input_delegates_to_adapter(self):
        """predict(PIL) calls adapter.predict() directly (no conversion)."""
        raw = _make_ocr_result("text")
        adapter = MagicMock()
        adapter.predict.return_value = raw
        wrapper = OCRWrapper(adapter)
        pil = _make_pil_image()

        result = wrapper.predict(pil)

        adapter.predict.assert_called_once_with(pil)
        # Raw OCRResult is returned (not wrapped)
        assert result is raw

    def test_wrap_ocr_factory_function(self):
        """wrap_ocr() returns an OCRWrapper instance."""
        adapter = MagicMock(spec=["predict"])
        result = wrap_ocr(adapter)
        assert isinstance(result, OCRWrapper)
        assert result.adapter is adapter


# ===========================================================================
# TestOCRNode
# ===========================================================================


class TestOCRNode:
    """Tests for the OCR graph node."""

    # ------------------------------------------------------------------ #
    # Basic image input                                                    #
    # ------------------------------------------------------------------ #

    def test_image_input_calls_recognize_once(self):
        """OCR with Image input calls recognizer.recognize() exactly once."""
        ocr_result = _make_ocr_text("invoice", "total: 100")
        recognizer = _make_mock_recognizer(ocr_result)
        ctx = _make_ctx({"ocr": {"tesseract": recognizer}})
        node = OCR(using="tesseract")
        image = _make_image_artifact()

        result = node.run(ctx, image=image)

        recognizer.recognize.assert_called_once()
        assert "ocr" in result
        assert isinstance(result["ocr"], OCRText)

    def test_image_input_returns_correct_text_blocks(self):
        """Output OCRText contains text blocks from the OCR provider."""
        ocr_result = _make_ocr_text("hello", "world")
        recognizer = _make_mock_recognizer(ocr_result)
        ctx = _make_ctx({"ocr": {"eng": recognizer}})
        node = OCR(using="eng")

        result = node.run(ctx, image=_make_image_artifact())["ocr"]

        assert len(result.text_blocks) == 2
        assert result.text_blocks[0].text == "hello"
        assert result.text_blocks[1].text == "world"

    # ------------------------------------------------------------------ #
    # ROIs input                                                           #
    # ------------------------------------------------------------------ #

    def test_rois_input_calls_recognize_per_crop(self):
        """OCR with ROIs calls recognize() once for each crop."""
        recognizer = _make_mock_recognizer(_make_ocr_text("sign"))
        ctx = _make_ctx({"ocr": {"eng": recognizer}})
        node = OCR(using="eng")
        rois = _make_rois(n=3)

        node.run(ctx, rois=rois)

        assert recognizer.recognize.call_count == 3

    def test_rois_instance_ids_preserved_in_output(self):
        """instance_ids in output OCRText align with the source ROI ids."""
        recognizer = _make_mock_recognizer(_make_ocr_text("text"))
        ctx = _make_ctx({"ocr": {"eng": recognizer}})
        node = OCR(using="eng")
        rois = _make_rois(n=3)

        result = node.run(ctx, rois=rois)["ocr"]

        # One text block per ROI, each tagged with its ROI id
        assert len(result.instance_ids) == 3
        for rid in result.instance_ids:
            assert rid in rois.instance_ids

    def test_empty_rois_returns_empty_ocr_text(self):
        """OCR on zero ROIs returns an empty OCRText without error."""
        recognizer = _make_mock_recognizer()
        ctx = _make_ctx({"ocr": {"eng": recognizer}})
        node = OCR(using="eng")
        empty_rois = ROIs(roi_images=[], instance_ids=[], source_boxes=[])

        result = node.run(ctx, rois=empty_rois)["ocr"]

        assert isinstance(result, OCRText)
        assert len(result.text_blocks) == 0
        recognizer.recognize.assert_not_called()

    def test_rois_multiple_blocks_per_crop(self):
        """Multiple text blocks per crop are all collected in the output."""
        # Two blocks per crop
        multi_block_ocr = _make_ocr_text("line1", "line2")
        recognizer = _make_mock_recognizer(multi_block_ocr)
        ctx = _make_ctx({"ocr": {"eng": recognizer}})
        node = OCR(using="eng")

        result = node.run(ctx, rois=_make_rois(n=2))["ocr"]

        # 2 crops × 2 blocks = 4 total
        assert len(result.text_blocks) == 4

    # ------------------------------------------------------------------ #
    # Output key / dynamic naming                                          #
    # ------------------------------------------------------------------ #

    def test_custom_output_name(self):
        """out='my_text' stores the result under 'my_text'."""
        recognizer = _make_mock_recognizer()
        ctx = _make_ctx({"ocr": {"eng": recognizer}})
        node = OCR(using="eng", out="my_text")

        result = node.run(ctx, image=_make_image_artifact())

        assert "my_text" in result
        assert "ocr" not in result

    def test_default_output_name_is_ocr(self):
        """Default output key is 'ocr'."""
        recognizer = _make_mock_recognizer()
        ctx = _make_ctx({"ocr": {"eng": recognizer}})
        node = OCR(using="eng")

        result = node.run(ctx, image=_make_image_artifact())

        assert "ocr" in result

    # ------------------------------------------------------------------ #
    # src override                                                         #
    # ------------------------------------------------------------------ #

    def test_src_override_resolves_custom_input_key(self):
        """src='my_image' resolves from inputs dict by that key."""
        recognizer = _make_mock_recognizer(_make_ocr_text("custom"))
        ctx = _make_ctx({"ocr": {"eng": recognizer}})
        node = OCR(using="eng", src="my_image")
        image = _make_image_artifact()

        result = node.run(ctx, my_image=image)

        assert "ocr" in result
        recognizer.recognize.assert_called_once()

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def test_record_metric_num_text_blocks_called(self):
        """ctx.record_metric() is called with num_text_blocks."""
        recognizer = _make_mock_recognizer(_make_ocr_text("a", "b", "c"))
        ctx = _make_ctx({"ocr": {"eng": recognizer}})
        ctx.record_metric = MagicMock()
        node = OCR(using="eng")

        node.run(ctx, image=_make_image_artifact())

        calls = [str(c) for c in ctx.record_metric.call_args_list]
        assert any("num_text_blocks" in c for c in calls)

    def test_record_metric_latency_called(self):
        """ctx.record_metric() is called with latency_ms."""
        recognizer = _make_mock_recognizer()
        ctx = _make_ctx({"ocr": {"eng": recognizer}})
        ctx.record_metric = MagicMock()
        node = OCR(using="eng")

        node.run(ctx, image=_make_image_artifact())

        calls = [str(c) for c in ctx.record_metric.call_args_list]
        assert any("latency_ms" in c for c in calls)

    # ------------------------------------------------------------------ #
    # Error handling                                                       #
    # ------------------------------------------------------------------ #

    def test_invalid_input_type_raises_value_error(self):
        """Non-Image / non-ROIs input raises ValueError."""
        recognizer = _make_mock_recognizer()
        ctx = _make_ctx({"ocr": {"eng": recognizer}})
        node = OCR(using="eng")

        with pytest.raises(ValueError, match="Image or ROIs"):
            node.run(ctx, image="not_an_artifact")

    def test_missing_provider_raises_key_error(self):
        """Using a provider that is not in the context raises KeyError."""
        ctx = _make_ctx({})
        node = OCR(using="missing_provider")

        with pytest.raises(KeyError):
            node.run(ctx, image=_make_image_artifact())

    def test_no_inputs_raises_value_error(self):
        """Calling run() with no recognized input key raises ValueError."""
        recognizer = _make_mock_recognizer()
        ctx = _make_ctx({"ocr": {"eng": recognizer}})
        node = OCR(using="eng")

        with pytest.raises(ValueError, match="no inputs"):
            node.run(ctx)

    # ------------------------------------------------------------------ #
    # Node name                                                            #
    # ------------------------------------------------------------------ #

    def test_node_name_defaults_to_class_name(self):
        """Node name defaults to 'OCR' (the class name)."""
        node = OCR(using="eng")
        assert node.name == "OCR"

    def test_node_name_can_be_overridden(self):
        """Custom name is stored on the node."""
        node = OCR(using="eng", name="sign_reader")
        assert node.name == "sign_reader"

    # ------------------------------------------------------------------ #
    # Recognizer returns OCRText vs OCRResult                              #
    # ------------------------------------------------------------------ #

    def test_recognizer_returning_ocr_result_is_wrapped(self):
        """If recognize() returns a raw OCRResult it is converted to OCRText."""
        raw = _make_ocr_result("raw text")
        recognizer = MagicMock()
        recognizer.recognize.return_value = raw
        ctx = _make_ctx({"ocr": {"eng": recognizer}})
        node = OCR(using="eng")

        result = node.run(ctx, image=_make_image_artifact())["ocr"]

        assert isinstance(result, OCRText)
        assert result.text_blocks[0].text == "raw text"

    def test_repr_includes_provider_and_output(self):
        """__repr__ includes using and out."""
        node = OCR(using="tesseract", out="text")
        r = repr(node)
        assert "tesseract" in r
        assert "text" in r


# ===========================================================================
# TestOCRGraphComposition
# ===========================================================================


class TestOCRGraphComposition:
    """Graph composition tests with mock adapters.

    These tests validate end-to-end data flow through mata.infer() using fully
    mocked providers — no real model weights are needed.
    """

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @pytest.fixture(autouse=True)
    def _ensure_pil(self):
        """Guard: skip composition tests if PIL is not available."""
        pytest.importorskip("PIL")

    def _make_detect_adapter(self) -> MagicMock:
        """Mock adapter that returns two detections."""
        from mata.core.artifacts.detections import Detections
        from mata.core.types import Instance, VisionResult

        instances = [
            Instance(bbox=(0, 0, 64, 32), score=0.9, label=0, label_name="sign"),
            Instance(bbox=(64, 0, 128, 32), score=0.85, label=0, label_name="sign"),
        ]
        vr = VisionResult(instances=instances, meta={})
        adapter = MagicMock()
        adapter.predict.return_value = Detections.from_vision_result(vr)
        return adapter

    def _make_ocr_adapter(self, text: str = "STOP") -> MagicMock:
        """Mock OCR adapter whose recognize() returns a fixed OCRText."""
        adapter = MagicMock()
        adapter.predict.return_value = _make_ocr_result(text)
        # Needs recognize() for OCRWrapper — returned by the wrapper layer
        recognizer = OCRWrapper(adapter)
        return recognizer

    def _make_pil_image(self) -> PILImage.Image:
        return PILImage.new("RGB", (128, 32), color=(100, 100, 100))

    # ------------------------------------------------------------------ #
    # Sequential: Image -> OCR                                            #
    # ------------------------------------------------------------------ #

    def test_sequential_image_to_ocr(self):
        """Image -> OCR graph: output dict contains OCRText under 'ocr'."""
        recognizer = _make_mock_recognizer(_make_ocr_text("hello graph"))

        result = mata.infer(
            image=self._make_pil_image(),
            graph=[OCR(using="ocr_eng")],
            providers={"ocr": {"ocr_eng": recognizer}},
        )

        assert result.has_channel("ocr")
        assert isinstance(result.channels["ocr"], OCRText)
        assert result.channels["ocr"].text_blocks[0].text == "hello graph"

    # ------------------------------------------------------------------ #
    # Pipeline: Detect -> Filter -> ExtractROIs -> OCR                    #
    # ------------------------------------------------------------------ #

    def test_pipeline_detect_filter_rois_ocr(self):
        """Full detect->filter->ROI->OCR pipeline passes instance_ids correctly."""
        from mata.nodes.detect import Detect
        from mata.nodes.filter import Filter
        from mata.nodes.roi import ExtractROIs

        detect_adapter = self._make_detect_adapter()
        ocr_recognizer = _make_mock_recognizer(_make_ocr_text("STOP"))

        result = mata.infer(
            image=self._make_pil_image(),
            graph=[
                Detect(using="detector", out="dets"),
                Filter(src="dets", label_in=["sign"], out="sign_dets"),
                ExtractROIs(src_dets="sign_dets", out="rois"),
                OCR(using="ocr_eng", src="rois", out="ocr_result"),
            ],
            providers={
                "detect": {"detector": detect_adapter},
                "ocr": {"ocr_eng": ocr_recognizer},
            },
        )

        assert result.has_channel("ocr_result")
        ocr_out = result.channels["ocr_result"]
        assert isinstance(ocr_out, OCRText)

    def test_pipeline_instance_ids_in_ocr_match_roi_ids(self):
        """instance_ids in OCRText align with the source ROI instance_ids."""
        from mata.nodes.detect import Detect
        from mata.nodes.filter import Filter
        from mata.nodes.roi import ExtractROIs

        detect_adapter = self._make_detect_adapter()
        ocr_recognizer = _make_mock_recognizer(_make_ocr_text("text"))

        result = mata.infer(
            image=self._make_pil_image(),
            graph=[
                Detect(using="detector", out="dets"),
                Filter(src="dets", out="filtered"),
                ExtractROIs(src_dets="filtered", out="rois"),
                OCR(using="ocr_eng", src="rois", out="ocr_out"),
            ],
            providers={
                "detect": {"detector": detect_adapter},
                "ocr": {"ocr_eng": ocr_recognizer},
            },
        )

        ocr_result = result.channels["ocr_out"]
        # Each text block must carry a non-empty instance_id that traces back
        # to a source detection (ExtractROIs propagates detection instance_ids)
        assert len(ocr_result.instance_ids) > 0
        for iid in ocr_result.instance_ids:
            assert isinstance(iid, str) and len(iid) > 0

    # ------------------------------------------------------------------ #
    # Pipeline: Detect -> ExtractROIs -> OCR -> Fuse                      #
    # ------------------------------------------------------------------ #

    def test_pipeline_with_fuse_multiresult(self):
        """Detect -> ExtractROIs -> OCR -> Fuse yields MultiResult with both channels."""
        from mata.core.artifacts.result import MultiResult
        from mata.nodes.detect import Detect
        from mata.nodes.fuse import Fuse
        from mata.nodes.roi import ExtractROIs

        detect_adapter = self._make_detect_adapter()
        ocr_recognizer = _make_mock_recognizer(_make_ocr_text("YIELD"))

        result = mata.infer(
            image=self._make_pil_image(),
            graph=[
                Detect(using="detector", out="dets"),
                ExtractROIs(src_dets="dets", out="rois"),
                OCR(using="ocr_eng", src="rois", out="ocr_out"),
                Fuse(dets="dets", ocr="ocr_out", out="fused"),
            ],
            providers={
                "detect": {"detector": detect_adapter},
                "ocr": {"ocr_eng": ocr_recognizer},
            },
        )

        assert result.has_channel("fused")
        fused = result.channels["fused"]
        assert isinstance(fused, MultiResult)
        assert "dets" in fused.channels
        assert "ocr" in fused.channels

    # ------------------------------------------------------------------ #
    # DSL (NodePipe >>)                                                    #
    # ------------------------------------------------------------------ #

    def test_dsl_pipe_builds_without_error(self):
        """NodePipe (>>) DSL constructs an OCR pipeline without raising."""
        from mata.core.graph.dsl import NodePipe
        from mata.nodes.detect import Detect
        from mata.nodes.roi import ExtractROIs

        graph = (
            NodePipe(Detect(using="detector", out="dets"))
            >> ExtractROIs(src_dets="dets", out="rois")
            >> OCR(using="ocr_eng", src="rois", out="ocr_out")
        )
        assert graph is not None

    def test_dsl_pipe_result_is_executable(self):
        """DSL-built pipeline can be executed via mata.infer()."""
        from mata.core.graph.dsl import NodePipe
        from mata.nodes.detect import Detect
        from mata.nodes.roi import ExtractROIs

        detect_adapter = self._make_detect_adapter()
        ocr_recognizer = _make_mock_recognizer(_make_ocr_text("DSL"))

        graph = (
            NodePipe(Detect(using="detector", out="dets"))
            >> ExtractROIs(src_dets="dets", out="rois")
            >> OCR(using="ocr_eng", src="rois", out="ocr_out")
        ).build()

        result = mata.infer(
            image=self._make_pil_image(),
            graph=graph,
            providers={
                "detect": {"detector": detect_adapter},
                "ocr": {"ocr_eng": ocr_recognizer},
            },
        )

        assert result.has_channel("ocr_out")

    # ------------------------------------------------------------------ #
    # Parallel: parallel([Detect, OCR]) -> Fuse                           #
    # ------------------------------------------------------------------ #

    def test_parallel_detect_ocr_fuse(self):
        """Detect + OCR both operating on Image feeds Fuse: both channels in MultiResult."""
        from mata.core.artifacts.result import MultiResult
        from mata.nodes.detect import Detect
        from mata.nodes.fuse import Fuse

        detect_adapter = self._make_detect_adapter()
        ocr_recognizer = _make_mock_recognizer(_make_ocr_text("PARALLEL"))

        # Detect and OCR both consume Image independently, then Fuse aggregates
        result = mata.infer(
            image=self._make_pil_image(),
            graph=[
                Detect(using="detector", out="dets"),
                OCR(using="ocr_eng", out="ocr_out"),
                Fuse(dets="dets", ocr="ocr_out", out="fused"),
            ],
            providers={
                "detect": {"detector": detect_adapter},
                "ocr": {"ocr_eng": ocr_recognizer},
            },
        )

        assert result.has_channel("fused")
        assert isinstance(result.channels["fused"], MultiResult)

    def test_parallel_channels_accessible_in_multiresult(self):
        """Individual channels (dets, ocr) are accessible on the MultiResult."""
        from mata.core.artifacts.detections import Detections
        from mata.nodes.detect import Detect
        from mata.nodes.fuse import Fuse

        detect_adapter = self._make_detect_adapter()
        ocr_recognizer = _make_mock_recognizer(_make_ocr_text("ACCESS"))

        result = mata.infer(
            image=self._make_pil_image(),
            graph=[
                Detect(using="detector", out="dets"),
                OCR(using="ocr_eng", out="ocr_out"),
                Fuse(dets="dets", ocr="ocr_out", out="fused"),
            ],
            providers={
                "detect": {"detector": detect_adapter},
                "ocr": {"ocr_eng": ocr_recognizer},
            },
        )

        fused = result.channels["fused"]
        assert isinstance(fused.channels["dets"], Detections)
        assert isinstance(fused.channels["ocr"], OCRText)

    # ------------------------------------------------------------------ #
    # Provider isolation                                                   #
    # ------------------------------------------------------------------ #

    def test_standalone_ocr_node_without_detect(self):
        """Standalone OCR node (no detect stage) works directly on Image."""
        ocr_recognizer = _make_mock_recognizer(_make_ocr_text("standalone"))

        result = mata.infer(
            image=self._make_pil_image(),
            graph=[OCR(using="ocr_eng")],
            providers={"ocr": {"ocr_eng": ocr_recognizer}},
        )

        assert result.has_channel("ocr")
        assert result.channels["ocr"].full_text == "standalone"


# ===========================================================================
# TestOCRVLMTool
# ===========================================================================


class TestOCRVLMTool:
    """Tests for OCR integration with the VLM tool schema and registry."""

    # ------------------------------------------------------------------ #
    # Schema defaults                                                      #
    # ------------------------------------------------------------------ #

    def test_ocr_in_task_schema_defaults(self):
        """TASK_SCHEMA_DEFAULTS contains an 'ocr' entry."""
        assert "ocr" in TASK_SCHEMA_DEFAULTS

    def test_ocr_schema_task_field(self):
        """The schema's task field is 'ocr'."""
        schema = TASK_SCHEMA_DEFAULTS["ocr"]
        assert schema.task == "ocr"

    def test_ocr_schema_name_field(self):
        """The schema's name field is 'ocr'."""
        schema = TASK_SCHEMA_DEFAULTS["ocr"]
        assert schema.name == "ocr"

    def test_ocr_schema_has_description(self):
        """The OCR schema has a non-empty description."""
        schema = TASK_SCHEMA_DEFAULTS["ocr"]
        assert schema.description

    # ------------------------------------------------------------------ #
    # _format_provider_result                                              #
    # ------------------------------------------------------------------ #

    def _make_registry_with_ocr(self, provider_name: str, adapter: Any) -> ToolRegistry:
        """Build a ToolRegistry with an OCR provider registered."""
        ctx = ExecutionContext(
            providers={"ocr": {provider_name: OCRWrapper(adapter)}},
            device="cpu",
        )
        return ToolRegistry(ctx, [provider_name])

    def test_format_result_single_region(self):
        """1 text region → summary contains region count."""
        adapter = MagicMock()
        adapter.predict.return_value = _make_ocr_result("Hello")
        registry = self._make_registry_with_ocr("my_ocr", adapter)

        result = _make_ocr_result("Hello")
        tool_result = registry._format_provider_result("my_ocr", "ocr", result)

        assert tool_result.success is True
        assert "1" in tool_result.summary
        assert "Hello" in tool_result.summary

    def test_format_result_three_regions(self):
        """3 text regions → all shown in summary (no +N more)."""
        adapter = MagicMock()
        adapter.predict.return_value = _make_ocr_result("A")
        registry = self._make_registry_with_ocr("my_ocr", adapter)

        result = _make_ocr_result("A", "B", "C")
        tool_result = registry._format_provider_result("my_ocr", "ocr", result)

        assert tool_result.success is True
        assert "+0 more" not in tool_result.summary

    def test_format_result_five_regions_shows_plus_more(self):
        """5 text regions → summary includes '+2 more'."""
        adapter = MagicMock()
        adapter.predict.return_value = _make_ocr_result("A")
        registry = self._make_registry_with_ocr("my_ocr", adapter)

        result = _make_ocr_result("A", "B", "C", "D", "E")
        tool_result = registry._format_provider_result("my_ocr", "ocr", result)

        assert "+2 more" in tool_result.summary

    def test_format_result_empty_returns_no_text_found(self):
        """Empty OCR result → 'No text found' (not an error)."""
        adapter = MagicMock()
        adapter.predict.return_value = _make_ocr_result("x")  # just for registry init
        registry = self._make_registry_with_ocr("my_ocr", adapter)

        result = _make_ocr_result()  # zero regions
        tool_result = registry._format_provider_result("my_ocr", "ocr", result)

        assert tool_result.success is True
        assert "No text" in tool_result.summary

    def test_format_result_ocr_text_artifact(self):
        """_format_provider_result handles OCRText artifacts as well as OCRResult."""
        adapter = MagicMock()
        adapter.predict.return_value = _make_ocr_result("x")
        registry = self._make_registry_with_ocr("my_ocr", adapter)

        artifact = _make_ocr_text("from artifact")
        tool_result = registry._format_provider_result("my_ocr", "ocr", artifact)

        assert tool_result.success is True
        assert "from artifact" in tool_result.summary

    def test_format_result_artifact_stored_in_artifacts(self):
        """The raw result is stored in tool_result.artifacts under 'ocr_result'."""
        adapter = MagicMock()
        adapter.predict.return_value = _make_ocr_result("stored")
        registry = self._make_registry_with_ocr("my_ocr", adapter)

        result = _make_ocr_result("stored")
        tool_result = registry._format_provider_result("my_ocr", "ocr", result)

        assert "ocr_result" in tool_result.artifacts

    def test_ocr_tool_registered_as_provider_executes(self):
        """execute_tool() dispatches correctly to an OCR provider."""
        raw = _make_ocr_result("dispatched")
        adapter = MagicMock()
        adapter.predict.return_value = raw
        registry = self._make_registry_with_ocr("my_ocr", adapter)

        image = _make_image_artifact()
        tool_call = ToolCall(tool_name="my_ocr", arguments={}, raw_text="")
        tool_result = registry.execute_tool(tool_call, image)

        assert tool_result.success is True
        assert "dispatched" in tool_result.summary

    def test_ocr_tool_with_no_text_does_not_raise(self):
        """execute_tool() does not raise even when OCR returns zero regions."""
        empty_result = _make_ocr_result()
        adapter = MagicMock()
        adapter.predict.return_value = empty_result
        registry = self._make_registry_with_ocr("empty_ocr", adapter)

        image = _make_image_artifact()
        tool_call = ToolCall(tool_name="empty_ocr", arguments={}, raw_text="")
        tool_result = registry.execute_tool(tool_call, image)

        assert tool_result.success is True
        assert "No text" in tool_result.summary
