"""Tests for the BarcodeData artifact, Barcode graph node, and barcode tool schema.

Test groups:

- TestBarcodeDataArtifact  (~12 tests) — BarcodeData / BarcodeEntry construction,
                                          validation, serialisation, frozen semantics.
- TestBarcodeNode          (~20 tests) — Barcode node execution, ROIs pipeline,
                                          metrics, error handling, provider resolution.
- TestBarcodeToolSchema    (~8 tests)  — TASK_SCHEMA_DEFAULTS barcode entry,
                                          schema_for_task(), parameter checks.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

import mata
from mata.core.artifacts.barcode_data import BarcodeData, BarcodeEntry
from mata.core.artifacts.base import Artifact
from mata.core.artifacts.image import Image
from mata.core.artifacts.rois import ROIs
from mata.core.graph.context import ExecutionContext
from mata.core.tool_schema import TASK_SCHEMA_DEFAULTS, schema_for_task
from mata.core.types import BarcodeRegion, BarcodeResult
from mata.nodes.barcode import Barcode

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_pil_image(width: int = 64, height: int = 32) -> PILImage.Image:
    return PILImage.new("RGB", (width, height), color=(180, 180, 180))


def _make_image_artifact() -> Image:
    return Image.from_pil(_make_pil_image())


def _make_rois(n: int = 3) -> ROIs:
    crops = [_make_pil_image() for _ in range(n)]
    boxes = [(i * 10, 0, i * 10 + 10, 10) for i in range(n)]
    ids = [f"roi_{i:04d}" for i in range(n)]
    return ROIs(roi_images=crops, instance_ids=ids, source_boxes=boxes)


def _make_barcode_result(*payloads: str, btype: str = "QR_CODE") -> BarcodeResult:
    regions = [BarcodeRegion(data=p, type=btype) for p in payloads]
    return BarcodeResult(barcodes=regions)


def _make_mock_provider(return_value: Any = None) -> MagicMock:
    mock = MagicMock()
    if return_value is None:
        return_value = _make_barcode_result("https://example.com")
    mock.predict.return_value = return_value
    return mock


def _make_ctx(providers: dict[str, dict[str, Any]] | None = None) -> ExecutionContext:
    return ExecutionContext(providers=providers or {}, device="cpu")


# ===========================================================================
# TestBarcodeDataArtifact
# ===========================================================================


class TestBarcodeDataArtifact:
    """Unit tests for the BarcodeData artifact and BarcodeEntry dataclass."""

    def test_empty_artifact(self):
        """BarcodeData() with no arguments is valid and empty."""
        artifact = BarcodeData()
        assert artifact.is_empty
        assert artifact.num_barcodes == 0
        assert artifact.entries == ()
        assert artifact.instance_ids == ()

    def test_from_barcode_result(self):
        """from_barcode_result() converts BarcodeResult into a BarcodeData."""
        region = BarcodeRegion(
            data="https://example.com",
            type="QR_CODE",
            bbox=(10.0, 20.0, 110.0, 120.0),
            score=0.99,
        )
        result = BarcodeResult(barcodes=[region], meta={"engine": "pyzbar"})

        artifact = BarcodeData.from_barcode_result(result)

        assert artifact.num_barcodes == 1
        entry = artifact.entries[0]
        assert entry.data == "https://example.com"
        assert entry.type == "QR_CODE"
        assert entry.confidence == 0.99
        assert entry.bbox == (10.0, 20.0, 110.0, 120.0)
        assert artifact.meta["engine"] == "pyzbar"

    def test_validate_valid(self):
        """validate() passes silently for a correctly-constructed artifact."""
        entries = (BarcodeEntry(data="12345", type="EAN_13"),)
        artifact = BarcodeData(entries=entries, instance_ids=("roi_0001",))
        artifact.validate()  # must not raise

    def test_validate_invalid_entries(self):
        """validate() raises ValueError when entries is not a tuple."""
        # Bypass type-checking at construction (frozen only prevents mutation)
        bad = BarcodeData.__new__(BarcodeData)
        object.__setattr__(bad, "entries", [BarcodeEntry("x", "QR_CODE")])
        object.__setattr__(bad, "instance_ids", ())
        object.__setattr__(bad, "meta", {})

        with pytest.raises(ValueError, match="entries must be a tuple"):
            bad.validate()

    def test_validate_invalid_instance_ids(self):
        """validate() raises ValueError when instance_ids is not a tuple."""
        bad = BarcodeData.__new__(BarcodeData)
        object.__setattr__(bad, "entries", ())
        object.__setattr__(bad, "instance_ids", ["roi_0"])
        object.__setattr__(bad, "meta", {})

        with pytest.raises(ValueError, match="instance_ids must be a tuple"):
            bad.validate()

    def test_to_dict(self):
        """to_dict() produces a JSON-compatible dict with expected structure."""
        entry = BarcodeEntry(data="CODE128VAL", type="CODE_128", bbox=(0.0, 1.0, 50.0, 51.0))
        artifact = BarcodeData(
            entries=(entry,),
            instance_ids=("roi_0001",),
            meta={"engine": "zxing"},
        )

        d = artifact.to_dict()

        assert d["entries"][0]["data"] == "CODE128VAL"
        assert d["entries"][0]["type"] == "CODE_128"
        assert d["entries"][0]["bbox"] == [0.0, 1.0, 50.0, 51.0]
        assert d["instance_ids"] == ["roi_0001"]
        assert d["meta"]["engine"] == "zxing"

    def test_to_json(self):
        """to_json() returns a valid JSON string round-trippable via from_json()."""
        entry = BarcodeEntry(data="ROUND_TRIP", type="QR_CODE")
        artifact = BarcodeData(entries=(entry,))

        json_str = artifact.to_json()
        restored = BarcodeData.from_json(json_str)

        assert restored.num_barcodes == 1
        assert restored.entries[0].data == "ROUND_TRIP"
        assert restored.entries[0].type == "QR_CODE"

    def test_instance_ids_alignment(self):
        """instance_ids tuple aligns one-to-one with entries tuple."""
        entries = (
            BarcodeEntry(data="A", type="QR_CODE"),
            BarcodeEntry(data="B", type="EAN_13"),
        )
        ids = ("roi_0001", "roi_0002")
        artifact = BarcodeData(entries=entries, instance_ids=ids)

        assert len(artifact.entries) == len(artifact.instance_ids)
        assert artifact.instance_ids[0] == "roi_0001"
        assert artifact.instance_ids[1] == "roi_0002"

    def test_is_frozen(self):
        """Attempting to mutate a frozen BarcodeData raises AttributeError."""
        artifact = BarcodeData()
        with pytest.raises(AttributeError):
            artifact.entries = ()  # type: ignore[misc]

    def test_inherits_artifact(self):
        """BarcodeData is a subclass of the base Artifact."""
        artifact = BarcodeData()
        assert isinstance(artifact, Artifact)

    def test_entries_are_tuple(self):
        """entries defaults to an empty tuple, not a list."""
        artifact = BarcodeData()
        assert isinstance(artifact.entries, tuple)

    def test_meta_preserved(self):
        """meta dict values are preserved through to_dict() serialisation."""
        artifact = BarcodeData(meta={"engine": "pyzbar", "elapsed_ms": 12.5})
        d = artifact.to_dict()
        assert d["meta"]["engine"] == "pyzbar"
        assert d["meta"]["elapsed_ms"] == 12.5


# ===========================================================================
# TestBarcodeNode
# ===========================================================================


class TestBarcodeNode:
    """Unit tests for the Barcode graph node."""

    # ------------------------------------------------------------------ #
    # Class-level attributes                                               #
    # ------------------------------------------------------------------ #

    def test_node_has_correct_inputs_outputs(self):
        """Barcode node declares 'image' input and 'barcodes' output by default."""
        node = Barcode(using="pyzbar")
        assert "image" in node.inputs
        assert "barcodes" in node.outputs
        assert node.outputs["barcodes"] is BarcodeData

    def test_init_with_name(self):
        """Custom node name is stored on the instance."""
        node = Barcode(using="pyzbar", name="qr_scanner")
        assert node.name == "qr_scanner"

    # ------------------------------------------------------------------ #
    # Image input                                                          #
    # ------------------------------------------------------------------ #

    def test_run_on_image(self):
        """run() with an Image artifact returns BarcodeData under default key."""
        mock = _make_mock_provider(_make_barcode_result("HELLO", "WORLD"))
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})
        node = Barcode(using="pyzbar")

        result = node.run(ctx, image=_make_image_artifact())

        assert "barcodes" in result
        assert isinstance(result["barcodes"], BarcodeData)
        assert result["barcodes"].num_barcodes == 2

    def test_run_image_empty_result(self):
        """run() with a provider returning no barcodes produces empty BarcodeData."""
        mock = _make_mock_provider(BarcodeResult(barcodes=[]))
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})
        node = Barcode(using="pyzbar")

        result = node.run(ctx, image=_make_image_artifact())["barcodes"]

        assert result.is_empty

    # ------------------------------------------------------------------ #
    # ROIs input                                                           #
    # ------------------------------------------------------------------ #

    def test_run_on_rois(self):
        """run() with a ROIs artifact calls predict() once per crop."""
        mock = _make_mock_provider(_make_barcode_result("CODE"))
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})
        node = Barcode(using="pyzbar")
        rois = _make_rois(n=3)

        node.run(ctx, rois=rois)

        assert mock.predict.call_count == 3

    def test_run_on_rois_preserves_instance_ids(self):
        """instance_ids in BarcodeData output align with the source ROI ids."""
        mock = _make_mock_provider(_make_barcode_result("CODE"))
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})
        node = Barcode(using="pyzbar")
        rois = _make_rois(n=3)

        result = node.run(ctx, rois=rois)["barcodes"]

        # One entry per crop, each carrying the corresponding ROI instance_id
        assert len(result.instance_ids) == 3
        for iid in result.instance_ids:
            assert iid in rois.instance_ids

    def test_run_rois_empty_crops(self):
        """Empty ROIs produce an empty BarcodeData without calling predict()."""
        mock = _make_mock_provider()
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})
        node = Barcode(using="pyzbar")
        empty_rois = ROIs(roi_images=[], instance_ids=[], source_boxes=[])

        result = node.run(ctx, rois=empty_rois)["barcodes"]

        assert result.is_empty
        mock.predict.assert_not_called()

    def test_run_rois_multiple_barcodes_per_crop(self):
        """Multiple barcodes per crop are all collected in the output."""
        # predict returns 2 barcodes per crop
        double_result = _make_barcode_result("A", "B")
        mock = _make_mock_provider(double_result)
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})
        node = Barcode(using="pyzbar")
        rois = _make_rois(n=2)

        result = node.run(ctx, rois=rois)["barcodes"]

        # 2 crops × 2 barcodes = 4 total
        assert result.num_barcodes == 4
        assert len(result.instance_ids) == 4

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def test_run_records_latency_metric(self):
        """ctx.record_metric() is called with 'latency_ms'."""
        mock = _make_mock_provider()
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})
        ctx.record_metric = MagicMock()
        node = Barcode(using="pyzbar")

        node.run(ctx, image=_make_image_artifact())

        calls = [str(c) for c in ctx.record_metric.call_args_list]
        assert any("latency_ms" in c for c in calls)

    def test_run_records_num_barcodes_metric(self):
        """ctx.record_metric() is called with 'num_barcodes'."""
        mock = _make_mock_provider(_make_barcode_result("X", "Y", "Z"))
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})
        ctx.record_metric = MagicMock()
        node = Barcode(using="pyzbar")

        node.run(ctx, image=_make_image_artifact())

        calls = [str(c) for c in ctx.record_metric.call_args_list]
        assert any("num_barcodes" in c for c in calls)

    # ------------------------------------------------------------------ #
    # Configuration options                                                #
    # ------------------------------------------------------------------ #

    def test_output_name_customizable(self):
        """out='codes' stores the result under 'codes', not 'barcodes'."""
        mock = _make_mock_provider()
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})
        node = Barcode(using="pyzbar", out="codes")

        result = node.run(ctx, image=_make_image_artifact())

        assert "codes" in result
        assert "barcodes" not in result

    def test_src_override(self):
        """src='my_img' resolves the input from that key instead of 'image'."""
        mock = _make_mock_provider()
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})
        node = Barcode(using="pyzbar", src="my_img")
        image = _make_image_artifact()

        result = node.run(ctx, my_img=image)

        assert "barcodes" in result
        mock.predict.assert_called_once()

    def test_kwargs_forwarded_to_predict(self):
        """Extra kwargs passed at construction are forwarded to provider.predict()."""
        mock = _make_mock_provider()
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})
        node = Barcode(using="pyzbar", symbols=["QRCODE"])

        node.run(ctx, image=_make_image_artifact())

        _, call_kwargs = mock.predict.call_args
        assert call_kwargs.get("symbols") == ["QRCODE"]

    def test_provider_resolved_from_context(self):
        """Provider is looked up by capability='barcode' and the given name."""
        mock_a = _make_mock_provider(_make_barcode_result("provider_a"))
        mock_b = _make_mock_provider(_make_barcode_result("provider_b"))
        ctx = _make_ctx({"barcode": {"engine_a": mock_a, "engine_b": mock_b}})
        node = Barcode(using="engine_b")

        result = node.run(ctx, image=_make_image_artifact())["barcodes"]

        mock_b.predict.assert_called_once()
        mock_a.predict.assert_not_called()
        assert result.entries[0].data == "provider_b"

    # ------------------------------------------------------------------ #
    # Error handling                                                       #
    # ------------------------------------------------------------------ #

    def test_run_no_input_raises(self):
        """run() with no recognised input raises ValueError."""
        mock = _make_mock_provider()
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})
        node = Barcode(using="pyzbar")

        with pytest.raises(ValueError, match="no inputs"):
            node.run(ctx)

    def test_run_invalid_input_type_raises(self):
        """Non-Image / non-ROIs input raises ValueError."""
        mock = _make_mock_provider()
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})
        node = Barcode(using="pyzbar")

        with pytest.raises(ValueError, match="Image or ROIs"):
            node.run(ctx, image="not_an_artifact")

    # ------------------------------------------------------------------ #
    # BarcodeData pass-through                                             #
    # ------------------------------------------------------------------ #

    def test_run_with_barcode_data_passthrough(self):
        """If provider.predict() returns BarcodeData directly it is passed through."""
        prebuilt = BarcodeData(entries=(BarcodeEntry(data="pre", type="QR_CODE"),))
        mock = MagicMock()
        mock.predict.return_value = prebuilt
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})
        node = Barcode(using="pyzbar")

        result = node.run(ctx, image=_make_image_artifact())["barcodes"]

        assert result is prebuilt

    # ------------------------------------------------------------------ #
    # Input priority                                                       #
    # ------------------------------------------------------------------ #

    def test_input_priority_src_then_rois_then_image(self):
        """src key takes priority when both src and 'image' are supplied."""
        mock = _make_mock_provider()
        ctx = _make_ctx({"barcode": {"pyzbar": mock}})

        # node wired to 'custom_key', not 'image' or 'rois'
        node = Barcode(using="pyzbar", src="custom_key")
        img_a = _make_image_artifact()
        img_b = _make_image_artifact()

        # Both custom_key and image supplied — custom_key must win
        result = node.run(ctx, custom_key=img_a, image=img_b)

        assert "barcodes" in result
        # Only one call since we resolved a single Image (not ROIs loop)
        assert mock.predict.call_count == 1

    # ------------------------------------------------------------------ #
    # Exports                                                              #
    # ------------------------------------------------------------------ #

    def test_node_exported_from_nodes_package(self):
        """Barcode is importable directly from mata.nodes."""
        from mata.nodes import Barcode as BarcodeFromPackage

        assert BarcodeFromPackage is Barcode

    # ------------------------------------------------------------------ #
    # Graph composition                                                    #
    # ------------------------------------------------------------------ #

    def test_graph_composition(self):
        """Barcode node executes correctly inside mata.infer()."""
        mock = _make_mock_provider(_make_barcode_result("GRAPH_CODE"))

        result = mata.infer(
            image=_make_pil_image(),
            graph=[Barcode(using="pyzbar", out="codes")],
            providers={"barcode": {"pyzbar": mock}},
        )

        assert result.has_channel("codes")
        assert isinstance(result.channels["codes"], BarcodeData)
        assert result.channels["codes"].entries[0].data == "GRAPH_CODE"


# ===========================================================================
# TestBarcodeToolSchema
# ===========================================================================


class TestBarcodeToolSchema:
    """Tests for the 'barcode' entry in TASK_SCHEMA_DEFAULTS."""

    def test_schema_exists_in_defaults(self):
        """TASK_SCHEMA_DEFAULTS contains a 'barcode' key."""
        assert "barcode" in TASK_SCHEMA_DEFAULTS

    def test_schema_for_task_barcode(self):
        """schema_for_task('barcode') returns a valid ToolSchema without raising."""
        schema = schema_for_task("barcode")
        assert schema is not None

    def test_schema_task_is_barcode(self):
        """The schema's task field is 'barcode'."""
        schema = TASK_SCHEMA_DEFAULTS["barcode"]
        assert schema.task == "barcode"

    def test_schema_has_region_parameter(self):
        """The barcode schema includes a 'region' parameter."""
        schema = TASK_SCHEMA_DEFAULTS["barcode"]
        param_names = [p.name for p in schema.parameters]
        assert "region" in param_names

    def test_schema_region_not_required(self):
        """The 'region' parameter is optional (required=False)."""
        schema = TASK_SCHEMA_DEFAULTS["barcode"]
        region_param = next(p for p in schema.parameters if p.name == "region")
        assert region_param.required is False

    def test_schema_not_builtin(self):
        """The barcode schema is a provider-based tool, not a built-in."""
        schema = TASK_SCHEMA_DEFAULTS["barcode"]
        assert schema.builtin is False

    def test_schema_description_mentions_qr(self):
        """The schema description references QR codes."""
        schema = TASK_SCHEMA_DEFAULTS["barcode"]
        assert "qr" in schema.description.lower() or "QR" in schema.description

    def test_schema_description_mentions_barcode(self):
        """The schema description references barcodes."""
        schema = TASK_SCHEMA_DEFAULTS["barcode"]
        assert "barcode" in schema.description.lower()
