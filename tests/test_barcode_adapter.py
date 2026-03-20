"""Comprehensive mock-based unit tests for all barcode adapters.

Covers:
- BarcodeRegion / BarcodeResult core types
- ModelType enum (PYZBAR / ZXING entries)
- PyzbarAdapter (predict, bbox, type normalisation, filter, error handling)
- ZxingAdapter (predict, bbox, filter, error handling)

No real pyzbar or zxingcpp is installed — all external dependencies are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pil_image(width: int = 64, height: int = 32) -> PILImage.Image:
    return PILImage.new("RGB", (width, height), color=(200, 200, 200))


def _make_pyzbar_decoded(
    type_str: str = "QRCODE",
    data_bytes: bytes = b"https://example.com",
    left: int = 10,
    top: int = 20,
    width: int = 90,
    height: int = 80,
) -> Mock:
    """Build a mock pyzbar decoded object."""
    rect = Mock()
    rect.left = left
    rect.top = top
    rect.width = width
    rect.height = height
    obj = Mock()
    obj.type = type_str
    obj.data = data_bytes
    obj.rect = rect
    return obj


def _make_zxing_result(
    format_name: str = "QRCode",
    text: str = "https://example.com",
    raw_bytes: bytes | None = b"\x00\x01",
    corners: tuple[tuple[int, int], ...] = ((10, 20), (100, 20), (100, 100), (10, 100)),
) -> Mock:
    """Build a mock zxingcpp result object."""
    fmt = Mock()
    fmt.name = format_name
    pos = Mock()
    pos.top_left = Mock(x=corners[0][0], y=corners[0][1])
    pos.top_right = Mock(x=corners[1][0], y=corners[1][1])
    pos.bottom_left = Mock(x=corners[3][0], y=corners[3][1])
    pos.bottom_right = Mock(x=corners[2][0], y=corners[2][1])
    r = Mock()
    r.format = fmt
    r.text = text
    r.position = pos
    r.bytes = raw_bytes
    return r


# ===========================================================================
# Group 1 – BarcodeRegion and BarcodeResult core types (~20 tests)
# ===========================================================================


class TestBarcodeRegionAndResult:
    """Tests for BarcodeRegion, BarcodeResult dataclasses and their helpers."""

    # ------------------------------------------------------------------
    # BarcodeRegion construction
    # ------------------------------------------------------------------

    def test_barcode_region_minimal(self):
        from mata.core.types import BarcodeRegion

        r = BarcodeRegion(data="123456789012", type="EAN_13")
        assert r.data == "123456789012"
        assert r.type == "EAN_13"
        assert r.bbox is None
        assert r.score == 1.0
        assert r.raw_bytes is None

    def test_barcode_region_with_all_fields(self):
        from mata.core.types import BarcodeRegion

        r = BarcodeRegion(
            data="https://example.com",
            type="QR_CODE",
            bbox=(10.0, 20.0, 110.0, 120.0),
            score=1.0,
            raw_bytes=b"\x68\x74\x74\x70",
        )
        assert r.data == "https://example.com"
        assert r.type == "QR_CODE"
        assert r.bbox == (10.0, 20.0, 110.0, 120.0)
        assert r.score == 1.0
        assert r.raw_bytes == b"\x68\x74\x74\x70"

    def test_barcode_region_is_frozen(self):
        from mata.core.types import BarcodeRegion

        r = BarcodeRegion(data="abc", type="QR_CODE")
        with pytest.raises((AttributeError, TypeError)):
            r.data = "changed"  # type: ignore[misc]

    def test_barcode_region_to_dict_with_bbox(self):
        from mata.core.types import BarcodeRegion

        r = BarcodeRegion(data="hello", type="CODE_128", bbox=(1.0, 2.0, 101.0, 52.0), score=1.0)
        d = r.to_dict()
        assert d["data"] == "hello"
        assert d["type"] == "CODE_128"
        assert d["score"] == 1.0
        assert d["bbox"] == [1.0, 2.0, 101.0, 52.0]

    def test_barcode_region_to_dict_no_bbox(self):
        from mata.core.types import BarcodeRegion

        r = BarcodeRegion(data="hello", type="QR_CODE")
        d = r.to_dict()
        assert "bbox" not in d

    def test_barcode_region_to_dict_with_raw_bytes(self):
        from mata.core.types import BarcodeRegion

        raw = b"\xde\xad\xbe\xef"
        r = BarcodeRegion(data="binary", type="QR_CODE", raw_bytes=raw)
        d = r.to_dict()
        assert d["raw_bytes"] == raw.hex()

    def test_barcode_region_from_dict(self):
        from mata.core.types import BarcodeRegion

        d = {"data": "test", "type": "EAN_13", "score": 0.99, "bbox": [5.0, 6.0, 55.0, 56.0]}
        r = BarcodeRegion.from_dict(d)
        assert r.data == "test"
        assert r.type == "EAN_13"
        assert r.score == 0.99
        assert r.bbox == (5.0, 6.0, 55.0, 56.0)
        assert r.raw_bytes is None

    def test_barcode_region_from_dict_roundtrip(self):
        from mata.core.types import BarcodeRegion

        original = BarcodeRegion(
            data="roundtrip",
            type="CODE_39",
            bbox=(0.0, 0.0, 200.0, 100.0),
            score=1.0,
            raw_bytes=b"\x01\x02\x03",
        )
        restored = BarcodeRegion.from_dict(original.to_dict())
        assert restored == original

    # ------------------------------------------------------------------
    # BarcodeResult construction
    # ------------------------------------------------------------------

    def test_barcode_result_empty(self):
        from mata.core.types import BarcodeResult

        result = BarcodeResult(barcodes=[])
        assert len(result) == 0

    def test_barcode_result_with_barcodes(self):
        from mata.core.types import BarcodeRegion, BarcodeResult

        barcodes = [
            BarcodeRegion(data="A", type="QR_CODE"),
            BarcodeRegion(data="B", type="EAN_13"),
        ]
        result = BarcodeResult(barcodes=barcodes)
        assert len(result) == 2

    def test_barcode_result_len(self):
        from mata.core.types import BarcodeRegion, BarcodeResult

        result = BarcodeResult(barcodes=[BarcodeRegion(data="X", type="QR_CODE")] * 5)
        assert len(result) == 5

    def test_barcode_result_iter(self):
        from mata.core.types import BarcodeRegion, BarcodeResult

        codes = [BarcodeRegion(data=str(i), type="CODE_128") for i in range(3)]
        result = BarcodeResult(barcodes=codes)
        collected = list(result)
        assert len(collected) == 3
        assert collected[0].data == "0"
        assert collected[2].data == "2"

    def test_barcode_result_filter_by_type(self):
        from mata.core.types import BarcodeRegion, BarcodeResult

        result = BarcodeResult(
            barcodes=[
                BarcodeRegion(data="q1", type="QR_CODE"),
                BarcodeRegion(data="e1", type="EAN_13"),
                BarcodeRegion(data="q2", type="QR_CODE"),
            ]
        )
        filtered = result.filter_by_type("QR_CODE")
        assert len(filtered) == 2
        assert all(b.type == "QR_CODE" for b in filtered.barcodes)

    def test_barcode_result_filter_by_type_case_insensitive(self):
        from mata.core.types import BarcodeRegion, BarcodeResult

        result = BarcodeResult(
            barcodes=[
                BarcodeRegion(data="q", type="QR_CODE"),
                BarcodeRegion(data="e", type="EAN_13"),
            ]
        )
        filtered = result.filter_by_type("qr_code")
        assert len(filtered) == 1
        assert filtered.barcodes[0].data == "q"

    def test_barcode_result_to_json(self):
        from mata.core.types import BarcodeRegion, BarcodeResult

        result = BarcodeResult(
            barcodes=[BarcodeRegion(data="hello", type="QR_CODE")],
            meta={"engine": "pyzbar"},
        )
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert len(parsed["barcodes"]) == 1
        assert parsed["barcodes"][0]["data"] == "hello"
        assert parsed["meta"]["engine"] == "pyzbar"

    def test_barcode_result_from_json(self):
        from mata.core.types import BarcodeResult

        payload = json.dumps(
            {
                "barcodes": [{"data": "parsed", "type": "EAN_13", "score": 1.0}],
                "meta": {"engine": "zxing"},
            }
        )
        result = BarcodeResult.from_json(payload)
        assert len(result.barcodes) == 1
        assert result.barcodes[0].data == "parsed"
        assert result.meta["engine"] == "zxing"

    def test_barcode_result_to_json_roundtrip(self):
        from mata.core.types import BarcodeRegion, BarcodeResult

        original = BarcodeResult(
            barcodes=[
                BarcodeRegion(data="abc", type="CODE_39", bbox=(1.0, 2.0, 50.0, 30.0)),
                BarcodeRegion(data="xyz", type="QR_CODE", raw_bytes=b"\xde\xad"),
            ],
            meta={"engine": "pyzbar"},
        )
        restored = BarcodeResult.from_json(original.to_json())
        assert len(restored.barcodes) == 2
        assert restored.barcodes[0].data == "abc"
        assert restored.barcodes[1].raw_bytes == b"\xde\xad"
        assert restored.meta["engine"] == "pyzbar"

    def test_barcode_result_save_json(self, tmp_path):
        from mata.core.types import BarcodeRegion, BarcodeResult

        result = BarcodeResult(
            barcodes=[BarcodeRegion(data="save-test", type="QR_CODE")],
            meta={"engine": "pyzbar"},
        )
        outfile = str(tmp_path / "barcodes.json")
        result.save(outfile)
        with open(outfile) as f:
            data = json.load(f)
        assert "barcodes" in data
        assert data["barcodes"][0]["data"] == "save-test"

    def test_barcode_result_save_csv(self, tmp_path):
        from mata.core.types import BarcodeRegion, BarcodeResult

        result = BarcodeResult(
            barcodes=[
                BarcodeRegion(data="1234567890128", type="EAN_13", bbox=(0.0, 0.0, 100.0, 50.0)),
            ]
        )
        outfile = str(tmp_path / "barcodes.csv")
        result.save(outfile)
        with open(outfile) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert lines[0].strip() == "data,type,score,x1,y1,x2,y2"
        assert "1234567890128" in lines[1]

    def test_barcode_result_save_unsupported_raises(self, tmp_path):
        from mata.core.types import BarcodeResult

        result = BarcodeResult(barcodes=[])
        with pytest.raises(ValueError, match="Unsupported"):
            result.save(str(tmp_path / "out.xml"))


# ===========================================================================
# Group 2 – ModelType enum entries (~3 tests)
# ===========================================================================


class TestBarcodeModelType:
    """Tests for ModelType.PYZBAR and ModelType.ZXING enum entries."""

    def test_pyzbar_model_type_exists(self):
        from mata.core.types import ModelType

        assert hasattr(ModelType, "PYZBAR")
        assert ModelType.PYZBAR is not None

    def test_zxing_model_type_exists(self):
        from mata.core.types import ModelType

        assert hasattr(ModelType, "ZXING")
        assert ModelType.ZXING is not None

    def test_model_type_string_values(self):
        from mata.core.types import ModelType

        assert ModelType.PYZBAR == "pyzbar"
        assert ModelType.ZXING == "zxing"


# ===========================================================================
# Group 3 – PyzbarAdapter (~20 tests, all mocked)
# ===========================================================================


class TestPyzbarAdapter:
    """Tests for PyzbarAdapter — all pyzbar calls are mocked."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_init_calls_ensure(self, mock_ensure):
        mock_ensure.return_value = Mock()
        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        PyzbarAdapter()
        mock_ensure.assert_called()

    # ------------------------------------------------------------------
    # predict() — basic contract
    # ------------------------------------------------------------------

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_predict_returns_barcode_result(self, mock_ensure):
        mock_pz = Mock()
        mock_pz.decode.return_value = []
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter
        from mata.core.types import BarcodeResult

        adapter = PyzbarAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert isinstance(result, BarcodeResult)

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_predict_qr_code(self, mock_ensure):
        mock_pz = Mock()
        mock_pz.decode.return_value = [_make_pyzbar_decoded("QRCODE", b"https://example.com", 10, 20, 90, 80)]
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert len(result.barcodes) == 1
        assert result.barcodes[0].data == "https://example.com"
        assert result.barcodes[0].type == "QR_CODE"

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_predict_ean13(self, mock_ensure):
        mock_pz = Mock()
        mock_pz.decode.return_value = [_make_pyzbar_decoded("EAN13", b"4006381333931")]
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert result.barcodes[0].type == "EAN_13"
        assert result.barcodes[0].data == "4006381333931"

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_predict_multiple_barcodes(self, mock_ensure):
        mock_pz = Mock()
        mock_pz.decode.return_value = [
            _make_pyzbar_decoded("QRCODE", b"qr1", 0, 0, 50, 50),
            _make_pyzbar_decoded("EAN13", b"1234567890123", 100, 0, 100, 50),
            _make_pyzbar_decoded("CODE128", b"ABCDEF", 200, 0, 100, 50),
        ]
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert len(result.barcodes) == 3

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_predict_no_barcodes_empty_result(self, mock_ensure):
        mock_pz = Mock()
        mock_pz.decode.return_value = []
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert len(result.barcodes) == 0

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_predict_bbox_xyxy_format(self, mock_ensure):
        mock_pz = Mock()
        # rect: left=10, top=20, width=90, height=40
        mock_pz.decode.return_value = [_make_pyzbar_decoded("QRCODE", b"data", left=10, top=20, width=90, height=40)]
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        bbox = result.barcodes[0].bbox
        assert bbox is not None
        # xyxy: (left, top, left+width, top+height)
        assert bbox == (10.0, 20.0, 100.0, 60.0)

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_predict_type_normalization(self, mock_ensure):
        """Known pyzbar types are normalized to MATA canonical names; unknown passed through."""
        mock_pz = Mock()
        mock_pz.decode.return_value = [
            _make_pyzbar_decoded("UPCA", b"012345678905"),
            _make_pyzbar_decoded("AZTEC", b"HELLO"),  # not in map → kept as-is
        ]
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert result.barcodes[0].type == "UPC_A"
        assert result.barcodes[1].type == "AZTEC"  # unmapped → passed through

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_predict_score_always_one(self, mock_ensure):
        mock_pz = Mock()
        mock_pz.decode.return_value = [_make_pyzbar_decoded("QRCODE", b"test")]
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert result.barcodes[0].score == 1.0

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_predict_raw_bytes_preserved(self, mock_ensure):
        raw = b"\x00\x01\x02\x03"
        mock_pz = Mock()
        mock_pz.decode.return_value = [_make_pyzbar_decoded("QRCODE", raw)]
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert result.barcodes[0].raw_bytes == raw

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_predict_pil_image_input(self, mock_ensure):
        mock_pz = Mock()
        mock_pz.decode.return_value = []
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)) as mock_load:
            adapter.predict(pil_img)
            mock_load.assert_called_once_with(pil_img)

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_predict_numpy_input(self, mock_ensure):
        import numpy as np

        mock_pz = Mock()
        mock_pz.decode.return_value = []
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter()
        pil_img = _make_pil_image()
        arr = np.zeros((32, 64, 3), dtype="uint8")
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)) as mock_load:
            adapter.predict(arr)
            mock_load.assert_called_once_with(arr)

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_predict_file_path_input(self, mock_ensure):
        mock_pz = Mock()
        mock_pz.decode.return_value = []
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)) as mock_load:
            adapter.predict("image.jpg")
            mock_load.assert_called_once_with("image.jpg")

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_symbols_filter(self, mock_ensure):
        """Only barcodes in the symbols allow-list are returned."""
        mock_pz = Mock()
        mock_pz.decode.return_value = [
            _make_pyzbar_decoded("QRCODE", b"qr-only"),
            _make_pyzbar_decoded("EAN13", b"not-wanted"),
        ]
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter(symbols=["QR_CODE"])
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert len(result.barcodes) == 1
        assert result.barcodes[0].type == "QR_CODE"

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_symbols_filter_excludes(self, mock_ensure):
        """When symbols filter is set, non-matching types are excluded."""
        mock_pz = Mock()
        mock_pz.decode.return_value = [
            _make_pyzbar_decoded("CODE128", b"filtered-out"),
        ]
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter(symbols=["QR_CODE"])
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert len(result.barcodes) == 0

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_info_returns_metadata(self, mock_ensure):
        mock_ensure.return_value = Mock()

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter(symbols=["QR_CODE"])
        info = adapter.info()
        assert info["name"] == "pyzbar"
        assert info["task"] == "barcode"
        assert info["symbols"] == ["QR_CODE"]

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_name_and_task_attributes(self, mock_ensure):
        mock_ensure.return_value = Mock()

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter()
        assert adapter.name == "pyzbar"
        assert adapter.task == "barcode"

    def test_import_error_helpful_message(self):
        """Missing pyzbar raises ImportError with install instructions."""
        with patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar") as mock_ensure:
            mock_ensure.side_effect = ImportError(
                "pyzbar is required for PyzbarAdapter. " "Install with: pip install pyzbar"
            )
            from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

            with pytest.raises(ImportError, match="pyzbar"):
                PyzbarAdapter()

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_meta_contains_engine(self, mock_ensure):
        mock_pz = Mock()
        mock_pz.decode.return_value = []
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert result.meta["engine"] == "pyzbar"

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_utf8_decode_errors_handled(self, mock_ensure):
        """Non-UTF-8 bytes are replaced rather than raising."""
        # Bytes that are invalid in UTF-8 strict mode
        invalid_utf8 = b"\xff\xfe binary \x00"
        mock_pz = Mock()
        mock_pz.decode.return_value = [_make_pyzbar_decoded("QRCODE", invalid_utf8)]
        mock_ensure.return_value = mock_pz

        from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter

        adapter = PyzbarAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        # Should succeed (errors="replace") — result.data is a string
        assert isinstance(result.barcodes[0].data, str)


# ===========================================================================
# Group 4 – ZxingAdapter (~15 tests, all mocked)
# ===========================================================================


class TestZxingAdapter:
    """Tests for ZxingAdapter — all zxingcpp calls are mocked."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_init_calls_ensure(self, mock_ensure):
        mock_ensure.return_value = Mock()
        from mata.adapters.barcode.zxing_adapter import ZxingAdapter

        ZxingAdapter()
        mock_ensure.assert_called()

    # ------------------------------------------------------------------
    # predict() — basic contract
    # ------------------------------------------------------------------

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_predict_returns_barcode_result(self, mock_ensure):
        mock_zx = Mock()
        mock_zx.read_barcodes.return_value = []
        mock_ensure.return_value = mock_zx

        from mata.adapters.barcode.zxing_adapter import ZxingAdapter
        from mata.core.types import BarcodeResult

        adapter = ZxingAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert isinstance(result, BarcodeResult)

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_predict_qr_code(self, mock_ensure):
        mock_zx = Mock()
        mock_zx.read_barcodes.return_value = [_make_zxing_result("QRCode", "https://example.com")]
        mock_ensure.return_value = mock_zx

        from mata.adapters.barcode.zxing_adapter import ZxingAdapter

        adapter = ZxingAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert len(result.barcodes) == 1
        assert result.barcodes[0].data == "https://example.com"
        assert result.barcodes[0].type == "QRCode"

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_predict_bbox_from_position(self, mock_ensure):
        """Bbox is computed as min/max of all four corner coordinates."""
        mock_zx = Mock()
        corners = ((10, 20), (110, 20), (110, 120), (10, 120))
        mock_zx.read_barcodes.return_value = [_make_zxing_result("QRCode", "data", corners=corners)]
        mock_ensure.return_value = mock_zx

        from mata.adapters.barcode.zxing_adapter import ZxingAdapter

        adapter = ZxingAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        bbox = result.barcodes[0].bbox
        assert bbox is not None
        assert bbox[0] == 10.0  # min x
        assert bbox[1] == 20.0  # min y
        assert bbox[2] == 110.0  # max x
        assert bbox[3] == 120.0  # max y

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_predict_no_barcodes(self, mock_ensure):
        mock_zx = Mock()
        mock_zx.read_barcodes.return_value = []
        mock_ensure.return_value = mock_zx

        from mata.adapters.barcode.zxing_adapter import ZxingAdapter

        adapter = ZxingAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert len(result.barcodes) == 0

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_predict_multiple_barcodes(self, mock_ensure):
        mock_zx = Mock()
        mock_zx.read_barcodes.return_value = [
            _make_zxing_result("QRCode", "qr1"),
            _make_zxing_result("EAN13", "1234567890123"),
            _make_zxing_result("Code128", "ABCD"),
        ]
        mock_ensure.return_value = mock_zx

        from mata.adapters.barcode.zxing_adapter import ZxingAdapter

        adapter = ZxingAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert len(result.barcodes) == 3

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_formats_filter(self, mock_ensure):
        """Only barcodes in the formats allow-list are returned."""
        mock_zx = Mock()
        mock_zx.read_barcodes.return_value = [
            _make_zxing_result("QRCode", "keep"),
            _make_zxing_result("EAN13", "discard"),
        ]
        mock_ensure.return_value = mock_zx

        from mata.adapters.barcode.zxing_adapter import ZxingAdapter

        adapter = ZxingAdapter(formats=["QRCode"])
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert len(result.barcodes) == 1
        assert result.barcodes[0].data == "keep"

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_formats_filter_excludes(self, mock_ensure):
        """Non-matching formats are excluded when a filter is set."""
        mock_zx = Mock()
        mock_zx.read_barcodes.return_value = [
            _make_zxing_result("Code128", "excluded"),
        ]
        mock_ensure.return_value = mock_zx

        from mata.adapters.barcode.zxing_adapter import ZxingAdapter

        adapter = ZxingAdapter(formats=["QRCode"])
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert len(result.barcodes) == 0

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_info_returns_metadata(self, mock_ensure):
        mock_ensure.return_value = Mock()

        from mata.adapters.barcode.zxing_adapter import ZxingAdapter

        adapter = ZxingAdapter(formats=["QRCode", "EAN13"])
        info = adapter.info()
        assert info["name"] == "zxing"
        assert info["task"] == "barcode"
        assert info["formats"] == ["QRCode", "EAN13"]

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_name_and_task_attributes(self, mock_ensure):
        mock_ensure.return_value = Mock()

        from mata.adapters.barcode.zxing_adapter import ZxingAdapter

        adapter = ZxingAdapter()
        assert adapter.name == "zxing"
        assert adapter.task == "barcode"

    def test_import_error_helpful_message(self):
        """Missing zxingcpp raises ImportError with install instructions."""
        with patch("mata.adapters.barcode.zxing_adapter._ensure_zxing") as mock_ensure:
            mock_ensure.side_effect = ImportError(
                "zxingcpp is required for ZxingAdapter. " "Install with: pip install zxing-cpp"
            )
            from mata.adapters.barcode.zxing_adapter import ZxingAdapter

            with pytest.raises(ImportError, match="zxingcpp"):
                ZxingAdapter()

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_meta_contains_engine(self, mock_ensure):
        mock_zx = Mock()
        mock_zx.read_barcodes.return_value = []
        mock_ensure.return_value = mock_zx

        from mata.adapters.barcode.zxing_adapter import ZxingAdapter

        adapter = ZxingAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert result.meta["engine"] == "zxing"

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_predict_pil_image_input(self, mock_ensure):
        mock_zx = Mock()
        mock_zx.read_barcodes.return_value = []
        mock_ensure.return_value = mock_zx

        from mata.adapters.barcode.zxing_adapter import ZxingAdapter

        adapter = ZxingAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)) as mock_load:
            adapter.predict(pil_img)
            mock_load.assert_called_once_with(pil_img)

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_score_always_one(self, mock_ensure):
        mock_zx = Mock()
        mock_zx.read_barcodes.return_value = [_make_zxing_result("QRCode", "data")]
        mock_ensure.return_value = mock_zx

        from mata.adapters.barcode.zxing_adapter import ZxingAdapter

        adapter = ZxingAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert result.barcodes[0].score == 1.0

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_raw_bytes_preserved(self, mock_ensure):
        raw = b"\xca\xfe\xba\xbe"
        mock_zx = Mock()
        mock_zx.read_barcodes.return_value = [_make_zxing_result("QRCode", "binary-data", raw_bytes=raw)]
        mock_ensure.return_value = mock_zx

        from mata.adapters.barcode.zxing_adapter import ZxingAdapter

        adapter = ZxingAdapter()
        pil_img = _make_pil_image()
        with patch.object(adapter, "_load_image", return_value=(pil_img, None)):
            result = adapter.predict(pil_img)

        assert result.barcodes[0].raw_bytes == raw
