"""Integration tests for barcode task — loader dispatch and public API flow.

Tests the full flow from ``mata.run("barcode", ...)`` / ``mata.load("barcode", ...)``
through UniversalLoader source-type detection to adapter dispatch.

Groups:
    TestBarcodeSourceDetection  — _detect_source_type() for barcode sources
    TestBarcodeEngineDispatch   — External engine loading and dispatch
    TestBarcodeAPI              — Public mata.run() / mata.load() API
    TestBarcodeExtras           — pyproject.toml optional-dependency extras

All external dependencies (pyzbar, zxingcpp) are mocked — no real library
install required to run this test suite.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

try:
    import tomllib
except ImportError:  # Python < 3.11
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore[assignment]

from PIL import Image

import mata
from mata.adapters.barcode.pyzbar_adapter import PyzbarAdapter
from mata.adapters.barcode.zxing_adapter import ZxingAdapter
from mata.core.exceptions import ModelNotFoundError, UnsupportedModelError
from mata.core.model_loader import UniversalLoader
from mata.core.types import BarcodeResult, ModelType

# ─── helpers ──────────────────────────────────────────────────────────────────

_PYPROJECT_PATH = Path(__file__).parent.parent / "pyproject.toml"


def _make_pil_image(width: int = 100, height: int = 100) -> Image.Image:
    """Return a minimal RGB PIL Image for test inputs."""
    return Image.new("RGB", (width, height), color=(200, 200, 200))


def _load_pyproject() -> dict:
    """Load and parse pyproject.toml."""
    if tomllib is None:
        pytest.skip("tomllib/tomli not available on this Python version")
    with open(_PYPROJECT_PATH, "rb") as fh:
        return tomllib.load(fh)


# ──────────────────────────────────────────────────────────────────────────────
# Group 1 — Source type detection (~6 tests)
# ──────────────────────────────────────────────────────────────────────────────


class TestBarcodeSourceDetection:
    """Tests for UniversalLoader._detect_source_type() on barcode-related sources."""

    def test_pyzbar_detected_as_external_engine(self):
        loader = UniversalLoader()
        source_type, resolved = loader._detect_source_type("barcode", "pyzbar")
        assert source_type == "external_engine"
        assert resolved == "pyzbar"

    def test_zxing_detected_as_external_engine(self):
        loader = UniversalLoader()
        source_type, resolved = loader._detect_source_type("barcode", "zxing")
        assert source_type == "external_engine"
        assert resolved == "zxing"

    def test_pyzbar_case_insensitive(self):
        loader = UniversalLoader()
        for variant in ("PYZBAR", "PyZbar", "Pyzbar"):
            source_type, resolved = loader._detect_source_type("barcode", variant)
            assert source_type == "external_engine", f"Failed for variant '{variant}'"
            assert resolved == "pyzbar", f"Expected 'pyzbar', got '{resolved}' for '{variant}'"

    def test_barcode_hf_model_detected_as_huggingface(self):
        loader = UniversalLoader()
        source_type, resolved = loader._detect_source_type("barcode", "org/barcode-model")
        assert source_type == "huggingface"
        assert resolved == "org/barcode-model"

    def test_barcode_alias_detected_as_config(self):
        loader = UniversalLoader()
        loader.registry = Mock()
        loader.registry.has_alias.return_value = True
        source_type, resolved = loader._detect_source_type("barcode", "my-barcode-alias")
        assert source_type == "config_alias"
        assert resolved == "my-barcode-alias"
        loader.registry.has_alias.assert_called_once_with("barcode", "my-barcode-alias")

    def test_barcode_onnx_detected_as_local_file(self):
        # Path-extension detection fires even without an existing file on disk.
        loader = UniversalLoader()
        source_type, resolved = loader._detect_source_type("barcode", "barcode_model.onnx")
        assert source_type == "local_file"
        assert resolved == "barcode_model.onnx"


# ──────────────────────────────────────────────────────────────────────────────
# Group 2 — External engine dispatch (~8 tests)
# ──────────────────────────────────────────────────────────────────────────────


class TestBarcodeEngineDispatch:
    """Tests for external engine loading, task validation, and adapter dispatch."""

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_load_pyzbar_returns_adapter(self, mock_ensure):
        mock_ensure.return_value = Mock()
        adapter = mata.load("barcode", "pyzbar")
        assert isinstance(adapter, PyzbarAdapter)

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_load_zxing_returns_adapter(self, mock_ensure):
        mock_ensure.return_value = Mock()
        adapter = mata.load("barcode", "zxing")
        assert isinstance(adapter, ZxingAdapter)

    def test_load_pyzbar_wrong_task_raises(self):
        # pyzbar is barcode-only — requesting it as an OCR engine must fail.
        with pytest.raises(UnsupportedModelError, match="barcode"):
            mata.load("ocr", "pyzbar")

    def test_load_zxing_wrong_task_raises(self):
        with pytest.raises(UnsupportedModelError, match="barcode"):
            mata.load("ocr", "zxing")

    @patch("mata.adapters.ocr.easyocr_adapter._ensure_easyocr")
    def test_load_ocr_engines_still_work(self, mock_ensure):
        """Adding barcode engines must not regress existing OCR engine dispatch."""
        from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter

        mock_easyocr = Mock()
        mock_easyocr.Reader.return_value = Mock()
        mock_ensure.return_value = mock_easyocr
        adapter = mata.load("ocr", "easyocr")
        assert isinstance(adapter, EasyOCRAdapter)

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_model_type_pyzbar_dispatch(self, mock_ensure):
        mock_ensure.return_value = Mock()
        adapter = mata.load("barcode", model_type=ModelType.PYZBAR)
        assert isinstance(adapter, PyzbarAdapter)

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_model_type_zxing_dispatch(self, mock_ensure):
        mock_ensure.return_value = Mock()
        adapter = mata.load("barcode", model_type=ModelType.ZXING)
        assert isinstance(adapter, ZxingAdapter)

    def test_unknown_engine_raises(self):
        # An unrecognised string falls through to config-alias lookup and fails.
        with pytest.raises((ModelNotFoundError, UnsupportedModelError)):
            mata.load("barcode", "totally_unknown_engine_xyz")


# ──────────────────────────────────────────────────────────────────────────────
# Group 3 — API integration (~6 tests)
# ──────────────────────────────────────────────────────────────────────────────


class TestBarcodeAPI:
    """Public mata.run() / mata.load() barcode API integration tests."""

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_run_barcode_pyzbar(self, mock_ensure):
        """Full flow through pyzbar engine produces a BarcodeResult."""
        mock_pyzbar_lib = Mock()
        mock_pyzbar_lib.decode.return_value = []
        mock_ensure.return_value = mock_pyzbar_lib

        result = mata.run("barcode", _make_pil_image(), model="pyzbar")

        assert isinstance(result, BarcodeResult)
        assert result.meta.get("engine") == "pyzbar"

    @patch("mata.adapters.barcode.zxing_adapter._ensure_zxing")
    def test_run_barcode_zxing(self, mock_ensure):
        """Full flow through zxing engine produces a BarcodeResult."""
        mock_zxing_lib = Mock()
        mock_zxing_lib.read_barcodes.return_value = []
        mock_ensure.return_value = mock_zxing_lib

        result = mata.run("barcode", _make_pil_image(), model="zxing")

        assert isinstance(result, BarcodeResult)
        assert result.meta.get("engine") == "zxing"

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_load_barcode(self, mock_ensure):
        """mata.load('barcode', 'pyzbar') returns adapter with correct task/name."""
        mock_ensure.return_value = Mock()
        adapter = mata.load("barcode", "pyzbar")
        assert adapter.task == "barcode"
        assert adapter.name == "pyzbar"

    def test_run_barcode_not_supported_error_includes_barcode(self):
        """Error raised when wrong task requests a barcode engine mentions 'barcode'."""
        with pytest.raises(UnsupportedModelError, match="barcode"):
            mata.load("detect", "pyzbar")

    @patch("mata.adapters.barcode.pyzbar_adapter._ensure_pyzbar")
    def test_run_barcode_returns_barcode_result(self, mock_ensure):
        """mata.run returns a BarcodeResult with correct decoded barcode data."""
        mock_decoded = Mock()
        mock_decoded.data = b"12345678"
        mock_decoded.type = "CODE128"
        mock_decoded.rect = Mock(left=10, top=20, width=80, height=40)

        mock_pyzbar_lib = Mock()
        mock_pyzbar_lib.decode.return_value = [mock_decoded]
        mock_ensure.return_value = mock_pyzbar_lib

        result = mata.run("barcode", _make_pil_image(), model="pyzbar")

        assert isinstance(result, BarcodeResult)
        assert len(result.barcodes) == 1
        assert result.barcodes[0].data == "12345678"
        assert result.barcodes[0].type == "CODE_128"  # normalised via _PYZBAR_TYPE_MAP

    def test_run_track_still_raises(self):
        """mata.run('track', ...) must still raise ValueError — no regression."""
        with pytest.raises(ValueError, match="track"):
            mata.run("track", _make_pil_image())


# ──────────────────────────────────────────────────────────────────────────────
# Group 4 — Pyproject extras verification (~5 tests)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def optional_deps() -> dict[str, list[str]]:
    """Return the [project.optional-dependencies] table from pyproject.toml."""
    return _load_pyproject().get("project", {}).get("optional-dependencies", {})


class TestBarcodeExtras:
    """Verify barcode optional-dependency extras are declared in pyproject.toml."""

    def test_barcode_extra_defined(self, optional_deps):
        assert "barcode" in optional_deps, "'barcode' extra not found in pyproject.toml"

    def test_barcode_zxing_extra_defined(self, optional_deps):
        assert "barcode-zxing" in optional_deps, "'barcode-zxing' extra not found"

    def test_barcode_all_extra_defined(self, optional_deps):
        assert "barcode-all" in optional_deps, "'barcode-all' extra not found"

    def test_barcode_extra_contains_pyzbar(self, optional_deps):
        barcode_deps = optional_deps.get("barcode", [])
        assert any(
            dep.startswith("pyzbar") for dep in barcode_deps
        ), f"pyzbar not listed in 'barcode' extra: {barcode_deps}"

    def test_barcode_zxing_extra_contains_zxingcpp(self, optional_deps):
        zxing_deps = optional_deps.get("barcode-zxing", [])
        assert any(
            "zxing-cpp" in dep for dep in zxing_deps
        ), f"zxing-cpp not listed in 'barcode-zxing' extra: {zxing_deps}"
