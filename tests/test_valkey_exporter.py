"""Tests for Valkey/Redis exporter — Task A1, A2, C1.

All tests use unittest.mock to simulate the Valkey/Redis client.
No real Valkey or Redis server is required.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Result factory helpers ────────────────────────────────────────────────────


def _make_vision_result():
    from mata.core.types import Instance, VisionResult

    return VisionResult(
        instances=[
            Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(50, 60, 150, 250), score=0.75, label=1, label_name="dog"),
        ],
        meta={"model": "detr-resnet-50"},
    )


def _make_classify_result():
    from mata.core.types import Classification, ClassifyResult

    return ClassifyResult(
        predictions=[
            Classification(label=0, score=0.95, label_name="cat"),
            Classification(label=1, score=0.04, label_name="dog"),
        ],
        meta={"model": "clip-vit"},
    )


def _make_depth_result():
    import numpy as np

    from mata.core.types import DepthResult

    depth = np.linspace(0.1, 1.0, 16, dtype=np.float32).reshape(4, 4)
    return DepthResult(depth=depth, meta={"model": "depth-anything"})


def _make_ocr_result():
    from mata.core.types import OCRResult, TextRegion

    return OCRResult(
        regions=[
            TextRegion(text="hello", score=0.98, bbox=(0.0, 0.0, 50.0, 20.0)),
            TextRegion(text="world", score=0.92, bbox=(60.0, 0.0, 120.0, 20.0)),
        ],
        meta={"engine": "easyocr"},
    )


def _mock_client():
    return MagicMock()


# ── TestExportValkey ──────────────────────────────────────────────────────────


class TestExportValkey:
    """Tests for export_valkey() function."""

    def test_export_vision_result_json(self):
        """JSON export of VisionResult stores JSON string via client.set()."""
        from mata.core.exporters.valkey_exporter import export_valkey

        result = _make_vision_result()
        client = _mock_client()
        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            export_valkey(result, url="valkey://localhost:6379", key="test:vision")

        client.set.assert_called_once_with("test:vision", result.to_json())
        client.setex.assert_not_called()

    def test_export_classify_result_json(self):
        """JSON export of ClassifyResult stored via client.set()."""
        from mata.core.exporters.valkey_exporter import export_valkey

        result = _make_classify_result()
        client = _mock_client()
        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            export_valkey(result, url="valkey://localhost:6379", key="test:classify")

        client.set.assert_called_once_with("test:classify", result.to_json())

    def test_export_depth_result_json(self):
        """JSON export of DepthResult stored via client.set()."""
        from mata.core.exporters.valkey_exporter import export_valkey

        result = _make_depth_result()
        client = _mock_client()
        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            export_valkey(result, url="valkey://localhost:6379", key="test:depth")

        client.set.assert_called_once_with("test:depth", result.to_json())

    def test_export_ocr_result_json(self):
        """JSON export of OCRResult stored via client.set()."""
        from mata.core.exporters.valkey_exporter import export_valkey

        result = _make_ocr_result()
        client = _mock_client()
        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            export_valkey(result, url="valkey://localhost:6379", key="test:ocr")

        client.set.assert_called_once_with("test:ocr", result.to_json())

    def test_export_with_ttl(self):
        """When ttl is provided, client.setex() is used instead of set()."""
        from mata.core.exporters.valkey_exporter import export_valkey

        result = _make_vision_result()
        client = _mock_client()
        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            export_valkey(result, url="valkey://localhost:6379", key="test:ttl", ttl=3600)

        client.setex.assert_called_once_with("test:ttl", 3600, result.to_json())
        client.set.assert_not_called()

    def test_export_without_ttl(self):
        """When ttl is None (default), client.set() is used and setex() is not."""
        from mata.core.exporters.valkey_exporter import export_valkey

        result = _make_classify_result()
        client = _mock_client()
        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            export_valkey(result, url="valkey://localhost:6379", key="test:nottl")

        client.set.assert_called_once()
        client.setex.assert_not_called()

    def test_export_msgpack_serializer(self):
        """msgpack serializer packs result.to_dict() and stores bytes."""
        from mata.core.exporters.valkey_exporter import export_valkey

        result = _make_vision_result()
        client = _mock_client()
        fake_packed = b"\x82\xa3foo\xa3bar"
        mock_msgpack = MagicMock()
        mock_msgpack.packb.return_value = fake_packed

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            with patch.dict(sys.modules, {"msgpack": mock_msgpack}):
                export_valkey(
                    result,
                    url="valkey://localhost:6379",
                    key="test:mp",
                    serializer="msgpack",
                )

        mock_msgpack.packb.assert_called_once_with(result.to_dict(), use_bin_type=True)
        client.set.assert_called_once_with("test:mp", fake_packed)

    def test_export_invalid_serializer_raises(self):
        """An unsupported serializer name raises ValueError with helpful message."""
        from mata.core.exporters.valkey_exporter import export_valkey

        result = _make_vision_result()
        client = _mock_client()
        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            with pytest.raises(ValueError, match="Unsupported serializer"):
                export_valkey(
                    result,
                    url="valkey://localhost:6379",
                    key="k",
                    serializer="xml",
                )

    def test_import_error_no_client(self):
        """ImportError raised with helpful message when neither valkey nor redis is available."""
        # Use a fresh call to _get_valkey_client to trigger the import error path
        from mata.core.exporters import valkey_exporter

        # Patch the imports inside the function
        with patch.dict(sys.modules, {"valkey": None, "redis": None}):
            with pytest.raises(ImportError, match="pip install datamata"):
                valkey_exporter._get_valkey_client("valkey://localhost:6379")

    def test_valkey_client_fallback_to_redis(self):
        """When valkey-py is absent, the client is obtained from redis-py."""
        from mata.core.exporters import valkey_exporter

        mock_redis_module = MagicMock()
        mock_client_instance = MagicMock()
        mock_redis_module.from_url.return_value = mock_client_instance

        with patch.dict(sys.modules, {"valkey": None, "redis": mock_redis_module}):
            client = valkey_exporter._get_valkey_client("valkey://localhost:6379")

        # URL scheme must be translated for redis-py
        mock_redis_module.from_url.assert_called_once_with("redis://localhost:6379")
        assert client is mock_client_instance

    def test_url_password_not_logged(self):
        """Passwords embedded in connection URLs must never appear in log messages."""
        from mata.core.exporters.valkey_exporter import export_valkey

        result = _make_vision_result()
        client = _mock_client()
        url_with_password = "valkey://admin:s3cr3tpassword@localhost:6379"

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            with patch("mata.core.exporters.valkey_exporter.logger") as mock_logger:
                export_valkey(result, url=url_with_password, key="test:pwd")

        for logged_call in mock_logger.info.call_args_list:
            logged_text = str(logged_call)
            assert "s3cr3tpassword" not in logged_text, f"Password leaked in log message: {logged_text}"


# ── TestLoadValkey ────────────────────────────────────────────────────────────


class TestLoadValkey:
    """Tests for load_valkey() function."""

    def test_load_vision_result(self):
        """Loading a VisionResult JSON returns a VisionResult instance."""
        from mata.core.exporters.valkey_exporter import load_valkey
        from mata.core.types import VisionResult

        original = _make_vision_result()
        raw_json = original.to_json().encode()
        client = _mock_client()
        client.get.return_value = raw_json

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            loaded = load_valkey("valkey://localhost:6379", key="test:vision", result_type="vision")

        assert isinstance(loaded, VisionResult)
        assert len(loaded.instances) == len(original.instances)

    def test_load_classify_result(self):
        """Loading a ClassifyResult JSON returns a ClassifyResult instance."""
        from mata.core.exporters.valkey_exporter import load_valkey
        from mata.core.types import ClassifyResult

        original = _make_classify_result()
        raw_json = original.to_json().encode()
        client = _mock_client()
        client.get.return_value = raw_json

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            loaded = load_valkey("valkey://localhost:6379", key="test:classify", result_type="classify")

        assert isinstance(loaded, ClassifyResult)
        assert len(loaded.predictions) == len(original.predictions)

    def test_load_depth_result(self):
        """Loading a DepthResult JSON returns a DepthResult instance."""
        from mata.core.exporters.valkey_exporter import load_valkey
        from mata.core.types import DepthResult

        original = _make_depth_result()
        raw_json = original.to_json().encode()
        client = _mock_client()
        client.get.return_value = raw_json

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            loaded = load_valkey("valkey://localhost:6379", key="test:depth", result_type="depth")

        assert isinstance(loaded, DepthResult)
        assert loaded.depth.shape == original.depth.shape

    def test_load_ocr_result(self):
        """Loading an OCRResult JSON returns an OCRResult instance."""
        from mata.core.exporters.valkey_exporter import load_valkey
        from mata.core.types import OCRResult

        original = _make_ocr_result()
        raw_json = original.to_json().encode()
        client = _mock_client()
        client.get.return_value = raw_json

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            loaded = load_valkey("valkey://localhost:6379", key="test:ocr", result_type="ocr")

        assert isinstance(loaded, OCRResult)
        assert len(loaded.regions) == len(original.regions)

    def test_load_auto_detect_type(self):
        """Auto-detection picks the right type from serialized dict keys."""
        from mata.core.exporters.valkey_exporter import load_valkey
        from mata.core.types import VisionResult

        original = _make_vision_result()
        raw_json = original.to_json().encode()
        client = _mock_client()
        client.get.return_value = raw_json

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            # result_type="auto" is the default
            loaded = load_valkey("valkey://localhost:6379", key="test:auto")

        assert isinstance(loaded, VisionResult)

    def test_load_explicit_type(self):
        """Explicit result_type bypasses auto-detection."""
        from mata.core.exporters.valkey_exporter import load_valkey
        from mata.core.types import VisionResult

        original = _make_vision_result()
        raw_json = original.to_json().encode()
        client = _mock_client()
        client.get.return_value = raw_json

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            # Force "detect" which maps to VisionResult
            loaded = load_valkey("valkey://localhost:6379", key="test:explicit", result_type="detect")

        assert isinstance(loaded, VisionResult)

    def test_load_missing_key_raises(self):
        """A missing Valkey key raises KeyError with the key name in the message."""
        from mata.core.exporters.valkey_exporter import load_valkey

        client = _mock_client()
        client.get.return_value = None  # key doesn't exist

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            with pytest.raises(KeyError, match="missing_key"):
                load_valkey("valkey://localhost:6379", key="missing_key")

    def test_load_unknown_type_raises(self):
        """An unknown result_type string raises ValueError."""
        from mata.core.exporters.valkey_exporter import load_valkey

        # Use VisionResult data but ask for an unsupported type
        raw_json = json.dumps({"instances": [], "meta": {}}).encode()
        client = _mock_client()
        client.get.return_value = raw_json

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            with pytest.raises(ValueError, match="Unknown result_type"):
                load_valkey("valkey://localhost:6379", key="k", result_type="segment_panoptic")

    def test_roundtrip_vision_result(self):
        """VisionResult survives export → load round-trip intact."""
        from mata.core.exporters.valkey_exporter import export_valkey, load_valkey
        from mata.core.types import VisionResult

        original = _make_vision_result()
        stored_value: list[str] = []

        export_client = _mock_client()
        export_client.set.side_effect = lambda k, v: stored_value.append(v)

        load_client = _mock_client()

        with patch(
            "mata.core.exporters.valkey_exporter._get_valkey_client",
            return_value=export_client,
        ):
            export_valkey(original, url="valkey://localhost:6379", key="rt:vision")

        assert stored_value, "export_valkey must call client.set()"

        load_client.get.return_value = stored_value[0].encode() if isinstance(stored_value[0], str) else stored_value[0]

        with patch(
            "mata.core.exporters.valkey_exporter._get_valkey_client",
            return_value=load_client,
        ):
            loaded = load_valkey("valkey://localhost:6379", key="rt:vision")

        assert isinstance(loaded, VisionResult)
        assert len(loaded.instances) == len(original.instances)
        assert loaded.instances[0].score == pytest.approx(original.instances[0].score)
        assert loaded.instances[0].label == original.instances[0].label
        assert loaded.instances[0].label_name == original.instances[0].label_name

    def test_roundtrip_classify_result(self):
        """ClassifyResult survives export → load round-trip intact."""
        from mata.core.exporters.valkey_exporter import export_valkey, load_valkey
        from mata.core.types import ClassifyResult

        original = _make_classify_result()
        stored_value: list[str] = []

        export_client = _mock_client()
        export_client.set.side_effect = lambda k, v: stored_value.append(v)

        with patch(
            "mata.core.exporters.valkey_exporter._get_valkey_client",
            return_value=export_client,
        ):
            export_valkey(original, url="valkey://localhost:6379", key="rt:classify")

        assert stored_value

        load_client = _mock_client()
        load_client.get.return_value = stored_value[0].encode() if isinstance(stored_value[0], str) else stored_value[0]

        with patch(
            "mata.core.exporters.valkey_exporter._get_valkey_client",
            return_value=load_client,
        ):
            loaded = load_valkey("valkey://localhost:6379", key="rt:classify")

        assert isinstance(loaded, ClassifyResult)
        assert len(loaded.predictions) == len(original.predictions)
        assert loaded.predictions[0].score == pytest.approx(original.predictions[0].score)
        assert loaded.predictions[0].label_name == original.predictions[0].label_name


# ── TestParseValkeyURI ────────────────────────────────────────────────────────


class TestParseValkeyURI:
    """Tests for _parse_valkey_uri() helper."""

    def test_simple_uri(self):
        """Simple valkey://host:port/key parses correctly."""
        from mata.core.exporters.valkey_exporter import _parse_valkey_uri

        base_url, key = _parse_valkey_uri("valkey://localhost:6379/my_key")

        assert base_url == "valkey://localhost:6379"
        assert key == "my_key"

    def test_uri_with_db_number(self):
        """valkey://host:port/db/key parses DB number into base_url."""
        from mata.core.exporters.valkey_exporter import _parse_valkey_uri

        base_url, key = _parse_valkey_uri("valkey://localhost:6379/0/my_key")

        assert base_url == "valkey://localhost:6379/0"
        assert key == "my_key"

    def test_uri_with_password(self):
        """URI with credentials passes through without stripping the password."""
        from mata.core.exporters.valkey_exporter import _parse_valkey_uri

        base_url, key = _parse_valkey_uri("redis://user:pass@host:6379/0/my_key")

        assert base_url == "redis://user:pass@host:6379/0"
        assert key == "my_key"

    def test_redis_scheme(self):
        """redis:// scheme is parsed the same way as valkey://."""
        from mata.core.exporters.valkey_exporter import _parse_valkey_uri

        base_url, key = _parse_valkey_uri("redis://localhost:6379/detections")

        assert base_url == "redis://localhost:6379"
        assert key == "detections"

    def test_invalid_uri_raises(self):
        """A URI with no key component raises ValueError."""
        from mata.core.exporters.valkey_exporter import _parse_valkey_uri

        with pytest.raises(ValueError, match="Invalid Valkey URI"):
            _parse_valkey_uri("valkey://localhost:6379/")

    def test_empty_key_raises(self):
        """A URI without any path raises ValueError."""
        from mata.core.exporters.valkey_exporter import _parse_valkey_uri

        with pytest.raises(ValueError, match="Invalid Valkey URI"):
            _parse_valkey_uri("valkey://localhost:6379")

    def test_key_with_colon(self):
        """Keys containing colon separators are preserved intact."""
        from mata.core.exporters.valkey_exporter import _parse_valkey_uri

        base_url, key = _parse_valkey_uri("valkey://localhost:6379/pipeline:detections:frame_001")

        assert base_url == "valkey://localhost:6379"
        assert key == "pipeline:detections:frame_001"

    def test_db_number_with_key_containing_colon(self):
        """DB number + key with colon separators parses correctly."""
        from mata.core.exporters.valkey_exporter import _parse_valkey_uri

        base_url, key = _parse_valkey_uri("valkey://localhost:6379/2/ns:my_key")

        assert base_url == "valkey://localhost:6379/2"
        assert key == "ns:my_key"


# ── TestSaveValkeyIntegration ─────────────────────────────────────────────────


class TestSaveValkeyIntegration:
    """Tests for result.save('valkey://...') URI scheme dispatch."""

    def test_vision_result_save_valkey(self):
        """VisionResult.save() routes valkey:// URI to export_valkey."""
        result = _make_vision_result()
        with (
            patch("mata.core.exporters.valkey_exporter.export_valkey") as mock_export,
            patch("mata.core.exporters.valkey_exporter._get_valkey_client"),
        ):
            result.save("valkey://localhost:6379/vision_key")

        mock_export.assert_called_once()
        _, kwargs = mock_export.call_args
        assert kwargs.get("key") == "vision_key" or mock_export.call_args[0][0] is result

    def test_vision_result_save_valkey_dispatches(self):
        """Verify export_valkey receives correct url and key from VisionResult.save()."""
        result = _make_vision_result()
        client = _mock_client()

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            result.save("valkey://localhost:6379/my_vision_key")

        client.set.assert_called_once()
        stored_data = client.set.call_args[0]
        assert stored_data[0] == "my_vision_key"
        # Verify the stored content is valid JSON
        parsed = json.loads(stored_data[1])
        assert "instances" in parsed

    def test_classify_result_save_valkey(self):
        """ClassifyResult.save() routes valkey:// URI to export_valkey."""
        result = _make_classify_result()
        client = _mock_client()

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            result.save("valkey://localhost:6379/classify_key")

        client.set.assert_called_once()
        stored_key = client.set.call_args[0][0]
        assert stored_key == "classify_key"

    def test_depth_result_save_valkey(self):
        """DepthResult.save() routes valkey:// URI to export_valkey."""
        result = _make_depth_result()
        client = _mock_client()

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            result.save("valkey://localhost:6379/depth_key")

        client.set.assert_called_once()
        stored_key = client.set.call_args[0][0]
        assert stored_key == "depth_key"

    def test_ocr_result_save_valkey(self):
        """OCRResult.save() routes valkey:// URI to export_valkey."""
        result = _make_ocr_result()
        client = _mock_client()

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            result.save("valkey://localhost:6379/ocr_key")

        client.set.assert_called_once()
        stored_key = client.set.call_args[0][0]
        assert stored_key == "ocr_key"

    def test_detect_result_save_valkey(self):
        """DetectResult.save() routes valkey:// URI to export_valkey."""
        from mata.core.types import Detection, DetectResult

        result = DetectResult(
            detections=[Detection(bbox=(0, 0, 10, 10), score=0.8, label=0)],
            meta={},
        )
        client = _mock_client()

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            result.save("valkey://localhost:6379/detect_key")

        client.set.assert_called_once()
        stored_key = client.set.call_args[0][0]
        assert stored_key == "detect_key"

    def test_segment_result_save_valkey(self):
        """SegmentResult.save() routes valkey:// URI to export_valkey."""
        import numpy as np

        from mata.core.types import SegmentMask, SegmentResult

        mask_arr = np.zeros((8, 8), dtype=bool)
        result = SegmentResult(
            masks=[SegmentMask(mask=mask_arr, score=0.85, label=0)],
            meta={},
        )
        client = _mock_client()

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            result.save("valkey://localhost:6379/segment_key")

        client.set.assert_called_once()
        stored_key = client.set.call_args[0][0]
        assert stored_key == "segment_key"

    def test_redis_scheme_also_dispatches(self):
        """redis:// scheme is also routed to export_valkey (not file export)."""
        result = _make_classify_result()
        client = _mock_client()

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            result.save("redis://localhost:6379/redis_key")

        client.set.assert_called_once()

    def test_save_json_still_works(self, tmp_path):
        """file.json save is unaffected — no regression."""
        from mata.core.exporters import export_json

        result = _make_vision_result()
        output = tmp_path / "detections.json"

        with patch("mata.core.exporters.json_exporter.export_json", wraps=export_json) as _spy:
            result.save(str(output))

        assert output.exists()
        data = json.loads(output.read_text())
        assert "instances" in data

    def test_save_csv_still_works(self, tmp_path):
        """file.csv save is unaffected — no regression."""
        result = _make_vision_result()
        output = tmp_path / "detections.csv"

        result.save(str(output))

        assert output.exists()
        content = output.read_text()
        assert len(content) > 0

    def test_save_image_still_works(self, tmp_path):
        """Image path does not get mistakenly routed to valkey exporter."""
        import numpy as np

        result = _make_vision_result()
        # Provide a dummy image so export_image doesn't fail
        dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        output = tmp_path / "overlay.png"

        # Only check that export_valkey is NOT called for a .png save
        with patch("mata.core.exporters.valkey_exporter.export_valkey") as mock_export_valkey:
            result.save(str(output), image=dummy_img)

        mock_export_valkey.assert_not_called()

    def test_save_ttl_forwarded_via_save(self):
        """Extra kwargs (e.g., ttl) are forwarded from result.save() to export_valkey."""
        result = _make_vision_result()
        client = _mock_client()

        with patch("mata.core.exporters.valkey_exporter._get_valkey_client", return_value=client):
            result.save("valkey://localhost:6379/ttl_key", ttl=7200)

        client.setex.assert_called_once()
        args = client.setex.call_args[0]
        assert args[0] == "ttl_key"
        assert args[1] == 7200

    def test_importable_without_valkey_installed(self):
        """export_valkey is importable even when valkey-py is not installed."""
        # This test simply calls import — the real guard is at call time, not import time
        with patch.dict(sys.modules, {"valkey": None, "redis": None}):
            # Re-execute the import to confirm no top-level ImportError
            import importlib

            import mata.core.exporters.valkey_exporter as mod

            importlib.reload(mod)
            # Should still be callable (just raises at execution time)
            assert callable(mod.export_valkey)
