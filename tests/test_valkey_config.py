"""Tests for Valkey config and Pub/Sub — Task C2, D1."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from mata.core.exceptions import ModelNotFoundError
from mata.core.model_registry import ModelRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(yaml_content: str) -> ModelRegistry:
    """Create a ModelRegistry backed by a temporary YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        fh.write(yaml_content)
        tmp_path = fh.name
    return ModelRegistry(custom_config_path=tmp_path)


def _minimal_vision_result():
    """Return a minimal mock VisionResult."""
    mock = MagicMock()
    mock.to_json.return_value = '{"instances": []}'
    mock.to_dict.return_value = {"instances": []}
    return mock


def _minimal_classify_result():
    mock = MagicMock()
    mock.to_json.return_value = '{"predictions": []}'
    mock.to_dict.return_value = {"predictions": []}
    return mock


# ---------------------------------------------------------------------------
# TestValkeyConfig
# ---------------------------------------------------------------------------


class TestValkeyConfig:
    """Tests for YAML config integration (Task C2)."""

    BASIC_YAML = """\
storage:
  valkey:
    default:
      url: "valkey://localhost:6379"
      db: 0
      ttl: 3600
    production:
      url: "valkey://prod-cluster:6380"
      db: 1
"""

    def test_get_default_connection(self):
        """get_valkey_connection('default') returns the default connection dict."""
        registry = _make_registry(self.BASIC_YAML)
        conn = registry.get_valkey_connection("default")

        assert conn["url"] == "valkey://localhost:6379"
        assert conn["db"] == 0
        assert conn["ttl"] == 3600

    def test_get_named_connection(self):
        """get_valkey_connection('production') returns non-default connection."""
        registry = _make_registry(self.BASIC_YAML)
        conn = registry.get_valkey_connection("production")

        assert conn["url"] == "valkey://prod-cluster:6380"
        assert conn["db"] == 1

    def test_missing_connection_raises(self):
        """Unknown connection name raises ModelNotFoundError with helpful message."""
        registry = _make_registry(self.BASIC_YAML)

        with pytest.raises(ModelNotFoundError, match="nonexistent"):
            registry.get_valkey_connection("nonexistent")

    def test_missing_connection_lists_available(self):
        """ModelNotFoundError message lists the available connection names."""
        registry = _make_registry(self.BASIC_YAML)

        with pytest.raises(ModelNotFoundError) as exc_info:
            registry.get_valkey_connection("missing")

        error_msg = str(exc_info.value)
        # Error should list at least one of the existing connection names
        assert "default" in error_msg or "production" in error_msg

    def test_password_env_resolved(self, monkeypatch):
        """password_env is resolved from the environment and replaced by 'password'."""
        yaml_content = """\
storage:
  valkey:
    default:
      url: "valkey://secure-host:6379"
      password_env: "TEST_VALKEY_PASS_E3"
"""
        monkeypatch.setenv("TEST_VALKEY_PASS_E3", "s3cr3t")
        registry = _make_registry(yaml_content)
        conn = registry.get_valkey_connection("default")

        assert conn.get("password") == "s3cr3t"
        assert "password_env" not in conn

    def test_password_env_missing_graceful(self, monkeypatch):
        """Missing env var does not raise; 'password' key is simply absent."""
        yaml_content = """\
storage:
  valkey:
    default:
      url: "valkey://secure-host:6379"
      password_env: "TEST_VALKEY_UNDEFINED_ENV_VAR_XYZ"
"""
        monkeypatch.delenv("TEST_VALKEY_UNDEFINED_ENV_VAR_XYZ", raising=False)
        registry = _make_registry(yaml_content)
        conn = registry.get_valkey_connection("default")

        assert "password_env" not in conn
        assert "password" not in conn

    def test_no_storage_section_graceful(self):
        """A config file with no 'storage' section is handled without error."""
        yaml_content = """\
models:
  detect:
    rtdetr-fast:
      source: "facebook/detr-resnet-50"
"""
        registry = _make_registry(yaml_content)

        with pytest.raises(ModelNotFoundError):
            registry.get_valkey_connection("default")

    def test_existing_models_config_unaffected(self):
        """Adding a storage section does not alter the existing models section."""
        yaml_content = """\
detect:
  my-model:
    source: "facebook/detr-resnet-50"
    threshold: 0.5

storage:
  valkey:
    default:
      url: "valkey://localhost:6379"
"""
        registry = _make_registry(yaml_content)

        # models section still works
        assert registry.has_alias("detect", "my-model")
        config = registry.get_config("detect", "my-model")
        assert config["source"] == "facebook/detr-resnet-50"

        # storage section also works
        conn = registry.get_valkey_connection("default")
        assert conn["url"] == "valkey://localhost:6379"

    def test_tls_flag_passthrough(self):
        """tls flag in YAML is returned in the connection dict."""
        yaml_content = """\
storage:
  valkey:
    secure:
      url: "valkey://tls-host:6380"
      tls: true
      db: 0
"""
        registry = _make_registry(yaml_content)
        conn = registry.get_valkey_connection("secure")

        assert conn.get("tls") is True


# ---------------------------------------------------------------------------
# TestPublishValkey
# ---------------------------------------------------------------------------


class TestPublishValkey:
    """Tests for publish_valkey() Pub/Sub function (Task D1)."""

    def test_publish_vision_result(self):
        """publish_valkey() calls client.publish() with a VisionResult."""
        from mata.core.exporters.valkey_exporter import publish_valkey

        result = _minimal_vision_result()
        mock_client = MagicMock()
        mock_client.publish.return_value = 3

        with patch(
            "mata.core.exporters.valkey_exporter._get_valkey_client",
            return_value=mock_client,
        ):
            count = publish_valkey(result, url="valkey://localhost:6379", channel="detections")

        mock_client.publish.assert_called_once()
        call_args = mock_client.publish.call_args
        assert call_args[0][0] == "detections"
        assert count == 3

    def test_publish_returns_subscriber_count(self):
        """publish_valkey() returns the integer subscriber count from client.publish()."""
        from mata.core.exporters.valkey_exporter import publish_valkey

        result = _minimal_vision_result()
        mock_client = MagicMock()
        mock_client.publish.return_value = 7

        with patch(
            "mata.core.exporters.valkey_exporter._get_valkey_client",
            return_value=mock_client,
        ):
            count = publish_valkey(result, url="valkey://localhost:6379", channel="ch")

        assert count == 7

    def test_publish_json_serializer(self):
        """publish_valkey() with serializer='json' calls result.to_json()."""
        from mata.core.exporters.valkey_exporter import publish_valkey

        result = _minimal_vision_result()
        mock_client = MagicMock()
        mock_client.publish.return_value = 1

        with patch(
            "mata.core.exporters.valkey_exporter._get_valkey_client",
            return_value=mock_client,
        ):
            publish_valkey(result, url="valkey://localhost:6379", channel="ch", serializer="json")

        result.to_json.assert_called_once()
        result.to_dict.assert_not_called()

    def test_publish_msgpack_serializer(self):
        """publish_valkey() with serializer='msgpack' calls result.to_dict() and packs."""
        from mata.core.exporters.valkey_exporter import publish_valkey

        result = _minimal_classify_result()
        mock_client = MagicMock()
        mock_client.publish.return_value = 2

        msgpack_mock = MagicMock()
        msgpack_mock.packb.return_value = b"\x81\xabpredictions\x90"

        with patch(
            "mata.core.exporters.valkey_exporter._get_valkey_client",
            return_value=mock_client,
        ):
            with patch.dict("sys.modules", {"msgpack": msgpack_mock}):
                publish_valkey(
                    result,
                    url="valkey://localhost:6379",
                    channel="ch",
                    serializer="msgpack",
                )

        result.to_dict.assert_called_once()
        msgpack_mock.packb.assert_called_once_with(result.to_dict.return_value, use_bin_type=True)
        mock_client.publish.assert_called_once_with("ch", b"\x81\xabpredictions\x90")

    def test_publish_invalid_serializer_raises(self):
        """publish_valkey() raises ValueError for an unsupported serializer."""
        from mata.core.exporters.valkey_exporter import publish_valkey

        result = _minimal_vision_result()
        mock_client = MagicMock()

        with patch(
            "mata.core.exporters.valkey_exporter._get_valkey_client",
            return_value=mock_client,
        ):
            with pytest.raises(ValueError, match="Unsupported serializer"):
                publish_valkey(
                    result,
                    url="valkey://localhost:6379",
                    channel="ch",
                    serializer="pickle",
                )
