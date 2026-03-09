"""Valkey/Redis exporter for MATA result types.

Exports any result with a to_dict()/to_json() interface to a Valkey key.
Supports TTL, JSON serialization, and optional msgpack for binary efficiency.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from mata.core.logging import get_logger

if TYPE_CHECKING:
    from mata.core.types import ClassifyResult, DepthResult, OCRResult, VisionResult

logger = get_logger(__name__)


def _parse_valkey_uri(uri: str) -> tuple[str, str]:
    """Parse a Valkey/Redis URI into (base_url, key).

    Handles the following formats:
      - valkey://host:port/key_name
      - valkey://host:port/0/key_name       (with DB number)
      - redis://user:pass@host:port/0/key   (with credentials)

    Args:
        uri: Full Valkey/Redis URI string

    Returns:
        Tuple of (base_url, key)

    Raises:
        ValueError: If URI format is invalid or key is missing
    """
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    path_parts = parsed.path.strip("/").split("/", 1)

    if len(path_parts) == 2 and path_parts[0].isdigit():
        # Has DB number: valkey://host:port/0/key
        db = path_parts[0]
        key = path_parts[1]
        base_url = f"{parsed.scheme}://{parsed.netloc}/{db}"
    elif len(path_parts) == 1 and path_parts[0]:
        key = path_parts[0]
        base_url = f"{parsed.scheme}://{parsed.netloc}"
    else:
        raise ValueError(
            f"Invalid Valkey URI: '{uri}'. Expected format: valkey://host:port/key " "or valkey://host:port/db/key"
        )

    return base_url, key


def _get_valkey_client(url: str, **kwargs: Any):
    """Lazy-import and connect to Valkey/Redis.

    Tries valkey-py first, falls back to redis-py for compatibility.

    Args:
        url: Valkey connection URL (e.g., "valkey://localhost:6379/0")
        **kwargs: Additional connection parameters

    Returns:
        Connected Valkey/Redis client instance

    Raises:
        ImportError: If neither valkey-py nor redis-py is installed
    """
    try:
        import valkey

        return valkey.from_url(url, **kwargs)
    except ImportError:
        try:
            import redis

            # valkey:// scheme → redis:// for redis-py compatibility
            redis_url = url.replace("valkey://", "redis://", 1)
            return redis.from_url(redis_url, **kwargs)
        except ImportError:
            raise ImportError(
                "Valkey export requires 'valkey' or 'redis' package. "
                "Install with: pip install datamata[valkey] or pip install datamata[redis]"
            )


def export_valkey(
    result: VisionResult | ClassifyResult | DepthResult | OCRResult,
    url: str,
    key: str,
    ttl: int | None = None,
    serializer: str = "json",
    **kwargs: Any,
) -> None:
    """Export result to a Valkey/Redis key.

    Args:
        result: Any MATA result object with to_dict()/to_json()
        url: Valkey connection URL
        key: Key name to store under
        ttl: Time-to-live in seconds (None = no expiry)
        serializer: "json" (default) or "msgpack"
        **kwargs: Additional connection parameters

    Raises:
        ImportError: If valkey/redis client not installed
        ConnectionError: If Valkey server unreachable
    """
    client = _get_valkey_client(url, **kwargs)

    if serializer == "json":
        data = result.to_json()
    elif serializer == "msgpack":
        import msgpack

        data = msgpack.packb(result.to_dict(), use_bin_type=True)
    else:
        raise ValueError(f"Unsupported serializer: '{serializer}'. Use 'json' or 'msgpack'.")

    if ttl is not None:
        client.setex(key, ttl, data)
    else:
        client.set(key, data)

    logger.info(f"Exported result to Valkey key '{key}' (ttl={ttl})")


def load_valkey(
    url: str,
    key: str,
    result_type: str = "auto",
    **kwargs: Any,
) -> VisionResult | ClassifyResult | DepthResult | OCRResult:
    """Load a MATA result from a Valkey/Redis key.

    Args:
        url: Valkey connection URL
        key: Key name to load from
        result_type: "auto" (detect from data), "vision", "classify", "depth", "ocr"
        **kwargs: Additional connection parameters

    Returns:
        Reconstructed result object

    Raises:
        KeyError: If key does not exist
        ImportError: If valkey/redis client not installed
    """
    client = _get_valkey_client(url, **kwargs)
    raw = client.get(key)

    if raw is None:
        raise KeyError(f"Valkey key '{key}' not found")

    data = json.loads(raw)

    if result_type == "auto":
        result_type = _detect_result_type(data)

    return _deserialize_result(data, result_type)


def _detect_result_type(data: dict) -> str:
    """Auto-detect result type from serialized dict keys."""
    if "instances" in data:
        return "vision"
    elif "predictions" in data:
        return "classify"
    elif "depth" in data:
        return "depth"
    elif "regions" in data:
        return "ocr"
    else:
        raise ValueError(
            f"Cannot auto-detect result type from keys: {list(data.keys())}. " "Specify result_type explicitly."
        )


def _deserialize_result(data: dict, result_type: str):
    """Reconstruct a typed result from dict."""
    from mata.core.types import ClassifyResult, DepthResult, OCRResult, VisionResult

    type_map = {
        "vision": VisionResult,
        "detect": VisionResult,
        "classify": ClassifyResult,
        "depth": DepthResult,
        "ocr": OCRResult,
    }

    cls = type_map.get(result_type)
    if cls is None:
        raise ValueError(f"Unknown result_type: '{result_type}'. Use: {list(type_map.keys())}")

    return cls.from_dict(data)


def publish_valkey(
    result: VisionResult | ClassifyResult | DepthResult | OCRResult,
    url: str,
    channel: str,
    serializer: str = "json",
    **kwargs: Any,
) -> int:
    """Publish result to a Valkey Pub/Sub channel.

    This is a fire-and-forget operation. Messages are delivered only to
    active subscribers and are NOT persisted — if no subscriber is listening
    when ``publish_valkey`` is called, the message is silently dropped.

    Channel names should never be derived from user-controlled input without
    prior validation, as they are passed directly to the Valkey server.

    Args:
        result: Any MATA result object with to_dict()/to_json()
        url: Valkey connection URL (e.g., "valkey://localhost:6379")
        channel: Pub/Sub channel name to publish to
        serializer: "json" (default) or "msgpack"
        **kwargs: Additional connection parameters

    Returns:
        Number of subscribers that received the message

    Raises:
        ImportError: If valkey/redis client not installed
        ValueError: If an unsupported serializer is specified
    """
    client = _get_valkey_client(url, **kwargs)

    if serializer == "json":
        data = result.to_json()
    elif serializer == "msgpack":
        import msgpack

        data = msgpack.packb(result.to_dict(), use_bin_type=True)
    else:
        raise ValueError(f"Unsupported serializer: '{serializer}'. Use 'json' or 'msgpack'.")

    num_receivers = client.publish(channel, data)
    logger.info(f"Published result to channel '{channel}' ({num_receivers} subscribers)")
    return num_receivers
