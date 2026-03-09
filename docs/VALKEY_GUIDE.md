# MATA Valkey/Redis Integration Guide

**Version**: 1.9.2  
**Last Updated**: March 9, 2026  
**Status**: ✅ Production Ready

---

## Table of Contents

1. [Installation & Setup](#1-installation--setup)
2. [Basic Usage (save/load)](#2-basic-usage-saveload)
3. [Graph Pipeline Integration](#3-graph-pipeline-integration)
4. [YAML Configuration (Named Connections)](#4-yaml-configuration-named-connections)
5. [Streaming Patterns (Per-Frame Tracking)](#5-streaming-patterns-per-frame-tracking)
6. [Pub/Sub Event-Driven Architecture](#6-pubsub-event-driven-architecture)
7. [Security](#7-security)
8. [Performance Tuning](#8-performance-tuning)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Installation & Setup

### Install the client library

MATA supports [Valkey](https://valkey.io/) (the open-source Redis fork) and Redis via optional extras:

```bash
pip install datamata[valkey]   # valkey-py >= 6.0.0 (recommended)
pip install datamata[redis]    # redis-py >= 5.0.0  (alternative, wire-compatible)
```

Both clients are optional — `import mata` succeeds without either installed. An `ImportError` with an actionable message is raised only when a storage operation is actually executed.

### Start a local Valkey server

The quickest way to get a local server running is with Docker:

```bash
docker run -d --name valkey-server -p 6379:6379 valkey/valkey:latest
```

Or install natively:

```bash
# macOS
brew install valkey

# Ubuntu / Debian
apt-get install valkey-server
```

Verify connectivity:

```bash
valkey-cli ping   # → PONG
# or
redis-cli ping    # works against Valkey too
```

### Supported URI schemes

| Scheme                              | Client    | Notes                    |
| ----------------------------------- | --------- | ------------------------ |
| `valkey://host:port/key`            | valkey-py | Recommended              |
| `valkey://host:port/0/key`          | valkey-py | With DB number           |
| `redis://host:port/key`             | redis-py  | Wire-compatible fallback |
| `redis://user:pass@host:port/0/key` | redis-py  | With credentials         |
| `rediss://host:port/key`            | redis-py  | TLS-encrypted connection |

---

## 2. Basic Usage (save/load)

### Save any result to Valkey

All MATA result types (`VisionResult`, `ClassifyResult`, `DepthResult`, `OCRResult`, `DetectResult`, `SegmentResult`) support a `valkey://` URI directly in their `save()` method:

```python
import mata

# Run detection
result = mata.run("detect", "photo.jpg", model="PekingU/rtdetr_r18vd", threshold=0.4)

# Save to Valkey — same API as saving to a file
result.save("valkey://localhost:6379/detections:frame_001")

# With a TTL (expires after 5 minutes)
result.save("valkey://localhost:6379/detections:latest", ttl=300)

# With a DB number
result.save("valkey://localhost:6379/1/detections:frame_001")
```

The same pattern works for all tasks:

```python
depth = mata.run("depth", "scene.jpg", model="depth-anything/Depth-Anything-V2-Small-hf")
depth.save("valkey://localhost:6379/depth:latest")

classes = mata.run("classify", "cat.jpg", model="microsoft/resnet-50")
classes.save("valkey://localhost:6379/classify:latest")

text = mata.run("ocr", "scan.png", model="ucaslcl/GOT-OCR2_0")
text.save("valkey://localhost:6379/ocr:document_001")
```

### Load a result back

```python
from mata.core.exporters import load_valkey

# Auto-detect result type from stored data
result = load_valkey(url="valkey://localhost:6379", key="detections:frame_001")
print(type(result))   # <class 'mata.core.types.VisionResult'>

# Explicit result type (faster, skips auto-detection)
result = load_valkey(
    url="valkey://localhost:6379",
    key="detections:frame_001",
    result_type="vision",   # "vision", "classify", "depth", "ocr"
)
```

### Low-level exporter API

For more control, use the exporter functions directly:

```python
from mata.core.exporters import export_valkey, load_valkey

# Export with all options
export_valkey(
    result=detection_result,
    url="valkey://localhost:6379",
    key="pipeline:output",
    ttl=3600,           # 1 hour TTL
    serializer="json",  # "json" (default) or "msgpack"
)

# Load back
loaded = load_valkey(url="valkey://localhost:6379", key="pipeline:output")
```

### Round-trip example

```python
import mata
from mata.core.exporters import export_valkey, load_valkey

# Step 1: run inference
result = mata.run("detect", "image.jpg", model="PekingU/rtdetr_r18vd")

# Step 2: persist
export_valkey(result, url="valkey://localhost:6379", key="my:result", ttl=600)

# Step 3: load in another process / service
loaded = load_valkey(url="valkey://localhost:6379", key="my:result")

# Results are equivalent
assert len(loaded.instances) == len(result.instances)
```

---

## 3. Graph Pipeline Integration

### `ValkeyStore` — sink node

`ValkeyStore` writes an artifact to Valkey during graph execution and passes it through unchanged, so downstream nodes can still consume it.

```python
import mata
from mata.nodes import Detect, Filter, ValkeyStore
from mata.core.graph import Graph

detector = mata.load("detect", "PekingU/rtdetr_r18vd")

graph = (
    Graph()
    .then(Detect(using="detr", out="dets"))
    .then(Filter(src="dets", score_gt=0.4, out="filtered"))
    .then(ValkeyStore(
        src="filtered",
        url="valkey://localhost:6379",
        key="pipeline:detections:{timestamp}",   # {timestamp} is Unix epoch
        ttl=3600,
    ))
    # Downstream nodes still see "filtered" — ValkeyStore is a pass-through
)

result = mata.infer("frame.jpg", graph=graph, providers={"detr": detector})
print(result.filtered)   # still accessible after store
```

**Key template placeholders:**

| Placeholder   | Resolved value                            |
| ------------- | ----------------------------------------- |
| `{node}`      | Node's `name` attribute (`"ValkeyStore"`) |
| `{timestamp}` | Unix epoch (integer seconds at run time)  |

Only these two placeholders are supported — user data is never interpolated.

### `ValkeyLoad` — source node

`ValkeyLoad` loads a stored result from Valkey and injects it as the first artifact in a graph. Use this to build pipelines that consume results produced by another service.

```python
from mata.nodes import ValkeyLoad, Filter, Fuse
from mata.core.graph import Graph

graph = (
    Graph()
    .then(ValkeyLoad(
        url="valkey://localhost:6379",
        key="upstream:detections:latest",
        result_type="vision",   # "auto" also works
        out="dets",
    ))
    .then(Filter(src="dets", score_gt=0.6, out="hi_conf"))
    .then(Fuse(detections="hi_conf", out="annotated"))
)

result = mata.infer("frame.jpg", graph=graph, providers={})
```

### Complete cross-pipeline example

This pattern enables two independent services to share detection results through Valkey:

```python
import mata
from mata.nodes import Detect, Filter, ValkeyStore, ValkeyLoad, Fuse
from mata.core.graph import Graph

detector = mata.load("detect", "PekingU/rtdetr_r18vd")

# ── Service A: Camera ingestion ──────────────────────────────────────────
store_graph = (
    Graph()
    .then(Detect(using="detr", out="dets"))
    .then(Filter(src="dets", score_gt=0.3, out="filtered"))
    .then(ValkeyStore(
        src="filtered",
        url="valkey://prod-cluster:6379",
        key="cam01:detections:latest",
        ttl=10,   # fresh for 10 s; overwritten each frame
    ))
)

for frame in camera_frames():
    mata.infer(frame, graph=store_graph, providers={"detr": detector})


# ── Service B: Downstream analytics (separate process / machine) ─────────
load_graph = (
    Graph()
    .then(ValkeyLoad(
        url="valkey://prod-cluster:6379",
        key="cam01:detections:latest",
        out="dets",
    ))
    .then(Filter(src="dets", score_gt=0.7, out="hi_conf"))
    .then(Fuse(detections="hi_conf", out="annotated"))
)

annotated = mata.infer(latest_frame, graph=load_graph, providers={})
```

---

## 4. YAML Configuration (Named Connections)

Instead of hard-coding URLs in your code, define named connection profiles in `.mata/models.yaml` (project-local) or `~/.mata/models.yaml` (user-global):

```yaml
# .mata/models.yaml
models:
  detect:
    rtdetr-fast:
      source: "PekingU/rtdetr_r18vd"
      threshold: 0.4

# Storage section — new in v1.9.0
storage:
  valkey:
    default:
      url: "valkey://localhost:6379"
      db: 0
      ttl: 3600

    staging:
      url: "valkey://staging-host:6379"
      db: 1
      ttl: 600

    production:
      url: "valkey://prod-cluster:6379"
      password_env: "VALKEY_PASSWORD" # ← resolved from environment variable
      db: 0
      tls: true
      ttl: 86400
```

### Retrieve a connection profile

```python
from mata.core.model_registry import ModelRegistry

registry = ModelRegistry()

# Get the default connection
conn = registry.get_valkey_connection()            # name="default"

# Get a named connection
conn = registry.get_valkey_connection("production")

# conn is a plain dict — pass to export_valkey as **kwargs
# { "url": "valkey://prod-cluster:6379", "password": "<from env>", "tls": True, "ttl": 86400 }
```

### Password management with `password_env`

**Never store passwords in YAML.** Use the `password_env` key to reference an environment variable:

```yaml
production:
  url: "valkey://prod-cluster:6379"
  password_env: "VALKEY_PASSWORD" # the env-var NAME, not the value
```

At runtime, `ModelRegistry.get_valkey_connection()` resolves `os.environ["VALKEY_PASSWORD"]` and replaces `password_env` with `password` in the returned dict. If the variable is not set, the `password` key is omitted entirely (no error).

```bash
# Set before running your application
export VALKEY_PASSWORD="s3cr3tP@ssw0rd"
```

The env-var name itself is never logged or returned — only the resolved password is passed to the client.

---

## 5. Streaming Patterns (Per-Frame Tracking)

For real-time video pipelines, write each frame's tracking results to Valkey with a rolling TTL so the key always holds the latest state:

```python
import mata
import time

tracker = mata.load("track", "PekingU/rtdetr_r18vd", tracker="botsort")

VALKEY_URL = "valkey://localhost:6379"
ROLLING_KEY = "track:cam01:latest"
TTL = 5   # seconds; overwritten each frame, auto-expires if feed drops

cap = cv2.VideoCapture("rtsp://camera/stream")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = tracker.update(frame, persist=True)

    # Overwrite the rolling key — downstream consumers always read "latest"
    result.save(f"{VALKEY_URL}/{ROLLING_KEY}", ttl=TTL)

cap.release()
```

### Per-frame keyed history

For audit trails, write each frame to a unique key using the frame index or timestamp:

```python
for frame_idx, frame in enumerate(video_frames("recording.mp4")):
    result = mata.run("detect", frame, model="PekingU/rtdetr_r18vd")
    result.save(f"valkey://localhost:6379/recording:frame:{frame_idx:06d}", ttl=3600)
```

Retrieve a specific frame later:

```python
from mata.core.exporters import load_valkey

frame_result = load_valkey(
    url="valkey://localhost:6379",
    key="recording:frame:000042",
)
```

### Using `ValkeyStore` in `mata.track()` stream mode

```python
import mata
from mata.nodes import ValkeyStore
from mata.core.graph import Graph

# Build a small post-processing graph around each tracking result
store_node = ValkeyStore(
    src="tracked",
    url="valkey://localhost:6379",
    key="track:{timestamp}",
    ttl=30,
)

for result in mata.track(
    "rtsp://cam/stream",
    model="PekingU/rtdetr_r18vd",
    tracker="bytetrack",
    stream=True,
):
    # Export directly from the result object in stream mode
    result.save("valkey://localhost:6379/track:stream:latest", ttl=5)
```

---

## 6. Pub/Sub Event-Driven Architecture

Use `publish_valkey()` to broadcast results to real-time subscribers without holding state. Messages are fire-and-forget: if no subscriber is listening, the message is dropped.

```python
from mata.core.exporters import publish_valkey

result = mata.run("detect", "frame.jpg", model="PekingU/rtdetr_r18vd")

n = publish_valkey(
    result=result,
    url="valkey://localhost:6379",
    channel="detections:stream",   # Pub/Sub channel name
)
print(f"Delivered to {n} subscriber(s)")
```

### Subscriber (separate process)

```python
import valkey
import json

client = valkey.from_url("valkey://localhost:6379")
pubsub = client.pubsub()
pubsub.subscribe("detections:stream")

for message in pubsub.listen():
    if message["type"] != "message":
        continue

    data = json.loads(message["data"])
    # Reconstruct the result object if needed
    from mata.core.exporters.valkey_exporter import _deserialize_result, _detect_result_type
    result_type = _detect_result_type(data)
    result = _deserialize_result(data, result_type)
    process(result)
```

### Event-driven pipeline with Pub/Sub and ValkeyStore combined

For patterns where you need both real-time notifications **and** persistent storage:

```python
from mata.core.exporters import export_valkey, publish_valkey

result = mata.run("detect", "frame.jpg", model="PekingU/rtdetr_r18vd")

# Persist for later retrieval (with TTL)
export_valkey(result, url="valkey://localhost:6379", key="det:latest", ttl=60)

# Broadcast to live subscribers
publish_valkey(result, url="valkey://localhost:6379", channel="det:events")
```

**Channel naming guidelines:**

- Use hierarchical names separated by `:` — e.g., `cam01:detections`, `pipeline:alerts`
- Do not interpolate user-controlled strings directly into channel names without sanitization
- Pattern subscriptions (`PSUBSCRIBE cam*:detections`) work natively with Valkey pub/sub

---

## 7. Security

### TLS connections

For production connections over untrusted networks, use TLS. With `redis-py` you can use the `rediss://` scheme (note the double `s`):

```python
result.save("rediss://prod-host:6380/detections:latest")
```

With a named config:

```yaml
production:
  url: "valkey://prod-host:6380"
  tls: true
  password_env: "VALKEY_PASSWORD"
```

The `tls: true` flag is passed through to the client as `ssl=True` when calling `from_url()`.

### Credentials — never log, never hard-code

**Never put passwords in:**

- Source code
- YAML config files (use `password_env` instead)
- Log messages

MATA enforces this in the exporter layer: the raw connection URL is never passed to the logger. If the URL contains a password segment (e.g., `valkey://user:pass@host/key`), only the key name is logged, not the URL.

Bad practice:

```python
# ❌ Hard-coded credentials in source
result.save("valkey://admin:secret@host:6379/key")
```

Correct practice:

```python
# ✅ Credentials from environment variable via named connection
import os
from mata.core.model_registry import ModelRegistry

registry = ModelRegistry()
conn = registry.get_valkey_connection("production")
# The password came from os.environ["VALKEY_PASSWORD"] — never written in code
result.save(conn["url"] + "/my_key", password=conn.get("password"))
```

### SSRF prevention

When your application exposes an API that accepts Valkey URIs from users, validate them before use:

```python
import ipaddress
from urllib.parse import urlparse

ALLOWED_VALKEY_HOSTS = {"valkey-internal", "localhost", "127.0.0.1"}

def safe_valkey_uri(uri: str) -> str:
    """Validate that a Valkey URI points to an allowed host."""
    parsed = urlparse(uri)
    host = parsed.hostname or ""

    # Reject private/loopback IPs if they're not explicitly allowed
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private and host not in ALLOWED_VALKEY_HOSTS:
            raise ValueError(f"Disallowed Valkey host: {host!r}")
    except ValueError:
        pass   # not an IP address — proceed

    if host not in ALLOWED_VALKEY_HOSTS:
        raise ValueError(f"Valkey host '{host}' not in allowlist")

    return uri
```

Never pass externally-supplied URLs to `export_valkey()` or `load_valkey()` without validation.

### Key name sanitization

Key names derived from user input should be sanitized to prevent overwriting critical keys:

```python
import re

def safe_key(user_input: str, prefix: str = "user") -> str:
    """Sanitize a user-supplied string for use as a Valkey key segment."""
    # Allow only alphanumeric, hyphens, underscores
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_input)
    return f"{prefix}:{clean}"
```

---

## 8. Performance Tuning

### Serializer choice: JSON vs msgpack

|             | `json` (default)           | `msgpack`                                 |
| ----------- | -------------------------- | ----------------------------------------- |
| Format      | UTF-8 text                 | binary                                    |
| Size        | larger (Base64 for arrays) | ~30–60% smaller for numeric arrays        |
| Speed       | fast                       | faster for large payloads                 |
| Dependency  | stdlib                     | `pip install msgpack`                     |
| Readability | human-readable             | binary (use `msgpack.unpackb` to inspect) |

Use `msgpack` for high-throughput tracking pipelines with dense bounding-box arrays:

```python
export_valkey(
    result,
    url="valkey://localhost:6379",
    key="track:latest",
    serializer="msgpack",
    ttl=10,
)
```

> **Note:** `DepthResult` stores a `(H, W)` float array that can be several MB as JSON. Use `msgpack` or consider downsampling the depth map before storing.

### TTL strategies

| Scenario                      | Recommended TTL                                                 |
| ----------------------------- | --------------------------------------------------------------- |
| Rolling latest frame          | 5–30 s (overwritten each frame)                                 |
| Frame history / audit trail   | 1–24 h (depends on retention needs)                             |
| Cross-pipeline handoff        | 60–300 s (long enough for downstream to pick up)                |
| Development / debugging       | `None` (inspect keys with `valkey-cli`)                         |
| Production with memory budget | 10–60 min + set `maxmemory-policy allkeys-lru` in Valkey config |

### Connection pooling

`valkey-py` and `redis-py` both maintain an internal connection pool. If you're calling `export_valkey()` in a tight loop, reuse the client instead of reconnecting each call:

```python
import valkey

client = valkey.from_url("valkey://localhost:6379", max_connections=10)

for result in results:
    data = result.to_json()
    client.setex(f"frame:{i}", 30, data)
```

For the public API, you can pass a pre-built client via `**kwargs` if the exporter supports it, or use the node-level graph integration which reuses the connection within a single graph execution.

### Async patterns

MATA's core exporters are synchronous. For async use cases (e.g., FastAPI), run exports in a thread pool:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from mata.core.exporters import export_valkey

executor = ThreadPoolExecutor(max_workers=4)

async def async_export(result, url, key, ttl=None):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        executor,
        lambda: export_valkey(result, url=url, key=key, ttl=ttl),
    )
```

---

## 9. Troubleshooting

### Issue 1: `ImportError` — no client installed

**Symptom:**

```
ImportError: Valkey export requires 'valkey' or 'redis' package.
Install with: pip install datamata[valkey] or pip install datamata[redis]
```

**Solution:**

```bash
pip install datamata[valkey]   # or pip install datamata[redis]
```

This error only occurs when a storage operation is actually called — `import mata` succeeds without either package.

---

### Issue 2: `ConnectionError` — server unreachable

**Symptom:**

```
valkey.exceptions.ConnectionError: Error 111 connecting to localhost:6379. Connection refused.
```

**Cause:** No Valkey/Redis server is running, or the host/port is wrong.

**Solution:**

```bash
# Start a local server with Docker
docker run -d --name valkey-server -p 6379:6379 valkey/valkey:latest

# Verify it's reachable
valkey-cli -h localhost -p 6379 ping   # → PONG
```

Check that firewall rules allow the port if connecting to a remote host.

---

### Issue 3: `KeyError` — key not found on load

**Symptom:**

```
KeyError: "Valkey key 'detections:frame_001' not found"
```

**Possible causes:**

- The TTL expired before you loaded the key
- The key was written to a different DB number
- A typo in the key name

**Solution:**

```bash
# Inspect keys matching a pattern
valkey-cli keys "detections:*"

# Check if a specific key exists and its TTL
valkey-cli exists detections:frame_001
valkey-cli ttl detections:frame_001       # -1 = no TTL, -2 = does not exist

# Check the DB number (default is 0)
valkey-cli -n 1 keys "*"
```

---

### Issue 4: `ValueError` — cannot auto-detect result type

**Symptom:**

```
ValueError: Cannot auto-detect result type from keys: ['foo', 'bar'].
Specify result_type explicitly.
```

**Cause:** The stored JSON does not have the expected top-level keys (`instances`, `predictions`, `depth`, `regions`). This can happen if a non-MATA value was stored under the same key.

**Solution:**

```bash
# Inspect the raw value
valkey-cli get my_key | python -c "import sys, json; print(json.dumps(json.loads(sys.stdin.read()), indent=2))"
```

If the data is valid MATA JSON but from an older format, specify the type explicitly:

```python
result = load_valkey(url=URL, key="my_key", result_type="vision")
```

---

### Issue 5: Large memory usage / OOM

**Symptom:** Valkey server memory grows unboundedly; keys are never evicted.

**Cause:** TTL was not set, or the eviction policy is `noeviction` (default).

**Solution:**

Set a `maxmemory` limit and an eviction policy in your Valkey config (`valkey.conf`):

```
maxmemory 512mb
maxmemory-policy allkeys-lru
```

Or configure via `valkey-cli`:

```bash
valkey-cli config set maxmemory 512mb
valkey-cli config set maxmemory-policy allkeys-lru
```

Always set a TTL on keys written by high-frequency pipelines:

```python
result.save("valkey://localhost:6379/track:latest", ttl=30)
```

---

## See Also

- [Graph API Reference — Storage Nodes](GRAPH_API_REFERENCE.md#storage-nodes) — full parameter reference for `ValkeyStore` and `ValkeyLoad`
- [QUICK_REFERENCE.md — Valkey section](../QUICK_REFERENCE.md#️-valkeyredis-storage-quick-reference-v19) — cheatsheet
- [Valkey official documentation](https://valkey.io/documentation/)
- [MATA Validation Guide](VALIDATION_GUIDE.md)
