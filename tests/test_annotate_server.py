"""Unit tests for AnnotateServer and AnnotateHandler (Task G1).

Tests cover:
1.  test_server_starts_and_stops          — start background thread, verify port, shutdown
2.  test_server_default_host_is_localhost  — default host is 127.0.0.1
3.  test_health_endpoint                  — GET /api/health → 200 {"status": "ok"}
4.  test_health_endpoint_content_type     — Content-Type is application/json
5.  test_serve_index_html                 — GET / → 200 text/html
6.  test_serve_index_html_via_explicit    — GET /index.html → 200
7.  test_static_file_serving              — GET /static/index.html → 200
8.  test_static_path_traversal_blocked    — GET /static/../../../etc/passwd → 403
9.  test_static_traversal_double_dot_url  — GET /static/../../secret → 403
10. test_request_body_size_limit          — POST >10 MB → 413
11. test_json_response_content_type       — JSON response has correct Content-Type
12. test_unknown_route_returns_404        — GET /nonexistent → 404
13. test_unknown_api_route_returns_404    — GET /api/unknown/path → 404
14. test_method_not_allowed_patch         — PATCH /api/datasets (no ann id) → 404
15. test_server_url_property             — url property = http://host:port
16. test_port_zero_gets_random_port      — port=0 → port > 0 after bind
17. test_invalid_json_body_returns_400   — POST with malformed JSON → 400
18. test_get_api_datasets_success        — GET /api/datasets → 200 list
19. test_server_stores_data_root         — data_root stored on instance
20. test_server_has_dataset_manager      — dataset_manager initialized
21. test_server_has_empty_coco_state     — coco_state initialized as dict
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from pathlib import Path

import pytest

from mata.annotate.dataset_manager import DatasetManager
from mata.annotate.server import AnnotateServer


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


class _ServerContext:
    """Context manager that starts a server thread and cleans up on exit."""

    def __init__(self, data_root: Path, host: str = "127.0.0.1", port: int = 0) -> None:
        self.server = AnnotateServer(data_root=str(data_root), host=host, port=port)
        self._thread = threading.Thread(
            target=self.server._httpd.serve_forever,
            daemon=True,
        )

    def __enter__(self) -> "AnnotateServer":
        self._thread.start()
        # Give the server a moment to start
        time.sleep(0.05)
        return self.server

    def __exit__(self, *_: object) -> None:
        self.server._httpd.shutdown()
        self._thread.join(timeout=5)
        self.server._httpd.server_close()


def _get(server: AnnotateServer, path: str) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    conn.request("GET", path)
    return conn.getresponse()


def _post(
    server: AnnotateServer,
    path: str,
    body: bytes | None = None,
    headers: dict | None = None,
) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    h = headers or {}
    if body is not None and "Content-Length" not in h:
        h["Content-Length"] = str(len(body))
    conn.request("POST", path, body=body, headers=h)
    return conn.getresponse()


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


def test_server_starts_and_stops(tmp_path: Path) -> None:
    """Server can start in a background thread and cleanly shut down."""
    with _ServerContext(tmp_path) as srv:
        assert srv.port > 0
        resp = _get(srv, "/api/health")
        assert resp.status == 200
    # After __exit__ the server has been shut down — nothing more to assert


def test_server_default_host_is_localhost() -> None:
    """Default host is 127.0.0.1 (localhost only)."""
    srv = AnnotateServer.__new__(AnnotateServer)
    # Check the constructor sets host=127.0.0.1 by default
    import inspect
    sig = inspect.signature(AnnotateServer.__init__)
    host_default = sig.parameters["host"].default
    assert host_default == "127.0.0.1"


def test_port_zero_gets_random_port(tmp_path: Path) -> None:
    """port=0 produces an OS-assigned port > 0."""
    srv = AnnotateServer(data_root=str(tmp_path), port=0)
    try:
        assert srv.port > 0
    finally:
        srv._httpd.server_close()


def test_server_url_property(tmp_path: Path) -> None:
    """url property returns http://host:port."""
    srv = AnnotateServer(data_root=str(tmp_path), host="127.0.0.1", port=0)
    try:
        assert srv.url == f"http://127.0.0.1:{srv.port}"
    finally:
        srv._httpd.server_close()


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


def test_health_endpoint(tmp_path: Path) -> None:
    """GET /api/health returns 200 with status:ok."""
    with _ServerContext(tmp_path) as srv:
        resp = _get(srv, "/api/health")
        data = json.loads(resp.read())
        assert resp.status == 200
        assert data == {"status": "ok"}


def test_health_endpoint_content_type(tmp_path: Path) -> None:
    """GET /api/health Content-Type is application/json."""
    with _ServerContext(tmp_path) as srv:
        resp = _get(srv, "/api/health")
        resp.read()
        ct = resp.getheader("Content-Type", "")
        assert "application/json" in ct


def test_json_response_content_type(tmp_path: Path) -> None:
    """All JSON API responses include application/json in Content-Type."""
    with _ServerContext(tmp_path) as srv:
        resp = _get(srv, "/api/datasets")
        resp.read()
        ct = resp.getheader("Content-Type", "")
        assert "application/json" in ct


def test_serve_index_html(tmp_path: Path) -> None:
    """GET / serves index.html with 200 text/html."""
    with _ServerContext(tmp_path) as srv:
        resp = _get(srv, "/")
        body = resp.read()
        assert resp.status == 200
        ct = resp.getheader("Content-Type", "")
        assert "text/html" in ct
        assert len(body) > 0


def test_serve_index_html_via_explicit_path(tmp_path: Path) -> None:
    """GET /index.html also returns index.html."""
    with _ServerContext(tmp_path) as srv:
        resp = _get(srv, "/index.html")
        body = resp.read()
        assert resp.status == 200
        assert len(body) > 0


def test_static_file_serving(tmp_path: Path) -> None:
    """GET /static/index.html serves the index file."""
    with _ServerContext(tmp_path) as srv:
        resp = _get(srv, "/static/index.html")
        body = resp.read()
        assert resp.status == 200
        assert len(body) > 0


def test_static_path_traversal_blocked(tmp_path: Path) -> None:
    """GET /static/../../../etc/passwd returns 403 Forbidden."""
    with _ServerContext(tmp_path) as srv:
        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
        # Send a raw request to bypass URL normalisation in http.client
        conn._send_request = None  # type: ignore[assignment]
        conn.connect()
        conn.sock.sendall(b"GET /static/../../../etc/passwd HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        resp = http.client.HTTPResponse(conn.sock)
        resp.begin()
        resp.read()
        assert resp.status == 403


def test_static_traversal_double_dot_url(tmp_path: Path) -> None:
    """_resolve_static_path returns None for paths that escape static dir."""
    from mata.annotate.server import _resolve_static_path

    result = _resolve_static_path("/static/../../secret.txt")
    assert result is None


def test_unknown_route_returns_404(tmp_path: Path) -> None:
    """GET /nonexistent returns 404."""
    with _ServerContext(tmp_path) as srv:
        resp = _get(srv, "/nonexistent-route")
        resp.read()
        assert resp.status == 404


def test_unknown_api_route_returns_404(tmp_path: Path) -> None:
    """GET /api/does_not_exist returns 404."""
    with _ServerContext(tmp_path) as srv:
        resp = _get(srv, "/api/does_not_exist")
        resp.read()
        assert resp.status == 404


def test_method_not_allowed_patch(tmp_path: Path) -> None:
    """PATCH /api/datasets (no annotation id) returns 404 — no route match."""
    import socket

    with _ServerContext(tmp_path) as srv:
        body = b"{}"
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect(("127.0.0.1", srv.port))
        s.sendall(
            b"PATCH /api/datasets HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        )
        response = b""
        try:
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                response += chunk
        except OSError:
            pass
        finally:
            s.close()

        # PATCH /api/datasets has no matching route → 404
        assert b"404" in response[:50]


# ---------------------------------------------------------------------------
# Request body / size limit tests
# ---------------------------------------------------------------------------


def test_request_body_size_limit(tmp_path: Path) -> None:
    """POST with Content-Length >10 MB returns 413 Payload Too Large."""
    import socket

    with _ServerContext(tmp_path) as srv:
        # Send only the header with a huge Content-Length — avoids uploading 10 MB
        big_length = 10 * 1024 * 1024 + 1
        request = (
            b"POST /api/datasets/test_ds HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(big_length).encode() + b"\r\n\r\n"
        )
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect(("127.0.0.1", srv.port))
        s.sendall(request)
        response = b""
        try:
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                response += chunk
        except OSError:
            pass
        finally:
            s.close()

        assert b"413" in response[:50]


def test_invalid_json_body_returns_400(tmp_path: Path) -> None:
    """POST with malformed JSON body returns 400 Bad Request."""
    with _ServerContext(tmp_path) as srv:
        body = b"{not valid json"
        resp = _post(
            srv,
            "/api/datasets/test_ds",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        resp.read()
        assert resp.status == 400


# ---------------------------------------------------------------------------
# Dataset API via HTTP
# ---------------------------------------------------------------------------


def test_get_api_datasets_success(tmp_path: Path) -> None:
    """GET /api/datasets returns 200 and a JSON list."""
    with _ServerContext(tmp_path) as srv:
        resp = _get(srv, "/api/datasets")
        data = json.loads(resp.read())
        assert resp.status == 200
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# Server state tests
# ---------------------------------------------------------------------------


def test_server_stores_data_root(tmp_path: Path) -> None:
    """AnnotateServer.data_root stores the provided path as string."""
    srv = AnnotateServer(data_root=str(tmp_path), port=0)
    try:
        assert srv.data_root == str(tmp_path)
    finally:
        srv._httpd.server_close()


def test_server_has_dataset_manager(tmp_path: Path) -> None:
    """AnnotateServer creates a DatasetManager on init."""
    srv = AnnotateServer(data_root=str(tmp_path), port=0)
    try:
        assert isinstance(srv.dataset_manager, DatasetManager)
    finally:
        srv._httpd.server_close()


def test_server_has_empty_coco_state(tmp_path: Path) -> None:
    """AnnotateServer.coco_state is an empty dict on init."""
    srv = AnnotateServer(data_root=str(tmp_path), port=0)
    try:
        assert isinstance(srv.coco_state, dict)
        assert srv.coco_state == {}
    finally:
        srv._httpd.server_close()


# ---------------------------------------------------------------------------
# Task F4: Theme tests
# ---------------------------------------------------------------------------


def _read_static_file(name: str) -> str:
    """Return the text content of a packaged static asset."""
    from mata.annotate.server import _resolve_static_dir

    return (_resolve_static_dir() / name).read_text(encoding="utf-8")


def test_index_html_contains_dark_theme_css_block() -> None:
    """index.html must contain a [data-theme="dark"] CSS variable block."""
    html = _read_static_file("index.html")
    assert '[data-theme="dark"]' in html, (
        'index.html is missing the [data-theme="dark"] CSS block'
    )


def test_index_html_contains_light_theme_css_variables() -> None:
    """index.html must contain a light-theme CSS variable block (:root or [data-theme="light"])."""
    html = _read_static_file("index.html")
    has_root = ":root" in html
    has_light_attr = '[data-theme="light"]' in html
    assert has_root or has_light_attr, (
        "index.html is missing a light-theme CSS variable block (:root or [data-theme=\"light\"])"
    )


def test_app_js_contains_theme_manager() -> None:
    """app.js must contain a ThemeManager initialization."""
    js = _read_static_file("app.js")
    assert "ThemeManager" in js, "app.js does not define or reference ThemeManager"
    # Verify it is actually initialised (not just referenced in a comment)
    assert "ThemeManager.init" in js or "ThemeManager =" in js, (
        "app.js references ThemeManager but does not initialise it"
    )


def test_app_js_uses_localstorage_key() -> None:
    """app.js must reference the 'mata-annotate-theme' localStorage key."""
    js = _read_static_file("app.js")
    assert "mata-annotate-theme" in js, (
        "app.js does not reference the 'mata-annotate-theme' localStorage key"
    )
