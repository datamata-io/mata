from __future__ import annotations

from dataclasses import asdict, is_dataclass
import http.server
import json
import mimetypes
import socketserver
import threading
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mata.annotate.ai_assist import AIAssist
from mata.annotate.dataset_manager import DatasetManager
from mata.core.logging import get_logger

logger = get_logger(__name__)

_MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB


def _json_safe(value: Any) -> Any:
    """Convert status payloads into JSON-safe values."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolve_static_dir() -> Path:
    """Resolve the packaged static asset directory."""
    return (Path(__file__).parent / "static").resolve()


_STATIC_DIR = _resolve_static_dir()


def _resolve_static_path(url_path: str) -> Path | None:
    """Resolve a URL path to a file under the static directory.

    Returns ``None`` if the path would escape the static directory.
    """
    # Strip leading /static/ prefix if present
    rel = url_path.lstrip("/")
    if rel.startswith("static/"):
        rel = rel[len("static/"):]

    candidate = (_STATIC_DIR / rel).resolve()
    try:
        candidate.relative_to(_STATIC_DIR.resolve())
    except ValueError:
        return None
    return candidate


class AnnotateHandler(http.server.BaseHTTPRequestHandler):
    """Request handler for the annotation server."""

    # ``server`` is typed as AnnotateServer at runtime
    server: "AnnotateServer"

    # ------------------------------------------------------------------
    # HTTP verb dispatch
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_index()
        elif path.startswith("/static/"):
            self._serve_static(path)
        elif path.startswith("/api/"):
            self._dispatch_api("GET", self.path)
        else:
            self._send_error("Not found", 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            self._dispatch_api("POST", self.path)
        else:
            self._send_error("Not found", 404)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            self._dispatch_api("DELETE", self.path)
        else:
            self._send_error("Not found", 404)

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            self._dispatch_api("PUT", self.path)
        else:
            self._send_error("Not found", 404)

    def do_PATCH(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            self._dispatch_api("PATCH", self.path)
        else:
            self._send_error("Not found", 404)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _dispatch_api(self, method: str, path: str) -> None:
        # Inline health endpoint — no external dependency needed.
        if urlparse(path).path == "/api/health":
            self._send_json({"status": "ok"})
            return

        # Delegate to api_handler module (wired in Task A3/C3).
        from mata.annotate import api_handler  # lazy import

        try:
            body = self._read_json_body() if method in ("POST", "PUT", "PATCH") else {}
        except ValueError as exc:
            self._send_error(str(exc), 400)
            return

        try:
            result = api_handler.dispatch(self.server, method, self.path, body)
        except NotImplementedError:
            self._send_error("Not implemented", 501)
            return
        except (ValueError, FileNotFoundError) as exc:
            self._send_error(str(exc), 400 if isinstance(exc, ValueError) else 404)
            return
        except Exception as exc:
            logger.exception("Unhandled error in API dispatch: %s %s", method, self.path)
            self._send_error(f"Internal server error: {exc}", 500)
            return

        if result is None:
            self._send_error("Not found", 404)
            return

        status, response = result

        # Binary responses (images, thumbnails) are returned as (status, bytes, content_type)
        if isinstance(response, (bytes, bytearray)):
            content_type = "application/octet-stream"
            self._send_binary(response, content_type, status)
        elif isinstance(response, tuple) and len(response) == 2 and isinstance(response[0], (bytes, bytearray)):
            data, content_type = response
            self._send_binary(data, content_type, status)
        else:
            self._send_json(response, status)

    # ------------------------------------------------------------------
    # Static file serving
    # ------------------------------------------------------------------

    def _serve_index(self) -> None:
        index = _resolve_static_dir() / "index.html"
        if not index.exists():
            self._send_error("index.html not found", 404)
            return
        self._serve_static_file(index, "text/html; charset=utf-8")

    def _serve_static(self, url_path: str) -> None:
        file_path = _resolve_static_path(url_path)
        if file_path is None:
            self._send_error("Forbidden", 403)
            return
        if not file_path.exists() or not file_path.is_file():
            self._send_error("Not found", 404)
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        if mime is None:
            mime = "application/octet-stream"
        self._serve_static_file(file_path, mime)

    def _serve_static_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (ConnectionAbortedError, BrokenPipeError):
            pass

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionAbortedError, BrokenPipeError):
            pass

    def _send_binary(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (ConnectionAbortedError, BrokenPipeError):
            pass

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message, "code": status}, status)

    def _read_json_body(self) -> dict:
        """Read and parse the JSON request body, enforcing 10 MB limit."""
        length_str = self.headers.get("Content-Length")
        if length_str is None:
            return {}

        try:
            length = int(length_str)
        except ValueError:
            raise ValueError("Invalid Content-Length")

        if length > _MAX_BODY_BYTES:
            # Consume headers so the response is sent cleanly
            self.send_response(413)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            body = b'{"error": "Payload Too Large", "code": 413}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            raise ValueError("Payload Too Large")

        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("%s - - [%s] %s", self.address_string(), self.log_date_time_string(), fmt % args)


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Thread-per-request HTTP server so long AI-assist calls don't block."""

    daemon_threads = True


class AnnotateServer:
    """MATA annotation web server.

    Wraps a ``ThreadingHTTPServer`` and holds shared state (dataset manager,
    AI models) accessible to every request handler via ``handler.server``.
    """

    def __init__(
        self,
        data_root: str = "data",
        host: str = "127.0.0.1",
        port: int = 8710,
        detect_model: str | None = None,
        vlm_model: str | None = None,
        embed_model: str | None = None,
        zeroshot_model: str | None = None,
    ) -> None:
        self.data_root = str(data_root)
        self.host = host
        self.port = port
        self.detect_model = detect_model
        self.vlm_model = vlm_model
        self.embed_model = embed_model

        self.dataset_manager = DatasetManager(Path(data_root).resolve())
        self.coco_state: dict[str, dict[str, Any]] = {}
        self.coco_io: dict[str, dict[str, Any]] | None = self.coco_state
        self.ai_models: dict[str, Any] = {}
        self.ai_assist: Any = AIAssist(
            detect_model=detect_model,
            vlm_model=vlm_model,
            embed_model=embed_model,
        )
        if detect_model is not None:
            self.ai_assist.load_detect()
        if vlm_model is not None:
            self.ai_assist.load_vlm()
        if zeroshot_model is not None:
            self.ai_assist.load_zeroshot(zeroshot_model)

        self._train_thread: threading.Thread | None = None
        self._train_status: dict[str, Any] = {"status": "idle"}
        self._train_stop = threading.Event()
        self._train_lock = threading.Lock()

        self._rescan_jobs: dict[str, dict] = {}
        self._rescan_lock = threading.Lock()

        self._redistribute_jobs: dict[str, dict] = {}
        self._redistribute_lock = threading.Lock()

        self._httpd = _ThreadingHTTPServer((host, port), AnnotateHandler)
        self.port = self._httpd.server_port

        # BaseHTTPRequestHandler exposes the HTTPServer instance as self.server.
        # Mirror shared annotate state onto that live server object so routed API
        # handlers can access the dataset manager, COCO cache, and AI state.
        self._httpd.dataset_manager = self.dataset_manager  # type: ignore[attr-defined]
        self._httpd.coco_state = self.coco_state  # type: ignore[attr-defined]
        self._httpd.coco_io = self.coco_io  # type: ignore[attr-defined]
        self._httpd.ai_models = self.ai_models  # type: ignore[attr-defined]
        self._httpd.ai_assist = self.ai_assist  # type: ignore[attr-defined]
        self._httpd.start_training = self.start_training  # type: ignore[attr-defined]
        self._httpd.get_training_status = self.get_training_status  # type: ignore[attr-defined]
        self._httpd.stop_training = self.stop_training  # type: ignore[attr-defined]
        self._httpd.start_rescan = self.start_rescan  # type: ignore[attr-defined]
        self._httpd.get_rescan_status = self.get_rescan_status  # type: ignore[attr-defined]
        self._httpd.start_redistribute = self.start_redistribute  # type: ignore[attr-defined]
        self._httpd.get_redistribute_status = self.get_redistribute_status  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _set_train_status(self, status: dict[str, Any]) -> None:
        with self._train_lock:
            self._train_status = _json_safe(status)

    def _resolve_training_data_path(self, data: str) -> Path:
        candidate = Path(data)
        if not candidate.is_absolute():
            candidate = self.dataset_manager._root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.dataset_manager._root)
        except ValueError as exc:
            raise PermissionError("Training data path must stay within the annotation data root.") from exc
        if not resolved.exists():
            raise FileNotFoundError(f"Training data path not found: {resolved}")
        return resolved

    def _run_training(
        self,
        mode: str,
        task: str,
        model: str,
        data: str,
        kwargs: dict[str, Any],
    ) -> None:
        """Execute a train or fine-tune request on a background thread."""
        import mata

        with self._train_lock:
            stop_requested = self._train_status.get("stop_requested", False)

        self._set_train_status(
            {
                "status": "running",
                "mode": mode,
                "task": task,
                "model": model,
                "data": data,
                "stop_requested": stop_requested,
            }
        )

        train_fn = mata.finetune if mode == "finetune" else mata.train
        try:
            result = train_fn(task, model=model, data=data, **kwargs)
            self._set_train_status(
                {
                    "status": "done",
                    "mode": mode,
                    "task": task,
                    "model": model,
                    "data": data,
                    "stop_requested": self._train_stop.is_set(),
                    "best_checkpoint": getattr(result, "best_checkpoint", ""),
                    "last_checkpoint": getattr(result, "last_checkpoint", ""),
                    "metrics": _json_safe(getattr(result, "final_metrics", None)),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Background %s request failed for task=%s model=%s", mode, task, model)
            self._set_train_status(
                {
                    "status": "error",
                    "mode": mode,
                    "task": task,
                    "model": model,
                    "data": data,
                    "stop_requested": self._train_stop.is_set(),
                    "error": str(exc),
                }
            )
        finally:
            with self._train_lock:
                self._train_thread = None

    def start_training(self, request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Validate and launch a background train/finetune request."""
        task = str(request["task"])
        model = str(request["model"])
        data = str(request["data"])
        mode = str(request.get("mode", request.get("action", "train"))).lower()
        if mode not in {"train", "finetune"}:
            raise ValueError("Training mode must be 'train' or 'finetune'.")

        resolved_data = self._resolve_training_data_path(data)
        kwargs = {
            key: value
            for key, value in request.items()
            if key not in {"task", "model", "data", "mode", "action"}
        }

        with self._train_lock:
            if self._train_thread is not None and self._train_thread.is_alive():
                return 409, {"error": "Training is already running.", "code": 409}

            self._train_stop.clear()
            self._train_status = {
                "status": "starting",
                "mode": mode,
                "task": task,
                "model": model,
                "data": str(resolved_data),
                "stop_requested": False,
            }
            self._train_thread = threading.Thread(
                target=self._run_training,
                args=(mode, task, model, str(resolved_data), kwargs),
                name="mata-annotate-training",
                daemon=True,
            )
            self._train_thread.start()

        return 202, {
            "status": "started",
            "mode": mode,
            "task": task,
            "model": model,
            "data": str(resolved_data),
        }

    def get_training_status(self) -> dict[str, Any]:
        with self._train_lock:
            return dict(self._train_status)

    def stop_training(self) -> dict[str, Any]:
        with self._train_lock:
            active = self._train_thread is not None and self._train_thread.is_alive()
            if active:
                self._train_stop.set()
                self._train_status = {**self._train_status, "stop_requested": True}
                return {"status": "stop_requested"}
            return {"status": "idle"}

    def start_rescan(self, name: str) -> dict[str, Any]:
        """Launch a background rescan of dataset *name* to populate/refresh its cache.

        Returns ``{"status": "started"}`` immediately; the caller should poll
        ``get_rescan_status(name)`` to check for ``"done"`` or ``"error"``.
        Returns ``{"status": "already_running"}`` if a job is still active.
        Returns ``{"status": "not_found"}`` if the dataset directory does not exist.
        """
        from mata.annotate.dataset_manager import _run_rescan_worker

        dataset_dir = self.dataset_manager._safe_resolve(name)
        if not dataset_dir.is_dir():
            return {"status": "not_found"}

        with self._rescan_lock:
            existing = self._rescan_jobs.get(name, {})
            if existing.get("status") == "running":
                return {"status": "already_running"}
            self._rescan_jobs[name] = {"status": "running"}

        t = threading.Thread(
            target=_run_rescan_worker,
            args=(self.dataset_manager, name, self._rescan_jobs, self._rescan_lock),
            name=f"mata-rescan-{name}",
            daemon=True,
        )
        t.start()
        return {"status": "started"}

    def get_rescan_status(self, name: str) -> dict[str, Any]:
        """Return current rescan status for *name* (``idle``, ``running``, ``done``, or ``error``)."""
        with self._rescan_lock:
            return dict(self._rescan_jobs.get(name, {"status": "idle"}))

    def start_redistribute(
        self,
        name: str,
        train_pct: int,
        val_pct: int,
        test_pct: int,
        seed: int | None = None,
        annotated_first: bool = True,
    ) -> dict[str, Any]:
        """Launch a background redistribution of *name*'s images across split dirs.

        Returns ``{"status": "started"}`` immediately; poll
        ``get_redistribute_status(name)`` for ``"done"`` or ``"error"``.
        """
        from mata.annotate.dataset_manager import _run_redistribute_worker

        dataset_dir = self.dataset_manager._safe_resolve(name)
        if not dataset_dir.is_dir():
            return {"status": "not_found"}

        with self._redistribute_lock:
            existing = self._redistribute_jobs.get(name, {})
            if existing.get("status") == "running":
                return {"status": "already_running"}
            self._redistribute_jobs[name] = {"status": "running"}

        params = {
            "train": train_pct,
            "val": val_pct,
            "test": test_pct,
            "seed": seed,
            "annotated_first": annotated_first,
        }
        t = threading.Thread(
            target=_run_redistribute_worker,
            args=(self.dataset_manager, name, params, self._redistribute_jobs, self._redistribute_lock),
            name=f"mata-redistribute-{name}",
            daemon=True,
        )
        t.start()
        return {"status": "started"}

    def get_redistribute_status(self, name: str) -> dict[str, Any]:
        """Return current redistribute status for *name*."""
        with self._redistribute_lock:
            return dict(self._redistribute_jobs.get(name, {"status": "idle"}))

    def serve_forever(self) -> None:
        logger.info("MATA Annotate server running at %s  (press Ctrl+C to stop)", self.url)
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        self._httpd.shutdown()
        logger.info("MATA Annotate server stopped.")


def start_server(
    data: str = "data",
    *,
    host: str = "127.0.0.1",
    port: int = 8710,
    open_browser: bool = True,
    block: bool = True,
    detect_model: str | None = None,
    vlm_model: str | None = None,
    embed_model: str | None = None,
    zeroshot_model: str | None = None,
    **kwargs: Any,
) -> AnnotateServer:
    """Launch the MATA annotation web server.

    Args:
        data: Root data directory to manage. Defaults to ``"data"``.
        host: Bind address. Defaults to ``"127.0.0.1"`` (localhost only).
        port: Port number. Defaults to ``8710``.
        open_browser: Open the default browser automatically. Defaults to ``True``.
        block: If ``True`` (default) block until the server is stopped (Ctrl+C).
               If ``False``, start in a daemon thread and return the server instance.
        detect_model: Detection model alias/ID for AI-assist pre-labeling.
        vlm_model: VLM model alias/ID for AI-assist auto-annotation.
        embed_model: Embedding model alias/ID for CLIP classify suggestions.
        zeroshot_model: Grounding DINO model alias/ID for zero-shot detection AI-assist.
        **kwargs: Reserved for future server configuration.

    Returns:
        The running :class:`AnnotateServer` instance.
    """
    server = AnnotateServer(
        data_root=data,
        host=host,
        port=port,
        detect_model=detect_model,
        vlm_model=vlm_model,
        embed_model=embed_model,
        zeroshot_model=zeroshot_model,
    )

    if open_browser:
        # Open after a short delay so the server is ready
        t = threading.Timer(0.5, lambda: webbrowser.open(server.url))
        t.daemon = True
        t.start()

    if block:
        server.serve_forever()
        return server

    thread = threading.Thread(target=server._httpd.serve_forever, name="mata-annotate", daemon=True)
    thread.start()
    logger.info("MATA Annotate server started at %s (background thread)", server.url)
    return server
