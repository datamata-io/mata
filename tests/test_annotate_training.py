from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import mata

from mata.annotate import api_handler
from mata.annotate.server import AnnotateServer


def _make_server(data_root: Path) -> AnnotateServer:
    return AnnotateServer(data_root=str(data_root), port=0)


def _write_dataset_yaml(root: Path, relative_path: str = "project/dataset.yaml") -> Path:
    yaml_path = root / relative_path
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("path: .\ntrain: train\nval: val\n", encoding="utf-8")
    return yaml_path


def _wait_for_status(server: AnnotateServer, expected: str, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = server.get_training_status()
        if status.get("status") == expected:
            return status
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for training status '{expected}': {server.get_training_status()}")


def _join_training_thread(server: AnnotateServer, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        thread = server._train_thread
        if thread is None:
            return
        thread.join(timeout=0.05)
        if not thread.is_alive():
            return
    raise AssertionError("Training thread did not finish before timeout.")


def test_server_initializes_training_state(tmp_path: Path) -> None:
    server = _make_server(tmp_path)
    try:
        assert server.get_training_status() == {"status": "idle"}
        assert server._train_thread is None
        assert server._train_stop.is_set() is False
    finally:
        server._httpd.server_close()


def test_start_training_returns_202_and_reports_running(tmp_path: Path, monkeypatch) -> None:
    yaml_path = _write_dataset_yaml(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def fake_train(task: str, *, model: str, data: str, **kwargs):
        assert task == "detect"
        assert model == "facebook/detr-resnet-50"
        assert data == str(yaml_path.resolve())
        assert kwargs["epochs"] == 3
        started.set()
        release.wait(timeout=1.0)
        return SimpleNamespace(
            best_checkpoint="runs/train/best",
            last_checkpoint="runs/train/last",
            final_metrics={"map50": 0.91},
        )

    monkeypatch.setattr(mata, "train", fake_train)

    server = _make_server(tmp_path)
    try:
        status, payload = api_handler.dispatch(
            server,
            "POST",
            "/api/train",
            {
                "task": "detect",
                "model": "facebook/detr-resnet-50",
                "data": "project/dataset.yaml",
                "epochs": 3,
            },
        )

        assert status == 202
        assert payload["status"] == "started"
        assert payload["data"] == str(yaml_path.resolve())
        assert started.wait(timeout=1.0)
        running = _wait_for_status(server, "running")
        assert running["task"] == "detect"
        assert running["model"] == "facebook/detr-resnet-50"

        release.set()
        _wait_for_status(server, "done")
        _join_training_thread(server)
    finally:
        server._httpd.server_close()


def test_training_status_after_completion_includes_checkpoints(tmp_path: Path, monkeypatch) -> None:
    _write_dataset_yaml(tmp_path)

    def fake_train(*args, **kwargs):
        return SimpleNamespace(
            best_checkpoint="runs/train/best",
            last_checkpoint="runs/train/last",
            final_metrics={"loss": 0.12, "map50": 0.88},
        )

    monkeypatch.setattr(mata, "train", fake_train)

    server = _make_server(tmp_path)
    try:
        status, _ = api_handler.dispatch(
            server,
            "POST",
            "/api/train",
            {
                "task": "detect",
                "model": "facebook/detr-resnet-50",
                "data": "project/dataset.yaml",
            },
        )

        assert status == 202
        done = _wait_for_status(server, "done")
        assert done["best_checkpoint"] == "runs/train/best"
        assert done["last_checkpoint"] == "runs/train/last"
        assert done["metrics"] == {"loss": 0.12, "map50": 0.88}
        _join_training_thread(server)
    finally:
        server._httpd.server_close()


def test_concurrent_training_rejected_409(tmp_path: Path, monkeypatch) -> None:
    _write_dataset_yaml(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def fake_train(*args, **kwargs):
        started.set()
        release.wait(timeout=1.0)
        return SimpleNamespace(best_checkpoint="", last_checkpoint="", final_metrics={})

    monkeypatch.setattr(mata, "train", fake_train)

    server = _make_server(tmp_path)
    try:
        first_status, _ = api_handler.dispatch(
            server,
            "POST",
            "/api/train",
            {
                "task": "detect",
                "model": "facebook/detr-resnet-50",
                "data": "project/dataset.yaml",
            },
        )
        assert first_status == 202
        assert started.wait(timeout=1.0)

        second_status, second_payload = api_handler.dispatch(
            server,
            "POST",
            "/api/train",
            {
                "task": "detect",
                "model": "facebook/detr-resnet-50",
                "data": "project/dataset.yaml",
            },
        )

        assert second_status == 409
        assert second_payload["code"] == 409

        release.set()
        _wait_for_status(server, "done")
        _join_training_thread(server)
    finally:
        server._httpd.server_close()


def test_training_data_path_outside_root_rejected(tmp_path: Path) -> None:
    server = _make_server(tmp_path)
    outside = tmp_path.parent / "outside-dataset.yaml"
    outside.write_text("path: .\n", encoding="utf-8")
    try:
        status, payload = api_handler.dispatch(
            server,
            "POST",
            "/api/train",
            {
                "task": "detect",
                "model": "facebook/detr-resnet-50",
                "data": str(outside),
            },
        )

        assert status == 403
        assert payload["code"] == 403
    finally:
        server._httpd.server_close()


def test_training_stop_request_sets_status_flag(tmp_path: Path, monkeypatch) -> None:
    _write_dataset_yaml(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def fake_train(*args, **kwargs):
        started.set()
        release.wait(timeout=1.0)
        return SimpleNamespace(best_checkpoint="", last_checkpoint="", final_metrics={})

    monkeypatch.setattr(mata, "train", fake_train)

    server = _make_server(tmp_path)
    try:
        api_handler.dispatch(
            server,
            "POST",
            "/api/train",
            {
                "task": "detect",
                "model": "facebook/detr-resnet-50",
                "data": "project/dataset.yaml",
            },
        )
        assert started.wait(timeout=1.0)

        status, payload = api_handler.dispatch(server, "POST", "/api/train/stop", {})

        assert status == 200
        assert payload == {"status": "stop_requested"}
        running = server.get_training_status()
        assert running["stop_requested"] is True

        release.set()
        _wait_for_status(server, "done")
        _join_training_thread(server)
    finally:
        server._httpd.server_close()


def test_training_error_captured_in_status(tmp_path: Path, monkeypatch) -> None:
    _write_dataset_yaml(tmp_path)

    def fake_train(*args, **kwargs):
        raise RuntimeError("training failed")

    monkeypatch.setattr(mata, "train", fake_train)

    server = _make_server(tmp_path)
    try:
        status, _ = api_handler.dispatch(
            server,
            "POST",
            "/api/train",
            {
                "task": "detect",
                "model": "facebook/detr-resnet-50",
                "data": "project/dataset.yaml",
            },
        )

        assert status == 202
        error_status = _wait_for_status(server, "error")
        assert error_status["error"] == "training failed"
        _join_training_thread(server)
    finally:
        server._httpd.server_close()


def test_finetune_mode_dispatches_to_mata_finetune(tmp_path: Path, monkeypatch) -> None:
    yaml_path = _write_dataset_yaml(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_finetune(task: str, *, model: str, data: str, **kwargs):
        calls.append((task, data))
        return SimpleNamespace(
            best_checkpoint="runs/finetune/best",
            last_checkpoint="runs/finetune/last",
            final_metrics={"top1": 0.97},
        )

    monkeypatch.setattr(mata, "finetune", fake_finetune)

    server = _make_server(tmp_path)
    try:
        status, payload = api_handler.dispatch(
            server,
            "POST",
            "/api/train",
            {
                "mode": "finetune",
                "task": "classify",
                "model": "microsoft/resnet-50",
                "data": "project/dataset.yaml",
            },
        )

        assert status == 202
        assert payload["mode"] == "finetune"
        done = _wait_for_status(server, "done")
        assert done["mode"] == "finetune"
        assert calls == [("classify", str(yaml_path.resolve()))]
        _join_training_thread(server)
    finally:
        server._httpd.server_close()


def test_get_training_status_api_endpoint_returns_idle_initially(tmp_path: Path) -> None:
    server = _make_server(tmp_path)
    try:
        status, payload = api_handler.dispatch(server, "GET", "/api/train/status", {})
        assert status == 200
        assert payload["status"] == "idle"
    finally:
        server._httpd.server_close()


def test_stop_when_idle_returns_idle_status(tmp_path: Path) -> None:
    server = _make_server(tmp_path)
    try:
        status, payload = api_handler.dispatch(server, "POST", "/api/train/stop", {})
        assert status == 200
        assert payload["status"] == "idle"
    finally:
        server._httpd.server_close()


def test_missing_required_fields_returns_400(tmp_path: Path) -> None:
    server = _make_server(tmp_path)
    try:
        # Missing "model" and "data"
        status, payload = api_handler.dispatch(
            server,
            "POST",
            "/api/train",
            {"task": "detect"},
        )
        assert status == 400
        assert "error" in payload
    finally:
        server._httpd.server_close()


def test_training_data_path_not_found_returns_404(tmp_path: Path) -> None:
    server = _make_server(tmp_path)
    try:
        # File under data_root but does not exist — should return 404 Not Found
        status, payload = api_handler.dispatch(
            server,
            "POST",
            "/api/train",
            {
                "task": "detect",
                "model": "facebook/detr-resnet-50",
                "data": "nonexistent/dataset.yaml",
            },
        )
        assert status == 404
        assert "code" in payload
        assert payload["code"] == 404
    finally:
        server._httpd.server_close()


def test_extra_kwargs_forwarded_to_train(tmp_path: Path, monkeypatch) -> None:
    _write_dataset_yaml(tmp_path)
    captured: dict = {}

    def fake_train(task: str, *, model: str, data: str, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(best_checkpoint="", last_checkpoint="", final_metrics={})

    monkeypatch.setattr(mata, "train", fake_train)

    server = _make_server(tmp_path)
    try:
        api_handler.dispatch(
            server,
            "POST",
            "/api/train",
            {
                "task": "detect",
                "model": "facebook/detr-resnet-50",
                "data": "project/dataset.yaml",
                "epochs": 5,
                "batch_size": 4,
                "lr": 0.001,
            },
        )
        _wait_for_status(server, "done")
        _join_training_thread(server)

        assert captured["epochs"] == 5
        assert captured["batch_size"] == 4
        assert captured["lr"] == 0.001
    finally:
        server._httpd.server_close()


def test_train_thread_cleared_after_completion(tmp_path: Path, monkeypatch) -> None:
    _write_dataset_yaml(tmp_path)

    def fake_train(*args, **kwargs):
        return SimpleNamespace(best_checkpoint="", last_checkpoint="", final_metrics={})

    monkeypatch.setattr(mata, "train", fake_train)

    server = _make_server(tmp_path)
    try:
        api_handler.dispatch(
            server,
            "POST",
            "/api/train",
            {
                "task": "detect",
                "model": "facebook/detr-resnet-50",
                "data": "project/dataset.yaml",
            },
        )
        _wait_for_status(server, "done")
        _join_training_thread(server)
        assert server._train_thread is None
    finally:
        server._httpd.server_close()


def test_result_with_no_final_metrics_stores_none(tmp_path: Path, monkeypatch) -> None:
    _write_dataset_yaml(tmp_path)

    def fake_train(*args, **kwargs):
        # result has no final_metrics attribute
        return SimpleNamespace(best_checkpoint="runs/best", last_checkpoint="runs/last")

    monkeypatch.setattr(mata, "train", fake_train)

    server = _make_server(tmp_path)
    try:
        api_handler.dispatch(
            server,
            "POST",
            "/api/train",
            {
                "task": "detect",
                "model": "facebook/detr-resnet-50",
                "data": "project/dataset.yaml",
            },
        )
        done = _wait_for_status(server, "done")
        _join_training_thread(server)
        assert done["best_checkpoint"] == "runs/best"
        assert done["metrics"] is None
    finally:
        server._httpd.server_close()


def test_can_start_new_training_after_previous_completes(tmp_path: Path, monkeypatch) -> None:
    _write_dataset_yaml(tmp_path)
    call_count = {"n": 0}

    def fake_train(*args, **kwargs):
        call_count["n"] += 1
        return SimpleNamespace(best_checkpoint="", last_checkpoint="", final_metrics={})

    monkeypatch.setattr(mata, "train", fake_train)

    server = _make_server(tmp_path)
    try:
        for _ in range(2):
            status, _ = api_handler.dispatch(
                server,
                "POST",
                "/api/train",
                {
                    "task": "detect",
                    "model": "facebook/detr-resnet-50",
                    "data": "project/dataset.yaml",
                },
            )
            assert status == 202
            _wait_for_status(server, "done")
            _join_training_thread(server)

        assert call_count["n"] == 2
    finally:
        server._httpd.server_close()


def test_stop_flag_preserved_in_done_status(tmp_path: Path, monkeypatch) -> None:
    _write_dataset_yaml(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def fake_train(*args, **kwargs):
        started.set()
        release.wait(timeout=1.0)
        return SimpleNamespace(best_checkpoint="", last_checkpoint="", final_metrics={})

    monkeypatch.setattr(mata, "train", fake_train)

    server = _make_server(tmp_path)
    try:
        api_handler.dispatch(
            server,
            "POST",
            "/api/train",
            {
                "task": "detect",
                "model": "facebook/detr-resnet-50",
                "data": "project/dataset.yaml",
            },
        )
        assert started.wait(timeout=1.0)
        api_handler.dispatch(server, "POST", "/api/train/stop", {})
        release.set()
        done = _wait_for_status(server, "done")
        _join_training_thread(server)
        assert done["stop_requested"] is True
    finally:
        server._httpd.server_close()