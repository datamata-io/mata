"""Unit tests for the `mata annotate` CLI subcommand (Task G1).

Uses argparse parsing and mocked mata.annotate calls — no server starts.
Run independently: pytest tests/test_annotate_cli.py -v
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mata.cli import _build_parser, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(args: list[str]) -> int:
    """Invoke main() with explicit argv; return exit code."""
    return main(args)


def _capture(args: list[str]) -> tuple[int, str, str]:
    """Return (exit_code, stdout, stderr)."""
    out = StringIO()
    err = StringIO()
    with patch("sys.stdout", out), patch("sys.stderr", err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# Help / parser tests
# ---------------------------------------------------------------------------


def test_annotate_help_exits_zero() -> None:
    """mata annotate --help prints usage and exits 0."""
    with pytest.raises(SystemExit) as exc_info:
        _capture(["annotate", "--help"])
    assert exc_info.value.code == 0


def test_annotate_help_mentions_annotate() -> None:
    """mata annotate --help output references 'annotate'."""
    out = StringIO()
    with patch("sys.stdout", out), pytest.raises(SystemExit):
        main(["annotate", "--help"])
    assert "annotate" in out.getvalue().lower()


def test_annotate_subcommand_appears_in_global_help() -> None:
    """mata --help lists 'annotate' as an available subcommand."""
    out = StringIO()
    with patch("sys.stdout", out), pytest.raises(SystemExit):
        main(["--help"])
    assert "annotate" in out.getvalue()


def test_annotate_parser_default_data() -> None:
    """--data defaults to 'data'."""
    p = _build_parser()
    ns = p.parse_args(["annotate"])
    assert ns.data == "data"


def test_annotate_parser_default_host() -> None:
    """--host defaults to '127.0.0.1'."""
    p = _build_parser()
    ns = p.parse_args(["annotate"])
    assert ns.host == "127.0.0.1"


def test_annotate_parser_default_port() -> None:
    """--port defaults to 8710."""
    p = _build_parser()
    ns = p.parse_args(["annotate"])
    assert ns.port == 8710


def test_annotate_parser_no_browser_flag() -> None:
    """--no-browser sets no_browser=True."""
    p = _build_parser()
    ns = p.parse_args(["annotate", "--no-browser"])
    assert ns.no_browser is True


def test_annotate_parser_no_browser_default_false() -> None:
    """no_browser is False by default."""
    p = _build_parser()
    ns = p.parse_args(["annotate"])
    assert ns.no_browser is False


def test_annotate_port_argument() -> None:
    """--port N stores integer N."""
    p = _build_parser()
    ns = p.parse_args(["annotate", "--port", "9000"])
    assert ns.port == 9000


def test_annotate_data_argument() -> None:
    """--data PATH stores the supplied path string."""
    p = _build_parser()
    ns = p.parse_args(["annotate", "--data", "/tmp/mydata"])
    assert ns.data == "/tmp/mydata"


def test_annotate_host_argument() -> None:
    """--host ADDR stores the supplied host string."""
    p = _build_parser()
    ns = p.parse_args(["annotate", "--host", "0.0.0.0"])
    assert ns.host == "0.0.0.0"


def test_annotate_detect_model_argument() -> None:
    """--detect-model stores the model string."""
    p = _build_parser()
    ns = p.parse_args(["annotate", "--detect-model", "facebook/detr-resnet-50"])
    assert ns.detect_model == "facebook/detr-resnet-50"


def test_annotate_vlm_model_argument() -> None:
    """--vlm-model stores the model string."""
    p = _build_parser()
    ns = p.parse_args(["annotate", "--vlm-model", "Qwen/Qwen3-VL-2B-Instruct"])
    assert ns.vlm_model == "Qwen/Qwen3-VL-2B-Instruct"


def test_annotate_embed_model_argument() -> None:
    """--embed-model stores the model string."""
    p = _build_parser()
    ns = p.parse_args(["annotate", "--embed-model", "openai/clip-vit-base-patch32"])
    assert ns.embed_model == "openai/clip-vit-base-patch32"


# ---------------------------------------------------------------------------
# _cmd_annotate handler tests
# ---------------------------------------------------------------------------


def test_annotate_invalid_data_dir_returns_exit_code_1() -> None:
    """mata annotate --data /nonexistent returns exit code 1."""
    code, _out, _err = _capture(["annotate", "--data", "/nonexistent_path_xyz_abc"])
    assert code == 1


def test_annotate_invalid_data_dir_prints_error_to_stderr() -> None:
    """mata annotate with missing data dir prints error to stderr."""
    _code, _out, err = _capture(["annotate", "--data", "/nonexistent_path_xyz_abc"])
    assert "ERROR" in err or "error" in err.lower() or "not found" in err.lower()


def test_annotate_calls_mata_annotate_with_correct_args(tmp_path: Path) -> None:
    """_cmd_annotate calls mata.annotate() with parsed arguments."""
    mock_annotate = MagicMock(return_value=None)

    with patch("mata.annotate", mock_annotate, create=True):
        # Use a real directory so the path-existence check passes
        code = _run(["annotate", "--data", str(tmp_path), "--no-browser", "--port", "9999"])

    mock_annotate.assert_called_once()
    call_kwargs = mock_annotate.call_args.kwargs
    assert call_kwargs["data"] == str(tmp_path)
    assert call_kwargs["port"] == 9999
    assert call_kwargs["open_browser"] is False


def test_annotate_propagates_no_browser(tmp_path: Path) -> None:
    """--no-browser causes open_browser=False to be passed to mata.annotate."""
    mock_annotate = MagicMock(return_value=None)

    with patch("mata.annotate", mock_annotate, create=True):
        _run(["annotate", "--data", str(tmp_path), "--no-browser"])

    kwargs = mock_annotate.call_args.kwargs
    assert kwargs.get("open_browser") is False


def test_annotate_keyboard_interrupt_exits_zero(tmp_path: Path) -> None:
    """KeyboardInterrupt during mata.annotate() results in exit code 0."""
    with patch("mata.annotate", side_effect=KeyboardInterrupt, create=True):
        code, out, _err = _capture(["annotate", "--data", str(tmp_path)])

    assert code == 0
    assert "stopped" in out.lower() or "server stopped" in out.lower()


def test_annotate_exception_returns_exit_code_1(tmp_path: Path) -> None:
    """Unexpected exception from mata.annotate() returns exit code 1."""
    with patch("mata.annotate", side_effect=RuntimeError("boom"), create=True):
        code, _out, err = _capture(["annotate", "--data", str(tmp_path)])

    assert code == 1
    assert "ERROR" in err or "boom" in err
