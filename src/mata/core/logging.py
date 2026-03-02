"""Simple console logging for MATA framework."""

import logging
import os
import sys
import warnings
from contextlib import contextmanager

# Global logger instance
_logger: logging.Logger | None = None

# Global verbosity level:
#   0 = silent   — suppress MATA logs AND third-party noise
#   1 = quiet    — suppress third-party noise only (default)
#   2 = verbose  — show everything (third-party noise visible)
_verbosity: int = 1


def get_logger(name: str = "mata") -> logging.Logger:
    """Get or create the MATA logger.

    Args:
        name: Logger name (default: "mata")

    Returns:
        Configured logger instance
    """
    global _logger

    if _logger is None:
        _logger = logging.getLogger(name)
        _logger.setLevel(logging.INFO)

        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Simple format
        formatter = logging.Formatter(fmt="[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)

        _logger.addHandler(handler)
        _logger.propagate = False

    return _logger


def set_log_level(level: str) -> None:
    """Set the logging level.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger = get_logger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)


# Convenience functions
def debug(msg: str) -> None:
    """Log debug message."""
    get_logger().debug(msg)


def info(msg: str) -> None:
    """Log info message."""
    get_logger().info(msg)


def warning(msg: str) -> None:
    """Log warning message."""
    get_logger().warning(msg)


def error(msg: str) -> None:
    """Log error message."""
    get_logger().error(msg)


def critical(msg: str) -> None:
    """Log critical message."""
    get_logger().critical(msg)


def verbose(level: int = 2) -> None:
    """Control MATA's output verbosity.

    Args:
        level: Verbosity level:
            - ``0`` (silent): Suppress *all* output — both MATA's ``[INFO]``
              messages **and** third-party noise (tqdm, transformers, etc.).
            - ``1`` (quiet, **default**): Show MATA logs, suppress third-party
              noise during model loading.
            - ``2`` (verbose): Show everything — useful for debugging model
              loading issues.

    Examples::

        import mata

        mata.verbose(0)   # total silence
        mata.verbose(1)   # only MATA logs (default)
        mata.verbose(2)   # MATA logs + third-party output
    """
    global _verbosity
    if level not in (0, 1, 2):
        raise ValueError(f"verbosity must be 0, 1, or 2 — got {level}")
    _verbosity = level

    logger = get_logger()
    if level == 0:
        # Silence MATA's own logs
        logger.setLevel(logging.WARNING)
        for handler in logger.handlers:
            handler.setLevel(logging.WARNING)
    else:
        # Restore MATA logs
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            handler.setLevel(logging.INFO)


def get_verbosity() -> int:
    """Return the current verbosity level (0, 1, or 2)."""
    return _verbosity


@contextmanager
def suppress_third_party_logs():
    """Suppress noisy third-party library output during model loading.

    Temporarily silences:
    - transformers warnings (fast processor notices, unexpected keys)
    - tqdm/huggingface_hub progress bars
    - matplotlib backend messages

    Usage::

        from mata.core.logging import suppress_third_party_logs

        with suppress_third_party_logs():
            model = Model.from_pretrained(model_id)
    """
    # When verbosity >= 2, skip all suppression (pass-through).
    if _verbosity >= 2:
        yield
        return

    # --- transformers logging ---
    prev_transformers_verbosity = None
    try:
        import transformers

        prev_transformers_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
    except ImportError:
        pass

    # --- huggingface_hub progress bars ---
    prev_hf_disable = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # --- tqdm progress bars (safetensors weight materialization, etc.) ---
    prev_tqdm_disable = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"

    # --- Redirect stderr to suppress tqdm progress bars ---
    # tqdm writes directly to stderr, bypassing Python logging.
    # Redirecting stderr is the only reliable way to silence it.
    # MATA's own logs go to stdout, so they remain visible.
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")

    # --- Python warnings (ViTImageProcessor, weight-only deprecation, etc.) ---
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*fast processor.*", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*ViTImageProcessor.*")
        warnings.filterwarnings("ignore", message=".*Torch was not compiled.*")

        # --- stdlib loggers for tqdm, matplotlib, urllib3, etc. ---
        noisy_loggers = [
            "transformers",
            "transformers.modeling_utils",
            "transformers.configuration_utils",
            "transformers.tokenization_utils_base",
            "huggingface_hub",
            "huggingface_hub.file_download",
            "matplotlib",
            "matplotlib.pyplot",
            "PIL",
            "tqdm",
            "urllib3",
            "filelock",
            "safetensors",
        ]
        saved_levels: dict[str, int] = {}
        for name in noisy_loggers:
            lg = logging.getLogger(name)
            saved_levels[name] = lg.level
            lg.setLevel(logging.ERROR)

        try:
            yield
        finally:
            # --- Restore stderr ---
            sys.stderr.close()
            sys.stderr = old_stderr

            # --- Restore logger levels ---
            for name, level in saved_levels.items():
                logging.getLogger(name).setLevel(level)

            if prev_transformers_verbosity is not None:
                try:
                    transformers.logging.set_verbosity(prev_transformers_verbosity)
                except Exception:
                    pass

            if prev_hf_disable is None:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            else:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_hf_disable

            if prev_tqdm_disable is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = prev_tqdm_disable
