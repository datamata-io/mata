"""Checkpoint manager for saving, loading, and exporting training state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mata.core.exceptions import TrainingError
from mata.core.logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Save, load, and export training checkpoints.

    Can be instantiated with an optional *checkpoint_dir* for convenience::

        ckpt = CheckpointManager("runs/train/detect/best")
        ckpt.export_for_inference(output_dir="runs/train/detect/best/export")

    Or used statelessly by passing *checkpoint_dir* to each method directly.

    Checkpoint directory layout::

        checkpoint_dir/
        ├── model_state.pth        # torch.save(model.state_dict())
        ├── optimizer_state.pth    # torch.save({optimizer, scheduler state dicts})
        ├── training_state.json    # {"epoch": N, "best_metric": X, "history": {...}}
        └── config.json            # {"model_source": "...", "task": "...", "engine": "..."}

    HuggingFace inference export layout::

        export_dir/
        ├── config.json                 # HF model config
        ├── model.safetensors           # HF model weights
        └── preprocessor_config.json   # processor config (if processor supplied)

    Torchvision inference export layout::

        export_dir/
        ├── model.pth      # torch.save(model.state_dict())
        └── metadata.json  # {"model_source": "...", "task": "...", "num_classes": N, ...}
    """

    def __init__(self, checkpoint_dir: str | Path | None = None) -> None:
        """Create a CheckpointManager.

        Args:
            checkpoint_dir: Optional default checkpoint directory.  When set,
                :meth:`export_for_inference` and :meth:`load` will use this
                path if no explicit *checkpoint_dir* argument is supplied.
        """
        self.checkpoint_dir: Path | None = Path(checkpoint_dir) if checkpoint_dir is not None else None

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        epoch: int,
        metrics: Any,
        config: Any,
        path: str | Path,
    ) -> Path:
        """Save a training checkpoint.

        Creates the directory at *path* (if needed) and writes four files:
        ``model_state.pth``, ``optimizer_state.pth``, ``training_state.json``,
        and ``config.json``.

        Args:
            model: PyTorch model (``nn.Module``).
            optimizer: Optimizer (``torch.optim.Optimizer``), or ``None``.
            scheduler: LR scheduler, or ``None``.
            epoch: Current epoch index (0-based or 1-based — caller's choice).
            metrics: Metrics object or dict (used to extract ``best_metric``
                scalar for JSON storage).
            config: ``TrainingConfig`` instance (or any object / dict whose
                attributes describe the run).
            path: Directory to write the checkpoint into.

        Returns:
            Resolved ``Path`` of the checkpoint directory.

        Raises:
            TrainingError: If any file cannot be written.
        """
        try:
            import torch  # lazy import
        except ImportError as exc:  # pragma: no cover
            raise TrainingError("PyTorch is required to save checkpoints.") from exc

        ckpt_dir = Path(path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Model weights (state_dict only — safe serialisation)
            torch.save(model.state_dict(), ckpt_dir / "model_state.pth")

            # 2. Optimizer + scheduler state
            opt_state: dict[str, Any] = {}
            if optimizer is not None:
                opt_state["optimizer"] = optimizer.state_dict()
            if scheduler is not None:
                opt_state["scheduler"] = scheduler.state_dict()
            torch.save(opt_state, ckpt_dir / "optimizer_state.pth")

            # 3. Training state (pure JSON — no pickle)
            best_metric = _extract_scalar_metric(metrics)
            history: dict[str, list[float]] = {}
            if hasattr(config, "history"):  # populated externally by engines
                history = config.history  # type: ignore[attr-defined]
            training_state = {
                "epoch": epoch,
                "best_metric": best_metric,
                "history": history,
            }
            with open(ckpt_dir / "training_state.json", "w", encoding="utf-8") as fh:
                json.dump(training_state, fh, indent=2)

            # 4. Run config (pure JSON)
            config_data = _config_to_dict(config)
            with open(ckpt_dir / "config.json", "w", encoding="utf-8") as fh:
                json.dump(config_data, fh, indent=2)

        except OSError as exc:
            raise TrainingError(f"Failed to write checkpoint to '{ckpt_dir}': {exc}") from exc

        logger.debug("Checkpoint saved to %s", ckpt_dir)
        return ckpt_dir

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a checkpoint from *path*.

        Uses ``weights_only=True`` for model weights (CVE-2025-32434
        mitigation) and ``weights_only=False`` for optimizer state (safe
        because these files are written by MATA internally, never by end
        users).

        Args:
            path: Checkpoint directory (same directory passed to
                :meth:`save`).

        Returns:
            Dict with keys:
            - ``"model_state"`` — ``OrderedDict`` suitable for
              ``model.load_state_dict()``.
            - ``"optimizer_state"`` — dict with ``"optimizer"`` and
              optionally ``"scheduler"`` state dicts (empty dict if file
              absent).
            - ``"training_state"`` — parsed ``training_state.json``.
            - ``"config"`` — parsed ``config.json``.

        Raises:
            TrainingError: If required files are missing or cannot be parsed.
        """
        try:
            import torch  # lazy import
        except ImportError as exc:  # pragma: no cover
            raise TrainingError("PyTorch is required to load checkpoints.") from exc

        ckpt_dir = Path(path)
        if not ckpt_dir.is_dir():
            raise TrainingError(
                f"Checkpoint directory not found: '{ckpt_dir}'. "
                "Ensure the path points to the directory created by CheckpointManager.save()."
            )

        result: dict[str, Any] = {}

        # Model weights — weights_only=True (security)
        model_pth = ckpt_dir / "model_state.pth"
        if not model_pth.exists():
            raise TrainingError(f"model_state.pth not found in '{ckpt_dir}'.")
        try:
            result["model_state"] = torch.load(
                model_pth, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise TrainingError(f"Failed to load model_state.pth: {exc}") from exc

        # Optimizer / scheduler state — weights_only=False (internal file)
        opt_pth = ckpt_dir / "optimizer_state.pth"
        if opt_pth.exists():
            try:
                result["optimizer_state"] = torch.load(
                    opt_pth, map_location="cpu", weights_only=False
                )
            except Exception as exc:
                raise TrainingError(f"Failed to load optimizer_state.pth: {exc}") from exc
        else:
            result["optimizer_state"] = {}

        # Training state JSON
        ts_path = ckpt_dir / "training_state.json"
        if not ts_path.exists():
            raise TrainingError(f"training_state.json not found in '{ckpt_dir}'.")
        try:
            with open(ts_path, "r", encoding="utf-8") as fh:
                result["training_state"] = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingError(f"Failed to parse training_state.json: {exc}") from exc

        # Config JSON (optional — older checkpoints may not have it)
        cfg_path = ckpt_dir / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    result["config"] = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                raise TrainingError(f"Failed to parse config.json: {exc}") from exc
        else:
            result["config"] = {}

        logger.debug("Checkpoint loaded from %s", ckpt_dir)
        return result

    # ------------------------------------------------------------------
    # export_for_inference
    # ------------------------------------------------------------------

    def export_for_inference(
        self,
        checkpoint_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        model: Any = None,
        processor: Any = None,
        engine: str | None = None,
    ) -> Path:
        """Export a training checkpoint for inference via ``mata.load()``.

        Detects whether the checkpoint was produced by a HuggingFace or
        torchvision-based training run by inspecting ``config.json`` in
        *checkpoint_dir*.  The *engine* argument overrides auto-detection when
        provided.

        - **HuggingFace model**: calls ``model.save_pretrained(output_dir)``
          and optionally ``processor.save_pretrained(output_dir)``.
        - **Torchvision model**: saves ``model.pth`` + ``metadata.json``.

        If *model* is ``None`` the function only copies / rewrites files from
        the checkpoint directory — useful when the model object is unavailable
        at export time.

        Args:
            checkpoint_dir: Directory previously created by :meth:`save`.
            output_dir: Target directory for the inference-ready export.
            model: Live model object (``nn.Module`` or HF ``PreTrainedModel``).
                If ``None``, the raw ``model_state.pth`` is copied over.
            processor: HF processor/image-processor to export alongside the
                model.  Ignored for torchvision exports.

        Returns:
            Resolved ``Path`` of *output_dir*.

        Raises:
            TrainingError: If checkpoint metadata cannot be read or export fails.
        """
        try:
            import torch  # lazy import
        except ImportError as exc:  # pragma: no cover
            raise TrainingError("PyTorch is required for checkpoint export.") from exc

        # Resolve checkpoint_dir: explicit arg > instance default
        resolved_dir = checkpoint_dir if checkpoint_dir is not None else self.checkpoint_dir
        if resolved_dir is None:
            raise TrainingError(
                "checkpoint_dir must be provided either to the constructor "
                "or as an argument to export_for_inference()."
            )
        if output_dir is None:
            raise TrainingError("output_dir must be provided.")

        ckpt_dir = Path(resolved_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load config.json to determine engine; engine arg overrides auto-detection
        cfg_path = ckpt_dir / "config.json"
        detected_engine = "huggingface"  # default
        config_data: dict[str, Any] = {}
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    config_data = json.load(fh)
                detected_engine = config_data.get("engine", "huggingface")
            except (OSError, json.JSONDecodeError) as exc:
                raise TrainingError(f"Cannot read config.json from checkpoint: {exc}") from exc

        resolved_engine = engine if engine is not None else detected_engine

        if resolved_engine == "torchvision":
            self._export_torchvision(
                ckpt_dir, out_dir, model, config_data, torch
            )
        else:
            self._export_huggingface(
                ckpt_dir, out_dir, model, processor, config_data, torch
            )

        logger.info("Exported inference checkpoint to %s", out_dir)
        return out_dir

    def _export_huggingface(
        self,
        ckpt_dir: Path,
        out_dir: Path,
        model: Any,
        processor: Any,
        config_data: dict,
        torch: Any,
    ) -> None:
        if model is not None and hasattr(model, "save_pretrained"):
            model.save_pretrained(out_dir)
        else:
            # No live model — copy the raw state dict
            import shutil
            src = ckpt_dir / "model_state.pth"
            if src.exists():
                shutil.copy2(src, out_dir / "model_state.pth")
            # Copy HF config files if present
            for fname in ("config.json", "generation_config.json"):
                src_f = ckpt_dir / fname
                if src_f.exists():
                    import shutil as _shutil
                    _shutil.copy2(src_f, out_dir / fname)

        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(out_dir)

    def _export_torchvision(
        self,
        ckpt_dir: Path,
        out_dir: Path,
        model: Any,
        config_data: dict,
        torch: Any,
    ) -> None:
        if model is not None:
            torch.save(model.state_dict(), out_dir / "model.pth")
        else:
            # Copy existing model_state.pth → model.pth
            import shutil
            src = ckpt_dir / "model_state.pth"
            if src.exists():
                shutil.copy2(src, out_dir / "model.pth")

        metadata = {
            "model_source": config_data.get("model_source", ""),
            "task": config_data.get("task", ""),
            "engine": "torchvision",
            "num_classes": config_data.get("num_classes"),
            "class_names": config_data.get("class_names"),
        }
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    # ------------------------------------------------------------------
    # list_checkpoints
    # ------------------------------------------------------------------

    def list_checkpoints(self, save_dir: str | Path) -> list[str]:
        """Return sorted list of checkpoint directory paths under *save_dir*.

        A directory is considered a valid checkpoint if it contains
        ``model_state.pth``.

        Args:
            save_dir: Root training output directory (e.g. ``runs/train/detect``).

        Returns:
            Sorted list of absolute checkpoint directory path strings.
        """
        base = Path(save_dir)
        if not base.is_dir():
            return []

        checkpoints: list[str] = []
        for entry in base.iterdir():
            if entry.is_dir() and (entry / "model_state.pth").exists():
                checkpoints.append(str(entry.resolve()))

        return sorted(checkpoints)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_scalar_metric(metrics: Any) -> float | None:
    """Try to extract a single float from a metrics object or dict."""
    if metrics is None:
        return None
    if isinstance(metrics, (int, float)):
        return float(metrics)
    if isinstance(metrics, dict):
        for key in ("map50", "map", "top1", "fitness", "loss"):
            if key in metrics:
                val = metrics[key]
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return None
    # Object with common metric attributes
    for attr in ("map50", "map", "top1", "fitness"):
        val = getattr(metrics, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def _config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a TrainingConfig (or any object/dict) to a JSON-safe dict."""
    if isinstance(config, dict):
        return {k: v for k, v in config.items() if _json_serialisable(v)}
    data: dict[str, Any] = {}
    for attr in (
        "task",
        "model",
        "data",
        "val_data",
        "epochs",
        "batch_size",
        "lr",
        "optimizer",
        "weight_decay",
        "scheduler",
        "warmup_epochs",
        "device",
        "amp",
        "save_dir",
        "save_every",
        "val_every",
        "patience",
        "freeze_backbone",
        "freeze_layers",
        "augment",
        "resume",
        "num_workers",
        "seed",
        "verbose",
        # engine / export metadata fields (may not exist on TrainingConfig)
        "engine",
        "model_source",
        "num_classes",
        "class_names",
    ):
        val = getattr(config, attr, None)
        if val is not None and _json_serialisable(val):
            data[attr] = val
    # Ensure engine key always present (training engines set this)
    if "engine" not in data:
        data["engine"] = getattr(config, "engine", "huggingface")
    # Alias model → model_source for export helpers
    if "model_source" not in data and "model" in data:
        data["model_source"] = data["model"]
    return data


def _json_serialisable(value: Any) -> bool:
    """Return True if *value* is directly JSON-serialisable."""
    return isinstance(value, (str, int, float, bool, list, dict, type(None)))
