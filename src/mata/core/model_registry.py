"""Model registry for managing config-based model aliases.

This module provides YAML-based configuration for model aliases, allowing users to
define shortcuts for commonly used models.

Config file locations (in order of precedence):
1. Project-local: .mata/models.yaml (in current working directory)
2. User global: ~/.mata/models.yaml
"""

from pathlib import Path
from typing import Any

import yaml

from mata.core.exceptions import ModelNotFoundError
from mata.core.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Registry for model configuration aliases.

    Manages model configurations from YAML files, supporting both user-global
    and project-local configurations.

    Config file format:
        detect:
          rtdetr-fast:
            source: "facebook/detr-resnet-50"
            threshold: 0.3
            device: auto

          rtdetr-onnx:
            source: "/path/to/model.onnx"
            threshold: 0.4

        segment:
          mask2former:
            source: "facebook/mask2former-swin-base-coco-instance"
            threshold: 0.5

    Examples:
        >>> registry = ModelRegistry()
        >>> # Check if alias exists
        >>> registry.has_alias("detect", "rtdetr-fast")
        True
        >>> # Get config for alias
        >>> config = registry.get_config("detect", "rtdetr-fast")
        >>> # List all aliases for a task
        >>> aliases = registry.list_aliases("detect")
    """

    def __init__(self, custom_config_path: str | None = None):
        """Initialize model registry.

        Args:
            custom_config_path: Optional custom config file path.
                If not provided, searches standard locations.
        """
        self._configs: dict[str, dict[str, dict[str, Any]]] = {}
        self._runtime_configs: dict[str, dict[str, dict[str, Any]]] = {}
        self._loaded = False
        self._custom_path = custom_config_path

    def _ensure_loaded(self):
        """Ensure config files are loaded (lazy loading)."""
        if not self._loaded:
            self._load_configs()
            self._loaded = True

    def _load_configs(self):
        """Load configurations from YAML files.

        Search order:
        1. Custom path (if provided)
        2. Project-local: .mata/models.yaml
        3. User global: ~/.mata/models.yaml
        """
        configs_to_load = []

        if self._custom_path:
            configs_to_load.append(Path(self._custom_path))
        else:
            # User global config
            user_config = Path.home() / ".mata" / "models.yaml"
            if user_config.exists():
                configs_to_load.append(user_config)

            # Project-local config (higher precedence)
            project_config = Path.cwd() / ".mata" / "models.yaml"
            if project_config.exists():
                configs_to_load.append(project_config)

        # Load configs (later ones override earlier ones)
        for config_path in configs_to_load:
            try:
                self._load_yaml(config_path)
                logger.info(f"Loaded model config from: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

    def _load_yaml(self, config_path: Path):
        """Load and merge a YAML config file.

        Args:
            config_path: Path to YAML config file
        """
        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning(f"Empty config file: {config_path}")
                return

            # Merge into existing configs
            for task, aliases in data.items():
                if task not in self._configs:
                    self._configs[task] = {}
                if isinstance(aliases, dict):
                    self._configs[task].update(aliases)

        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading {config_path}: {e}")
            raise

    def has_alias(self, task: str, alias: str) -> bool:
        """Check if an alias exists for a task.

        Args:
            task: Task type (e.g., "detect", "segment")
            alias: Alias name

        Returns:
            True if alias exists
        """
        self._ensure_loaded()

        # Check runtime configs first
        if task in self._runtime_configs and alias in self._runtime_configs[task]:
            return True

        # Check file-based configs
        return task in self._configs and alias in self._configs[task]

    def get_config(self, task: str, alias: str) -> dict[str, Any]:
        """Get configuration for a model alias.

        Validates and normalizes model_type field if present in config.
        Provides lint-level warnings for unknown config fields.

        Args:
            task: Task type
            alias: Alias name

        Returns:
            Configuration dictionary with normalized model_type

        Raises:
            ModelNotFoundError: If alias not found
        """
        self._ensure_loaded()

        # Check runtime configs first (higher precedence)
        if task in self._runtime_configs and alias in self._runtime_configs[task]:
            config = self._runtime_configs[task][alias].copy()
            return self._validate_and_normalize_config(task, alias, config)

        # Check file-based configs
        if task in self._configs and alias in self._configs[task]:
            config = self._configs[task][alias].copy()
            return self._validate_and_normalize_config(task, alias, config)

        # Not found
        available = self.list_aliases(task)
        raise ModelNotFoundError(
            f"Model alias '{alias}' not found for task '{task}'. "
            f"Available aliases: {available if available else 'none'}. "
            f"You can add aliases in ~/.mata/models.yaml or .mata/models.yaml"
        )

    def _validate_and_normalize_config(self, task: str, alias: str, config: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize config fields (lint-level).

        Args:
            task: Task type
            alias: Alias name
            config: Raw config dictionary

        Returns:
            Normalized config dictionary
        """
        from mata.core.logging import get_logger
        from mata.core.types import ModelType

        logger = get_logger(__name__)

        # Known valid config fields
        KNOWN_FIELDS = {  # noqa: N806
            "source",
            "model_type",
            "threshold",
            "device",
            "config",
            "input_size",
            "id2label",
            "checkpoint_path",
            "model_path",
            "engine_path",
            "normalize",
            "target_size",
            # Tracking-specific fields (task == "track")
            "tracker",
            "tracker_config",
            "frame_rate",
        }

        # Lint-level warning for unknown fields (non-blocking)
        unknown_fields = set(config.keys()) - KNOWN_FIELDS
        if unknown_fields:
            logger.warning(
                f"Config '{task}/{alias}' has unknown fields: {unknown_fields}. "
                f"Known fields: {sorted(KNOWN_FIELDS)}. "
                f"These may be ignored by adapters."
            )

        # Normalize model_type if present
        if "model_type" in config:
            raw_type = config["model_type"]
            try:
                # Use ModelType.normalize() for consistent handling
                normalized = ModelType.normalize(raw_type)
                if normalized:
                    config["model_type"] = normalized
                else:
                    # normalize() returned None (invalid type)
                    logger.warning(
                        f"Config '{task}/{alias}' has invalid model_type: '{raw_type}'. "
                        f"Removing from config, will use auto-detection."
                    )
                    del config["model_type"]
            except Exception as e:
                logger.warning(
                    f"Error normalizing model_type in config '{task}/{alias}': {e}. "
                    f"Removing from config, will use auto-detection."
                )
                if "model_type" in config:
                    del config["model_type"]

        return config

    def list_aliases(self, task: str) -> list[str]:
        """List all aliases for a task.

        Args:
            task: Task type

        Returns:
            List of alias names
        """
        self._ensure_loaded()

        aliases = set()

        # Add file-based aliases
        if task in self._configs:
            aliases.update(self._configs[task].keys())

        # Add runtime aliases
        if task in self._runtime_configs:
            aliases.update(self._runtime_configs[task].keys())

        return sorted(aliases)

    def register(self, task: str, alias: str, config: dict[str, Any]):
        """Register a model alias at runtime.

        This allows programmatic registration without modifying config files.
        Runtime registrations take precedence over file-based configs.

        Args:
            task: Task type
            alias: Alias name
            config: Configuration dictionary (must include 'source')

        Raises:
            ValueError: If config is invalid

        Examples:
            >>> registry = ModelRegistry()
            >>> registry.register("detect", "my-model", {
            ...     "source": "/path/to/model.onnx",
            ...     "threshold": 0.5,
            ...     "device": "cuda"
            ... })
        """
        if "source" not in config:
            raise ValueError("Config must include 'source' field")

        if task not in self._runtime_configs:
            self._runtime_configs[task] = {}

        self._runtime_configs[task][alias] = config.copy()
        logger.info(f"Registered runtime alias '{alias}' for task '{task}'")

    def get_default(self, task: str) -> str | None:
        """Get the default alias for a task.

        Checks for a special 'default' alias in the config.

        Args:
            task: Task type

        Returns:
            Default alias name or None if not configured
        """
        self._ensure_loaded()

        # Check if there's a 'default' alias defined
        if self.has_alias(task, "default"):
            config = self.get_config(task, "default")
            # If default points to a source, return None (use source directly)
            # If default points to another alias, return that alias
            source = config.get("source", "")
            if "/" not in source and not Path(source).exists():
                # Likely an alias reference
                return source

        return None

    def save_to_file(self, file_path: str | None = None):
        """Save current runtime configurations to a YAML file.

        Args:
            file_path: Target file path. If None, uses ~/.mata/models.yaml
        """
        if not file_path:
            file_path = str(Path.home() / ".mata" / "models.yaml")

        # Create directory if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Merge runtime configs with existing file configs
        merged = {}
        for task, aliases in {**self._configs, **self._runtime_configs}.items():
            merged[task] = aliases

        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(merged, f, default_flow_style=False, sort_keys=True)

        logger.info(f"Saved model registry to: {file_path}")

    def get_valkey_connection(self, name: str = "default") -> dict[str, Any]:
        """Get named Valkey connection config from the YAML ``storage`` section.

        Args:
            name: Connection profile name defined under ``storage.valkey`` in
                the YAML config (e.g., "default", "production").

        Returns:
            Dict with connection parameters (url, db, ttl, tls, password, …).
            ``password_env`` is resolved from the environment and replaced by
            ``password``; the env-var name is never returned.

        Raises:
            ModelNotFoundError: If the named connection is not present in the
                config.  Missing ``storage`` section is treated as an empty
                config (not an error).

        Examples:
            >>> registry = ModelRegistry()
            >>> conn = registry.get_valkey_connection("default")
            >>> # {"url": "valkey://localhost:6379", "db": 0, "ttl": 3600}
        """
        self._ensure_loaded()

        storage = self._configs.get("storage", {}).get("valkey", {})
        if name not in storage:
            raise ModelNotFoundError(
                f"Valkey connection '{name}' not found in config. " f"Available: {list(storage.keys())}"
            )

        conn = dict(storage[name])

        # Resolve password from environment variable — never expose plaintext
        if "password_env" in conn:
            import os

            env_var = conn.pop("password_env")
            password = os.environ.get(env_var)
            if password:
                conn["password"] = password

        return conn
