"""Configuration management for MATA framework."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MATAConfig:
    """MATA framework configuration.

    Attributes:
        default_device: Default device for inference ("cuda", "cpu", or "auto")
        default_models: Mapping of task to default model name
        cache_dir: Directory for model weights and cache
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """

    default_device: str = "auto"
    default_models: dict[str, str | None] = field(
        default_factory=lambda: {
            "detect": "rtdetr",
            "segment": None,
            "classify": None,
            "track": None,
        }
    )
    cache_dir: Path | None = None
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate and set defaults."""
        if self.cache_dir is None:
            # Default to ~/.cache/mata
            self.cache_dir = Path.home() / ".cache" / "mata"
        elif isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> MATAConfig:
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            MATAConfig instance
        """
        return cls(**config_dict)

    @classmethod
    def from_file(cls, config_path: Path | str) -> MATAConfig:
        """Load configuration from JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            MATAConfig instance
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path) as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_device": self.default_device,
            "default_models": self.default_models,
            "cache_dir": str(self.cache_dir),
            "log_level": self.log_level,
        }

    def save(self, config_path: Path | str) -> None:
        """Save configuration to JSON file.

        Args:
            config_path: Path to save configuration
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Global configuration instance
_config: MATAConfig | None = None


def get_config() -> MATAConfig:
    """Get global configuration instance.

    Returns:
        Current MATAConfig instance
    """
    global _config
    if _config is None:
        # Try to load from environment variable
        config_path = os.environ.get("MATA_CONFIG")
        if config_path and Path(config_path).exists():
            _config = MATAConfig.from_file(config_path)
        else:
            _config = MATAConfig()
    return _config


def set_config(config: MATAConfig) -> None:
    """Set global configuration instance.

    Args:
        config: New configuration instance
    """
    global _config
    _config = config
