"""Tests for configuration management."""

import tempfile
from pathlib import Path

from mata.core.config import MATAConfig, get_config, set_config


def test_default_config():
    """Test default configuration."""
    config = MATAConfig()

    assert config.default_device == "auto"
    assert config.default_models["detect"] == "rtdetr"
    assert config.log_level == "INFO"
    assert config.cache_dir is not None


def test_config_from_dict():
    """Test creating config from dictionary."""
    config_dict = {"default_device": "cuda", "default_models": {"detect": "dino"}, "log_level": "DEBUG"}

    config = MATAConfig.from_dict(config_dict)
    assert config.default_device == "cuda"
    assert config.default_models["detect"] == "dino"
    assert config.log_level == "DEBUG"


def test_config_save_load():
    """Test saving and loading config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"

        # Create and save config
        config1 = MATAConfig(default_device="cuda", log_level="DEBUG")
        config1.save(config_path)

        # Load config
        config2 = MATAConfig.from_file(config_path)

        assert config2.default_device == "cuda"
        assert config2.log_level == "DEBUG"


def test_global_config():
    """Test global config get/set."""
    custom_config = MATAConfig(default_device="cpu")
    set_config(custom_config)

    retrieved = get_config()
    assert retrieved.default_device == "cpu"
