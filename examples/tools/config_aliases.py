"""Config Aliases — MATA Framework

Demonstrates how to set up and use model aliases via configuration files,
allowing you to define shortcuts for commonly used models.

Covers:
- Writing a project-local .mata/models.yaml config
- Loading models by alias
- Runtime model registration with mata.register_model()

Run: python examples/tools/config_aliases.py
"""

from pathlib import Path

import mata


def setup_example_config():
    """Create an example config file at .mata/models.yaml."""
    config_dir = Path.cwd() / ".mata"
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / "models.yaml"

    config_content = """# Project-specific model configuration
detect:
  # Fast model for development/testing
  dev:
    source: "facebook/detr-resnet-50"
    threshold: 0.3
    device: cpu

  # Production model with higher accuracy
  prod:
    source: "PekingU/rtdetr_v2_r101vd"
    threshold: 0.5
    device: cuda

  # Custom local ONNX model (uncomment to use)
  # custom:
  #   source: "models/custom_detector.onnx"
  #   threshold: 0.4
"""

    config_path.write_text(config_content)
    return config_path


def main():
    print("MATA — Config Alias Examples\n")

    # Setup example config
    config_path = setup_example_config()
    print(f"✓ Created config: {config_path}\n")

    # Load model using alias defined in .mata/models.yaml
    detector = mata.load("detect", "dev")
    print(f"✓ Loaded 'dev' alias: {detector.info()}")

    # Runtime registration (no config file needed)
    mata.register_model("detect", "runtime-model", "PekingU/rtdetr_v2_r50vd", threshold=0.45, device="auto")

    list_models = mata.list_models()
    print(f"✓ Registered models: {list_models}\n")

    detector = mata.load("detect", "runtime-model")
    print(f"✓ Loaded 'runtime-model' alias: {detector.info()}\n")


if __name__ == "__main__":
    main()
