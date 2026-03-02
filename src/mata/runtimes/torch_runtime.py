"""PyTorch runtime implementation."""

from typing import Any

import torch


class TorchRuntime:
    """PyTorch inference runtime.

    Executes models using PyTorch backend with automatic device management.
    """

    name: str = "torch"

    def __init__(self, device: str = "auto") -> None:
        """Initialize PyTorch runtime.

        Args:
            device: Device to use ("cuda", "cpu", or "auto" for automatic selection)
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def load_model(self, model_path: str, **kwargs: Any) -> torch.nn.Module:
        """Load a PyTorch model from path.

        Args:
            model_path: Path to .pt or .pth file
            **kwargs: Additional torch.load options

        Returns:
            Loaded PyTorch model
        """
        model = torch.load(model_path, map_location=self.device, **kwargs)
        if isinstance(model, torch.nn.Module):
            model.eval()
        return model

    def infer(self, model: torch.nn.Module, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Run inference with PyTorch model.

        Args:
            model: PyTorch model
            inputs: Dictionary of input tensors
            **kwargs: Additional inference options

        Returns:
            Dictionary of output tensors
        """
        with torch.no_grad():
            # Move inputs to device
            device_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Run model
            outputs = model(**device_inputs)

            # Convert outputs to dict if needed
            if isinstance(outputs, torch.Tensor):
                return {"output": outputs}
            elif isinstance(outputs, dict):
                return outputs
            else:
                # Handle model-specific output types (e.g., transformers outputs)
                return {"output": outputs}

    def get_device(self) -> str:
        """Get the device this runtime uses.

        Returns:
            Device string (e.g., "cuda:0", "cpu")
        """
        return str(self.device)
