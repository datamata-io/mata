"""Runtime abstraction layer for backend-agnostic inference.

Runtimes handle the actual model execution, allowing models to run
on different backends (PyTorch, ONNX, TensorRT) without changing adapter code.
"""

from typing import Any, Protocol


class Runtime(Protocol):
    """Protocol for inference runtimes.

    Runtimes abstract the execution backend, allowing the same model
    to run on PyTorch, ONNX Runtime, TensorRT, etc.
    """

    name: str

    def load_model(self, model_path: str, **kwargs: Any) -> Any:
        """Load a model from path.

        Args:
            model_path: Path to model file
            **kwargs: Backend-specific loading options

        Returns:
            Loaded model object
        """
        ...

    def infer(self, model: Any, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Run inference on inputs.

        Args:
            model: Loaded model object
            inputs: Input tensors/arrays as dictionary
            **kwargs: Backend-specific inference options

        Returns:
            Output tensors/arrays as dictionary
        """
        ...

    def get_device(self) -> str:
        """Get the device this runtime uses.

        Returns:
            Device string (e.g., "cuda:0", "cpu")
        """
        ...
