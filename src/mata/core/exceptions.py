"""Custom exceptions for MATA framework."""


class MATAError(Exception):
    """Base exception for all MATA errors."""

    pass


class TaskNotSupportedError(MATAError):
    """Raised when a task type is not supported."""

    def __init__(self, task: str, supported: list[str]) -> None:
        self.task = task
        self.supported = supported

        msg = f"Task '{task}' is not supported. " f"Supported tasks: {', '.join(supported)}"
        super().__init__(msg)


class MATARuntimeError(MATAError):
    """Raised when a runtime error occurs during execution."""

    pass


# Backward compatibility alias — deprecated, use MATARuntimeError
RuntimeError = MATARuntimeError


class InvalidInputError(MATAError):
    """Raised when input validation fails.

    This occurs when:
    - Image path does not exist
    - Image format is not supported
    - Input type is invalid
    """

    def __init__(self, message: str, input_value: any = None) -> None:
        self.input_value = input_value
        super().__init__(message)


class ModelLoadError(MATAError):
    """Raised when model loading fails.

    This typically occurs when:
    - Model checkpoint not found
    - Network error downloading model
    - Insufficient memory
    - Invalid model configuration
    """

    def __init__(self, model_id: str, reason: str) -> None:
        self.model_id = model_id
        self.reason = reason

        msg = (
            f"Failed to load model '{model_id}': {reason}. "
            f"Please check network connection, disk space, and model availability."
        )
        super().__init__(msg)


class ConfigurationError(MATAError):
    """Raised when configuration is invalid."""

    pass


class ModelNotFoundError(MATAError):
    """Raised when a model source cannot be found or resolved.

    This occurs when:
    - Model file path does not exist
    - HuggingFace model ID is invalid
    - Config alias not defined
    - Model source cannot be determined
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ValidationError(MATAError):
    """Raised when artifact or node validation fails.

    This occurs when:
    - Node inputs don't match declared types
    - Node outputs don't match declared types
    - Missing required inputs/outputs
    - Artifact validation fails
    - Type mismatches in graph wiring
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnsupportedModelError(MATAError):
    """Raised when a model format or architecture is not supported.

    This occurs when:
    - File extension is not recognized
    - Model architecture cannot be detected
    - Required adapter not implemented
    - Model format version incompatible
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
