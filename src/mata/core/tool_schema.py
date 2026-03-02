"""Tool schema definitions for VLM agent tool-calling system.

This module defines the core data structures that describe tools available to VLMs,
parse tool invocations from VLM output, and format tool execution results.

Design principles:
- Frozen dataclasses for immutability (matches Entity, Instance pattern)
- Decoupled from adapter implementations
- Multiple serialization formats (prompt text, OpenAI schema)
- Support for both MATA task tools and built-in image tools

Version: 1.7.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolParameter:
    """Describes a single parameter for a tool.

    Attributes:
        name: Parameter name (e.g., "threshold", "region")
        type: Type descriptor string ("float", "int", "str", "list[str]", "bbox")
        description: Human-readable description for VLM prompt
        required: Whether this parameter must be provided
        default: Default value if not provided (only valid when required=False)

    Examples:
        >>> param = ToolParameter("threshold", "float", "Confidence threshold", required=False, default=0.3)
        >>> param.name
        'threshold'
    """

    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
            "default": self.default,
        }


@dataclass(frozen=True)
class ToolSchema:
    """Describes a tool that can be invoked by a VLM.

    A tool can be either:
    - A MATA task adapter (detect, classify, segment, depth)
    - A built-in image operation (zoom, crop)

    The schema is injected into the VLM's system prompt so it knows what tools
    are available and how to call them.

    Attributes:
        name: Tool identifier (e.g., "detect", "classify", "zoom")
        description: Human-readable description for VLM prompt
        task: MATA task type ("detect", "classify", "segment", "depth", "image")
        parameters: List of parameters this tool accepts
        builtin: True for built-in image tools (zoom, crop), False for task adapters

    Examples:
        >>> schema = ToolSchema(
        ...     name="detect",
        ...     description="Run object detection",
        ...     task="detect",
        ...     parameters=[ToolParameter("threshold", "float", "Confidence threshold")],
        ... )
        >>> schema.name
        'detect'
    """

    name: str
    description: str
    task: str
    parameters: list[ToolParameter] = field(default_factory=list)
    builtin: bool = False

    def to_prompt_str(self) -> str:
        """Format as text block for VLM system prompt injection.

        Produces a human-readable tool description that VLMs can understand.
        Format is optimized for Qwen3-VL, LLaVA, and similar open-weight VLMs.

        Returns:
            Formatted string describing the tool and its parameters.

        Examples:
            >>> schema = ToolSchema("detect", "Run object detection", "detect",
            ...     [ToolParameter("threshold", "float", "Min confidence", False, 0.3)])
            >>> print(schema.to_prompt_str())  # doctest: +NORMALIZE_WHITESPACE
            Tool: detect
            Description: Run object detection
            Parameters:
            - threshold (float, optional, default=0.3): Min confidence
        """
        lines = [
            f"Tool: {self.name}",
            f"Description: {self.description}",
        ]

        if self.parameters:
            lines.append("Parameters:")
            for param in self.parameters:
                req_str = "required" if param.required else "optional"
                default_str = f", default={param.default}" if not param.required and param.default is not None else ""
                lines.append(f"  - {param.name} ({param.type}, {req_str}{default_str}): {param.description}")
        else:
            lines.append("Parameters: None")

        return "\n".join(lines)

    def to_openai_schema(self) -> dict[str, Any]:
        """Format as OpenAI function-calling schema.

        Produces a JSON schema compatible with OpenAI's function-calling API.
        This enables future support for API-based VLMs (GPT-4V, Claude, etc.).

        Returns:
            Dictionary following OpenAI function schema format.

        Examples:
            >>> schema = ToolSchema("detect", "Run object detection", "detect",
            ...     [ToolParameter("threshold", "float", "Min confidence", False, 0.3)])
            >>> api_schema = schema.to_openai_schema()
            >>> api_schema["name"]
            'detect'
            >>> api_schema["parameters"]["type"]
            'object'
        """
        # Map MATA type strings to JSON Schema types
        type_map = {
            "float": "number",
            "int": "integer",
            "str": "string",
            "bool": "boolean",
            "list[str]": "array",
            "bbox": "array",  # [x1, y1, x2, y2]
        }

        properties = {}
        required = []

        for param in self.parameters:
            json_type = type_map.get(param.type, "string")
            param_schema: dict[str, Any] = {
                "type": json_type,
                "description": param.description,
            }

            # Add array item types
            if param.type == "list[str]":
                param_schema["items"] = {"type": "string"}
            elif param.type == "bbox":
                param_schema["items"] = {"type": "number"}
                param_schema["minItems"] = 4
                param_schema["maxItems"] = 4

            # Add default value
            if not param.required and param.default is not None:
                param_schema["default"] = param.default

            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "task": self.task,
            "parameters": [p.to_dict() for p in self.parameters],
            "builtin": self.builtin,
        }


@dataclass(frozen=True)
class ToolCall:
    """Parsed tool invocation from VLM output.

    Represents a single tool call extracted from VLM's text response.
    The agent loop parses these from VLM output and dispatches them to the registry.

    Attributes:
        tool_name: Name of the tool to invoke
        arguments: Dictionary of parameter name → value
        raw_text: Original VLM text that produced this call (for debugging)

    Examples:
        >>> call = ToolCall("detect", {"threshold": 0.5}, '{"tool": "detect", "arguments": {"threshold": 0.5}}')
        >>> call.tool_name
        'detect'
        >>> call.arguments["threshold"]
        0.5
    """

    tool_name: str
    arguments: dict[str, Any]
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "raw_text": self.raw_text,
        }


@dataclass(frozen=True)
class ToolResult:
    """Result of a tool invocation, formatted for VLM consumption.

    After executing a tool, this result is formatted as text and fed back to the VLM
    in the next conversation turn. The VLM uses this information to decide whether
    to call another tool or provide a final answer.

    Attributes:
        tool_name: Name of the tool that was executed
        success: Whether the tool execution succeeded
        summary: Human-readable summary for VLM (e.g., "Found 3 objects: person, car, dog")
        artifacts: Typed results (Detections, Classifications, depth maps, etc.)

    Examples:
        >>> result = ToolResult("detect", True, "Found 2 objects", {"instances": []})
        >>> result.success
        True
        >>> result.summary
        'Found 2 objects'
    """

    tool_name: str
    success: bool
    summary: str
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "summary": self.summary,
            "artifacts": self.artifacts,
        }

    def to_conversation_message(self) -> str:
        """Format as a conversation message for VLM history.

        Returns text suitable for appending to conversation history.
        Includes success/failure status and summary.

        Returns:
            Formatted string for conversation history.

        Examples:
            >>> result = ToolResult("detect", True, "Found 2 cats and 1 dog")
            >>> result.to_conversation_message()
            '[Tool: detect] ✓ Found 2 cats and 1 dog'
        """
        status = "✓" if self.success else "✗"
        return f"[Tool: {self.tool_name}] {status} {self.summary}"


# Default tool schemas for MATA task types
# These are auto-generated schemas that can be used without manual configuration
TASK_SCHEMA_DEFAULTS: dict[str, ToolSchema] = {
    "detect": ToolSchema(
        name="detect",
        task="detect",
        description="Run object detection on the image or a cropped region. Returns bounding boxes with labels and confidence scores.",
        parameters=[
            ToolParameter(
                "region",
                "bbox",
                "Crop region as [x1, y1, x2, y2] in pixels, or null for full image",
                required=False,
                default=None,
            ),
            ToolParameter(
                "threshold",
                "float",
                "Minimum confidence threshold (0.0-1.0)",
                required=False,
                default=0.3,
            ),
            ToolParameter(
                "text_prompts",
                "str",
                "Object classes to detect, dot-separated (for zero-shot models)",
                required=False,
                default=None,
            ),
        ],
        builtin=False,
    ),
    "classify": ToolSchema(
        name="classify",
        task="classify",
        description="Classify the image or a specific region. Returns top-k class labels with confidence scores.",
        parameters=[
            ToolParameter(
                "region",
                "bbox",
                "Crop region as [x1, y1, x2, y2] in pixels, or null for full image",
                required=False,
                default=None,
            ),
            ToolParameter(
                "text_prompts",
                "list[str]",
                "Zero-shot class labels to evaluate (for CLIP-like models)",
                required=False,
                default=None,
            ),
            ToolParameter(
                "top_k",
                "int",
                "Number of top classifications to return",
                required=False,
                default=5,
            ),
        ],
        builtin=False,
    ),
    "segment": ToolSchema(
        name="segment",
        task="segment",
        description="Run instance or semantic segmentation on the image. Returns masks for detected objects or regions.",
        parameters=[
            ToolParameter(
                "region",
                "bbox",
                "Crop region as [x1, y1, x2, y2] in pixels, or null for full image",
                required=False,
                default=None,
            ),
            ToolParameter(
                "text_prompts",
                "str",
                "Object or region to segment (for zero-shot models like SAM)",
                required=False,
                default=None,
            ),
            ToolParameter(
                "threshold",
                "float",
                "Minimum confidence threshold (0.0-1.0)",
                required=False,
                default=0.5,
            ),
        ],
        builtin=False,
    ),
    "depth": ToolSchema(
        name="depth",
        task="depth",
        description="Estimate depth map for the image. Returns a depth map where closer objects are brighter.",
        parameters=[
            ToolParameter(
                "region",
                "bbox",
                "Crop region as [x1, y1, x2, y2] in pixels, or null for full image",
                required=False,
                default=None,
            ),
        ],
        builtin=False,
    ),
    "ocr": ToolSchema(
        name="ocr",
        task="ocr",
        description=(
            "Run OCR (optical character recognition) on the image or a cropped region. "
            "Returns all recognized text with confidence scores. "
            "Use this when you need to read text visible in the image."
        ),
        parameters=[
            ToolParameter(
                name="region",
                type="bbox",
                description="Optional crop region [x1, y1, x2, y2] to run OCR on a specific area",
                required=False,
                default=None,
            ),
        ],
        builtin=False,
    ),
}


def schema_for_task(task: str) -> ToolSchema:
    """Generate default ToolSchema for a MATA task type.

    Args:
        task: MATA task type ("detect", "classify", "segment", "depth")

    Returns:
        Default ToolSchema for the task.

    Raises:
        ValueError: If task type is not recognized.

    Examples:
        >>> schema = schema_for_task("detect")
        >>> schema.name
        'detect'
        >>> schema.task
        'detect'
        >>> len(schema.parameters)
        3
    """
    if task not in TASK_SCHEMA_DEFAULTS:
        raise ValueError(f"Unknown task type: {task}. " f"Available: {', '.join(sorted(TASK_SCHEMA_DEFAULTS.keys()))}")
    return TASK_SCHEMA_DEFAULTS[task]
