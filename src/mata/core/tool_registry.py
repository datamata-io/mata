"""Tool registry for VLM agent tool-calling system.

Resolves tool names to schemas and providers, validates tool availability,
and dispatches tool calls to the appropriate execution path (adapter-based
or built-in image operations).

Design:
- Tool names reference provider keys from the ExecutionContext
- Built-in tools (zoom, crop) are always available
- Provider-based tools are resolved via ctx.get_provider(capability, name)
- All tools are validated at construction time (fail fast)

Version: 1.7.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mata.core.tool_schema import (
    TASK_SCHEMA_DEFAULTS,
    ToolCall,
    ToolParameter,
    ToolResult,
    ToolSchema,
)

if TYPE_CHECKING:
    from mata.core.artifacts.image import Image
    from mata.core.graph.context import ExecutionContext

logger = logging.getLogger(__name__)


# ============================================================================
# Built-in Tool Schemas
# ============================================================================

BUILTIN_SCHEMAS: dict[str, ToolSchema] = {
    "zoom": ToolSchema(
        name="zoom",
        description="Zoom into a specific region of the image by cropping and upscaling. Use this to examine details in a specific area.",
        task="image",
        parameters=[
            ToolParameter(
                "region",
                "bbox",
                "Region to zoom into as [x1, y1, x2, y2] in pixels",
                required=True,
            ),
            ToolParameter(
                "scale",
                "float",
                "Upscaling factor (e.g., 2.0 = 2x zoom)",
                required=False,
                default=2.0,
            ),
        ],
        builtin=True,
    ),
    "crop": ToolSchema(
        name="crop",
        description="Crop a specific region from the image without upscaling. Use this to isolate a region for further analysis.",
        task="image",
        parameters=[
            ToolParameter(
                "region",
                "bbox",
                "Region to crop as [x1, y1, x2, y2] in pixels",
                required=True,
            ),
        ],
        builtin=True,
    ),
}


# ============================================================================
# ToolRegistry
# ============================================================================


class ToolRegistry:
    """Resolves tool names to schemas and providers.

    The registry bridges the gap between tool names in the user's `tools=` list
    and the actual provider instances in the ExecutionContext. It handles:

    1. Built-in tools (zoom, crop) - always available
    2. Provider-based tools (detect, classify, etc.) - resolved from context
    3. Schema generation for all available tools
    4. Tool call dispatch with argument validation

    Example:
        ```python
        # In an agent loop
        ctx = ExecutionContext(providers={
            "detect": {"detr": detr_adapter},
            "classify": {"clip": clip_adapter}
        })

        registry = ToolRegistry(ctx, ["detr", "clip", "zoom"])

        # Get schemas for prompt
        system_prompt = registry.build_system_prompt_block()

        # Execute a tool call
        call = ToolCall("detr", {"threshold": 0.5}, "...")
        result = registry.execute_tool(call, image)
        ```

    Attributes:
        _context: ExecutionContext with provider registry
        _tool_map: Maps tool name → (capability, provider) for provider-based tools
        _schemas: Maps tool name → ToolSchema for all available tools
    """

    def __init__(self, ctx: ExecutionContext, tool_names: list[str]):
        """Initialize registry and validate all tool names exist.

        Args:
            ctx: ExecutionContext with provider registry
            tool_names: List of tool names to make available. Can be:
                - Built-in tool names ("zoom", "crop")
                - Provider names that exist in ctx.providers

        Raises:
            KeyError: If any tool name is not found as a built-in or provider

        Example:
            ```python
            ctx = ExecutionContext(providers={
                "detect": {"detr": detr_adapter}
            })
            registry = ToolRegistry(ctx, ["detr", "zoom"])  # OK
            registry = ToolRegistry(ctx, ["unknown"])       # KeyError
            ```
        """
        self._context = ctx
        self._tool_map: dict[str, tuple[str, Any]] = {}  # tool_name → (capability, provider)
        self._schemas: dict[str, ToolSchema] = {}

        # Validate and resolve all tool names
        for tool_name in tool_names:
            if tool_name in BUILTIN_SCHEMAS:
                # Built-in tool - always available
                self._schemas[tool_name] = BUILTIN_SCHEMAS[tool_name]
                logger.debug(f"Registered built-in tool: {tool_name}")
            else:
                # Provider-based tool - search all capabilities
                capability, provider = self._resolve_provider(tool_name)
                self._tool_map[tool_name] = (capability, provider)
                self._schemas[tool_name] = self._schema_for_capability(capability, tool_name, provider)
                logger.debug(f"Registered provider tool: {tool_name} (capability: {capability})")

    def _resolve_provider(self, tool_name: str) -> tuple[str, Any]:
        """Resolve a tool name to its capability and provider instance.

        Searches all capabilities in the context to find which one contains
        the given provider name.

        Args:
            tool_name: Provider name to resolve

        Returns:
            Tuple of (capability, provider_instance)

        Raises:
            KeyError: If tool name not found in any capability
        """
        if not self._context.providers:
            raise KeyError(
                f"Tool '{tool_name}' not found. No providers registered in context. "
                f"Available built-in tools: {', '.join(BUILTIN_SCHEMAS.keys())}"
            )

        # Search all capabilities for this provider name
        for capability, cap_providers in self._context.providers.items():
            if tool_name in cap_providers:
                provider = cap_providers[tool_name]
                return capability, provider

        # Not found - build helpful error message
        available_providers = []
        for capability, cap_providers in self._context.providers.items():
            for name in cap_providers.keys():
                available_providers.append(f"{name} ({capability})")

        available_str = ", ".join(available_providers) if available_providers else "none"
        builtin_str = ", ".join(BUILTIN_SCHEMAS.keys())

        raise KeyError(
            f"Tool '{tool_name}' not found in provider registry. "
            f"Available providers: {available_str}. "
            f"Available built-in tools: {builtin_str}"
        )

    def _is_zero_shot_provider(self, provider: Any) -> bool:
        """Return True if the provider wraps a zero-shot adapter.

        Unwraps one level of wrapper (e.g. DetectorWrapper) then checks the
        class name for the 'ZeroShot' marker used by all MATA zero-shot adapters.

        Args:
            provider: Provider instance (may be a wrapper or raw adapter)

        Returns:
            True if the underlying adapter is a zero-shot model
        """
        # Unwrap through a single wrapper layer (DetectorWrapper, ClassifierWrapper, etc.)
        adapter = getattr(provider, "adapter", provider)
        return "ZeroShot" in type(adapter).__name__

    def _schema_for_capability(self, capability: str, tool_name: str, provider: Any) -> ToolSchema:
        """Generate ToolSchema for a capability, tailored to the actual provider.

        Uses the default task schemas from TASK_SCHEMA_DEFAULTS as a base, but
        customizes the name and — for zero-shot providers — upgrades
        ``text_prompts`` to ``required=True`` so the VLM's system prompt
        correctly instructs the model to always supply the parameter.

        Args:
            capability: Task capability ("detect", "classify", "segment", "depth")
            tool_name: Provider name to use in schema (e.g. "detector", "detr")
            provider: The resolved provider instance (wrapper or raw adapter)

        Returns:
            ToolSchema for this capability, with text_prompts required when the
            provider is a zero-shot model.
        """
        if capability not in TASK_SCHEMA_DEFAULTS:
            # For unknown capabilities (e.g., "vlm"), create a minimal schema
            return ToolSchema(
                name=tool_name,
                description=f"Execute {capability} task",
                task=capability,
                parameters=[],
                builtin=False,
            )

        # Clone the default schema but use the provider name
        default = TASK_SCHEMA_DEFAULTS[capability]

        # For zero-shot providers, upgrade text_prompts to required so the VLM
        # knows it must always supply the classes it wants to detect/classify.
        params = default.parameters
        if self._is_zero_shot_provider(provider):
            params = [
                ToolParameter(
                    p.name,
                    p.type,
                    (
                        (
                            "Object classes to detect, dot-separated (e.g. 'cat . dog . person'). "
                            "REQUIRED — this is a zero-shot model and cannot run without class names."
                        )
                        if p.name == "text_prompts"
                        else p.description
                    ),
                    required=True if p.name == "text_prompts" else p.required,
                    default=None if p.name == "text_prompts" else p.default,
                )
                for p in default.parameters
            ]

        return ToolSchema(
            name=tool_name,  # Use provider name, not capability
            description=default.description,
            task=default.task,
            parameters=params,
            builtin=False,
        )

    def get_schema(self, tool_name: str) -> ToolSchema:
        """Get ToolSchema for a tool name.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolSchema for the tool

        Raises:
            KeyError: If tool name not in registry
        """
        if tool_name not in self._schemas:
            raise KeyError(
                f"Tool '{tool_name}' not in registry. " f"Available tools: {', '.join(self._schemas.keys())}"
            )
        return self._schemas[tool_name]

    def get_provider(self, tool_name: str) -> Any:
        """Get provider instance for a tool name.

        Only valid for provider-based tools (not built-ins).

        Args:
            tool_name: Name of the tool

        Returns:
            Provider instance (wrapper)

        Raises:
            KeyError: If tool is built-in or not found
        """
        if tool_name in BUILTIN_SCHEMAS:
            raise KeyError(f"Tool '{tool_name}' is a built-in tool and has no provider")

        if tool_name not in self._tool_map:
            raise KeyError(
                f"Tool '{tool_name}' not in registry. " f"Available provider tools: {', '.join(self._tool_map.keys())}"
            )

        _, provider = self._tool_map[tool_name]
        return provider

    def all_schemas(self) -> list[ToolSchema]:
        """Get all available tool schemas.

        Returns:
            List of ToolSchema objects for all registered tools
        """
        return list(self._schemas.values())

    def build_system_prompt_block(self) -> str:
        """Generate the tool-description block for VLM system prompt.

        Renders all registered tools as human-readable descriptions that
        can be injected into the VLM's system prompt.

        Returns:
            Formatted string with all tool descriptions

        Example:
            >>> registry = ToolRegistry(ctx, ["detr", "zoom"])
            >>> print(registry.build_system_prompt_block())
            Tool: detr
            Description: Run object detection...
            Parameters:
              - threshold (float, optional, default=0.3): Minimum confidence
            <BLANKLINE>
            Tool: zoom
            Description: Zoom into a region...
            Parameters:
              - region (bbox, required): Region to zoom...
        """
        blocks = []
        for schema in self._schemas.values():
            blocks.append(schema.to_prompt_str())
        return "\n\n".join(blocks)

    def execute_tool(self, tool_call: ToolCall, image: Image) -> ToolResult:
        """Dispatch a parsed ToolCall to the appropriate provider.

        Handles both adapter-based tools (via wrapper.predict()) and built-in
        tools (zoom, crop). Validates arguments and formats results for VLM
        consumption.

        Args:
            tool_call: Parsed tool invocation from VLM output
            image: Image to operate on

        Returns:
            ToolResult with success status, summary text, and artifacts

        Raises:
            KeyError: If tool name not in registry
            Exception: If tool execution fails (propagated from underlying tool)

        Example:
            >>> call = ToolCall("detr", {"threshold": 0.5}, "...")
            >>> result = registry.execute_tool(call, image)
            >>> print(result.summary)
            'Found 3 objects: person (0.95), car (0.87), dog (0.72)'
        """
        tool_name = tool_call.tool_name

        if tool_name not in self._schemas:
            raise KeyError(
                f"Tool '{tool_name}' not in registry. " f"Available tools: {', '.join(self._schemas.keys())}"
            )

        # Dispatch to built-in or provider-based execution
        if tool_name in BUILTIN_SCHEMAS:
            return self._execute_builtin(tool_call, image)
        else:
            return self._execute_provider(tool_call, image)

    def _execute_builtin(self, tool_call: ToolCall, image: Image) -> ToolResult:
        """Execute a built-in image tool (zoom, crop).

        Args:
            tool_call: Tool call with arguments
            image: Image to operate on

        Returns:
            ToolResult with cropped/zoomed image and summary
        """
        tool_name = tool_call.tool_name
        args = tool_call.arguments

        try:
            if tool_name == "crop":
                return self._crop_tool(image, args.get("region"))
            elif tool_name == "zoom":
                return self._zoom_tool(image, args.get("region"), args.get("scale", 2.0))
            else:
                raise ValueError(f"Unknown built-in tool: {tool_name}")

        except Exception as e:
            logger.error(f"Built-in tool '{tool_name}' failed: {e}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                summary=f"Tool '{tool_name}' failed: {str(e)}",
                artifacts={},
            )

    def _execute_provider(self, tool_call: ToolCall, image: Image) -> ToolResult:
        """Execute a provider-based tool (detect, classify, segment, depth).

        Args:
            tool_call: Tool call with arguments
            image: Image to operate on

        Returns:
            ToolResult with task-specific results and summary
        """
        tool_name = tool_call.tool_name
        capability, provider = self._tool_map[tool_name]
        args = tool_call.arguments.copy()

        try:
            # Handle region-based cropping if specified
            target_image = image
            region_offset = None

            if "region" in args and args["region"] is not None:
                region = args.pop("region")  # Remove from args
                crop_result = self._crop_tool(image, region)
                if crop_result.success:
                    target_image = crop_result.artifacts["image"]
                    # Store clamped integer coordinates for remapping
                    x1, y1, x2, y2 = region
                    x1 = int(max(0, min(x1, image.width)))
                    y1 = int(max(0, min(y1, image.height)))
                    region_offset = [x1, y1, x2, y2]  # For coordinate remapping
                else:
                    # Crop failed - return error
                    return crop_result

            # Execute provider
            result = provider.predict(target_image, **args)

            # Format result based on capability
            return self._format_provider_result(tool_name, capability, result, region_offset)

        except Exception as e:
            logger.error(f"Provider tool '{tool_name}' ({capability}) failed: {e}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                summary=f"Tool '{tool_name}' failed: {str(e)}",
                artifacts={},
            )

    def _format_provider_result(
        self,
        tool_name: str,
        capability: str,
        result: Any,
        region_offset: list[float] | None = None,
    ) -> ToolResult:
        """Format a provider result as a ToolResult.

        Args:
            tool_name: Name of the tool that was executed
            capability: Task capability (detect, classify, etc.)
            result: Raw result from provider.predict()
            region_offset: If result is from a cropped region, the [x1, y1, x2, y2] offset

        Returns:
            ToolResult with formatted summary and artifacts
        """
        # Import here to avoid circular dependency
        from mata.core.artifacts.classifications import Classifications
        from mata.core.artifacts.detections import Detections
        from mata.core.types import Instance

        try:
            if capability == "detect":
                # Result should be Detections artifact
                if not isinstance(result, Detections):
                    return ToolResult(
                        tool_name=tool_name,
                        success=False,
                        summary=f"Unexpected result type from {tool_name}: {type(result)}",
                        artifacts={},
                    )

                # Remap coordinates if from cropped region
                if region_offset:
                    # region_offset contains clamped integer coordinates [x1, y1, x2, y2]
                    x1_offset, y1_offset = region_offset[0], region_offset[1]

                    # Remap bbox coordinates from crop space to original image space
                    remapped_instances = []
                    for inst in result.instances:
                        if inst.bbox is not None:
                            # Offset bbox coordinates
                            x1, y1, x2, y2 = inst.bbox
                            remapped_bbox = (
                                x1 + x1_offset,
                                y1 + y1_offset,
                                x2 + x1_offset,
                                y2 + y1_offset,
                            )

                            # Create new Instance with remapped bbox
                            remapped_inst = Instance(
                                bbox=remapped_bbox,
                                mask=inst.mask,
                                score=inst.score,
                                label=inst.label,
                                label_name=inst.label_name,
                                area=inst.area,
                                is_stuff=inst.is_stuff,
                                embedding=inst.embedding,
                                track_id=inst.track_id,
                                keypoints=inst.keypoints,
                            )
                            remapped_instances.append(remapped_inst)
                        else:
                            # Mask-only instance - keep as-is
                            remapped_instances.append(inst)

                    # Create new Detections with remapped instances
                    result = Detections(
                        instances=remapped_instances,
                        entities=result.entities,
                        meta={**result.meta, "region_offset": region_offset},
                    )

                # Build summary
                num_dets = len(result.instances)
                if num_dets == 0:
                    summary = "No objects detected"
                else:
                    # Top 3 detections
                    top_labels = [inst.label for inst in result.instances[:3]]
                    top_scores = [f"{inst.score:.2f}" for inst in result.instances[:3]]
                    top_str = ", ".join([f"{lbl} ({s})" for lbl, s in zip(top_labels, top_scores)])
                    summary = f"Found {num_dets} objects: {top_str}"
                    if num_dets > 3:
                        summary += f" (and {num_dets - 3} more)"

                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    summary=summary,
                    artifacts={"detections": result},
                )

            elif capability == "classify":
                # Result should be Classifications artifact
                if not isinstance(result, Classifications):
                    return ToolResult(
                        tool_name=tool_name,
                        success=False,
                        summary=f"Unexpected result type from {tool_name}: {type(result)}",
                        artifacts={},
                    )

                # Build summary
                if not result.predictions:
                    summary = "No classifications returned"
                else:
                    top1 = result.top1
                    label1 = top1.label_name if top1.label_name else str(top1.label)
                    summary = f"Top classification: {label1} ({top1.score:.2f})"
                    if len(result.predictions) > 1:
                        top2 = result.predictions[1]
                        label2 = top2.label_name if top2.label_name else str(top2.label)
                        summary += f", second: {label2} ({top2.score:.2f})"

                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    summary=summary,
                    artifacts={"classifications": result},
                )

            elif capability == "ocr":
                from mata.core.types import OCRResult

                if isinstance(result, OCRResult):
                    texts = [r.text for r in result.regions]
                elif hasattr(result, "text_blocks"):  # OCRText artifact
                    texts = [b.text for b in result.text_blocks]
                else:
                    texts = []

                if texts:
                    preview = " | ".join(texts[:3])
                    if len(preview) > 80:
                        preview = preview[:77] + "..."
                    summary = f"Recognized {len(texts)} text region(s): {preview}"
                    if len(texts) > 3:
                        summary += f" (+{len(texts) - 3} more)"
                else:
                    summary = "No text found in image"

                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    summary=summary,
                    artifacts={"ocr_result": result},
                )

            else:
                # Generic handling for segment, depth, etc.
                summary = f"{capability.capitalize()} task completed successfully"
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    summary=summary,
                    artifacts={capability: result},
                )

        except Exception as e:
            logger.error(f"Failed to format result from {tool_name}: {e}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                summary=f"Failed to format result: {str(e)}",
                artifacts={},
            )

    def _crop_tool(self, image: Image, region: list[float] | None) -> ToolResult:
        """Crop a region from the image.

        Args:
            image: Source image
            region: [x1, y1, x2, y2] in pixels, or None

        Returns:
            ToolResult with cropped Image artifact
        """
        if region is None:
            return ToolResult(
                tool_name="crop",
                success=False,
                summary="Crop requires a 'region' parameter",
                artifacts={},
            )

        try:
            # Validate and clamp region
            x1, y1, x2, y2 = region
            x1 = max(0, min(x1, image.width))
            y1 = max(0, min(y1, image.height))
            x2 = max(0, min(x2, image.width))
            y2 = max(0, min(y2, image.height))

            if x2 <= x1 or y2 <= y1:
                return ToolResult(
                    tool_name="crop",
                    success=False,
                    summary=f"Invalid region: [{x1}, {y1}, {x2}, {y2}] (empty or inverted)",
                    artifacts={},
                )

            # Crop using PIL
            pil_img = image.to_pil()
            cropped_pil = pil_img.crop((int(x1), int(y1), int(x2), int(y2)))

            # Import here to avoid circular dependency
            from mata.core.artifacts.image import Image as ImageArtifact

            cropped_image = ImageArtifact.from_pil(
                cropped_pil,
                color_space=image.color_space,
            )

            summary = f"Cropped region [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}] -> {cropped_image.width}x{cropped_image.height}px"

            return ToolResult(
                tool_name="crop",
                success=True,
                summary=summary,
                artifacts={"image": cropped_image},
            )

        except Exception as e:
            logger.error(f"Crop tool failed: {e}")
            return ToolResult(
                tool_name="crop",
                success=False,
                summary=f"Crop failed: {str(e)}",
                artifacts={},
            )

    def _zoom_tool(self, image: Image, region: list[float] | None, scale: float = 2.0) -> ToolResult:
        """Zoom into a region by cropping and upscaling.

        Args:
            image: Source image
            region: [x1, y1, x2, y2] in pixels, or None
            scale: Upscaling factor (default 2.0)

        Returns:
            ToolResult with zoomed Image artifact
        """
        # First crop
        crop_result = self._crop_tool(image, region)
        if not crop_result.success:
            crop_result.tool_name = "zoom"  # Update tool name
            return crop_result

        try:
            cropped_image = crop_result.artifacts["image"]

            # Upscale
            pil_img = cropped_image.to_pil()
            new_width = int(cropped_image.width * scale)
            new_height = int(cropped_image.height * scale)

            from PIL import Image as PILImage

            zoomed_pil = pil_img.resize((new_width, new_height), PILImage.LANCZOS)

            # Import here to avoid circular dependency
            from mata.core.artifacts.image import Image as ImageArtifact

            zoomed_image = ImageArtifact.from_pil(
                zoomed_pil,
                color_space=cropped_image.color_space,
            )

            if region:
                x1, y1, x2, y2 = region
                summary = f"Zoomed region [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}] by {scale}x -> {zoomed_image.width}x{zoomed_image.height}px"
            else:
                summary = f"Zoomed by {scale}x -> {zoomed_image.width}x{zoomed_image.height}px"

            return ToolResult(
                tool_name="zoom",
                success=True,
                summary=summary,
                artifacts={"image": zoomed_image},
            )

        except Exception as e:
            logger.error(f"Zoom tool failed: {e}")
            return ToolResult(
                tool_name="zoom",
                success=False,
                summary=f"Zoom failed: {str(e)}",
                artifacts={},
            )
