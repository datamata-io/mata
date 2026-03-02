"""JSON extraction and entity parsing for VLM structured output.

Handles the messiness of LLM text output: markdown code fences,
partial JSON, multiple JSON blocks, and malformed responses.
Designed for graceful degradation — never raises on bad input.

Version 1.7.0: Added tool-call parsing for VLM agent system.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from mata.core.logging import get_logger
from mata.core.types import Entity, Instance

if TYPE_CHECKING:
    from mata.core.tool_schema import ToolCall, ToolSchema

logger = get_logger(__name__)

# Default key mappings for flexible entity parsing
DEFAULT_LABEL_KEYS = ["label", "name", "object", "class", "category", "type", "item"]
DEFAULT_SCORE_KEYS = ["score", "confidence", "probability", "prob", "certainty"]
DEFAULT_LIST_KEYS = ["objects", "items", "detections", "entities", "results"]
DEFAULT_BBOX_KEYS = ["bbox", "box", "bounding_box", "bounds"]
DEFAULT_MASK_KEYS = ["mask", "segmentation"]


def extract_json_from_text(text: str) -> dict | list | None:
    """Extract JSON from LLM text output.

    Handles multiple formats:
    - Markdown fenced JSON (```json ... ```)
    - Raw JSON objects or arrays
    - JSON embedded in surrounding text

    Extraction priority (try in order, return first success):
    1. JSON inside markdown code fences
    2. JSON array at top level
    3. JSON object at top level
    4. Raw text as JSON

    Args:
        text: Raw text output from VLM

    Returns:
        Parsed JSON as dict or list, or None if no valid JSON found

    Examples:
        >>> extract_json_from_text('```json\\n{"label": "cat"}\\n```')
        {'label': 'cat'}
        >>> extract_json_from_text('[{"label": "dog"}]')
        [{'label': 'dog'}]
        >>> extract_json_from_text('No JSON here')
        None
    """
    if not text or not text.strip():
        logger.debug("Empty text provided, no JSON to extract")
        return None

    # Strategy 1: Try markdown code fences (```json or ```)
    # Match both ```json and ``` without language specifier
    fence_pattern = r"```(?:json)?\s*\n(.*?)\n```"
    fence_matches = re.findall(fence_pattern, text, re.DOTALL | re.IGNORECASE)

    for match in fence_matches:
        try:
            # Strip trailing commas (common LLM mistake)
            cleaned = _strip_trailing_commas(match.strip())
            parsed = json.loads(cleaned)
            logger.debug("Successfully extracted JSON from code fence")
            return parsed
        except json.JSONDecodeError:
            continue

    # Strategy 2: Try to find JSON array at top level
    array_pattern = r"\[.*\]"
    array_matches = re.findall(array_pattern, text, re.DOTALL)

    for match in array_matches:
        try:
            cleaned = _strip_trailing_commas(match.strip())
            parsed = json.loads(cleaned)
            logger.debug("Successfully extracted JSON array from text")
            return parsed
        except json.JSONDecodeError:
            continue

    # Strategy 3: Try to find JSON object at top level
    # Use a more sophisticated approach to find balanced braces
    obj_match = _extract_balanced_json_object(text)
    if obj_match:
        try:
            cleaned = _strip_trailing_commas(obj_match.strip())
            parsed = json.loads(cleaned)
            logger.debug("Successfully extracted JSON object from text")
            return parsed
        except json.JSONDecodeError:
            pass

    # Strategy 4: Try raw text as JSON
    try:
        cleaned = _strip_trailing_commas(text.strip())
        parsed = json.loads(cleaned)
        logger.debug("Successfully parsed raw text as JSON")
        return parsed
    except json.JSONDecodeError:
        pass

    logger.debug("No valid JSON found in text")
    return None


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before closing brackets/braces (common LLM error)."""
    # Remove trailing comma before ] or }
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def _extract_balanced_json_object(text: str) -> str | None:
    """Extract first balanced JSON object from text.

    Finds opening { and matching closing } accounting for nesting.
    """
    start_idx = text.find("{")
    if start_idx == -1:
        return None

    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(start_idx, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx : i + 1]

    return None


def parse_entities(
    data: dict | list,
    key_mapping: dict[str, str] | None = None,
    auto_promote: bool = False,
) -> list[Entity | Instance]:
    """Parse JSON data into Entity or Instance objects.

    Handles various JSON shapes from VLM output:
    - List of objects: [{"label": "cat", "confidence": 0.9}, ...]
    - Dict with items key: {"objects": [...], "count": 3}
    - Single object: {"label": "cat", "confidence": 0.9}

    Key mapping supports flexible field names from VLM output:
    - "name" → label, "object" → label, "class" → label
    - "confidence" → score, "probability" → score, "prob" → score
    - "bbox" → bbox, "box" → bbox, "bounding_box" → bbox
    - "mask" → mask, "segmentation" → mask
    - All other keys → attributes

    Auto-promotion feature (v1.5.4):
    When auto_promote=True, entities with bbox/mask data in JSON are
    automatically promoted to Instance objects. This enables direct
    comparison with other spatial detection models when using VLMs
    that output bounding boxes (e.g., Qwen3-VL grounding mode).

    Args:
        data: Parsed JSON (dict or list)
        key_mapping: Optional custom key mapping overrides
        auto_promote: If True, promote Entity to Instance when bbox/mask found

    Returns:
        List of Entity or Instance objects (may be empty if parsing fails)

    Examples:
        >>> # Default: Entity objects (semantic only)
        >>> parse_entities([{"label": "cat", "confidence": 0.9}])
        [Entity(label='cat', score=0.9, attributes={})]

        >>> # With bbox but no auto_promote: bbox in attributes
        >>> parse_entities([{"label": "cat", "bbox": [10, 20, 100, 150]}])
        [Entity(label='cat', score=1.0, attributes={'bbox': [10, 20, 100, 150]})]

        >>> # With auto_promote: Instance objects (spatial)
        >>> parse_entities([{"label": "cat", "bbox": [10, 20, 100, 150]}], auto_promote=True)
        [Instance(bbox=(10, 20, 100, 150), score=1.0, label=0, label_name='cat', ...)]
    """
    entities = []

    # Handle list of items
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                entity = _parse_single_entity(item, key_mapping, auto_promote)
                if entity:
                    entities.append(entity)
        return entities

    # Handle dict
    if isinstance(data, dict):
        # Check if dict contains a list under a known key
        for list_key in DEFAULT_LIST_KEYS:
            # Case-insensitive check
            matching_keys = [k for k in data.keys() if k.lower() == list_key.lower()]
            if matching_keys:
                list_data = data[matching_keys[0]]
                if isinstance(list_data, list):
                    return parse_entities(list_data, key_mapping, auto_promote)

        # Otherwise, treat as single entity
        entity = _parse_single_entity(data, key_mapping, auto_promote)
        if entity:
            entities.append(entity)
        return entities

    logger.debug(f"Unexpected data type for entity parsing: {type(data)}")
    return entities


def _parse_single_entity(
    item: dict,
    key_mapping: dict[str, str] | None = None,
    auto_promote: bool = False,
) -> Entity | Instance | None:
    """Parse a single dict into an Entity or Instance.

    Args:
        item: Dictionary representing one entity
        key_mapping: Optional custom key mapping
        auto_promote: If True, promote to Instance when bbox/mask found

    Returns:
        Entity if label found (or Instance if auto_promote=True and spatial data present),
        None otherwise
    """
    # Find label (required)
    label = None
    label_key_used = None

    for key in DEFAULT_LABEL_KEYS:
        # Case-insensitive check
        matching_keys = [k for k in item.keys() if k.lower() == key.lower()]
        if matching_keys:
            label = item[matching_keys[0]]
            label_key_used = matching_keys[0]
            break

    # Apply custom key mapping if provided
    if key_mapping and "label" in key_mapping:
        custom_label_key = key_mapping["label"]
        if custom_label_key in item:
            label = item[custom_label_key]
            label_key_used = custom_label_key

    if not label:
        logger.debug(f"No label found in item: {item}")
        return None

    # Find score (optional, defaults to 1.0)
    score = 1.0
    score_key_used = None

    for key in DEFAULT_SCORE_KEYS:
        matching_keys = [k for k in item.keys() if k.lower() == key.lower()]
        if matching_keys:
            try:
                score = float(item[matching_keys[0]])
                score_key_used = matching_keys[0]
                break
            except (ValueError, TypeError):
                logger.debug(f"Invalid score value: {item[matching_keys[0]]}")

    # Apply custom key mapping for score
    if key_mapping and "score" in key_mapping:
        custom_score_key = key_mapping["score"]
        if custom_score_key in item:
            try:
                score = float(item[custom_score_key])
                score_key_used = custom_score_key
            except (ValueError, TypeError):
                logger.debug(f"Invalid score value: {item[custom_score_key]}")

    # Collect all other keys as attributes (initially)
    attributes = {}
    used_keys = {label_key_used, score_key_used}

    for key, value in item.items():
        if key not in used_keys:
            attributes[key] = value

    # Auto-promotion: check for bbox/mask in attributes
    if auto_promote:
        bbox = None
        mask = None
        bbox_key_to_remove = None
        mask_key_to_remove = None

        # Try to find bbox in attributes
        for bbox_key in DEFAULT_BBOX_KEYS:
            matching_keys = [k for k in attributes.keys() if k.lower() == bbox_key.lower()]
            if matching_keys:
                bbox_data = attributes[matching_keys[0]]
                if isinstance(bbox_data, list) and len(bbox_data) >= 4:
                    # Convert to tuple (xyxy format expected)
                    bbox = tuple(bbox_data[:4])
                    bbox_key_to_remove = matching_keys[0]
                    logger.debug(f"Auto-promoted entity with bbox: {bbox}")
                break

        # Try to find mask in attributes
        for mask_key in DEFAULT_MASK_KEYS:
            matching_keys = [k for k in attributes.keys() if k.lower() == mask_key.lower()]
            if matching_keys:
                mask = attributes[matching_keys[0]]
                mask_key_to_remove = matching_keys[0]
                logger.debug(f"Auto-promoted entity with mask (type={type(mask).__name__})")
                break

        # If bbox or mask found, promote to Instance
        if bbox is not None or mask is not None:
            try:
                instance = Instance(
                    bbox=bbox,
                    mask=mask,
                    score=score,
                    label=0,  # Default label_id
                    label_name=str(label),
                )
                # Only remove bbox/mask from attributes after successful promotion
                if bbox_key_to_remove:
                    attributes.pop(bbox_key_to_remove)
                if mask_key_to_remove:
                    attributes.pop(mask_key_to_remove)
                return instance
            except (ValueError, TypeError) as e:
                # Instance validation failed (e.g., invalid bbox/mask format)
                # Fall back to Entity with bbox/mask in attributes
                logger.warning(
                    f"Failed to promote entity to Instance: {e}. " f"Returning Entity with spatial data in attributes."
                )
                # bbox/mask stay in attributes since we didn't pop them

    return Entity(label=str(label), score=score, attributes=attributes)


def get_json_schema(mode: str) -> str:
    """Get JSON format instruction for a given output mode.

    Used by VLM adapter to append formatting instructions to the prompt.

    Args:
        mode: Output mode ("json", "detect", "classify", "describe")

    Returns:
        Instruction string to append to system prompt

    Examples:
        >>> "JSON" in get_json_schema("json")
        True
        >>> "label" in get_json_schema("detect")
        True
        >>> get_json_schema("unknown")
        ''
    """
    mode_lower = mode.lower() if mode else ""

    schemas = {
        "json": "Respond with a valid JSON object.",
        "detect": (
            "Respond with a JSON array containing ALL detected objects in the image. "
            "Include every object you can identify, regardless of confidence level. "
            'Each object should have "label" (string) and "confidence" (float 0-1, your estimated confidence). '
            'Optionally include "bbox" as [x1, y1, x2, y2] coordinates (top-left to bottom-right). '
            "Example with varying confidence: "
            '[{"label": "cat", "confidence": 0.95, "bbox": [535, 52, 1000, 771]}, '
            '{"label": "remote", "confidence": 0.7, "bbox": [519, 158, 578, 388]}, '
            '{"label": "couch", "confidence": 0.85, "bbox": [0, 0, 1000, 1000]}]. '
            "Respond ONLY with JSON, no other text."
        ),
        "classify": (
            "Respond with a JSON array of possible classifications. "
            'Each should have "label" (string) and "confidence" (float 0-1). '
            "Respond ONLY with JSON, no other text."
        ),
        "describe": (
            'Respond with a JSON object containing "description" (string), '
            '"objects" (array of {"label": string, "confidence": float}), '
            'and "scene" (string). '
            "Respond ONLY with JSON, no other text."
        ),
    }

    if mode_lower in schemas:
        return schemas[mode_lower]

    if mode:
        logger.warning(f"Unknown output mode '{mode}', no schema instruction added")

    return ""


# ============================================================================
# Tool-Call Parsing (v1.7.0 — VLM Agent System)
# ============================================================================


def parse_tool_calls(text: str) -> list[ToolCall] | None:
    """Extract tool call(s) from VLM output text.

    Attempts multiple parsing strategies to robustly extract tool calls from
    diverse VLM output formats. Returns None if no tool call is found, which
    signals a final answer (agent loop termination).

    Parsing strategies (tried in order):
    1. Fenced code blocks: ```tool_call\n{...}\n```
    2. XML-style tags: <tool_call>{...}</tool_call>
    3. Raw JSON with "tool" key: {"tool": "...", "arguments": {...}}
    4. Raw JSON with "action" key: {"action": "...", "parameters": {...}}

    Args:
        text: Raw VLM output text (may contain mixed text + tool call)

    Returns:
        List of ToolCall objects (may contain 1+ calls), or None if no tool call found.
        None indicates final answer mode (VLM synthesis without tool invocation).

    Examples:
        >>> # Fenced block format
        >>> text = '```tool_call\\n{"tool": "detect", "arguments": {"threshold": 0.5}}\\n```'
        >>> calls = parse_tool_calls(text)
        >>> calls[0].tool_name
        'detect'

        >>> # XML format
        >>> text = '<tool_call>{"tool": "classify", "arguments": {}}</tool_call>'
        >>> calls = parse_tool_calls(text)
        >>> calls[0].tool_name
        'classify'

        >>> # Final answer (no tool call)
        >>> text = 'I found 3 cats and 2 dogs in the image.'
        >>> parse_tool_calls(text)
        None
    """
    if not text or not text.strip():
        logger.debug("Empty text provided, no tool call to extract")
        return None

    # Lazy import to avoid circular dependency
    from mata.core.tool_schema import ToolCall

    # Strategy 1: Fenced code blocks (```tool_call ... ```)
    fence_pattern = r"```tool_call\s*\n(.*?)\n```"
    fence_matches = re.findall(fence_pattern, text, re.DOTALL | re.IGNORECASE)

    for match in fence_matches:
        parsed = _parse_tool_call_json(match.strip())
        if parsed:
            tool_name, arguments, raw_text = parsed
            logger.debug(f"Extracted tool call from fenced block: {tool_name}")
            return [ToolCall(tool_name=tool_name, arguments=arguments, raw_text=match.strip())]

    # Strategy 2: XML-style tags (<tool_call>...</tool_call>)
    xml_pattern = r"<tool_call>(.*?)</tool_call>"
    xml_matches = re.findall(xml_pattern, text, re.DOTALL | re.IGNORECASE)

    for match in xml_matches:
        parsed = _parse_tool_call_json(match.strip())
        if parsed:
            tool_name, arguments, raw_text = parsed
            logger.debug(f"Extracted tool call from XML tag: {tool_name}")
            return [ToolCall(tool_name=tool_name, arguments=arguments, raw_text=match.strip())]

    # Strategy 3: Raw JSON anywhere in text
    # Try to extract any JSON object from the text
    json_obj = _extract_balanced_json_object(text)
    if json_obj:
        parsed = _parse_tool_call_json(json_obj)
        if parsed:
            tool_name, arguments, raw_text = parsed
            logger.debug(f"Extracted tool call from raw JSON: {tool_name}")
            return [ToolCall(tool_name=tool_name, arguments=arguments, raw_text=json_obj)]

    logger.debug("No tool call found in text (final answer mode)")
    return None


def _parse_tool_call_json(json_str: str) -> tuple[str, dict, str] | None:
    """Parse JSON string into (tool_name, arguments, raw_text) tuple.

    Handles multiple key naming conventions:
    - {"tool": "...", "arguments": {...}}
    - {"action": "...", "parameters": {...}}
    - {"tool": "...", "args": {...}}
    - {"name": "...", "arguments": {...}}

    Args:
        json_str: JSON string to parse

    Returns:
        (tool_name, arguments_dict, raw_json) tuple, or None if parsing failed
    """
    try:
        data = json.loads(json_str.strip())
    except json.JSONDecodeError:
        logger.debug(f"Failed to parse JSON: {json_str[:100]}")
        return None

    if not isinstance(data, dict):
        return None

    # Extract tool name (try multiple keys)
    tool_name = None
    for key in ["tool", "action", "name"]:
        if key in data and isinstance(data[key], str):
            tool_name = data[key]
            break

    if not tool_name:
        logger.debug(f"No 'tool', 'action', or 'name' key found in JSON: {data}")
        return None

    # Extract arguments (try multiple keys)
    arguments = {}
    for key in ["arguments", "parameters", "args", "params"]:
        if key in data:
            if isinstance(data[key], dict):
                arguments = data[key]
                break
            else:
                logger.warning(f"Found '{key}' key but value is not a dict: {type(data[key])}")

    # If no arguments key found, check if the rest of the dict is arguments
    # (for format like {"tool": "detect", "threshold": 0.5})
    if not arguments:
        # All keys except the tool name key are arguments
        used_keys = {k for k in ["tool", "action", "name"] if k in data}
        potential_args = {k: v for k, v in data.items() if k not in used_keys}
        if potential_args:
            arguments = potential_args
            logger.debug(f"Using flat argument format: {arguments}")

    return (tool_name, arguments, json_str)


def validate_tool_call(call: ToolCall, schemas: list[ToolSchema]) -> ToolCall:
    """Validate and normalize a parsed tool call against known schemas.

    Performs three key operations:
    1. Fuzzy-match tool names ("detection" → "detect", "classifier" → "classify")
    2. Type-coerce arguments ("0.5" → 0.5 for float, "[1,2,3]" → [1,2,3] for list)
    3. Fill defaults for missing optional parameters

    Args:
        call: Parsed tool call from VLM output
        schemas: List of available tool schemas for validation

    Returns:
        Validated and normalized ToolCall (new instance, original unchanged)

    Raises:
        ValueError: If tool name cannot be matched to any schema

    Examples:
        >>> from mata.core.tool_schema import ToolCall, ToolSchema, ToolParameter
        >>> schema = ToolSchema("detect", "Run detection", "detect",
        ...     [ToolParameter("threshold", "float", "Conf threshold", False, 0.3)])
        >>> call = ToolCall("detection", {"threshold": "0.5"})  # Wrong name, string arg
        >>> validated = validate_tool_call(call, [schema])
        >>> validated.tool_name
        'detect'
        >>> validated.arguments["threshold"]
        0.5
    """
    # Lazy import to avoid circular dependency
    from mata.core.tool_schema import ToolCall

    # Step 1: Fuzzy-match tool name
    matched_schema = _fuzzy_match_tool_schema(call.tool_name, schemas)
    if not matched_schema:
        available = ", ".join([s.name for s in schemas])
        raise ValueError(
            f"Tool '{call.tool_name}' not found. Available tools: {available}. "
            f"Check for typos or verify the tool is registered."
        )

    normalized_tool_name = matched_schema.name

    # Step 2: Type-coerce arguments based on parameter schemas
    coerced_arguments = {}
    param_map = {p.name: p for p in matched_schema.parameters}

    for arg_name, arg_value in call.arguments.items():
        if arg_name in param_map:
            param = param_map[arg_name]
            coerced_value = _coerce_argument_type(arg_value, param.type)
            coerced_arguments[arg_name] = coerced_value
        else:
            # Unknown parameter — keep it as-is (may be ignored by tool executor)
            logger.warning(f"Unknown parameter '{arg_name}' for tool '{normalized_tool_name}', keeping as-is")
            coerced_arguments[arg_name] = arg_value

    # Step 3: Fill defaults for missing optional parameters
    for param in matched_schema.parameters:
        if param.name not in coerced_arguments:
            if not param.required:
                coerced_arguments[param.name] = param.default
                logger.debug(f"Filled default value for '{param.name}': {param.default}")

    return ToolCall(
        tool_name=normalized_tool_name,
        arguments=coerced_arguments,
        raw_text=call.raw_text,
    )


def _fuzzy_match_tool_schema(tool_name: str, schemas: list[ToolSchema]) -> ToolSchema | None:
    """Fuzzy-match a tool name against available schemas.

    Matching strategies (in priority order):
    1. Exact match (case-insensitive)
    2. Common root match (shared prefix ≥4 chars and ≥70% of shorter string)
    3. Edit distance (Levenshtein distance ≤ 3)

    This handles common VLM variations:
    - "detection" ↔ "detect" (common root)
    - "classifier" ↔ "classify" (common root)
    - "segmentation" ↔ "segment" (common root)
    - "ditect" → "detect" (1-char typo)

    Args:
        tool_name: Tool name from VLM output
        schemas: List of available tool schemas

    Returns:
        Matched ToolSchema, or None if no match found
    """
    tool_lower = tool_name.lower().strip()

    # Strategy 1: Exact match (case-insensitive)
    for schema in schemas:
        if schema.name.lower() == tool_lower:
            return schema

    # Strategy 2: Common root matching (handles common suffixes)
    # Check if one string is the root of another (common prefix with reasonable suffix)
    for schema in schemas:
        schema_lower = schema.name.lower()
        # Find common prefix
        common_len = 0
        for i in range(min(len(tool_lower), len(schema_lower))):
            if tool_lower[i] == schema_lower[i]:
                common_len += 1
            else:
                break

        # If common prefix is at least 4 chars and covers at least 70% of the shorter string
        shorter_len = min(len(tool_lower), len(schema_lower))
        if common_len >= 4 and common_len >= shorter_len * 0.7:
            logger.debug(
                f"Fuzzy-matched '{tool_name}' → '{schema.name}' (common root: {common_len}/{shorter_len} chars)"
            )
            return schema

    # Strategy 3: Simple edit distance (Levenshtein ≤ 3)
    for schema in schemas:
        distance = _levenshtein_distance(tool_lower, schema.name.lower())
        if distance <= 3:
            logger.debug(f"Fuzzy-matched '{tool_name}' → '{schema.name}' (edit distance {distance})")
            return schema

    return None


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Pure Python implementation (no external dependencies).

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (minimum number of edits to transform s1 into s2)
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _coerce_argument_type(value: any, expected_type: str) -> any:
    """Coerce an argument value to the expected type.

    VLMs often output incorrect types (e.g., "0.5" instead of 0.5).
    This function attempts to coerce values to match parameter schemas.

    Type conversions:
    - "float" → float("0.5")
    - "int" → int("42")
    - "bool" → bool (handles "true"/"false" strings)
    - "list[str]" → list (parses "[a,b,c]" or "a,b,c")
    - "bbox" → list[float] (parses "[1,2,3,4]")
    - "str" → str (no conversion)

    Args:
        value: Raw argument value from VLM output
        expected_type: Expected type string from ToolParameter

    Returns:
        Coerced value, or original value if coercion fails

    Examples:
        >>> _coerce_argument_type("0.5", "float")
        0.5
        >>> _coerce_argument_type("true", "bool")
        True
        >>> _coerce_argument_type("cat,dog,bird", "list[str]")
        ['cat', 'dog', 'bird']
    """
    # Already correct type - fast path
    if expected_type == "float" and isinstance(value, (float, int)):
        return float(value)
    elif expected_type == "int" and isinstance(value, int):
        return value
    elif expected_type == "bool" and isinstance(value, bool):
        return value
    elif expected_type == "str" and isinstance(value, str):
        return value
    elif expected_type in ("list[str]", "bbox") and isinstance(value, list):
        return value

    # String → float
    if expected_type == "float":
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Failed to coerce '{value}' to float, keeping as-is")
            return value

    # String → int
    if expected_type == "int":
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Failed to coerce '{value}' to int, keeping as-is")
            return value

    # String → bool
    if expected_type == "bool":
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                return True
            elif value.lower() in ("false", "0", "no"):
                return False
        logger.warning(f"Failed to coerce '{value}' to bool, keeping as-is")
        return value

    # String → list[str]
    if expected_type == "list[str]":
        if isinstance(value, str):
            # Try parsing as JSON array first
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass

            # Fallback: comma-separated string
            return [item.strip() for item in value.split(",") if item.strip()]

        logger.warning(f"Failed to coerce '{value}' to list[str], keeping as-is")
        return value

    # String → bbox (list of 4 floats)
    if expected_type == "bbox":
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list) and len(parsed) >= 4:
                    return [float(x) for x in parsed[:4]]
            except (json.JSONDecodeError, ValueError):
                pass

        logger.warning(f"Failed to coerce '{value}' to bbox, keeping as-is")
        return value

    # No conversion needed or unknown type
    return value
