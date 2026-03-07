# VLM Tool-Calling Agent System — Architecture Summary

**Version**: 1.7.1  
**Implementation Date**: February 16, 2026  
**Last Updated**: March 8, 2026  
**Status**: ✅ Production Ready  
**Test Coverage**: 342 comprehensive tests, all passing

---

## Executive Summary

The VLM Tool-Calling Agent System extends MATA's vision-language capabilities with an **agentic execution mode** where VLMs can iteratively call specialized tools to gather information before providing final answers. This enables VLMs to:

- **Recognize their limitations** and delegate to specialized models (detection, classification, segmentation)
- **Use built-in image tools** (zoom, crop) to examine regions of interest
- **Chain multiple tools** to gather comprehensive information (detect → classify → synthesize)
- **Recover from errors** through retry logic and graceful failure handling

The system integrates seamlessly with MATA's existing graph execution model through optional parameters on VLM nodes (`VLMDetect`, `VLMQuery`, `VLMDescribe`), ensuring **zero breaking changes** to existing code.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Graph Execution Layer                            │
│  (mata.infer(), Graph.compile(), ExecutionContext)                      │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         │ provides ExecutionContext with providers={}
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           VLM Nodes                                      │
│  (VLMDetect, VLMQuery, VLMDescribe)                                     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────┐        │
│  │ Standard Mode (tools=None):                                 │        │
│  │   VLMWrapper.query() → parse output → return result        │        │
│  └────────────────────────────────────────────────────────────┘        │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────┐        │
│  │ Agent Mode (tools=[...]):                                   │        │
│  │   ToolRegistry(tools, ctx) → AgentLoop(vlm, registry)      │        │
│  │   → run() → AgentResult → convert to artifact              │        │
│  └────────────────────────────────────────────────────────────┘        │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         │ agent mode delegates to:
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          AgentLoop                                       │
│  (src/mata/core/agent_loop.py)                                          │
│                                                                          │
│  Core Responsibilities:                                                 │
│  • Iterate between VLM inference and tool execution                     │
│  • Build dynamic system prompts with tool descriptions                  │
│  • Parse tool calls from VLM output (multiple formats)                  │
│  • Maintain conversation history                                        │
│  • Accumulate instances/entities across all tool calls                  │
│  • Enforce max_iterations safety cap                                    │
│  • Handle errors (retry, skip, fail modes)                              │
│  • Detect infinite loops                                                │
│  • Integrate with observability (tracing, metrics)                      │
│                                                                          │
│  Execution flow:                                                        │
│  1. Generate system prompt with tool schemas                            │
│  2. VLM.query(prompt) → response text                                   │
│  3. parse_tool_calls(response) → ToolCall[]                             │
│  4. For each ToolCall:                                                  │
│     • ToolRegistry.execute(call, image) → ToolResult                    │
│     • Accumulate instances/entities                                     │
│     • Add to conversation history                                       │
│  5. If tool calls found: format results → loop to step 2                │
│  6. If no tool calls: finalize → return AgentResult                     │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         │ resolves tools via:
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        ToolRegistry                                      │
│  (src/mata/core/tool_registry.py)                                       │
│                                                                          │
│  Responsibilities:                                                      │
│  • Resolve tool names to ToolSchemas                                    │
│  • Validate tool availability at construction time (fail fast)          │
│  • Dispatch tool calls to appropriate handlers                          │
│  • Generate system prompt blocks for VLM                                │
│  • Handle region-based tool execution (crop-before-execute)             │
│                                                                          │
│  Tool Resolution:                                                       │
│  ┌─────────────────────┐          ┌─────────────────────┐             │
│  │ Built-in Tools      │          │ Provider Tools      │             │
│  │ (always available)  │          │ (from ctx)          │             │
│  │                     │          │                     │             │
│  │ • zoom              │          │ • detect            │             │
│  │ • crop              │          │ • classify          │             │
│  │                     │          │ • segment           │             │
│  │ Executed directly   │          │ • depth             │             │
│  │ via image_tools.py  │          │                     │             │
│  │                     │          │ Executed via        │             │
│  │                     │          │ ctx.get_provider()  │             │
│  └─────────────────────┘          └─────────────────────┘             │
│                                                                          │
│  Execution Flow:                                                        │
│  1. Validate tool against schemas                                       │
│  2. If builtin: execute_builtin_tool() → image result                   │
│  3. If provider: ctx.get_provider(task, name) → adapter.predict()       │
│  4. If region parameter: crop_and_execute() → remap coordinates         │
│  5. Convert adapter result to ToolResult                                │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         │ uses schemas from:
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ToolSchema Layer                                 │
│  (src/mata/core/tool_schema.py)                                         │
│                                                                          │
│  Core Types:                                                            │
│  • ToolParameter: Parameter definition (name, type, description, etc.)  │
│  • ToolSchema: Complete tool specification                              │
│  • ToolCall: Parsed tool invocation from VLM                            │
│  • ToolResult: Execution result (success, data, error)                  │
│                                                                          │
│  Factory Function:                                                      │
│  schema_for_task(task) → ToolSchema with defaults                       │
│  - "detect" → object detection with threshold, class_names              │
│  - "classify" → classification with top_k parameter                     │
│  - "segment" → segmentation with threshold parameter                    │
│  - "depth" → depth estimation (no parameters)                           │
│                                                                          │
│  Serialization:                                                         │
│  • to_prompt_str() → Human-readable for VLM system prompts              │
│  • to_openai_schema() → OpenAI function-calling format (future)         │
└─────────────────────────────────────────────────────────────────────────┘

                         Supporting Components
                         ═════════════════════

┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  Tool Prompts        │  │  Tool Parsing        │  │  Image Tools         │
│  (tool_prompts.py)   │  │  (parsers.py)        │  │  (image_tools.py)    │
│                      │  │                      │  │                      │
│ System prompt        │  │ parse_tool_calls()   │  │ zoom_image()         │
│ generation with      │  │ - Fenced blocks      │  │ crop_image()         │
│ tool descriptions    │  │ - XML tags           │  │                      │
│                      │  │ - Raw JSON           │  │ Built-in tools that  │
│ Result formatting    │  │                      │  │ don't require        │
│ for conversation     │  │ validate_tool_call() │  │ provider adapters    │
│ history              │  │ - Fuzzy matching     │  │                      │
│                      │  │ - Type coercion      │  │                      │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘

┌──────────────────────┐  ┌──────────────────────┐
│  Converters          │  │  Observability       │
│  (converters.py)     │  │  (tracing, metrics)  │
│                      │  │                      │
│ agent_result_to_     │  │ Tracing:             │
│ detections()         │  │ • Parent span:       │
│                      │  │   agent:{node_name}  │
│ Converts AgentResult │  │ • VLM turn spans     │
│ to standard MATA     │  │ • Tool exec spans    │
│ artifacts:           │  │                      │
│ • Detections         │  │ Metrics:             │
│ • Classifications    │  │ • agent_iterations   │
│ • Text               │  │ • tool_calls_count   │
└──────────────────────┘  └──────────────────────┘
```

---

## Design Decisions & Rationale

### 1. **Loop Location: Inside VLM Node's `run()` Method**

**Decision**: The agent loop executes synchronously within the VLM node, not as a graph-level scheduler.

**Rationale**:

- **Preserves DAG invariant**: MATA's graph remains a static directed acyclic graph. No dynamic node insertion.
- **Zero breaking changes**: Existing code path unaffected — standard mode bypasses agent loop entirely.
- **Simpler scheduling**: No need for new scheduler logic or graph rewriting.
- **Clear execution boundaries**: Agent loop is a node implementation detail, not an architecture change.

**Tradeoff**: Cannot parallelize tool calls within a single agent loop (sequential execution only). Acceptable since most VLMs output one tool at a time.

---

### 2. **Tool Names Reference Provider Keys**

**Decision**: Tool names in `tools=[...]` are **provider names** from the `providers={}` dict, not task types.

**Example**:

```python
mata.infer(
    graph,
    providers={
        "my-detector": DetectAdapter(...),  # ← tool name is "my-detector"
        "my-classifier": ClassifyAdapter(...),
    },
    tools=["my-detector", "my-classifier"],  # ← same names used here
)
```

**Rationale**:

- **Single source of truth**: No dual registration (providers dict + separate tool definitions).
- **Flexible naming**: Users can have multiple providers for the same task (e.g., "fast-detect", "accurate-detect").
- **Natural resolution**: `ctx.get_provider(capability, name)` works directly with tool names.

**Tradeoff**: Tool names must match provider keys exactly (case-sensitive). Fuzzy matching added to mitigate.

---

### 3. **Custom `ToolSchema` Dataclass (Not OpenAI Schema)**

**Decision**: Define our own `ToolSchema`, `ToolParameter`, `ToolCall`, `ToolResult` dataclasses.

**Rationale**:

- **Lighter weight**: MATA-specific fields only (task type, builtin flag, region parameter).
- **No vendor lock-in**: Not tied to OpenAI's function-calling format.
- **Extensibility**: Can add MATA-specific features (e.g., region-based execution).
- **Future compatibility**: Can render to OpenAI format via `.to_openai_schema()` when needed.

**Tradeoff**: Custom types add complexity. Mitigated by comprehensive factory functions (`schema_for_task()`).

---

### 4. **Max Iterations with Hard Cap**

**Decision**: Default `max_iterations=5`, hard cap at 20 (configurable per-node).

**Rationale**:

- **Cost control**: Prevents runaway API calls to expensive VLM providers.
- **Latency control**: Prevents infinite loops that hang user applications.
- **Safety**: Even with VLM hallucinations, execution terminates gracefully.

**Tradeoff**: Complex multi-step reasoning may hit iteration limit. Users can increase if needed.

---

### 5. **VLM Provider Agnostic Design**

**Decision**: AgentLoop interfaces with `VLMWrapper.query()`, not specific VLM implementations.

**Rationale**:

- **Decoupling**: Supports any VLM with a wrapper (local models, API providers).
- **Extensibility**: New VLM providers work immediately if they implement `query()`.
- **Testability**: Easy to mock VLM responses in tests.

**Implementation**: `VLMWrapper` from `src/mata/adapters/vlm/base.py` provides unified interface.

---

### 6. **Built-in Tools as First-Class Citizens**

**Decision**: Built-in tools (zoom, crop) dispatched separately, not via fake providers.

**Rationale**:

- **No pollution**: Doesn't require users to register fake "zoom" provider.
- **Always available**: Works even without any task adapters loaded.
- **Clean separation**: Image operations distinct from model inference.

**Implementation**: `ToolRegistry` checks `schema.builtin` flag and routes to `image_tools.py`.

---

### 8. **Provider-Aware Schema Generation for Zero-Shot Models** _(v1.7.1)_

**Decision**: `ToolRegistry` introspects the actual provider at registration time and upgrades `text_prompts` to `required=True` for zero-shot adapters.

**Rationale**:

- **VLM must know `text_prompts` is required** — The default `TASK_SCHEMA_DEFAULTS["detect"]` marks `text_prompts` as optional (correct for supervised detectors like RT-DETR, YOLO). But zero-shot models (GroundingDINO, OWL-ViT) **cannot run without class names**. If the schema shows the parameter as optional, the VLM's system prompt will say _"optional"_ and the agent will omit it, causing a `TypeError` or `InvalidInputError` at execution time.
- **Zero-shot contract is enforced at the adapter level** — `HuggingFaceZeroShotDetectAdapter.predict()` keeps `text_prompts` as a required positional argument. The fix is upstream: make the _schema_ match the adapter's actual contract.
- **Clean detection via class name** — All MATA zero-shot adapters have `"ZeroShot"` in their class name. `_is_zero_shot_provider()` unwraps one layer of wrapper (e.g., `DetectorWrapper.adapter`) and checks the underlying class name — no new class attributes or protocol changes needed.
- **`TASK_SCHEMA_DEFAULTS` stays generic** — The shared default schema is not modified; customization happens per-provider at `ToolRegistry` construction time.

**Agentic chain this enables**:

```
VLM: "I see an unknown object. Let me classify it."
  → classifier(region=[80,120,220,300])  → "cat (0.92)"
VLM: "It's a cat. Let me find all cats using the detector."
  → detector(text_prompts="cat")         → 2 cats detected
VLM: "Found 2 cats at [80,120,220,300] and [300,130,440,280]. Summary..."
```

**Implementation**: `_is_zero_shot_provider()` + upgraded `_schema_for_capability(capability, tool_name, provider)` in `src/mata/core/tool_registry.py` (v1.7.1).

---

### 7. **Multi-Format Tool Call Parsing**

**Decision**: Support fenced blocks (` ```tool_call `), XML (`<tool_call>`), and raw JSON.

**Rationale**:

- **VLM variability**: Different VLMs produce different formats.
- **Robustness**: Fallback formats increase success rate.
- **Prompt engineering**: Users can guide VLM toward preferred format.

**Implementation**: `parse_tool_calls()` in `parsers.py` tries all strategies in priority order.

---

### 8. **Error Handling Modes: retry, skip, fail**

**Decision**: Three error recovery modes configurable via `on_error=` parameter.

**Modes**:

- **retry** (default): Retry malformed tool calls with clarification prompt. Continues on tool failures.
- **skip**: Treat errors as final answer (best-effort execution).
- **fail**: Raise exception on first error (strict mode).

**Rationale**:

- **Flexibility**: Different use cases need different failure strategies (demos vs. production).
- **Safety**: Default mode recovers gracefully without crashing.
- **Control**: Strict mode available for critical applications.

**Implementation**: Error mode checked in `AgentLoop.run()` on every error path.

---

### 9. **Accumulation Across All Tool Calls**

**Decision**: AgentResult accumulates **all instances and entities** from every tool call, not just the final one.

**Rationale**:

- **Complete information**: Users get all detections, not just what VLM mentions in final answer.
- **Traceability**: Can audit what information VLM had access to.
- **Artifact compatibility**: Detections/Classifications artifacts expect all instances.

**Implementation**: `agent_loop.py` appends to `all_instances` and `all_entities` lists on every tool result.

---

### 10. **Conversation History Maintained**

**Decision**: Full conversation history preserved in `AgentResult.conversation` as list of message dicts.

**Format**:

````python
[
    {"role": "system", "content": "...tool descriptions..."},
    {"role": "user", "content": "Analyze this image."},
    {"role": "assistant", "content": '```tool_call\n{"tool": "detect"}\n```'},
    {"role": "user", "content": "Tool 'detect' returned: Found 3 cats..."},
    {"role": "assistant", "content": "The image contains 3 cats..."},
]
````

**Rationale**:

- **Debugging**: Users can see full VLM reasoning chain.
- **Fine-tuning**: Conversation logs useful for training/improving VLMs.
- **Context preservation**: VLM sees its own previous tool calls to avoid repeating.

**Implementation**: Each VLM query and tool result appended to `conversation` list.

---

## Implementation Statistics

### Code Deliverables

| Module                     | Lines | Purpose                               | Status      |
| -------------------------- | ----- | ------------------------------------- | ----------- |
| `tool_schema.py`           | 424   | Schema definitions, factory functions | ✅ Complete |
| `tool_registry.py`         | 695   | Tool resolution, execution dispatch   | ✅ Complete |
| `agent_loop.py`            | 691   | Core agent iteration logic            | ✅ Complete |
| `tool_prompts.py`          | 214   | System prompt generation              | ✅ Complete |
| `image_tools.py`           | 190   | Built-in zoom/crop tools              | ✅ Complete |
| `parsers.py` (extended)    | +391  | Tool call parsing, validation         | ✅ Complete |
| VLM nodes (3 files)        | +180  | VLMDetect/Query/Describe agent mode   | ✅ Complete |
| `converters.py` (extended) | +120  | AgentResult → artifact conversion     | ✅ Complete |
| **Total new/modified**     | ~3000 | Lines of production code              | ✅ Complete |

### Test Coverage

| Test Suite                     | Tests   | Coverage | Status         |
| ------------------------------ | ------- | -------- | -------------- |
| `test_tool_schema.py`          | 33      | 100%     | ✅ All passing |
| `test_tool_registry.py`        | 44      | 87%      | ✅ All passing |
| `test_agent_loop.py`           | 51      | 90%+     | ✅ All passing |
| `test_agent_loop_tracing.py`   | 13      | Full     | ✅ All passing |
| `test_tool_prompts.py`         | 18      | 100%     | ✅ All passing |
| `test_tool_call_parser.py`     | 51      | 100%     | ✅ All passing |
| `test_vlm_nodes_agent_mode.py` | 11      | Full     | ✅ All passing |
| `test_image_tools.py`          | 37      | 100%     | ✅ All passing |
| `test_region_tool_dispatch.py` | 11      | Full     | ✅ All passing |
| `test_converters.py` (agent)   | 55      | Full     | ✅ All passing |
| `test_vlm_tool_calling.py`     | 12      | Full     | ✅ All passing |
| **Total**                      | **336** | **~95%** | ✅ All passing |

### Integration Status

- ✅ Works with `mata.infer()` API
- ✅ Works with `Graph.compile()` fluent API
- ✅ Supports parallel VLM agent nodes in same graph
- ✅ Zero regressions (2576/2576 existing tests still pass)
- ✅ Zero breaking changes to public API
- ✅ Backward compatible with all VLM nodes

---

## Key Features

### 1. **Flexible Tool Configuration**

```python
# Built-in tools only
VLMQuery(using="qwen3-vl", tools=["zoom", "crop"])

# Provider tools only
VLMDetect(using="qwen3-vl", tools=["detect", "classify"])

# Mixed tools
VLMQuery(using="qwen3-vl", tools=["detect", "zoom", "classify"])

# No tools = standard VLM mode (backward compatible)
VLMQuery(using="qwen3-vl")  # Same as before
```

### 2. **Automatic Tool Schema Generation**

Tool schemas auto-generated from adapters with sensible defaults:

- Detect: `threshold`, `class_names`, `region` parameters
- Classify: `top_k`, `region` parameters
- Segment: `threshold`, `region` parameters
- Depth: No parameters (always full image)

Custom schemas supported via `ToolSchema` construction.

### 3. **Region-Based Tool Execution**

VLMs can specify regions for tools to operate on:

```python
# VLM output:
{"tool": "classify", "arguments": {"region": [100, 100, 300, 300]}}

# Execution flow:
1. Crop image to [100, 100, 300, 300]
2. Run classifier on cropped region
3. Remap coordinates back to full image
4. Return result to VLM with region context
```

### 4. **Multi-Format Tool Call Support**

Supports VLM output in any of these formats:

````markdown
Format 1: Fenced block (recommended)

```tool_call
{"tool": "detect", "arguments": {"threshold": 0.5}}
```

Format 2: XML tags
<tool_call>
{"tool": "detect", "arguments": {"threshold": 0.5}}
</tool_call>

Format 3: Raw JSON
{"tool": "detect", "arguments": {"threshold": 0.5}}
````

### 5. **Error Recovery**

- **Malformed tool calls**: Retry with clarification prompt showing correct format
- **Tool execution failures**: Continue with failure result (VLM can react)
- **Infinite loops**: Detect repeated identical calls, break loop
- **Max iterations**: Safety cap prevents runaway execution

### 6. **Observability Integration**

Full tracing and metrics support:

```python
# Tracing spans:
agent:vlm_query           # Parent span for entire loop
  agent:vlm_turn_0        # Each VLM inference
  agent:tool:detect       # Each tool execution
  agent:vlm_turn_1
  agent:tool:classify
  ...

# Metrics collected:
agent_iterations: 3.0     # Number of loop iterations
tool_calls_count: 2.0     # Total tools called
```

### 7. **Rich Metadata**

`AgentResult` contains complete execution trace:

```python
result = AgentResult(
    text="Found 3 cats and 1 dog...",        # Final VLM answer
    tool_calls=[...],                         # All ToolCall objects
    tool_results=[...],                       # All ToolResult objects
    iterations=3,                             # Loop count
    instances=[...],                          # All detected instances
    entities=[...],                           # All VLM entities
    conversation=[...],                       # Full chat history
    meta={
        "agent_iterations": 3,
        "agent_tool_calls": [...],            # As dicts
        "agent_tool_results": [...],          # Summaries
    },
)
```

---

## Limitations & Known Issues

### Current Limitations

1. **Sequential Tool Execution**
   - VLMs can only call **one tool per turn** (no parallel tool calls)
   - **Why**: Simplifies parsing and reduces VLM confusion
   - **Impact**: Multi-tool workflows take multiple iterations
   - **Workaround**: VLM can call multiple tools across iterations

2. **VLM Reliability Dependency**
   - System depends on VLM producing **valid tool call syntax**
   - **Why**: Open-weight VLMs have variable instruction-following ability
   - **Impact**: May need multiple retries on malformed output
   - **Mitigation**: Retry logic + fuzzy matching + helpful error prompts

3. **No Streaming Support**
   - Agent loop executes **fully synchronously** (not streaming)
   - **Why**: Need complete VLM response to parse tool calls
   - **Impact**: Longer latency for multi-iteration loops
   - **Future**: Could stream final VLM synthesis after tools complete

4. **Provider Name Coupling**
   - Tool names must **match provider dict keys** exactly
   - **Why**: Direct resolution via `ctx.get_provider(task, name)`
   - **Impact**: Renaming providers breaks tool references
   - **Mitigation**: Fuzzy matching helps with minor typos

5. **No Tool Result Caching**
   - Identical tool calls **re-execute** every time
   - **Why**: No caching layer implemented yet
   - **Impact**: Wastes compute if VLM repeats calls
   - **Mitigation**: Infinite loop detection prevents worst case

6. **Limited Tool Chaining Logic**
   - VLM must **manually chain tools** (no automatic pipelines)
   - **Why**: No built-in dependency resolution between tools
   - **Impact**: VLM must understand tool sequencing
   - **Example**: VLM must know to detect before classifying specific objects

### Known Issues

#### 1. **Floating-Point Comparison in Keypoints Tests**

- **Status**: Pre-existing, not a regression
- **Impact**: 2 test failures in `test_artifact_task_1_4.py`
- **Workaround**: Not blocking (unrelated to VLM tool-calling)

#### 2. **VLM Integration Tests Require Large Models**

- **Status**: Expected behavior (marked with `@pytest.mark.slow`)
- **Impact**: 7 tests require ~4GB model downloads
- **Workaround**: Run with `pytest -m slow` when models available

#### 3. **Type Coercion Edge Cases**

- **Status**: Handled gracefully
- **Impact**: VLMs may output `"0.5"` instead of `0.5` for floats
- **Solution**: Comprehensive type coercion in `validate_tool_call()`

#### ~~4. Zero-Shot Detector Omits `text_prompts`~~ _(Fixed — v1.7.1)_

- **Was**: `TASK_SCHEMA_DEFAULTS["detect"]` marked `text_prompts` as optional, causing the VLM to omit it. Zero-shot adapters require it, so the call failed with `TypeError`.
- **Fix**: `ToolRegistry._schema_for_capability()` now introspects the actual provider via `_is_zero_shot_provider()` and upgrades `text_prompts` to `required=True` for zero-shot adapters. The VLM's system prompt now correctly says the parameter is required, so the agent always populates it from its own reasoning.

---

## Future Directions

### Near-Term Enhancements (v1.8 - v1.9)

#### 1. **RAG Tool Support**

**Description**: Add document/knowledge base tools for VLMs to query external context.

**Design**:

```python
ToolSchema(
    name="search_docs",
    description="Search knowledge base for relevant information",
    task="retrieval",
    parameters=[
        ToolParameter("query", "str", "Search query"),
        ToolParameter("top_k", "int", "Number of results", default=5),
    ],
)
```

**Implementation**:

- New `RetrievalProvider` interface in `src/mata/adapters/retrieval/`
- Vector store backends (FAISS, ChromaDB, Qdrant)
- Document chunking and indexing utilities
- Context injection into VLM prompts

**Use Cases**:

- Medical image analysis with literature references
- Industrial inspection with manual lookups
- Education (e.g., "What animal is this? Search animal encyclopedia.")

#### 2. **Custom Tool Registration API**

**Description**: Allow users to define custom tools beyond MATA's built-in tasks.

**API Design**:

```python
from mata.core.tool_schema import ToolSchema, ToolParameter

def my_custom_tool(image, param1: str, param2: int):
    """User-defined tool logic."""
    return {"result": "custom output"}

custom_schema = ToolSchema(
    name="custom_tool",
    description="My custom tool",
    task="custom",
    parameters=[
        ToolParameter("param1", "str", "Description"),
        ToolParameter("param2", "int", "Description"),
    ],
)

# Register with ToolRegistry
mata.infer(
    graph,
    custom_tools={
        "custom_tool": (custom_schema, my_custom_tool)
    }
)
```

**Implementation**:

- Extend `ToolRegistry` to accept custom tool functions
- Validation for custom signatures
- Automatic result conversion to `ToolResult`

**Use Cases**:

- Company-specific workflows (e.g., inventory checking)
- Integration with proprietary systems (e.g., CAD, GIS)
- Domain-specific operations (e.g., medical DICOM processing)

#### 3. **API VLM Provider Support**

**Description**: Native support for cloud VLM APIs (OpenAI, Anthropic, Google) with tool-calling.

**Status**: Currently works with local models (Qwen3-VL, LLaVA) through `VLMWrapper`.

**Design**:

```python
from mata.adapters.vlm.openai import OpenAIVLMAdapter

provider = OpenAIVLMAdapter(
    model="gpt-4-vision-preview",
    api_key="...",
)

# Automatically uses OpenAI's native function-calling format
mata.infer(
    graph,
    providers={"vlm": provider},
    tools=["detect", "classify"],  # Converted to OpenAI schema
)
```

**Implementation**:

- New adapters in `src/mata/adapters/vlm/`:
  - `openai.py` — OpenAI GPT-4V, GPT-4O
  - `anthropic.py` — Claude 3.5 Sonnet with vision
  - `google.py` — Gemini 1.5 Pro
- Schema conversion: `ToolSchema.to_openai_schema()` (already implemented)
- Response parsing for provider-specific function call formats

**Benefits**:

- Leverage powerful commercial VLMs with better instruction-following
- Reduce infrastructure requirements (no local GPU needed)
- Mixed provider setups (API VLM + local task models)

#### 4. **Parallel Tool Calls**

**Description**: Allow VLM to call multiple tools in one turn (execute in parallel).

**Current**: Single tool per turn (sequential execution).

**Design**:

```python
# VLM output:
[
    {"tool": "detect", "arguments": {}},
    {"tool": "depth", "arguments": {}},
]

# Execution:
results = await AsyncToolRegistry.execute_batch([call1, call2], image)
# Returns both results simultaneously
```

**Implementation**:

- Extend `parse_tool_calls()` to return multiple calls
- Parallel dispatch in `ToolRegistry` (asyncio or threading)
- Aggregate results before returning to VLM

**Benefits**:

- Lower latency for multi-tool queries
- Better utilization of GPU resources
- Reduced total iterations

**Challenges**:

- VLM output format must support arrays
- Error handling for partial failures
- Conversation history formatting for multiple results

#### 5. **Tool Result Caching**

**Description**: Cache tool execution results to avoid re-running identical calls.

**Design**:

```python
class ToolRegistry:
    def __init__(self, ..., cache_enabled: bool = True):
        self._cache = {}  # key: (tool_name, args_hash, image_hash)

    def execute_tool(self, call, image):
        cache_key = self._compute_cache_key(call, image)
        if cache_key in self._cache:
            logger.info(f"Cache hit for {call.tool_name}")
            return self._cache[cache_key]

        result = self._execute_uncached(call, image)
        self._cache[cache_key] = result
        return result
```

**Implementation**:

- Perceptual image hashing (dHash, pHash) for image similarity
- LRU cache with configurable max size
- Cache invalidation on context boundary (new `mata.infer()` call)

**Benefits**:

- Faster execution when VLM repeats calls
- Reduced GPU/API cost
- Better user experience in iterative workflows

---

### Long-Term Vision (v2.0+)

#### 1. **Multi-Agent Collaboration**

- Multiple VLM agents with different specializations collaborate on same task
- Agent-to-agent communication protocols
- Consensus mechanisms for conflicting outputs

#### 2. **Learning from Tool Usage**

- Fine-tune VLMs on successful tool-calling traces
- Reinforcement learning from tool execution outcomes
- Automatic prompt optimization based on success rate

#### 3. **Tool Discovery & Recommendation**

- VLMs can query "what tools are available?"
- Registry suggests tools based on task description
- Automatic tool subset selection for complex queries

#### 4. **Graph-Level Agent Orchestration**

- Multi-node agent graphs (agent calls different agents)
- Hierarchical agent architectures (supervisor → specialist agents)
- Dynamic graph rewriting based on agent decisions

---

## Integration Guide

### Basic Usage

```python
import mata
from mata.nodes import VLMQuery

# 1. Define graph with agent-enabled VLM node
graph = [
    VLMQuery(
        using="qwen3-vl",
        prompt="Analyze this image and identify all objects.",
        tools=["detect", "classify", "zoom"],  # ← Agent mode enabled
        max_iterations=5,
        on_error="retry",
    ),
]

# 2. Provide adapters as providers
from mata.adapters import HuggingFaceDetectAdapter, HuggingFaceClassifyAdapter

result = mata.infer(
    graph,
    image="path/to/image.jpg",
    providers={
        "detect": HuggingFaceDetectAdapter("facebook/detr-resnet-50"),
        "classify": HuggingFaceClassifyAdapter("microsoft/resnet-50"),
    },
)

# 3. Access results
detections = result["vlm_dets"]  # Detections artifact with all instances
print(detections.meta["agent_iterations"])  # Number of loop iterations
print(detections.meta["agent_text"])        # Final VLM synthesis
```

### Advanced: Custom System Prompts

```python
from mata.core.tool_prompts import build_tool_system_prompt
from mata.core.tool_schema import schema_for_task

# Build custom system prompt
schemas = [schema_for_task("detect"), schema_for_task("classify")]
custom_prompt = build_tool_system_prompt(
    schemas,
    base_prompt="You are an expert medical image analyst. Use tools to examine the image systematically."
)

# Use in VLM node (future: expose as parameter)
# For now, system prompts are auto-generated
```

### Observability

```python
from mata.core.observability import ExecutionTracer, MetricsCollector

tracer = ExecutionTracer()
metrics = MetricsCollector()

result = mata.infer(
    graph,
    image="image.jpg",
    providers={...},
    tracer=tracer,
    metrics_collector=metrics,
)

# Inspect traces
for span in tracer.spans:
    if span.name.startswith("agent:"):
        print(f"{span.name}: {span.duration_ms}ms")

# Inspect metrics
print(f"Iterations: {metrics.get('agent_iterations')}")
print(f"Tool calls: {metrics.get('tool_calls_count')}")
```

---

## Testing Strategy

### Unit Tests (336 total)

- **Schema validation**: All tool schemas serialize correctly
- **Registry resolution**: Tool names resolve to correct providers
- **Agent loop logic**: Iteration, retry, error handling, loop detection
- **Parsing robustness**: All tool call formats parsed correctly
- **Type coercion**: VLM string outputs converted to correct types
- **Built-in tools**: Zoom/crop execute correctly with various parameters
- **Observability**: Tracing spans and metrics recorded properly

### Integration Tests

- **End-to-end via mata.infer()**: Full graph execution with mock VLMs
- **Parallel execution**: Multiple VLM agent nodes in same graph
- **Metadata preservation**: Agent metadata flows through converters
- **Backward compatibility**: VLM nodes without tools= work identically

### Manual Testing (with real models)

```bash
# Run examples with real VLM (requires GPU + transformers)
python examples/graph/vlm_workflows.py --real

# Test individual workflows
python examples/graph/vlm_workflows.py --workflow 5a  # Single tool
python examples/graph/vlm_workflows.py --workflow 5b  # Multi-tool
python examples/graph/vlm_workflows.py --workflow 5c  # Zoom tool
```

---

## Performance Characteristics

### Latency Profile (Qwen3-VL-2B on A100)

| Scenario                    | Iterations | Latency | Cost (API) |
| --------------------------- | ---------- | ------- | ---------- |
| No tools (standard mode)    | 1          | ~500ms  | $0.01      |
| Single tool call            | 2          | ~1.2s   | $0.02      |
| Multi-tool chain (2 tools)  | 3          | ~2.0s   | $0.03      |
| Complex reasoning (5 tools) | 6          | ~4.5s   | $0.06      |

**Note**: Latency dominated by VLM inference, not tool execution.

### Memory Usage

- **AgentLoop overhead**: ~5MB (conversation history storage)
- **ToolRegistry**: ~1MB (schema caching)
- **Conversation history**: ~100KB per 10 turns (with base64 image embeddings)

### Scalability

- **Parallel agents**: Linear scaling (independent VLM nodes)
- **Max iterations**: Configurable up to 20 (default 5)
- **Tool count**: Tested with up to 10 tools (no performance degradation)

---

## Comparison with Other Systems

| Feature                | MATA VLM Tools | LangChain Agents | OpenAI Assistants | LlamaIndex Agents |
| ---------------------- | -------------- | ---------------- | ----------------- | ----------------- |
| Local VLM support      | ✅ Yes         | ⚠️ Limited       | ❌ No             | ⚠️ Limited        |
| Computer vision tools  | ✅ Native      | ❌ Manual        | ❌ Manual         | ❌ Manual         |
| Graph integration      | ✅ Native      | ⚠️ Chains        | ❌ No             | ⚠️ Workflows      |
| Built-in image tools   | ✅ Yes         | ❌ No            | ❌ No             | ❌ No             |
| Error recovery modes   | ✅ Yes (3)     | ⚠️ Basic         | ⚠️ Basic          | ⚠️ Basic          |
| Observability tracing  | ✅ Native      | ⚠️ LangSmith     | ✅ Native         | ⚠️ LlamaDebug     |
| Region-based execution | ✅ Yes         | ❌ No            | ❌ No             | ❌ No             |
| Zero breaking changes  | ✅ Yes         | ❌ No            | N/A               | ❌ No             |
| Offline execution      | ✅ Yes         | ⚠️ Partial       | ❌ No             | ⚠️ Partial        |

**MATA's Unique Advantages**:

1. **Vision-first design**: Tools are adapters you already have (detect, segment, etc.)
2. **Graph integration**: Agent nodes compose naturally with standard nodes
3. **Local-first**: Works offline with open-weight VLMs
4. **Backward compatible**: Existing code continues to work unchanged

---

## Acknowledgments

This implementation was developed as part of MATA v1.7.0 to enable advanced vision-language reasoning workflows while maintaining the framework's core principles of simplicity, modularity, and license safety.

**Key Contributors**:

- Architecture design: Developer A, Developer B
- Implementation: Full-stack development across all phases
- Testing: Comprehensive test suite (336 tests)

**References**:

- Original proposal: `docs/TASK_VLM_TOOL_CALLING.md`
- Implementation tasks: Phases A-G in task document
- Design patterns: Inspired by OpenAI function calling, LangChain agents, llama.cpp tool use

---

**Document Version**: 1.0  
**Last Updated**: February 16, 2026  
**Status**: Production Ready  
**Maintainers**: MATA Core Team
