# MATA Annotation Guide

> **Version**: v2.0  
> **Module**: `mata.annotate`  
> **Interfaces**: `mata annotate`, `mata.annotate()`

MATA ships a browser-based annotation tool that runs on Python's built-in `http.server`. There is no separate web stack to install, no Node.js toolchain, and no frontend build step. If MATA is installed, the annotation server is already available.

The v2.0 annotation workflow is built around two distinct views:

- **Browser View** — a paginated thumbnail grid for navigating datasets, filtering by split, searching by filename, and tracking annotation progress.
- **Editor View** — a three-column canvas workspace for drawing bounding boxes and polygons, managing classes, reviewing annotation properties, and running AI-assisted labeling.

Both views share a persistent theme (dark / light / system) and are served by the same local process with no external dependencies.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Browser View](#2-browser-view)
3. [Editor View](#3-editor-view)
4. [Annotation Tools](#4-annotation-tools)
5. [AI-Assisted Labeling](#5-ai-assisted-labeling)
6. [Zoom, Pan, and Brightness](#6-zoom-pan-and-brightness)
7. [Theme Toggle](#7-theme-toggle)
8. [Export and Training](#8-export-and-training)
9. [CLI Reference](#9-cli-reference)
10. [Python API Reference](#10-python-api-reference)
11. [Security Notes](#11-security-notes)
12. [Keyboard Shortcuts](#12-keyboard-shortcuts)

---

## 1. Quick Start

### What you need

- A working MATA install.
- A data root that contains one or more datasets. By default, the annotation server uses `data/`.
- A modern browser (Chrome, Firefox, Edge, Safari).

The annotation server itself adds no extra web dependency. Optional AI-assist models reuse MATA's normal model-loading stack.

### Start in 30 seconds

```bash
mata annotate --data data
```

By default, MATA:

- binds to `127.0.0.1`
- uses port `8710`
- opens your default browser automatically

Open `http://127.0.0.1:8710` if the browser does not open on its own.

You will land in the **Browser View**: a paginated thumbnail grid showing every dataset in your data root.

### Start with AI-assist models configured

```bash
mata annotate \
  --data data \
  --detect-model facebook/detr-resnet-50 \
  --vlm-model Qwen/Qwen3-VL-2B-Instruct \
  --embed-model openai/clip-vit-base-patch32

mata annotate --data data --detect-model PekingU/rtdetr_v2_r18vd --vlm-model Qwen/Qwen3-VL-2B-Instruct --embed-model openai/clip-vit-base-patch32
```

Models are loaded lazily. They are not downloaded or initialized until the first AI-assist request arrives.

### Start from Python

```python
import mata

server = mata.annotate(
    data="data",
    open_browser=False,
    block=False,
)
print(server.url)

# Later, when you are done:
server.shutdown()
```

---

## 2. Browser View

The Browser View is the entry point. It shows a paginated thumbnail grid of all images in the selected dataset with sidebar controls for filtering, sorting, and tracking progress.

### Navigating the Browser View

When you open `http://127.0.0.1:8710`, you see:

- A **left sidebar** listing every dataset in your data root. Each dataset shows its type badge (`coco`, `imagefolder`, `voc`, or `empty`).
- A **main area** with a thumbnail grid for the selected dataset.
- A **stats bar** at the top showing total image count, annotated count, and browse progress as a percentage.

Click any dataset name in the sidebar to load it. The thumbnail grid updates immediately.

### Split Tabs

Above the thumbnail grid, horizontal tabs let you filter by split:

| Tab       | Shows                                |
| --------- | ------------------------------------ |
| **All**   | Every image in the dataset (default) |
| **Train** | Images in the `train` split only     |
| **Val**   | Images in the `val` split only       |
| **Test**  | Images in the `test` split only      |

Tabs that have no images are not hidden, but their count badge shows `0`.

### Search and Filter

A search box above the thumbnail grid filters by filename (case-insensitive). Typing `face` shows only images whose filename contains `face`. The `total` counter and pagination update to reflect the filtered count.

### Sort Order

A dropdown beside the search box controls image ordering:

| Option     | Behavior                     |
| ---------- | ---------------------------- |
| `Name A→Z` | Alphabetical ascending       |
| `Name Z→A` | Alphabetical descending      |
| `Newest`   | Most recently modified first |
| `Largest`  | Largest file size first      |

### Pagination

Pagination controls appear below the grid. Use the `<` and `>` buttons to step through pages, or click a page number directly. A **per-page** dropdown lets you choose how many thumbnails appear per page.

Each thumbnail shows:

- The image itself (JPEG thumbnail served by the backend).
- A green checkmark badge (bottom-left) when the image has at least one saved annotation.
- The filename below the thumbnail.

### Opening an Image in the Editor

Click any thumbnail to open it in the **Editor View**. The browser history is updated so the Back button returns you to the same page and scroll position in the Browser View.

On **keyboard navigation**, `ArrowLeft` / `ArrowRight` move focus between thumbnails, and `Enter` opens the focused thumbnail.

### Dataset Stats

The stats bar above the grid shows:

- **Total images** in the currently filtered view.
- **Annotated** — images with at least one annotation.
- **Browse progress** — percentage of images annotated, displayed as a fill bar.

A per-split summary is also shown: how many images exist in each split (train / val / test).

### Supported Dataset Layouts

#### COCO-style detection or segmentation datasets

Minimal editable layout:

```text
data/my_coco_dataset/
├── images/
│   ├── img001.jpg
│   └── img002.jpg
└── annotations/
    └── instances.json
```

Split-style layout (also supported):

```text
data/my_coco_dataset/
├── train/
├── val/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
└── dataset.yaml
```

#### ImageFolder classification datasets

```text
data/my_classes/
├── train/
│   ├── cat/
│   └── dog/
└── val/
    ├── cat/
    └── dog/
```

### Creating a Dataset via API

```bash
curl -X POST http://127.0.0.1:8710/api/datasets \
  -H "Content-Type: application/json" \
  -d '{"name": "my_project"}'
```

This creates:

```text
data/my_project/
├── images/
└── annotations/
```

Dataset names are restricted to letters, numbers, underscores, and hyphens (max 64 characters).

---

## 3. Editor View

Clicking a thumbnail navigates to the Editor View at `/#edit?dataset=...&image=...`. The Editor View has three columns:

```
┌─────────────────────────────────────────────────────────────────────┐
│  [← Dataset]  filename.jpg   [◀ Prev]  [Next ▶]   [🌙 Theme]       │
└─────────────────────────────────────────────────────────────────────┘
┌──────────────┬────────────────────────────────────┬────────────────┐
│  Left Panel  │         Canvas (center)             │  Tool Palette  │
│  (220 px)    │                                     │  (48 px)       │
│              │                                     │                │
│  [Labels]    │                                     │  ✦ Select      │
│  [Attributes]│                                     │  ☐ BBox        │
│  [Raw Data]  │                                     │  ⬡ Polygon     │
│              │                                     │  ─ Polyline†   │
│  Layers      │                                     │  ↻ Rotate†     │
│  Classes     │                                     │  ◈ AI          │
│  Auto Annotate│                                    │  ⊢ Split†      │
│  Properties  │                                     │  ⊕ Merge†      │
│              │                                     │  ────────      │
│              │                                     │  ↩ Undo        │
│              │                                     │  ↪ Redo†       │
│              │                                     │  ────────      │
│              │                                     │  🗑 Delete     │
└──────────────┴────────────────────────────────────┴────────────────┘
│  [−] [100%] [+] [RESET]                    [☀ Brightness]          │
└─────────────────────────────────────────────────────────────────────┘
```

_† Coming Soon — visible but disabled._

### Top Bar

- **← Dataset** breadcrumb — returns to the Browser View.
- **Filename** — current image name.
- **◀ Prev / Next ▶** buttons — or use `ArrowLeft` / `ArrowRight`.
- **Theme toggle** button — cycles between Light, Dark, and System mode (see [Section 7](#7-theme-toggle)).

### Left Panel Tabs

The left panel has three tabs:

#### Labels Tab

The default tab. Contains:

1. **Layers list** — one row per annotation on the current image. Click a row to select that annotation and highlight it on the canvas. The active row is highlighted.
2. **Classes sub-tab** — a color legend mapping each category to its canvas color. Colors use a centralized 8-color palette to guarantee that legend swatches always match canvas annotations.
3. **Auto Annotate section** — run AI-assisted labeling (see [Section 5](#5-ai-assisted-labeling)).
4. **Annotation Properties panel** — appears below the layers list when an annotation is selected (see below).

#### Annotation Properties Panel

When an annotation is selected on the canvas, the bottom of the Labels tab shows its properties inline:

| Field        | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| **Bbox**     | Bounding box in `xyxy` format (read-only; edit by dragging)  |
| **Area**     | Bounding box area in px² (computed, read-only)               |
| **Category** | Dropdown — changing it updates the annotation and auto-saves |
| **Score**    | Confidence badge (only shown for AI-assist candidates)       |

Properties update live during drag and resize operations.

Deselecting an annotation (pressing `Escape` or clicking empty canvas) replaces the panel with a _"Select an annotation to view its properties"_ placeholder.

#### Attributes Tab

Shows key-value metadata attached to the selected annotation.

- Each attribute is shown as a read-only key field and an editable value field.
- Click the **×** button to delete an attribute.
- Click **+ Add Attribute** to append a new key-value pair inline. Confirm with `Enter` or the ✓ button, cancel with `Escape` or the × button.
- Value changes are persisted automatically (auto-save debounce).

When no annotation is selected, the tab shows a placeholder.

#### Raw Data Tab

Displays the current image's COCO JSON (image record + annotations + categories) in a syntax-highlighted `<pre>` block.

- Keys are highlighted in the accent color.
- String values in the highlight color.
- Numbers and booleans in distinct colors.
- A **Copy JSON** button copies the raw string to the clipboard.
- The view updates automatically whenever annotations are added, modified, or deleted.

### Right Tool Palette

A vertical strip of 40×40 px buttons on the right edge of the editor. Only one tool is active at a time (radio behavior).

| Button     | Shortcut | State       | Action                                      |
| ---------- | -------- | ----------- | ------------------------------------------- |
| ✦ Select   | `V`      | Active      | Click/drag to select and move annotations   |
| ☐ BBox     | `B`      | Active      | Draw bounding boxes                         |
| ⬡ Polygon  | `P`      | Active      | Draw polygons, vertex by vertex             |
| — Polyline | —        | Coming Soon | Disabled (tooltip: "Coming Soon")           |
| ↻ Rotate   | —        | Coming Soon | Disabled                                    |
| ◈ AI       | `A`      | Active      | Scrolls left panel to Auto Annotate section |
| ⊢ Split    | —        | Coming Soon | Disabled                                    |
| ⊕ Merge    | —        | Coming Soon | Disabled                                    |
| ↩ Undo     | `Ctrl+Z` | Active      | Undo the last annotation action             |
| ↪ Redo     | `Ctrl+Y` | Coming Soon | Disabled                                    |
| 🗑 Delete  | `Delete` | Active      | Delete the selected annotation              |

Active tool has a highlighted background (`--accent-soft` fill). Tool state resets to **Select** when switching to a new image.

### Bottom Bar

- **−** button — zoom out 10%.
- **Zoom percentage** label — shows current zoom level (e.g., `100%`).
- **+** button — zoom in 10%.
- **RESET** button — restores 100% zoom, centers the image.
- **☀ Brightness** button — opens a popover with Brightness and Contrast sliders (see [Section 6](#6-zoom-pan-and-brightness)).

---

## 4. Annotation Tools

### Bounding Box Tool (`B`)

- Click and drag on the canvas to create a box.
- Release the pointer to open the class picker.
- Click an existing box to select it; a move handle and 8 resize handles appear.
- Drag the box body to move it; drag a resize handle to reshape it.

Stored coordinates are always in image-pixel space (xyxy). The canvas scales the image for display and maps pointer events back to original pixels at any zoom level.

### Polygon Tool (`P`)

- Click once per vertex to place points.
- Double-click the first vertex, double-click anywhere, or press `Enter` to close the polygon (minimum 3 vertices required).
- Assign a class in the class picker.
- Select an existing polygon to move the full shape.
- Drag individual vertices to refine the boundary.

Polygons are stored in COCO `segmentation: [[x1, y1, ...]]` format. A tight bounding box is computed automatically.

### Select Tool (`V`)

- Click any annotation to select it.
- Drag the selected annotation to move it.
- Drag resize handles to reshape bounding boxes. Individual polygon vertices are draggable in polygon-select mode.
- Press `Escape` to deselect.

### Autosave and Undo

- Every action is immediately pushed to the undo stack (last 50 states).
- Changes are autosaved after a short debounce window.
- `Ctrl+S` / `Cmd+S` saves immediately.
- `Ctrl+Z` / `Cmd+Z` restores the previous state.
- `Ctrl+Y` / `Cmd+Y` — Redo (Coming Soon).

### Class Management

- The class picker appears after drawing a new annotation; type to filter or click to assign.
- The **last used class** pre-selects on the next draw action.
- To change a class after drawing, use the **Category** dropdown in the Annotation Properties panel.
- Add new categories via the **+ add class** chip in the class picker.

---

## 5. AI-Assisted Labeling

### Overview

The **Auto Annotate** section in the Labels tab provides three modes. A mode dropdown switches between them:

| Mode       | Uses                   | Prompt input                  |
| ---------- | ---------------------- | ----------------------------- |
| **Detect** | Detection model        | Confidence threshold (0–1)    |
| **VLM**    | VLM model              | Free-text prompt textarea     |
| **CLIP**   | Embedding / CLIP model | Class names (comma-separated) |

Press the **ANNOTATE** button to run. A status indicator shows:

- Spinner while the request is in-flight.
- ✓ when results arrive.
- Error message if the required model is not loaded.

AI-generated candidates appear on the canvas as **dashed-border draft annotations**. Each draft carries the assigned class and confidence score. Accept a draft by clicking it to confirm, or press `Delete` to dismiss it.

### Starting the Server with Models

```bash
# Detection mode
mata annotate --data data --detect-model facebook/detr-resnet-50

# VLM mode
mata annotate --data data --vlm-model Qwen/Qwen3-VL-2B-Instruct

# CLIP mode
mata annotate --data data --embed-model openai/clip-vit-base-patch32

# All three
mata annotate --data data \
  --detect-model facebook/detr-resnet-50 \
  --vlm-model Qwen/Qwen3-VL-2B-Instruct \
  --embed-model openai/clip-vit-base-patch32
```

When a mode's required model is not loaded, clicking **ANNOTATE** returns a `501 Not Implemented` response and the UI shows a descriptive message.

### CLIP Mode Behavior

CLIP mode pre-fills the class names field from the dataset's existing categories, reducing setup friction. You can edit this list before running.

### API Access (Advanced)

The same endpoints are available directly:

```bash
# Detection
curl -X POST http://127.0.0.1:8710/api/assist/auto-annotate \
  -H "Content-Type: application/json" \
  -d '{"dataset": "coco_mini", "image_filename": "images/000001.jpg", "threshold": 0.30}'

# VLM
curl -X POST http://127.0.0.1:8710/api/assist/vlm \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/abs/path/to/image.jpg", "class_names": ["person", "car"], "max_new_tokens": 512}'

# CLIP
curl -X POST http://127.0.0.1:8710/api/assist/classify \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/abs/path/to/image.jpg", "class_names": ["cat", "dog"], "top_k": 3}'
```

---

## 6. Zoom, Pan, and Brightness

All zoom and pan interactions work correctly at any annotation zoom level. Drawing, selecting, moving, and resizing annotations behave identically at 50% or 500%.

### Zoom

| Action                        | Effect                                    |
| ----------------------------- | ----------------------------------------- |
| **Mouse wheel** (scroll up)   | Zoom in ~10%; anchors to cursor position  |
| **Mouse wheel** (scroll down) | Zoom out ~10%; anchors to cursor position |
| **+** button (bottom bar)     | Zoom in 10%                               |
| **−** button (bottom bar)     | Zoom out 10%                              |
| **RESET** button              | Restore 100% zoom, center the image       |

Zoom range: **10% – 500%**. The current percentage is always shown in the bottom bar.

Zoom anchors to the mouse cursor position so the point under the cursor stays fixed during a scroll event.

### Pan

| Action              | Effect         |
| ------------------- | -------------- |
| Hold `Space` + drag | Pan the canvas |
| Middle-click + drag | Pan the canvas |

The canvas clamps panning so the image edge cannot move past the canvas center — you can never lose the image off-screen.

### Brightness and Contrast

Click the **☀** button in the bottom bar to open the brightness popover. It contains:

- **Brightness** slider — 0% to 200% (default 100%).
- **Contrast** slider — 0% to 200% (default 100%).
- **Reset** button — restores both to 100%.

These adjustments are display-only (CSS filter). They do not modify the source image or any saved annotation.

---

## 7. Theme Toggle

The theme button in the top bar of the Editor View and the corresponding control in the Browser View cycle through three modes:

| Mode       | Behavior                                                    |
| ---------- | ----------------------------------------------------------- |
| **Light**  | Light background, dark text                                 |
| **Dark**   | Dark background, light text                                 |
| **System** | Follows the OS / browser `prefers-color-scheme` media query |

The selected mode is stored in `localStorage` under the key `mata-annotate-theme` and persists across page reloads and browser restarts.

Theme transitions animate smoothly (300 ms ease on `background`, `color`, and `border-color` properties). Both views observe the theme: all panels, canvas elements, badges, and buttons re-theme together.

---

## 8. Export and Training

> [!NOTE]
> The training integration (`mata.train()`) is a beta feature in v2.0.0. See the [Training Guide](TRAINING_GUIDE.md) for details.

### Exporting a Dataset

```bash
curl -X POST http://127.0.0.1:8710/api/datasets/my_project/export \
  -H "Content-Type: application/json" \
  -d '{"class_names": ["person", "car"], "split_ratio": 0.8}'
```

Export writes:

- `dataset.yaml`
- `annotations/instances_train.json`
- `annotations/instances_val.json`
- `train/` and `val/` image directories

Example generated YAML:

```yaml
path: D:/data/my_project
train: train
val: val
train_annotations: annotations/instances_train.json
val_annotations: annotations/instances_val.json
names:
  0: person
  1: car
```

For the full training data contract, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md).

### Launching Training from the Browser

The browser sidebar includes a training panel where you can configure:

- mode: `train` or `finetune`
- task: `detect`, `classify`, or `segment`
- model, data path, epochs, batch size, learning rate, device

Clicking **Start Training** sends a background request to `/api/train` and polls `/api/train/status` until completion.

Behavior:

- only one job runs at a time
- stop requests are best-effort
- the server stays responsive while training runs

### Training from Python

```python
import mata

result = mata.train(
    "detect",
    model="facebook/detr-resnet-50",
    data="data/my_project/dataset.yaml",
    epochs=10,
    batch_size=4,
)

print(result.best_checkpoint)
print(result.final_metrics)
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for full options.

---

## 9. CLI Reference

```bash
mata annotate [options]
```

### Options

| Flag             | Default     | Description                           |
| ---------------- | ----------- | ------------------------------------- |
| `--data`         | `data`      | Data directory to manage              |
| `--host`         | `127.0.0.1` | Bind address                          |
| `--port`         | `8710`      | Server port                           |
| `--no-browser`   | `False`     | Do not auto-open the browser          |
| `--detect-model` | `None`      | Detection model for AI-assist         |
| `--vlm-model`    | `None`      | VLM model for AI-assist               |
| `--embed-model`  | `None`      | Embedding model for CLIP-style assist |

### Common Examples

```bash
# Default local launch
mata annotate

# Different data root, no auto-browser
mata annotate --data D:/datasets --no-browser

# Custom port
mata annotate --data data --port 9000

# All interfaces (use with caution)
mata annotate --data data --host 0.0.0.0

# With assist models
mata annotate \
  --data data \
  --detect-model facebook/detr-resnet-50 \
  --vlm-model Qwen/Qwen3-VL-2B-Instruct
```

### Stopping the Server

Press `Ctrl+C`. The CLI prints `Server stopped.` and exits cleanly.

For broader CLI coverage, see [CLI_REFERENCE.md](CLI_REFERENCE.md).

---

## 10. Python API Reference

### Signature

```python
mata.annotate(
    data: str = "data",
    *,
    host: str = "127.0.0.1",
    port: int = 8710,
    open_browser: bool = True,
    block: bool = True,
    detect_model: str | None = None,
    vlm_model: str | None = None,
    embed_model: str | None = None,
    **kwargs,
) -> Any
```

### Parameters

| Parameter      | Default       | Description                                                    |
| -------------- | ------------- | -------------------------------------------------------------- |
| `data`         | `"data"`      | Root data directory to manage                                  |
| `host`         | `"127.0.0.1"` | Bind address                                                   |
| `port`         | `8710`        | Requested port                                                 |
| `open_browser` | `True`        | Open the default browser automatically                         |
| `block`        | `True`        | Block until shutdown, or return immediately in background mode |
| `detect_model` | `None`        | Detection model ID or alias                                    |
| `vlm_model`    | `None`        | VLM model ID or alias                                          |
| `embed_model`  | `None`        | Embedding model ID or alias                                    |
| `**kwargs`     | —             | Reserved for future server configuration                       |

### Return Value

`mata.annotate()` returns the running `AnnotateServer` instance. In non-blocking mode this is your shutdown handle.

Useful attributes and methods:

- `server.url` — full bind URL, e.g. `http://127.0.0.1:8710`
- `server.serve_forever()` — block the calling thread
- `server.shutdown()` — stop the server
- `server.get_training_status()` — poll background training state

### Examples

```python
import mata

# Start in background mode
server = mata.annotate(
    data="data",
    block=False,
    open_browser=False,
    detect_model="facebook/detr-resnet-50",
)
print("Annotate running at", server.url)

# Custom port, non-blocking
server = mata.annotate(
    data="D:/datasets",
    host="127.0.0.1",
    port=9000,
    block=False,
    open_browser=False,
)
```

---

## 11. Security Notes

### Localhost-First Binding

- Default host is `127.0.0.1`.
- Use `0.0.0.0` only when you explicitly want remote access and have network controls in place.

### Path Traversal Protection

User-controlled paths (static files, dataset images, thumbnails, training data) are validated against the configured data root. Requests that attempt to escape are rejected.

### Safe Identifiers

Dataset names, class names, and category names are restricted to letters, numbers, underscores, and hyphens. Path-like names such as `../evil` are rejected.

### Request Size Limit

JSON request bodies are limited to 10 MB. Larger bodies are rejected with `413 Payload Too Large`.

### Training Path Enforcement

Training requests must reference a `data` path inside the configured annotation data root. This prevents the browser from launching training against arbitrary filesystem paths.

---

## 12. Keyboard Shortcuts

Shortcuts are **view-scoped**. All editor shortcuts are guarded by the active view check and do not fire while focus is in a text `<input>` or `<textarea>`.

### Browser View Shortcuts

| Shortcut     | Action                                           |
| ------------ | ------------------------------------------------ |
| `ArrowLeft`  | Move focus to the previous thumbnail in the grid |
| `ArrowRight` | Move focus to the next thumbnail in the grid     |
| `Enter`      | Open the focused thumbnail in the Editor View    |

### Editor View Shortcuts

| Shortcut               | Scope       | Action                                                                   |
| ---------------------- | ----------- | ------------------------------------------------------------------------ |
| `V`                    | Editor only | Activate the Select tool                                                 |
| `B`                    | Editor only | Activate the Bounding Box tool                                           |
| `P`                    | Editor only | Activate the Polygon tool                                                |
| `A`                    | Editor only | Scroll left panel to Auto Annotate section                               |
| `Enter`                | Editor only | Close the in-progress polygon (≥ 3 vertices required)                    |
| `Escape`               | Editor only | Cancel current action → deselect annotation → close mobile panel overlay |
| `Delete` / `Backspace` | Editor only | Delete the selected annotation                                           |
| `Ctrl+Z` / `Cmd+Z`     | Editor only | Undo the previous action                                                 |
| `Ctrl+Y` / `Cmd+Y`     | Editor only | Redo (Coming Soon — currently a no-op with `preventDefault`)             |
| `Ctrl+S` / `Cmd+S`     | Editor only | Save annotations immediately (prevents browser Save Page dialog)         |
| `ArrowRight`           | Editor only | Navigate to the next image                                               |
| `ArrowLeft`            | Editor only | Navigate to the previous image                                           |
| `Space` + drag         | Editor only | Pan the canvas while the key is held                                     |

---

## Related Documentation

- [CLI_REFERENCE.md](CLI_REFERENCE.md)
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- [GRAPH_API_REFERENCE.md](GRAPH_API_REFERENCE.md)
