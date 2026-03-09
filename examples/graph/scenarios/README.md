# MATA Real-World Scenario Examples

This directory contains production-grade example scripts demonstrating MATA's application to real-world computer vision problems across multiple industries.

## Overview

Each example script:

- ✅ Solves a specific industry problem using existing MATA graph presets
- ✅ Includes comprehensive documentation with problem context
- ✅ Supports **mock mode** (default, no model downloads) and **real mode** (actual inference)
- ✅ Follows consistent code patterns for easy understanding

## Quick Start

```bash
# Mock mode — verify graph construction, no downloads
python retail_shelf_analysis.py

# Real mode — actual inference with models
python retail_shelf_analysis.py --real shelf_image.jpg
```

## Industries & Scenarios

### 🏭 Manufacturing (4 scenarios)

| Script                               | Problem                                   | Models               | Graph Pattern             |
| ------------------------------------ | ----------------------------------------- | -------------------- | ------------------------- |
| `manufacturing_defect_classify.py`   | Surface defect detection & classification | GroundingDINO + CLIP | Detect > ROI > Classify   |
| `manufacturing_defect_segment.py`    | Defect segmentation & measurement         | GroundingDINO + SAM  | Detect > Segment > Refine |
| `manufacturing_assembly_verify.py`   | Assembly verification with VLM            | Qwen3-VL + DETR      | VLM ‖ Detect > Fuse       |
| `manufacturing_component_inspect.py` | Per-component VLM inspection              | DETR + Qwen3-VL      | Detect > ROI > VLM        |

### 🛒 Retail (3 scenarios)

| Script                     | Problem                                  | Models                 | Graph Pattern                 |
| -------------------------- | ---------------------------------------- | ---------------------- | ----------------------------- |
| `retail_shelf_analysis.py` | Product detection + brand classification | Faster R-CNN + CLIP    | Detect > NMS > ROI > Classify |
| `retail_product_search.py` | Zero-shot product search & segmentation  | GroundingDINO + SAM    | Detect > Segment              |
| `retail_stock_level.py`    | Multi-modal stock assessment             | Qwen3-VL + DETR + CLIP | VLM ‖ Detect ‖ Classify       |

### 🚗 Autonomous Driving (4 scenarios)

| Script                           | Problem                               | Models                                   | Graph Pattern             |
| -------------------------------- | ------------------------------------- | ---------------------------------------- | ------------------------- |
| `driving_distance_estimation.py` | Vehicle distance estimation           | DETR + Depth Anything                    | Detect ‖ Depth > Fuse     |
| `driving_road_scene.py`          | Complete road scene analysis          | 4 models (detect/segment/depth/classify) | 4-way parallel > Fuse     |
| `driving_traffic_tracking.py`    | Traffic object tracking               | RT-DETR + BYTETrack                      | Detect > Track > Annotate |
| `driving_obstacle_vlm.py`        | Obstacle detection with VLM reasoning | Qwen3-VL + GroundingDINO + Depth         | VLM ‖ Detect ‖ Depth      |

### 🔒 Security/Surveillance (3 scenarios)

| Script                              | Problem                                    | Models                         | Graph Pattern           |
| ----------------------------------- | ------------------------------------------ | ------------------------------ | ----------------------- |
| `security_crowd_monitoring.py`      | Person detection + tracking                | DETR + BYTETrack               | Detect > Filter > Track |
| `security_suspicious_object.py`     | Suspicious object detection + VLM analysis | GroundingDINO + SAM + Qwen3-VL | Detect > Segment > VLM  |
| `security_situational_awareness.py` | Situational awareness monitoring           | Qwen3-VL + GroundingDINO       | VLM > PromoteEntities   |

### 🌾 Agriculture (3 scenarios)

| Script                            | Problem                       | Models               | Graph Pattern           |
| --------------------------------- | ----------------------------- | -------------------- | ----------------------- |
| `agriculture_disease_classify.py` | Crop disease classification   | GroundingDINO + CLIP | Detect > ROI > Classify |
| `agriculture_aerial_crop.py`      | Aerial crop segmentation      | Mask2Former + Depth  | Segment ‖ Depth > Fuse  |
| `agriculture_pest_detection.py`   | Pest detection & area mapping | GroundingDINO + SAM  | Detect > Segment        |

### 🏥 Healthcare (3 scenarios)

⚠️ **DISCLAIMER**: Research and demonstration purposes only. NOT for clinical use.

| Script                         | Problem                         | Models                 | Graph Pattern                 |
| ------------------------------ | ------------------------------- | ---------------------- | ----------------------------- |
| `medical_roi_segmentation.py`  | ROI segmentation & measurement  | GroundingDINO + SAM    | Detect > Segment              |
| `medical_report_generation.py` | Medical image report generation | Qwen3-VL               | VLM > Fuse                    |
| `medical_pathology_triage.py`  | Pathology triage workflow       | DETR + CLIP + Qwen3-VL | Detect > ROI > Classify > VLM |

## Usage Patterns

### Mock Mode (Default)

No model downloads required. Verifies graph construction and prints expected output structure.

```bash
python retail_shelf_analysis.py
```

**Output:**

```
=== Retail: Shelf Product Analysis (Mock Mode) ===

Real-World Problem:
  Retailers need automated shelf monitoring...

Graph Flow:
  Detect > Filter > NMS > ExtractROIs > Classify > Fuse

✓ Graph 'shelf_product_analysis' constructed with 6 nodes
```

### Real Mode

Downloads and runs actual models for inference.

```bash
python retail_shelf_analysis.py --real path/to/image.jpg
```

**Output:**

```
=== Retail Shelf Analysis Results ===

Detected 15 products on shelf:
  • coca_cola: confidence 0.92 at (100, 50, 150, 200)
  • pepsi: confidence 0.88 at (160, 55, 210, 205)
  ...

=== Brand Summary ===
  coca_cola: 5 units
  pepsi: 4 units
  ...
```

## Key Features

### 1. Zero-Shot Capabilities

Many examples use **zero-shot models** (GroundingDINO, CLIP) that work without task-specific training:

```python
# Natural language prompts — no training needed
text_prompts = "red can . blue bottle . cereal box"
```

### 2. Multi-Modal Analysis

Examples combine multiple vision tasks for comprehensive insights:

```python
# VLM + Detection + Classification in parallel
stock_level_analysis()  # > semantic + quantitative + categorical
```

### 3. Preset Reuse

Demonstrates how to reuse existing presets with custom prompts:

```python
# Same preset, different domains
grounding_dino_sam()
# Retail: "red can . blue bottle"
# Manufacturing: "scratch . crack . dent"
# Agriculture: "pest . diseased_leaf"
```

## File Structure

```
scenarios/
├── README.md                              # This file
├── manufacturing_defect_classify.py       # Manufacturing: Defect classification
├── manufacturing_defect_segment.py        # Manufacturing: Defect segmentation
├── manufacturing_assembly_verify.py       # Manufacturing: Assembly verification
├── manufacturing_component_inspect.py     # Manufacturing: Component inspection
├── retail_shelf_analysis.py               # Retail: Shelf product analysis
├── retail_product_search.py               # Retail: Product search
├── retail_stock_level.py                  # Retail: Stock level assessment
├── driving_distance_estimation.py         # Driving: Distance estimation
├── driving_road_scene.py                  # Driving: Road scene analysis
├── driving_traffic_tracking.py            # Driving: Traffic tracking
├── driving_obstacle_vlm.py                # Driving: Obstacle detection
├── security_crowd_monitoring.py           # Security: Crowd monitoring
├── security_suspicious_object.py          # Security: Suspicious object detection
├── security_situational_awareness.py      # Security: Situational awareness
├── agriculture_disease_classify.py        # Agriculture: Disease classification
├── agriculture_aerial_crop.py             # Agriculture: Aerial crop analysis
├── agriculture_pest_detection.py          # Agriculture: Pest detection
├── medical_roi_segmentation.py            # Healthcare: ROI segmentation
├── medical_report_generation.py           # Healthcare: Report generation
└── medical_pathology_triage.py            # Healthcare: Pathology triage
```

## Learning Path

**Beginners**:

1. Start with `retail_shelf_analysis.py` — simple detect > classify pattern
2. Try `retail_product_search.py` — learn zero-shot detection
3. Explore `manufacturing_defect_classify.py` — see domain transfer

**Intermediate**:

1. Study `retail_stock_level.py` — multi-modal parallel execution
2. Try `driving_road_scene.py` — 4-way parallel analysis
3. Explore `manufacturing_assembly_verify.py` — VLM integration

**Advanced**:

1. Customize preset parameters in any example
2. Combine presets from multiple examples
3. Build your own industry-specific preset

## Model Requirements

### Common Models

- **Detection**: `facebook/detr-resnet-50`, `torchvision/fasterrcnn_resnet50_fpn_v2`
- **Zero-Shot Detection**: `IDEA-Research/grounding-dino-tiny`
- **Classification**: `openai/clip-vit-base-patch32`
- **Segmentation**: `facebook/sam-vit-base`
- **Depth**: `depth-anything/Depth-Anything-V2-Small-hf`
- **VLM**: `Qwen/Qwen3-VL-2B-Instruct`

### Installation

```bash
# Install MATA with all dependencies
pip install datamata[all]

# Or install specific model types
pip install datamata[transformers]  # HuggingFace models
pip install datamata[torchvision]   # Torchvision models
```

## Support

- **Showcase Documentation**: See [docs/REAL_WORLD_SCENARIOS.md](../../../docs/REAL_WORLD_SCENARIOS.md) for comprehensive industry guide
- **Implementation Details**: See [docs/TASK_REAL_WORLD_SCENARIOS.md](../../../docs/TASK_REAL_WORLD_SCENARIOS.md) for development documentation
- **Presets API**: `src/mata/presets/` for available graph factories
- **Core API**: `mata.load()`, `mata.infer()` for model loading and execution

## Contributing

These examples serve as reference implementations. When creating new scenarios:

1. Follow the established docstring pattern
2. Support both mock and real modes
3. Include real-world problem context
4. Provide actionable output analysis
5. Test in mock mode before committing

---

**Version**: MATA v1.6.1  
**Last Updated**: February 15, 2026  
**Total Scenarios**: 20 across 6 industries (all complete ✅)
