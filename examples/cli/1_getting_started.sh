#!/usr/bin/env bash
# MATA CLI -- Getting Started
#
# A progressive tutorial covering the most common mata commands.
# Run from the examples/cli/ directory:
#
#   cd examples/cli
#   bash 1_getting_started.sh
#
# Prerequisites:
#   pip install datamata
#
# Download the test image if it is not already present:
#   curl -o ../images/000000039769.jpg \
#        http://images.cocodataset.org/val2017/000000039769.jpg

set -euo pipefail

IMAGE="../images/000000039769.jpg"
DETECT_MODEL="facebook/detr-resnet-50"
CLASSIFY_MODEL="openai/clip-vit-base-patch32"

echo "=== mata -- Getting Started ==="

# -----------------------------------------------
# Version and help
# -----------------------------------------------

# Print the installed version
mata --version

# List all available subcommands
mata --help

# Full help text for a specific subcommand
mata run --help

# -----------------------------------------------
# First detection
# -----------------------------------------------

# Detect objects and print a human-readable summary
mata run detect "$IMAGE" --model "$DETECT_MODEL"

# -----------------------------------------------
# Confidence filtering
# -----------------------------------------------

# Only show predictions above 50% confidence
mata run detect "$IMAGE" --model "$DETECT_MODEL" --conf 0.5

# -----------------------------------------------
# Save output
# -----------------------------------------------

# Save an annotated result image to runs/
mata run detect "$IMAGE" --model "$DETECT_MODEL" --save

# -----------------------------------------------
# JSON output
# -----------------------------------------------

# Emit raw JSON to stdout -- useful for shell pipelines
mata run detect "$IMAGE" --model "$DETECT_MODEL" --json

# -----------------------------------------------
# Classification
# -----------------------------------------------

# Zero-shot CLIP classification -- always requires --text labels
mata run classify "$IMAGE" --model "$CLASSIFY_MODEL" --text "cat,dog,bird,car,person"

# Zero-shot: supply your own class labels (comma-separated)
mata run classify "$IMAGE" --model "$CLASSIFY_MODEL" \
    --text "cat,dog,bird,car,person"

# -----------------------------------------------
# Verbosity control
# -----------------------------------------------

# -v  : reduced logging (warnings only)
mata -v run detect "$IMAGE" --model "$DETECT_MODEL"

# -vv : verbose logging (debug output)
mata -vv run detect "$IMAGE" --model "$DETECT_MODEL"

echo "Done."
