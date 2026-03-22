#!/usr/bin/env bash
# mata track -- multi-object tracking in video files and streams.
#
# Run from: examples/cli/
#   bash 3_track_examples.sh
#
# Prerequisites:
#   pip install datamata
#   Video: ../videos/cup.mp4  (included in the repo)

set -euo pipefail

VIDEO="../videos/cup.mp4"
MODEL="facebook/detr-resnet-50"

# -----------------------------------------------
# Basic tracking
# -----------------------------------------------

# Track objects using the default BotSort tracker
mata track "$VIDEO" --model "$MODEL"

# -----------------------------------------------
# Tracker selection
# -----------------------------------------------

# BotSort (default) -- appearance-assisted motion tracking
mata track "$VIDEO" --model "$MODEL" --tracker botsort

# ByteTrack -- pure motion, lighter-weight
mata track "$VIDEO" --model "$MODEL" --tracker bytetrack

# -----------------------------------------------
# Detection thresholds
# -----------------------------------------------

# Lower confidence threshold catches more objects
mata track "$VIDEO" --model "$MODEL" --conf 0.3

# Tune both confidence and IoU simultaneously
mata track "$VIDEO" --model "$MODEL" --conf 0.3 --iou 0.5

# -----------------------------------------------
# Save output video
# -----------------------------------------------

mata track "$VIDEO" --model "$MODEL" --save --save-dir runs/track/

# -----------------------------------------------
# JSON output (per-frame track data)
# -----------------------------------------------

mata track "$VIDEO" --model "$MODEL" --json

# -----------------------------------------------
# Appearance-based ReID (BotSort only)
# pip install datamata
# -----------------------------------------------

# Activate ReID to maintain consistent IDs through occlusions
mata track "$VIDEO" \
    --model "$MODEL" \
    --tracker botsort \
    --reid-model openai/clip-vit-base-patch32

# ReID with saved annotated output
mata track "$VIDEO" \
    --model "$MODEL" \
    --tracker botsort \
    --reid-model openai/clip-vit-base-patch32 \
    --save --save-dir runs/track_reid/

# -----------------------------------------------
# RTSP stream (replace with your camera URL)
# -----------------------------------------------

# mata track "rtsp://admin:password@192.168.1.1:554/stream" \
#     --model "$MODEL" \
#     --tracker botsort \
#     --conf 0.4

# -----------------------------------------------
# Webcam (device index 0) with live display
# -----------------------------------------------

# mata track "0" --model "$MODEL" --tracker botsort --show
