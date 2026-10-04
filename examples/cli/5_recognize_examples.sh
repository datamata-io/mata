#!/usr/bin/env bash
# mata recognize -- gallery-based identity matching.
#
# Run from: examples/cli/
#   bash 5_recognize_examples.sh
#
# Prerequisites:
#   pip install datamata
#
# A .npz gallery file must be built before running mata recognize.
# Create one with the following Python snippet (run once):
#
#   python - << 'EOF'
#   import mata
#   gallery = mata.Gallery(similarity_thresh=0.7)
#   for label, path in [("cat_scene", "../images/000000039769.jpg"),
#                       ("other_scene", "../images/000000015338.jpg")]:
#       emb = mata.run("embed", path, model="openai/clip-vit-base-patch32")
#       gallery.add(label, emb[0])
#   gallery.save("gallery.npz")
#   print("gallery.npz written")
#   EOF

set -euo pipefail

IMAGE="../images/000000039769.jpg"
GALLERY="./gallery.npz"
MODEL="openai/clip-vit-base-patch32"

# -----------------------------------------------
# Basic recognition
# -----------------------------------------------

# Return the single best match from the gallery
mata recognize "$IMAGE" --gallery "$GALLERY" --model "$MODEL"

# -----------------------------------------------
# Top-k results
# -----------------------------------------------

# Return the top-3 closest matches
mata recognize "$IMAGE" --gallery "$GALLERY" --model "$MODEL" --top-k 3

# -----------------------------------------------
# Similarity threshold
# -----------------------------------------------

# Only return matches above 0.80 cosine similarity
mata recognize "$IMAGE" --gallery "$GALLERY" --model "$MODEL" --threshold 0.80

# -----------------------------------------------
# JSON output
# -----------------------------------------------

# Structured JSON with scores and labels
mata recognize "$IMAGE" --gallery "$GALLERY" --model "$MODEL" --json

# Combined: top-5, threshold, JSON
mata recognize "$IMAGE" \
    --gallery "$GALLERY" \
    --model "$MODEL" \
    --top-k 5 \
    --threshold 0.70 \
    --json

# -----------------------------------------------
# Device selection
# -----------------------------------------------

# Run embedding extraction on CUDA
# mata recognize "$IMAGE" --gallery "$GALLERY" --model "$MODEL" --device cuda
