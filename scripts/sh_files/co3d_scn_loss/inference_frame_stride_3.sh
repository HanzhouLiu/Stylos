#!/bin/bash
# ================================================
# Batch StyleAnySplat Inference with Multiple Scenes
# ================================================
# This script runs style-based inference for a list of CO3D scenes.
# For each scene, it:
#   - Loads the pretrained checkpoint
#   - Applies all style images in STYLE_DIR
#   - Saves rendered results into OUT_DIR/{EXP_NAME}/{scene}/styles/{style}/images/
#   - Optionally saves processed GT frames into GT_DIR/{scene}/images/ (only once)
# =================================================

# Path to pretrained checkpoint
CHECKPOINT="output/released_version/exp_co3d_style_scene_loss_4gpus/2025-09-20_16-18-24/hf_checkpoints/step_015000"

# Experiment name (used in output directory)
EXP_NAME="exp_scn_loss"

# Directory with style images (e.g., 00.jpg ... 49.jpg)
STYLE_DIR="datasets/styles"

# Directory to save processed GT frames (only saved once, skipped if already exists)
GT_DIR="output_renders/co3d/final_version/GT_FRAMES_16"

# Directory to save inference outputs
OUT_DIR="output_renders/co3d/final_version/STRIDE_3_VIEW_SIZE_16"

# ------------------------------------------------
# Frame processing parameters
# ------------------------------------------------
MAX_FRAMES=16      # Use only the first 16 frames (continuous sequence)
FRAME_STRIDE=3     # Skip 2 frames every time
BATCH_SIZE=16      # Number of content frames per inference batch
RENDER_BATCH=0     # 0 = render all views at once (faster if GPU memory allows)
# ------------------------------------------------

# List of content scenes to process
SCENES=(
  "datasets/CO3D/skateboard/20_766_1737/images"
  "datasets/CO3D/skateboard/168_18360_34837/images"
  "datasets/CO3D/skateboard/251_26915_54746/images"
  "datasets/CO3D/skateboard/374_42007_84045/images"
  "datasets/CO3D/skateboard/428_60215_117479/images"
  "datasets/CO3D/pizza/78_8049_16641/images"
  "datasets/CO3D/pizza/102_11950_20611/images"
  "datasets/CO3D/pizza/228_23990_48325/images"
  "datasets/CO3D/pizza/350_36818_69005/images"
  "datasets/CO3D/pizza/424_59147_114833/images"
  "datasets/CO3D/donut/34_1472_4748/images"
  "datasets/CO3D/donut/78_8064_17103/images"
  "datasets/CO3D/donut/129_14950_29917/images"
  "datasets/CO3D/donut/400_51425_101070/images"
  "datasets/CO3D/donut/430_60711_118886/images"
)

# Loop through each scene
for CONTENT_DIR in "${SCENES[@]}"; do
  echo "================================================"
  echo "[RUN] Processing Content Scene: $CONTENT_DIR"
  
  python -m scripts.python_files.co3d_inference \
    --checkpoint "$CHECKPOINT" \
    --content_dir "$CONTENT_DIR" \
    --style_dir "$STYLE_DIR" \
    --exp_name "$EXP_NAME" \
    --max_frames $MAX_FRAMES \
    --frame_stride $FRAME_STRIDE \
    --batch_size $BATCH_SIZE \
    --render_batch $RENDER_BATCH \
    --gt_dir "$GT_DIR" \
    --out_dir "$OUT_DIR"
done
