#!/bin/bash
# ================================================
# Batch evaluation for StyleAnySplat CO3D results
# ================================================
# This script:
#   - Calls co3d_eval.py with --base_dir and --artscore_model
#   - The Python script iterates over all scenes/styles
#   - Saves results into BASE_DIR/eval_results.csv
# =================================================

# Path to trained RAFT checkpoint
RAFT_MODEL="checkpoints/raft-things.pth"

# Path to ArtScore checkpoint
ARTSCORE_MODEL="checkpoints/loss@listMLE_model@resnet50_denseLayer@True_batch_size@16_lr@0.0001_dropout@0.5_E_8.pth"

# Base directory where stylized outputs are saved
BASE_DIR="output_renders/co3d/final_version/STRIDE_3_VIEW_SIZE_16/exp_scn_loss"

# Mode: "short", "long", or "both"
EVAL_MODE="both"

# Long-range gap (number of frames apart for long-range evaluation)
LONG_GAP=7

# Run evaluation (Python script handles looping & averaging, ArtScore + flow-based metrics)
python3 -m scripts.python_files.co3d_eval \
  --model "$RAFT_MODEL" \
  --artscore_model "$ARTSCORE_MODEL" \
  --base_dir "$BASE_DIR" \
  --mode "$EVAL_MODE" \
  --long_gap "$LONG_GAP"
