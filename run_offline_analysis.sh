#!/bin/bash
# Offline analysis scripts (no simulation needed).
# Run after feature_analysis and crossattn_analysis complete.
# Executed inside Singularity container.

export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/data/bilgehan.sakai/cosmos-policy/cache/huggingface

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

ANALYSIS_DIR="cosmos_policy/experiments/robot/robocasa/analysis/results"
FEATURES_NPZ="$ANALYSIS_DIR/action_features/features.npz"
STEP_ACTIONS_NPZ="$ANALYSIS_DIR/action_denoising/step_actions.npz"

echo "=== [1/3] Layer-wise Analysis ==="
python -m cosmos_policy.experiments.robot.robocasa.analysis.layer_analysis \
    --npz_path "$FEATURES_NPZ" \
    --out_dir "$ANALYSIS_DIR/action_layer" \
    --task_name PnPCounterToCab \
    --success_rate 0.70

echo "=== [2/3] Linear Probe Analysis ==="
python -m cosmos_policy.experiments.robot.robocasa.analysis.linear_probe \
    --feat_npz "$FEATURES_NPZ" \
    --actions_npz "$STEP_ACTIONS_NPZ" \
    --out_dir "$ANALYSIS_DIR/action_probe" \
    --task_name PnPCounterToCab \
    --success_rate 0.70

echo "=== [3/3] Dimension Analysis ==="
python -m cosmos_policy.experiments.robot.robocasa.analysis.dimension_analysis

echo "=== Offline analysis complete ==="
