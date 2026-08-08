#!/bin/bash
# sanity_checks — T4/T5/T9/T10 オフラインサニティテスト
# torch 不要; ホスト venv から直接実行可能

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

ANALYSIS_DIR="cosmos_policy/experiments/robot/robocasa/analysis/results"

# Step 1: run_manifest.json を生成 (T10 の前提)
python -m cosmos_policy.experiments.robot.robocasa.analysis.run_manifest_gen \
    --results_root "$ANALYSIS_DIR" \
    --out_path     "$ANALYSIS_DIR/run_manifest.json" \
    --task_name    PnPCounterToCab \
    --n_episodes   50 \
    --seed         195 \
    --success_rate 0.60 \
    --success_count 30 \
    --n_policy_calls 1108

# Step 2: サニティテスト実行
python -m cosmos_policy.experiments.robot.robocasa.analysis.sanity_checks \
    --feat_npz "$ANALYSIS_DIR/action_features/features.npz" \
    --manifest "$ANALYSIS_DIR/run_manifest.json" \
    --out_dir  "$ANALYSIS_DIR/sanity"
