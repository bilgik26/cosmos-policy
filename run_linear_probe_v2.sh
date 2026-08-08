#!/bin/bash
# linear_probe_v2 — 設計書 §3.G 準拠の改訂版線形プローブ
# fold内PCA + LogisticRegression + Permutation + BH FDR
# torch 不要; ホスト venv から直接実行可能

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

ANALYSIS_DIR="cosmos_policy/experiments/robot/robocasa/analysis/results"

python -m cosmos_policy.experiments.robot.robocasa.analysis.linear_probe_v2 \
    --feat_npz    "$ANALYSIS_DIR/action_features/features.npz" \
    --actions_npz "$ANALYSIS_DIR/action_denoising/step_actions.npz" \
    --out_dir     "$ANALYSIS_DIR/action_probe_v2" \
    --task_name   PnPCounterToCab \
    --success_rate 0.60 \
    --n_pca    30 \
    --n_perm  100 \
    --n_boot 1000
