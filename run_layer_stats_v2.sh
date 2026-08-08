#!/bin/bash
# layer_stats_v2 — 設計書 §3.D/E/K 準拠のオフライン層別統計解析
# torch 不要; ホスト venv から直接実行可能

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

ANALYSIS_DIR="cosmos_policy/experiments/robot/robocasa/analysis/results"

python -m cosmos_policy.experiments.robot.robocasa.analysis.layer_stats_v2 \
    --feat_npz "$ANALYSIS_DIR/action_features/features.npz" \
    --out_dir  "$ANALYSIS_DIR/action_layer_v2" \
    --task_name PnPCounterToCab \
    --success_rate 0.60
