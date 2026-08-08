#!/bin/bash
# dim_analysis_v2 — 設計書 §3.F 準拠の次元別アクション変化量解析
# torch 不要; ホスト venv から直接実行可能

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

ANALYSIS_DIR="cosmos_policy/experiments/robot/robocasa/analysis/results"

python -m cosmos_policy.experiments.robot.robocasa.analysis.dim_analysis_v2 \
    --actions_npz "$ANALYSIS_DIR/action_denoising/step_actions.npz" \
    --out_dir     "$ANALYSIS_DIR/action_dim_analysis" \
    --task_name   PnPCounterToCab
