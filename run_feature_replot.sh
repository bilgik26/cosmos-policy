#!/bin/bash
# feature_replot — 既存の features.npz を読み込んでプロットのみ再実行する。
# torch・cosmos_policy・robosuite を使わないため Singularity 不要。
# ホスト側 venv から直接実行できる。

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

ANALYSIS_DIR="cosmos_policy/experiments/robot/robocasa/analysis/results"

python -m cosmos_policy.experiments.robot.robocasa.analysis.feature_replot \
    --npz_path "$ANALYSIS_DIR/action_features/features.npz" \
    --out_dir  "$ANALYSIS_DIR/action_features" \
    --task_name PnPCounterToCab \
    --success_rate 0.60 \
    --success_count 30 \
    --total_episodes 50
