#!/bin/bash
# mechanism_reanalysis — 設計書 §3.A/B/C 準拠のオフライン再解析
# torch 不要; ホスト venv から直接実行可能

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

ANALYSIS_DIR="cosmos_policy/experiments/robot/robocasa/analysis/results"

python -m cosmos_policy.experiments.robot.robocasa.analysis.verification.mechanism.mechanism_reanalysis \
    --actions_npz "$ANALYSIS_DIR/action_denoising/step_actions.npz" \
    --light_json  "$ANALYSIS_DIR/action_denoising/denoising_light_records.json" \
    --out_dir     "$ANALYSIS_DIR/action_denoising_reanalysis" \
    --task_name   PnPCounterToCab \
    --success_rate 0.60
