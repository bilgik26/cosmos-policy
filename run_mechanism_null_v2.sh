#!/bin/bash
# mechanism_null_v2 — §3.A/B ヌルモデル + §3.D 捕捉対応表
# torch 不要; ホスト venv から直接実行可能

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

ANALYSIS_DIR="cosmos_policy/experiments/robot/robocasa/analysis/results"

python -m cosmos_policy.experiments.robot.robocasa.analysis.mechanism_null_v2 \
    --actions_npz "$ANALYSIS_DIR/action_denoising/step_actions.npz" \
    --feat_npz    "$ANALYSIS_DIR/action_features/features.npz" \
    --light_json  "$ANALYSIS_DIR/action_denoising/denoising_light_records.json" \
    --stats_json  "$ANALYSIS_DIR/action_denoising_reanalysis/mechanism_reanalysis_stats.json" \
    --out_dir     "$ANALYSIS_DIR/mechanism_null_v2" \
    --task_name   PnPCounterToCab \
    --n_perm      1000
