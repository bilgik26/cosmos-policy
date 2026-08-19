#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

python3 -m cosmos_policy.experiments.robot.robocasa.analysis.verification.intervention_v3.steerability_audit \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/intervention_v3_steerability_audit \
    --pair_types L T P G S \
    --n_pairs_online 24 \
    --online_task PnPCounterToCab \
    --n_perm 50 \
    --seed 195
