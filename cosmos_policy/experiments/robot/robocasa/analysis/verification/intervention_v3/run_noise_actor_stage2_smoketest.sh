#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

python3 -m cosmos_policy.experiments.robot.robocasa.analysis.verification.intervention_v3.noise_actor_stage2 \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/_smoketest_noise_actor_stage2 \
    --task_name PnPCounterToCab --seed 195 \
    --n_target_success 2 --max_attempts 6 --calls_per_episode 2 --m_invert 4 --damping 0.5 \
    --epochs 20 --val_episodes 1 \
    --n_episodes_eval 2 --max_call_eval 6
