#!/bin/bash
# Phase 6 (ldv_design_v2.md フェーズ6): prompt fadeout / entrainment verification, PnPCounterToCab.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

python3 -m cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.prompt_fadeout_entrainment \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --embedding_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification \
    --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification \
    --task_name PnPCounterToCab \
    --seed 195 \
    --n_episodes 8 \
    --alpha 40.0 \
    --fadeout_calls 5
