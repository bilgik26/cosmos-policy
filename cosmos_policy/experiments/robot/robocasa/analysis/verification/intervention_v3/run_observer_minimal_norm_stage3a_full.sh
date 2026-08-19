#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES}"

python3 -m cosmos_policy.experiments.robot.robocasa.analysis.verification.intervention_v3.observer_minimal_norm_stage3a \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/intervention_v3_observer_minimal_norm \
    --task_name PnPCounterToCab --seed 195 --target_layer 13 \
    --n_episodes_eval 8 --max_call_eval 40 --conditions off force_open force_closed
