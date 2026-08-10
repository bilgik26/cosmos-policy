#!/bin/bash
# Phase 7 (ldv_design_v2.md フェーズ7) data collection: same methodology as
# attractor/collect_multitask.py (4 tasks x 2 seed series x 20 episodes), plus the generated
# action chunk (X_hat_0, 32x7) per call, which collect/collect_v2 never saved.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1

python3 -m cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.collect_action_chunks \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --output_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification/collect_actions \
    --tasks PnPCounterToCab,CloseDrawer,TurnOnStove,CoffeePressButton \
    --seed_series 195,196 \
    --n_episodes_per_task 20 \
    --use_wrist_image true --num_wrist_images 1 --use_proprio true --normalize_proprio true \
    --unnormalize_actions true --trained_with_image_aug true --chunk_size 32 --num_open_loop_steps 16 \
    --randomize_seed false --deterministic true --use_variance_scale false --use_jpeg_compression true \
    --flip_images true --num_denoising_steps_action 5 --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 --data_collection false
