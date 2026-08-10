#!/bin/bash
# Phase 5 (ldv_design_v2.md フェーズ5): single-affordance task re-verification, CloseDrawer.
# Reuses dynamic_vector_field_steering.py / phase4_evaluation_metrics.py unmodified. Runs on
# GPU1 (CoffeePressButton runs concurrently on GPU0).
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

export HF_HOME=/mnt/data/bilgehan.sakai/cosmos-policy/cache/huggingface
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

set -e

python3 -m cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamic_vector_field_steering \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --collect_dir cosmos_policy/experiments/robot/robocasa/analysis/results/attractor_verification/collect \
    --embedding_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification \
    --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification \
    --task_name CloseDrawer \
    --seed 195 \
    --n_episodes 8 \
    --alpha 40.0

python3 -m cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.phase4_evaluation_metrics \
    --steering_json cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification/dynamic_vector_field_steering_CloseDrawer.json \
    --embedding_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification \
    --collect_dir cosmos_policy/experiments/robot/robocasa/analysis/results/attractor_verification/collect_v2 \
    --task_name CloseDrawer \
    --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification
