#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

export HF_HOME=/mnt/data/bilgehan.sakai/cosmos-policy/cache/huggingface
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

python3 -m cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.graded_dv_rerun \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --collect_dir cosmos_policy/experiments/robot/robocasa/analysis/results/attractor_verification/collect \
    --embedding_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification \
    --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification/smoketest \
    --task_name PnPCounterToCab \
    --seed 195 \
    --n_episodes 1 \
    --alpha 40.0
