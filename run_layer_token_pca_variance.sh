#!/bin/bash
# 層ごとの潜在表現(全トークン)PCA分散説明率解析
# layer_token_pca_variance.py: feature_analysis.pyのaction-token限定・空間平均済み
# 特徴とは異なり、プーリングを一切行わず全トークン(T×H×W)をPCAサンプルとして扱う。
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

export HF_HOME=/mnt/data/bilgehan.sakai/cosmos-policy/cache/huggingface
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

python -m cosmos_policy.experiments.robot.robocasa.analysis.layer_token_pca_variance \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True --num_wrist_images 1 \
    --use_proprio True --normalize_proprio True --unnormalize_actions True \
    --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 32 --num_open_loop_steps 16 \
    --seed 195 --randomize_seed False --deterministic True \
    --use_variance_scale False --use_jpeg_compression True --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 --num_denoising_steps_value 1 \
    --data_collection False \
    --tasks "PnPCounterToCab,CloseDrawer,CoffeePressButton,TurnOnStove" \
    --n_episodes_per_task 15 \
    --tokens_per_call 128 \
    --output_dir cosmos_policy/experiments/robot/robocasa/analysis/results/layer_token_pca
