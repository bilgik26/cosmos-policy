#!/bin/bash
# feature_analysis (DiT特徴量収集) — 50エピソード実行スクリプト。
# GPU 0 is occupied by qwen3-vllm; use GPU 1 via CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

EGL_ICD_DIR="/tmp/singularity_egl_icd"
mkdir -p "$EGL_ICD_DIR"
cat > "$EGL_ICD_DIR/10_nvidia.json" << 'JSONEOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "/.singularity.d/libs/libEGL_nvidia.so.0"
    }
}
JSONEOF
export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_ICD_DIR/10_nvidia.json"

export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/data/bilgehan.sakai/cosmos-policy/cache/huggingface

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

python -m cosmos_policy.experiments.robot.robocasa.analysis.verification.collection.feature_analysis \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True --num_wrist_images 1 \
    --use_proprio True --normalize_proprio True --unnormalize_actions True \
    --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 32 --num_open_loop_steps 16 \
    --task_name PnPCounterToCab \
    --seed 195 --randomize_seed False --deterministic True \
    --use_variance_scale False --use_jpeg_compression True --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 --num_denoising_steps_value 1 \
    --data_collection False \
    --num_analysis_episodes 50 \
    --output_dir cosmos_policy/experiments/robot/robocasa/analysis/results/action_features
