#!/bin/bash
# Pad マスク・アテンション寄与の直接検証 — レビュー §3.7
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=1

EGL_ICD_DIR="/tmp/singularity_egl_icd"
mkdir -p "$EGL_ICD_DIR"
if [ -f "/.singularity.d/libs/libEGL_nvidia.so.0" ]; then
    EGL_LIB="/.singularity.d/libs/libEGL_nvidia.so.0"
else
    EGL_LIB="/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0"
fi
cat > "$EGL_ICD_DIR/10_nvidia.json" << JSONEOF
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "$EGL_LIB"
    }
}
JSONEOF
export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_ICD_DIR/10_nvidia.json"

export HF_HOME=/mnt/data/bilgehan.sakai/cosmos-policy/cache/huggingface
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

python -m cosmos_policy.experiments.robot.robocasa.analysis.verification.attention.t_pad_attn_verify \
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
    --num_calls "${NUM_CALLS:-15}" \
    --output_dir cosmos_policy/experiments/robot/robocasa/analysis/results/pad_attn_verify
