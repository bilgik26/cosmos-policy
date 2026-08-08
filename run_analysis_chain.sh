#!/bin/bash
# Chain: wait for attention_analysis_v2 → selfattn rownorm → §3.C → §5
# Run from project root. Logs everything to results/logs/.

set -e

ATTN_PID=3497853
LOG_DIR="cosmos_policy/experiments/robot/robocasa/analysis/results/logs"
ATTN_META="cosmos_policy/experiments/robot/robocasa/analysis/results/self_attention_v2/attn_meta_v2.json"

mkdir -p "$LOG_DIR"

echo "[chain] Waiting for attention_analysis_v2 (PID $ATTN_PID) to complete..."
while kill -0 "$ATTN_PID" 2>/dev/null; do
    sleep 30
done
echo "[chain] PID $ATTN_PID finished at $(date)"

if [ ! -f "$ATTN_META" ]; then
    echo "[chain] ERROR: $ATTN_META not found — attention run may have crashed"
    exit 1
fi

# Step 1: Row-normalize T matrices and compute effect sizes (CPU only, ~1 min)
echo "[chain] Step 1: compute_selfattn_rownorm..."
source .venv/bin/activate
python -m cosmos_policy.experiments.robot.robocasa.analysis.compute_selfattn_rownorm \
    --input_dir cosmos_policy/experiments/robot/robocasa/analysis/results/self_attention_v2 \
    2>&1 | tee "$LOG_DIR/compute_selfattn_rownorm.log"
echo "[chain] selfattn rownorm done at $(date)"

# EGL setup for simulator scripts
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
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/data/bilgehan.sakai/cosmos-policy/cache/huggingface
export CUDA_VISIBLE_DEVICES=1

# Step 2: §3.C F_θ preconditioning analysis (15 episodes — only needs good statistics on norms)
echo "[chain] Step 2: §3.C precond_ftheta_analysis (15 episodes)..."
python -m cosmos_policy.experiments.robot.robocasa.analysis.precond_ftheta_analysis \
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
    --num_episodes 15 \
    --output_dir cosmos_policy/experiments/robot/robocasa/analysis/results/precond_ftheta \
    2>&1 | tee "$LOG_DIR/run_precond_ftheta.log"
echo "[chain] §3.C done at $(date)"

# Step 3: §5 null model analysis (10 episodes — random model always fails, 10 episodes enough for PR comparison)
echo "[chain] Step 3: §5 null_model_analysis (10 episodes)..."
python -m cosmos_policy.experiments.robot.robocasa.analysis.null_model_analysis \
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
    --num_episodes 10 \
    --output_dir cosmos_policy/experiments/robot/robocasa/analysis/results/null_model_random_init \
    --trained_features_path cosmos_policy/experiments/robot/robocasa/analysis/results/action_features/features.npz \
    --random_init_seed 42 \
    2>&1 | tee "$LOG_DIR/run_null_model.log"
echo "[chain] §5 done at $(date)"

echo "[chain] All steps complete at $(date)"
echo "[chain] Artifacts:"
echo "  - self_attention_v2/selfattn_Tmatrix_rownorm.png"
echo "  - self_attention_v2/selfattn_modality_effectsize.json"
echo "  - precond_ftheta/precond_normalized_by_sqrt_d.json"
echo "  - null_model_random_init/null_vs_trained_pr.json"
echo "  - null_model_random_init/null_vs_trained_pr.png"
echo "  - null_model_random_init/null_vs_trained_cka_k{0,4}.png"
