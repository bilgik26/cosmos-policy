#!/bin/bash
# critical_boundary_test.py の小規模動作確認 (1タスク・1エピソード・1プローブ・低解像度)。
# 本番実行前にクラッシュ有無と1プローブあたりの所要時間を確認するためのスモークテスト。
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

export HF_HOME=/mnt/data/bilgehan.sakai/cosmos-policy/cache/huggingface
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

python -m cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.critical_boundary_test \
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
    --tasks "CoffeePressButton" \
    --seed_series "195" \
    --n_episodes_per_task 1 \
    --n_probes_per_episode 1 \
    --probe_call_stride 2 \
    --fine_steps 9 \
    --n_hutchinson_probes 6 \
    --n_convergence_seeds 4 \
    --output_dir cosmos_policy/experiments/robot/robocasa/analysis/results/attractor_verification/critical_boundary_smoketest
