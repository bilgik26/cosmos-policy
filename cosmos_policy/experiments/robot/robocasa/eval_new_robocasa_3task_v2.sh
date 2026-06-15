#!/bin/bash
# Evaluation command for the new_robocasa_3task_v2 checkpoint (iter_000001800),
# 30 episodes per task for PickPlaceCabinetToCounter, CoffeeSetupMug, StartCoffeeMachine.
#
# MUJOCO_GL=osmesa is required on compute-only GPU driver environments where
# EGL offscreen rendering is unavailable.

export MUJOCO_GL=osmesa
source /workspace/.venv/bin/activate
cd /workspace

CKPT=/workspace/outputs/cosmos_policy/cosmos_v2_finetune/new_robocasa_3task_v2/checkpoints/iter_000001800/model_consolidated.pt

for TASK in PickPlaceCabinetToCounter CoffeeSetupMug StartCoffeeMachine; do
  echo "=== Evaluating $TASK ==="
  python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval_new \
    --config cosmos_predict2_2b_480p_new_robocasa_pretrain_human__inference \
    --ckpt_path $CKPT \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --num_wrist_images 1 \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path /workspace/robocasa/datasets/new_robocasa_dataset_statistics.json \
    --t5_text_embeddings_path /workspace/robocasa/datasets/new_robocasa_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 32 \
    --num_open_loop_steps 16 \
    --task_name $TASK \
    --obj_instance_split target \
    --num_trials_per_task 30 \
    --run_id_note eval30 \
    --local_log_dir cosmos_policy/experiments/robot/robocasa/logs/ \
    --seed 195 \
    --randomize_seed False \
    --deterministic True \
    --use_variance_scale False \
    --use_jpeg_compression True \
    --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 \
    --data_collection False
  echo "=== Done $TASK ==="
done
