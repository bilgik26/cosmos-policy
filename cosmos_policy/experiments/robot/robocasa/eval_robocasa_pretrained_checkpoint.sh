#!/bin/bash
# Evaluation command using the pretrained checkpoint from ROBOCASA.md:
#   nvidia/Cosmos-Policy-RoboCasa-Predict2-2B
# (trained on 24 RoboCasa tasks, 50 demos/task, old robocasa-cosmos-policy dataset)
#
# Notes:
# - The checkpoint, dataset stats, and T5 embeddings are auto-downloaded from
#   HuggingFace by `get_model()` / `load_dataset_stats()` / the embeddings cache loader.
# - This uses run_robocasa_eval_new.py (compatible with the currently installed
#   bilgik26/robocasa v1.0.1 package). The old run_robocasa_eval.py cannot be used
#   here because it imports SINGLE_STAGE_TASK_DATASETS/MULTI_STAGE_TASK_DATASETS,
#   which only exist in the old moojink/robocasa-cosmos-policy package.
# - Model architecture (state_t=11, chunk_duration=41, 5 conditioning frames) matches
#   the new_robocasa config, and dataset_statistics.json has the same dims
#   (actions: 7, proprio: 9), so it loads cleanly with this script.
# - Any task description not present in the pretrained T5 embeddings pkl will be
#   computed on-the-fly (slower for the first occurrence of each new description).
#
# MUJOCO_GL=osmesa is required on compute-only GPU driver environments where
# EGL offscreen rendering is unavailable.

export MUJOCO_GL=osmesa
source /workspace/.venv/bin/activate
cd /workspace

CKPT=nvidia/Cosmos-Policy-RoboCasa-Predict2-2B

# Task name mapping (new_robocasa naming -> approximate old 24-task equivalents):
#   PickPlaceCabinetToCounter ~ PnPCabToCounter
#   CoffeeSetupMug            ~ CoffeeSetupMug (same name)
#   StartCoffeeMachine        ~ CoffeePressButton
#
# task_name below uses the new_robocasa (ATOMIC_TASK_DATASETS) naming, which is what
# run_robocasa_eval_new.py validates against and what the installed robocasa env uses.
for TASK in PickPlaceCabinetToCounter CoffeeSetupMug StartCoffeeMachine; do
  echo "=== Evaluating $TASK ==="
  python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval_new \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path $CKPT \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --num_wrist_images 1 \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 32 \
    --num_open_loop_steps 16 \
    --task_name $TASK \
    --obj_instance_split target \
    --num_trials_per_task 5 \
    --run_id_note pretrained_eval30 \
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
