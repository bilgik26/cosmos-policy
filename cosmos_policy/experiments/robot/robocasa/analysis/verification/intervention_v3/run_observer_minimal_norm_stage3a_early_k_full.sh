#!/bin/bash
# report_v3.md §6.2 follow-up E-3: rerun Stage 3-A's setpoint intervention at earlier
# denoising steps (k=0,1,2) instead of the auto-selected best-CV-acc k=4, to test the
# separability-vs-causal-leverage tradeoff noted in §6.2's interpretation.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

for K in 0 1 2; do
    python3 -m cosmos_policy.experiments.robot.robocasa.analysis.verification.intervention_v3.observer_minimal_norm_stage3a \
        --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
        --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
        --config_file cosmos_policy/config/config.py \
        --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
        --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/intervention_v3_observer_minimal_norm_k${K} \
        --task_name PnPCounterToCab --seed 195 --target_layer 13 --target_k ${K} \
        --n_episodes_eval 8 --max_call_eval 40 \
        --conditions off force_open force_closed
done
