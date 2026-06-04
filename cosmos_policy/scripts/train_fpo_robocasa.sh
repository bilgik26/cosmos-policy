#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# FPO++ online RL training of Cosmos Policy on RoboCasa
#
# Usage:
#   cosmos_policy/scripts/train_fpo_robocasa.sh [TASK_NAME] [EXTRA_ARGS...]
#
# Examples:
#   # Run with defaults (TurnOffMicrowave, LoRA rank 8)
#   cosmos_policy/scripts/train_fpo_robocasa.sh
#
#   # Specify task
#   cosmos_policy/scripts/train_fpo_robocasa.sh PnPCounterToCab
#
#   # Full DiT fine-tuning
#   cosmos_policy/scripts/train_fpo_robocasa.sh TurnOffMicrowave \
#       --finetune_mode full_dit
#
#   # Dry-run with fewer timesteps and W&B disabled
#   cosmos_policy/scripts/train_fpo_robocasa.sh TurnOffMicrowave \
#       --total_timesteps 10000 --wandb_enable False
#
# W&B API key:
#   Set the WANDB_API_KEY environment variable before running, or run
#   `wandb login` once to store the key in ~/.netrc.
#   See: https://docs.wandb.ai/quickstart
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Resolve repo root ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── Task (first positional arg, default PnPCounterToSink) ────────────────────
TASK="${1:-PnPCounterToSink}"
shift 2>/dev/null || true

# ── Timestamp for log dir ────────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/fpo_${TASK}_${TIMESTAMP}"

echo "============================================================"
echo "  FPO++ Cosmos Policy — RoboCasa"
echo "  task        : ${TASK}"
echo "  log_dir     : ${LOG_DIR}"
echo "  extra args  : $*"
echo "============================================================"

# ── Run ───────────────────────────────────────────────────────────────────────
uv run --extra cu128 --group robocasa --python 3.10 \
    python -m cosmos_policy.experiments.robot.robocasa.train_fpo_robocasa \
        --task_name          "${TASK}"   \
        --num_envs           4           \
        --img_res            224         \
        --obj_instance_split B           \
        \
        --finetune_mode      lora        \
        --lora_rank          8           \
        --lora_alpha         16.0        \
        --lora_dropout       0.0         \
        --lora_targets       "q_proj,k_proj,v_proj,output_proj" \
        \
        --chunk_size         32          \
        --n_open_loop        16          \
        \
        --steps_per_iter     96          \
        --n_cfm_samples      8           \
        \
        --gamma              0.99        \
        --gae_lambda         0.95        \
        \
        --update_epochs      4           \
        --num_mini_batches   4           \
        --clip_coef          0.01        \
        --vf_coef            0.5         \
        --aux_coef           1.0         \
        --max_grad_norm      1.0         \
        --trust_region_mode  ppo         \
        \
        --lr_lora            1e-5        \
        --adam_eps           1e-5        \
        --weight_decay       0.0         \
        \
        --lr_scheduler_name         constant \
        --lr_scheduler_warmup_steps 5        \
        \
        --total_timesteps    1000000     \
        --seed               42          \
        \
        --eval_rollout_freq  10          \
        --eval_num_episodes  10          \
        \
        --wandb_enable       True        \
        --wandb_project      fpo-cosmos-robocasa \
        \
        --log_dir            "${LOG_DIR}" \
        --save_interval      10           \
        --log_interval       1            \
        "$@"
