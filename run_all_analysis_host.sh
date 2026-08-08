#!/bin/bash
# Host-level master script for running all analysis scripts in sequence.
# Runs all 5 simulation-based scripts, then 3 offline scripts.
# Total estimated time: 6-12 hours (50 episodes each x 5 scripts).

set -e  # Exit on error

SIF_PATH="/mnt/data/bilgehan.sakai/singularity/sif/cosmos-policy.sif"
PROJECT_ROOT="/home/bilgehan.sakai/cosmos-policy"
HOST_CACHE_ROOT="/mnt/data/bilgehan.sakai/cosmos-policy/cache"
GLOBAL_PYTHONUSERBASE="/mnt/data/bilgehan.sakai/cache/python_local"
TEXTURE_BIND="/mnt/data/bilgehan.sakai/tmp_home:/mnt/data/bilgehan.sakai/tmp_home"

LOG_DIR="$PROJECT_ROOT/cosmos_policy/experiments/robot/robocasa/analysis/results/logs"
mkdir -p "$LOG_DIR"

run_in_singularity() {
    local script_name="$1"
    local script_path="$PROJECT_ROOT/$script_name"
    local log_file="$LOG_DIR/${script_name%.sh}.log"

    echo ""
    echo "============================================================"
    echo "  Running: $script_name"
    echo "  Log: $log_file"
    echo "  Started: $(date)"
    echo "============================================================"

    singularity exec --nv \
        --bind "${HOST_CACHE_ROOT}:${HOST_CACHE_ROOT}" \
        --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
        --bind "/home/bilgehan.sakai:/home/bilgehan.sakai" \
        --bind "${GLOBAL_PYTHONUSERBASE}:${GLOBAL_PYTHONUSERBASE}" \
        --bind "${TEXTURE_BIND}" \
        --env "PYTHONUSERBASE=${GLOBAL_PYTHONUSERBASE}" \
        --env "HF_HOME=${HOST_CACHE_ROOT}/huggingface" \
        "$SIF_PATH" \
        bash "$script_path" 2>&1 | tee "$log_file"

    echo ""
    echo "  Finished: $(date)"
}

# ── Phase 1: Simulation-based analysis (needs GPU + RoboCasa) ──────────────
echo "=== Phase 1: Simulation-based analysis (5 scripts x 50 episodes) ==="

run_in_singularity "run_mechanism_analysis.sh"
run_in_singularity "run_feature_analysis.sh"
run_in_singularity "run_crossattn_analysis.sh"
run_in_singularity "run_attention_analysis.sh"
run_in_singularity "run_image_analysis.sh"

# ── Phase 2: Offline analysis (reads npz files, no simulation) ────────────
echo "=== Phase 2: Offline analysis ==="

run_in_singularity "run_offline_analysis.sh"

echo ""
echo "============================================================"
echo "  ALL ANALYSIS COMPLETE: $(date)"
echo "  Results: $PROJECT_ROOT/cosmos_policy/experiments/robot/robocasa/analysis/results/"
echo "============================================================"
