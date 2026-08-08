#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/mnt/data/bilgehan.sakai/cosmos-policy/cache/huggingface
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"

cd /home/bilgehan.sakai/cosmos-policy
source .venv/bin/activate

python3 -m cosmos_policy.experiments.robot.robocasa.analysis.precompute_text_directions \
    --out_path cosmos_policy/experiments/robot/robocasa/analysis/results/attractor_verification/text_directions.pt
