#!/bin/bash
# Offline replot, no GPU/model needed.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

python3 -m cosmos_policy.experiments.robot.robocasa.analysis.verification.intervention_v3.plot_constraint_vs_success \
    --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/intervention_v3_constraint_vs_success
