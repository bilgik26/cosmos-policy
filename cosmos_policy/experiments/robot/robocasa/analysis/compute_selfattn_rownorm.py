"""
§3.I post-processing: row-normalize T matrices and compute modality effect sizes.

Reads: results/self_attention_v2/attn_stats.json
Writes:
  results/self_attention_v2/selfattn_Tmatrix_rownorm.png
  results/self_attention_v2/selfattn_modality_effectsize.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import (
    STATE_T, T_NAMES, INPUT_T_IDXS, OUTPUT_T_IDXS, PATCHES_PER_T,
    PROBE_BLOCKS, NUM_DENOISE_STEPS,
)
from cosmos_policy.experiments.robot.robocasa.analysis.attention_analysis import (
    SIGMA_SCHEDULE,
)

UNIFORM = 1.0 / STATE_T  # 1/11 ≈ 0.0909 per token-type after row normalization


def load_t_matrices(stats_path: Path):
    """Load raw T matrices from attn_stats.json, return as dict (blk, k) -> np.ndarray [11,11]."""
    with open(stats_path) as f:
        stats = json.load(f)
    t_mats = {}
    for key, mat in stats["t_matrices"].items():
        # key format: "block{blk}_step{k}"
        parts = key.split("_")
        blk = int(parts[0].replace("block", ""))
        k = int(parts[1].replace("step", ""))
        t_mats[(blk, k)] = np.array(mat, dtype=np.float32)
    return t_mats, stats


def row_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-normalize so each row sums to 1 (fraction of attention per input token type)."""
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return mat / row_sums


def compute_effect_sizes(t_mats_rownorm):
    """Compute excess over uniform for each (blk, k, t_out, t_in).

    Effect size = (observed - 1/STATE_T) / (1/STATE_T) — fractional excess.
    Positive = attends more than uniform; negative = attends less.
    """
    results = []
    for (blk, k), mat in sorted(t_mats_rownorm.items()):
        for t_out in OUTPUT_T_IDXS:
            for t_in in range(STATE_T):
                obs = float(mat[t_out, t_in])
                effect = (obs - UNIFORM) / UNIFORM
                results.append({
                    "block": blk,
                    "k": k,
                    "sigma": float(SIGMA_SCHEDULE[k]) if k < len(SIGMA_SCHEDULE) else None,
                    "t_out": T_NAMES[t_out],
                    "t_in": T_NAMES[t_in],
                    "observed": obs,
                    "uniform_null": UNIFORM,
                    "effect_fractional": effect,
                })
    return results


def plot_rownorm_matrices(t_mats_rownorm, output_path: Path):
    """Plot row-normalized T matrices for each probe block (subplots = k steps)."""
    blocks = sorted(set(b for b, _ in t_mats_rownorm))
    steps = sorted(set(k for _, k in t_mats_rownorm))
    n_steps = len(steps)

    fig, axes = plt.subplots(len(blocks), n_steps, figsize=(4 * n_steps, 3.5 * len(blocks)))
    if len(blocks) == 1:
        axes = axes[np.newaxis, :]
    if n_steps == 1:
        axes = axes[:, np.newaxis]

    tname_list = [T_NAMES[i] for i in range(STATE_T)]
    short_names = [n.replace("curr_", "").replace("future_", "f_") for n in tname_list]

    for bi, blk in enumerate(blocks):
        for ki, k in enumerate(steps):
            ax = axes[bi, ki]
            mat = t_mats_rownorm.get((blk, k), np.full((STATE_T, STATE_T), UNIFORM))
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                           vmin=0.0, vmax=min(1.0, mat.max() * 1.2))
            sigma_str = f"{SIGMA_SCHEDULE[k]:.1f}" if k < len(SIGMA_SCHEDULE) else "?"
            ax.set_title(f"Blk{blk} σ={sigma_str}(k={k})", fontsize=8)
            ax.set_xticks(range(STATE_T))
            ax.set_xticklabels(short_names, rotation=60, ha="right", fontsize=6)
            ax.set_yticks(range(STATE_T))
            ax.set_yticklabels(short_names, fontsize=6)
            # Draw uniform contour line at 1/STATE_T
            ax.axhline(-0.5, color="gray", lw=0.3)
            plt.colorbar(im, ax=ax, shrink=0.7, label="frac")
            if ki == 0:
                ax.set_ylabel(f"Block {blk}", fontsize=8)
            if bi == 0:
                ax.set_xlabel("Key (input T)", fontsize=7)

    fig.suptitle(f"Row-Normalized T Matrix (row sum=1); uniform={UNIFORM:.3f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_action_attention_profile(t_mats_rownorm, output_path: Path):
    """Plot action token attention to each input type across blocks and k steps."""
    blocks = sorted(set(b for b, _ in t_mats_rownorm))
    steps = sorted(set(k for _, k in t_mats_rownorm))

    ACTION_T = 5  # ACTION_T_IDX
    input_names = [T_NAMES[i] for i in INPUT_T_IDXS]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 4), sharey=True)
    if len(steps) == 1:
        axes = [axes]

    for ki, k in enumerate(steps):
        ax = axes[ki]
        for ci, t_in in enumerate(INPUT_T_IDXS):
            vals = [t_mats_rownorm.get((blk, k), np.full((STATE_T, STATE_T), UNIFORM))[ACTION_T, t_in]
                    for blk in blocks]
            ax.plot(blocks, vals, "o-", color=colors[ci], label=input_names[ci], lw=1.5, ms=4)
        ax.axhline(UNIFORM, color="k", ls="--", lw=0.8, label=f"uniform={UNIFORM:.3f}")
        sigma_str = f"{SIGMA_SCHEDULE[k]:.1f}" if k < len(SIGMA_SCHEDULE) else "?"
        ax.set_title(f"k={k} (σ={sigma_str})", fontsize=9)
        ax.set_xlabel("Block", fontsize=8)
        ax.set_xticks(blocks)
        if ki == 0:
            ax.set_ylabel("Fraction of attention (row-norm)", fontsize=8)
            ax.legend(fontsize=7)

    fig.suptitle("Action token → input type attention (row-normalized)", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str,
                        default="cosmos_policy/experiments/robot/robocasa/analysis/results/self_attention_v2")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    stats_path = input_dir / "attn_stats.json"

    if not stats_path.exists():
        print(f"ERROR: {stats_path} not found. Run attention_analysis_v2.py first.")
        return

    print(f"Loading {stats_path}...")
    t_mats, stats = load_t_matrices(stats_path)

    t_mats_rownorm = {k: row_normalize(v) for k, v in t_mats.items()}

    # 1. Plot row-normalized T matrices
    plot_rownorm_matrices(t_mats_rownorm, input_dir / "selfattn_Tmatrix_rownorm.png")

    # 2. Plot action attention profile (action token only)
    plot_action_attention_profile(t_mats_rownorm, input_dir / "selfattn_action_profile.png")

    # 3. Compute effect sizes
    effect_sizes = compute_effect_sizes(t_mats_rownorm)

    # Summarize: action token, k=4 (final step), averaged over mid blocks
    mid_blocks = [b for b in PROBE_BLOCKS if 9 <= b <= 22]
    final_k = max(k for _, k in t_mats_rownorm)
    action_summary = {}
    for t_in in range(STATE_T):
        vals = [t_mats_rownorm.get((b, final_k), np.full((STATE_T, STATE_T), UNIFORM))[5, t_in]
                for b in mid_blocks if (b, final_k) in t_mats_rownorm]
        if vals:
            mean_obs = float(np.mean(vals))
            action_summary[T_NAMES[t_in]] = {
                "mean_fraction": mean_obs,
                "uniform_null": UNIFORM,
                "effect_fractional": (mean_obs - UNIFORM) / UNIFORM,
                "ratio_vs_uniform": mean_obs / UNIFORM,
            }

    # Per-block summary for k=4
    per_block = {}
    for blk in sorted(set(b for b, _ in t_mats_rownorm)):
        mat = t_mats_rownorm.get((blk, final_k))
        if mat is None:
            continue
        row = {}
        for t_out in OUTPUT_T_IDXS:
            row[T_NAMES[t_out]] = {
                T_NAMES[t_in]: {
                    "fraction": float(mat[t_out, t_in]),
                    "effect_vs_uniform": float((mat[t_out, t_in] - UNIFORM) / UNIFORM),
                }
                for t_in in range(STATE_T)
            }
        per_block[str(blk)] = row

    output = {
        "method": "row_normalized_T_matrix",
        "uniform_null": UNIFORM,
        "state_t": STATE_T,
        "final_k": final_k,
        "mid_blocks": mid_blocks,
        "action_token_summary_mid_blocks_k4": action_summary,
        "per_block_k4": per_block,
        "all_effect_sizes": effect_sizes,
    }

    out_path = input_dir / "selfattn_modality_effectsize.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {out_path}")

    # Print summary table
    print(f"\n=== Action token → input type (mid blocks {mid_blocks}, k={final_k}) ===")
    print(f"{'input_type':<20} {'fraction':>10} {'vs uniform':>12} {'ratio':>8}")
    print("-" * 54)
    for t_in_name, v in sorted(action_summary.items(), key=lambda x: -x[1]["mean_fraction"]):
        print(f"{t_in_name:<20} {v['mean_fraction']:>10.4f} {v['effect_fractional']:>+12.3f}  {v['ratio_vs_uniform']:>7.2f}x")
    print(f"{'[uniform null]':<20} {UNIFORM:>10.4f} {'0.000':>12}  {'1.00x':>8}")


if __name__ == "__main__":
    main()
