"""
attention_replot.py — 既存 t_matrices.npy から Attention Rollout を計算してプロットを再生成する。

シミュレーション再実行なしで attention_analysis.py の Rollout 解析を追加する場合に使用する。

実行:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.attention_replot \
      --npy_path cosmos_policy/experiments/robot/robocasa/analysis/results/self_attention/t_matrices.npy \
      --stats_path cosmos_policy/experiments/robot/robocasa/analysis/results/self_attention/attn_stats.json \
      --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/self_attention
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from cosmos_policy.experiments.robot.robocasa.analysis.attention_analysis import (
    compute_attention_rollout,
    plot_rollout_matrices,
    STATE_T,
    T_NAMES,
    INPUT_T_IDXS,
    OUTPUT_T_IDXS,
    SIGMA_SCHEDULE,
)
from cosmos_policy.experiments.robot.robot_utils import log_message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", required=True)
    parser.add_argument("--stats_path", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_message(f"Loading {args.npy_path}")
    raw = np.load(args.npy_path, allow_pickle=True).item()

    # キーが文字列 "(block, k_step)" → タプルに変換
    t_matrices = {}
    for key_str, mat in raw.items():
        key = ast.literal_eval(key_str)  # "(0, 0)" → (0, 0)
        t_matrices[key] = mat

    log_message(f"Loaded {len(t_matrices)} (block, step) T-matrices")
    block_list = sorted(set(k[0] for k in t_matrices))
    step_list = sorted(set(k[1] for k in t_matrices))
    log_message(f"Blocks: {block_list}, Steps: {step_list}")

    # Attention Rollout を計算
    log_message("\nComputing Attention Rollout...")
    rollout = compute_attention_rollout(t_matrices)
    log_message(f"Computed rollout for {len(rollout)} (block, step) pairs")

    # Rollout プロット生成
    log_message("\nGenerating rollout plots...")
    plot_rollout_matrices(rollout, out_dir)

    # attn_stats.json に rollout_matrices と rollout_summary を追加
    log_message(f"\nUpdating {args.stats_path}")
    with open(args.stats_path, encoding="utf-8") as f:
        stats = json.load(f)

    stats["rollout_matrices"] = {
        f"block{blk}_step{k}": mat.tolist()
        for (blk, k), mat in rollout.items()
    }

    last_blk = max(block_list)
    final_step = max(step_list)
    r_mat = rollout.get((last_blk, final_step))
    if r_mat is not None:
        stats["rollout_summary"] = {}
        for t_out in range(STATE_T):
            row = {T_NAMES[t_in]: float(r_mat[t_out, t_in]) for t_in in range(STATE_T)}
            stats["rollout_summary"][T_NAMES[t_out]] = row

    with open(args.stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    log_message(f"Updated: {args.stats_path}")

    # コンソールに Rollout サマリー出力
    if r_mat is not None:
        log_message(f"\n=== Rollout Summary (Block-{last_blk}, k={final_step}) ===")
        header = f"{'output':>16}" + "".join(f" {T_NAMES[t]:>14}" for t in INPUT_T_IDXS)
        log_message(header)
        log_message("-" * (16 + 15 * len(INPUT_T_IDXS)))
        for t_out in OUTPUT_T_IDXS:
            row = stats["rollout_summary"].get(T_NAMES[t_out], {})
            vals = "".join(f" {row.get(T_NAMES[t], 0):>14.6f}" for t in INPUT_T_IDXS)
            log_message(f"{T_NAMES[t_out]:>16}{vals}")

    log_message(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
