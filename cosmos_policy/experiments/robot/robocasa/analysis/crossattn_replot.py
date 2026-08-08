"""
crossattn_replot.py — 既存 crossattn.npz から全プロット（H_real 含む）を再生成する。

シミュレーション再実行なしで crossattn_analysis.py のプロットを更新する場合に使用する。

実行:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.crossattn_replot \
      --npz_path cosmos_policy/experiments/robot/robocasa/analysis/results/action_crossattn/crossattn.npz \
      --meta_path cosmos_policy/experiments/robot/robocasa/analysis/results/action_crossattn/crossattn_meta.json \
      --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/action_crossattn
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cosmos_policy.experiments.robot.robocasa.analysis.crossattn_analysis import (
    NUM_DENOISE_STEPS,
    plot_all,
)
from cosmos_policy.experiments.robot.robot_utils import log_message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_message(f"Loading {args.npz_path}")
    data = np.load(args.npz_path, allow_pickle=True)

    log_message(f"Loading {args.meta_path}")
    with open(args.meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    probe_layers = meta["probe_layers"]
    Sk = int(data["Sk"][0])
    ep_arr = data["episode_labels"]
    ci_arr = data["call_idx_labels"]
    n_real = meta["token_info"]["n_real"]

    # per_call[layer][k]: (N, Sk) ndarray
    per_call = {l: {} for l in probe_layers}
    for l in probe_layers:
        for k in range(NUM_DENOISE_STEPS):
            per_call[l][k] = data[f"attn_layer{l}_k{k}"]

    # token_info の再構成
    real_tokens = meta["token_info"]["tokens"]
    pad_count = Sk - len(real_tokens)
    all_tokens = real_tokens + ["<PAD>"] * max(0, pad_count)
    attn_mask = data["token_attention_mask"]

    token_info = {
        "tokens": all_tokens,
        "attention_mask": attn_mask,
        "n_real": n_real,
    }

    primary_desc = meta.get("primary_desc", meta.get("task_descriptions", [""])[0])
    log_message(f"Task: {primary_desc}")
    log_message(f"Probe layers: {probe_layers}, n_real={n_real}, Sk={Sk}")
    log_message(f"N calls: {len(ep_arr)}")

    log_message("\nGenerating plots...")
    for k in range(NUM_DENOISE_STEPS):
        log_message(f"  k={k}...")
        plot_all(
            per_call=per_call,
            ep_arr=ep_arr,
            ci_arr=ci_arr,
            token_info=token_info,
            task_desc=primary_desc,
            probe_layers=probe_layers,
            out_dir=out_dir,
            Sk=Sk,
            k_focus=k,
        )

    log_message(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
