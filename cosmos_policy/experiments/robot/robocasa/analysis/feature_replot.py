"""
feature_replot.py — 既存 features.npz からプロット・統計を再生成するオフラインスクリプト。

torch・cosmos_policy モデルコード・robosuite を一切インポートしないため
Singularity コンテナ外（ホスト venv）から OOM なしで実行できる。

Usage:
    python -m cosmos_policy.experiments.robot.robocasa.analysis.feature_replot \
        --npz_path results/action_features/features.npz \
        --out_dir  results/action_features \
        --task_name PnPCounterToCab \
        --success_rate 0.60 \
        --success_count 30 \
        --total_episodes 50
"""
import argparse
import json
from pathlib import Path

from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import (
    PROBE_LAYERS,
    NUM_DENOISE_STEPS,
)
from cosmos_policy.experiments.robot.robocasa.analysis.feature_plot import (
    load_features,
    plot_pca_per_layer,
    plot_tsne_per_layer,
    plot_feature_change_by_layer,
    plot_step_change_per_layer,
    plot_cka_matrix,
    plot_feature_variance_by_layer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--task_name", default="PnPCounterToCab")
    parser.add_argument("--success_rate", type=float, default=0.60)
    parser.add_argument("--success_count", type=int, default=30)
    parser.add_argument("--total_episodes", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {args.npz_path}")
    feats, episode_labels, call_idx_labels = load_features(
        args.npz_path, PROBE_LAYERS, NUM_DENOISE_STEPS
    )
    total_calls = len(episode_labels)
    print(f"Loaded {total_calls} call records from {args.total_episodes} episodes")

    stats = {
        "task": args.task_name,
        "success_rate": args.success_rate,
        "success_count": args.success_count,
        "total_episodes": args.total_episodes,
        "total_policy_calls": total_calls,
        "probe_layers": PROBE_LAYERS,
    }

    print("\n--- PCA plots (k=0..4) ---")
    for k in range(NUM_DENOISE_STEPS):
        print(f"  PCA k={k}...")
        plot_pca_per_layer(feats, episode_labels, call_idx_labels,
                           PROBE_LAYERS, out_dir, args.task_name, args.success_rate,
                           denoise_step=k)

    print("\n--- t-SNE plots (k=0..4) ---")
    for k in range(NUM_DENOISE_STEPS):
        print(f"  t-SNE k={k}...")
        plot_tsne_per_layer(feats, episode_labels, call_idx_labels,
                            PROBE_LAYERS, out_dir, args.task_name, args.success_rate,
                            denoise_step=k)

    print("\n--- Feature variance ---")
    var_stats = plot_feature_variance_by_layer(
        feats, PROBE_LAYERS, out_dir, args.task_name, args.success_rate
    )
    stats["feature_variance"] = var_stats

    print("\n--- Feature change (k=0→k=4) by layer ---")
    change_stats = plot_feature_change_by_layer(
        feats, PROBE_LAYERS, out_dir, args.task_name, args.success_rate
    )
    stats.update(change_stats)

    print("\n--- Step change per layer ---")
    plot_step_change_per_layer(
        feats, PROBE_LAYERS, out_dir, args.task_name, args.success_rate
    )

    print("\n--- CKA matrices (k=0..4) ---")
    for k in range(NUM_DENOISE_STEPS):
        print(f"  CKA k={k}...")
        cka_s = plot_cka_matrix(
            feats, PROBE_LAYERS, out_dir, args.task_name, args.success_rate,
            denoise_step=k,
        )
        if k == NUM_DENOISE_STEPS - 1:
            stats.update(cka_s)

    stats_path = out_dir / "feature_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {stats_path}")
    print("feature_replot complete!")


if __name__ == "__main__":
    main()
