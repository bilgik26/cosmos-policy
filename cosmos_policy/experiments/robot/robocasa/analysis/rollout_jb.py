"""
§3.J 注意 Rollout — J-b 代替手法

設計書 §3.J の判断:
  J-a (全28層 rollout) は不採用 — 全28ブロックにフックが必要で
  現在のデータには7ブロック分しかない。
  7ブロックだけの rollout は「部分層での累積フロー主張」を禁止する
  設計書の規定に違反する。

J-b 代替: ブロック貢献プロファイル
  各 (隣接) ブロック間の特徴変化量 ‖Δfeat‖/‖feat‖ を層別に計算。
  これは attn + MLP + norm の合計変化量であり、
  attention rollout の代替的な「どの層で最大変換が起きるか」指標になる。

加えて「一様ヌル B_J」:
  A=I (attention=0) rollout の場合、各ブロック出力 = 入力 (残差経由のみ)
  → 特徴変化は 0 になるはず。
  この「ゼロ変化ヌル」との差として、実測変化量の有意性を表す。

入力: results/action_features/features.npz
出力: results/rollout_jb/
  - block_contribution.json
  - block_contribution.png
"""

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")

PROBE_LAYERS = [0, 4, 9, 13, 18, 22, 27]
SIGMA_SCHEDULE = [80.0, 42.29, 20.97, 9.62, 4.0]
N_BOOTSTRAP = 2000
RNG_SEED = 195


def load_features(npz_path: Path) -> dict:
    """Load features and return dict: key → (N, 2048) arrays."""
    d = np.load(npz_path)
    feats = {}
    for k in range(5):
        for l in PROBE_LAYERS:
            feats[(k, l)] = d[f"feat_k{k}_layer{l}"]  # (N, 2048)
    ep_labels = d["episode_labels"]  # (N,)
    return feats, ep_labels


def compute_relative_change(
    feats: dict, ep_labels: np.ndarray, rng: np.random.Generator
) -> dict:
    """
    For each k, compute ‖feat_l2 - feat_l1‖_F / ‖feat_l1‖_F
    across adjacent probe layers.
    Also compute within-k, same-layer norm to use as baseline.
    """
    results = {}
    pairs = list(zip(PROBE_LAYERS[:-1], PROBE_LAYERS[1:]))  # (0,4), (4,9), ...

    for k in range(5):
        results[k] = {}
        for l1, l2 in pairs:
            f1 = feats[(k, l1)]  # (N, 2048)
            f2 = feats[(k, l2)]  # (N, 2048)

            delta = f2 - f1
            delta_norm = np.linalg.norm(delta, axis=1)  # (N,)
            f1_norm = np.linalg.norm(f1, axis=1)         # (N,)
            rel_change = delta_norm / (f1_norm + 1e-8)

            # Bootstrap CI (episode-level)
            eps = np.unique(ep_labels)
            ep_means = np.array([rel_change[ep_labels == ep].mean() for ep in eps])
            boot_means = np.array([
                rng.choice(ep_means, len(eps), replace=True).mean()
                for _ in range(N_BOOTSTRAP)
            ])
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))

            results[k][f"{l1}→{l2}"] = {
                "layer_pair": [l1, l2],
                "mean_rel_change": float(rel_change.mean()),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "mean_delta_norm": float(delta_norm.mean()),
                "mean_f1_norm": float(f1_norm.mean()),
            }

    return results


def compute_within_layer_k_change(feats: dict, ep_labels: np.ndarray,
                                   rng: np.random.Generator) -> dict:
    """
    For each layer, ‖feat_k1 - feat_k0‖ / ‖feat_k0‖ (step-to-step change).
    This quantifies how much each block's representation evolves across denoising steps.
    """
    results = {}
    step_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]

    for l in PROBE_LAYERS:
        results[l] = {}
        for k1, k2 in step_pairs:
            f1 = feats[(k1, l)]
            f2 = feats[(k2, l)]
            delta_norm = np.linalg.norm(f2 - f1, axis=1)
            f1_norm = np.linalg.norm(f1, axis=1)
            rel_change = delta_norm / (f1_norm + 1e-8)

            eps = np.unique(ep_labels)
            ep_means = np.array([rel_change[ep_labels == ep].mean() for ep in eps])
            boot_means = np.array([
                rng.choice(ep_means, len(eps), replace=True).mean()
                for _ in range(N_BOOTSTRAP)
            ])
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))

            results[l][f"k{k1}→k{k2}"] = {
                "step_pair": [k1, k2],
                "mean_rel_change": float(rel_change.mean()),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }

    return results


def plot_block_contributions(layer_changes: dict, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    pairs = list(zip(PROBE_LAYERS[:-1], PROBE_LAYERS[1:]))
    x_labels = [f"{l1}→{l2}" for l1, l2 in pairs]
    x = range(len(x_labels))

    for ki, k in enumerate(range(5)):
        ax = axes[ki]
        means = [layer_changes[k][lbl]["mean_rel_change"] for lbl in x_labels]
        ci_los = [layer_changes[k][lbl]["ci_lo"] for lbl in x_labels]
        ci_his = [layer_changes[k][lbl]["ci_hi"] for lbl in x_labels]

        ax.errorbar(
            x, means,
            yerr=[np.array(means) - np.array(ci_los),
                  np.array(ci_his) - np.array(means)],
            marker='o', capsize=4, color='steelblue',
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(x_labels, rotation=45, fontsize=8)
        ax.set_xlabel("Layer pair")
        ax.set_ylabel("‖Δfeat‖/‖feat‖" if ki == 0 else "")
        ax.set_title(f"k={k} (σ={SIGMA_SCHEDULE[k]:.1f})")
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle("§3.J J-b: Block Contribution Profile\n"
                 "‖feat(l+1) − feat(l)‖ / ‖feat(l)‖ per adjacent probe layer pair\n"
                 "(attn + MLP + norm; null: ‖Δ‖=0 when A=I)", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "block_contribution.png", dpi=150)
    plt.close()

    # Also plot k-step change by layer
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    step_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for (k1, k2), color in zip(step_pairs, colors):
        means = [None]  # placeholder
        # from within_layer_changes
        pass  # will be called separately
    plt.close()


def main():
    rng = np.random.default_rng(RNG_SEED)

    npz_path = Path(
        "cosmos_policy/experiments/robot/robocasa/analysis/results/"
        "action_features/features.npz"
    )
    output_dir = Path(
        "cosmos_policy/experiments/robot/robocasa/analysis/results/rollout_jb"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== §3.J J-b: Block Contribution Profile ===")
    feats, ep_labels = load_features(npz_path)
    print(f"  N={len(ep_labels)}, n_episodes={len(np.unique(ep_labels))}")

    print("  Computing layer-to-layer relative change...")
    layer_changes = compute_relative_change(feats, ep_labels, rng)

    print("  Computing step-to-step relative change per layer...")
    step_changes = compute_within_layer_k_change(feats, ep_labels, rng)

    # Summary
    print("\n  Layer-pair contribution (mean ‖Δ‖/‖f‖, k=4):")
    pairs = list(zip(PROBE_LAYERS[:-1], PROBE_LAYERS[1:]))
    for l1, l2 in pairs:
        lbl = f"{l1}→{l2}"
        v = layer_changes[4][lbl]
        print(f"    {lbl:8s}: {v['mean_rel_change']:.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")

    print("\n  Step-change at k=3→4 per layer (largest denoising change):")
    for l in PROBE_LAYERS:
        v = step_changes[l]["k3→k4"]
        print(f"    Layer {l:2d}: {v['mean_rel_change']:.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")

    # J-b null model note
    null_note = (
        "J-b null (A=I rollout): if attention were zero (residual only), "
        "block outputs = block inputs → ‖Δ‖/‖f‖ = 0. "
        "Observed ‖Δ‖/‖f‖ > 0 indicates the transformer blocks DO transform representations. "
        "This is an always-true fact for a trained network, so the interesting comparison "
        "is the LAYER-SPECIFIC profile: which layers contribute most."
    )

    result = {
        "section": "3J_jb_block_contribution",
        "method": (
            "J-b: Attention Rollout replaced by block contribution profile. "
            "J-a (28-layer rollout) not feasible: only 7 probe blocks available in features.npz. "
            "Partial-layer rollout forbidden by design doc §3.J."
        ),
        "null_model": "A=I rollout → ‖Δfeat‖/‖feat‖ = 0 (residual-only, no transformation)",
        "null_note": null_note,
        "layer_pair_relative_change": layer_changes,
        "step_relative_change_per_layer": step_changes,
    }

    with open(output_dir / "block_contribution.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {output_dir}/block_contribution.json")

    plot_block_contributions(layer_changes, output_dir)
    print(f"Saved: {output_dir}/block_contribution.png")

    print("\n§3.J decision: J-b adopted (rollout dropped)")
    print(f"  {null_note}")


if __name__ == "__main__":
    main()
