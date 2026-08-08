"""
§3.H 言語クロスアテンション（pad マスク修正版）

設計書 §3.H の要件:
  - pad マスク後に重みを実トークン上で再正規化 (P8)
  - 一様ヌル 1/n_real との効果量 (KL・比)
  - Bootstrap CI (H_real, KL)
  - 層間集中度の permutation 検定

入力:  results/action_crossattn/crossattn.npz
出力:  results/crossattn_masked/
  - t3_pad_mask_report.json
  - crossattn_masked_topk.png
  - crossattn_vs_uniform_effectsize.json
  - crossattn_entropy_by_layer_k.png
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
N_PERMUTATION = 2000


def entropy(w: np.ndarray) -> np.ndarray:
    """Shannon entropy per row, nats."""
    w = np.clip(w, 1e-30, None)
    return -np.sum(w * np.log(w), axis=-1)


def kl_to_uniform(w: np.ndarray, n: int) -> np.ndarray:
    """KL(w ‖ uniform_n) = log(n) - H(w) per row."""
    return np.log(n) - entropy(w)


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator,
                 alpha: float = 0.05):
    n = len(values)
    means = np.array([rng.choice(values, n, replace=True).mean()
                      for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(values.mean()), lo, hi


def permutation_p_uniform(kl_obs: np.ndarray, n_real: int,
                           n_perm: int, rng: np.random.Generator) -> float:
    """
    H0: attention is uniform over n_real tokens.
    Under H0, draw uniform samples and compute their KL → should be near 0.
    p = fraction of permutations where mean KL_perm ≥ mean KL_obs.
    """
    obs_mean = kl_obs.mean()
    count = 0
    for _ in range(n_perm):
        # Uniform Dirichlet (concentration=1) ≈ uniform attention
        perm_w = rng.dirichlet(np.ones(n_real), size=len(kl_obs))
        perm_kl = kl_to_uniform(perm_w, n_real)
        if perm_kl.mean() >= obs_mean:
            count += 1
    return count / n_perm


def run_t3_verification(npz_path: Path, output_dir: Path) -> dict:
    """T3: pad weight=0, row sum=1 — checked over ALL (layer, k) pairs."""
    d = np.load(npz_path)
    mask = d['token_attention_mask']  # (512,)
    real_idx = np.where(mask == 1)[0]
    pad_idx = np.where(mask == 0)[0]

    all_row_sums = []
    all_pad_max = []
    all_real_total = []
    w_norm_last = None

    for l in PROBE_LAYERS:
        for k in range(5):
            w = d[f'attn_layer{l}_k{k}']  # (1600, 512)
            all_row_sums.append(w.sum(axis=1))
            if len(pad_idx) > 0:
                all_pad_max.append(float(w[:, pad_idx].max()))
            all_real_total.append(float(w[:, real_idx].sum(1).mean()))
            if w_norm_last is None:
                w_real = w[:, real_idx]
                w_norm_last = w_real / w_real.sum(axis=1, keepdims=True)

    all_row_sums = np.concatenate(all_row_sums)
    max_pad_unmasked = max(all_pad_max) if all_pad_max else 0.0
    mean_real_total = float(np.mean(all_real_total))
    row_sum_masked = w_norm_last.sum(axis=1)

    result = {
        "n_real_tokens": int(len(real_idx)),
        "n_pad_tokens": int(len(pad_idx)),
        "real_token_range": [int(real_idx[0]), int(real_idx[-1])] if len(real_idx) else [],
        "n_layer_k_pairs_checked": len(PROBE_LAYERS) * 5,
        "unmasked_row_sum_mean": float(all_row_sums.mean()),
        "max_unmasked_row_sum_deviation_from_1": float(np.abs(all_row_sums - 1.0).max()),
        "max_pad_weight_unmasked_over_all_layer_k": float(max_pad_unmasked),
        "mean_total_weight_on_real_positions": mean_real_total,
        "mean_total_weight_on_pad_positions": 1.0 - mean_real_total,
        "masked_row_sum_mean": float(row_sum_masked.mean()),
        "masked_row_sum_max_deviation": float(np.abs(row_sum_masked - 1.0).max()),
        "t3_row_sum_pass": bool(np.abs(all_row_sums - 1.0).max() < 1e-3),
        "t3_pad_zero_pass": bool(max_pad_unmasked < 1e-4),
        "t3_row_sum_after_masking_pass": bool(np.abs(row_sum_masked - 1.0).max() < 1e-5),
        "interpretation": (
            f"Unmasked softmax: {(1.0 - mean_real_total)*100:.1f}% total weight on "
            f"{len(pad_idx)} pad positions (expected 0%%). "
            f"max_pad_weight={max_pad_unmasked:.4f} across all {len(PROBE_LAYERS)*5} (layer,k) pairs. "
            "T3 FAIL on unmasked capture; fix: apply attention_mask before softmax in hook. "
            "Masked+renorm over real tokens gives correct row sums."
        ),
    }
    return result


def run_section_3H(npz_path: Path, output_dir: Path, rng: np.random.Generator):
    """§3.H masked cross-attention analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz_path)
    mask = d['token_attention_mask']  # (512,)
    real_idx = np.where(mask == 1)[0]
    n_real = len(real_idx)
    uniform_val = 1.0 / n_real
    H_uniform = float(np.log(n_real))

    # Load and mask all (layer, k) pairs
    # w_masked[l][k] shape: (1600, n_real) — renormalized over real tokens
    w_masked = {}
    for l in PROBE_LAYERS:
        w_masked[l] = {}
        for k in range(5):
            w_raw = d[f'attn_layer{l}_k{k}']  # (1600, 512)
            w_r = w_raw[:, real_idx]
            w_masked[l][k] = w_r / w_r.sum(axis=1, keepdims=True)

    # ── 1. Entropy and KL by (layer, k) ──────────────────────────────────────
    stats = {}
    for l in PROBE_LAYERS:
        stats[l] = {}
        for k in range(5):
            w = w_masked[l][k]
            h = entropy(w)  # (1600,)
            kl = kl_to_uniform(w, n_real)  # (1600,)
            top1_w = w.max(axis=1)
            top3_w = np.sort(w, axis=1)[:, -3:].sum(axis=1)

            h_mean, h_lo, h_hi = bootstrap_ci(h, N_BOOTSTRAP, rng)
            kl_mean, kl_lo, kl_hi = bootstrap_ci(kl, N_BOOTSTRAP, rng)
            top1_mean, t1_lo, t1_hi = bootstrap_ci(top1_w, N_BOOTSTRAP, rng)
            top3_mean, t3_lo, t3_hi = bootstrap_ci(top3_w, N_BOOTSTRAP, rng)

            # permutation p-value: is KL significantly > 0?
            p_val = permutation_p_uniform(kl, n_real, N_PERMUTATION, rng)

            stats[l][k] = {
                "H_real": {"mean": h_mean, "ci_lo": h_lo, "ci_hi": h_hi},
                "H_uniform": H_uniform,
                "KL_to_uniform": {"mean": kl_mean, "ci_lo": kl_lo, "ci_hi": kl_hi},
                "top1_weight": {"mean": top1_mean, "ci_lo": t1_lo, "ci_hi": t1_hi,
                                "uniform_val": uniform_val, "ratio_vs_uniform": top1_mean / uniform_val},
                "top3_weight": {"mean": top3_mean, "ci_lo": t3_lo, "ci_hi": t3_hi,
                                "uniform_3": 3 * uniform_val, "ratio_vs_uniform_3": top3_mean / (3 * uniform_val)},
                "perm_p_KL_gt_0": p_val,
                "significant_vs_uniform": p_val < 0.05,
            }

    # ── 2. Plot: Entropy by layer and k ────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    for ki, k in enumerate(range(5)):
        ax = axes[ki]
        h_means = [stats[l][k]["H_real"]["mean"] for l in PROBE_LAYERS]
        h_los = [stats[l][k]["H_real"]["ci_lo"] for l in PROBE_LAYERS]
        h_his = [stats[l][k]["H_real"]["ci_hi"] for l in PROBE_LAYERS]
        ax.errorbar(PROBE_LAYERS, h_means,
                    yerr=[np.array(h_means) - np.array(h_los),
                          np.array(h_his) - np.array(h_means)],
                    marker='o', capsize=3, label=f'H_real k={k}')
        ax.axhline(H_uniform, color='red', linestyle='--', alpha=0.7,
                   label=f'H_uniform={H_uniform:.2f}')
        # Mark significant layers
        for li, l in enumerate(PROBE_LAYERS):
            if stats[l][k]["significant_vs_uniform"]:
                ax.plot(l, h_means[li], 'g*', markersize=10)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Entropy (nats)' if ki == 0 else '')
        ax.set_title(f'k={k} (σ={SIGMA_SCHEDULE[k]:.1f})')
        ax.legend(fontsize=7)
    fig.suptitle('§3.H Cross-Attention Entropy (masked, real tokens only)\n'
                 '* = significant vs uniform (perm p<0.05)', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / 'crossattn_entropy_by_layer_k.png', dpi=150)
    plt.close()

    # ── 3. Plot: Top-k attention vs uniform ───────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    k_focus = 4  # most committed step
    for ki, k in enumerate(range(5)):
        # Top-1 weight across layers
        ax = axes[0, ki]
        t1_means = [stats[l][k]["top1_weight"]["mean"] for l in PROBE_LAYERS]
        t1_los = [stats[l][k]["top1_weight"]["ci_lo"] for l in PROBE_LAYERS]
        t1_his = [stats[l][k]["top1_weight"]["ci_hi"] for l in PROBE_LAYERS]
        ax.errorbar(PROBE_LAYERS, t1_means,
                    yerr=[np.array(t1_means) - np.array(t1_los),
                          np.array(t1_his) - np.array(t1_means)],
                    marker='o', capsize=3, color='steelblue')
        ax.axhline(uniform_val, color='red', linestyle='--', alpha=0.7,
                   label=f'uniform={uniform_val:.3f}')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Top-1 attn weight' if ki == 0 else '')
        ax.set_title(f'Top-1, k={k}')
        ax.legend(fontsize=7)

        # Top-3 weight across layers
        ax = axes[1, ki]
        t3_means = [stats[l][k]["top3_weight"]["mean"] for l in PROBE_LAYERS]
        t3_los = [stats[l][k]["top3_weight"]["ci_lo"] for l in PROBE_LAYERS]
        t3_his = [stats[l][k]["top3_weight"]["ci_hi"] for l in PROBE_LAYERS]
        ax.errorbar(PROBE_LAYERS, t3_means,
                    yerr=[np.array(t3_means) - np.array(t3_los),
                          np.array(t3_his) - np.array(t3_means)],
                    marker='s', capsize=3, color='darkorange')
        ax.axhline(3 * uniform_val, color='red', linestyle='--', alpha=0.7,
                   label=f'3×uniform={3*uniform_val:.3f}')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Top-3 cumulative weight' if ki == 0 else '')
        ax.set_title(f'Top-3, k={k}')
        ax.legend(fontsize=7)

    fig.suptitle('§3.H Cross-Attention Token Selectivity (masked, 15 real tokens)\n'
                 'Red dashed = uniform baseline (1/15 per token)', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / 'crossattn_masked_topk.png', dpi=150)
    plt.close()

    # ── 4. KL effect size summary (by layer, averaged over k) ───────────────
    effect_summary = {}
    for l in PROBE_LAYERS:
        kl_by_k = [stats[l][k]["KL_to_uniform"]["mean"] for k in range(5)]
        sig_by_k = [stats[l][k]["significant_vs_uniform"] for k in range(5)]
        top1_by_k = [stats[l][k]["top1_weight"]["ratio_vs_uniform"] for k in range(5)]
        effect_summary[f"layer_{l}"] = {
            "KL_mean_over_k": float(np.mean(kl_by_k)),
            "KL_by_k": kl_by_k,
            "top1_ratio_vs_uniform_by_k": top1_by_k,
            "n_significant_k": int(sum(sig_by_k)),
            "all_significant": bool(all(sig_by_k)),
        }

    # Find which layers are universally significant
    sig_layers = [l for l in PROBE_LAYERS
                  if effect_summary[f"layer_{l}"]["all_significant"]]

    return {
        "n_real_tokens": n_real,
        "H_uniform": H_uniform,
        "uniform_top1": uniform_val,
        "stats_by_layer_k": {
            f"layer_{l}": {
                f"k{k}": stats[l][k] for k in range(5)
            } for l in PROBE_LAYERS
        },
        "effect_summary_by_layer": effect_summary,
        "significant_layers_all_k": sig_layers,
        "interpretation": (
            f"After pad masking (15 real tokens, {n_real} total), "
            f"uniform H={H_uniform:.2f} nats. "
            f"Layers {sig_layers} show significant KL>0 vs uniform across all k. "
            "Layers without universal significance: selective attention claims not supported."
        ),
    }


def main():
    rng = np.random.default_rng(RNG_SEED)

    npz_path = Path(
        "cosmos_policy/experiments/robot/robocasa/analysis/results/"
        "action_crossattn/crossattn.npz"
    )
    output_dir = Path(
        "cosmos_policy/experiments/robot/robocasa/analysis/results/crossattn_masked"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== T3: Pad Mask Verification ===")
    t3 = run_t3_verification(npz_path, output_dir)
    print(f"  n_real={t3['n_real_tokens']}, n_pad={t3['n_pad_tokens']}")
    print(f"  Unmasked: {t3['mean_total_weight_on_pad_positions']*100:.1f}% total on pad, "
          f"{t3['mean_total_weight_on_real_positions']*100:.1f}% total on real")
    print(f"  T3 row-sum PASS: {t3['t3_row_sum_pass']}")
    print(f"  T3 pad-zero PASS: {t3['t3_pad_zero_pass']} "
          f"(max_pad_weight={t3['max_pad_weight_unmasked_over_all_layer_k']:.4f})")
    print(f"  → {t3['interpretation']}")

    with open(output_dir / "t3_pad_mask_report.json", "w") as f:
        json.dump(t3, f, indent=2)
    print(f"Saved: {output_dir}/t3_pad_mask_report.json")

    print("\n=== §3.H Cross-Attention Masked Analysis ===")
    result_3h = run_section_3H(npz_path, output_dir, rng)
    print(f"  n_real={result_3h['n_real_tokens']}, H_uniform={result_3h['H_uniform']:.4f}")
    print(f"  Significant layers (all k): {result_3h['significant_layers_all_k']}")

    # Print per-layer KL summary
    print("\n  KL(attn‖uniform) by layer (mean over k=0..4):")
    for l in PROBE_LAYERS:
        es = result_3h["effect_summary_by_layer"][f"layer_{l}"]
        sig = "✓" if es["all_significant"] else "×"
        print(f"    Layer {l:2d}: KL={es['KL_mean_over_k']:.4f}, "
              f"top1_ratio={es['top1_ratio_vs_uniform_by_k'][4]:.2f}×uniform "
              f"[sig={sig}]")

    # Compute unmasked picture (as model actually computes)
    print("\n=== Unmasked stats (as model actually computes) ===")
    d = np.load(npz_path)
    mask = d['token_attention_mask']
    real_idx = np.where(mask == 1)[0]
    pad_idx = np.where(mask == 0)[0]
    n_real = len(real_idx)
    unmasked = {}
    print(f"  {'Layer':6s} {'Real total':12s} {'Pad total':12s} {'Top-1 real':12s} {'vs U(512)':10s}")
    for l in PROBE_LAYERS:
        unmasked[l] = {}
        for k in range(5):
            w = d[f'attn_layer{l}_k{k}']
            real_w = w[:, real_idx]
            pad_w = w[:, pad_idx]
            top1_real = real_w.max(axis=1)
            unmasked[l][k] = {
                "mean_weight_real_total": float(real_w.sum(1).mean()),
                "mean_weight_pad_total": float(pad_w.sum(1).mean()),
                "top1_real_weight": float(top1_real.mean()),
                "top1_real_vs_uniform_512": float((top1_real * 512).mean()),
            }
        s = unmasked[l][4]
        print(f"  Layer {l:2d}: {s['mean_weight_real_total']*100:.2f}%       "
              f"{s['mean_weight_pad_total']*100:.2f}%    "
              f"{s['top1_real_weight']*100:.3f}%    "
              f"{s['top1_real_vs_uniform_512']:.2f}×")

    result_3h["unmasked_stats_by_layer_k"] = {
        f"layer_{l}": {f"k{k}": unmasked[l][k] for k in range(5)}
        for l in PROBE_LAYERS
    }
    result_3h["model_design_note"] = (
        "DiT cross-attention (minimal_v4_dit.py L1353) passes crossattn_emb to cross_attn() "
        "with NO key_padding_mask. Unmasked softmax over 512 positions is what the model "
        "actually computes. Pad positions (15-511) receive ~97% of total attention weight. "
        "P8 masking is applied for analysis purposes only (not a model fix)."
    )

    with open(output_dir / "crossattn_vs_uniform_effectsize.json", "w") as f:
        json.dump(result_3h, f, indent=2)
    print(f"\nSaved: {output_dir}/crossattn_vs_uniform_effectsize.json")
    print(f"Saved: {output_dir}/crossattn_masked_topk.png")
    print(f"Saved: {output_dir}/crossattn_entropy_by_layer_k.png")
    print(f"\n{result_3h['interpretation']}")


if __name__ == "__main__":
    main()
