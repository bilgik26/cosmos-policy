"""
§3 (テーマ2: 中間層内部表現の構造的崩壊) の再計算。3seed成功epのみマージデータ
(results/v4_merged/action_features.npz) を用いる。

計算内容:
  §3.1 PR (Participation Ratio) の k=0..4 推移 + episode符号検定(k3->4)
  §3.2 特徴ノルム(k=0..4 全ステップ、要件2で追加) + ステップ間コサイン類似度
  §3.3 ステップ間CKA (サンプルブートストラップ n=200 CI付き)
  §3.4 3指標クロス一致 + Cohen's d (episode単位ペア)

実行:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.v4_section3 \
      --merged_dir results/v4_merged --out_json results/v4_merged/section3_results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cosmos_policy.experiments.robot.robocasa.analysis.v4_stats_lib import (
    participation_ratio, linear_cka, cohens_d_paired, episode_sign_test,
)

PROBE_LAYERS = [0, 4, 9, 13, 18, 22, 27]
NUM_DENOISE_STEPS = 5


def load_feats(merged_dir: Path):
    d = np.load(merged_dir / "action_features.npz")
    ep = d["episode_labels"]
    feats = {}
    for k in range(NUM_DENOISE_STEPS):
        feats[k] = {}
        for l in PROBE_LAYERS:
            key = f"feat_k{k}_layer{l}"
            if key in d.files:
                feats[k][l] = d[key]
    return feats, ep


def section_3_1_pr(feats, ep, n_boot=300, seed=0):
    """PR (k=0..4) + episode単位bootstrap CI + k3->4符号検定。"""
    rng = np.random.RandomState(seed)
    eps = np.unique(ep)
    results = {}
    for l in PROBE_LAYERS:
        pr_by_k = {}
        pr_ep_by_k = {}  # per-episode PR (for sign test / Cohen's d)
        for k in range(NUM_DENOISE_STEPS):
            X = feats[k][l]
            pr_point = participation_ratio(X)
            # episode単位bootstrap
            ep_to_idx = {e: np.where(ep == e)[0] for e in eps}
            boots = []
            for _ in range(n_boot):
                samp = rng.choice(eps, size=len(eps), replace=True)
                idxs = np.concatenate([ep_to_idx[e] for e in samp])
                boots.append(participation_ratio(X[idxs]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            pr_by_k[k] = {"point": pr_point, "ci_lo": float(lo), "ci_hi": float(hi)}
            # per-episode PR (小サンプルなのでepisode毎に計算、符号検定・Cohen's d用)
            pr_ep_by_k[k] = np.array([
                participation_ratio(X[ep == e]) if (ep == e).sum() >= 3 else np.nan
                for e in eps
            ])
        results[l] = {"pr_by_k": pr_by_k}
        # k3->4 sign test (call-levelではなくepisode単位PRの対比)
        a, b = pr_ep_by_k[3], pr_ep_by_k[4]
        valid = ~(np.isnan(a) | np.isnan(b))
        if valid.sum() >= 3:
            from scipy.stats import binomtest
            n_gt = int((b[valid] < a[valid]).sum())  # PR低下 = b<a
            res = binomtest(n_gt, valid.sum(), 0.5)
            results[l]["k3to4_sign_test"] = {
                "n_episodes": int(valid.sum()), "n_decreased": n_gt, "p_value": float(res.pvalue)
            }
            results[l]["k3to4_cohens_d"] = cohens_d_paired(a[valid], b[valid])
    return results


def section_3_2_norm_cosine(feats, ep):
    """特徴ノルム(k=0..4、要件2で追加) + ステップ間コサイン類似度。"""
    results = {"norm": {}, "cosine": {}}
    for l in PROBE_LAYERS:
        norms = {}
        for k in range(NUM_DENOISE_STEPS):
            X = feats[k][l]
            norms[k] = float(np.linalg.norm(X, axis=1).mean())
        results["norm"][l] = norms
        cos = {}
        for k in range(NUM_DENOISE_STEPS - 1):
            A, B = feats[k][l], feats[k + 1][l]
            num = (A * B).sum(axis=1)
            den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
            cos[f"{k}to{k+1}"] = float((num / den).mean())
        results["cosine"][l] = cos
    return results


def section_3_3_cka(feats, n_boot=200, sample_size=200, seed=0):
    """ステップ間CKA + サンプルブートストラップCI。"""
    rng = np.random.RandomState(seed)
    results = {}
    for l in PROBE_LAYERS:
        cka_by_transition = {}
        for k in range(NUM_DENOISE_STEPS - 1):
            A, B = feats[k][l], feats[k + 1][l]
            N = A.shape[0]
            point = linear_cka(A, B)
            boots = []
            ss = min(sample_size, N)
            for _ in range(n_boot):
                idx = rng.choice(N, size=ss, replace=True)
                boots.append(linear_cka(A[idx], B[idx]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            se = float(np.std(boots))
            cka_by_transition[f"{k}to{k+1}"] = {
                "point": point, "ci_lo": float(lo), "ci_hi": float(hi), "se": se,
            }
        results[l] = cka_by_transition
    return results


def section_3_4_cross_agreement(pr_results, cka_results, norm_cos_results):
    """3指標のクロス一致テーブル(k3->4)。"""
    out = {}
    for l in PROBE_LAYERS:
        pr34 = pr_results[l]["pr_by_k"][4]["point"] - pr_results[l]["pr_by_k"][3]["point"]
        cka34 = cka_results[l].get("3to4", {}).get("point")
        cos34 = norm_cos_results["cosine"][l].get("3to4")
        d = pr_results[l].get("k3to4_cohens_d")
        out[l] = {
            "pr_delta_3to4": pr34, "cka_3to4": cka34, "cosine_3to4": cos34,
            "cohens_d_pr_3to4": d,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", default="cosmos_policy/experiments/robot/robocasa/analysis/results/v4_merged")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    merged_dir = Path(args.merged_dir)
    out_json = Path(args.out_json) if args.out_json else merged_dir / "section3_results.json"

    feats, ep = load_feats(merged_dir)
    n_calls = len(ep)
    n_eps = len(np.unique(ep))
    print(f"Loaded: N_calls={n_calls}, N_episodes={n_eps}")

    print("Computing §3.1 PR...")
    pr_results = section_3_1_pr(feats, ep)
    print("Computing §3.2 norm/cosine...")
    nc_results = section_3_2_norm_cosine(feats, ep)
    print("Computing §3.3 CKA...")
    cka_results = section_3_3_cka(feats)
    print("Computing §3.4 cross-agreement...")
    cross = section_3_4_cross_agreement(pr_results, cka_results, nc_results)

    for l in PROBE_LAYERS:
        pr_str = " ".join(f"k{k}={pr_results[l]['pr_by_k'][k]['point']:.1f}" for k in range(5))
        print(f"  Blk-{l}: PR[{pr_str}]  CKA(3->4)={cka_results[l].get('3to4',{}).get('point'):.3f}  "
              f"cos(3->4)={nc_results['cosine'][l].get('3to4'):.4f}  d={pr_results[l].get('k3to4_cohens_d'):.2f}")

    out = {
        "n_calls": n_calls, "n_episodes": n_eps,
        "pr": pr_results, "norm_cosine": nc_results, "cka": cka_results, "cross_agreement": cross,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
