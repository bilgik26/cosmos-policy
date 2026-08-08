"""
§8 (テーマ7: 画像ラテント生成の解析) の再計算・全面拡充。
3seed成功epのみマージデータ (results/v4_merged/{image_features,image_hidfeat,
image_sigma_data}.npz) を用い、§2-3と同水準の統計的厳密さを適用する:
  §8.1: Δlatentノルム + episode bootstrap CI + 符号検定 + 線形ガウスヌル対比
  §8.2: FFT空間周波数(radial spectrum、収集時に事前計算済み) + B_B置換検定
        + 低/高周波帯域 + Spearman順位相関
  §8.3: (a) トークン間CKA(action vs future_primary vs curr_primary, 同一call)
        + bootstrap CI
        (b) 新規: future_primary のPR/CKA/コサインのkステップ推移
        （§3.1-3.4と全く同じ設計、画像ラテントトークンでのk=3→4崩壊の直接検証）
  §8.4: fold内標準化PCA + LogisticRegression + LOEO + permutationテスト + BH-FDR
        によるprogress_3プロービング(future_primary、§5.1と同一パイプライン)

実行:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.verification.image_latent.v4_section8 \
      --merged_dir results/v4_merged --out_json results/v4_merged/section8_results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.v4_stats_lib import (
    episode_bootstrap_ci, bb_permutation_test, gaussian_null_delta,
    participation_ratio, linear_cka, cohens_d_paired, bh_fdr,
    loeo_probe_permutation_test,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.probing.linear_probe import labels_from_progress

PROBE_LAYERS = [0, 4, 9, 13, 18, 22, 27]
NUM_DENOISE_STEPS = 5
SIGMA_SCHEDULE = [80.0, 42.3, 21.0, 9.6, 4.0]
LATENT_D = 16 * 28 * 28  # C'xH'xW'
TOKENS_MAIN = ["future_wrist", "future_primary", "future_secondary", "curr_primary", "action"]


def section_8_1_delta_norm(image_features_npz, n_boot=300):
    d = image_features_npz
    ep = d["episode_labels"]
    log_sigma = np.log(SIGMA_SCHEDULE)
    dlog = [abs(log_sigma[k + 1] - log_sigma[k]) for k in range(4)]
    results = {}
    for name in TOKENS_MAIN:
        tok_result = {"raw": {}, "schednorm": {}}
        deltas_by_k = {}
        for k in range(1, NUM_DENOISE_STEPS):
            key = f"delta_norm_{name}_k{k}"
            if key not in d.files:
                continue
            raw = d[key]
            sched = raw / dlog[k - 1]
            deltas_by_k[k] = sched
            tok_result["raw"][k] = episode_bootstrap_ci(raw, ep, n_boot=n_boot)
            tok_result["schednorm"][k] = episode_bootstrap_ci(sched, ep, n_boot=n_boot)
        # sign test k1 vs k4 (schednorm)
        if 1 in deltas_by_k and 4 in deltas_by_k:
            eps = np.unique(ep)
            n_gt, n_tot = 0, 0
            for e in eps:
                m = ep == e
                v1, v4 = deltas_by_k[1][m].mean(), deltas_by_k[4][m].mean()
                if v1 == v4:
                    continue
                n_tot += 1
                if v4 > v1:
                    n_gt += 1
            p = binomtest(n_gt, n_tot, 0.5).pvalue if n_tot > 0 else float("nan")
            tok_result["sign_test_k1_vs_k4"] = {"n_episodes": n_tot, "n_greater": n_gt, "p_value": float(p)}
        results[name] = tok_result
    return results


def section_8_1_gaussian_null(image_sigma_data_npz, delta_results):
    d = image_sigma_data_npz
    null_results = {}
    for name in ["future_wrist", "future_primary", "future_secondary"]:
        key = f"sigma_data_{name}"
        if key not in d.files:
            continue
        sigma_data = float(d[key][0])
        n = int(d[f"sigma_data_n_{name}"][0])
        null_raw = gaussian_null_delta(SIGMA_SCHEDULE, sigma_data, LATENT_D)
        log_sigma = np.log(SIGMA_SCHEDULE)
        dlog = np.array([abs(log_sigma[k + 1] - log_sigma[k]) for k in range(4)])
        null_sched = null_raw / dlog
        observed = [delta_results.get(name, {}).get("schednorm", {}).get(k, {}).get("point")
                    for k in range(1, 5)]
        ratio = [float(o / n_) if (o is not None and n_ > 0) else None for o, n_ in zip(observed, null_sched)]
        null_results[name] = {
            "sigma_data": sigma_data, "sigma_data_n_calls": n,
            "null_schednorm": null_sched.tolist(), "observed_schednorm": observed,
            "ratio_observed_over_null": ratio,
        }
    return null_results


def section_8_2_fft(image_features_npz, n_perm=1000):
    d = image_features_npz
    ep = d["episode_labels"]
    results = {}
    for name in ["future_wrist", "future_primary", "future_secondary"]:
        specs_by_k = []
        for k in range(NUM_DENOISE_STEPS):
            key = f"fft_{name}_k{k}"
            if key not in d.files:
                specs_by_k = None
                break
            specs_by_k.append(d[key])  # (N_calls, 14)
        if specs_by_k is None:
            continue
        n_calls = specs_by_k[0].shape[0]
        radii = np.arange(14)
        centroids = np.zeros((n_calls, NUM_DENOISE_STEPS))
        low_power = np.zeros((n_calls, NUM_DENOISE_STEPS))
        high_power = np.zeros((n_calls, NUM_DENOISE_STEPS))
        for k in range(NUM_DENOISE_STEPS):
            spec = specs_by_k[k]  # (N,14)
            denom = spec.sum(axis=1) + 1e-12
            centroids[:, k] = (spec * radii[None, :]).sum(axis=1) / denom
            low_power[:, k] = spec[:, :4].mean(axis=1)
            high_power[:, k] = spec[:, 4:].mean(axis=1)
        bb = bb_permutation_test(centroids, np.arange(n_calls), n_perm=n_perm)
        per_step_mean = centroids.mean(axis=0)
        rho, p_rho = spearmanr(np.arange(NUM_DENOISE_STEPS), per_step_mean)
        results[name] = {
            "centroid_per_step": per_step_mean.tolist(),
            "bb_permutation_test": bb, "spearman_rho": float(rho), "spearman_p": float(p_rho),
            "low_freq_power_per_step": low_power.mean(axis=0).tolist(),
            "high_freq_power_per_step": high_power.mean(axis=0).tolist(),
        }
    return results


def section_8_3a_cross_token_cka(image_hidfeat_npz, n_boot=200, sample_size=200, seed=0):
    """action vs future_primary vs curr_primary のトークン間CKA (同一call)。"""
    rng = np.random.RandomState(seed)
    d = image_hidfeat_npz
    pairs = [("action", "future_primary"), ("action", "curr_primary")]
    results = {}
    for (a_name, b_name) in pairs:
        pair_key = f"{a_name}_vs_{b_name}"
        results[pair_key] = {}
        for l in PROBE_LAYERS:
            per_k = {}
            for k in range(NUM_DENOISE_STEPS):
                ka, kb = f"hidfeat_{a_name}_k{k}_layer{l}", f"hidfeat_{b_name}_k{k}_layer{l}"
                if ka not in d.files or kb not in d.files:
                    continue
                A, B = d[ka], d[kb]
                N = A.shape[0]
                point = linear_cka(A, B)
                ss = min(sample_size, N)
                boots = [linear_cka(A[idx], B[idx]) for idx in
                         (rng.choice(N, size=ss, replace=True) for _ in range(n_boot))]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                per_k[k] = {"point": point, "ci_lo": float(lo), "ci_hi": float(hi)}
            results[pair_key][l] = per_k
    return results


def section_8_3b_pr_cka_cosine_kcollapse(image_hidfeat_npz, ep, token="future_primary", n_boot=300, seed=0):
    """§3.1-3.4と同一設計: future_primaryのPR/CKA/コサインのk推移+Cohen's d。"""
    rng = np.random.RandomState(seed)
    d = image_hidfeat_npz
    feats = {}
    for k in range(NUM_DENOISE_STEPS):
        feats[k] = {}
        for l in PROBE_LAYERS:
            key = f"hidfeat_{token}_k{k}_layer{l}"
            if key in d.files:
                feats[k][l] = d[key]
    if not feats[0]:
        return {"available": False}

    eps = np.unique(ep)
    pr_results, norm_results, cos_results, cka_results, cross = {}, {}, {}, {}, {}
    for l in PROBE_LAYERS:
        if l not in feats[0]:
            continue
        pr_by_k = {}
        pr_ep_by_k = {}
        for k in range(NUM_DENOISE_STEPS):
            X = feats[k][l]
            pr_point = participation_ratio(X)
            ep_to_idx = {e: np.where(ep == e)[0] for e in eps}
            boots = []
            for _ in range(n_boot):
                samp = rng.choice(eps, size=len(eps), replace=True)
                idxs = np.concatenate([ep_to_idx[e] for e in samp])
                boots.append(participation_ratio(X[idxs]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            pr_by_k[k] = {"point": pr_point, "ci_lo": float(lo), "ci_hi": float(hi)}
            pr_ep_by_k[k] = np.array([
                participation_ratio(X[ep == e]) if (ep == e).sum() >= 3 else np.nan for e in eps
            ])
        pr_results[l] = pr_by_k
        a, b = pr_ep_by_k[3], pr_ep_by_k[4]
        valid = ~(np.isnan(a) | np.isnan(b))
        if valid.sum() >= 3:
            from scipy.stats import binomtest as bt
            n_dec = int((b[valid] < a[valid]).sum())
            res = bt(n_dec, valid.sum(), 0.5)
            cross.setdefault(l, {})["k3to4_sign_test_p"] = float(res.pvalue)
            cross[l]["cohens_d_pr_3to4"] = cohens_d_paired(a[valid], b[valid])

        norms = {k: float(np.linalg.norm(feats[k][l], axis=1).mean()) for k in range(NUM_DENOISE_STEPS)}
        norm_results[l] = norms
        cos = {}
        for k in range(NUM_DENOISE_STEPS - 1):
            A, B = feats[k][l], feats[k + 1][l]
            num = (A * B).sum(axis=1)
            den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
            cos[f"{k}to{k+1}"] = float((num / den).mean())
        cos_results[l] = cos
        cross.setdefault(l, {})["cosine_3to4"] = cos.get("3to4")

        cka_t = {}
        for k in range(NUM_DENOISE_STEPS - 1):
            A, B = feats[k][l], feats[k + 1][l]
            N = A.shape[0]
            point = linear_cka(A, B)
            ss = min(200, N)
            boots = [linear_cka(A[idx], B[idx]) for idx in
                     (rng.choice(N, size=ss, replace=True) for _ in range(200))]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            cka_t[f"{k}to{k+1}"] = {"point": point, "ci_lo": float(lo), "ci_hi": float(hi)}
        cka_results[l] = cka_t
        cross[l]["cka_3to4"] = cka_t.get("3to4", {}).get("point")

    return {"available": True, "pr": pr_results, "norm": norm_results,
            "cosine": cos_results, "cka": cka_results, "cross_agreement": cross}


def section_8_4_probing(image_hidfeat_npz, n_perm=100):
    d = image_hidfeat_npz
    if "hidden_episode_labels" not in d.files or "hidden_call_idx_labels" not in d.files:
        return {"available": False, "reason": "no call_idx labels in merged image_hidfeat"}
    ep, ci = d["hidden_episode_labels"], d["hidden_call_idx_labels"]
    labels, _ = labels_from_progress(ep, ci, n_classes=3)
    results = {}
    all_p, all_keys = [], []
    for k in range(NUM_DENOISE_STEPS):
        results[k] = {}
        for l in PROBE_LAYERS:
            key = f"hidfeat_future_primary_k{k}_layer{l}"
            if key not in d.files:
                continue
            X = d[key]
            res = loeo_probe_permutation_test(X, labels, ep, n_components=30, n_perm=n_perm)
            results[k][l] = res
            all_p.append(res["p_value"])
            all_keys.append((k, l))
    sig = bh_fdr(all_p)
    for (k, l), s in zip(all_keys, sig):
        results[k][l]["bh_significant"] = bool(s)
    return {"available": True, "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", default="cosmos_policy/experiments/robot/robocasa/analysis/results/v4_merged")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--n_perm_probe", type=int, default=100)
    args = ap.parse_args()
    merged_dir = Path(args.merged_dir)
    out_json = Path(args.out_json) if args.out_json else merged_dir / "section8_results.json"

    img_feat = np.load(merged_dir / "image_features.npz")
    img_hid = np.load(merged_dir / "image_hidfeat.npz")
    img_sigma = np.load(merged_dir / "image_sigma_data.npz")
    ep = img_feat["episode_labels"]
    print(f"image_features: N_calls={len(ep)}, N_episodes={len(np.unique(ep))}")

    print("§8.1 Δlatent norm...")
    delta_results = section_8_1_delta_norm(img_feat)
    print("§8.1 gaussian null...")
    null_results = section_8_1_gaussian_null(img_sigma, delta_results)
    for name, r in null_results.items():
        print(f"  {name}: sigma_data={r['sigma_data']:.3f} (N={r['sigma_data_n_calls']}) "
              f"ratio={[round(x,2) if x else None for x in r['ratio_observed_over_null']]}")

    print("§8.2 FFT...")
    fft_results = section_8_2_fft(img_feat)

    print("§8.3a cross-token CKA...")
    cross_cka = section_8_3a_cross_token_cka(img_hid)

    print("§8.3b PR/CKA/cosine k-collapse (future_primary)...")
    ep_hid = img_hid["hidden_episode_labels"] if "hidden_episode_labels" in img_hid.files else ep
    kcollapse = section_8_3b_pr_cka_cosine_kcollapse(img_hid, ep_hid)
    if kcollapse.get("available"):
        for l in PROBE_LAYERS:
            if l in kcollapse["pr"]:
                pr_str = " ".join(f"k{k}={kcollapse['pr'][l][k]['point']:.1f}" for k in range(5))
                print(f"  Blk-{l}: PR[{pr_str}]")

    print(f"§8.4 probing (n_perm={args.n_perm_probe})...")
    probe_results = section_8_4_probing(img_hid, n_perm=args.n_perm_probe)
    print("  available:", probe_results.get("available"))

    out = {
        "n_calls": int(len(ep)), "n_episodes": int(len(np.unique(ep))),
        "delta_norm": delta_results, "gaussian_null": null_results,
        "fft": fft_results, "cross_token_cka": cross_cka,
        "kcollapse_future_primary": kcollapse, "probing": probe_results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
