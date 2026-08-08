"""
§2 (テーマ1: 行動出力レベルのデノイジング動態) の再計算。3seed成功epのみ
マージデータ(results/v4_merged/step_actions.npz)を用いる。

計算内容:
  §2.1 Δx̂₀ raw/relative/schednorm + episode bootstrap CI + 符号検定(k0→1 vs k3→4)
       + 線形ガウス最適デノイザヌルとの対比 + sigma_data感度分析
  §2.2 FFTスペクトル重心 + B_B置換検定 + 低/高周波帯域分割
  §2.3 次元別RMS + グリッパー切替分析 + 空間オフセット
  §2.4 (可能な場合のみ) noise_pred_norm/F_theta の測度集中解析

実行:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.v4_section2 \
      --merged_dir results/v4_merged --out_json results/v4_merged/section2_results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

from cosmos_policy.experiments.robot.robocasa.analysis.v4_stats_lib import (
    episode_bootstrap_ci, bb_permutation_test, gaussian_null_delta, estimate_sigma_data,
)

NUM_DENOISE_STEPS = 5
SIGMA_SCHEDULE = [80.0, 42.3, 21.0, 9.6, 4.0]
CHUNK_SIZE = 32
ACTION_DIM = 7


def load_step_actions(merged_dir: Path):
    d = np.load(merged_dir / "step_actions.npz")
    acts = {k: d[str(k)] for k in range(NUM_DENOISE_STEPS)}
    ep = {k: d[f"episode_labels_{k}"] for k in range(NUM_DENOISE_STEPS)}
    return acts, ep, d


def section_2_1_delta(acts, ep, n_boot=1000, seed=0):
    """Δx̂₀ raw/relative/schednorm + bootstrap CI + 符号検定 + ガウスヌル対比。"""
    D = CHUNK_SIZE * ACTION_DIM
    log_sigma = np.log(SIGMA_SCHEDULE)
    dlog = [abs(log_sigma[k + 1] - log_sigma[k]) for k in range(4)]

    # 全てのkでcall数が一致することを前提とする(mismatchがあれば警告)
    ns = [acts[k].shape[0] for k in range(5)]
    n_min = min(ns)
    if len(set(ns)) > 1:
        print(f"  [warn] step call counts differ: {ns}, truncating to n_min={n_min}")

    results = {"raw": {}, "relative": {}, "schednorm": {}}
    ep_k = ep[0][:n_min]  # 各kで同じcall集合のはず(mechanism_analysis.pyの単一ロールアウト由来)

    raw_deltas = {}
    for k in range(4):
        a = acts[k][:n_min].reshape(n_min, -1)
        b = acts[k + 1][:n_min].reshape(n_min, -1)
        raw = np.linalg.norm(b - a, axis=1)
        rel = raw / (np.linalg.norm(a, axis=1) + 1e-12)
        sched = raw / dlog[k]
        raw_deltas[k] = raw
        results["raw"][k] = episode_bootstrap_ci(raw, ep_k, n_boot=n_boot)
        results["relative"][k] = episode_bootstrap_ci(rel, ep_k, n_boot=n_boot)
        results["schednorm"][k] = episode_bootstrap_ci(sched, ep_k, n_boot=n_boot)

    # 符号検定: k0->1 vs k3->4 (schednorm)
    sched0 = raw_deltas[0] / dlog[0]
    sched3 = raw_deltas[3] / dlog[3]
    eps = np.unique(ep_k)
    n_gt, n_tot = 0, 0
    for e in eps:
        m = ep_k == e
        if m.sum() == 0:
            continue
        v0, v3 = sched0[m].mean(), sched3[m].mean()
        if v0 == v3:
            continue
        n_tot += 1
        if v3 > v0:
            n_gt += 1
    p = binomtest(n_gt, n_tot, 0.5).pvalue if n_tot > 0 else float("nan")
    results["sign_test_k0to1_vs_k3to4"] = {"n_episodes": n_tot, "n_greater": n_gt, "p_value": float(p)}

    # 線形ガウスヌルとの対比
    x0_final = acts[4][:n_min].reshape(n_min, -1)
    sigma_data_emp = estimate_sigma_data(x0_final)
    null_emp = gaussian_null_delta(SIGMA_SCHEDULE, sigma_data_emp, D)
    null_emp_sched = null_emp / np.array(dlog)

    sigma_data_edm = 0.5
    null_edm = gaussian_null_delta(SIGMA_SCHEDULE, sigma_data_edm, D)
    null_edm_sched = null_edm / np.array(dlog)

    observed_sched = [results["schednorm"][k]["point"] for k in range(4)]
    results["gaussian_null"] = {
        "sigma_data_empirical": sigma_data_emp,
        "null_schednorm_empirical_sigma": null_emp_sched.tolist(),
        "null_schednorm_edm_sigma": null_edm_sched.tolist(),
        "observed_schednorm": observed_sched,
        "ratio_empirical_sigma": [float(o / n) for o, n in zip(observed_sched, null_emp_sched)],
        "ratio_edm_sigma": [float(o / n) for o, n in zip(observed_sched, null_edm_sched)],
    }
    return results, n_min, ep_k


def section_2_2_fft(acts, ep_k, n_min, n_perm=1000):
    """FFTスペクトル重心 + B_B置換検定 + 低/高周波帯域。
    対象は各 policy call の Δx̂₀ = x̂₀(k+1)-x̂₀(k)（行動差分、4遷移分）であり、
    x̂₀ 自体ではない（report §2.2 methodology: "各policy callのΔx̂₀を次元ごとに
    1次元FFTし..."）。"""
    n_transitions = NUM_DENOISE_STEPS - 1
    centroids_per_call = np.zeros((n_min, n_transitions))
    low_power = np.zeros((n_min, n_transitions))
    high_power = np.zeros((n_min, n_transitions))
    for k in range(n_transitions):
        delta = acts[k + 1][:n_min] - acts[k][:n_min]  # (n_min, 32, 7)
        for i in range(n_min):
            dim_centroids = []
            for dim in range(ACTION_DIM):
                series = delta[i, :, dim]
                fft_mag = np.abs(np.fft.rfft(series - series.mean()))
                freqs = np.fft.rfftfreq(len(series))
                if fft_mag.sum() > 1e-12:
                    centroid = float((fft_mag * freqs).sum() / fft_mag.sum())
                else:
                    centroid = 0.0
                dim_centroids.append(centroid)
                low_power[i, k] += fft_mag[freqs <= 0.25].mean() if (freqs <= 0.25).any() else 0.0
                high_power[i, k] += fft_mag[freqs > 0.25].mean() if (freqs > 0.25).any() else 0.0
            centroids_per_call[i, k] = np.mean(dim_centroids)
        low_power[:, k] /= ACTION_DIM
        high_power[:, k] /= ACTION_DIM

    bb = bb_permutation_test(centroids_per_call, np.arange(n_min), n_perm=n_perm)
    per_transition_mean = centroids_per_call.mean(axis=0)
    rho, p_rho = spearmanr(np.arange(n_transitions), per_transition_mean)

    return {
        "centroid_per_transition": per_transition_mean.tolist(),  # [k0->1, k1->2, k2->3, k3->4]
        "bb_permutation_test": bb,
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "low_freq_power_per_transition": low_power.mean(axis=0).tolist(),
        "high_freq_power_per_transition": high_power.mean(axis=0).tolist(),
    }


def section_2_3_dimwise(acts, n_min):
    """次元別RMS(Δx_t、チャンク内の隣接timestep差)+ グリッパー切替分析 + 空間オフセット。
    report §2.3 methodology: "次元グループ別のRMS(Δx_t)" — 生の値のRMSではなく、
    32-stepチャンク内での隣接timestep間の差分のRMSである点に注意
    （生値のRMSでは常に≈1付近に飽和し、切替の有無による差が実質的に消えてしまう
    ことをこのデータで確認し、report記述に立ち返って修正した）。"""
    dim_names = ["X", "Y", "Z", "Rx", "Ry", "Rz", "Grip"]
    rms_by_dim_k = {d: {} for d in range(ACTION_DIM)}
    for k in range(NUM_DENOISE_STEPS):
        a = acts[k][:n_min]
        d_t = a[:, 1:, :] - a[:, :-1, :]  # (n_min, 31, 7) チャンク内の隣接timestep差
        for d in range(ACTION_DIM):
            rms_by_dim_k[d][k] = float(np.sqrt((d_t[:, :, d] ** 2).mean()))

    # グリッパー切替 (dim=6, 閾値0で開閉判定)。RMSはΔx_t(チャンク内隣接差)基準。
    grip = {k: acts[k][:n_min, :, 6] for k in range(NUM_DENOISE_STEPS)}
    switch_results = {}
    for k in range(NUM_DENOISE_STEPS):
        g = grip[k]
        opened = g > 0
        switches = np.any(opened[:, :-1] != opened[:, 1:], axis=1)
        g_dt = g[:, 1:] - g[:, :-1]  # (n_min, 31)
        n_switch = int(switches.sum())
        rms_switch = float(np.sqrt((g_dt[switches] ** 2).mean())) if switches.any() else float("nan")
        rms_noswitch = float(np.sqrt((g_dt[~switches] ** 2).mean())) if (~switches).any() else float("nan")
        switch_results[k] = {"n_switch_calls": n_switch, "rms_switch": rms_switch, "rms_noswitch": rms_noswitch}

    # 空間オフセット (k=0→4)
    offset = acts[4][:n_min] - acts[0][:n_min]
    offset_mean = offset.mean(axis=(0, 1))
    offset_std = offset.std(axis=(0, 1))

    return {
        "rms_by_dim": {dim_names[d]: rms_by_dim_k[d] for d in range(ACTION_DIM)},
        "gripper_switch": switch_results,
        "offset_mean": {dim_names[d]: float(offset_mean[d]) for d in range(ACTION_DIM)},
        "offset_std": {dim_names[d]: float(offset_std[d]) for d in range(ACTION_DIM)},
    }


def section_2_4_precond(merged_dir: Path, n_min_hint=None):
    """noise_pred_norm/F_theta測度集中解析（norm_k/sigma_kフィールドがある場合のみ）。"""
    d = np.load(merged_dir / "step_actions.npz")
    if "norm_0" not in d.files:
        return {"available": False, "reason": "norm_k fields not present in merged step_actions.npz"}
    results = {"available": True, "per_step": {}}
    for k in range(NUM_DENOISE_STEPS):
        norms = d[f"norm_{k}"]
        sigmas = d[f"sigma_{k}"]
        cv = float(norms.std() / norms.mean()) if norms.mean() > 0 else float("nan")
        results["per_step"][k] = {
            "mean_norm": float(norms.mean()), "cv_percent": cv * 100,
            "mean_sigma": float(sigmas.mean()), "n": int(len(norms)),
        }
    d_inferred = results["per_step"][0]["mean_norm"] ** 2
    results["d_inferred_from_cv"] = float(d_inferred)
    results["theoretical_cv_1_over_sqrt2d"] = float(1.0 / np.sqrt(2 * d_inferred)) * 100
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", default="cosmos_policy/experiments/robot/robocasa/analysis/results/v4_merged")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    merged_dir = Path(args.merged_dir)
    out_json = Path(args.out_json) if args.out_json else merged_dir / "section2_results.json"

    acts, ep, _ = load_step_actions(merged_dir)
    print(f"Loaded step_actions: N per k = {[acts[k].shape[0] for k in range(5)]}")

    print("Computing §2.1 Δx̂₀ + gaussian null...")
    delta_results, n_min, ep_k = section_2_1_delta(acts, ep)
    print("  schednorm:", {k: round(delta_results['schednorm'][k]['point'], 4) for k in range(4)})
    print("  null ratio (empirical sigma_data):", delta_results["gaussian_null"]["ratio_empirical_sigma"])

    print("Computing §2.2 FFT...")
    fft_results = section_2_2_fft(acts, ep_k, n_min)
    print("  centroid per transition:", fft_results["centroid_per_transition"])

    print("Computing §2.3 dim-wise + gripper switch...")
    dimwise_results = section_2_3_dimwise(acts, n_min)

    print("Computing §2.4 precond (if available)...")
    precond_results = section_2_4_precond(merged_dir)
    print(" ", precond_results.get("available"))

    out = {
        "n_calls": n_min, "n_episodes": int(len(np.unique(ep_k))),
        "delta_x0": delta_results, "fft": fft_results,
        "dimwise": dimwise_results, "precond": precond_results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
