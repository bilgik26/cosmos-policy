"""
rollout_transition.py — attractor_verification_design.md §5.C 準拠

ロールアウト中 (t軸, policy call を跨ぐ) の表現遷移をセグメント化し、
「離散フェーズ状態を遷移し、境界で滞在が短い」という H を検証する。

**実装上の代替 (時間予算の都合、正直に明記)**: 設計書は Sticky HDP-HMM を指定するが、
bnpy/pyhsmm 等の専用パッケージは本環境に未導入・導入コストが高いため、
`hmmlearn.GaussianHMM` (固定 K, BIC で 2〜6 から選択) で代替する。「sticky」性は
学習された遷移行列の対角成分 (自己遷移確率) で評価する。

手順:
  1. 各タスクについて、シーン残差化 (Z - Z̄_episode) した Z_pool(k=last, layer=mid) を
     episode ごとに call_idx 順に並べた系列を作る。
  2. PELT (ruptures) で連続特徴の変化点を検出 → 変化点密度・平均セグメント長。
  3. GaussianHMM (BIC選択) で状態列を推定 → 遷移行列・自己遷移確率(=stickiness)・
     状態↔phase_label / 状態↔episode(scene) の MI。
  4. Null: episode 内で call の順序をシャッフル (シーン分布は保持、時間構造のみ破壊) し、
     同じ K で HMM を再学習、自己遷移確率と対数尤度の帰無分布を作る。
"""

import json
from pathlib import Path

import numpy as np
import ruptures as rpt
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from cosmos_policy.experiments.robot.robot_utils import log_message

LAYER = 13
K_STEP = 4


def load_task_seed_data(collect_dir: Path, fname: str, layer: int, k: int):
    feat_path = collect_dir / fname
    phase_path = collect_dir / fname.replace(".npz", "_phases.npz")
    fd = np.load(feat_path)
    pd = np.load(phase_path)
    key = f"feat_k{k}_layer{layer}"
    idx_key = key + "_idx"
    keep_idx = fd[idx_key]
    return {
        "feats": fd[key],
        "episode": fd["episode"][keep_idx],
        "call_idx": fd["call_idx"][keep_idx],
        "phase_label": pd["phase_label"][keep_idx],
    }


def scene_residualize(X, episode):
    Xr = X.copy()
    for e in np.unique(episode):
        mask = episode == e
        Xr[mask] = X[mask] - X[mask].mean(axis=0, keepdims=True)
    return Xr


def order_by_episode_call(X, episode, call_idx, phase_label):
    order = np.lexsort((call_idx, episode))
    lengths = [np.sum(episode == e) for e in np.unique(episode)]
    return X[order], episode[order], phase_label[order], lengths


def pelt_changepoints(X_seq, lengths, pen=None):
    """Per-episode PELT changepoint count on the (low-dim) continuous sequence."""
    offsets = np.cumsum([0] + lengths)
    n_cps_per_ep = []
    for i, L in enumerate(lengths):
        seg = X_seq[offsets[i]:offsets[i + 1]]
        if L < 6:
            n_cps_per_ep.append(0)
            continue
        p = pen if pen is not None else 3 * np.log(L) * seg.shape[1]
        algo = rpt.Pelt(model="rbf").fit(seg)
        cps = algo.predict(pen=p)
        n_cps_per_ep.append(max(len(cps) - 1, 0))  # last cp == L always included
    return n_cps_per_ep


def fit_hmm_bic(X_seq, lengths, k_range=range(2, 7), seed=0):
    best = None
    for k in k_range:
        try:
            model = GaussianHMM(n_components=k, covariance_type="diag",
                                 n_iter=100, random_state=seed)
            model.fit(X_seq, lengths)
            ll = model.score(X_seq, lengths)
            n_params = k * k - 1 + k * X_seq.shape[1] * 2 + k - 1
            bic = -2 * ll + n_params * np.log(X_seq.shape[0])
            if best is None or bic < best[0]:
                best = (bic, k, model, ll)
        except Exception:
            continue
    return best  # (bic, k, model, loglik)


def self_transition_mean(transmat):
    return float(np.mean(np.diag(transmat)))


def shuffle_order_within_episode(X, episode, lengths, rng):
    offsets = np.cumsum([0] + lengths)
    X_shuf = X.copy()
    for i in range(len(lengths)):
        sl = slice(offsets[i], offsets[i + 1])
        perm = rng.permutation(lengths[i])
        X_shuf[sl] = X[sl][perm]
    return X_shuf


def analyze_task(collect_dir, task, fnames_by_seed, n_null=20):
    result = {"task": task, "per_seed_series": {}}
    for seed_series, fname in fnames_by_seed.items():
        d = load_task_seed_data(collect_dir, fname, LAYER, K_STEP)
        Xr = scene_residualize(d["feats"], d["episode"])
        X_seq, episode_seq, phase_seq, lengths = order_by_episode_call(
            Xr, d["episode"], d["call_idx"], d["phase_label"]
        )
        # reduce dim for stable HMM/PELT fitting
        Xs = StandardScaler().fit_transform(X_seq)
        Xp = PCA(n_components=min(10, Xs.shape[0] - 1, Xs.shape[1]), random_state=0).fit_transform(Xs)

        cps_per_ep = pelt_changepoints(Xp, lengths)
        mean_seg_len = float(np.mean([L / max(c + 1, 1) for L, c in zip(lengths, cps_per_ep)]))

        bic, k_hat, model, ll = fit_hmm_bic(Xp, lengths, seed=0)
        states = model.predict(Xp, lengths)
        stickiness = self_transition_mean(model.transmat_)
        dwell_time = float(1.0 / (1.0 - stickiness + 1e-9))

        mi_phase = normalized_mutual_info_score(states, phase_seq)
        mi_scene = normalized_mutual_info_score(states, episode_seq)

        rng = np.random.RandomState(3)
        null_stickiness = []
        for i in range(n_null):
            Xp_shuf = shuffle_order_within_episode(Xp, episode_seq, lengths, rng)
            try:
                m = GaussianHMM(n_components=k_hat, covariance_type="diag", n_iter=50,
                                 random_state=i).fit(Xp_shuf, lengths)
                null_stickiness.append(self_transition_mean(m.transmat_))
            except Exception:
                continue
        null_stickiness = np.array(null_stickiness)
        p_val = float((np.sum(null_stickiness >= stickiness) + 1) / (len(null_stickiness) + 1)) if len(null_stickiness) else 1.0

        seed_result = {
            "n_episodes": len(lengths),
            "mean_episode_len_calls": float(np.mean(lengths)),
            "pelt_mean_changepoints_per_episode": float(np.mean(cps_per_ep)),
            "pelt_mean_segment_len_calls": mean_seg_len,
            "hmm_k_bic_selected": k_hat,
            "hmm_transition_matrix": model.transmat_.tolist(),
            "hmm_self_transition_mean": stickiness,
            "hmm_dwell_time_calls": dwell_time,
            "mi_state_vs_phase": float(mi_phase),
            "mi_state_vs_scene": float(mi_scene),
            "null_order_shuffle_stickiness_mean": float(null_stickiness.mean()) if len(null_stickiness) else None,
            "null_order_shuffle_stickiness_std": float(null_stickiness.std()) if len(null_stickiness) else None,
            "p_value_real_stickiness_gt_null": p_val,
            "structured_transitions_supported": bool(p_val < 0.05 and mi_phase > mi_scene),
        }
        result["per_seed_series"][str(seed_series)] = seed_result
        log_message(
            f"[rollout_transition {task} seed={seed_series}] K={k_hat} stickiness={stickiness:.3f} "
            f"(null={seed_result['null_order_shuffle_stickiness_mean']:.3f}±"
            f"{seed_result['null_order_shuffle_stickiness_std']:.3f}, p={p_val:.3f}) "
            f"MI(phase)={mi_phase:.3f} MI(scene)={mi_scene:.3f} "
            f"PELT_segs/ep={seed_result['pelt_mean_changepoints_per_episode']+1:.1f}"
        )
    return result


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--collect_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_null", type=int, default=20)
    args = p.parse_args()

    collect_dir = Path(args.collect_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((collect_dir / "multitask_manifest.json").read_text())

    files_by_task = {}
    for fname, info in manifest["files"].items():
        files_by_task.setdefault(info["task"], {})[info["seed_base"]] = fname

    results = {}
    for task, fnames_by_seed in files_by_task.items():
        results[task] = analyze_task(collect_dir, task, fnames_by_seed, args.n_null)

    with open(out_dir / "rollout_transition.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'rollout_transition.json'}")


if __name__ == "__main__":
    main()
