"""
skill_count.py — attractor_verification_design.md §6.1 準拠

教師なしクラスタリングバッテリ (DP-GMM / HDBSCAN / k-means+silhouette / スペクトラル eigengap) を
シーン統制の下で実行し、K (スキル/フェーズ数) を推定する。

Level 1 (フェーズ, タスク内): エピソード(=シーン)平均を引いた残差特徴 (Z - Z̄_episode) を
  入力にする (design §2 統制2「シーン残差化」)。
  Null: エピソードIDをcall間でシャッフルしてから残差化した「シーンシャッフルnull」。
  採用基準: (a) ≥3手法でK一致, (b) 実測Kがシーンnullの分布を有意に超える,
            (c) クラスタ↔phase MI 高 / クラスタ↔episode(scene) MI 低,
            (d) 2 seed系列でK安定。

Level 2 (タスク間, 真に異種のスキル): 生特徴 (残差化しない、タスク間差を残すため) を入力に
  プールしたタスク間クラスタリング。Null: タスクラベルをepisode単位でシャッフル。
"""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import hdbscan

from cosmos_policy.experiments.robot.robot_utils import log_message

LAYER = 13
K_STEP = 4
MAX_K = 8


def load_task_seed_data(collect_dir: Path, fname: str, layer: int, k: int):
    feat_path = collect_dir / fname
    phase_path = collect_dir / fname.replace(".npz", "_phases.npz")
    fd = np.load(feat_path)
    pd = np.load(phase_path)
    key = f"feat_k{k}_layer{layer}"
    idx_key = key + "_idx"
    feats = fd[key]
    keep_idx = fd[idx_key]
    return {
        "feats": feats,
        "episode": fd["episode"][keep_idx],
        "call_idx": fd["call_idx"][keep_idx],
        "phase_label": pd["phase_label"][keep_idx],
        "task_idx": fd["task_idx"][keep_idx],
    }


def scene_residualize(X, episode):
    Xr = X.copy()
    for e in np.unique(episode):
        mask = episode == e
        Xr[mask] = X[mask] - X[mask].mean(axis=0, keepdims=True)
    return Xr


def preprocess(X, pca_var=0.95, seed=0):
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=pca_var, svd_solver="full", random_state=seed)
    return pca.fit_transform(Xs)


def spectral_eigengap_k(X, n_neighbors=10, max_k=MAX_K):
    n = X.shape[0]
    n_neighbors = min(n_neighbors, n - 1)
    A = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity", include_self=False)
    A = 0.5 * (A + A.T)
    A = (A > 0).astype(float)
    deg = np.asarray(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1e-12
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    L = np.eye(n) - D_inv_sqrt @ A.toarray() @ D_inv_sqrt
    eigvals = np.sort(np.linalg.eigvalsh(L))
    m = min(max_k + 2, n)
    gaps = np.diff(eigvals[:m])
    k_hat = int(np.argmax(gaps[1:m - 1]) + 2) if m > 3 else 1
    k_hat = max(1, min(k_hat, max_k))
    return k_hat, eigvals[:m].tolist()


def kmeans_silhouette_k(X, max_k=MAX_K, seed=0):
    best_k, best_score = 1, -1.0
    scores = {}
    for k in range(2, min(max_k, X.shape[0] - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        if len(np.unique(km.labels_)) < 2:
            continue
        s = silhouette_score(X, km.labels_)
        scores[k] = float(s)
        if s > best_score:
            best_score, best_k = s, k
    return best_k, scores


def dpgmm_k(X, max_k=MAX_K, seed=0, weight_thresh=0.05):
    dpgmm = BayesianGaussianMixture(
        n_components=max_k, weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1e-2, max_iter=300, random_state=seed,
    ).fit(X)
    labels = dpgmm.predict(X)
    active = np.unique(labels)
    weights = dpgmm.weights_[active]
    k_hat = int((weights > weight_thresh).sum())
    return max(k_hat, 1), labels


def hdbscan_k(X, min_cluster_size=None):
    if min_cluster_size is None:
        min_cluster_size = max(5, X.shape[0] // 20)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(X)
    k_hat = len(set(labels)) - (1 if -1 in labels else 0)
    return max(k_hat, 0), labels


def run_battery(X, seed=0):
    """Returns dict of method -> K_hat, plus best labels for MI (kmeans at consensus K)."""
    k_km, sil_scores = kmeans_silhouette_k(X, seed=seed)
    k_dpgmm, dpgmm_labels = dpgmm_k(X, seed=seed)
    k_hdb, hdb_labels = hdbscan_k(X)
    k_spec, eigvals = spectral_eigengap_k(X)
    return {
        "kmeans_silhouette": k_km,
        "dpgmm": k_dpgmm,
        "hdbscan": k_hdb,
        "spectral_eigengap": k_spec,
        "kmeans_silhouette_scores": sil_scores,
        "dpgmm_labels": dpgmm_labels,
        "kmeans_labels_at_consensus": KMeans(n_clusters=max(k_km, 2), n_init=10, random_state=seed).fit_predict(X),
    }


def consensus_k(battery: dict):
    ks = [battery["kmeans_silhouette"], battery["dpgmm"], battery["hdbscan"], battery["spectral_eigengap"]]
    vals, counts = np.unique(ks, return_counts=True)
    order = np.argsort(-counts)
    top_k, top_count = int(vals[order[0]]), int(counts[order[0]])
    return top_k, top_count, ks


def scene_shuffle_null_k(X_raw, episode, n_shuffles, seed=0):
    """Shuffle episode assignment among calls, re-residualize, re-run battery, collect consensus K."""
    rng = np.random.RandomState(seed)
    null_ks = []
    for i in range(n_shuffles):
        ep_shuffled = rng.permutation(episode)
        Xr_null = scene_residualize(X_raw, ep_shuffled)
        Xp_null = preprocess(Xr_null, seed=seed + i)
        battery_null = run_battery(Xp_null, seed=seed + i)
        top_k, _, _ = consensus_k(battery_null)
        null_ks.append(top_k)
    return null_ks


def level1_task_analysis(collect_dir, manifest, task, fnames_by_seed, n_null=20):
    result = {"task": task, "per_seed_series": {}}
    consensus_by_seed = {}

    for seed_series, fname in fnames_by_seed.items():
        d = load_task_seed_data(collect_dir, fname, LAYER, K_STEP)
        X_raw, episode, phase_label = d["feats"], d["episode"], d["phase_label"]

        Xr = scene_residualize(X_raw, episode)
        Xp = preprocess(Xr, seed=0)

        battery = run_battery(Xp, seed=0)
        top_k, top_count, ks = consensus_k(battery)

        cluster_labels = battery["kmeans_labels_at_consensus"]
        mi_phase = normalized_mutual_info_score(cluster_labels, phase_label)
        mi_scene = normalized_mutual_info_score(cluster_labels, episode)

        null_ks = scene_shuffle_null_k(X_raw, episode, n_null, seed=1)
        null_ks_arr = np.array(null_ks)
        p_val = float((np.sum(null_ks_arr >= top_k) + 1) / (len(null_ks_arr) + 1))

        seed_result = {
            "n_calls": int(len(episode)),
            "n_episodes": int(len(np.unique(episode))),
            "battery_k": {"kmeans_silhouette": battery["kmeans_silhouette"],
                          "dpgmm": battery["dpgmm"], "hdbscan": battery["hdbscan"],
                          "spectral_eigengap": battery["spectral_eigengap"]},
            "consensus_k": top_k,
            "consensus_agreement_count": top_count,
            "criterion_a_3plus_agree": bool(top_count >= 3),
            "mi_cluster_vs_phase": float(mi_phase),
            "mi_cluster_vs_scene": float(mi_scene),
            "criterion_c_phase_gt_scene_mi": bool(mi_phase > mi_scene),
            "scene_shuffle_null_k_distribution": null_ks,
            "scene_shuffle_null_p_value": p_val,
            "criterion_b_exceeds_null": bool(p_val < 0.05),
        }
        result["per_seed_series"][str(seed_series)] = seed_result
        consensus_by_seed[seed_series] = top_k
        log_message(
            f"[Level1 {task} seed_series={seed_series}] battery={seed_result['battery_k']} "
            f"consensus_K={top_k} (agree={top_count}/4) MI(phase)={mi_phase:.3f} "
            f"MI(scene)={mi_scene:.3f} null_p={p_val:.3f}"
        )

    ks_across_seeds = list(consensus_by_seed.values())
    result["criterion_d_seed_stable"] = bool(len(set(ks_across_seeds)) == 1) if ks_across_seeds else False
    result["consensus_k_by_seed_series"] = {str(k): v for k, v in consensus_by_seed.items()}

    all_pass = (
        all(v["criterion_a_3plus_agree"] for v in result["per_seed_series"].values())
        and all(v["criterion_b_exceeds_null"] for v in result["per_seed_series"].values())
        and all(v["criterion_c_phase_gt_scene_mi"] for v in result["per_seed_series"].values())
        and result["criterion_d_seed_stable"]
    )
    result["all_criteria_pass"] = all_pass
    result["verdict"] = (
        f"モデルは Level1 (フェーズ, {task}) について K={ks_across_seeds[0] if ks_across_seeds else '?'} 個の"
        "安定したクラスタを形成 (a-d全基準を満たす)。"
        if all_pass else
        f"Level1 (フェーズ, {task}) はクラスタリング基準の少なくとも1つを満たさなかった。"
        "スキル/フェーズアトラクタの主張根拠として採用しない。"
    )
    return result


def level2_cross_task_analysis(collect_dir, manifest, n_null=20):
    Xs, eps, tasks_ = [], [], []
    ep_offset = 0
    for fname in sorted(manifest["files"].keys()):
        d = load_task_seed_data(collect_dir, fname, LAYER, K_STEP)
        episode_local = d["episode"] + ep_offset
        ep_offset += d["episode"].max() + 1 if len(d["episode"]) else 0
        Xs.append(d["feats"])
        eps.append(episode_local)
        tasks_.append(d["task_idx"])
    X_raw = np.concatenate(Xs)
    episode = np.concatenate(eps)
    task_idx = np.concatenate(tasks_)

    Xp = preprocess(X_raw, seed=0)  # NOT residualized: task-level between-scene signal is the target
    battery = run_battery(Xp, seed=0)
    top_k, top_count, ks = consensus_k(battery)
    cluster_labels = battery["kmeans_labels_at_consensus"]
    mi_task = normalized_mutual_info_score(cluster_labels, task_idx)
    mi_scene = normalized_mutual_info_score(cluster_labels, episode)

    # null: shuffle task label at the episode level (not call level, since task is constant per episode)
    rng = np.random.RandomState(2)
    ep_to_task = {}
    for e, t in zip(episode, task_idx):
        ep_to_task.setdefault(e, t)
    uniq_eps = np.array(list(ep_to_task.keys()))
    null_ks = []
    for i in range(n_null):
        shuffled_task_vals = rng.permutation(list(ep_to_task.values()))
        ep_to_task_shuffled = dict(zip(uniq_eps, shuffled_task_vals))
        task_null = np.array([ep_to_task_shuffled[e] for e in episode])
        mi_null = normalized_mutual_info_score(cluster_labels, task_null)
        null_ks.append(mi_null)
    null_ks = np.array(null_ks)
    p_val = float((np.sum(null_ks >= mi_task) + 1) / (len(null_ks) + 1))

    result = {
        "battery_k": {"kmeans_silhouette": battery["kmeans_silhouette"], "dpgmm": battery["dpgmm"],
                      "hdbscan": battery["hdbscan"], "spectral_eigengap": battery["spectral_eigengap"]},
        "consensus_k": top_k,
        "consensus_agreement_count": top_count,
        "n_true_tasks": int(len(np.unique(task_idx))),
        "mi_cluster_vs_task": float(mi_task),
        "mi_cluster_vs_scene": float(mi_scene),
        "task_shuffle_null_mi_mean": float(null_ks.mean()),
        "task_shuffle_null_p_value": p_val,
        "note": (
            "Level2はタスクが変われば視点・キッチン領域・言語指示が全て変わるため、"
            "分離できることは真に異種スキル間の分離の弱い証拠に留まる"
            "(視覚的シーン差との交絡を否定できない、サニティチェックとして解釈)。"
        ),
    }
    log_message(
        f"[Level2 cross-task] battery={result['battery_k']} consensus_K={top_k} "
        f"(true n_tasks={result['n_true_tasks']}) MI(task)={mi_task:.3f} "
        f"MI(scene)={mi_scene:.3f} null_p={p_val:.3f}"
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

    level1_results = {}
    for task, fnames_by_seed in files_by_task.items():
        level1_results[task] = level1_task_analysis(collect_dir, manifest, task, fnames_by_seed, args.n_null)

    level2_result = level2_cross_task_analysis(collect_dir, manifest, args.n_null)

    out = {"layer": LAYER, "k_step": K_STEP, "level1": level1_results, "level2": level2_result}
    with open(out_dir / "skill_count.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log_message(f"Saved: {out_dir / 'skill_count.json'}")


if __name__ == "__main__":
    main()
