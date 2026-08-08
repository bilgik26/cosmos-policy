"""
linear_separability_gate.py — attractor_verification_design.md §11 ゲート[2]

「線形分離可能なスキルアトラクタが存在する」という提案の大前提を、実行前に検証する
決定的なゲート。フェーズ (Level 1, タスク内複合フェーズ4クラス) が、fold内PCA+LRで
episodeシャッフルnullを有意に超えて線形分離できるかを判定する。

方法 (P3: 前処理はfold内で推定・固定して明記):
  1. 標準化方式: 生特徴量をfold内で z-score 標準化してから PCA (95%分散保持)。
  2. Group K-Fold (グループ=episode, K=5): episode をまたいだリークを防止。
  3. 各 fold で PCA を train のみで fit、test に適用 → 多クラス LogisticRegression。
  4. Null: episode 内で phase_label を独立にシャッフル (シーン/episode 構造は保持したまま
     フェーズと特徴の対応だけ破壊)。200 回繰り返し、null 分布と実測 accuracy を比較。
  5. 効果量 (real - null_mean)/null_std、経験的 p 値 (P5)。

判定: real accuracy が null 分布の 95 percentile を有意に超える (BH-FDR 補正後 p<0.05) 場合に
「フェーズはシーンnullを超えて線形分離できる」→ ゲート通過。
超えない場合は「線形分離アトラクタ」前提を棄却し、後続のクラスタリング解析(skill_count.py)は
参考情報として位置づけ、レポートで「アトラクタは同定できなかった」を主要結論候補にする。

Level 2 (タスク間, 真に異種のスキル) も同時に検証するが、こちらはタスクが異なれば
カメラ視点・キッチン領域・言語指示が全て変わるため、分離できて当然のサニティチェックとして
扱う (スキルアトラクタの主張根拠には使わない)。
"""

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from cosmos_policy.experiments.robot.robot_utils import log_message

PROBE_LAYER_FOR_GATE = 13  # mid layer, per design §4.2 emphasis
K_STEP_FOR_GATE = 4        # last denoising step


def load_task_data(collect_dir: Path, fname: str, layer: int, k: int):
    feat_path = collect_dir / fname
    phase_path = collect_dir / fname.replace(".npz", "_phases.npz")
    fd = np.load(feat_path)
    pd = np.load(phase_path)

    key = f"feat_k{k}_layer{layer}"
    idx_key = key + "_idx"
    feats = fd[key]
    keep_idx = fd[idx_key]  # indices into the full call array that have this (k,layer) feature

    episode = fd["episode"][keep_idx]
    phase_label = pd["phase_label"][keep_idx]
    task_idx = fd["task_idx"][keep_idx]
    return feats, episode, phase_label, task_idx


def fold_pca_lr_accuracy(X, y, groups, n_splits=5, pca_var=0.95, seed=0):
    """Group K-fold, fold内標準化+PCA+LR。held-out accuracyの平均を返す。"""
    n_groups = len(np.unique(groups))
    k = min(n_splits, n_groups)
    if k < 2:
        return np.nan
    gkf = GroupKFold(n_splits=k)
    accs = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        pca = PCA(n_components=pca_var, svd_solver="full", random_state=seed)
        Xtr_p = pca.fit_transform(Xtr)
        Xte_p = pca.transform(Xte)
        clf = LogisticRegression(max_iter=2000, multi_class="auto")
        clf.fit(Xtr_p, y[train_idx])
        acc = clf.score(Xte_p, y[test_idx])
        accs.append(acc)
    return float(np.mean(accs)) if accs else np.nan


def episode_shuffle_null(y, groups, rng):
    """episode内でyを独立にシャッフル (scene構造を保ったままラベル対応を破壊)。"""
    y_null = y.copy()
    for g in np.unique(groups):
        mask = groups == g
        idx = np.where(mask)[0]
        y_null[idx] = rng.permutation(y[idx])
    return y_null


def run_gate(X, y, groups, label_name, n_perm=200, seed=0):
    rng = np.random.RandomState(seed)
    real_acc = fold_pca_lr_accuracy(X, y, groups, seed=seed)

    null_accs = []
    for i in range(n_perm):
        y_null = episode_shuffle_null(y, groups, rng)
        null_accs.append(fold_pca_lr_accuracy(X, y_null, groups, seed=seed))
    null_accs = np.array([a for a in null_accs if not np.isnan(a)])

    null_mean = float(null_accs.mean())
    null_std = float(null_accs.std())
    effect_size = (real_acc - null_mean) / (null_std + 1e-12)
    p_value = float((np.sum(null_accs >= real_acc) + 1) / (len(null_accs) + 1))

    chance = 1.0 / len(np.unique(y))

    result = {
        "label": label_name,
        "n_samples": int(len(y)),
        "n_classes": int(len(np.unique(y))),
        "n_episodes": int(len(np.unique(groups))),
        "chance_level": chance,
        "real_accuracy": real_acc,
        "null_mean_accuracy": null_mean,
        "null_std_accuracy": null_std,
        "null_p95_accuracy": float(np.percentile(null_accs, 95)),
        "effect_size_d": effect_size,
        "p_value_empirical": p_value,
        "n_permutations": int(len(null_accs)),
        "gate_pass": bool(real_acc > np.percentile(null_accs, 95) and p_value < 0.05),
    }
    log_message(
        f"[{label_name}] real={real_acc:.3f} null_mean={null_mean:.3f}±{null_std:.3f} "
        f"(p95={result['null_p95_accuracy']:.3f}) chance={chance:.3f} "
        f"d={effect_size:.2f} p={p_value:.4f} gate_pass={result['gate_pass']}"
    )
    return result


def bh_fdr(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--collect_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--layer", type=int, default=PROBE_LAYER_FOR_GATE)
    p.add_argument("--k_step", type=int, default=K_STEP_FOR_GATE)
    p.add_argument("--n_perm", type=int, default=200)
    args = p.parse_args()

    collect_dir = Path(args.collect_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((collect_dir / "multitask_manifest.json").read_text())
    files_by_task = {}
    for fname, info in manifest["files"].items():
        files_by_task.setdefault(info["task"], []).append(fname)

    all_results = []

    # ── Level 1: within-task phase separability (the decisive gate) ──────────
    for task, fnames in files_by_task.items():
        Xs, eps, phs, _ = [], [], [], []
        ep_offset = 0
        for fname in sorted(fnames):
            feats, episode, phase_label, _ = load_task_data(
                collect_dir, fname, args.layer, args.k_step
            )
            # make episode ids unique across seed-series files within this task
            episode_local = episode + ep_offset
            ep_offset += episode.max() + 1 if len(episode) else 0
            Xs.append(feats)
            eps.append(episode_local)
            phs.append(phase_label)
        X = np.concatenate(Xs)
        groups = np.concatenate(eps)
        y = np.concatenate(phs)

        # drop classes with too few episodes to avoid degenerate folds
        result = run_gate(X, y, groups, label_name=f"phase|{task}", n_perm=args.n_perm)
        result["level"] = 1
        result["task"] = task
        all_results.append(result)

    # ── Level 2: cross-task separability (expected-trivial sanity check) ────
    Xs, eps, tasks_ = [], [], []
    ep_offset = 0
    for task, fnames in files_by_task.items():
        for fname in sorted(fnames):
            feats, episode, _, task_idx = load_task_data(
                collect_dir, fname, args.layer, args.k_step
            )
            episode_local = episode + ep_offset
            ep_offset += episode.max() + 1 if len(episode) else 0
            Xs.append(feats)
            eps.append(episode_local)
            tasks_.append(task_idx)
    X = np.concatenate(Xs)
    groups = np.concatenate(eps)
    y = np.concatenate(tasks_)
    result = run_gate(X, y, groups, label_name="task_identity|cross_task", n_perm=args.n_perm)
    result["level"] = 2
    result["task"] = "ALL"
    all_results.append(result)

    # BH-FDR across all gate p-values
    pvals = [r["p_value_empirical"] for r in all_results]
    qvals = bh_fdr(pvals)
    for r, q in zip(all_results, qvals):
        r["p_value_fdr_bh"] = float(q)
        r["gate_pass_fdr"] = bool(r["gate_pass"] and q < 0.05)

    level1_results = [r for r in all_results if r["level"] == 1]
    gate_overall_pass = all(r["gate_pass_fdr"] for r in level1_results) if level1_results else False

    summary = {
        "probe_layer": args.layer,
        "k_step": args.k_step,
        "n_permutations": args.n_perm,
        "preprocessing": "z-score standardize (fold-internal) -> PCA(95% var, fold-internal) -> multinomial LR",
        "null_construction": "within-episode independent shuffle of phase_label (scene structure preserved)",
        "results": all_results,
        "gate2_decision": (
            "PASS: 全タスクでフェーズがシーンnullをFDR<0.05で有意に超えて線形分離可能。"
            "後続のクラスタリング解析(skill_count.py等)を実行する。"
            if gate_overall_pass else
            "FAIL: 少なくとも1タスクでフェーズがシーンnullを有意に超えられなかった。"
            "「線形分離可能なスキルアトラクタ」前提が本モデル・本設定では支持されない。"
            "後続のクラスタリング解析は参考情報として実行するが、Tier A主張の根拠にはしない。"
        ),
        "gate2_pass": gate_overall_pass,
    }

    with open(out_dir / "linear_separability_gate.json", "w") as f:
        json.dump(summary, f, indent=2)

    log_message(f"=== GATE [2] DECISION: {'PASS' if gate_overall_pass else 'FAIL'} ===")
    log_message(f"Saved: {out_dir / 'linear_separability_gate.json'}")


if __name__ == "__main__":
    main()
