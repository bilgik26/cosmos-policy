"""
v4改訂: 3seed(195,196,197)x50episode収集データを、成功episodeのみに
フィルタしてマージするスクリプト。

verification_report_v3.md の改訂（2026-08-06、3seed成功epのみへのデータ統一）で
使用する、以下5種類の収集データをそれぞれマージする:
  - feature    (feature_analysis.py出力):      action DiT特徴量
  - mechanism  (mechanism_analysis.py出力):    デコード済み行動チャンク軌跡
  - image      (image_analysis.py出力):        画像ラテント特徴量
  - attn       (attention_analysis_v2.py出力): 自己注意T-matrix
  - crossattn  (crossattn_analysis.py出力):    クロスアテンション重み

各episodeのグローバルID = seed*1000 + ep_idx_within_seed（seed跨ぎの衝突回避）。

実行:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.v4_merge \
      --collect_dir results/v4_collect --out_dir results/v4_merged
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

SEEDS = [195, 196, 197]


def global_id(seed: int, ep_idx: int) -> int:
    return seed * 1000 + int(ep_idx)


def merge_feature_like(
    collect_dir: Path,
    subdir: str,
    npz_filename: str,
    out_path: Path,
    array_keys_predicate,
):
    """
    feature_analysis / crossattn_analysis 共通パターン:
    npz に episode_labels, call_idx_labels, episode_success_index/flag と
    複数の (N_calls, ...) 配列があり、call毎にepisode_labelsでフィルタする。
    """
    merged: Dict[str, List[np.ndarray]] = {}
    all_episode_labels = []
    all_call_idx_labels = []
    all_success_index = []
    all_success_flag = []
    meta = {"seeds_used": [], "n_success_per_seed": {}, "n_raw_per_seed": {}}

    for seed in SEEDS:
        p = collect_dir / subdir / f"seed{seed}" / npz_filename
        if not p.exists():
            print(f"  [skip] {p} not found")
            continue
        d = np.load(p, allow_pickle=False)
        if "episode_success_index" not in d.files:
            print(f"  [warn] {p} has no episode_success_index, skipping")
            continue
        ep_labels = d["episode_labels"]
        success_index = d["episode_success_index"]
        success_flag = d["episode_success_flag"]
        success_set = {int(e) for e, ok in zip(success_index, success_flag) if ok}
        n_raw = len(success_index)
        n_succ = len(success_set)
        meta["seeds_used"].append(seed)
        meta["n_raw_per_seed"][seed] = n_raw
        meta["n_success_per_seed"][seed] = n_succ
        print(f"  seed={seed}: {n_succ}/{n_raw} episodes successful")

        call_mask = np.array([int(e) in success_set for e in ep_labels])
        n_calls_kept = int(call_mask.sum())
        print(f"    -> {n_calls_kept}/{len(ep_labels)} calls kept")

        gids = np.array([global_id(seed, e) for e in ep_labels[call_mask]])
        all_episode_labels.append(gids)
        if "call_idx_labels" in d.files:
            all_call_idx_labels.append(d["call_idx_labels"][call_mask])
        for e in success_index[np.isin(success_index, list(success_set))]:
            all_success_index.append(global_id(seed, e))
            all_success_flag.append(True)

        for key in d.files:
            if key in ("episode_labels", "call_idx_labels", "episode_success_index", "episode_success_flag"):
                continue
            if not array_keys_predicate(key):
                continue
            arr = d[key]
            if arr.ndim >= 1 and arr.shape[0] == len(ep_labels):
                merged.setdefault(key, []).append(arr[call_mask])
            else:
                # not a per-call array (e.g. scalar meta) -- keep from first seed only
                merged.setdefault(key, []).append(arr)

    out = {}
    for key, parts in merged.items():
        try:
            out[key] = np.concatenate(parts, axis=0)
        except ValueError:
            out[key] = parts[0]  # fallback: non-per-call scalar/meta array
    out["episode_labels"] = np.concatenate(all_episode_labels) if all_episode_labels else np.array([])
    if all_call_idx_labels:
        out["call_idx_labels"] = np.concatenate(all_call_idx_labels)
    out["episode_success_index"] = np.array(all_success_index)
    out["episode_success_flag"] = np.array(all_success_flag)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=int)
    print(f"Saved: {out_path}  (N_calls={len(out['episode_labels'])}, "
          f"N_success_episodes={len(out['episode_success_index'])})")
    return meta


def merge_step_actions(collect_dir: Path, out_path: Path):
    """mechanism_analysis.py 出力 step_actions.npz のマージ。
    keys '0'-'4' は (N_calls, 32, 7)、episode_labels_k{k}/call_idx_labels_k{k} で
    call毎のepisode所属がわかる（kごとに有効call数が微妙に違う可能性があるため、
    kごとに独立してフィルタする）。"""
    merged_by_k: Dict[str, List[np.ndarray]] = {str(k): [] for k in range(5)}
    ep_labels_by_k: Dict[str, List[np.ndarray]] = {str(k): [] for k in range(5)}
    ci_labels_by_k: Dict[str, List[np.ndarray]] = {str(k): [] for k in range(5)}
    norm_by_k: Dict[int, List[np.ndarray]] = {k: [] for k in range(5)}
    sigma_by_k: Dict[int, List[np.ndarray]] = {k: [] for k in range(5)}
    norm_ep_by_k: Dict[int, List[np.ndarray]] = {k: [] for k in range(5)}
    meta = {"seeds_used": [], "n_success_per_seed": {}, "n_raw_per_seed": {}}

    for seed in SEEDS:
        p = collect_dir / "mechanism" / f"seed{seed}" / "step_actions.npz"
        if not p.exists():
            print(f"  [skip] {p} not found")
            continue
        d = np.load(p)
        if "episode_success_index" not in d.files:
            print(f"  [warn] {p} has no episode_success_index, skipping")
            continue
        success_index = d["episode_success_index"]
        success_flag = d["episode_success_flag"]
        success_set = {int(e) for e, ok in zip(success_index, success_flag) if ok}
        meta["seeds_used"].append(seed)
        meta["n_raw_per_seed"][seed] = len(success_index)
        meta["n_success_per_seed"][seed] = len(success_set)
        print(f"  seed={seed}: {len(success_set)}/{len(success_index)} episodes successful")

        for k in range(5):
            ks = str(k)
            if ks not in d.files or f"episode_labels_{k}" not in d.files:
                continue
            acts = d[ks]
            ep_lab = d[f"episode_labels_{k}"]
            mask = np.array([int(e) in success_set for e in ep_lab])
            merged_by_k[ks].append(acts[mask])
            ep_labels_by_k[ks].append(np.array([global_id(seed, e) for e in ep_lab[mask]]))
            if f"call_idx_labels_{k}" in d.files:
                ci_labels_by_k[ks].append(d[f"call_idx_labels_{k}"][mask])

            # §2.4 (noise_pred_norm/F_theta) 用: norm_{k}/sigma_{k} は
            # act抽出成否によらない別母集団(norm_episode_labels_{k})を持つため
            # 独立にフィルタする。
            if f"norm_{k}" in d.files and f"norm_episode_labels_{k}" in d.files:
                norm_ep_lab = d[f"norm_episode_labels_{k}"]
                norm_mask = np.array([int(e) in success_set for e in norm_ep_lab])
                norm_by_k[k].append(d[f"norm_{k}"][norm_mask])
                sigma_by_k[k].append(d[f"sigma_{k}"][norm_mask])
                norm_ep_by_k[k].append(np.array([global_id(seed, e) for e in norm_ep_lab[norm_mask]]))

    out = {}
    for k in range(5):
        ks = str(k)
        if merged_by_k[ks]:
            out[ks] = np.concatenate(merged_by_k[ks], axis=0)
            out[f"episode_labels_{k}"] = np.concatenate(ep_labels_by_k[ks], axis=0)
            if ci_labels_by_k[ks]:
                out[f"call_idx_labels_{k}"] = np.concatenate(ci_labels_by_k[ks], axis=0)
            print(f"  k={k}: N={out[ks].shape[0]} merged action chunks")
        if norm_by_k[k]:
            out[f"norm_{k}"] = np.concatenate(norm_by_k[k], axis=0)
            out[f"sigma_{k}"] = np.concatenate(sigma_by_k[k], axis=0)
            out[f"norm_episode_labels_{k}"] = np.concatenate(norm_ep_by_k[k], axis=0)
            print(f"  k={k}: N={out[f'norm_{k}'].shape[0]} merged norm/sigma records")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=int)
    print(f"Saved: {out_path}")
    return meta


def merge_attn_tmatrices(collect_dir: Path, out_path: Path):
    """attention_analysis_v2.py 出力 per_episode_t_matrices.npz のマージ。
    成功epのみをプールし、episode単位で均等加重平均する
    （2段階平均: call平均[既にepisode毎に平均済み] -> episode平均、
    call数の多いepisodeに引きずられないようにする設計をseedを跨いでも維持）。
    """
    per_ep_mats: Dict = {}  # (block,k) -> list of per-episode-avg matrices
    meta = {"seeds_used": [], "n_success_per_seed": {}, "n_raw_per_seed": {}, "n_success_episodes_pooled": 0}
    t1_all_passed = True
    t2_all_passed = True

    for seed in SEEDS:
        p = collect_dir / "attn" / f"seed{seed}" / "per_episode_t_matrices.npz"
        meta_p = collect_dir / "attn" / f"seed{seed}" / "attn_meta_v2.json"
        if not p.exists():
            print(f"  [skip] {p} not found")
            continue
        d = np.load(p)
        success_index = d["episode_success_index"]
        success_flag = d["episode_success_flag"]
        success_eps = [int(e) for e, ok in zip(success_index, success_flag) if ok]
        meta["seeds_used"].append(seed)
        meta["n_raw_per_seed"][seed] = len(success_index)
        meta["n_success_per_seed"][seed] = len(success_eps)
        meta["n_success_episodes_pooled"] += len(success_eps)
        print(f"  seed={seed}: {len(success_eps)}/{len(success_index)} episodes successful")

        if meta_p.exists():
            with open(meta_p) as f:
                jm = json.load(f)
            t1_all_passed = t1_all_passed and jm.get("t1_passed", False)
            t2_all_passed = t2_all_passed and jm.get("t2_passed", False)

        for ep in success_eps:
            for key in d.files:
                if not key.startswith(f"tmat_ep{ep}_block"):
                    continue
                # key = tmat_ep{ep}_block{b}_k{k}
                rest = key[len(f"tmat_ep{ep}_"):]  # block{b}_k{k}
                block_str, k_str = rest.split("_k")
                block_idx = int(block_str.replace("block", ""))
                k_step = int(k_str)
                per_ep_mats.setdefault((block_idx, k_step), []).append(d[key])

    out = {}
    for (block_idx, k_step), mats in per_ep_mats.items():
        out[f"tmat_block{block_idx}_k{k_step}"] = np.mean(np.stack(mats, axis=0), axis=0)
        out[f"tmat_block{block_idx}_k{k_step}_n_episodes"] = np.array([len(mats)])
    meta["t1_all_passed"] = t1_all_passed
    meta["t2_all_passed"] = t2_all_passed

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=int)
    print(f"Saved: {out_path}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect_dir", default="cosmos_policy/experiments/robot/robocasa/analysis/results/v4_collect")
    ap.add_argument("--out_dir", default="cosmos_policy/experiments/robot/robocasa/analysis/results/v4_merged")
    args = ap.parse_args()
    collect_dir = Path(args.collect_dir)
    out_dir = Path(args.out_dir)

    print("=== Merging feature_analysis (action DiT features) ===")
    merge_feature_like(
        collect_dir, "feature", "features.npz", out_dir / "action_features.npz",
        array_keys_predicate=lambda k: k.startswith("feat_"),
    )

    print("\n=== Merging mechanism_analysis (decoded action chunks) ===")
    merge_step_actions(collect_dir, out_dir / "step_actions.npz")

    print("\n=== Merging image_analysis (image latent features) ===")
    merge_feature_like(
        collect_dir, "image", "image_features.npz", out_dir / "image_features.npz",
        array_keys_predicate=lambda k: (
            k.startswith("norm_") or k.startswith("delta_norm_") or k.startswith("fft_")
        ),
    )
    # hidfeat_* と sigma_data_* は別ラベル系列(hidden_episode_labels)なので個別処理
    merge_image_hidfeat(collect_dir, out_dir / "image_hidfeat.npz")
    merge_image_sigma_data(collect_dir, out_dir / "image_sigma_data.npz")

    print("\n=== Merging attention_analysis_v2 (self-attention T-matrices) ===")
    merge_attn_tmatrices(collect_dir, out_dir / "attn_tmatrices.npz")

    print("\n=== Merging crossattn_analysis (cross-attention weights) ===")
    merge_feature_like(
        collect_dir, "crossattn", "crossattn.npz", out_dir / "crossattn.npz",
        array_keys_predicate=lambda k: k.startswith("attn_layer"),
    )

    print("\nAll merges complete.")


def merge_image_hidfeat(collect_dir: Path, out_path: Path):
    merged: Dict[str, List[np.ndarray]] = {}
    all_gids = []
    all_cis = []
    for seed in SEEDS:
        p = collect_dir / "image" / f"seed{seed}" / "image_features.npz"
        if not p.exists():
            continue
        d = np.load(p)
        if "episode_success_index" not in d.files or "hidden_episode_labels" not in d.files:
            continue
        success_index = d["episode_success_index"]
        success_flag = d["episode_success_flag"]
        success_set = {int(e) for e, ok in zip(success_index, success_flag) if ok}
        ep_lab = d["hidden_episode_labels"]
        ci_lab = d["hidden_call_idx_labels"]
        mask = np.array([int(e) in success_set for e in ep_lab])
        gids = np.array([global_id(seed, e) for e in ep_lab[mask]])
        all_gids.append(gids)
        all_cis.append(ci_lab[mask])
        for key in d.files:
            if not key.startswith("hidfeat_"):
                continue
            merged.setdefault(key, []).append(d[key][mask])
    out = {k: np.concatenate(v, axis=0) for k, v in merged.items()}
    out["hidden_episode_labels"] = np.concatenate(all_gids) if all_gids else np.array([])
    out["hidden_call_idx_labels"] = np.concatenate(all_cis) if all_cis else np.array([])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"Saved: {out_path}  (N={len(out['hidden_episode_labels'])})")


def merge_image_sigma_data(collect_dir: Path, out_path: Path):
    """pooled variance formula で複数seedのsigma_data per-dim統計を正しく合成する。"""
    per_token: Dict[str, List] = {}  # name -> list of (mean_vec, std_vec, n)
    for seed in SEEDS:
        p = collect_dir / "image" / f"seed{seed}" / "image_features.npz"
        if not p.exists():
            continue
        d = np.load(p)
        for key in d.files:
            if not key.startswith("sigma_data_per_dim_mean_"):
                continue
            name = key[len("sigma_data_per_dim_mean_"):]
            mean_vec = d[key]
            std_vec = d[f"sigma_data_per_dim_std_{name}"]
            n = int(d[f"sigma_data_n_{name}"][0])
            per_token.setdefault(name, []).append((mean_vec, std_vec, n))

    out = {}
    for name, parts in per_token.items():
        total_n = sum(n for _, _, n in parts)
        if total_n < 2:
            continue
        pooled_mean = sum(m * n for m, _, n in parts) / total_n
        # pooled variance: within-group + between-group
        within = sum((n - 1) * (s ** 2) for m, s, n in parts if n > 1)
        between = sum(n * (m - pooled_mean) ** 2 for m, s, n in parts)
        pooled_var = (within + between) / max(total_n - 1, 1)
        pooled_std = np.sqrt(np.maximum(pooled_var, 0.0))
        out[f"sigma_data_{name}"] = np.array([float(pooled_std.mean())])
        out[f"sigma_data_n_{name}"] = np.array([total_n])
        print(f"  {name}: pooled sigma_data={float(pooled_std.mean()):.4f} (N={total_n})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
