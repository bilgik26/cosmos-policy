"""
§5 (テーマ4: 線形プロービング) の再計算。3seed成功epのみマージデータを用いる。

重要な設計判断: action_features.npz (feature_analysis.py由来) と
step_actions.npz (mechanism_analysis.py由来) は独立した別ロールアウトである。
同一seed・同一episode数で実行しても、GPU非決定性により極少数のepisodeで
成否が食い違いうることを実データで確認済み(seed195: feature 29/50 vs
mechanism 30/50成功)。位置(row index)だけを頼りに単純連結すると特徴量と
ラベルが不整合になるリスクがあるため、(global_episode_id, call_idx_in_ep)
の組をキーとして両者の積集合のみを使う。

実行:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.v4_section5 \
      --merged_dir results/v4_merged --out_json results/v4_merged/section5_results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cosmos_policy.experiments.robot.robocasa.analysis.v4_stats_lib import (
    loeo_probe, loeo_probe_permutation_test, bh_fdr,
)
from cosmos_policy.experiments.robot.robocasa.analysis.linear_probe import (
    labels_from_progress, labels_from_gripper,
)

PROBE_LAYERS = [0, 4, 9, 13, 18, 22, 27]
NUM_DENOISE_STEPS = 5


def build_common_index(feat_ep, feat_ci, act_ep, act_ci):
    """(episode, call_idx) をキーに両者の積集合の対応indexペアを返す。"""
    feat_keys = {(int(e), int(c)): i for i, (e, c) in enumerate(zip(feat_ep, feat_ci))}
    act_keys = {(int(e), int(c)): i for i, (e, c) in enumerate(zip(act_ep, act_ci))}
    common = sorted(set(feat_keys) & set(act_keys))
    feat_idx = np.array([feat_keys[k] for k in common])
    act_idx = np.array([act_keys[k] for k in common])
    ep_of_common = np.array([k[0] for k in common])
    return feat_idx, act_idx, ep_of_common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", default="cosmos_policy/experiments/robot/robocasa/analysis/results/v4_merged")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--n_perm", type=int, default=100)
    args = ap.parse_args()
    merged_dir = Path(args.merged_dir)
    out_json = Path(args.out_json) if args.out_json else merged_dir / "section5_results.json"

    fd = np.load(merged_dir / "action_features.npz")
    ad = np.load(merged_dir / "step_actions.npz")

    feat_ep, feat_ci = fd["episode_labels"], fd["call_idx_labels"]
    # step_actionsのk=4基準のepisode/call_idxラベルを使う(gripper_2/3の定義がk=4依存のため)
    act_ep, act_ci = ad["episode_labels_4"], ad["call_idx_labels_4"]

    feat_idx, act_idx, ep_common = build_common_index(feat_ep, feat_ci, act_ep, act_ci)
    n_feat_total, n_act_total = len(feat_ep), len(act_ep)
    n_common = len(feat_idx)
    print(f"feature calls={n_feat_total}, action calls={n_act_total}, "
          f"common(matched by episode+call_idx)={n_common}")

    # actions dict (k=4を基準にjoinしたが、gripper labelにはk=4の値をそのまま使う。
    # ラベル自体はk非依存なので、feat_idxで揃えたあとは全kで共通に使える)
    actions_k4 = ad["4"][act_idx]  # (n_common, 32, 7)
    actions_for_labels = {4: actions_k4}

    ep_arr = ep_common
    ci_arr = feat_ci[feat_idx]

    progress_labels, progress_values = labels_from_progress(ep_arr, ci_arr, n_classes=3)
    gripper_2 = labels_from_gripper(actions_for_labels, k=4, n_classes=2)
    gripper_3 = labels_from_gripper(actions_for_labels, k=4, n_classes=3)

    label_configs = {
        "progress_3": progress_labels,
        "gripper_2": gripper_2,
        "gripper_3": gripper_3,
    }

    results = {"n_common_calls": int(n_common), "n_episodes": int(len(np.unique(ep_arr)))}
    all_pvals = []
    all_keys = []

    for ltype, labels in label_configs.items():
        results[ltype] = {}
        for k in range(NUM_DENOISE_STEPS):
            results[ltype][k] = {}
            for l in PROBE_LAYERS:
                key = f"feat_k{k}_layer{l}"
                if key not in fd.files:
                    continue
                X = fd[key][feat_idx]
                perm_res = loeo_probe_permutation_test(
                    X, labels, ep_arr, n_components=30, n_perm=args.n_perm
                )
                results[ltype][k][l] = perm_res
                all_pvals.append(perm_res["p_value"])
                all_keys.append((ltype, k, l))
            print(f"  {ltype} k={k} done")

    sig = bh_fdr(all_pvals, alpha=0.05)
    for (ltype, k, l), s in zip(all_keys, sig):
        results[ltype][k][l]["bh_significant"] = bool(s)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
