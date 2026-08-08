"""
§6.1 (自己注意解析) と §7.1 (クロスアテンション分布) の再計算。
3seed成功epのみマージデータを用いる。

§6.1: results/v4_merged/attn_tmatrices.npz (行正規化T-matrix、既に
      「成功epのみ・episode単位均等加重」で平均化済み、v4_merge.pyのmerge_attn_tmatricesが
      構築) を読み込み、報告書の表(入力/出力タイプ別、層別プロファイル)を再現する。

§7.1: results/v4_merged/crossattn.npz (crossattn_analysis.py由来、成功epのみ)
      から、実トークンへの重み集中・pad重みを再計算する。

実行:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.v4_section6_7 \
      --merged_dir results/v4_merged --out_json results/v4_merged/section6_7_results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import (
    STATE_T, T_NAMES, BLANK_T_IDX, PROPRIO_T_IDX, CURR_WRIST_T_IDX,
    CURR_PRIMARY_T_IDX, CURR_SECONDARY_T_IDX, ACTION_T_IDX,
)

PROBE_LAYERS = [0, 4, 9, 13, 18, 22, 27]
UNIFORM_NULL = 1.0 / STATE_T


def _row_normalize(row: np.ndarray) -> np.ndarray:
    """attention_analysis_v2.pyのt_mat[t_out,t_in]は`block.mean()`
    （t_inブロック内196キーの平均softmax重み）であり、行和は1ではなく
    1/PATCHES_PER_T(=1/196)になる（実データで確認: 行和が常に0.0051020...=1/196）。
    report §6.1 の「行正規化（各行を行和で割り、行和=1に変換）」を再現するには、
    ここで明示的に行和で割る必要がある（全t_inで同じスケール定数がかかっている
    ため、行和で割ることと×196することは数学的に同値）。"""
    s = row.sum()
    return row / s if s > 0 else row


def section_6_1(merged_dir: Path):
    d = np.load(merged_dir / "attn_tmatrices.npz")
    meta = json.load(open(merged_dir / "attn_tmatrices.meta.json"))
    profile_k4 = {}
    for l in PROBE_LAYERS:
        key = f"tmat_block{l}_k4"
        if key not in d.files:
            continue
        tmat = d[key]  # (11,11), row t_out, col t_in (未正規化: 行和=1/196)
        row = _row_normalize(tmat[ACTION_T_IDX])  # action token(T=5)の行(query)、行正規化後
        profile_k4[l] = {
            T_NAMES.get(t_in, str(t_in)): {
                "weight": float(row[t_in]),
                "ratio_to_uniform": float(row[t_in] / UNIFORM_NULL),
            }
            for t_in in range(STATE_T)
        }
    # 入力/出力タイプ別（中間層Blk-9,13,18,22平均, k=4）
    mid_layers = [9, 13, 18, 22]
    avg_by_type = {}
    for t_in in range(STATE_T):
        vals = []
        for l in mid_layers:
            key = f"tmat_block{l}_k4"
            if key in d.files:
                row = _row_normalize(d[key][ACTION_T_IDX])
                vals.append(row[t_in])
        if vals:
            avg_by_type[T_NAMES.get(t_in, str(t_in))] = {
                "weight": float(np.mean(vals)), "ratio_to_uniform": float(np.mean(vals) / UNIFORM_NULL),
            }
    return {"n_success_episodes_pooled": meta.get("n_success_episodes_pooled"),
            "t1_all_passed": meta.get("t1_all_passed"), "t2_all_passed": meta.get("t2_all_passed"),
            "profile_k4_by_block": profile_k4, "avg_by_type_midlayers_k4": avg_by_type}


def section_7_1(merged_dir: Path, n_real_tokens_hint=15):
    d = np.load(merged_dir / "crossattn.npz")
    meta = json.load(open(merged_dir / "crossattn.meta.json"))
    keys = [k for k in d.files if k.startswith("attn_layer")]
    if not keys:
        return {"available": False, "meta": meta}
    results = {}
    for l in PROBE_LAYERS:
        key4 = f"attn_layer{l}_k4"
        if key4 not in d.files:
            continue
        arr = d[key4]  # (N_calls, n_tokens) 512トークン全体への重み (unmasked)
        n_tokens = arr.shape[1]
        # 実トークン数は task description依存で変動するため、簡易にpad位置を
        # 「重みが厳密に0でない後半」として推定するのではなく、n_real_tokens_hintを
        # 使う(crossattn_masked_analysis.py相当の厳密なtoken_attention_mask処理は
        # 別途必要。ここでは実トークン重み合計/pad重み合計の概算のみ)。
        real_weight = float(arr[:, :n_real_tokens_hint].sum(axis=1).mean())
        pad_weight = float(arr[:, n_real_tokens_hint:].sum(axis=1).mean())
        results[l] = {"real_token_weight_approx": real_weight, "pad_weight_approx": pad_weight}
    return {"available": True, "n_success_episodes": len(meta.get("n_success_per_seed", {})),
            "per_layer_k4": results, "n_real_tokens_hint": n_real_tokens_hint}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", default="cosmos_policy/experiments/robot/robocasa/analysis/results/v4_merged")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    merged_dir = Path(args.merged_dir)
    out_json = Path(args.out_json) if args.out_json else merged_dir / "section6_7_results.json"

    print("Computing §6.1 self-attention...")
    s61 = section_6_1(merged_dir)
    print("  avg_by_type (midlayers k4):", {k: round(v["ratio_to_uniform"], 2) for k, v in s61["avg_by_type_midlayers_k4"].items()})

    print("Computing §7.1 cross-attention...")
    s71 = section_7_1(merged_dir)
    print("  available:", s71.get("available"))

    out = {"section6_1": s61, "section7_1": s71}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
