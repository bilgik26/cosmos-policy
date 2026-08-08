"""
gradient_structure_test.py — attractor_verification_report.md §4.1 の主張
「（離散クラスタではなく）連続的な勾配(gradient-like)構造の可能性」を直接検証する。

背景: skill_count.py の Level1 クラスタリングバッテリは、フェーズが強く線形デコード可能
(Gate[2] PASS) であるにもかかわらず、安定した離散クラスタを検出できなかった (0/4タスク)。
レポートはこれを「線形分離可能 = 離散クラスタ」ではなく「連続的勾配構造」の可能性として
解釈したが、この解釈自体は直接検証されていなかった。本スクリプトはその検証を行う。

方法（P5: fold内前処理 / P6: 循環回避 を厳守）:
  1. gripper_state, motion_state (phase_labelの構成要素、二値) それぞれについて、
     GroupKFold(group=episode) で学習用foldにLogisticRegressionを fit し、held-out foldの
     決定関数値 (1次元射影) だけをプールする。射影軸をテストデータ自身から求めない
     ことで「軸を最適分離するように選んだから双峰に見える」という循環を避ける。
  2. held-out射影に対して:
     (a) Hartigan's dip test (diptest package) で単峰性を検定。p>0.05 → 単峰性を棄却できない
         (勾配的)。p<0.05 → 有意に非単峰 (離散クラスタ的)。
     (b) GMM(1コンポーネント) vs GMM(2コンポーネント) のBIC差 (1D)。ΔBIC=BIC(1)-BIC(2) が
         大きく正 → 2峰モデルが有意に良い (離散クラスタ的)。ΔBIC≈0 or 負 → 追加コンポーネント
         は説明力を増やさない (勾配的)。
     (c) KDEの谷/山比 (valley-to-peak ratio): 2クラス条件付き平均の間の密度谷が両ピークの
         最小値に対してどれだけ深いか。1に近い→谷なし(勾配)。0に近い→明確な谷(離散)。
  3. 比較対象として、閾値化される前の生の物理量 (gripper opening width, motion speed) 自体の
     単峰性も同じ3指標で検定する。これにより「表現Zの勾配性は、そもそも物理量自体が連続的
     だからか、それとも表現が物理的な閾値構造を滑らかにしているからか」を切り分ける。

注意: 本スクリプトはあくまで補助検証であり、GATE等の採否基準には影響しない
(design原案には存在しない、レポート執筆後の追加分析)。
"""

import json
from pathlib import Path

import diptest
import numpy as np
from scipy.stats import gaussian_kde, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.skill_count import (
    LAYER, K_STEP, load_task_seed_data, scene_residualize, preprocess,
)


def gmm_bic_delta(x):
    """BIC(1comp) - BIC(2comp)。正で大きいほど2峰モデルが有意に良い(離散的)。"""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    reg = max(1e-8, float(np.var(x)) * 1e-3)
    bic1 = GaussianMixture(n_components=1, random_state=0, reg_covar=reg).fit(x).bic(x)
    bic2 = GaussianMixture(n_components=2, random_state=0, n_init=5, reg_covar=reg).fit(x).bic(x)
    return float(bic1 - bic2)


def valley_peak_ratio(x, y_binary):
    """1D KDEで、2クラス条件付き平均の間の谷の深さを (谷密度 / 小さい方のピーク密度) で返す。
    1に近い=谷なし(勾配), 0に近い=明確な谷(離散クラスタ)。"""
    x = np.asarray(x, dtype=np.float64)
    if len(np.unique(y_binary)) < 2:
        return None
    m0 = x[y_binary == 0].mean()
    m1 = x[y_binary == 1].mean()
    lo, hi = sorted([m0, m1])
    if hi - lo < 1e-9:
        return None
    kde = gaussian_kde(x)
    grid_valley = np.linspace(lo, hi, 200)
    valley_density = kde(grid_valley).min()
    peak0 = kde(np.array([m0]))[0]
    peak1 = kde(np.array([m1]))[0]
    peak_min = min(peak0, peak1)
    if peak_min < 1e-12:
        return None
    return float(valley_density / peak_min)


def unimodality_battery(x, y_binary=None, label=""):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    dip_stat, dip_p = diptest.diptest(x)
    out = {
        "n": int(len(x)),
        "dip_statistic": float(dip_stat),
        "dip_p_value": float(dip_p),
        "unimodal_not_rejected": bool(dip_p > 0.05),
        "gmm_bic_delta_1minus2": gmm_bic_delta(x),
        "gmm2_preferred": bool(gmm_bic_delta(x) > 10.0),  # standard "strong evidence" BIC threshold
    }
    if y_binary is not None:
        out["valley_to_peak_ratio"] = valley_peak_ratio(x, y_binary)
    return out


def analyze_task_seed_full(collect_dir, fname, seed=0):
    d = load_task_seed_data(collect_dir, fname, LAYER, K_STEP)
    X_raw, episode = d["feats"], d["episode"]
    phase_path = collect_dir / fname.replace(".npz", "_phases.npz")
    keep_idx = np.load(collect_dir / fname)[f"feat_k{K_STEP}_layer{LAYER}_idx"]
    ph = np.load(phase_path)
    gripper_state = ph["gripper_state"][keep_idx]
    motion_state = ph["motion_state"][keep_idx]
    gripper_width = ph["gripper_mean"][keep_idx]
    speed_log = np.log1p(ph["speed"][keep_idx])

    Xr = scene_residualize(X_raw, episode)
    Xp = preprocess(Xr, seed=seed)

    out = {}
    for axis_name, y, raw in [
        ("gripper", gripper_state, gripper_width),
        ("motion", motion_state, speed_log),
    ]:
        n_groups = len(np.unique(episode))
        n_splits = min(5, n_groups)
        if n_splits < 2 or len(np.unique(y)) < 2:
            out[axis_name] = {"skipped": "degenerate label or too few episode groups"}
            continue
        gkf = GroupKFold(n_splits=n_splits)
        proj = np.full(len(y), np.nan)
        for train_idx, test_idx in gkf.split(Xp, y, groups=episode):
            if len(np.unique(y[train_idx])) < 2:
                continue
            scaler = StandardScaler().fit(Xp[train_idx])
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            clf.fit(scaler.transform(Xp[train_idx]), y[train_idx])
            proj[test_idx] = clf.decision_function(scaler.transform(Xp[test_idx]))
        valid = ~np.isnan(proj)
        proj_v, y_v, raw_v = proj[valid], y[valid], raw[valid]

        rep_battery = unimodality_battery(proj_v, y_v)
        raw_battery = unimodality_battery(raw_v, y_v)
        rho, rho_p = spearmanr(proj_v, raw_v)

        out[axis_name] = {
            "n_holdout": int(valid.sum()),
            "representation_projection_unimodality": rep_battery,
            "raw_physical_quantity_unimodality": raw_battery,
            "spearman_proj_vs_raw_quantity": {"rho": float(rho), "p_value": float(rho_p)},
            "interpretation": (
                "representation側がunimodal_not_rejected=Trueかつraw側もTrue → "
                "物理量・表現とも連続的で、勾配構造の解釈と整合。"
                if rep_battery["unimodal_not_rejected"]
                else "representation側で単峰性が棄却された(有意に非単峰) → "
                "この軸については離散クラスタ的構造がGroupKFold held-out射影でも支持される。"
            ),
        }
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--collect_dir", required=True)
    p.add_argument("--out_dir", required=True)
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
        results[task] = {}
        for seed_series, fname in fnames_by_seed.items():
            log_message(f"[gradient_structure_test] {task} seed_series={seed_series}")
            res = analyze_task_seed_full(collect_dir, fname, seed=0)
            results[task][str(seed_series)] = res
            for axis_name, r in res.items():
                if "skipped" in r:
                    log_message(f"  {axis_name}: skipped ({r['skipped']})")
                    continue
                rp = r["representation_projection_unimodality"]
                rw = r["raw_physical_quantity_unimodality"]
                log_message(
                    f"  {axis_name}: rep dip_p={rp['dip_p_value']:.3f} "
                    f"(unimodal_not_rejected={rp['unimodal_not_rejected']}) "
                    f"ΔBIC(1-2)={rp['gmm_bic_delta_1minus2']:.1f} "
                    f"valley/peak={rp.get('valley_to_peak_ratio')} "
                    f"| raw dip_p={rw['dip_p_value']:.3f} "
                    f"| spearman(proj,raw)={r['spearman_proj_vs_raw_quantity']['rho']:.3f}"
                )

    with open(out_dir / "gradient_structure_test.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'gradient_structure_test.json'}")


if __name__ == "__main__":
    main()
