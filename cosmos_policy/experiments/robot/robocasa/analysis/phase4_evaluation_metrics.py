"""
phase4_evaluation_metrics.py — latent_dynamics_verification_design.md フェーズ4 対応。

フェーズ3(dynamic_vector_field_steering.py)が出力した各条件(C0〜C5)のロールアウトログ
(dynamic_vector_field_steering_<task>.json)から、design.md §フェーズ4が要求する3指標を計算する:

  1. **動的因果効力 (Dynamic Causal Efficacy)**: 各ロールアウトのエンドエフェクタ軌跡が、
     成功エピソード(教師)の軌跡群とどの程度形状的に一致するかを、DTW距離と離散Fréchet距離
     (いずれも自前で実装。標準ライブラリ・追加依存なしで実行可能な軽量なDP実装)で測定する。
     教師軌跡は collect_v2/ の成功エピソードのエンドエフェクタ位置系列(call粒度)を用いる。
     本ロールアウトログのeef_trajはenv-step粒度(教師よりも高解像度)であるため、比較の公平性の
     ため教師と本ロールアウトの双方を同一個数のウェイポイントにリサンプルしてから距離を計算する。
  2. **多様体逸脱度 (Manifold Deviation)**: フェーズ1(dynamics_embedding_test.py)が保存した
     成功エピソードのみの進行度ビン参照分布(§5.6と同じ fit_bin_reference/deviation_from_
     reference)を、各条件のロールアウトのXp軌跡(フェーズ3が既にオンラインで計算・保存済み、
     xp_traj/xp_call_idx)に適用し、逸脱量を条件間で比較する。
  3. **感覚・環境フィードバックへの従属性**: design.mdは「観測を変化させた際、出力アクションが
     Reactiveな一定出力にならず、不変量を維持したまま柔軟に調整されるか」を問う。本検証では
     C2(通常observation)とC5(observationをcall0の1枚に固定、環境自体は実際に進行)を比較する
     直接統制条件を用意した(フェーズ3で実装済み)。両条件で同一のsteering設定を使うため、
     行動系列の複雑さ・可変性(エンドエフェクタ経路長、速度の分散)がC2で明確にC5を上回れば、
     動的介入下の行動が観測に従属して柔軟に変化していること(=単なる固定的な既定動作の再生では
     ないこと)の証拠になる。

出力: phase4_evaluation_metrics.json (各条件・各episodeの指標、条件間比較の要約統計)。
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.success_failure_trajectory_test import (
    fit_bin_reference, deviation_from_reference, load_task_seed_data_with_success,
)
from cosmos_policy.experiments.robot.robocasa.analysis.skill_count import LAYER, K_STEP


# ─────────────────────────── lightweight DTW / discrete Fréchet (no new deps) ───────────────────────────

def resample_polyline(points, n_out):
    """Arc-length resampling of a (T,3) polyline to exactly n_out waypoints."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.repeat(points, n_out, axis=0) if len(points) == 1 else np.zeros((n_out, points.shape[-1]))
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return np.repeat(points[:1], n_out, axis=0)
    targets = np.linspace(0, total, n_out)
    out = np.zeros((n_out, points.shape[-1]))
    for i, t in enumerate(targets):
        j = np.searchsorted(cum, t, side="right") - 1
        j = np.clip(j, 0, len(points) - 2)
        span = cum[j + 1] - cum[j]
        frac = (t - cum[j]) / span if span > 1e-12 else 0.0
        out[i] = points[j] + frac * (points[j + 1] - points[j])
    return out


def dtw_distance(a, b):
    """Standard O(T_a*T_b) DTW with Euclidean local cost, normalized by path length."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[na, nb] / (na + nb))


def discrete_frechet(a, b):
    """Standard dynamic-programming discrete Fréchet distance (Eiter & Mannila 1994)."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    ca = np.full((na, nb), -1.0)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        d = np.linalg.norm(a[i] - b[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), d)
        else:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), d)
        return ca[i, j]

    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, na * nb + 100))
    try:
        return float(c(na - 1, nb - 1))
    finally:
        sys.setrecursionlimit(old_limit)


N_RESAMPLE = 20   # waypoints per trajectory for DTW/Frechet (keeps discrete_frechet's O(T^2) cheap)


# ─────────────────────────── metric 1: dynamic causal efficacy ───────────────────────────

def load_teacher_trajectories(collect_dir: Path, task: str, manifest, max_teachers=15):
    """Success-episode eef_pos sequences (call-granularity) from collect_v2/, held out from
    (i.e. never used to fit) the phase-3 steering vector field itself -- the field was built
    from Xp/eef *dynamics* (deltas), not from these absolute trajectories, so there is no
    circularity in using them as a held-out shape-similarity reference here."""
    teachers = []
    for fname, info in manifest["files"].items():
        if info["task"] != task:
            continue
        d = load_task_seed_data_with_success(collect_dir, fname, LAYER, K_STEP)
        fd = np.load(collect_dir / fname)
        key = f"feat_k{K_STEP}_layer{LAYER}"
        keep_idx = fd[key + "_idx"]
        eef_pos = fd["eef_pos"][keep_idx]
        episode, call_idx, success = d["episode"], d["call_idx"], d["success"]
        for e in np.unique(episode):
            if not bool(success[episode == e][0]):
                continue
            idx = np.where(episode == e)[0]
            order = idx[np.argsort(call_idx[idx])]
            teachers.append(eef_pos[order])
            if len(teachers) >= max_teachers:
                return teachers
    return teachers


def dynamic_causal_efficacy(episode_logs, teacher_trajs):
    if not teacher_trajs:
        return None
    teachers_rs = [resample_polyline(t, N_RESAMPLE) for t in teacher_trajs]
    per_episode = []
    for ep in episode_logs:
        traj = np.array(ep["eef_traj"])
        if len(traj) < 2:
            continue
        traj_rs = resample_polyline(traj, N_RESAMPLE)
        dtws = [dtw_distance(traj_rs, t) for t in teachers_rs]
        frechets = [discrete_frechet(traj_rs, t) for t in teachers_rs]
        per_episode.append({
            "min_dtw_to_teacher": float(np.min(dtws)), "mean_dtw_to_teacher": float(np.mean(dtws)),
            "min_frechet_to_teacher": float(np.min(frechets)), "mean_frechet_to_teacher": float(np.mean(frechets)),
        })
    if not per_episode:
        return None
    return {
        "n_episodes": len(per_episode),
        "mean_min_dtw": float(np.mean([e["min_dtw_to_teacher"] for e in per_episode])),
        "mean_min_frechet": float(np.mean([e["min_frechet_to_teacher"] for e in per_episode])),
        "per_episode": per_episode,
    }


# ─────────────────────────── metric 2: manifold deviation ───────────────────────────

def manifold_deviation(episode_logs, centers, variances, n_bins):
    from cosmos_policy.experiments.robot.robocasa.analysis.success_failure_trajectory_test import progress_bin
    per_episode_mean_dev = []
    for ep in episode_logs:
        xp = np.array(ep.get("xp_traj", []))
        if len(xp) < 2:
            continue
        progress = np.arange(len(xp)) / max(len(xp) - 1, 1)
        dev = deviation_from_reference(xp, progress, centers, variances, n_bins=n_bins)
        per_episode_mean_dev.append(float(np.mean(dev)))
    if not per_episode_mean_dev:
        return None
    return {"n_episodes": len(per_episode_mean_dev), "mean_deviation": float(np.mean(per_episode_mean_dev)),
            "per_episode_mean_deviation": per_episode_mean_dev}


# ─────────────────────────── metric 3: sensory feedback dependency ───────────────────────────

def motion_complexity(episode_logs):
    path_lengths, action_stds = [], []
    for ep in episode_logs:
        traj = np.array(ep["eef_traj"])
        if len(traj) < 2:
            continue
        path_lengths.append(float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))))
        actions = np.array(ep["action_log"])
        if len(actions) >= 2:
            action_stds.append(float(np.mean(np.std(actions, axis=0))))
    if not path_lengths:
        return None
    return {
        "n_episodes": len(path_lengths),
        "mean_path_length": float(np.mean(path_lengths)),
        "mean_action_std": float(np.mean(action_stds)) if action_stds else float("nan"),
        "path_lengths": path_lengths,
    }


def sensory_feedback_dependency(normal_logs, frozen_logs):
    m_normal = motion_complexity(normal_logs)
    m_frozen = motion_complexity(frozen_logs)
    if m_normal is None or m_frozen is None:
        return None
    u_stat, u_p = mannwhitneyu(m_normal["path_lengths"], m_frozen["path_lengths"], alternative="greater")
    return {
        "normal_obs": m_normal, "frozen_obs": m_frozen,
        "mannwhitney_p_normal_path_length_gt_frozen": float(u_p),
        "path_length_ratio_normal_over_frozen": float(m_normal["mean_path_length"] / (m_frozen["mean_path_length"] + 1e-9)),
    }


# ─────────────────────────── main ───────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steering_json", required=True)
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--collect_dir", required=True,
                    help="collect_v2/ (for teacher trajectories -- needs episode_success, unlike collect/)")
    p.add_argument("--task_name", default="PnPCounterToCab")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steering = json.loads(Path(args.steering_json).read_text())
    conditions = steering["conditions"]

    import pickle
    with open(Path(args.embedding_dir) / f"dynamics_embedding_artifact_{args.task_name}.pkl", "rb") as f:
        dyn_artifact = pickle.load(f)
    from cosmos_policy.experiments.robot.robocasa.analysis.manifold_trajectory_test import episode_progress
    from cosmos_policy.experiments.robot.robocasa.analysis.success_failure_trajectory_test import N_PROGRESS_BINS
    episode, call_idx, success = dyn_artifact["episode"], dyn_artifact["call_idx"], dyn_artifact["success"]
    Xp = dyn_artifact["Xp"]
    mask_succ = success.astype(bool)
    progress = episode_progress(episode, call_idx)
    centers, variances = fit_bin_reference(Xp[mask_succ], progress[mask_succ])

    manifest = json.loads((Path(args.collect_dir) / "multitask_manifest.json").read_text())
    teacher_trajs = load_teacher_trajectories(Path(args.collect_dir), args.task_name, manifest)
    log_message(f"[phase4] loaded {len(teacher_trajs)} teacher (successful) eef trajectories for {args.task_name}")

    results = {"task": args.task_name, "n_teacher_trajectories": len(teacher_trajs), "conditions": {}}
    for cond_name, cond in conditions.items():
        ep_logs = cond["episodes"]
        results["conditions"][cond_name] = {
            "success_rate": cond["success_rate"],
            "dynamic_causal_efficacy": dynamic_causal_efficacy(ep_logs, teacher_trajs),
            "manifold_deviation": manifold_deviation(ep_logs, centers, variances, N_PROGRESS_BINS),
            "motion_complexity": motion_complexity(ep_logs),
        }
        log_message(f"[phase4 {cond_name}] success_rate={cond['success_rate']:.2f} "
                    f"dce_mean_min_dtw={results['conditions'][cond_name]['dynamic_causal_efficacy']['mean_min_dtw'] if results['conditions'][cond_name]['dynamic_causal_efficacy'] else float('nan'):.3f} "
                    f"manifold_dev={results['conditions'][cond_name]['manifold_deviation']['mean_deviation'] if results['conditions'][cond_name]['manifold_deviation'] else float('nan'):.3f}")

    if "C2_dummy_dynamic_field" in conditions and "C5_dummy_dynamic_field_frozen_obs" in conditions:
        results["sensory_feedback_dependency_C2_vs_C5"] = sensory_feedback_dependency(
            conditions["C2_dummy_dynamic_field"]["episodes"],
            conditions["C5_dummy_dynamic_field_frozen_obs"]["episodes"],
        )
        sfd = results["sensory_feedback_dependency_C2_vs_C5"]
        if sfd:
            log_message(f"[phase4] sensory feedback dependency: path_length ratio (normal/frozen)="
                        f"{sfd['path_length_ratio_normal_over_frozen']:.3f} "
                        f"MW p(normal>frozen)={sfd['mannwhitney_p_normal_path_length_gt_frozen']:.4f}")

    with open(out_dir / f"phase4_evaluation_metrics_{args.task_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / f'phase4_evaluation_metrics_{args.task_name}.json'}")


if __name__ == "__main__":
    main()
