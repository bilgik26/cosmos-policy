"""
phase_labeling.py — attractor_verification_design.md §4.3 準拠

collect_multitask.py が保存した raw 物理量 (robot0_gripper_qpos, robot0_eef_pos) から
「フェーズラベル」を導出する。**潜在表現には一切触れない**（循環回避、P6）。

定義 (設計書の要求「運動＋グリッパーの複合、グリッパー単独は使わない」に対応):
  - gripper_state ∈ {0=open, 1=closed}: gripper_qpos の平均値をプールした全データに対する
    1次元 2-means で二値化 (データ駆動、タスク間で共通の閾値)。
  - motion_state ∈ {0=still, 1=moving}: call 間の eef_pos 変位ノルムを同様に 2-means で二値化。
  - phase_label ∈ {0,1,2,3} = gripper_state × motion_state の複合 (4クラス)。
  - transition_flag: 直前/直後の call で gripper_state が変化する call (grasp/release 遷移の近傍)。
  - progress_bin ∈ {0,1,2}: エピソード内進行度 (analysis_shared.skill_phase_labels と同じ規約)。

**限界の明記**: 真の接触センサ (contact force) は使用していない。motion (eef 変位) を
接触の代理指標として使う近似であり、これは報告書に限界として明記する。

出力: 各 collect/<task>_seed<S>.npz に対応する collect/<task>_seed<S>_phases.npz
       (行順序は入力 npz と完全に一致)
"""

import json
from pathlib import Path

import numpy as np

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import skill_phase_labels


def load_meta(path: Path):
    d = np.load(path, allow_pickle=False)
    return {
        "episode": d["episode"],
        "episode_in_task_seed": d["episode_in_task_seed"],
        "call_idx": d["call_idx"],
        "task_idx": d["task_idx"],
        "task_name": d["task_name"],
        "seed_series_idx": d["seed_series_idx"],
        "gripper_qpos": d["gripper_qpos"],
        "eef_pos": d["eef_pos"],
    }


def two_means_1d(x: np.ndarray, n_init: int = 10, seed: int = 0):
    """依存を増やさない単純な1次元2-meansクラスタリング。閾値(2クラス境界)を返す。"""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.RandomState(seed)
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    best_inertia = np.inf
    best_centers = None
    for _ in range(n_init):
        c = rng.uniform(lo, hi, size=2)
        for _ in range(100):
            d0 = np.abs(x - c[0])
            d1 = np.abs(x - c[1])
            assign = (d1 < d0).astype(int)
            new_c = np.array([
                x[assign == 0].mean() if (assign == 0).any() else c[0],
                x[assign == 1].mean() if (assign == 1).any() else c[1],
            ])
            if np.allclose(new_c, c):
                break
            c = new_c
        d0 = np.abs(x - c[0])
        d1 = np.abs(x - c[1])
        assign = (d1 < d0).astype(int)
        inertia = np.sum(np.minimum(d0, d1) ** 2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = np.sort(c)
    threshold = float(best_centers.mean())
    return threshold, best_centers


def compute_motion_speed(meta):
    """episode内でcall_idx順に隣接callのeef_pos変位ノルムを計算 (episodeをまたがない)。"""
    n = len(meta["episode"])
    speed = np.zeros(n, dtype=np.float64)
    order = np.lexsort((meta["call_idx"], meta["episode"]))
    for pos in range(len(order)):
        i = order[pos]
        if pos == 0 or meta["episode"][order[pos - 1]] != meta["episode"][i]:
            speed[i] = np.nan  # will backfill below
        else:
            prev = order[pos - 1]
            speed[i] = np.linalg.norm(meta["eef_pos"][i] - meta["eef_pos"][prev])
    # backfill first-call-of-episode speed from the next call in the same episode
    for pos in range(len(order)):
        i = order[pos]
        if np.isnan(speed[i]):
            if pos + 1 < len(order) and meta["episode"][order[pos + 1]] == meta["episode"][i]:
                speed[i] = speed[order[pos + 1]]
            else:
                speed[i] = 0.0
    return speed


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--collect_dir", required=True)
    args = p.parse_args()

    collect_dir = Path(args.collect_dir)
    manifest_path = collect_dir / "multitask_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = sorted(manifest["files"].keys())

    log_message(f"Loading metadata from {len(files)} files for global threshold fit...")
    all_meta = {}
    pooled_gripper = []
    pooled_speed = []
    for fname in files:
        meta = load_meta(collect_dir / fname)
        speed = compute_motion_speed(meta)
        meta["speed"] = speed
        # opening width proxy (NOT mean(): panda's 2 fingers have near-opposite-sign
        # qpos so a plain mean cancels to ~0 regardless of open/closed state).
        meta["gripper_mean"] = np.abs(meta["gripper_qpos"]).sum(axis=1)
        all_meta[fname] = meta
        pooled_gripper.append(meta["gripper_mean"])
        pooled_speed.append(speed)

    pooled_gripper = np.concatenate(pooled_gripper)
    pooled_speed = np.concatenate(pooled_speed)

    gripper_threshold, gripper_centers = two_means_1d(pooled_gripper)
    # speed is heavy-tailed; cluster in log1p space for stability
    speed_threshold_log, speed_centers_log = two_means_1d(np.log1p(pooled_speed))
    speed_threshold = float(np.expm1(speed_threshold_log))

    log_message(f"Global gripper threshold: {gripper_threshold:.5f} (centers={gripper_centers})")
    log_message(f"Global motion-speed threshold: {speed_threshold:.5f} (log-centers={speed_centers_log})")

    summary = {
        "gripper_threshold": gripper_threshold,
        "gripper_centers": gripper_centers.tolist(),
        "speed_threshold": speed_threshold,
        "speed_threshold_log_centers": speed_centers_log.tolist(),
        "n_total_calls": int(len(pooled_gripper)),
        "phase_label_definition": {
            0: "gripper_open + still",
            1: "gripper_open + moving",
            2: "gripper_closed + still",
            3: "gripper_closed + moving",
        },
        "limitation_note": (
            "接触センサ未使用。eef_pos の call間変位を接触/相互作用の近似代理指標として使用。"
            "gripper_state は robot0_gripper_qpos の開き幅 (|dim0|+|dim1|, 2指が反対符号のため"
            "単純平均は使えない) の全データプール2-meansで二値化 (幅小=closed=1)。"
        ),
        "per_file_phase_distribution": {},
    }

    for fname in files:
        meta = all_meta[fname]
        # gripper_mean = opening width; small width = closed = state 1
        gripper_state = (meta["gripper_mean"] < gripper_threshold).astype(int)
        motion_state = (meta["speed"] > speed_threshold).astype(int)
        phase_label = gripper_state * 2 + motion_state  # 0..3

        # transition_flag: gripper_state changes vs previous/next call within same episode
        n = len(gripper_state)
        transition_flag = np.zeros(n, dtype=int)
        order = np.lexsort((meta["call_idx"], meta["episode"]))
        for pos in range(len(order)):
            i = order[pos]
            changed = False
            if pos > 0 and meta["episode"][order[pos - 1]] == meta["episode"][i]:
                if gripper_state[order[pos - 1]] != gripper_state[i]:
                    changed = True
            if pos + 1 < len(order) and meta["episode"][order[pos + 1]] == meta["episode"][i]:
                if gripper_state[order[pos + 1]] != gripper_state[i]:
                    changed = True
            transition_flag[i] = int(changed)

        # progress_bin: reuse analysis_shared.skill_phase_labels (0/1/2 by within-episode progress)
        progress_bin = skill_phase_labels(meta["episode"], meta["call_idx"].astype(float))

        out_path = collect_dir / (fname.replace(".npz", "_phases.npz"))
        np.savez(
            out_path,
            gripper_state=gripper_state,
            motion_state=motion_state,
            phase_label=phase_label,
            transition_flag=transition_flag,
            progress_bin=progress_bin,
            speed=meta["speed"],
            gripper_mean=meta["gripper_mean"],
        )
        vals, counts = np.unique(phase_label, return_counts=True)
        dist = {int(v): int(c) for v, c in zip(vals, counts)}
        summary["per_file_phase_distribution"][fname] = dist
        log_message(f"Saved {out_path.name}: phase distribution {dist}, "
                    f"transitions={transition_flag.sum()}/{n}")

    with open(collect_dir / "phase_labeling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log_message(f"Saved phase_labeling_summary.json")


if __name__ == "__main__":
    main()
