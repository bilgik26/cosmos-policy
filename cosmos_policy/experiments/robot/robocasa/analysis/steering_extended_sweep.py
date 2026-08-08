"""
steering_extended_sweep.py — attractor_verification_report.md §9 の拡張検証。

元の steering_intervention.py は Blk-13・k=(0,1)（早期20-40%）・α∈{1,2,4} の1条件のみを
テストした。本スクリプトはユーザーの指摘を受け、以下の3方向に one-variable-at-a-time で
拡張する（層・スケジュール・αの全組み合わせは計算コストが爆発するため採用しない）:

  1. 層スイープ: STEER_K_RANGE=(0,1), alpha=4.0 固定、layer∈{4,9,18,22}
     （元のlayer=13は既存結果を再利用、collect_multitask.pyが捕捉した7層
     [0,4,9,13,18,22,27] のうち中間層を選択）。
  2. スケジュールスイープ: layer=13, alpha=4.0 固定、k_range∈{(2,3)=中期, (3,4)=後期,
     (0,4)=全ステップ}（元のk=(0,1)=早期は既存結果を再利用）。
  3. αスイープ: layer=13, k_range=(0,1) 固定、alpha∈{8.0, 16.0}（元のalpha≤4を拡張）。

alpha=0 (no-op) ベースラインは層/スケジュールに依存しない（フックがalpha=0で早期return
するため）ので、本スクリプト内で1回だけ計算し全条件で共有する。

vec_typeはreal/randomの2種のみ（コスト抑制のため）。scene_orth方向は元の実験で
real方向とほぼ同じ挙動を示しており、今回は割愛。

大きなalphaはdenoiserを破壊的レジームに押しやり、NaN/Infアクションを生む可能性がある。
run_condition (steering_intervention.py, sanitize_actions) 側でこれを検出・報告する。
"""

import json
from pathlib import Path

import numpy as np

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import PolicyEvalConfig
from cosmos_policy.experiments.robot.robocasa.analysis.steering_intervention import (
    CHECKPOINT_CALL_IDX, SimpleFeatCapture, SteeringHook, compute_steering_vectors,
    run_condition, t1_noop_check,
)
from cosmos_policy.utils.utils import set_seed_everywhere

LAYER_SWEEP = [4, 9, 18, 22]
SCHEDULE_SWEEP = [(2, 3), (3, 4), (0, 4)]
ALPHA_SWEEP = [8.0, 16.0]
FIXED_ALPHA_FOR_LAYER_AND_SCHEDULE_SWEEP = 4.0
BASELINE_LAYER = 13
BASELINE_K_RANGE = (0, 1)
VEC_TYPES = ["real", "random"]


def summarize_condition(ep_results, vecs):
    success_rate = float(np.mean([r["success"] for r in ep_results]))
    closed_vals = [r["closed_at_checkpoint"] for r in ep_results if r["closed_at_checkpoint"] is not None]
    frac_closed = float(np.mean(closed_vals)) if closed_vals else float("nan")
    nan_rate = float(np.mean([r["nan_detected"] for r in ep_results]))

    probe_feats = [r["probe_feat_at_checkpoint"] for r in ep_results if r["probe_feat_at_checkpoint"] is not None]
    if probe_feats:
        Xp = vecs["probe_scaler"].transform(np.stack(probe_feats))
        probe_closed_frac = float(vecs["probe_clf"].predict(Xp).mean())
    else:
        probe_closed_frac = float("nan")

    return {
        "success_rate": success_rate,
        "frac_closed_at_checkpoint": frac_closed,
        "independent_probe_closed_frac": probe_closed_frac,
        "n_episodes": len(ep_results),
        "nan_rate": nan_rate,
        "max_hook_delta_observed": float(np.max([r["max_hook_delta"] for r in ep_results])),
    }


def run_layer_group(cfg, model, dataset_stats, capture, task_name, seed, n_episodes,
                     gripper_threshold, layer, jobs, collect_dir, manifest):
    """jobs: list of (label, k_range, alpha, vec_type). Registers hook once for this layer,
    computes steering vectors once for this layer, runs each job, then tears down."""
    vecs = compute_steering_vectors(collect_dir, manifest, task_name, layer, 4, seed=0)
    log_message(f"[layer={layer}] v_steer_norm={vecs['v_steer_norm']:.3f} "
                f"ambient_feat_norm_mean={vecs['ambient_feat_norm_mean']:.3f} "
                f"cos(v_steer,v_scene)={vecs['cos_v_steer_vscene']:.3f} "
                f"probe_train_acc={vecs['probe_train_acc']:.3f}")

    hook = SteeringHook(layer, k_range=BASELINE_K_RANGE)
    hook.register(model)
    capture.layer = layer
    if capture.handle is not None:
        capture.remove()
    capture.register(model)

    results = {}
    for label, k_range, alpha, vec_type in jobs:
        hook.k_range = k_range
        vec = vecs["v_steer"] if vec_type == "real" else vecs["v_random"]
        cond_name = f"L{layer}_k{k_range[0]}-{k_range[1]}_a{alpha}_{vec_type}"
        log_message(f"=== Condition: {cond_name} ===")
        ep_results = run_condition(
            cfg, model, dataset_stats, hook, capture, task_name, cond_name,
            vec, alpha, gripper_threshold, n_episodes, seed,
        )
        summary = summarize_condition(ep_results, vecs)
        ambient_ratio = float(alpha * vecs["v_steer_norm"] / vecs["ambient_feat_norm_mean"])
        summary.update({
            "layer": layer, "k_range": list(k_range), "alpha": alpha, "vec_type": vec_type,
            "alpha_vecnorm_over_ambient_norm": ambient_ratio,
        })
        results[cond_name] = summary
        log_message(
            f"[{cond_name}] success_rate={summary['success_rate']:.2f} "
            f"frac_closed@ck={summary['frac_closed_at_checkpoint']:.2f} "
            f"probe_closed_frac={summary['independent_probe_closed_frac']:.2f} "
            f"nan_rate={summary['nan_rate']:.2f} ambient_ratio={ambient_ratio:.2f}"
        )

    capture.remove()
    hook.remove()
    return results, vecs


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--collect_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task_name", default="PnPCounterToCab")
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--n_episodes", type=int, default=8)
    args = p.parse_args()

    collect_dir = Path(args.collect_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((collect_dir / "multitask_manifest.json").read_text())
    phase_summary = json.loads((collect_dir / "phase_labeling_summary.json").read_text())
    gripper_threshold = phase_summary["gripper_threshold"]

    cfg = PolicyEvalConfig(
        config=args.config, ckpt_path=args.ckpt_path, config_file=args.config_file,
        use_wrist_image=True, num_wrist_images=1, use_proprio=True, normalize_proprio=True,
        unnormalize_actions=True, dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path, trained_with_image_aug=True,
        chunk_size=32, num_open_loop_steps=16, task_name=args.task_name, seed=args.seed,
        randomize_seed=False, deterministic=True, use_variance_scale=False,
        use_jpeg_compression=True, flip_images=True, num_denoising_steps_action=5,
        num_denoising_steps_future_state=1, num_denoising_steps_value=1, data_collection=False,
    )
    set_seed_everywhere(args.seed)

    model, _ = get_model(cfg)
    model.eval()
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    capture = SimpleFeatCapture(BASELINE_LAYER)

    # ── T1 no-op check at baseline layer (mechanism unchanged across layers) ──
    probe_hook = SteeringHook(BASELINE_LAYER, k_range=BASELINE_K_RANGE)
    probe_hook.register(model)
    t1_max_diff = t1_noop_check(cfg, model, dataset_stats, probe_hook, args.task_name, args.seed)
    log_message(f"=== T1 check: max|delta| with alpha=0 hook vs no hook = {t1_max_diff:.2e} ===")
    assert t1_max_diff < 1e-4, f"T1 FAIL: alpha=0 hook is not a no-op (max_diff={t1_max_diff})"
    probe_hook.remove()

    # ── shared alpha=0 baseline (layer/schedule-independent) ──────────────────
    vecs_baseline = compute_steering_vectors(collect_dir, manifest, args.task_name,
                                              BASELINE_LAYER, 4, seed=0)
    hook0 = SteeringHook(BASELINE_LAYER, k_range=BASELINE_K_RANGE)
    hook0.register(model)
    capture.register(model)
    log_message("=== Condition: alpha0 (shared baseline) ===")
    ep0 = run_condition(cfg, model, dataset_stats, hook0, capture, args.task_name, "alpha0",
                         vecs_baseline["v_steer"], 0.0, gripper_threshold, args.n_episodes, args.seed)
    alpha0_summary = summarize_condition(ep0, vecs_baseline)
    log_message(f"[alpha0] success_rate={alpha0_summary['success_rate']:.2f} "
                f"frac_closed@ck={alpha0_summary['frac_closed_at_checkpoint']:.2f}")
    capture.remove()
    hook0.remove()

    all_results = {"alpha0": alpha0_summary}
    vecs_by_layer = {}

    # ── layer sweep (k_range=baseline, alpha=4.0) ──────────────────────────────
    for layer in LAYER_SWEEP:
        jobs = [(None, BASELINE_K_RANGE, FIXED_ALPHA_FOR_LAYER_AND_SCHEDULE_SWEEP, vt)
                for vt in VEC_TYPES]
        res, vecs = run_layer_group(cfg, model, dataset_stats, capture, args.task_name,
                                     args.seed, args.n_episodes, gripper_threshold, layer, jobs,
                                     collect_dir, manifest)
        all_results.update(res)
        vecs_by_layer[layer] = {
            "v_steer_norm": vecs["v_steer_norm"], "ambient_feat_norm_mean": vecs["ambient_feat_norm_mean"],
            "cos_v_steer_v_scene": vecs["cos_v_steer_vscene"], "probe_train_acc": vecs["probe_train_acc"],
        }

    # ── schedule sweep (layer=13, alpha=4.0) ────────────────────────────────────
    jobs = [(None, kr, FIXED_ALPHA_FOR_LAYER_AND_SCHEDULE_SWEEP, vt)
            for kr in SCHEDULE_SWEEP for vt in VEC_TYPES]
    res, vecs = run_layer_group(cfg, model, dataset_stats, capture, args.task_name,
                                 args.seed, args.n_episodes, gripper_threshold, BASELINE_LAYER, jobs,
                                 collect_dir, manifest)
    all_results.update(res)
    vecs_by_layer[BASELINE_LAYER] = {
        "v_steer_norm": vecs["v_steer_norm"], "ambient_feat_norm_mean": vecs["ambient_feat_norm_mean"],
        "cos_v_steer_v_scene": vecs["cos_v_steer_vscene"], "probe_train_acc": vecs["probe_train_acc"],
    }

    # ── alpha sweep (layer=13, k_range=baseline, larger alpha) ─────────────────
    jobs = [(None, BASELINE_K_RANGE, a, vt) for a in ALPHA_SWEEP for vt in VEC_TYPES]
    res, _ = run_layer_group(cfg, model, dataset_stats, capture, args.task_name,
                              args.seed, args.n_episodes, gripper_threshold, BASELINE_LAYER, jobs,
                              collect_dir, manifest)
    all_results.update(res)

    out = {
        "task": args.task_name,
        "baseline_layer": BASELINE_LAYER,
        "baseline_k_range": list(BASELINE_K_RANGE),
        "baseline_note": (
            "layer=13, k_range=(0,1), alpha<=4 の結果は steering_intervention.json "
            "(元の実験) を参照。本ファイルはその一変数拡張 (層/スケジュール/alpha) のみを含む。"
        ),
        "n_episodes_per_condition": args.n_episodes,
        "t1_noop_max_diff": t1_max_diff,
        "vecs_by_layer": vecs_by_layer,
        "conditions": all_results,
    }
    with open(out_dir / "steering_extended_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    log_message(f"Saved: {out_dir / 'steering_extended_sweep.json'}")


if __name__ == "__main__":
    main()
