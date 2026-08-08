"""
steering_vector_variants.py — attractor_verification_report.md §9/§9.1 のさらなる拡張。

ユーザー提案(2件)への対応:

(A) Steeringベクトルの精緻化: 単純な平均差分 v_steer=μ(closed)-μ(open) に加え、
    - v_svm: `compute_steering_vectors()`が既に学習しているLogisticRegression決定境界の
      法線ベクトルを、標準化空間から生特徴空間へ逆変換(coef_/scaler.scale_)し、v_steerと
      同ノルムに揃えたもの。平均差分は等方性ノイズを仮定した最適方向だが、SVM/LRの法線は
      クラス内分散(共分散構造)を考慮する点で異なる。
    - v_actmax: 誤差逆伝播によるactivation-maximization方向。`model.get_x0_fn_from_batch()`
      を`get_action()`が使う`torch.inference_mode()`の外から直接1回呼び出し
      (denoising step k=0: x=x_sigma_max, sigma=sigma_max、STEER_K_RANGE=(0,1)の最初の
      呼び出しに相当)、予測アクションの生のgripperチャンネル(index=6, dim_analysis_v2.py/
      t8_fixed_input_control.py準拠)を対象にBlk-{layer}のaction-token出力への勾配を計算。
      N_GRAD_SAMPLES個の異なるepisode/ノイズドローで方向(単位ベクトル)を平均し、
      v_steerとのcos類似度でサインを決定(生モデル出力のgripperチャンネルの符号規約は
      検証していないため、既に「closed」ラベルで検証済みのv_steerとの整合性を使う防御的処理)。

    技術的注意: 全DiTブロックにselective activation checkpointing (SACConfig,
    mode="mm_only")がかかっているため、backward()実行時にforward hookが1回のepisodeあたり
    最大2回発火しうる(実フォワード1回 + backward時の再計算1回)。本スクリプトのフックは
    「reset後最初の1回だけ捕捉してretain_grad」する設計でこれに対応する
    (/tmp scratchpadのactmax_validate.pyで単一エピソードにてgrad finite/nonzeroを事前検証済み)。

(B) 「タスク破壊」現象のN=30追試: §9.1で示唆されたL13, k=(3,4)後期/k=(0,4)全域スケジュール
    でのreal_v特異的タスク成功率崩壊(p=0.077-0.20, n=8/群で非有意)について、既存の
    steering_extended_sweep.jsonの8episodeに22episode追加し(seed衝突回避のためオフセット)
    プールしてN=30でFisher正確検定を再計算する。

vec_typeごとの評価条件は元の実験(§9.1)と同一化: L13, k=(0,1)早期 / k=(3,4)後期, α=4.0,
8episode/条件(v_svm, v_actmaxの新規テストはコスト抑制のためN=8のまま、既存real/randomと
直接比較できるようにするため)。
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import fisher_exact

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_model, init_t5_text_embeddings_cache, load_dataset_stats,
    get_t5_embedding_from_cache, prepare_images_for_model, rescale_proprio,
    extract_action_chunk_from_latent_sequence, COSMOS_TEMPORAL_COMPRESSION_FACTOR, ACTION_DIM,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    PolicyEvalConfig, create_robocasa_env, prepare_observation,
)
from cosmos_policy.experiments.robot.robocasa.analysis.steering_intervention import (
    ACTION_T_IDX, SimpleFeatCapture, SteeringHook, compute_steering_vectors, run_condition,
    t1_noop_check,
)
from cosmos_policy.experiments.robot.robocasa.analysis.steering_extended_sweep import summarize_condition
from cosmos_policy.utils.utils import duplicate_array, set_seed_everywhere
from cosmos_policy._src.imaginaire.utils import misc

GRIPPER_ACTION_DIM = 6  # raw model action index for gripper (dim_analysis_v2.py:37, t8_fixed_input_control.py:312)
N_GRAD_SAMPLES = 12
N_EXTRA_EPISODES_FOR_N30 = 22  # existing 8 (from steering_extended_sweep.json) + 22 = 30
BASELINE_LAYER = 13
BASELINE_K_RANGE = (0, 1)
LATE_K_RANGE = (3, 4)
FULL_K_RANGE = (0, 4)
FIXED_ALPHA = 4.0


def build_data_batch(cfg, model, dataset_stats, observation, task_desc):
    """Replicates cosmos_utils.get_action()'s data_batch construction (lines ~890-1111),
    minus the enclosing torch.inference_mode() (caller controls the grad context)."""
    text_embedding = get_t5_embedding_from_cache(task_desc)
    all_camera_images = [observation["wrist_image"], observation["primary_image"], observation["secondary_image"]]
    WRIST_IMAGE_IDX, IMAGE_IDX, IMAGE2_IDX = 0, 1, 2
    all_camera_images = prepare_images_for_model(all_camera_images, cfg)

    proprio = observation["proprio"]
    if cfg.normalize_proprio:
        proprio = rescale_proprio(proprio, dataset_stats, non_negative_only=False, scale_multiplier=1.0)

    image_sequence = []
    idx = 0
    primary_image = all_camera_images[IMAGE_IDX]
    blank_image = np.zeros_like(primary_image)
    image_sequence.append(np.expand_dims(np.zeros_like(blank_image), axis=0)); idx += 1

    blank_dup = duplicate_array(blank_image.copy(), total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    image_sequence.append(blank_dup)
    current_proprio_latent_idx = idx; idx += 1

    wrist_image = all_camera_images[WRIST_IMAGE_IDX]
    wrist_dup = duplicate_array(wrist_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    image_sequence.append(wrist_dup)
    current_wrist_image_latent_idx = idx; idx += 1
    current_wrist_image2_latent_idx = -1

    primary_dup = duplicate_array(primary_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    image_sequence.append(primary_dup)
    current_image_latent_idx = idx; idx += 1

    secondary_image = all_camera_images[IMAGE2_IDX]
    secondary_dup = duplicate_array(secondary_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    image_sequence.append(secondary_dup)
    current_image2_latent_idx = idx; idx += 1

    image_sequence.append(blank_dup.copy())
    action_latent_idx = idx; idx += 1

    image_sequence.append(blank_dup.copy())
    future_proprio_latent_idx = idx; idx += 1

    image_sequence.append(wrist_dup.copy())
    future_wrist_image_latent_idx = idx; idx += 1
    future_wrist_image2_latent_idx = -1

    image_sequence.append(primary_dup.copy())
    future_image_latent_idx = idx; idx += 1

    image_sequence.append(secondary_dup.copy())
    future_image2_latent_idx = idx; idx += 1

    image_sequence.append(blank_dup.copy())
    value_latent_idx = idx; idx += 1

    raw = np.concatenate(image_sequence, axis=0)
    raw = np.expand_dims(raw, axis=0)
    raw = np.transpose(raw, (0, 4, 1, 2, 3))
    raw = torch.from_numpy(raw).to(dtype=torch.uint8).cuda()
    proprio_tensor = torch.from_numpy(proprio).reshape(1, -1).to(dtype=torch.bfloat16).cuda()

    data_batch = {
        "dataset_name": "video_data",
        "video": raw,
        "t5_text_embeddings": text_embedding.repeat(1, 1, 1).to(dtype=torch.bfloat16).cuda(),
        "fps": torch.tensor([16], dtype=torch.bfloat16).cuda(),
        "padding_mask": torch.zeros((1, 1, 224, 224), dtype=torch.bfloat16).cuda(),
        "num_conditional_frames": model.config.min_num_conditional_frames,
        "proprio": proprio_tensor,
        "current_proprio_latent_idx": torch.tensor([current_proprio_latent_idx], dtype=torch.int64).cuda(),
        "current_wrist_image_latent_idx": torch.tensor([current_wrist_image_latent_idx], dtype=torch.int64).cuda(),
        "current_wrist_image2_latent_idx": torch.tensor([current_wrist_image2_latent_idx], dtype=torch.int64).cuda(),
        "current_image_latent_idx": torch.tensor([current_image_latent_idx], dtype=torch.int64).cuda(),
        "current_image2_latent_idx": torch.tensor([current_image2_latent_idx], dtype=torch.int64).cuda(),
        "action_latent_idx": torch.tensor([action_latent_idx], dtype=torch.int64).cuda(),
        "future_proprio_latent_idx": torch.tensor([future_proprio_latent_idx], dtype=torch.int64).cuda(),
        "future_wrist_image_latent_idx": torch.tensor([future_wrist_image_latent_idx], dtype=torch.int64).cuda(),
        "future_wrist_image2_latent_idx": torch.tensor([future_wrist_image2_latent_idx], dtype=torch.int64).cuda(),
        "future_image_latent_idx": torch.tensor([future_image_latent_idx], dtype=torch.int64).cuda(),
        "future_image2_latent_idx": torch.tensor([future_image2_latent_idx], dtype=torch.int64).cuda(),
        "value_latent_idx": torch.tensor([value_latent_idx], dtype=torch.int64).cuda(),
    }
    return data_batch, action_latent_idx


def compute_svm_vector(vecs):
    """LogisticRegression decision-boundary normal, mapped from standardized space back to raw
    feature space (coef_/scale_), rescaled to ||v_steer|| for a fair equal-dose comparison."""
    scaler, probe = vecs["probe_scaler"], vecs["probe_clf"]
    v_raw = probe.coef_[0] / scaler.scale_
    v_raw_norm = np.linalg.norm(v_raw)
    v_svm = v_raw / (v_raw_norm + 1e-12) * vecs["v_steer_norm"]
    return v_svm


def compute_actmax_vector(cfg, model, dataset_stats, task_name, layer, base_seed, v_steer,
                           n_samples=N_GRAD_SAMPLES, gripper_dim=GRIPPER_ACTION_DIM):
    per_sample_dirs = []
    for i in range(n_samples):
        seed_i = base_seed + 50000 + i
        env, _ = create_robocasa_env(cfg, seed=seed_i, episode_idx=0)
        obs = env.reset()
        for _ in range(10):
            dummy = np.zeros(env.action_spec[0].shape)
            obs, _, _, _ = env.step(dummy)
        task_desc = env.get_ep_meta().get("lang", task_name)
        observation = prepare_observation(obs, cfg.flip_images)
        env.close()

        data_batch, action_latent_idx = build_data_batch(cfg, model, dataset_stats, observation, task_desc)

        captured = {}

        def hook_fn(module, inp, output, _captured=captured):
            if "out" not in _captured and isinstance(output, torch.Tensor) and output.dim() == 5:
                output.retain_grad()
                _captured["out"] = output
            return output

        handle = model.net.blocks[layer].register_forward_hook(hook_fn)
        try:
            with torch.enable_grad():
                model._normalize_video_databatch_inplace(data_batch)
                model._augment_image_dim_inplace(data_batch)
                is_image_batch = model.is_image_batch(data_batch)
                input_key = model.input_image_key if is_image_batch else model.input_data_key
                _T, _H, _W = data_batch[input_key].shape[-3:]
                state_shape = [
                    model.config.state_ch,
                    model.tokenizer.get_latent_num_frames(_T),
                    _H // model.tokenizer.spatial_compression_factor,
                    _W // model.tokenizer.spatial_compression_factor,
                ]
                x_sigma_max = misc.arch_invariant_rand(
                    (1,) + tuple(state_shape), torch.float32, model.tensor_kwargs["device"], seed_i,
                ) * model.sde.sigma_max
                x0_fn = model.get_x0_fn_from_batch(data_batch, 1.5, is_negative_prompt=False)
                sigma_cur = torch.tensor([model.sde.sigma_max], device=x_sigma_max.device, dtype=torch.float32)
                x0_pred = x0_fn(x_sigma_max, sigma_cur)

                action_indices = torch.full((1,), action_latent_idx, dtype=torch.int64, device=x0_pred.device)
                action_chunk = extract_action_chunk_from_latent_sequence(
                    x0_pred, action_shape=(cfg.chunk_size, ACTION_DIM), action_indices=action_indices,
                )
                loss = action_chunk[0, 0, gripper_dim]
                model.zero_grad(set_to_none=True)
                loss.backward()
        finally:
            handle.remove()

        grad = captured.get("out")
        grad_tensor = grad.grad if grad is not None else None
        if grad_tensor is None or not torch.isfinite(grad_tensor).all():
            log_message(f"  [actmax sample {i}] invalid grad, skipping")
            continue
        vec = grad_tensor[0, ACTION_T_IDX].float().mean(dim=(0, 1)).detach().cpu().numpy()
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            continue
        per_sample_dirs.append(vec / norm)
        log_message(f"  [actmax sample {i}] loss={loss.item():.4f} grad_norm={norm:.3e}")

    per_sample_dirs = np.stack(per_sample_dirs)
    mean_dir = per_sample_dirs.mean(axis=0)
    consistency = float(np.linalg.norm(mean_dir))  # ~1 = samples agree on direction, ~0 = noise
    v_actmax_unit = mean_dir / (consistency + 1e-12)

    cos_vs_vsteer = float(np.dot(v_actmax_unit, v_steer) / (np.linalg.norm(v_steer) + 1e-12))
    sign_flipped = cos_vs_vsteer < 0
    if sign_flipped:
        v_actmax_unit = -v_actmax_unit

    v_actmax = v_actmax_unit * np.linalg.norm(v_steer)
    return {
        "v_actmax": v_actmax,
        "n_valid_samples": int(len(per_sample_dirs)),
        "direction_consistency_norm_of_mean_unit_vec": consistency,
        "cos_vs_v_steer_before_sign_fix": cos_vs_vsteer,
        "sign_flipped_to_match_v_steer": sign_flipped,
    }


def run_vector_variant_conditions(cfg, model, dataset_stats, hook, capture, task_name, seed,
                                   n_episodes, gripper_threshold, vecs, v_svm, v_actmax):
    """v_svm / v_actmax at L13, k=(0,1) early and k=(3,4) late, alpha=4.0 -- directly
    comparable to the existing real/random results at these same conditions."""
    results = {}
    for k_range_name, k_range in [("early_k0-1", BASELINE_K_RANGE), ("late_k3-4", LATE_K_RANGE)]:
        hook.k_range = k_range
        for vec_name, vec in [("svm", v_svm), ("actmax", v_actmax)]:
            cond_name = f"L13_{k_range_name}_a{FIXED_ALPHA}_{vec_name}"
            log_message(f"=== Condition: {cond_name} ===")
            ep_results = run_condition(
                cfg, model, dataset_stats, hook, capture, task_name, cond_name,
                vec, FIXED_ALPHA, gripper_threshold, n_episodes, seed,
            )
            summary = summarize_condition(ep_results, vecs)
            ambient_ratio = float(FIXED_ALPHA * float(np.linalg.norm(vec)) / vecs["ambient_feat_norm_mean"])
            summary.update({"layer": BASELINE_LAYER, "k_range": list(k_range), "alpha": FIXED_ALPHA,
                             "vec_type": vec_name, "alpha_vecnorm_over_ambient_norm": ambient_ratio})
            results[cond_name] = summary
            log_message(
                f"[{cond_name}] success_rate={summary['success_rate']:.2f} "
                f"frac_closed@ck={summary['frac_closed_at_checkpoint']:.2f} "
                f"probe_closed_frac={summary['independent_probe_closed_frac']:.2f}"
            )
    return results


def run_n30_extension(cfg, model, dataset_stats, hook, capture, task_name, seed, gripper_threshold, vecs):
    """22 additional episodes (seed-offset to avoid collision with the original 8) for
    real_v/random_v at k=(3,4) late and k=(0,4) full schedules -- pooled with the existing
    8 (from steering_extended_sweep.json) to reach N=30 for the task-success-collapse retest."""
    results = {}
    for k_range_name, k_range in [("late_k3-4", LATE_K_RANGE), ("full_k0-4", FULL_K_RANGE)]:
        hook.k_range = k_range
        for vec_name, vec in [("real", vecs["v_steer"]), ("random", vecs["v_random"])]:
            cond_name = f"L13_{k_range_name}_a{FIXED_ALPHA}_{vec_name}_extra22"
            log_message(f"=== Condition: {cond_name} ===")
            # seed offset by +20000 so these episodes never collide with the original 8
            # (base_seed + ep for ep in 0..7 in the original run).
            ep_results = run_condition(
                cfg, model, dataset_stats, hook, capture, task_name, cond_name,
                vec, FIXED_ALPHA, gripper_threshold, N_EXTRA_EPISODES_FOR_N30, seed + 20000,
            )
            summary = summarize_condition(ep_results, vecs)
            summary.update({"layer": BASELINE_LAYER, "k_range": list(k_range), "alpha": FIXED_ALPHA,
                             "vec_type": vec_name, "n_episodes_this_batch": N_EXTRA_EPISODES_FOR_N30})
            results[cond_name] = summary
            log_message(f"[{cond_name}] success_rate={summary['success_rate']:.2f} "
                        f"(n={N_EXTRA_EPISODES_FOR_N30})")
    return results


def pool_n30_and_test(existing_sweep_json, extra_results):
    """Combine the original 8-episode success/fail counts (from steering_extended_sweep.json)
    with the new 22-episode batch to get N=30 per condition, then Fisher's exact test
    real_v vs random_v success counts at each schedule."""
    existing_conditions = existing_sweep_json["conditions"]
    pooled = {}
    for k_range_name in ["late_k3-4", "full_k0-4"]:
        orig_k_key = "k3-4" if k_range_name == "late_k3-4" else "k0-4"
        counts = {}
        for vec_name in ["real", "random"]:
            orig_cond_name = f"L13_{orig_k_key}_a{FIXED_ALPHA}_{vec_name}"
            orig = existing_conditions[orig_cond_name]
            n_orig = orig["n_episodes"]
            n_success_orig = round(orig["success_rate"] * n_orig)

            extra_cond_name = f"L13_{k_range_name}_a{FIXED_ALPHA}_{vec_name}_extra22"
            extra = extra_results[extra_cond_name]
            n_extra = extra["n_episodes"]
            n_success_extra = round(extra["success_rate"] * n_extra)

            n_total = n_orig + n_extra
            n_success_total = n_success_orig + n_success_extra
            counts[vec_name] = {
                "n_total": n_total, "n_success_total": n_success_total,
                "success_rate_pooled": n_success_total / n_total,
                "n_success_orig8": n_success_orig, "n_success_extra22": n_success_extra,
            }
        table = [[counts["real"]["n_success_total"], counts["real"]["n_total"] - counts["real"]["n_success_total"]],
                 [counts["random"]["n_success_total"], counts["random"]["n_total"] - counts["random"]["n_success_total"]]]
        odds_ratio, p_value = fisher_exact(table)
        pooled[k_range_name] = {
            "real": counts["real"], "random": counts["random"],
            "fisher_exact_p_value_n30": float(p_value), "fisher_exact_odds_ratio": float(odds_ratio),
            "contingency_table_success_fail": table,
        }
        log_message(f"[N30 pooled {k_range_name}] real={counts['real']['success_rate_pooled']:.2f} "
                    f"({counts['real']['n_success_total']}/{counts['real']['n_total']}) vs "
                    f"random={counts['random']['success_rate_pooled']:.2f} "
                    f"({counts['random']['n_success_total']}/{counts['random']['n_total']}) "
                    f"Fisher p={p_value:.4f}")
    return pooled


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
    p.add_argument("--existing_sweep_json", required=True,
                    help="path to steering_extended_sweep.json (for N=30 pooling)")
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
    existing_sweep = json.loads(Path(args.existing_sweep_json).read_text())

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

    vecs = compute_steering_vectors(collect_dir, manifest, args.task_name, BASELINE_LAYER, 4, seed=0)
    log_message(f"v_steer norm={vecs['v_steer_norm']:.3f} ambient_norm={vecs['ambient_feat_norm_mean']:.3f}")

    # ── T1 no-op check (reconfirm at this layer before any new conditions) ────────────
    hook = SteeringHook(BASELINE_LAYER, k_range=BASELINE_K_RANGE)
    hook.register(model)
    t1_max_diff = t1_noop_check(cfg, model, dataset_stats, hook, args.task_name, args.seed)
    log_message(f"=== T1 check: max|delta| with alpha=0 hook vs no hook = {t1_max_diff:.2e} ===")
    assert t1_max_diff < 1e-4, f"T1 FAIL: alpha=0 hook is not a no-op (max_diff={t1_max_diff})"
    hook.remove()

    # ── (A1) v_svm ──────────────────────────────────────────────────────────────────
    v_svm = compute_svm_vector(vecs)
    log_message(f"v_svm computed, norm={np.linalg.norm(v_svm):.3f} (rescaled to ||v_steer||) "
                f"cos(v_svm,v_steer)={float(np.dot(v_svm, vecs['v_steer']) / (np.linalg.norm(v_svm) * vecs['v_steer_norm'] + 1e-12)):.3f}")

    # ── (A2) v_actmax (backprop) ────────────────────────────────────────────────────
    log_message(f"Computing v_actmax via backprop over {N_GRAD_SAMPLES} fresh episode resets...")
    actmax_result = compute_actmax_vector(cfg, model, dataset_stats, args.task_name, BASELINE_LAYER,
                                           args.seed, vecs["v_steer"])
    v_actmax = actmax_result["v_actmax"]
    log_message(f"v_actmax computed: n_valid={actmax_result['n_valid_samples']}/{N_GRAD_SAMPLES} "
                f"consistency={actmax_result['direction_consistency_norm_of_mean_unit_vec']:.3f} "
                f"cos_vs_v_steer_before_sign_fix={actmax_result['cos_vs_v_steer_before_sign_fix']:.3f} "
                f"sign_flipped={actmax_result['sign_flipped_to_match_v_steer']}")

    # ── run vector-variant conditions (early/late x svm/actmax) ────────────────────
    capture = SimpleFeatCapture(BASELINE_LAYER)
    hook = SteeringHook(BASELINE_LAYER, k_range=BASELINE_K_RANGE)
    hook.register(model)
    capture.register(model)
    variant_results = run_vector_variant_conditions(
        cfg, model, dataset_stats, hook, capture, args.task_name, args.seed,
        args.n_episodes, gripper_threshold, vecs, v_svm, v_actmax,
    )
    capture.remove()
    hook.remove()

    # ── (B) N=30 extension (22 more episodes at late/full schedules, real/random) ──
    capture = SimpleFeatCapture(BASELINE_LAYER)
    hook = SteeringHook(BASELINE_LAYER, k_range=LATE_K_RANGE)
    hook.register(model)
    capture.register(model)
    n30_extra_results = run_n30_extension(
        cfg, model, dataset_stats, hook, capture, args.task_name, args.seed, gripper_threshold, vecs,
    )
    capture.remove()
    hook.remove()

    n30_pooled = pool_n30_and_test(existing_sweep, n30_extra_results)

    out = {
        "task": args.task_name,
        "baseline_layer": BASELINE_LAYER,
        "n_episodes_per_variant_condition": args.n_episodes,
        "t1_noop_max_diff": t1_max_diff,
        "v_svm_norm": float(np.linalg.norm(v_svm)),
        "v_svm_cos_vs_v_steer": float(np.dot(v_svm, vecs["v_steer"]) / (np.linalg.norm(v_svm) * vecs["v_steer_norm"] + 1e-12)),
        "actmax_computation": {k: v for k, v in actmax_result.items() if k != "v_actmax"},
        "v_actmax_norm": float(np.linalg.norm(v_actmax)),
        "v_actmax_cos_vs_v_steer": float(np.dot(v_actmax, vecs["v_steer"]) / (np.linalg.norm(v_actmax) * vecs["v_steer_norm"] + 1e-12)),
        "v_actmax_cos_vs_v_svm": float(np.dot(v_actmax, v_svm) / (np.linalg.norm(v_actmax) * np.linalg.norm(v_svm) + 1e-12)),
        "vector_variant_conditions": variant_results,
        "n30_extra_episode_conditions": n30_extra_results,
        "n30_pooled_fisher_test": n30_pooled,
    }
    with open(out_dir / "steering_vector_variants.json", "w") as f:
        json.dump(out, f, indent=2)
    log_message(f"Saved: {out_dir / 'steering_vector_variants.json'}")


if __name__ == "__main__":
    main()
