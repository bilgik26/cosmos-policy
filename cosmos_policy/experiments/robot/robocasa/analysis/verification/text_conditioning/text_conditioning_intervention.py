"""text_conditioning_intervention.py — attractor_verification_report.md §9系列の拡張。
§14 未実施項目 #8: 「中間層への直接書き込み(steering)」と「拡散モデル本来の条件付け機構
(CFG的なテキスト指示側介入)」の効果の違いを比較する、最後に残っていた未検証項目。

設計:
  steering_intervention.py の SteeringHook は model.net.blocks[13] の出力(隠れ状態)に
  直接 alpha*vec を加算する、モデルの本来の計算経路の「外側」からの介入だった。
  本スクリプトは、モデルに実際に実装されている(しかしRoboCasa推論では常時
  is_negative_prompt=False で無効化されている)classifier-free guidanceの式
      raw_x0 = cond_x0 + guidance*(cond_x0 - uncond_x0)
  (policy_video2world_model.py denoise()呼び出し元、get_x0_fn_from_batch内)を実際に
  起動し、uncond側のテキスト埋め込みを固定の「対比」フレーズに差し替えることで、
  モデル本来の条件付け経路(全27層のcross-attention)を介した介入を行う。
  cond側は各エピソードの実際のタスク指示文のまま変更しない(通常運用と同一)。

  対比フレーズ2種 (precompute_text_directions.py で事前計算、T5-11bはpolicyモデルと
  同時にGPUメモリを共有しないよう完全に別プロセスで計算済み):
    - real_text:   uncond = "Keep the gripper open." (グリッパー開閉に意味的に関連)
    - random_text: uncond = "The weather today is sunny." (意味的に無関係、内容非特異性の統制)
  guidance(alpha)を大きくするほど cond - uncond 方向への外挿が強まる。
  real_textでは「'gripper open'から離れる」方向 = 大まかに「'closed'に近づく」方向、
  random_textは同じ機構をタスクと無関係な内容差分に対して適用したコントロール。

  steeringのk_range(1エピソード内5ステップの一部)に相当する概念は、テキスト条件付けが
  1回のget_action呼び出し内の全デノイジングステップに一様に効くため直接は移植できない。
  代わりに「call_idx(行動チャンク呼び出し番号)」粒度でスケジュールを変える:
    - all:  call_idx=0から常時介入 (steeringの「全域k0-4」に相当する概念的アナロジー)
    - late: call_idx=CHECKPOINT_CALL_IDX(=2)以降のみ介入 (steeringの「後期k3-4」に相当)
  これは steering の「ステップ内」粒度とは異なる「呼び出し間」粒度であり、両者は
  完全に同一のスケジュール軸ではない — 本質的な構造の違いとして正直に報告する。

  指標は steering_intervention.py と共通化 (success_rate, frac_closed_at_checkpoint,
  independent_probe_closed_frac) し、直接比較可能にする。independent probeは
  steering_intervention.compute_steering_vectors() が学習した同一のLR(Blk-13, k=4)を
  再利用 — こちらは「介入していないBlk-13の隠れ状態が、条件付け側介入だけでどれだけ
  'closed'方向に動くか」を見る、完全に独立な読み出しである。
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from scipy.stats import fisher_exact

from cosmos_policy.experiments.robot.cosmos_utils import get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import TASK_MAX_STEPS, PolicyEvalConfig, create_robocasa_env, prepare_observation
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.steering_intervention import (
    SimpleFeatCapture, compute_steering_vectors, sanitize_actions,
)
from cosmos_policy.utils.utils import set_seed_everywhere

BASELINE_LAYER = 13  # steering実験と同一層で独立プローブを読み出す(比較可能性のため)
CHECKPOINT_CALL_IDX = 2
N_EPISODES_PER_CONDITION = 8
ALPHAS = [1.0, 2.0, 4.0]  # guidance strength, steeringのALPHAグリッドと同じ値を再利用


class TextCFGHook:
    """model.generate_samples_from_batch と model.get_x0_fn_from_batch をパッチし、
    is_active() の間だけ is_negative_prompt=True + カスタムneg_t5_text_embeddings +
    guidance=alpha でモデル本来のCFG経路を起動する。SteeringHookの「隠れ状態への
    直接加算」とは対照的な、「条件付け入力の置き換え」による介入。"""

    def __init__(self):
        self.neg_embedding = None  # (1,512,1024) tensor, 固定の対比フレーズ埋め込み
        self.alpha = 0.0  # guidance strength (dose)
        self.call_idx = -1
        self.active_from_call = None  # None=常に不介入; int=この呼び出し番号以降介入
        self.last_active = False  # 直近の呼び出しで実際にCFG経路を起動したか(検証用)

    def reset_call_counter(self):
        self.call_idx = -1

    def before_call(self):
        self.call_idx += 1

    def is_active(self):
        return (self.alpha != 0.0) and (self.active_from_call is not None) and (self.call_idx >= self.active_from_call)


def get_action_with_text_cfg(cfg, model, dataset_stats, obs, task_desc, hook, seed, n_steps, capture=None):
    hook.before_call()
    orig_generate = model.generate_samples_from_batch
    orig_x0fn = model.get_x0_fn_from_batch

    def patched_generate(data_batch, **kwargs):
        if hook.is_active():
            data_batch = dict(data_batch)
            data_batch["neg_t5_text_embeddings"] = hook.neg_embedding.to(
                dtype=data_batch["t5_text_embeddings"].dtype, device=data_batch["t5_text_embeddings"].device,
            )
            kwargs["is_negative_prompt"] = True
            kwargs["guidance"] = hook.alpha
            hook.last_active = True
        else:
            hook.last_active = False
        return orig_generate(data_batch, **kwargs)

    def patched_x0fn(data_batch, guidance, **kwargs):
        result = orig_x0fn(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            fn, extra = result

            def w(x, s):
                if capture is not None:
                    capture.before_denoise_step()
                return fn(x, s)

            return w, extra
        else:
            def w(x, s):
                if capture is not None:
                    capture.before_denoise_step()
                return result(x, s)

            return w

    model.generate_samples_from_batch = patched_generate
    model.get_x0_fn_from_batch = patched_x0fn
    try:
        res = get_action(cfg=cfg, model=model, dataset_stats=dataset_stats, obs=obs,
                          task_label_or_embedding=task_desc, seed=seed, randomize_seed=False,
                          num_denoising_steps_action=n_steps, generate_future_state_and_value_in_parallel=False)
    finally:
        model.generate_samples_from_batch = orig_generate
        model.get_x0_fn_from_batch = orig_x0fn
    return res


def _reset_env_get_task_desc(cfg, seed, episode_idx, task_name):
    env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=episode_idx)
    obs = env.reset()
    for _ in range(10):
        dummy = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(dummy)
    task_desc = env.get_ep_meta().get("lang", task_name)
    return env, obs, task_desc


def t1_noop_check(cfg, model, dataset_stats, hook, task_name, base_seed):
    """active_from_call=None (常に不介入) の時、パッチあり/なしで出力が完全一致することを
    確認する。SteeringHookのT1と同じ役割だが、こちらは不介入時に元の関数をそのまま
    呼ぶだけなので理論上は厳密に0差分になるはず(steeringのalpha=0はhookのコード経路は
    通るが効果が0になる設計だったのに対し、こちらは経路自体を素通りする、より強い保証)。"""
    env, obs, task_desc = _reset_env_get_task_desc(cfg, base_seed + 9999, 0, task_name)
    observation = prepare_observation(obs, cfg.flip_images)
    env.close()

    hook.neg_embedding = torch.zeros(1, 512, 1024)
    hook.alpha = 0.0
    hook.active_from_call = None
    res_with_patch = get_action_with_text_cfg(cfg, model, dataset_stats, observation, task_desc,
                                               hook, base_seed + 12345, cfg.num_denoising_steps_action)
    res_without_patch = get_action(cfg=cfg, model=model, dataset_stats=dataset_stats, obs=observation,
                                    task_label_or_embedding=task_desc, seed=base_seed + 12345,
                                    randomize_seed=False, num_denoising_steps_action=cfg.num_denoising_steps_action,
                                    generate_future_state_and_value_in_parallel=False)
    max_diff = float(np.max(np.abs(
        np.stack(res_with_patch["actions"]) - np.stack(res_without_patch["actions"])
    )))
    return max_diff


def sanity_nonzero_effect_check(cfg, model, dataset_stats, hook, task_name, base_seed, neg_embedding):
    """「介入機構が実際に配線されている」ことの陽性対照: alpha=4, active_from_call=0で
    ベースラインと異なる出力が出ることを事前に確認する(そうでなければ、後段のnull結果が
    「効果なし」なのか「配線ミスで何も起きていない」のか区別できない)。"""
    env, obs, task_desc = _reset_env_get_task_desc(cfg, base_seed + 8888, 0, task_name)
    observation = prepare_observation(obs, cfg.flip_images)
    env.close()

    hook.neg_embedding = neg_embedding
    hook.alpha = 4.0
    hook.active_from_call = 0
    res_active = get_action_with_text_cfg(cfg, model, dataset_stats, observation, task_desc,
                                           hook, base_seed + 54321, cfg.num_denoising_steps_action)
    assert hook.last_active, "sanity check: hook did not actually engage the CFG path"

    hook.alpha = 0.0
    hook.active_from_call = None
    res_baseline = get_action_with_text_cfg(cfg, model, dataset_stats, observation, task_desc,
                                             hook, base_seed + 54321, cfg.num_denoising_steps_action)
    max_diff = float(np.max(np.abs(
        np.stack(res_active["actions"]) - np.stack(res_baseline["actions"])
    )))
    return max_diff


def run_condition(cfg, model, dataset_stats, hook, capture, task_name, condition_name,
                   neg_embedding, alpha, active_from_call, gripper_threshold, n_episodes, base_seed):
    hook.neg_embedding = neg_embedding
    hook.alpha = alpha
    hook.active_from_call = active_from_call
    max_steps = TASK_MAX_STEPS.get(task_name, 500)

    ep_results = []
    for ep in range(n_episodes):
        env, _ = create_robocasa_env(cfg, seed=base_seed + ep, episode_idx=ep)
        obs = env.reset()
        for _ in range(10):
            dummy = np.zeros(env.action_spec[0].shape)
            obs, _, _, _ = env.step(dummy)
        task_desc = env.get_ep_meta().get("lang", task_name)

        action_queue = deque()
        success = False
        call_idx = 0
        closed_at_checkpoint = None
        probe_feat_at_checkpoint = None
        t_step = 0
        nan_detected = False
        hook.reset_call_counter()

        while call_idx <= max(CHECKPOINT_CALL_IDX, 20) and t_step < max_steps:
            if len(action_queue) == 0:
                observation = prepare_observation(obs, cfg.flip_images)
                call_seed = base_seed + ep * 131 + call_idx
                capture.reset()
                result = get_action_with_text_cfg(
                    cfg, model, dataset_stats, observation, task_desc, hook, call_seed,
                    cfg.num_denoising_steps_action, capture=capture,
                )

                if call_idx == CHECKPOINT_CALL_IDX:
                    gripper_width = float(np.abs(obs["robot0_gripper_qpos"]).sum())
                    closed_at_checkpoint = int(gripper_width < gripper_threshold)
                    probe_feat_at_checkpoint = capture.get_final(cfg.num_denoising_steps_action - 1)

                actions, had_nan = sanitize_actions(result["actions"])
                nan_detected = nan_detected or had_nan
                call_idx += 1
                for i in range(min(cfg.num_open_loop_steps, len(actions))):
                    a = actions[i]
                    if a.shape[-1] == 7 and env.action_dim == 12:
                        a = np.concatenate([a, np.array([0., 0., 0., 0., -1.])])
                    action_queue.append(a)

            if action_queue:
                action = action_queue.popleft()
                obs, _, _, _ = env.step(action)
                t_step += 1
                if env._check_success():
                    success = True
                    break

        while not success and t_step < max_steps:
            if len(action_queue) == 0:
                observation = prepare_observation(obs, cfg.flip_images)
                call_seed = base_seed + ep * 131 + call_idx
                result = get_action_with_text_cfg(
                    cfg, model, dataset_stats, observation, task_desc, hook, call_seed,
                    cfg.num_denoising_steps_action,
                )
                actions, had_nan = sanitize_actions(result["actions"])
                nan_detected = nan_detected or had_nan
                call_idx += 1
                for i in range(min(cfg.num_open_loop_steps, len(actions))):
                    a = actions[i]
                    if a.shape[-1] == 7 and env.action_dim == 12:
                        a = np.concatenate([a, np.array([0., 0., 0., 0., -1.])])
                    action_queue.append(a)
            if action_queue:
                action = action_queue.popleft()
                obs, _, _, _ = env.step(action)
                t_step += 1
                if env._check_success():
                    success = True
                    break

        env.close()
        ep_results.append({
            "success": success,
            "closed_at_checkpoint": closed_at_checkpoint,
            "probe_feat_at_checkpoint": probe_feat_at_checkpoint,
            "nan_detected": nan_detected,
        })
        log_message(
            f"  [{condition_name} ep{ep}] success={success} closed@ck={closed_at_checkpoint} nan={nan_detected}"
        )

    return ep_results


def summarize_condition(ep_results, vecs):
    success_rate = float(np.mean([r["success"] for r in ep_results]))
    closed_vals = [r["closed_at_checkpoint"] for r in ep_results if r["closed_at_checkpoint"] is not None]
    frac_closed = float(np.mean(closed_vals)) if closed_vals else float("nan")

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
        "nan_rate": float(np.mean([r["nan_detected"] for r in ep_results])),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--collect_dir", required=True)
    p.add_argument("--text_directions_path", required=True,
                    help="precompute_text_directions.py の出力 (.pt)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task_name", default="PnPCounterToCab")
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--n_episodes", type=int, default=N_EPISODES_PER_CONDITION)
    args = p.parse_args()

    collect_dir = Path(args.collect_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((collect_dir / "multitask_manifest.json").read_text())
    phase_summary = json.loads((collect_dir / "phase_labeling_summary.json").read_text())
    gripper_threshold = phase_summary["gripper_threshold"]

    text_dirs = torch.load(args.text_directions_path, map_location="cpu")
    neg_open = text_dirs["neg_open"].float()
    neg_filler = text_dirs["neg_filler"].float()
    log_message(f"Loaded text directions: neg_open shape={tuple(neg_open.shape)}, neg_filler shape={tuple(neg_filler.shape)}")

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

    # 独立プローブは steering_intervention.py と全く同じ held-out データ・同じ層/kで学習
    # (=circularityなし、直接比較可能)
    vecs = compute_steering_vectors(collect_dir, manifest, args.task_name, BASELINE_LAYER, 4, seed=0)
    log_message(f"probe_train_acc={vecs['probe_train_acc']:.3f} (n_train={vecs['n_train_calls']})")

    hook = TextCFGHook()

    # ── T1: 不介入時は完全に元の関数と同一(P7相当) ─────────────────────────
    t1_max_diff = t1_noop_check(cfg, model, dataset_stats, hook, args.task_name, args.seed)
    log_message(f"=== T1 check: max|delta| with inactive patch vs no patch = {t1_max_diff:.2e} ===")
    assert t1_max_diff < 1e-4, f"T1 FAIL: inactive patch is not a no-op (max_diff={t1_max_diff})"

    # ── 陽性対照: 介入機構が実際にCFG経路を起動し、出力を変えることを確認 ──────
    sanity_diff_real = sanity_nonzero_effect_check(cfg, model, dataset_stats, hook, args.task_name, args.seed, neg_open)
    sanity_diff_random = sanity_nonzero_effect_check(cfg, model, dataset_stats, hook, args.task_name, args.seed, neg_filler)
    log_message(f"=== sanity check: max|delta| at alpha=4,active_from_call=0 vs baseline: "
                f"real_text={sanity_diff_real:.4f} random_text={sanity_diff_random:.4f} ===")
    assert sanity_diff_real > 1e-6 and sanity_diff_random > 1e-6, (
        "sanity check FAIL: CFG intervention produced zero effect -- likely a wiring bug, "
        "not evidence of 'no causal effect'"
    )

    capture = SimpleFeatCapture(BASELINE_LAYER)
    capture.register(model)

    # ── baseline (alpha=0, 常に不介入) ──────────────────────────────────────
    log_message("=== Condition: baseline ===")
    baseline_results = run_condition(
        cfg, model, dataset_stats, hook, capture, args.task_name, "baseline",
        neg_open, 0.0, None, gripper_threshold, args.n_episodes, args.seed,
    )
    baseline_summary = summarize_condition(baseline_results, vecs)
    log_message(f"[baseline] success_rate={baseline_summary['success_rate']:.2f} "
                f"frac_closed@ck={baseline_summary['frac_closed_at_checkpoint']:.2f}")

    all_results = {"baseline": baseline_summary}
    for direction_name, neg_emb in [("real_text", neg_open), ("random_text", neg_filler)]:
        for schedule_name, active_from_call in [("all", 0), ("late", CHECKPOINT_CALL_IDX)]:
            for alpha in ALPHAS:
                cond_name = f"{direction_name}_{schedule_name}_g{alpha}"
                log_message(f"=== Condition: {cond_name} ===")
                ep_results = run_condition(
                    cfg, model, dataset_stats, hook, capture, args.task_name, cond_name,
                    neg_emb, alpha, active_from_call, gripper_threshold, args.n_episodes, args.seed,
                )
                summary = summarize_condition(ep_results, vecs)
                summary.update({"direction": direction_name, "schedule": schedule_name,
                                 "active_from_call": active_from_call, "guidance": alpha})
                all_results[cond_name] = summary
                log_message(
                    f"[{cond_name}] success_rate={summary['success_rate']:.2f} "
                    f"frac_closed@ck={summary['frac_closed_at_checkpoint']:.2f} "
                    f"probe_closed_frac={summary['independent_probe_closed_frac']:.2f}"
                )

    capture.remove()

    # dose-response slope (frac_closed_at_checkpoint vs guidance), real vs random, per schedule
    dose_response = {}
    for schedule_name in ["all", "late"]:
        real_fracs = [baseline_summary["frac_closed_at_checkpoint"]] + \
                     [all_results[f"real_text_{schedule_name}_g{a}"]["frac_closed_at_checkpoint"] for a in ALPHAS]
        random_fracs = [baseline_summary["frac_closed_at_checkpoint"]] + \
                       [all_results[f"random_text_{schedule_name}_g{a}"]["frac_closed_at_checkpoint"] for a in ALPHAS]
        doses = [0.0] + ALPHAS
        real_slope = float(np.polyfit(doses, real_fracs, 1)[0])
        random_slope = float(np.polyfit(doses, random_fracs, 1)[0])
        dose_response[schedule_name] = {
            "real_text_slope": real_slope, "random_text_slope": random_slope,
            "direction_specific": bool(real_slope > 2 * abs(random_slope) and real_slope > 0),
        }

    # success-rate collapse test (real vs random) at guidance=4, mirroring steering §9.1's
    # "task destruction" comparison -- Fisher exact test, n=8/群 (underpowered like the
    # original §9.1 pre-N=30 result; reported as suggestive only, consistent with that
    # earlier finding's own caveats).
    collapse_tests = {}
    for schedule_name in ["all", "late"]:
        real = all_results[f"real_text_{schedule_name}_g4.0"]
        random_ = all_results[f"random_text_{schedule_name}_g4.0"]
        n_real, n_random = real["n_episodes"], random_["n_episodes"]
        s_real = round(real["success_rate"] * n_real)
        s_random = round(random_["success_rate"] * n_random)
        table = [[s_real, n_real - s_real], [s_random, n_random - s_random]]
        odds_ratio, p_value = fisher_exact(table)
        collapse_tests[schedule_name] = {
            "real_text_success": f"{s_real}/{n_real}", "random_text_success": f"{s_random}/{n_random}",
            "fisher_exact_p_value": float(p_value), "contingency_table": table,
        }

    out = {
        "task": args.task_name,
        "checkpoint_call_idx": CHECKPOINT_CALL_IDX,
        "n_episodes_per_condition": args.n_episodes,
        "gripper_threshold_reused_from_phase_labeling": gripper_threshold,
        "t1_noop_max_diff": t1_max_diff,
        "sanity_nonzero_effect_max_diff": {"real_text": sanity_diff_real, "random_text": sanity_diff_random},
        "probe_train_acc": vecs["probe_train_acc"],
        "conditions": all_results,
        "dose_response": dose_response,
        "collapse_tests_guidance4": collapse_tests,
    }
    with open(out_dir / "text_conditioning_intervention.json", "w") as f:
        json.dump(out, f, indent=2)
    log_message(f"Saved: {out_dir / 'text_conditioning_intervention.json'}")


if __name__ == "__main__":
    main()
