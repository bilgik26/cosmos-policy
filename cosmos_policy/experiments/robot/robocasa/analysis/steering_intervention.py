"""
steering_intervention.py — attractor_verification_design.md §6.2 準拠 (因果テスト)

「早期kでのsteeringで、条件付けと異なるターゲットスキルが行動として発現する」を検証する
最も強い因果テスト。P6 (定義した空間の外＝行動で検証)・P7 (α=0でno-op) を厳守。

方向 v_steer = μ(gripper_closed) − μ(gripper_open) を **主収集データ (collect_multitask.py の
出力、本スクリプトのテストロールアウトとは完全に独立な held-out データ)** から計算する
(循環回避)。早期k (5ステップスケジュールの最初の2ステップ, 20-40%) で Blk-13 の
action-token 出力に α·v を加算し、フレッシュな closed-loop ロールアウトで行動を観測する。

コントロール:
  (c1) ランダム方向 (v と同ノルムの無作為ベクトル)
  (c2) シーン直交化 v (episode平均間の最大分散方向を除去)
  (c3) α=0 (no-op, T1でフックが無害であることを確認)

行動指標:
  - frac_closed_at_checkpoint: call_idx==CHECKPOINT_CALL 時点でgripperが閉じている割合
    (ターゲット行動「早期グラスプ」の発現率, Intervention Success Rate に相当)
  - task success_rate (元スキルの成功率、低下するかを見る)
  - independent probe: 主データで学習した (gripper open/closed) LR プローブを steered
    特徴量に適用した場合の "closed" 予測率 (行動と無関係な独立検証)
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    TASK_MAX_STEPS, PolicyEvalConfig, create_robocasa_env, prepare_observation,
)
from cosmos_policy.utils.utils import set_seed_everywhere

STEER_LAYER = 13
STEER_K_RANGE = (0, 1)  # inject during k=0,1 of the 5-step schedule (early 20-40%)
ACTION_T_IDX = 5
CHECKPOINT_CALL_IDX = 2
N_EPISODES_PER_CONDITION = 8
ALPHAS = [1.0, 2.0, 4.0]
# NOTE: STEER_LAYER/STEER_K_RANGE/ALPHAS above remain the historical defaults (used by the
# original single-condition-matrix run). The extended sweep (multi-layer / multi-schedule /
# larger alpha, see report §9.x) drives these via CLI args instead of module constants —
# see main()'s --steer_layer/--k_range/--alphas and SteeringHook.k_range.


def compute_steering_vectors(collect_dir: Path, manifest, task_name: str, layer: int, k_step: int, seed=0):
    """v_steer (closed - open centroid), scene direction (top PC of per-episode means),
    v_steer_orth (scene-direction removed), and a fitted open/closed LR probe.
    All computed from the MAIN collected data (held out from steering test rollouts)."""
    feats_all, gripper_all, episode_all = [], [], []
    ep_offset = 0
    for fname, info in manifest["files"].items():
        if info["task"] != task_name:
            continue
        fd = np.load(collect_dir / fname)
        pd = np.load(collect_dir / fname.replace(".npz", "_phases.npz"))
        key = f"feat_k{k_step}_layer{layer}"
        idx_key = key + "_idx"
        keep_idx = fd[idx_key]
        feats_all.append(fd[key])
        gripper_all.append(pd["gripper_state"][keep_idx])
        episode_all.append(fd["episode"][keep_idx] + ep_offset)
        ep_offset += int(fd["episode"].max()) + 1

    X = np.concatenate(feats_all)
    gripper = np.concatenate(gripper_all)
    episode = np.concatenate(episode_all)

    mu_open = X[gripper == 0].mean(axis=0)
    mu_closed = X[gripper == 1].mean(axis=0)
    v_steer = mu_closed - mu_open

    # scene direction: top PC of per-episode mean features
    ep_means = np.stack([X[episode == e].mean(axis=0) for e in np.unique(episode)])
    ep_means_c = ep_means - ep_means.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(ep_means_c, full_matrices=False)
    v_scene = Vt[0]
    v_scene = v_scene / (np.linalg.norm(v_scene) + 1e-12)

    v_steer_orth = v_steer - np.dot(v_steer, v_scene) * v_scene

    scaler = StandardScaler().fit(X)
    probe = LogisticRegression(max_iter=2000).fit(scaler.transform(X), gripper)
    probe_train_acc = probe.score(scaler.transform(X), gripper)

    rng = np.random.RandomState(seed)
    v_random = rng.normal(size=v_steer.shape)
    v_random = v_random / np.linalg.norm(v_random) * np.linalg.norm(v_steer)

    ambient_norm = float(np.linalg.norm(X, axis=1).mean())

    return {
        "v_steer": v_steer,
        "v_steer_orth": v_steer_orth,
        "v_random": v_random,
        "v_steer_norm": float(np.linalg.norm(v_steer)),
        "v_steer_orth_norm": float(np.linalg.norm(v_steer_orth)),
        # true cosine similarity (v_steer is not unit-norm, so this must be normalized
        # explicitly -- see report bug list #6, the un-normalized dot product was previously
        # mislabeled "cosine" and printed a value outside [-1,1]).
        "cos_v_steer_vscene": float(np.dot(v_steer, v_scene) / (np.linalg.norm(v_steer) + 1e-12)),
        "ambient_feat_norm_mean": ambient_norm,
        "probe_scaler": scaler,
        "probe_clf": probe,
        "probe_train_acc": float(probe_train_acc),
        "n_train_calls": int(len(gripper)),
    }


class SteeringHook:
    """model.net.blocks[STEER_LAYER] に登録し、早期k(STEER_K_RANGE)でaction-token出力に
    alpha*vec を加算する。step counter は get_x0_fn_from_batch のラップで駆動する。"""

    def __init__(self, layer, k_range=STEER_K_RANGE):
        self.layer = layer
        self.k_range = k_range
        self.step = -1
        self.alpha = 0.0
        self.vec = None
        self.handle = None
        self.last_delta_max = 0.0  # for T1 sanity check

    def register(self, model):
        block = model.net.blocks[self.layer]
        self.handle = block.register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()

    def reset_step(self):
        self.step = -1

    def before_step(self):
        self.step += 1

    def _hook(self, module, inp, output):
        if not (isinstance(output, torch.Tensor) and output.dim() == 5):
            return output
        if self.vec is None or self.alpha == 0.0 or not (self.k_range[0] <= self.step <= self.k_range[1]):
            self.last_delta_max = 0.0
            return output
        vec_t = torch.tensor(self.vec, dtype=output.dtype, device=output.device)
        delta = self.alpha * vec_t
        self.last_delta_max = float(delta.abs().max().item())
        output = output.clone()
        output[0, ACTION_T_IDX] = output[0, ACTION_T_IDX] + delta
        return output


def get_action_with_steering(cfg, model, dataset_stats, obs, task_desc, hook, seed, n_steps,
                              capture=None):
    hook.reset_step()
    orig = model.get_x0_fn_from_batch

    def patched(data_batch, guidance, **kwargs):
        result = orig(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            fn, extra = result
            def w(x, s):
                hook.before_step()
                if capture is not None:
                    capture.before_denoise_step()
                return fn(x, s)
            return w, extra
        else:
            def w(x, s):
                hook.before_step()
                if capture is not None:
                    capture.before_denoise_step()
                return result(x, s)
            return w

    model.get_x0_fn_from_batch = patched
    try:
        res = get_action(cfg=cfg, model=model, dataset_stats=dataset_stats, obs=obs,
                          task_label_or_embedding=task_desc, seed=seed, randomize_seed=False,
                          num_denoising_steps_action=n_steps, generate_future_state_and_value_in_parallel=False)
    finally:
        model.get_x0_fn_from_batch = orig
    return res


class SimpleFeatCapture:
    """Blk-13 の全kの action-token特徴を1 callぶん保持 (independent probe用)。"""
    def __init__(self, layer):
        self.layer = layer
        self._step = -1
        self._cur = {}
        self.handle = None

    def register(self, model):
        self.handle = model.net.blocks[self.layer].register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()

    def before_denoise_step(self):
        self._step += 1

    def reset(self):
        self._step = -1
        self._cur = {}

    def get_final(self, k):
        return self._cur.get(k)

    def _hook(self, module, inp, out):
        if self._step < 0 or not (isinstance(out, torch.Tensor) and out.dim() == 5):
            return
        feat = out[0, ACTION_T_IDX].float().mean(dim=(0, 1))
        self._cur[self._step] = feat.detach().cpu().numpy()


def sanitize_actions(actions):
    """Large alpha can push the denoiser into a regime that produces non-finite actions.
    Detect this rather than letting MuJoCo crash on a NaN/Inf action, and report it --
    a large-alpha condition with a high nan_rate is itself informative (destructive but
    non-specific effect, distinct from a direction-specific behavioral effect)."""
    arr = np.stack(actions) if len(actions) else np.zeros((0,))
    had_nan = bool(arr.size and not np.all(np.isfinite(arr)))
    if had_nan:
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return [arr[i] for i in range(len(actions))], had_nan


def run_condition(cfg, model, dataset_stats, hook, capture, task_name, condition_name,
                   vec, alpha, gripper_threshold, n_episodes, base_seed):
    hook.vec = vec
    hook.alpha = alpha
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
        max_delta_this_ep = 0.0
        nan_detected = False

        while call_idx <= max(CHECKPOINT_CALL_IDX, 20) and t_step < max_steps:
            if len(action_queue) == 0:
                observation = prepare_observation(obs, cfg.flip_images)
                call_seed = base_seed + ep * 131 + call_idx
                capture.reset()
                result = get_action_with_steering(
                    cfg, model, dataset_stats, observation, task_desc, hook, call_seed,
                    cfg.num_denoising_steps_action, capture=capture,
                )
                max_delta_this_ep = max(max_delta_this_ep, hook.last_delta_max)

                if call_idx == CHECKPOINT_CALL_IDX:
                    # opening-width proxy (NOT mean(): panda's 2 fingers have near-
                    # opposite-sign qpos, see phase_labeling.py).
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

        # finish episode to get a real success readout (don't cut early if still running)
        while not success and t_step < max_steps:
            if len(action_queue) == 0:
                observation = prepare_observation(obs, cfg.flip_images)
                call_seed = base_seed + ep * 131 + call_idx
                result = get_action_with_steering(
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
            "max_hook_delta": max_delta_this_ep,
            "nan_detected": nan_detected,
        })
        log_message(
            f"  [{condition_name} ep{ep}] success={success} "
            f"closed@ck={closed_at_checkpoint} max_delta={max_delta_this_ep:.4f} "
            f"nan={nan_detected}"
        )

    return ep_results


def t1_noop_check(cfg, model, dataset_stats, hook, task_name, base_seed):
    """alpha=0時、hook登録あり/なしで出力が完全一致することを確認 (P7)。"""
    env, _ = create_robocasa_env(cfg, seed=base_seed + 9999, episode_idx=0)
    obs = env.reset()
    for _ in range(10):
        dummy = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(dummy)
    task_desc = env.get_ep_meta().get("lang", task_name)
    observation = prepare_observation(obs, cfg.flip_images)
    env.close()

    hook.vec = np.ones(2048)
    hook.alpha = 0.0
    res_with_hook = get_action_with_steering(cfg, model, dataset_stats, observation, task_desc,
                                              hook, base_seed + 12345, cfg.num_denoising_steps_action)
    hook.remove()
    res_without_hook = get_action(cfg=cfg, model=model, dataset_stats=dataset_stats, obs=observation,
                                   task_label_or_embedding=task_desc, seed=base_seed + 12345,
                                   randomize_seed=False, num_denoising_steps_action=cfg.num_denoising_steps_action,
                                   generate_future_state_and_value_in_parallel=False)
    hook.register(model)

    max_diff = float(np.max(np.abs(
        np.stack(res_with_hook["actions"]) - np.stack(res_without_hook["actions"])
    )))
    return max_diff


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
    p.add_argument("--n_episodes", type=int, default=N_EPISODES_PER_CONDITION)
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

    vecs = compute_steering_vectors(collect_dir, manifest, args.task_name, STEER_LAYER, 4, seed=0)
    log_message(f"v_steer norm={vecs['v_steer_norm']:.3f} v_orth norm={vecs['v_steer_orth_norm']:.3f} "
                f"cos(v_steer,v_scene)={vecs['cos_v_steer_vscene']:.3f} "
                f"probe_train_acc={vecs['probe_train_acc']:.3f} (n_train={vecs['n_train_calls']})")

    hook = SteeringHook(STEER_LAYER)
    hook.register(model)
    capture = SimpleFeatCapture(STEER_LAYER)
    capture.register(model)

    # ── T1: alpha=0 no-op check (P7) ──────────────────────────────────────────
    t1_max_diff = t1_noop_check(cfg, model, dataset_stats, hook, args.task_name, args.seed)
    log_message(f"=== T1 check: max|delta| with alpha=0 hook vs no hook = {t1_max_diff:.2e} ===")
    assert t1_max_diff < 1e-4, f"T1 FAIL: alpha=0 hook is not a no-op (max_diff={t1_max_diff})"

    conditions = [("alpha0", vecs["v_steer"], 0.0)]
    for a in ALPHAS:
        conditions.append((f"real_v_alpha{a}", vecs["v_steer"], a))
    for a in ALPHAS:
        conditions.append((f"random_v_alpha{a}", vecs["v_random"], a))
    for a in ALPHAS:
        conditions.append((f"scene_orth_v_alpha{a}", vecs["v_steer_orth"], a))

    all_results = {}
    for cond_name, vec, alpha in conditions:
        log_message(f"=== Condition: {cond_name} (alpha={alpha}) ===")
        ep_results = run_condition(
            cfg, model, dataset_stats, hook, capture, args.task_name, cond_name,
            # NOTE: use the SAME base_seed (hence same episode/env seeds) across all
            # conditions -- a matched/paired design where the only thing that differs
            # between conditions is the steering intervention itself, not which
            # kitchen scenes get sampled.
            vec, alpha, gripper_threshold, args.n_episodes, args.seed,
        )
        success_rate = float(np.mean([r["success"] for r in ep_results]))
        closed_vals = [r["closed_at_checkpoint"] for r in ep_results if r["closed_at_checkpoint"] is not None]
        frac_closed = float(np.mean(closed_vals)) if closed_vals else float("nan")

        probe_feats = [r["probe_feat_at_checkpoint"] for r in ep_results if r["probe_feat_at_checkpoint"] is not None]
        if probe_feats:
            Xp = vecs["probe_scaler"].transform(np.stack(probe_feats))
            probe_closed_frac = float(vecs["probe_clf"].predict(Xp).mean())
        else:
            probe_closed_frac = float("nan")

        all_results[cond_name] = {
            "alpha": alpha,
            "vec_type": "real" if "real" in cond_name or cond_name == "alpha0"
                        else ("random" if "random" in cond_name else "scene_orth"),
            "success_rate": success_rate,
            "frac_closed_at_checkpoint": frac_closed,
            "independent_probe_closed_frac": probe_closed_frac,
            "n_episodes": len(ep_results),
            "max_hook_delta_observed": float(np.max([r["max_hook_delta"] for r in ep_results])),
        }
        log_message(
            f"[{cond_name}] success_rate={success_rate:.2f} frac_closed@ck={frac_closed:.2f} "
            f"probe_closed_frac={probe_closed_frac:.2f}"
        )

    capture.remove()
    hook.remove()

    # dose-response slope for real_v condition
    real_alphas = [0.0] + ALPHAS
    real_fracs = [all_results["alpha0"]["frac_closed_at_checkpoint"]] + \
                 [all_results[f"real_v_alpha{a}"]["frac_closed_at_checkpoint"] for a in ALPHAS]
    slope = float(np.polyfit(real_alphas, real_fracs, 1)[0]) if len(real_alphas) > 1 else float("nan")

    random_fracs = [all_results["alpha0"]["frac_closed_at_checkpoint"]] + \
                   [all_results[f"random_v_alpha{a}"]["frac_closed_at_checkpoint"] for a in ALPHAS]
    random_slope = float(np.polyfit(real_alphas, random_fracs, 1)[0])

    orth_fracs = [all_results["alpha0"]["frac_closed_at_checkpoint"]] + \
                 [all_results[f"scene_orth_v_alpha{a}"]["frac_closed_at_checkpoint"] for a in ALPHAS]
    orth_slope = float(np.polyfit(real_alphas, orth_fracs, 1)[0])

    summary = {
        "task": args.task_name,
        "steer_layer": STEER_LAYER,
        "steer_k_range": STEER_K_RANGE,
        "checkpoint_call_idx": CHECKPOINT_CALL_IDX,
        "n_episodes_per_condition": args.n_episodes,
        "gripper_threshold_reused_from_phase_labeling": gripper_threshold,
        "v_steer_norm": vecs["v_steer_norm"],
        "v_steer_orth_norm": vecs["v_steer_orth_norm"],
        "cos_v_steer_v_scene": vecs["cos_v_steer_vscene"],
        "probe_train_acc": vecs["probe_train_acc"],
        "t1_noop_max_diff": t1_max_diff,
        "conditions": all_results,
        "dose_response_slope_real_v": slope,
        "dose_response_slope_random_v": random_slope,
        "dose_response_slope_scene_orth_v": orth_slope,
        "direction_specific": bool(slope > 2 * abs(random_slope) and slope > 0),
        "verdict": (
            "real_v の dose-response 傾きが random_v の傾きを明確に上回り、方向特異的な"
            "行動因果効果が観測された(「特定スキルを活性化できる」の行動レベル証拠)。"
            if slope > 2 * abs(random_slope) and slope > 0 else
            "real_v の dose-response 傾きが random_v と明確に区別できない、または正でない。"
            "潜在レベルでの操作が行動レベルの因果を持つという主張は本実験では支持されない"
            "(反証: 「潜在は動くが行動に対する因果を持たない」の可能性を含む)。"
        ),
    }

    with open(out_dir / "steering_intervention.json", "w") as f:
        json.dump(summary, f, indent=2)
    log_message(f"Saved: {out_dir / 'steering_intervention.json'}")
    log_message(f"=== VERDICT: {summary['verdict']} ===")


if __name__ == "__main__":
    main()
