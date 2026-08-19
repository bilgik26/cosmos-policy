"""
must_fix_experiments.py — review_report.md (latent_dynamics_verification 査読) MUST-2 / MUST-4 /
M-3 対応。dynamic_vector_field_steering.py が実装したオンライン因果推定器・kNNフローライブラリ・
steeringフックをそのまま再利用し、査読で要求された3種の追加条件のみを新規実装する:

  MUST-2 (positive control / オラクル介入): 同一観測・同一seedで実プロンプト条件とダミープロンプト
    条件のBlk-13特徴の差分 Delta_oracle = h(o_t, e_real) - h(o_t, e_dummy) を、C2と全く同じ注入点
    ・alpha・k_rangeでダミープロンプト条件に注入する。これが「注入チャネルがそもそも所望の行動を
    誘発しうる帯域を持つか」を測る上限実験であり、review_report.md §1 D-2 の MUST-2 に対応する。
    C2/C3/C4同様の1-call遅延構造(このcallで測ったDelta_oracleを次のcallの注入に使う)を踏襲する。

  MUST-4 (言語アブレーションの階層化): dummy sentenceという単一のOOD条件付けだけでは「真に条件付け
    が無い」ことと「分布外文で条件付けられている」ことを区別できない、という review_report.md §1
    D-4 の指摘に対応する。既存のC1(=L4, dummy sentence)に加え:
      L1_zero_embedding    : 零ベクトル埋め込み(真の"条件付けなし"、attention maskは変更しない
                              — 後述の限界参照)
      L2_mean_task_embedding: 収集済み342種類のRoboCasaタスク記述文のT5埋め込みの平均
                              ("言語はあるがタスク特異性なし")
      L3_cross_task_prompt  : 他タスクの実プロンプト("close the left drawer"、PnPCounterToCabの
                              実行中に使う) — 言語の"選択"機能と"発火"機能を分離する最重要条件
    いずれもsteeringなし(C1と同じ比較条件)。

  M-3 (適応的alpha): 査読 §2 M-3 の指摘 — 単位ベクトルへの正規化でエネルギー場勾配/フローの
    "大きさ"(谷の深さ)情報を捨てている — への対応。alpha_t = ADAPTIVE_ALPHA_COEF * ||raw_dir_t||
    とする条件を追加する(dynamic_vector_field_steering.run_conditionのadaptive_alpha_coef引数を
    使うだけで実装できる)。ADAPTIVE_ALPHA_COEFはこのファイル末尾のオフライン較正手順
    (既存のD空間artifactに対しDynamicFlowField.query+xp_direction_to_rawを200点適用し、
    mean(||raw_dir||)*coef=40になるよう較正、実測mean(||raw_dir||)=10.34, coef=3.87)で
    事前に定めた値をハードコードしている(GPU不要、再現手順はこのdocstring末尾参照)。

すべての新条件はPnPCounterToCab、N=8/条件、alpha=40.0(オラクル・C2互換のため)、k_range=(0,1)
(既存条件と同一設定)で実行する。既存のdynamic_vector_field_steering.pyは変更せず(run_condition
へのadaptive_alpha_coef引数追加のみ、デフォルト値None時は既存条件と完全に同一の挙動)、
run_dynamic_vector_field_steering.shが依拠する既存の実行結果には影響しない。

オフライン較正の再現手順:
  python3 -c "
  import pickle, numpy as np
  from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamic_vector_field_steering import DynamicFlowField, xp_direction_to_raw
  with open('cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification/dynamics_embedding_artifact_PnPCounterToCab.pkl','rb') as f:
      art = pickle.load(f)
  field = DynamicFlowField(art, randomize=False, seed=0)
  D, Xp = art['D'], art['Xp']
  rng = np.random.RandomState(0)
  idx = rng.choice(len(D), size=200, replace=False)
  norms = [np.linalg.norm(xp_direction_to_raw(field.query(D[i], Xp[i]), art)) for i in idx]
  print(40.0/np.mean(norms))
  "
"""

import json
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import torch

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import TASK_MAX_STEPS, PolicyEvalConfig, create_robocasa_env, prepare_observation
from cosmos_policy.experiments.robot.cosmos_utils import get_model, load_dataset_stats, init_t5_text_embeddings_cache
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.skill_count import LAYER, K_STEP
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.steering_intervention import sanitize_actions
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamic_vector_field_steering import (
    ACTION_T_IDX, DUMMY_PROMPT, CausalStateEstimator, DynamicFlowField, DynamicFieldHook,
    get_action_with_dynamic_hook, xp_direction_to_raw, t1_noop_check, run_condition,
    preload_dummy_prompt_embedding,
)

ADAPTIVE_ALPHA_COEF = 3.87   # see docstring: calibrated so mean(coef*||raw_dir||) ~= 40 (== other conditions' alpha)
ROBOCASA_T5_EMBEDDINGS_PATH = (
    "/mnt/data/bilgehan.sakai/cosmos-policy/cache/huggingface/"
    "models--nvidia--Cosmos-Policy-RoboCasa-Predict2-2B/snapshots/"
    "4b2a04c80d97202f86127ebec80461e8016ec1dc/robocasa_t5_embeddings.pkl"
)
CROSS_TASK_PROMPT = "close the left drawer"   # a real, in-distribution RoboCasa instruction for A DIFFERENT task


class MultiStepFeatCapture:
    """Generalizes FinalFeatCapture (dynamic_vector_field_steering.py) to capture the mean-pooled
    action-token Blk-13 feature at an arbitrary SET of denoising steps in one forward pass,
    instead of only the last one."""

    def __init__(self, layer, steps):
        self.layer = layer
        self.steps = set(steps)
        self._step = -1
        self.feats = {}
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
        self.feats = {}

    def _hook(self, module, inp, out):
        if self._step in self.steps and isinstance(out, torch.Tensor) and out.dim() == 5:
            self.feats[self._step] = out[0, ACTION_T_IDX].float().mean(dim=(0, 1)).detach().cpu().numpy()

    @property
    def _final(self):
        """dynamic_vector_field_steering.run_condition() (reused verbatim here for the plain
        L1/L2/L3/adaptive-alpha conditions) expects a FinalFeatCapture-shaped `capture._final`
        attribute holding the LAST denoising step's feature -- expose that as the max captured
        step so this class is a drop-in replacement."""
        return self.feats.get(max(self.steps)) if self.feats else None


def build_zero_embedding(reference_path=None, ref_key=DUMMY_PROMPT):
    """L1: true null conditioning. Same shape/dtype as a real T5 embedding, all zeros.
    NOTE (limitation, disclosed in the report): this zeroes the embedding VALUES but does not
    additionally force the model's text cross-attention mask to fully exclude the text tokens --
    we did not locate a plumbed-through flag for that in cosmos_utils.get_action/PolicyEvalConfig,
    and adding one would touch model-forward code shared by every other verification script in
    this project. A zero embedding is still a much closer approximation to "no linguistic
    conditioning" than an OOD dummy sentence (which carries a full, if irrelevant, T5 semantic
    embedding) and is the condition review_report.md D-4 asks be added at minimum."""
    import torch as _torch
    ref = _torch.load(
        "cosmos_policy/experiments/robot/robocasa/analysis/results/attractor_verification/text_directions.pt",
        map_location="cpu",
    )["neg_filler"]
    return np.zeros(tuple(ref.shape), dtype=np.float32)


def build_mean_task_embedding(path=ROBOCASA_T5_EMBEDDINGS_PATH, exclude=(DUMMY_PROMPT,)):
    """L2: mean T5 embedding across all cached RoboCasa task-description strings (342 sentences,
    see docstring). 'Language is present but carries no task-specific selectivity.'"""
    with open(path, "rb") as f:
        cache = pickle.load(f)
    vecs = [v.float().numpy() for k, v in cache.items() if k not in exclude]
    mean_vec = np.mean(np.stack(vecs, axis=0), axis=0)
    return mean_vec.astype(np.float32)


# ─────────────────────────── MUST-2: oracle intervention ───────────────────────────

def run_condition_oracle(cfg, model, dataset_stats, hook, capture, estimator, task_name,
                          real_task_desc, n_episodes, base_seed, alpha, max_call=40):
    """Same rollout loop as dynamic_vector_field_steering.run_condition's C2, except the
    injected direction is Delta_oracle = h_real - h_dummy (both captured at the SAME observation,
    SAME call seed, at the injection step k_range[0]=0) instead of the kNN flow-field direction.
    hook.alpha is forced to 0.0 during the extra "shadow" real-prompt forward pass so that pass's
    captured feature is the model's natural real-prompt representation, uncontaminated by any
    steering already queued from the previous call."""
    max_steps = TASK_MAX_STEPS.get(task_name, 500)
    ep_logs = []
    for ep in range(n_episodes):
        env, _ = create_robocasa_env(cfg, seed=base_seed + ep, episode_idx=ep)
        obs = env.reset()
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))

        estimator.reset_episode()
        hook.vec, hook.alpha = None, 0.0
        action_queue = deque()
        success, call_idx, t_step = False, 0, 0
        eef_traj, grip_traj, action_log = [], [], []
        xp_traj, d_traj, xp_call_idx, oracle_delta_norm_log = [], [], [], []
        max_delta_this_ep, nan_detected = 0.0, False

        while call_idx < max_call and t_step < max_steps and not success:
            if len(action_queue) == 0:
                observation = prepare_observation(obs, cfg.flip_images)
                call_seed = base_seed + ep * 131 + call_idx

                # shadow pass: real prompt, injection forced off, capture step-0 + last-step feats
                saved_vec, saved_alpha = hook.vec, hook.alpha
                hook.alpha = 0.0
                get_action_with_dynamic_hook(cfg, model, dataset_stats, observation, real_task_desc,
                                              hook, capture, call_seed, cfg.num_denoising_steps_action)
                h_real_step0 = capture.feats.get(0)

                # actual pass: dummy prompt, injection = whatever was queued from the PREVIOUS call
                hook.vec, hook.alpha = saved_vec, saved_alpha
                result = get_action_with_dynamic_hook(cfg, model, dataset_stats, observation, DUMMY_PROMPT,
                                                        hook, capture, call_seed, cfg.num_denoising_steps_action)
                max_delta_this_ep = max(max_delta_this_ep, hook.last_delta_max)
                h_dummy_step0 = capture.feats.get(0)
                h_final = capture.feats.get(cfg.num_denoising_steps_action - 1)

                gripper_width = float(np.abs(obs["robot0_gripper_qpos"]).sum())
                eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                if h_final is not None:
                    D_t, Xp_t = estimator.update(h_final, eef_pos, gripper_width)
                    xp_traj.append(Xp_t.tolist())
                    d_traj.append(D_t.tolist())
                    xp_call_idx.append(call_idx)
                if h_real_step0 is not None and h_dummy_step0 is not None:
                    delta_raw = h_real_step0.astype(np.float64) - h_dummy_step0.astype(np.float64)
                    delta_norm = float(np.linalg.norm(delta_raw))
                    oracle_delta_norm_log.append(delta_norm)
                    delta_unit = delta_raw / (delta_norm + 1e-12)
                    hook.vec = delta_unit
                    hook.alpha = alpha

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
                eef_traj.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64).tolist())
                grip_traj.append(float(np.abs(obs["robot0_gripper_qpos"]).sum()))
                action_log.append(np.asarray(action, dtype=np.float64).tolist())
                if env._check_success():
                    success = True
                    break

        env.close()
        ep_logs.append({
            "success": success, "n_calls": call_idx, "n_steps": t_step,
            "eef_traj": eef_traj, "grip_traj": grip_traj, "action_log": action_log,
            "xp_traj": xp_traj, "d_traj": d_traj, "xp_call_idx": xp_call_idx,
            "max_hook_delta": max_delta_this_ep, "nan_detected": nan_detected,
            "oracle_delta_norm_log": oracle_delta_norm_log,
        })
        log_message(f"  [C_oracle ep{ep}] success={success} n_steps={t_step} "
                    f"max_delta={max_delta_this_ep:.4f} mean_oracle_delta_norm="
                    f"{np.mean(oracle_delta_norm_log) if oracle_delta_norm_log else float('nan'):.4f}")

    success_rate = float(np.mean([r["success"] for r in ep_logs]))
    log_message(f"[C_oracle] success_rate={success_rate:.2f} (n={n_episodes})")
    return {"success_rate": success_rate, "episodes": ep_logs}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task_name", default="PnPCounterToCab")
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--n_episodes", type=int, default=8)
    p.add_argument("--alpha", type=float, default=40.0)
    p.add_argument("--k_range", type=int, nargs=2, default=[0, 1])
    p.add_argument("--conditions", nargs="+",
                    default=["C_oracle", "L1_zero_embedding", "L2_mean_task_embedding",
                              "L3_cross_task_prompt", "C2_adaptive_alpha"])
    args = p.parse_args()

    embedding_dir = Path(args.embedding_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(embedding_dir / f"dynamics_embedding_artifact_{args.task_name}.pkl", "rb") as f:
        dyn_artifact = pickle.load(f)

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
    preload_dummy_prompt_embedding()

    hook = DynamicFieldHook(LAYER, k_range=tuple(args.k_range))
    hook.register(model)
    capture = MultiStepFeatCapture(LAYER, steps={0, cfg.num_denoising_steps_action - 1})
    capture.register(model)

    t1_max_diff = t1_noop_check(cfg, model, dataset_stats, hook, capture, args.task_name, args.seed)
    log_message(f"=== T1 check: max|delta| with alpha=0 hook vs no hook = {t1_max_diff:.2e} ===")
    assert t1_max_diff < 1e-4, f"T1 FAIL: alpha=0 hook is not a no-op (max_diff={t1_max_diff})"

    estimator = CausalStateEstimator(dyn_artifact)
    field = DynamicFlowField(dyn_artifact, randomize=False, seed=0)

    zero_embed = build_zero_embedding()
    mean_embed = build_mean_task_embedding()

    results = {
        "method_note": (
            "MUST-2 (C_oracle): delta = h_Blk13(o_t,e_real) - h_Blk13(o_t,e_dummy) at step=0 "
            "(same denoising step the injection targets), same injection point/alpha/k_range as "
            "C2_dummy_dynamic_field. MUST-4 (L1-L3): language-conditioning ablation hierarchy "
            "(L1 zero embedding, L2 mean-of-342-task-descriptions embedding, L3 a real "
            f"other-task prompt '{CROSS_TASK_PROMPT}'), no steering, otherwise identical to C1. "
            f"M-3 (C2_adaptive_alpha): alpha_t = {ADAPTIVE_ALPHA_COEF} * ||raw_dir_t|| replacing "
            "the constant alpha=40 used by C2, restoring field-magnitude information discarded "
            "by unit-normalization."
        ),
        "task": args.task_name, "alpha": args.alpha, "k_range": args.k_range,
        "adaptive_alpha_coef": ADAPTIVE_ALPHA_COEF, "cross_task_prompt": CROSS_TASK_PROMPT,
        "t1_noop_max_diff": t1_max_diff, "conditions": {},
    }

    for cond in args.conditions:
        log_message(f"=== Condition: {cond} ===")
        if cond == "C_oracle":
            real_task_desc = None  # resolved per-episode from env.get_ep_meta() inside the loop below
            # run_condition_oracle needs the real task description string; PnPCounterToCab's
            # per-episode "lang" field is looked up the same way C0 does it (task_desc=None ->
            # env.get_ep_meta().get("lang", task_name)), so resolve it once via a throwaway env.
            probe_env, _ = create_robocasa_env(cfg, seed=args.seed, episode_idx=0)
            probe_env.reset()
            real_task_desc = probe_env.get_ep_meta().get("lang", args.task_name)
            probe_env.close()
            r = run_condition_oracle(cfg, model, dataset_stats, hook, capture, estimator,
                                      args.task_name, real_task_desc, args.n_episodes, args.seed, args.alpha)
        elif cond == "L1_zero_embedding":
            r = run_condition(cfg, model, dataset_stats, hook, capture, estimator, field,
                               args.task_name, cond, zero_embed, False, 0.0, args.n_episodes, args.seed)
        elif cond == "L2_mean_task_embedding":
            r = run_condition(cfg, model, dataset_stats, hook, capture, estimator, field,
                               args.task_name, cond, mean_embed, False, 0.0, args.n_episodes, args.seed)
        elif cond == "L3_cross_task_prompt":
            r = run_condition(cfg, model, dataset_stats, hook, capture, estimator, field,
                               args.task_name, cond, CROSS_TASK_PROMPT, False, 0.0, args.n_episodes, args.seed)
        elif cond == "C2_adaptive_alpha":
            r = run_condition(cfg, model, dataset_stats, hook, capture, estimator, field,
                               args.task_name, cond, DUMMY_PROMPT, True, None, args.n_episodes, args.seed,
                               adaptive_alpha_coef=ADAPTIVE_ALPHA_COEF)
        else:
            raise ValueError(f"unknown condition {cond}")
        results["conditions"][cond] = r
        with open(out_dir / f"must_fix_experiments_{args.task_name}.json", "w") as f:
            json.dump(results, f, indent=2)   # incremental save after each condition

    capture.remove()
    hook.remove()
    with open(out_dir / f"must_fix_experiments_{args.task_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / f'must_fix_experiments_{args.task_name}.json'}")


if __name__ == "__main__":
    main()
