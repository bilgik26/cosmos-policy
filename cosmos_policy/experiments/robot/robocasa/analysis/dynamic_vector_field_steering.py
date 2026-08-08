"""
dynamic_vector_field_steering.py — latent_dynamics_verification_design.md フェーズ3・4 対応。

フェーズ1(dynamics_embedding_test.py, D空間)・フェーズ2(energy_field_test.py, KDEエネルギー場・
成功エピソードのD/Xp軌跡)の成果物を使い、**推論時ロールアウト中にオンラインで** 「対象の動態的
不変量(target dynamics)」の方向へ潜在表現の速度ベクトルを書き換える動的介入を実装し、言語プロンプト
を無効化(ダミー文に置換)した状態でタスク関連の行動が自律的に誘発されるかを検証する。

──────────────────────────── オンライン因果推定の設計 ────────────────────────────

フェーズ1のD空間はbackward-only(因果的)窓で設計済みだが(design.mdからの意図的逸脱として同ファイル
docstringに開示済み)、**offline構築時の V_t は scene_residualize (エピソード全体の平均を使う非因果
的操作)に依存していた**。ロールアウト実行中は「そのエピソードの全体平均」を知り得ないため、
オンラインでは以下の**因果的な代替**を用いる(この違いは限界として§末尾で開示する):

  - scene_residualize (エピソード全体平均を引く) の代わりに、**そのエピソードのcall 0からこの
    callまでの累積平均**を引く「running mean residualization」を使う (call 0は自明に0になる)。
  - V_t, Delta_eef_t, Delta_grip_t は元々 §フェーズ1のバグ修正で backward difference
    (t vs t-1) に直してあるため、そのままオンラインで使える。
  - S_t の窓padding (エピソード先頭でtau未満の場合) も、フェーズ1の `build_delay_embedding_input`
    と全く同じ規則 (最古のc_tを繰り返す) を再現する。

**「対象の動態的不変量」の定義**: フェーズ2で成功エピソードのみからD空間上の(D_t, 実際に次に
何が起きたか)のペアを「フローライブラリ」として構築済み。ただし設計上重要な非対称性がある:
オフラインのライブラリ点は完結したエピソードなので **未来(Xp_{t+1}-Xp_t の forward difference)**
を安全に使える(教師データとして「そこから実際に次に何が起きたか」を意味する)。一方オンラインの
クエリ点(現在のライブロールアウト)はその未来を原理的に持たない。したがって:
  - ライブラリ側: forward difference (Xp_{t+1}-Xp_t、成功エピソードのみ、call終端は除外)。
  - クエリ側: 現在のD_t (backward-onlyで構築、上記の通り)。
  - 現在のD_tのk近傍(D空間)にあるライブラリ点を探し、その"次に何が起きたか"(forward Xp flow)
    を距離加重平均したものを v_target_Xp とする。
  - v_target_Xp を、同じk近傍のXp位置から作った局所接平面(局所PCA)へ射影する
    (design.md §フェーズ3実装指示2「多様体の接平面への射影」)。
  - 追加で、フェーズ2のKDEエネルギー場をXp空間で再学習し(D空間ではなく、後段の2048次元への
    逆変換が明快なXp空間で行う)、そのスコア(勾配)を弱い重み beta で加算する
    ("エネルギー場の谷を人工的に深くする"という§3の仮説の直接実装)。
  - 最終的な v_target_Xp を、progression PCA (prog_pca) と ambient_scaler の線形性を使って
    Blk-13の2048次元 raw 特徴空間の方向 raw_direction に逆変換し、alpha倍してBlk-13の
    action-token出力に加算する(steering_intervention.pyのSteeringHookと同じ注入点・書式)。

P6/P7 (steering_intervention.py由来の本プロジェクトの不変原則) を遵守する: T1 (alpha=0 no-op)
を必ず確認してから実行する。

条件設計(全てPnPCounterToCab、ダミープロンプト"The weather today is sunny."を使用):
  C0_real_prompt_no_steer   : 実プロンプト・steeringなし (現状の性能を再確認する基準)
  C1_dummy_no_steer         : ダミープロンプト・steeringなし (言語なしでは何が起きるかの基準)
  C2_dummy_dynamic_field    : ダミープロンプト・動的フィールドsteering (本検証の主対象)
  C3_dummy_static_v_steer   : ダミープロンプト・静的steering (steering_intervention.pyのv_steer、
                              §10のもの、同じalphaで比較 — 動的介入が静的介入に優る/劣るかを見る)
  C4_dummy_random_field     : ダミープロンプト・動的フィールドsteeringだが方向をランダム化
                              (方向特異性コントロール、c1に相当)
  C5_dummy_dynamic_field_frozen_obs : C2と同じ介入だが、方策への観測入力を常にcall0の1枚に
                              固定する(環境自体は実際の物理で進行させ、そこで生成された行動を
                              実行する)。フェーズ4指標3「感覚フィードバックへの従属性」の統制。
"""

import json
import os
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import torch

import cosmos_policy.experiments.robot.cosmos_utils as cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    TASK_MAX_STEPS, PolicyEvalConfig, create_robocasa_env, prepare_observation,
)
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.experiments.robot.robocasa.analysis.skill_count import LAYER, K_STEP
from cosmos_policy.experiments.robot.robocasa.analysis.dynamics_embedding_test import WINDOW_TAU
from cosmos_policy.experiments.robot.robocasa.analysis.energy_field_test import GaussianKDEField
from cosmos_policy.experiments.robot.robocasa.analysis.steering_intervention import (
    compute_steering_vectors, sanitize_actions,
)

ACTION_T_IDX = 5
DUMMY_PROMPT = "The weather today is sunny."
# precomputed by precompute_text_directions.py (§10.4 of attractor_verification_report.md) as
# "neg_filler" -- reused here verbatim so this script never has to load the ~5.6B-param T5-11b
# encoder in the same process/GPU as the already-loaded ~2B-param policy DiT (that combination
# OOMs on a single 24GB GPU, see report bug list).
PRECOMPUTED_TEXT_DIRECTIONS_PATH = (
    "cosmos_policy/experiments/robot/robocasa/analysis/results/attractor_verification/text_directions.pt"
)


def preload_dummy_prompt_embedding(path=PRECOMPUTED_TEXT_DIRECTIONS_PATH, device="cuda:0"):
    directions = torch.load(path, map_location=device)
    cosmos_utils.t5_text_embeddings_cache[DUMMY_PROMPT] = directions["neg_filler"].to(device)
    log_message(f"Preloaded dummy-prompt T5 embedding '{DUMMY_PROMPT}' from {path} (no on-the-fly T5 load).")
CHECKPOINT_CALL_IDX = 2
K_NEIGHBORS = 15
TANGENT_DIM = 5
ENERGY_PULL_BETA = 0.3


# ─────────────────────────── causal online state estimator ───────────────────────────

class CausalStateEstimator:
    """Reproduces dynamics_embedding_test.py's feature pipeline online, causally (see module
    docstring for the scene_residualize -> running-mean substitution)."""

    def __init__(self, dyn_artifact):
        self.ambient_scaler = dyn_artifact["ambient_scaler"]
        self.prog_pca = dyn_artifact["prog_pca"]
        self.c_scaler = dyn_artifact["c_scaler"]
        self.encoder_pca = dyn_artifact["encoder_pca"]
        self.tau = dyn_artifact["window_tau"]
        self.reset_episode()

    def reset_episode(self):
        self._running_sum = None
        self._running_count = 0
        self._prev_Xp = None
        self._prev_eef = None
        self._prev_grip = None
        self._c_hist = []

    def update(self, raw_feat_2048, eef_pos, gripper_width):
        """Call once per completed policy call with THIS call's own Blk-13 k=K_STEP feature.
        Returns (D_t, Xp_t) usable to steer the *next* call."""
        raw_feat_2048 = np.asarray(raw_feat_2048, dtype=np.float64)
        if self._running_sum is None:
            self._running_sum = raw_feat_2048.copy()
        else:
            self._running_sum = self._running_sum + raw_feat_2048
        self._running_count += 1
        running_mean = self._running_sum / self._running_count
        residualized = raw_feat_2048 - running_mean

        Xs = self.ambient_scaler.transform(residualized[None, :])[0]
        Xp_t = self.prog_pca.transform(Xs[None, :])[0]

        if self._prev_Xp is None:
            V_t = np.zeros_like(Xp_t)
            d_eef = np.zeros(3)
            d_grip = np.zeros(1)
        else:
            V_t = Xp_t - self._prev_Xp
            d_eef = np.asarray(eef_pos) - self._prev_eef
            d_grip = np.array([gripper_width - self._prev_grip])

        c_t_raw = np.concatenate([Xp_t, V_t, d_eef, d_grip])
        c_t = self.c_scaler.transform(c_t_raw[None, :])[0]
        self._c_hist.append(c_t)

        window = ([self._c_hist[0]] * max(self.tau - len(self._c_hist), 0)) + self._c_hist[-self.tau:]
        S_t = np.concatenate(window)
        D_t = self.encoder_pca.transform(S_t[None, :])[0]

        self._prev_Xp, self._prev_eef, self._prev_grip = Xp_t, np.asarray(eef_pos), gripper_width
        return D_t, Xp_t


# ─────────────────────────── target-dynamics flow field (kNN, offline library) ───────────────────────────

class DynamicFlowField:
    """Given the causal D_t of the live rollout, look up the k nearest offline library points
    (successful-episode, causally-built D-space positions) and return the distance-weighted
    average of what actually happened next in those demonstrations (forward Xp flow), projected
    onto the local tangent plane, plus a weak energy-ascent pull towards the Xp-space density
    peak (Gaussian KDE score, see energy_field_test.py)."""

    def __init__(self, dyn_artifact, k=K_NEIGHBORS, tangent_dim=TANGENT_DIM, beta=ENERGY_PULL_BETA,
                 randomize=False, seed=0):
        self.dyn_artifact = dyn_artifact
        D, Xp, episode, call_idx, success = (
            dyn_artifact["D"], dyn_artifact["Xp"], dyn_artifact["episode"],
            dyn_artifact["call_idx"], dyn_artifact["success"],
        )
        success_episodes = np.array([e for e in np.unique(episode) if bool(success[episode == e][0])])
        D_pts, Xp_pts, V_fwd = [], [], []
        for e in success_episodes:
            idx = np.where(episode == e)[0]
            order = idx[np.argsort(call_idx[idx])]
            if len(order) < 2:
                continue
            D_pts.append(D[order[:-1]])
            Xp_pts.append(Xp[order[:-1]])
            V_fwd.append(Xp[order[1:]] - Xp[order[:-1]])   # forward diff: safe, library is offline/complete
        self.D_pts = np.concatenate(D_pts)
        self.Xp_pts = np.concatenate(Xp_pts)
        self.V_fwd = np.concatenate(V_fwd)
        self.k = min(k, len(self.D_pts))
        self.tangent_dim = tangent_dim
        self.beta = beta
        self.energy_kde = GaussianKDEField(np.concatenate([p for p in Xp_pts]))
        self.rng = np.random.RandomState(seed)
        self.randomize = randomize

    def query(self, D_t, Xp_t):
        dists = np.linalg.norm(self.D_pts - D_t[None, :], axis=1)
        knn = np.argsort(dists)[:self.k]
        d_knn = dists[knn]
        h = np.median(d_knn) + 1e-6
        w = np.exp(-0.5 * (d_knn / h) ** 2)
        w = w / (w.sum() + 1e-12)
        v_target = (w[:, None] * self.V_fwd[knn]).sum(axis=0)

        # local tangent plane from the neighbours' Xp positions, project v_target onto it
        Xp_local = self.Xp_pts[knn]
        Xp_local_c = Xp_local - Xp_local.mean(axis=0, keepdims=True)
        n_comp = min(self.tangent_dim, len(knn) - 1, Xp_local_c.shape[1])
        if n_comp >= 1:
            _, _, Vt = np.linalg.svd(Xp_local_c, full_matrices=False)
            tangent_basis = Vt[:n_comp]  # (n_comp, 10)
            v_target = tangent_basis.T @ (tangent_basis @ v_target)

        if self.beta > 0:
            score = self.energy_kde.score(Xp_t[None, :])[0]
            score_unit = score / (np.linalg.norm(score) + 1e-12)
            v_target = v_target + self.beta * np.linalg.norm(v_target) * score_unit

        if self.randomize:
            rand_dir = self.rng.normal(size=v_target.shape)
            v_target = rand_dir / (np.linalg.norm(rand_dir) + 1e-12) * np.linalg.norm(v_target)

        return v_target   # direction in Xp-space (10-dim)


def xp_direction_to_raw(v_xp, dyn_artifact):
    """Linear inverse of scene_residualize->ambient_scaler->prog_pca chain for a DELTA vector
    (mean-subtraction cancels for a delta; only scaling/rotation matter)."""
    components = dyn_artifact["prog_pca"].components_          # (10, 2048)
    scale = dyn_artifact["ambient_scaler"].scale_               # (2048,)
    raw_std_dir = v_xp @ components                             # (2048,)
    raw_dir = raw_std_dir * scale
    return raw_dir


# ─────────────────────────── steering hook (dynamic vec, updated per call) ───────────────────────────

class DynamicFieldHook:
    def __init__(self, layer, k_range=(0, 1)):
        self.layer = layer
        self.k_range = k_range
        self.step = -1
        self.alpha = 0.0
        self.vec = None   # raw 2048-dim direction (NOT normalized), updated once per call
        self.handle = None
        self.last_delta_max = 0.0

    def register(self, model):
        block = model.net.blocks[self.layer]
        self.handle = block.register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()

    def reset_step(self):
        # NOTE (bug found & fixed during this verification, copied from and also present in
        # steering_intervention.py's SteeringHook -- see report bug list): last_delta_max used to
        # be reset to 0.0 unconditionally on every hook firing that DIDN'T match k_range, inside
        # _hook() itself. Since a get_action() call fires this hook once per denoising step in
        # increasing step order, and k_range is typically an EARLY sub-range (e.g. (0,1) of
        # 0..4), the LAST step of every call (step=4, outside k_range) always fired last and
        # unconditionally overwrote last_delta_max back to 0.0 -- silently erasing the record of
        # a real, correctly-applied injection that happened earlier in the same call at step 0/1.
        # The actual steering injection (the output[...] += delta line) was NOT affected by this
        # bug and fired correctly; only the diagnostic max-delta bookkeeping was wrong. Fixed by
        # resetting once per call (here, in reset_step()) and using max() inside the hook instead
        # of unconditional overwrite.
        self.step = -1
        self.last_delta_max = 0.0

    def before_step(self):
        self.step += 1

    def _hook(self, module, inp, output):
        if not (isinstance(output, torch.Tensor) and output.dim() == 5):
            return output
        if self.vec is None or self.alpha == 0.0 or not (self.k_range[0] <= self.step <= self.k_range[1]):
            return output
        vec_t = torch.tensor(self.vec, dtype=output.dtype, device=output.device)
        delta = self.alpha * vec_t
        self.last_delta_max = max(self.last_delta_max, float(delta.abs().max().item()))
        output = output.clone()
        output[0, ACTION_T_IDX] = output[0, ACTION_T_IDX] + delta
        return output


class FinalFeatCapture:
    """Captures only the LAST denoising step's action-token feature at the steered layer, for
    the causal state estimator's next-call update."""

    def __init__(self, layer, n_steps):
        self.layer = layer
        self.n_steps = n_steps
        self._step = -1
        self._final = None
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
        self._final = None

    def _hook(self, module, inp, out):
        if self._step < 0 or not (isinstance(out, torch.Tensor) and out.dim() == 5):
            return
        if self._step == self.n_steps - 1:
            self._final = out[0, ACTION_T_IDX].float().mean(dim=(0, 1)).detach().cpu().numpy()


def get_action_with_dynamic_hook(cfg, model, dataset_stats, obs, task_desc, hook, capture, seed, n_steps):
    hook.reset_step()
    capture.reset()
    orig = model.get_x0_fn_from_batch

    def patched(data_batch, guidance, **kwargs):
        result = orig(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            fn, extra = result
            def w(x, s):
                hook.before_step()
                capture.before_denoise_step()
                return fn(x, s)
            return w, extra
        else:
            def w(x, s):
                hook.before_step()
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


# ─────────────────────────── rollout loop ───────────────────────────

def run_condition(cfg, model, dataset_stats, hook, capture, estimator, field, task_name,
                   condition_name, task_desc, use_steering, alpha, n_episodes, base_seed,
                   max_call=40, frozen_obs=False):
    """frozen_obs=True implements phase-4's "sensory feedback dependency" control (design.md
    フェーズ4指標3): the policy is always shown the FIRST call's observation, regardless of the
    real (still-physically-evolving) scene, while the environment still executes whatever
    actions that produces. If the resulting action trajectory collapses to something flat/
    repetitive relative to the frozen_obs=False condition (under the same steering config), that
    is evidence the dynamic-field-steered behaviour stays coupled to live sensory input rather
    than degenerating into a fixed, Reactive-independent canned motion."""
    max_steps = TASK_MAX_STEPS.get(task_name, 500)
    ep_logs = []
    for ep in range(n_episodes):
        env, _ = create_robocasa_env(cfg, seed=base_seed + ep, episode_idx=ep)
        obs = env.reset()
        for _ in range(10):
            dummy = np.zeros(env.action_spec[0].shape)
            obs, _, _, _ = env.step(dummy)
        this_task_desc = task_desc if task_desc is not None else env.get_ep_meta().get("lang", task_name)

        estimator.reset_episode()
        hook.vec = None
        hook.alpha = 0.0

        action_queue = deque()
        success = False
        call_idx = 0
        t_step = 0
        eef_traj, grip_traj, action_log = [], [], []
        xp_traj, d_traj, xp_call_idx = [], [], []   # for phase-4 manifold-deviation metric
        max_delta_this_ep = 0.0
        nan_detected = False
        frozen_observation = None

        while call_idx < max_call and t_step < max_steps and not success:
            if len(action_queue) == 0:
                if frozen_obs and frozen_observation is not None:
                    observation = frozen_observation
                else:
                    observation = prepare_observation(obs, cfg.flip_images)
                    if frozen_obs:
                        frozen_observation = observation
                call_seed = base_seed + ep * 131 + call_idx
                result = get_action_with_dynamic_hook(
                    cfg, model, dataset_stats, observation, this_task_desc, hook, capture,
                    call_seed, cfg.num_denoising_steps_action,
                )
                max_delta_this_ep = max(max_delta_this_ep, hook.last_delta_max)
                if os.environ.get("DVFS_DEBUG"):
                    log_message(f"    [debug] call_idx={call_idx} capture._step={capture._step} "
                                f"final_is_none={capture._final is None} hook.vec_is_none={hook.vec is None} "
                                f"hook.alpha={hook.alpha} hook.last_delta_max={hook.last_delta_max}")

                # ── update causal state & (if steering) compute NEXT call's field vector ──
                gripper_width = float(np.abs(obs["robot0_gripper_qpos"]).sum())
                eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                if capture._final is not None:
                    D_t, Xp_t = estimator.update(capture._final, eef_pos, gripper_width)
                    xp_traj.append(Xp_t.tolist())
                    d_traj.append(D_t.tolist())
                    xp_call_idx.append(call_idx)   # call_idx was not yet incremented at this point
                    if use_steering:
                        v_xp = field.query(D_t, Xp_t)
                        raw_dir = xp_direction_to_raw(v_xp, field.dyn_artifact)
                        raw_dir_unit = raw_dir / (np.linalg.norm(raw_dir) + 1e-12)
                        hook.vec = raw_dir_unit
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
        })
        log_message(f"  [{condition_name} ep{ep}] success={success} n_steps={t_step} "
                    f"max_delta={max_delta_this_ep:.4f} nan={nan_detected}")

    success_rate = float(np.mean([r["success"] for r in ep_logs]))
    log_message(f"[{condition_name}] success_rate={success_rate:.2f} (n={n_episodes})")
    return {"success_rate": success_rate, "episodes": ep_logs}


def t1_noop_check(cfg, model, dataset_stats, hook, capture, task_name, base_seed):
    env, _ = create_robocasa_env(cfg, seed=base_seed + 9999, episode_idx=0)
    obs = env.reset()
    for _ in range(10):
        dummy = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(dummy)
    task_desc = env.get_ep_meta().get("lang", task_name)
    observation = prepare_observation(obs, cfg.flip_images)
    env.close()

    hook.vec = np.ones(2048) / np.sqrt(2048)
    hook.alpha = 0.0
    res_with_hook = get_action_with_dynamic_hook(cfg, model, dataset_stats, observation, task_desc,
                                                  hook, capture, base_seed + 12345, cfg.num_denoising_steps_action)
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
    p.add_argument("--collect_dir", required=True,
                    help="Directory with the ORIGINAL collect/ (phase-labeled) data, used only "
                         "for the C3 static-v_steer baseline via compute_steering_vectors(). "
                         "NOTE: this must be collect/, not collect_v2/ -- collect_v2/ (used for "
                         "--embedding_dir's D-space artifacts, which need episode_success) was "
                         "never re-run through phase_labeling.py so it has no *_phases.npz files.")
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task_name", default="PnPCounterToCab")
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--alpha", type=float, default=6.0)
    p.add_argument("--k_range", type=int, nargs=2, default=[0, 1])
    p.add_argument("--conditions", nargs="+",
                    default=["C0_real_prompt_no_steer", "C1_dummy_no_steer", "C2_dummy_dynamic_field",
                             "C3_dummy_static_v_steer", "C4_dummy_random_field",
                             "C5_dummy_dynamic_field_frozen_obs"])
    args = p.parse_args()

    collect_dir = Path(args.collect_dir)
    embedding_dir = Path(args.embedding_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(embedding_dir / f"dynamics_embedding_artifact_{args.task_name}.pkl", "rb") as f:
        dyn_artifact = pickle.load(f)

    manifest = json.loads((collect_dir / "multitask_manifest.json").read_text())

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
    capture = FinalFeatCapture(LAYER, cfg.num_denoising_steps_action)
    capture.register(model)

    t1_max_diff = t1_noop_check(cfg, model, dataset_stats, hook, capture, args.task_name, args.seed)
    log_message(f"=== T1 check: max|delta| with alpha=0 hook vs no hook = {t1_max_diff:.2e} ===")
    assert t1_max_diff < 1e-4, f"T1 FAIL: alpha=0 hook is not a no-op (max_diff={t1_max_diff})"

    estimator = CausalStateEstimator(dyn_artifact)
    field = DynamicFlowField(dyn_artifact, randomize=False, seed=0)
    random_field = DynamicFlowField(dyn_artifact, randomize=True, seed=1)

    static_vecs = compute_steering_vectors(collect_dir, manifest, args.task_name, LAYER, K_STEP, seed=0)
    static_vec_unit = static_vecs["v_steer"] / np.linalg.norm(static_vecs["v_steer"])

    results = {"method_note": (
        "Phase3: online causal D-space position estimate (running-mean scene-residualization "
        "substitute) -> kNN(k=%d) lookup in an offline successful-episode flow library (forward "
        "Xp-flow, tangent-projected, local PCA dim=%d) + weak KDE energy-ascent pull (beta=%.2f) "
        "in Xp-space -> linear inverse (prog_pca/ambient_scaler) to a raw Blk-13 2048-dim "
        "direction -> alpha*dir added to the action-token hidden state at k=%s (layer=%d)."
    ) % (K_NEIGHBORS, TANGENT_DIM, ENERGY_PULL_BETA, str(tuple(args.k_range)), LAYER),
        "task": args.task_name, "alpha": args.alpha, "k_range": args.k_range,
        "t1_noop_max_diff": t1_max_diff, "conditions": {}}

    for cond in args.conditions:
        log_message(f"=== Condition: {cond} ===")
        if cond == "C0_real_prompt_no_steer":
            r = run_condition(cfg, model, dataset_stats, hook, capture, estimator, field,
                               args.task_name, cond, None, False, 0.0, args.n_episodes, args.seed)
        elif cond == "C1_dummy_no_steer":
            r = run_condition(cfg, model, dataset_stats, hook, capture, estimator, field,
                               args.task_name, cond, DUMMY_PROMPT, False, 0.0, args.n_episodes, args.seed)
        elif cond == "C2_dummy_dynamic_field":
            r = run_condition(cfg, model, dataset_stats, hook, capture, estimator, field,
                               args.task_name, cond, DUMMY_PROMPT, True, args.alpha, args.n_episodes, args.seed)
        elif cond == "C4_dummy_random_field":
            r = run_condition(cfg, model, dataset_stats, hook, capture, estimator, random_field,
                               args.task_name, cond, DUMMY_PROMPT, True, args.alpha, args.n_episodes, args.seed)
        elif cond == "C5_dummy_dynamic_field_frozen_obs":
            r = run_condition(cfg, model, dataset_stats, hook, capture, estimator, field,
                               args.task_name, cond, DUMMY_PROMPT, True, args.alpha, args.n_episodes, args.seed,
                               frozen_obs=True)
        elif cond == "C3_dummy_static_v_steer":
            # static condition: bypass field.query, hold hook.vec fixed for the whole episode
            # (but still runs the causal state estimator, unused for control here, purely so
            # phase-4's manifold-deviation metric has an xp_traj/d_traj to compare against for
            # this condition too).
            def _static_run():
                max_steps = TASK_MAX_STEPS.get(args.task_name, 500)
                ep_logs = []
                for ep in range(args.n_episodes):
                    env, _ = create_robocasa_env(cfg, seed=args.seed + ep, episode_idx=ep)
                    obs = env.reset()
                    for _ in range(10):
                        obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))
                    estimator.reset_episode()
                    action_queue, success, call_idx, t_step = deque(), False, 0, 0
                    eef_traj, grip_traj, action_log = [], [], []
                    xp_traj, d_traj, xp_call_idx = [], [], []
                    max_delta, nan_detected = 0.0, False
                    hook.vec, hook.alpha = static_vec_unit, args.alpha
                    while call_idx < 40 and t_step < max_steps and not success:
                        if len(action_queue) == 0:
                            observation = prepare_observation(obs, cfg.flip_images)
                            call_seed = args.seed + ep * 131 + call_idx
                            result = get_action_with_dynamic_hook(cfg, model, dataset_stats, observation,
                                                                   DUMMY_PROMPT, hook, capture, call_seed,
                                                                   cfg.num_denoising_steps_action)
                            max_delta = max(max_delta, hook.last_delta_max)
                            gripper_width = float(np.abs(obs["robot0_gripper_qpos"]).sum())
                            eef_pos_now = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                            if capture._final is not None:
                                D_t, Xp_t = estimator.update(capture._final, eef_pos_now, gripper_width)
                                xp_traj.append(Xp_t.tolist())
                                d_traj.append(D_t.tolist())
                                xp_call_idx.append(call_idx)
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
                    ep_logs.append({"success": success, "n_calls": call_idx, "n_steps": t_step,
                                     "eef_traj": eef_traj, "grip_traj": grip_traj, "action_log": action_log,
                                     "xp_traj": xp_traj, "d_traj": d_traj, "xp_call_idx": xp_call_idx,
                                     "max_hook_delta": max_delta, "nan_detected": nan_detected})
                    log_message(f"  [{cond} ep{ep}] success={success} n_steps={t_step}")
                sr = float(np.mean([r["success"] for r in ep_logs]))
                log_message(f"[{cond}] success_rate={sr:.2f} (n={args.n_episodes})")
                return {"success_rate": sr, "episodes": ep_logs}
            r = _static_run()
        else:
            raise ValueError(f"unknown condition {cond}")
        results["conditions"][cond] = r

    capture.remove()
    hook.remove()

    with open(out_dir / f"dynamic_vector_field_steering_{args.task_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / f'dynamic_vector_field_steering_{args.task_name}.json'}")


if __name__ == "__main__":
    main()
