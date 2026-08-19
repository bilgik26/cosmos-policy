"""
noise_inversion_control.py — design_v3.md Stage 1 準拠 (ノイズ逆変換 positive control)

Cosmos Policy の EDM サンプラ (2nd-order multistep RES solver, cosmos_policy/_src/imaginaire/
functional/{multi_step,runge_kutta}.py, cosmos_policy/modules/cosmos_sampler.py) を
Gauss-Seidel 型不動点反復で逆変換し、「教師 (実プロンプト) が生成した最終潜在 x0_final を
再現する初期ノイズ z を復元できるか」(再構成テスト)、および「その z をダミープロンプト下で
与えたときに同じ行動・タスク完遂が再現するか」(オラクル介入テスト) を検定する。

数式 (詳細は report_v3.md の方法論節、および本ファイル冒頭コメント参照):
  forward: x_0=z, x0_k=x0_fn(x_k, σ_k) (k=0..4)
    x_1 = E(x_0, x0_0; σ0,σ1)                          (reg_x0_euler_step)
    x_{k+1} = R(x_k, x0_k, x0_{k-1}; σ_k,σ_{k+1},σ_{k-1})  (res_x0_rk2_step, k=1,2,3)
    x0_4 = x0_fn(x_4, σ4)                                (sample_clean, "clean" target)
  inverse: 目標 A := x0_4 (教師の generated_latent) を固定し、5つの制約式を後ろ向き
    (C4→C3→C2→C1→C0) に Gauss-Seidel 型不動点反復で解いて z=x_0 を復元する。
    各制約式は「その場の x0 推定 (x0_fn 呼び出し)」を固定して線形代数的に閉形式で
    逆変換し、これを outer loop で M 回繰り返す (5回のx0_fn呼び出し/outer loop)。

CosmosPolicySampler / res_sampler.py の実装をそのまま模倣しているため、σスケジュール・
phi1/phi2 の実装はそちらから直接 import して再利用する (数値的な食い違いを避ける)。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cosmos_policy._src.imaginaire.functional.runge_kutta import phi1, phi2
from cosmos_policy._src.imaginaire.modules.res_sampler import get_rev_ts
from cosmos_policy.experiments.robot.cosmos_utils import (
    ACTION_DIM,
    extract_action_chunk_from_latent_sequence,
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    unnormalize_actions,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    TASK_MAX_STEPS,
    PolicyEvalConfig,
    create_robocasa_env,
    prepare_observation,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import (
    ACTION_T_IDX,
    CHUNK_SIZE,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamic_vector_field_steering import (
    DUMMY_PROMPT,
    preload_dummy_prompt_embedding,
)

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "cosmos-policy":
    REPO_ROOT = REPO_ROOT.parent

DEFAULT_OUT_DIR = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/intervention_v3_noise_inversion"
)


# ── EDM RES-solver の Gauss-Seidel 型不動点逆変換 ──────────────────────────────

def get_sigma_schedule(sigma_min: float, sigma_max: float, num_denoise_steps: int, rho: float = 7.0) -> np.ndarray:
    """CosmosPolicySampler と完全に同一の手順でσスケジュールを構成する
    (num_steps-1 nfe, get_rev_tsで5点)。"""
    num_steps = num_denoise_steps - 1  # CosmosPolicySampler: sample_clean かつ num_steps>1 なら -1
    sigmas = get_rev_ts(t_min=sigma_min, t_max=sigma_max, num_steps=num_steps, ts_order=rho)
    return sigmas.numpy()  # shape (num_denoise_steps,)


def _euler_forward(x_s, x0_s, s, t):
    coef_x0 = (s - t) / s
    coef_xs = t / s
    return coef_x0 * x0_s + coef_xs * x_s


def _euler_inverse_for_xs(x_t, x0_s, s, t):
    """x_t = coef_x0*x0_s + coef_xs*x_s  を x_s について解く。"""
    coef_x0 = (s - t) / s
    coef_xs = t / s
    return (x_t - coef_x0 * x0_s) / coef_xs


def _rk2_dt_b1_b2(s: float, t: float, s1: float):
    s_log = -np.log(s)
    t_log = -np.log(t)
    m_log = -np.log(s1)
    dt = t_log - s_log
    c2 = (m_log - s_log) / dt
    p1 = float(phi1(torch.tensor(-dt, dtype=torch.float64)))
    p2 = float(phi2(torch.tensor(-dt, dtype=torch.float64)))
    b1 = p1 - p2 / c2 if abs(c2) > 1e-12 else 0.0
    b2 = p2 / c2 if abs(c2) > 1e-12 else 0.0
    return dt, b1, b2


def _rk2_forward(x_s, x0_s, x0_s1, s, t, s1):
    dt, b1, b2 = _rk2_dt_b1_b2(s, t, s1)
    return np.exp(-dt) * x_s + dt * (b1 * x0_s + b2 * x0_s1)


def _rk2_inverse_for_xs(x_t, x0_s, x0_s1, s, t, s1):
    """x_t = exp(-dt)*x_s + dt*(b1*x0_s + b2*x0_s1) を x_s について解く。"""
    dt, b1, b2 = _rk2_dt_b1_b2(s, t, s1)
    return np.exp(dt) * (x_t - dt * b1 * x0_s - dt * b2 * x0_s1)


def invert_noise(x0_fn, target_A: torch.Tensor, sigmas: np.ndarray, M: int, device, dtype,
                  damping: float = 0.5):
    """design_v3.md Stage 1 §2: 目標 A (=x0_4, 教師の "clean" latent) から初期ノイズ z を
    Gauss-Seidel 型不動点反復 (outer loop M 回、backward C4->C3->C2->C1->C0) で復元する。

    NOTE (report_v3.md バグリスト参照): damping=1.0 (無緩和 Picard/Gauss-Seidel) は
    M>=8 前後で発散することを実測した (M=4: action_chunk_mse=0.032 -> M=8:
    action_chunk_mse=57.2)。x0_fn の実効ヤコビアンが恒等写像から十分離れているため、
    素朴な残差補正 x <- x + (target - F(x)) の縮小写像条件 |1-F'| < 1 が破れていると
    考えられる。damping<1 の緩和 (x_new <- (1-d)*x_old + d*x_solved) で安定化する。

    Returns: z (torch.Tensor, same shape as target_A), diagnostics dict (residual history)。
    """
    squeeze_back = target_A.dim() == 4
    A = target_A.unsqueeze(0) if squeeze_back else target_A
    A = A.to(torch.float64)
    s0, s1, s2, s3, s4 = [float(s) for s in sigmas]

    x1 = A.clone()
    x2 = A.clone()
    x3 = A.clone()
    x4 = A.clone()
    x0_hist = {0: A.clone(), 1: A.clone(), 2: A.clone(), 3: A.clone()}
    z = A.clone()  # x_0 estimate
    d = damping

    def call_x0_fn(x):
        with torch.no_grad():
            out = x0_fn(x.to(dtype=dtype, device=device), torch.tensor([1.0], device=device, dtype=dtype))
        return out.to(torch.float64).cpu()

    residual_history = []
    for m in range(M):
        # C4: A = x0_fn(x4, s4)  -> damped Picard residual correction
        x0_4_est = call_x0_fn(x4)
        x4 = x4 + d * (A - x0_4_est)

        # C3: x4 = R(x3, x0_fn(x3,s3), x0_hist[2]; s3,s4,s2)
        x0_3_est = call_x0_fn(x3)
        x0_hist[3] = x0_3_est
        x3_solved = _rk2_inverse_for_xs(x4, x0_3_est, x0_hist[2], s3, s4, s2)
        x3 = (1 - d) * x3 + d * x3_solved

        # C2: x3 = R(x2, x0_fn(x2,s2), x0_hist[1]; s2,s3,s1)
        x0_2_est = call_x0_fn(x2)
        x0_hist[2] = x0_2_est
        x2_solved = _rk2_inverse_for_xs(x3, x0_2_est, x0_hist[1], s2, s3, s1)
        x2 = (1 - d) * x2 + d * x2_solved

        # C1: x2 = R(x1, x0_fn(x1,s1), x0_hist[0]; s1,s2,s0)
        x0_1_est = call_x0_fn(x1)
        x0_hist[1] = x0_1_est
        x1_solved = _rk2_inverse_for_xs(x2, x0_1_est, x0_hist[0], s1, s2, s0)
        x1 = (1 - d) * x1 + d * x1_solved

        # C0: x1 = E(z, x0_fn(z,s0); s0,s1)
        x0_0_est = call_x0_fn(z)
        x0_hist[0] = x0_0_est
        z_solved = _euler_inverse_for_xs(x1, x0_0_est, s0, s1)
        z = (1 - d) * z + d * z_solved

        residual = float(torch.norm(A - x0_4_est).item() / (torch.norm(A).item() + 1e-8))
        residual_history.append(residual)
        log_message(f"    inversion outer_iter {m + 1}/{M}: relative residual (C4) = {residual:.5f}")

    z = z.to(torch.float32)
    if squeeze_back:
        z = z.squeeze(0)
    return z, {"residual_history": residual_history}


def forward_sample(x0_fn, z: torch.Tensor, sigmas: np.ndarray, device, dtype):
    """検算用: z から通常の5-call forward EDM sampler を再現し x0_final を返す
    (CosmosPolicySampler._forward_impl と数学的に同一の手順を素朴に実装)。
    z が (C,T,H,W) (バッチ次元なし) の場合は内部でバッチ次元を補い、戻り値も同じ
    次元数に揃える (invert_noise の入出力shape契約と揃える)。"""
    squeeze_back = z.dim() == 4
    s = [float(v) for v in sigmas]
    x = (z.unsqueeze(0) if squeeze_back else z).to(torch.float64)

    def call_x0_fn(x):
        with torch.no_grad():
            out = x0_fn(x.to(dtype=dtype, device=device), torch.tensor([1.0], device=device, dtype=dtype))
        return out.to(torch.float64).cpu()

    x0_0 = call_x0_fn(x)
    x1 = _euler_forward(x, x0_0, s[0], s[1])
    x0_1 = call_x0_fn(x1)
    x2 = _rk2_forward(x1, x0_1, x0_0, s[1], s[2], s[0])
    x0_2 = call_x0_fn(x2)
    x3 = _rk2_forward(x2, x0_2, x0_1, s[2], s[3], s[1])
    x0_3 = call_x0_fn(x3)
    x4 = _rk2_forward(x3, x0_3, x0_2, s[3], s[4], s[2])
    x0_4 = call_x0_fn(x4)
    x0_4 = x0_4.to(torch.float32)
    if squeeze_back:
        x0_4 = x0_4.squeeze(0)
    return x0_4


# ── 教師データ生成・action抽出ヘルパ ──────────────────────────────────────────

def action_chunk_mse(latent_a: torch.Tensor, latent_b: torch.Tensor, action_latent_idx: int, dataset_stats):
    idx = torch.full((1,), action_latent_idx, dtype=torch.int64)
    act_a = extract_action_chunk_from_latent_sequence(
        latent_a.unsqueeze(0) if latent_a.dim() == 4 else latent_a,
        action_shape=(CHUNK_SIZE, ACTION_DIM), action_indices=idx,
    ).float().numpy()
    act_b = extract_action_chunk_from_latent_sequence(
        latent_b.unsqueeze(0) if latent_b.dim() == 4 else latent_b,
        action_shape=(CHUNK_SIZE, ACTION_DIM), action_indices=idx,
    ).float().numpy()
    act_a = unnormalize_actions(act_a, dataset_stats)
    act_b = unnormalize_actions(act_b, dataset_stats)
    return float(np.mean((act_a - act_b) ** 2)), act_a, act_b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--task_name", default="PnPCounterToCab")
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--n_episodes_recon", type=int, default=6,
                    help="number of distinct (scene, teacher-call) samples for the reconstruction sweep")
    p.add_argument("--m_sweep", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--damping", type=float, default=0.5,
                    help="relaxation factor for the Gauss-Seidel/Picard fixed point (1.0=undamped)")
    p.add_argument("--n_episodes_oracle", type=int, default=8)
    p.add_argument("--m_oracle", type=int, default=16)
    p.add_argument("--max_call_oracle", type=int, default=40)
    p.add_argument("--skip_reconstruction", action="store_true")
    p.add_argument("--skip_oracle", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PolicyEvalConfig(
        config=args.config, ckpt_path=args.ckpt_path, config_file=args.config_file,
        dataset_stats_path=args.dataset_stats_path, t5_text_embeddings_path=args.t5_text_embeddings_path,
        task_name=args.task_name, seed=args.seed,
    )
    model, _ = get_model(cfg)
    model.eval()
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    preload_dummy_prompt_embedding(device="cuda:0")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    sigmas = get_sigma_schedule(model.sde.sigma_min, model.sde.sigma_max, cfg.num_denoising_steps_action)
    log_message(f"sigma schedule = {sigmas}")

    results = {"sigma_schedule": sigmas.tolist(), "damping": args.damping, "reconstruction": [], "oracle": []}

    # ── 教師 forward pass 1回から x0_fn と generated_latent (=A) を得るヘルパ ──
    def teacher_forward(observation, task_desc, seed):
        r = get_action(
            cfg=cfg, model=model, dataset_stats=dataset_stats, obs=observation,
            task_label_or_embedding=task_desc, seed=seed, randomize_seed=False,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=False,
        )
        data_batch = r["data_batch"]
        x0_fn = model.get_x0_fn_from_batch(data_batch, guidance=1.5, is_negative_prompt=False)
        action_latent_idx = int(data_batch["action_latent_idx"][0].item())
        return r, x0_fn, data_batch, action_latent_idx

    # ── Reconstruction test: 教師 (実プロンプト) の A から z を復元し、
    #    forward_sample(z) が A を再現するか (action chunk MSE) を M でスイープする ──
    if not args.skip_reconstruction:
        log_message("=== Stage 1: reconstruction test ===")
        for ep in range(args.n_episodes_recon):
            seed = args.seed + ep
            cfg.task_name = args.task_name
            env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=ep)
            obs = env.reset()
            for _ in range(10):
                obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))
            observation = prepare_observation(obs, cfg.flip_images)
            real_task_desc = env.get_ep_meta().get("lang", args.task_name)
            env.close()

            r, x0_fn, data_batch, action_latent_idx = teacher_forward(observation, real_task_desc, seed)
            A = r["generated_latent"][0].detach().cpu()  # (C',T',H',W')

            for M in args.m_sweep:
                z_hat, diag = invert_noise(x0_fn, A, sigmas, M, device, dtype, damping=args.damping)
                x0_recon = forward_sample(x0_fn, z_hat, sigmas, device, dtype)
                mse_latent = float(torch.mean((x0_recon - A) ** 2).item())
                mse_action, act_a, act_b = action_chunk_mse(x0_recon, A, action_latent_idx, dataset_stats)
                log_message(f"[recon ep={ep} M={M}] latent_mse={mse_latent:.6f} action_chunk_mse={mse_action:.6f}")
                results["reconstruction"].append({
                    "episode": ep, "seed": seed, "M": M,
                    "latent_mse": mse_latent, "action_chunk_mse": mse_action,
                    "final_residual": diag["residual_history"][-1],
                })

    # ── Oracle intervention test: ダミープロンプト下で z_hat (実プロンプト教師の
    #    最終step latentから復元) を初期ノイズとして与え、行動一致度と実環境成功率を見る ──
    if not args.skip_oracle:
        log_message("=== Stage 1: oracle intervention test ===")
        succ_c0, succ_c1, succ_oracle = [], [], []
        for ep in range(args.n_episodes_oracle):
            seed = args.seed + ep
            cfg.task_name = args.task_name
            env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=ep)
            obs = env.reset()
            for _ in range(10):
                obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))
            observation = prepare_observation(obs, cfg.flip_images)
            real_task_desc = env.get_ep_meta().get("lang", args.task_name)
            env.close()

            r, x0_fn_real, data_batch_real, action_latent_idx = teacher_forward(observation, real_task_desc, seed)
            A = r["generated_latent"][0].detach().cpu()
            z_hat, _ = invert_noise(x0_fn_real, A, sigmas, args.m_oracle, device, dtype, damping=args.damping)

            # dummy-prompt x0_fn built from the SAME data_batch, only the T5 embedding swapped
            data_batch_dummy = dict(data_batch_real)
            from cosmos_policy.experiments.robot.cosmos_utils import get_t5_embedding_from_cache
            data_batch_dummy["t5_text_embeddings"] = get_t5_embedding_from_cache(DUMMY_PROMPT).repeat(1, 1, 1).to(
                dtype=dtype
            )
            x0_fn_dummy = model.get_x0_fn_from_batch(data_batch_dummy, guidance=1.5, is_negative_prompt=False)
            x0_oracle = forward_sample(x0_fn_dummy, z_hat, sigmas, device, dtype)
            mse_action_oracle, act_real, act_oracle = action_chunk_mse(
                x0_oracle, A, action_latent_idx, dataset_stats
            )
            log_message(f"[oracle ep={ep}] action_chunk_mse(oracle_vs_teacher)={mse_action_oracle:.6f}")

            # rollout C0 (real prompt), C1 (dummy, ordinary sampling), C_oracle (dummy, z_hat as x_sigma_max)
            for cond_name, task_desc_cond, x_sigma_max_override in [
                ("C0_real", real_task_desc, None),
                ("C1_dummy", DUMMY_PROMPT, None),
                ("C_oracle_dummy_zhat", DUMMY_PROMPT, z_hat),
            ]:
                succ = run_single_rollout(
                    cfg, model, dataset_stats, args.task_name, seed, ep, task_desc_cond,
                    args.max_call_oracle, x_sigma_max_override,
                )
                {"C0_real": succ_c0, "C1_dummy": succ_c1, "C_oracle_dummy_zhat": succ_oracle}[cond_name].append(succ)
                log_message(f"  [oracle ep={ep}] {cond_name}: success={succ}")

            results["oracle"].append({
                "episode": ep, "seed": seed, "action_chunk_mse_oracle_vs_teacher": mse_action_oracle,
            })

        results["oracle_success_rates"] = {
            "C0_real": float(np.mean(succ_c0)) if succ_c0 else None,
            "C1_dummy": float(np.mean(succ_c1)) if succ_c1 else None,
            "C_oracle_dummy_zhat": float(np.mean(succ_oracle)) if succ_oracle else None,
            "n_episodes": args.n_episodes_oracle,
        }

    with open(out_dir / f"noise_inversion_{args.task_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved to {out_dir / f'noise_inversion_{args.task_name}.json'}")


def run_single_rollout(cfg, model, dataset_stats, task_name, seed, ep_idx, task_desc, max_call, x_sigma_max_override):
    """通常の closed-loop rollout。1 call目だけ x_sigma_max_override を強制し (oracle条件)、
    以降は通常の get_action (seedベースの乱数ノイズ) を使う — これは design_v3.md Stage 1
    ステップ4 の「初期ノイズとして z_hat を与える」を、closed-loop方策の1回目の呼び出しに
    限定して適用する運用上の選択である (詳細は report_v3.md の限界節を参照)。"""
    max_steps = TASK_MAX_STEPS.get(task_name, 500)
    cfg.task_name = task_name
    env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=ep_idx)
    obs = env.reset()
    from collections import deque
    action_queue = deque()
    success = False
    call_idx = 0
    t = 0
    for t in range(max_steps):
        if len(action_queue) == 0 and call_idx < max_call:
            observation = prepare_observation(obs, cfg.flip_images)
            if call_idx == 0 and x_sigma_max_override is not None:
                actions = get_action_with_fixed_noise(
                    cfg, model, dataset_stats, observation, task_desc, x_sigma_max_override
                )
            else:
                r = get_action(
                    cfg=cfg, model=model, dataset_stats=dataset_stats, obs=observation,
                    task_label_or_embedding=task_desc, seed=seed + call_idx, randomize_seed=False,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=False,
                )
                actions = r["actions"]
            call_idx += 1
            for i in range(min(cfg.num_open_loop_steps, len(actions))):
                a = actions[i]
                if a.shape[-1] == 7 and env.action_dim == 12:
                    a = np.concatenate([a, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
                action_queue.append(a)
        if not action_queue:
            break
        action = action_queue.popleft()
        obs, _, _, _ = env.step(action)
        if env._check_success():
            success = True
            break
    env.close()
    return success


def get_action_with_fixed_noise(cfg, model, dataset_stats, obs, task_desc, x_sigma_max):
    """get_action と同じ前処理 (data_batch構築) を1回だけ行い、以後は model.sampler
    (実運用と同一の CosmosPolicySampler インスタンス) を x_sigma_max=z_hat で直接呼ぶ。

    NOTE: generate_samples_from_batch は呼び出し時に data_batch を
    _normalize_video_databatch_inplace で in-place 正規化する。get_action が既に1回
    normalize 済みの data_batch に対して generate_samples_from_batch を再度呼ぶと
    二重正規化になるため、ここでは get_x0_fn_from_batch (正規化を行わない) 経由で
    x0_fn を取り、model.sampler を直接呼び出す (generate_samples_from_batch を
    再度呼ばない) ことで二重正規化を避ける。"""
    import cosmos_policy.experiments.robot.cosmos_utils as cu

    r = get_action(
        cfg=cfg, model=model, dataset_stats=dataset_stats, obs=obs,
        task_label_or_embedding=task_desc, seed=1, randomize_seed=False,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=False,
    )
    data_batch = r["data_batch"]  # already normalized in-place by the get_action call above
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        x0_fn = model.get_x0_fn_from_batch(data_batch, guidance=1.5, is_negative_prompt=False)
        generated_latent = model.sampler(
            x0_fn,
            x_sigma_max.to(device=device, dtype=dtype).unsqueeze(0),
            num_steps=cfg.num_denoising_steps_action,
            sigma_max=model.sde.sigma_max,
            sigma_min=model.sde.sigma_min,
            solver_option="2ab",
        )
        action_indices = torch.full((1,), r["latent_indices"]["action_latent_idx"], dtype=torch.int64, device=device)
        actions = cu.extract_action_chunk_from_latent_sequence(
            generated_latent, action_shape=(cfg.chunk_size, cu.ACTION_DIM), action_indices=action_indices
        ).to(torch.float32).cpu().numpy()
        if cfg.unnormalize_actions:
            actions = cu.unnormalize_actions(actions, dataset_stats)
    actions = actions[0]
    return [actions[i] for i in range(len(actions))]


if __name__ == "__main__":
    main()
