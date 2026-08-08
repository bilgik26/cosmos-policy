"""
critical_boundary_test.py — attractor_verification_report.md への追加検証 (新規: Critical Boundary Detector)。

ユーザー提案「Critical Boundary Detector」への対応:
  Cosmos Policy の逆拡散過程 (連続σスケジュール, σ_max→σ_min) において、denoiserの出力
  u_t (= x0_pred、前処理後のクリーンな予測) の入力 x_t に対する勾配のFrobeniusノルム
  ‖∂u_t/∂x_t‖_F が σ_t に対してピークを取る点を「臨界境界 t*」と定義し、その位置を検出、
  境界の前後で予測がどう変わるかを分析する (拡散モデル文献の "critical window" 現象)。

  実測で確認した重要な事実: このモデルは (denoise()内で self.scaling として使われる)
  RectifiedFlowScaling (EDMScaling ではない、cosmos_policy/_src/imaginaire/modules/
  denoiser_scaling.py で確認済み) を使用しており、係数は t=σ/(σ+1) として
  c_skip=c_in=1-t, c_out=-t で与えられる (EDM標準式とは異なる)。また本番のσスケジュールは
  σ_min=model.sde.sigma_min≈4.0 (0.002等の典型値ではない) までしか下がらないため、
  σ→0近傍でのc_skip→1飽和は本番の探索範囲では起こらない (探索範囲は常に[σ_min≈4, σ_max=80])。

方法論 (厳格レビュー対策):
  1. ‖∂u_t/∂x_t‖_F の推定: u_t は高次元 (C×T'×H'×W') でJacobianを陽に計算するのは不可能な
     ため、Hutchinson trace推定量 (Rademacher確率ベクトルv、E[‖J^T v‖^2]=‖J‖_F^2) を用いる。
     ‖J‖_F 自体の不偏推定量は sqrt(mean(‖J^T v_i‖^2)) であり、mean(‖J^T v_i‖) ではない
     (sqrtは凹関数なのでJensenの不等式により後者は真値を過小評価するバイアスを持つ —
     実装初期にこのバイアスに気づき修正した。grad_frob_curve()参照)。
     全DiTブロックにselective activation checkpointing (mode="mm_only") がかかっており
     同一計算グラフに対する複数回のbackwardが禁止されているため (実測でRuntimeError確認済み)、
     retain_graphでの使い回しはできず、N_HUTCHINSON_PROBES本のプローブそれぞれについて
     forward+backwardを独立に取り直す (_hutchinson_sq_norms参照、コスト増とのトレードオフ)。
  2. **前処理による機械的な交絡への対処**: x0_pred = c_skip(σ)·x_t + c_out(σ)·F_θ(c_in(σ)·x_t)
     という affine な形を取るため、c_skip(σ)の変化そのものが ‖∂x0_pred/∂x_t‖_F を
     "ネットワークの意味的判断" とは無関係に機械的に押し上げる可能性がある。これを「本物の
     critical window信号」と誤認しないよう、各σでの解析的な c_skip(σ), c_out(σ), c_in(σ)
     (model.scalingから直接取得、再実装しない) をログする。参照として c_skip(σ)·√D_full
     (D_full=潜在テンソルの全次元数) も重ねるが、条件付けフレーム (proprio/画像) は
     condition_video_input_mask によりx_tと無関係に上書きされるため実効自由度は D_full より
     小さく、この参照曲線は緩い (甘めの) 上限に過ぎないことに注意 (絶対値の直接比較ではなく、
     実測曲線とc_skip(σ)自身の「σ_max→σ_min間の相対増加率」を比較することで、機械的な寄与を
     超える増加があるかを判定する — 詳細はレポート本文)。
  3. **本番ステップ数(5)の解像度不足への対処**: 本番の5ステップだけではt*位置を安定推定できない
     可能性が高いため、解析専用に細かいステップ数 (FINE_STEPS) でも実行してσ_tに対する曲線を
     取得し、本番の5ステップでも同じ傾向 (どの区間に山があるか) が定性的に再現されるかを別途
     確認する。
  4. **独立な交差検証 (多シード収束分析)**: 同一の観測・条件付けに対し異なる初期ノイズ
     (x_sigma_max のseedのみ変更) でN_CONVERGENCE_SEEDS回、勾配計算なしでサンプリングし、
     各ステップでのアクションチャンク予測 (u_t の action token 部分) のシード間分散を追跡する。
     「臨界窓」仮説が正しければ、σが大きい早期ステップではシード間で予測が大きく異なり
     (未決定、高分散)、t*付近で急激に収束する (決定、低分散化) はず。この収束点のσと
     勾配ノルムピークのσが一致するかどうかで、勾配ベースの検出を独立に検証する
     (単一指標への依存を避ける、という本プロジェクトの標準方針)。

実装上の注意 (循環回避・忠実性):
  - CosmosPolicySampler.forward()/_forward_impl() は @torch.no_grad() でデコレートされており、
    さらに本番のcollect系スクリプトは get_action() の torch.inference_mode() 内で実行される
    ため、実際のサンプリングループ内では勾配が取得できない (かつ inference tensor は
    inference_mode の外でも autograd に使えない)。そこで本スクリプトは:
      (a) get_action() を経由せず、build_data_batch() (steering_vector_variants.py で
          既にactivation-maximization勾配計算のために確立済みのパターン) で
          inference_mode の外に data_batch を構築する。
      (b) 生産で実際に使われているsolver関数 (differential_equation_solver, get_rev_ts,
          "2ab" multistep, SolverConfig) をそのまま (再実装せず) importして呼び出すことで、
          実際の生成軌跡 (σスケジュール・多段階更新式) を忠実に再現しつつ勾配計算を可能にする。
      (c) 各ステップの出力は勾配計算後に .detach() してから次ステップに渡すため、実軌跡には
          一切影響しない (副計算)。
  - モデル呼び出しパラメータ (guidance=1.5, solver_option="2ab", rho=7, sigma_min/max=
    model.sde.sigma_min/max, use_variance_scale=False相当) は cosmos_utils.get_action() /
    policy_text2world_model.generate_samples_from_batch() の実際のデフォルト値に合わせている
    (該当箇所を直接grepして確認済み)。

出力:
  results/attractor_verification/critical_boundary/critical_boundary_test.json
  results/attractor_verification/critical_boundary/*.png
"""

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cosmos_policy._src.imaginaire.modules.res_sampler import SolverConfig, differential_equation_solver, get_rev_ts
from cosmos_policy._src.imaginaire.utils import misc
from cosmos_policy.experiments.robot.cosmos_utils import (
    ACTION_DIM, extract_action_chunk_from_latent_sequence, get_action, get_model,
    init_t5_text_embeddings_cache, load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import SPATIAL_H, SPATIAL_W, STATE_T
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.collect_multitask import DEFAULT_TASKS, git_commit_hash
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.steering_vector_variants import build_data_batch
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    TASK_MAX_STEPS, PolicyEvalConfig, create_robocasa_env, prepare_observation,
)
from cosmos_policy.utils.utils import set_seed_everywhere

GUIDANCE = 1.5
SOLVER_OPTION = "2ab"
RHO = 7.0


# ── 逆拡散ループ (production解決関数を再利用し、勾配計算を側計算として追加) ────────────

def _hutchinson_sq_norms(float64_x0_fn, x, s, n_probes: int):
    """各プローブごとに forward を取り直してから backward を1回だけ行う。
    全DiTブロックにselective activation checkpointing (mode="mm_only") がかかっており、
    同一計算グラフに対してbackwardを複数回呼ぶと
    "Trying to backward an extra time" で失敗するため (retain_graphの使い回し不可、実測確認済み)、
    グラフの使い回しはせずプローブごとに独立した forward+backward を行う
    (n_hutchinson_probes倍のforwardコストとのトレードオフ)。"""
    sq_norms = []
    u_out = None
    for i in range(n_probes):
        x_req = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            u_t = float64_x0_fn(x_req, s.detach())
        if u_out is None:
            u_out = u_t.detach()
        v = (torch.randint(0, 2, u_t.shape, device=u_t.device) * 2 - 1).to(u_t.dtype)
        vjp = torch.autograd.grad(u_t, x_req, grad_outputs=v, retain_graph=False)[0]
        sq_norms.append(float((vjp.double() ** 2).sum().item()))
    return sq_norms, u_out


def run_reverse_diffusion(x0_fn, x_sigma_max, num_steps, sigma_min, sigma_max, action_latent_idx,
                           chunk_size, compute_grad, n_hutchinson_probes, records_out):
    """CosmosPolicySampler._forward_impl (num_steps>1経路) の忠実な再現。
    各ステップで x0_fn 呼び出し結果を records_out に記録し、compute_grad=True なら
    Hutchinson推定によるヤコビアンFrobeniusノルム(の二乗)も記録する。実軌跡は不変。"""
    in_dtype = x_sigma_max.dtype

    def float64_x0_fn(x, t):
        return x0_fn(x.to(in_dtype), t.to(in_dtype)).to(torch.float64)

    ns = num_steps - 1 if num_steps > 1 else num_steps  # sample_clean=True (production既定)
    sigmas_L = get_rev_ts(sigma_min, sigma_max, ns, RHO).to(x_sigma_max.device)
    solver_cfg = SolverConfig(is_multi=True, multistep=SOLVER_OPTION)

    def record_step(sigma_val, u_out):
        rec = {"sigma": sigma_val}
        if action_latent_idx is not None:
            idx_t = torch.full((1,), action_latent_idx, dtype=torch.int64, device=u_out.device)
            act = extract_action_chunk_from_latent_sequence(
                u_out.float(), action_shape=(chunk_size, ACTION_DIM), action_indices=idx_t
            )
            rec["action_chunk"] = act[0].detach().cpu().numpy()
        records_out.append(rec)

    def wrapped_x0_fn(x, s):
        sigma_val = float(s.flatten()[0].item())
        if compute_grad:
            sq_norms, u_out = _hutchinson_sq_norms(float64_x0_fn, x, s, n_hutchinson_probes)
            record_step(sigma_val, u_out)
            records_out[-1]["grad_sq_norms"] = sq_norms
        else:
            with torch.no_grad():
                u_out = float64_x0_fn(x.detach(), s.detach())
            record_step(sigma_val, u_out)
        return u_out

    sample_fn = differential_equation_solver(wrapped_x0_fn, sigmas_L, solver_cfg)
    denoised = sample_fn(x_sigma_max)
    # sample_clean: 最終クリーンステップ (production _forward_impl と同じ)
    ones = torch.ones(denoised.size(0), device=denoised.device, dtype=denoised.dtype)
    wrapped_x0_fn(denoised, sigmas_L[-1] * ones)
    return records_out


def run_boundary_probe(cfg, model, dataset_stats, observation, task_desc, seed, num_steps,
                        compute_grad, n_hutchinson_probes):
    data_batch, action_latent_idx = build_data_batch(cfg, model, dataset_stats, observation, task_desc)
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
            (1,) + tuple(state_shape), torch.float32, model.tensor_kwargs["device"], seed,
        ) * model.sde.sigma_max
        x0_fn = model.get_x0_fn_from_batch(data_batch, GUIDANCE, is_negative_prompt=False)

        records: List[Dict] = []
        run_reverse_diffusion(
            x0_fn, x_sigma_max, num_steps, model.sde.sigma_min, model.sde.sigma_max,
            action_latent_idx, cfg.chunk_size, compute_grad, n_hutchinson_probes, records,
        )
    return records


def analytic_scaling_curve(model, sigmas, ambient_dim):
    """各σでの前処理係数 (c_skip, c_out, c_in; このモデルは RectifiedFlowScaling を使用
    — denoiser_scaling.py 参照) を model.scaling から直接取得 (再実装しない)。
    c_skip(σ)·√D_full を「恒等写像項のみの機械的な参照曲線 (緩い上限)」として返す。"""
    out = {"sigma": [], "c_skip": [], "c_out": [], "c_in": [], "identity_skip_frob_ref": []}
    with torch.no_grad():
        for sig in sigmas:
            try:
                sigma_t = torch.tensor([[[[[float(sig)]]]]], device=model.tensor_kwargs["device"])
                c_skip, c_out, c_in, _ = model.scaling(sigma=sigma_t)
                out["sigma"].append(float(sig))
                out["c_skip"].append(float(c_skip.flatten()[0].item()))
                out["c_out"].append(float(c_out.flatten()[0].item()))
                out["c_in"].append(float(c_in.flatten()[0].item()))
                out["identity_skip_frob_ref"].append(float(c_skip.flatten()[0].item()) * float(np.sqrt(ambient_dim)))
            except Exception as e:
                log_message(f"  [analytic_scaling_curve] skipped sigma={sig}: {e}")
    return out


# ── ロールアウト + プローブ ───────────────────────────────────────────────────

def rollout_and_probe(cfg, model, dataset_stats, task_name, seed_base, n_episodes,
                       n_probes_per_episode, probe_call_stride, fine_steps, production_steps,
                       n_hutchinson_probes, n_convergence_seeds, also_production):
    max_steps = TASK_MAX_STEPS.get(task_name, 500)
    probe_records = []
    ep_success = []

    for ep_idx in range(n_episodes):
        env, _ = create_robocasa_env(cfg, seed=seed_base + ep_idx, episode_idx=ep_idx)
        obs = env.reset()
        for _ in range(10):
            dummy = np.zeros(env.action_spec[0].shape)
            obs, _, _, _ = env.step(dummy)
        task_desc = env.get_ep_meta().get("lang", task_name)

        action_queue = deque()
        success = False
        call_idx_in_ep = 0
        n_probed_this_ep = 0
        t0 = time.time()

        for t in range(max_steps):
            observation = prepare_observation(obs, cfg.flip_images)

            if len(action_queue) == 0:
                call_seed = cfg.seed + ep_idx + t
                do_probe = (n_probed_this_ep < n_probes_per_episode and call_idx_in_ep % probe_call_stride == 0)

                if do_probe:
                    log_message(f"  [{task_name} seed{seed_base}] probing ep{ep_idx} call{call_idx_in_ep} (t={t})")
                    fine_recs = run_boundary_probe(
                        cfg, model, dataset_stats, observation, task_desc, call_seed,
                        fine_steps, compute_grad=True, n_hutchinson_probes=n_hutchinson_probes,
                    )
                    prod_recs = None
                    if also_production:
                        prod_recs = run_boundary_probe(
                            cfg, model, dataset_stats, observation, task_desc, call_seed,
                            production_steps, compute_grad=True, n_hutchinson_probes=n_hutchinson_probes,
                        )
                    conv_recs = []
                    for c_off in range(n_convergence_seeds):
                        c_seed = call_seed + 90000 + c_off
                        r = run_boundary_probe(
                            cfg, model, dataset_stats, observation, task_desc, c_seed,
                            fine_steps, compute_grad=False, n_hutchinson_probes=0,
                        )
                        conv_recs.append(r)
                    probe_records.append({
                        "task": task_name, "seed_base": seed_base, "episode_in_task_seed": ep_idx,
                        "call_idx": call_idx_in_ep, "sim_step_t": t, "call_seed": call_seed,
                        "fine_records": fine_recs, "production_records": prod_recs,
                        "convergence_records": conv_recs,
                    })
                    n_probed_this_ep += 1

                try:
                    result = get_action(
                        cfg=cfg, model=model, dataset_stats=dataset_stats, obs=observation,
                        task_label_or_embedding=task_desc, seed=call_seed, randomize_seed=False,
                        num_denoising_steps_action=cfg.num_denoising_steps_action,
                        generate_future_state_and_value_in_parallel=False,
                    )
                except Exception as e:
                    log_message(f"  [{task_name} seed{seed_base}] Error at ep{ep_idx} t={t}: {e}")
                    import traceback
                    traceback.print_exc()
                    break

                actions = result["actions"]
                call_idx_in_ep += 1
                for i in range(min(cfg.num_open_loop_steps, len(actions))):
                    a = actions[i]
                    if a.shape[-1] == 7 and env.action_dim == 12:
                        a = np.concatenate([a, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
                    action_queue.append(a)

            if action_queue:
                action = action_queue.popleft()
                obs, _, _, _ = env.step(action)
                if env._check_success():
                    success = True
                    break

        env.close()
        ep_success.append(success)
        dt = time.time() - t0
        log_message(
            f"  [{task_name} seed{seed_base}] ep {ep_idx + 1}/{n_episodes} "
            f"{'SUCCESS' if success else 'FAIL'} calls={call_idx_in_ep} probed={n_probed_this_ep} ({dt:.1f}s)"
        )

    return probe_records, ep_success


# ── 集計・プロット ────────────────────────────────────────────────────────────

def grad_frob_curve(records: List[Dict]):
    """1プローブのfine_records/production_recordsから (sigma[], frob_estimate[], frob_se[]) を計算。

    各Hutchinsonプローブ i は sq_i = ‖J^T v_i‖^2 という ‖J‖_F^2 の不偏推定量を与える
    (E[v^T J J^T v] = tr(J J^T) = ‖J‖_F^2、Rademacher v)。‖J‖_F 自体の推定量は
    sqrt(mean(sq_i)) であり、mean(sqrt(sq_i)) ではない (sqrtは凹関数なのでJensenの不等式より
    後者は真値を過小評価するバイアスを持つ — 実際に旧実装ではこのバイアスにより解析的な
    c_skip(σ)·√D 参照曲線と比べて非現実的に小さい値が出ていたため修正した)。
    frob_se は sq_i の標準誤差をデルタ法で ‖J‖_F のスケールに伝播したもの
    (se(sqrt(X)) ≈ se(X) / (2·sqrt(mean(X))))。"""
    sigmas, frob_est, frob_se = [], [], []
    for rec in records:
        sq = np.clip(np.array(rec["grad_sq_norms"], dtype=np.float64), 0, None)
        mean_sq = float(sq.mean())
        se_sq = float(sq.std(ddof=1) / np.sqrt(len(sq))) if len(sq) > 1 else 0.0
        frob = float(np.sqrt(mean_sq))
        se_frob = se_sq / (2 * frob) if frob > 1e-12 else 0.0
        sigmas.append(rec["sigma"])
        frob_est.append(frob)
        frob_se.append(se_frob)
    return np.array(sigmas), np.array(frob_est), np.array(frob_se)


def classify_curve_shape(sigmas: np.ndarray, values: np.ndarray) -> Dict:
    """曲線の形状を分類する。argmax/argminが端点(σ_maxまたはσ_min)にある場合は
    「単調増加/減少」であり、探索範囲内に真の内部極値 (interior peak/trough) は
    無いと判定する (端点をピークと誤って報告しないため)。"""
    idx_max, idx_min = int(np.argmax(values)), int(np.argmin(values))
    n = len(values)
    if idx_max in (0, n - 1) and idx_min in (0, n - 1):
        shape = "monotonic_increasing" if values[-1] > values[0] else "monotonic_decreasing"
        return {"shape": shape, "extremum_sigma": None, "extremum_value": None}
    if idx_max not in (0, n - 1):
        return {"shape": "interior_peak", "extremum_sigma": float(sigmas[idx_max]), "extremum_value": float(values[idx_max])}
    return {"shape": "interior_trough", "extremum_sigma": float(sigmas[idx_min]), "extremum_value": float(values[idx_min])}


def convergence_curve(conv_recs: List[List[Dict]]):
    """N_CONVERGENCE_SEEDS本のconvergence_recordsから、各ステップでのシード間分散
    (action_chunkのpairwise距離の平均) を計算。全seedで同一のsigmaスケジュールを仮定。"""
    n_seeds = len(conv_recs)
    n_steps = len(conv_recs[0])
    sigmas = [conv_recs[0][k]["sigma"] for k in range(n_steps)]
    interseed_var = []
    for k in range(n_steps):
        chunks = np.stack([conv_recs[s][k]["action_chunk"].reshape(-1) for s in range(n_seeds)])
        centered = chunks - chunks.mean(axis=0, keepdims=True)
        var_k = float(np.mean(np.sum(centered ** 2, axis=1)))  # trace(cov)相当
        interseed_var.append(var_k)
    return np.array(sigmas), np.array(interseed_var)


def plot_boundary_summary(all_fine_curves, all_conv_curves, analytic_curve, task, out_dir, tag=""):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    sigmas_ref = all_fine_curves[0][0]
    mean_stack = np.stack([m for _, m, _ in all_fine_curves])
    pooled_mean = mean_stack.mean(axis=0)
    pooled_std = mean_stack.std(axis=0)

    l1, = ax1.plot(sigmas_ref, pooled_mean, "-o", color="C0", markersize=4,
                    label="‖∂x0_pred/∂x_t‖_F (Hutchinson est., mean over probe locations)")
    ax1.fill_between(sigmas_ref, pooled_mean - pooled_std, pooled_mean + pooled_std, color="C0", alpha=0.2)
    lines = [l1]
    if analytic_curve is not None and len(analytic_curve["sigma"]) == len(sigmas_ref):
        l_ref, = ax1.plot(analytic_curve["sigma"], analytic_curve["identity_skip_frob_ref"], "--", color="gray",
                           label="c_skip(sigma)*sqrt(D) (mechanical identity-skip reference)")
        lines.append(l_ref)
    ax1.set_xscale("log")
    ax1.set_xlabel("sigma_t (log scale, sigma_max -> sigma_min direction)")
    ax1.set_ylabel("||d(x0_pred)/d(x_t)||_F", color="C0")
    ax1.invert_xaxis()

    if all_conv_curves:
        ax2 = ax1.twinx()
        conv_stack = np.stack([v for _, v in all_conv_curves])
        pooled_conv = conv_stack.mean(axis=0)
        sigmas_conv = all_conv_curves[0][0]
        l2, = ax2.plot(sigmas_conv, pooled_conv, "-s", color="C3", markersize=4,
                        label="inter-seed variance (independent cross-check)")
        lines.append(l2)
        ax2.set_ylabel("inter-seed variance (predicted action chunk)", color="C3")
        ax2.set_yscale("log")

    ax1.legend(handles=lines, fontsize=8, loc="best")
    ax1.set_title(f"{task} {tag}: gradient-norm curve vs. inter-seed convergence curve")
    fig.tight_layout()
    fname = f"critical_boundary_{task}{('_' + tag) if tag else ''}.png"
    fig_path = out_dir / fname
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path.name


def main():
    import draccus

    @dataclass
    class CriticalBoundaryConfig(PolicyEvalConfig):
        output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/attractor_verification/critical_boundary"
        tasks: str = ",".join(DEFAULT_TASKS)
        seed_series: str = "195"
        n_episodes_per_task: int = 2
        n_probes_per_episode: int = 2
        probe_call_stride: int = 2
        fine_steps: int = 9
        production_steps: int = 5
        n_hutchinson_probes: int = 6
        n_convergence_seeds: int = 6
        also_production: bool = True

    cfg: CriticalBoundaryConfig = draccus.parse(CriticalBoundaryConfig)

    tasks = cfg.tasks.split(",")
    seed_series = [int(s) for s in cfg.seed_series.split(",")]
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_message("=== critical_boundary_test.py: Critical Boundary Detector ===")
    log_message(f"Tasks: {tasks}  seed_series: {seed_series}")
    log_message(f"fine_steps={cfg.fine_steps} production_steps={cfg.production_steps} "
                f"n_hutchinson_probes={cfg.n_hutchinson_probes} n_convergence_seeds={cfg.n_convergence_seeds}")

    model, _ = get_model(cfg)
    model.eval()
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    # 潜在状態の全次元数 D = C' × T'(=STATE_T=11) × H'(=SPATIAL_H=14) × W'(=SPATIAL_W=14)
    # (analysis_shared.pyの定数を再利用。c_skip(σ)·√D 参照曲線の計算に使う)
    D_latent = int(model.config.state_ch) * STATE_T * SPATIAL_H * SPATIAL_W
    log_message(f"D_latent (ambient dim of x_t) = {D_latent}")

    manifest = {
        "run_id": f"critical_boundary_{int(time.time())}",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit_hash(),
        "tasks": tasks,
        "seed_series": seed_series,
        "config": {
            "n_episodes_per_task": cfg.n_episodes_per_task,
            "n_probes_per_episode": cfg.n_probes_per_episode,
            "probe_call_stride": cfg.probe_call_stride,
            "fine_steps": cfg.fine_steps,
            "production_steps": cfg.production_steps,
            "n_hutchinson_probes": cfg.n_hutchinson_probes,
            "n_convergence_seeds": cfg.n_convergence_seeds,
            "guidance": GUIDANCE,
            "solver_option": SOLVER_OPTION,
            "rho": RHO,
        },
        "tasks_results": {},
    }

    for seed_base in seed_series:
        for task_name in tasks:
            set_seed_everywhere(seed_base)
            cfg.task_name = task_name
            cfg.seed = seed_base
            t_start = time.time()
            probe_records, ep_success = rollout_and_probe(
                cfg, model, dataset_stats, task_name, seed_base, cfg.n_episodes_per_task,
                cfg.n_probes_per_episode, cfg.probe_call_stride, cfg.fine_steps,
                cfg.production_steps, cfg.n_hutchinson_probes, cfg.n_convergence_seeds,
                cfg.also_production,
            )
            elapsed = time.time() - t_start

            all_fine_curves = [grad_frob_curve(p["fine_records"]) for p in probe_records]
            all_prod_curves = [grad_frob_curve(p["production_records"]) for p in probe_records if p["production_records"]]
            all_conv_curves = [convergence_curve(p["convergence_records"]) for p in probe_records if p["convergence_records"]]

            sigmas_fine = all_fine_curves[0][0].tolist() if all_fine_curves else []
            analytic_curve = analytic_scaling_curve(model, sigmas_fine, D_latent) if sigmas_fine else None

            grad_shape = None
            growth_ratio_empirical = None
            growth_ratio_c_skip = None
            if all_fine_curves:
                pooled_grad = np.stack([m for _, m, _ in all_fine_curves]).mean(axis=0)
                grad_shape = classify_curve_shape(np.array(sigmas_fine), pooled_grad)
                if pooled_grad[0] > 1e-12:
                    growth_ratio_empirical = float(pooled_grad[-1] / pooled_grad[0])
                if analytic_curve is not None and analytic_curve["c_skip"][0] > 1e-12:
                    growth_ratio_c_skip = float(analytic_curve["c_skip"][-1] / analytic_curve["c_skip"][0])

            conv_shape = None
            if all_conv_curves:
                conv_stack = np.stack([v for _, v in all_conv_curves])
                pooled_conv = conv_stack.mean(axis=0)
                sigmas_conv = all_conv_curves[0][0]
                conv_shape = classify_curve_shape(np.array(sigmas_conv), pooled_conv)

            fig_name = None
            if all_fine_curves:
                fig_name = plot_boundary_summary(all_fine_curves, all_conv_curves, analytic_curve, task_name, out_dir, tag=f"seed{seed_base}")

            fig_prod_name = None
            if all_prod_curves:
                fig_prod_name = plot_boundary_summary(all_prod_curves, [], None, task_name, out_dir, tag=f"seed{seed_base}_production{cfg.production_steps}steps")

            manifest["tasks_results"][f"{task_name}_seed{seed_base}"] = {
                "n_episodes": cfg.n_episodes_per_task,
                "success_rate": float(np.mean(ep_success)) if ep_success else None,
                "n_probes": len(probe_records),
                "elapsed_sec": elapsed,
                "sigmas_fine": sigmas_fine,
                "fine_grad_frob_mean_per_probe": [m.tolist() for _, m, _ in all_fine_curves],
                "fine_grad_frob_std_per_probe": [s.tolist() for _, _, s in all_fine_curves],
                "production_grad_frob_mean_per_probe": [m.tolist() for _, m, _ in all_prod_curves],
                "sigmas_production": all_prod_curves[0][0].tolist() if all_prod_curves else [],
                "convergence_sigmas": all_conv_curves[0][0].tolist() if all_conv_curves else [],
                "convergence_interseed_var_per_probe": [v.tolist() for _, v in all_conv_curves],
                "analytic_scaling_curve": analytic_curve,
                "D_latent": D_latent,
                "grad_curve_shape": grad_shape,
                "convergence_curve_shape": conv_shape,
                "growth_ratio_empirical_grad_frob": growth_ratio_empirical,
                "growth_ratio_c_skip": growth_ratio_c_skip,
                "plot_png": fig_name,
                "plot_production_png": fig_prod_name,
            }
            log_message(
                f"=== Done {task_name}/seed{seed_base}: grad_shape={grad_shape} conv_shape={conv_shape} "
                f"growth_ratio(empirical/c_skip)={growth_ratio_empirical}/{growth_ratio_c_skip} ({elapsed:.1f}s) ==="
            )
            with open(out_dir / "critical_boundary_test.json", "w") as f:
                json.dump(manifest, f, indent=2)

    log_message("=== critical_boundary_test.py complete ===")
    log_message(f"Manifest: {out_dir / 'critical_boundary_test.json'}")


if __name__ == "__main__":
    main()
