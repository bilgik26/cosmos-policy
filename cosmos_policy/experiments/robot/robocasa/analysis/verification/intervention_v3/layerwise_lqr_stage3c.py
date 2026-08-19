"""
layerwise_lqr_stage3c.py — design_v3.md Stage 3-C 準拠 (層方向 LQR, WA-LQR)

WA-LQR (arXiv:2607.14943) の「フローへの介入を正しい力学系 (層方向) の上で実装する」という
主張を、Cosmos-Policy の7つのprobe層 (Blk 0,4,9,13,18,22,27) を離散時間の状態遷移系とみなす
ことで検証する。design_v3.md §3 Stage3-Cの一般形 (低次元部分空間 d_z への streaming randomized
SVD 射影 + chunk-index減衰付きL-step LQR) を、**d_z=1のスカラー簡略版**として実装する
(3-Aが観測器ベース最小ノルム制御でグリッパー開閉という1次元特徴から始めたのと同じ理由——
まず低次元で再現を取る、design_v3.md §3 Stage3-A冒頭の指示に倣う)。

**数学的定式化**:
  状態 s_j = ζ_j(x) = W_raw_j・x + b_raw_j (j=0..6, probe_layers[j] のロジスティック回帰観測器、
  fit_probe() を7層それぞれで再利用)。
  隣接層間の遷移をオフラインG-pairデータ (同一forward pass内で7層すべてが同時に記録されている
  ため、行が完全に整列している——本ファイル冒頭の検証済み) から最小二乗であてはめる:
    s_{j+1} = a_j・s_j + c_j   (j=0..5, 6遷移)
  制御 u_j を層jの出力に (3-Aと同じ最小ノルム変換 u_j・W_raw_j/‖W_raw_j‖² で) 注入すると
  s_{j+1} = a_j・(s_j+u_j) + c_j となる。u_jのs_6への影響ゲインは
    gain_j = Π_{m=j}^{5} a_m
  であり、終端コスト q・(s_6−β*)² のみ (中間コストなし) の場合、
    minimize Σ_j r_j(τ)・u_j²  s.t.  Σ_j gain_j・u_j = Δ  (Δ = β* − s_6^nat)
  の閉形式解は重み付き最小ノルム解 u_j* = Δ・(gain_j/r_j) / Σ_m(gain_m²/r_m) であり、
  これは終端コストのみのスカラーLQRを後退Riccati再帰で解いた場合と数学的に同値である
  (単一等式制約下の最小ノルム問題として自明——本質的に3-Aのsetpoint公式のマルチステージ
  拡張になっている)。design_v3.md §3が要求する「chunk indexによる減衰スケジュール」は
  r_j(τ) = min(R_final, R_init・exp(τ/τ_R)) (τ = ロールアウト内のget_action呼び出し回数)
  として全jで共有し、早いchunkほどr_jが小さく (=制御コストが安い=強くsteer)、進むにつれ
  r_jが増大し介入が消えるようにする。

  s_0 (block0での観測) は**各callで実測**し (訓練データの平均ではなく)、そこから
  (a_j,c_j)チェーンで自然伝播したs_6^natを予測、Δを都度計算する——これは閉ループの
  観測器フィードバックであり、Buurmeijerの精神 (観測器とペアの介入) をマルチステージ化した
  ものになっている。

design_v3.md §3が指摘する通り、「固定方向加算/観測器+最小ノルム(3-A)/分布輸送(3-B)/
層方向LQR(3-C)」の比較自体が主要な比較対象であり、本実装は3-Aと同じBlk13単独最小ノルム
公式の直接拡張であるため、3-Aとの直接比較が可能である。
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
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
    PROBE_LAYERS,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.v4_stats_lib import episode_bootstrap_ci
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamic_vector_field_steering import (
    DUMMY_PROMPT,
    preload_dummy_prompt_embedding,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.intervention_v3.steerability_audit import (
    load_offline_pairs,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.intervention_v3.observer_minimal_norm_stage3a import (
    fit_probe,
)

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "cosmos-policy":
    REPO_ROOT = REPO_ROOT.parent

DEFAULT_COLLECT_DIR = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/attractor_verification/collect"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/intervention_v3_layerwise_lqr"
)


# ────────────────────────────── 1. per-layer observers + AR(1) transition fit ──────────────────────────────

def fit_layerwise_observers_and_transitions(collect_dir, target_k, seed=0):
    """7つのprobe層すべてでG-pair観測器を学習し (fit_probe再利用)、隣接層間のスカラーAR(1)
    遷移モデル (a_j, c_j) を最小二乗であてはめる。行の整列は collect_multitask.py が同一forward
    passで全層を同時記録しているため保証されている (idx配列が全層で完全一致、本ファイル docstring
    冒頭の事前確認済み)。"""
    offline = load_offline_pairs(collect_dir, "G", probe_layers=PROBE_LAYERS)
    probes = {}
    zetas = {}
    for l in PROBE_LAYERS:
        d = offline[target_k][l]
        y01 = (d["y"] == 1).astype(int)
        res = fit_probe(d["X"], y01, d["groups"], seed=seed)
        probes[l] = res
        zetas[l] = res["zeta"]
        log_message(f"  [3C probe] layer={l} k={target_k} cv_acc={res['cv_acc']:.3f} n={res['n_samples']}")

    n = len(zetas[PROBE_LAYERS[0]])
    for l in PROBE_LAYERS:
        assert len(zetas[l]) == n, "row count mismatch across probe layers — alignment assumption violated"

    transitions = {}
    for j in range(len(PROBE_LAYERS) - 1):
        l_from, l_to = PROBE_LAYERS[j], PROBE_LAYERS[j + 1]
        a_j, c_j = np.polyfit(zetas[l_from], zetas[l_to], deg=1)
        resid = zetas[l_to] - (a_j * zetas[l_from] + c_j)
        r2 = 1.0 - np.var(resid) / (np.var(zetas[l_to]) + 1e-12)
        transitions[j] = {"a": float(a_j), "c": float(c_j), "r2": float(r2),
                           "from_layer": l_from, "to_layer": l_to}
        log_message(f"  [3C transition] Blk{l_from}->Blk{l_to}: a={a_j:.4f} c={c_j:.4f} R2={r2:.4f}")
    return probes, transitions


# ────────────────────────────── 2. weighted-min-norm multi-stage solve (== terminal-cost LQR) ──────────────────────────────

def decay_schedule(tau, r_init, r_final, tau_r):
    return float(min(r_final, r_init * np.exp(tau / tau_r)))


def solve_lqr_plan(s0, transitions, beta_target, tau, r_init, r_final, tau_r):
    """s0 (実測) から6遷移を自然伝播してs6^natを予測し、終端コストのみのスカラーLQR
    (重み付き最小ノルム、docstring冒頭の導出参照) でu_0..u_5を解く。"""
    n_stage = len(transitions)  # 6
    a = np.array([transitions[j]["a"] for j in range(n_stage)])
    c = np.array([transitions[j]["c"] for j in range(n_stage)])

    s_nat = s0
    for j in range(n_stage):
        s_nat = a[j] * s_nat + c[j]
    delta = beta_target - s_nat

    gains = np.ones(n_stage)
    for j in range(n_stage - 1, -1, -1):
        gains[j] = a[j] * (gains[j + 1] if j + 1 < n_stage else 1.0)
    # gains[j] = prod_{m=j}^{n_stage-1} a[m]
    r = np.array([decay_schedule(tau, r_init, r_final, tau_r) for _ in range(n_stage)])
    denom = float(np.sum(gains ** 2 / r)) + 1e-12
    u = delta * (gains / r) / denom
    return u.tolist(), float(s_nat), float(delta)


# ────────────────────────────── 3. multi-layer intervention hook ──────────────────────────────

class LayerLQRHook:
    """PROBE_LAYERSの各blockに登録。target_kでのみ発火。position 0 (Blk0) で実測s0から
    LQRプランを都度solve、position 1..5にprecomputed u_jを注入、position 6 (最終probe層)は
    測定のみ (終端状態のログ用)。mode='off'なら常にno-op。"""

    def __init__(self, probes, transitions, zeta_target_open, zeta_target_closed,
                 target_k, r_init, r_final, tau_r):
        self.probes = probes
        self.transitions = transitions
        self.zeta_target_open = zeta_target_open
        self.zeta_target_closed = zeta_target_closed
        self.target_k = target_k
        self.r_init = r_init
        self.r_final = r_final
        self.tau_r = tau_r
        self.mode = "off"  # "off" | "force_open" | "force_closed"
        self.step = -1
        self.call_idx = 0
        self.handles = []
        self.plan = None
        self.last_log = None

    def register(self, model):
        for l in PROBE_LAYERS:
            h = model.net.blocks[l].register_forward_hook(self._make_hook(l))
            self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset_step(self):
        self.step = -1
        self.plan = None
        self.last_log = None

    def before_step(self):
        self.step += 1

    def _make_hook(self, layer):
        pos = PROBE_LAYERS.index(layer)

        def hook(module, inp, output):
            if not (isinstance(output, torch.Tensor) and output.dim() == 5):
                return output
            if self.mode == "off" or self.step != self.target_k:
                return output
            probe = self.probes[layer]
            feat = output[0, ACTION_T_IDX].float().mean(dim=(0, 1)).detach().cpu().numpy().astype(np.float64)
            zeta = float(probe["W_raw"] @ feat + probe["b_raw"])

            if pos == 0:
                beta_target = self.zeta_target_open if self.mode == "force_open" else self.zeta_target_closed
                u_list, s_nat, delta = solve_lqr_plan(
                    zeta, self.transitions, beta_target, self.call_idx,
                    self.r_init, self.r_final, self.tau_r,
                )
                self.plan = u_list
                self.last_log = {"s0": zeta, "s_nat": s_nat, "delta": delta, "target": beta_target,
                                  "u_plan": u_list, "call_idx": self.call_idx, "intervened": True}

            if pos < 6 and self.plan is not None:
                u_j = self.plan[pos]
                W_norm_sq = float(probe["W_raw"] @ probe["W_raw"])
                corr = u_j * probe["W_raw"] / W_norm_sq
                feat_after = feat + corr
                u_t = torch.tensor(corr, dtype=output.dtype, device=output.device)
                output = output.clone()
                output[0, ACTION_T_IDX] = output[0, ACTION_T_IDX] + u_t
                zeta_after = float(probe["W_raw"] @ feat_after + probe["b_raw"])
                if pos == 0:
                    self.last_log["zeta0_after"] = zeta_after
            elif pos == 6:
                # terminal probe layer: measurement only, no control
                if self.last_log is not None:
                    self.last_log["s6_actual"] = zeta
            return output

        return hook


def get_action_with_hook(cfg, model, dataset_stats, obs, task_desc, hook, seed):
    hook.reset_step()
    orig = model.get_x0_fn_from_batch

    def patched(data_batch, guidance, **kwargs):
        result = orig(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            fn, extra = result

            def w(x, s):
                hook.before_step()
                return fn(x, s)

            return w, extra
        else:
            def w(x, s):
                hook.before_step()
                return result(x, s)

            return w

    model.get_x0_fn_from_batch = patched
    try:
        r = get_action(
            cfg=cfg, model=model, dataset_stats=dataset_stats, obs=obs,
            task_label_or_embedding=task_desc, seed=seed, randomize_seed=False,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=False,
        )
    finally:
        model.get_x0_fn_from_batch = orig
    return r


def t1_noop_check(cfg, model, dataset_stats, hook, task_name, base_seed):
    env, _ = create_robocasa_env(cfg, seed=base_seed + 9999, episode_idx=0)
    obs = env.reset()
    task_desc = env.get_ep_meta().get("lang", task_name)
    observation = prepare_observation(obs, cfg.flip_images)
    env.close()

    hook.mode = "off"
    res_with = get_action_with_hook(cfg, model, dataset_stats, observation, task_desc, hook, base_seed + 12345)
    res_without = get_action(
        cfg=cfg, model=model, dataset_stats=dataset_stats, obs=observation,
        task_label_or_embedding=task_desc, seed=base_seed + 12345, randomize_seed=False,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=False,
    )
    return float(np.max(np.abs(np.stack(res_with["actions"]) - np.stack(res_without["actions"]))))


# ────────────────────────────── 4. closed-loop eval ──────────────────────────────

def run_condition(cfg, model, dataset_stats, hook, task_name, condition_name, mode, n_episodes, base_seed, max_call):
    max_steps = TASK_MAX_STEPS.get(task_name, 500)
    ep_logs = []
    for ep in range(n_episodes):
        seed = base_seed + ep
        cfg.task_name = task_name
        env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=ep)
        obs = env.reset()
        hook.mode = mode
        action_queue = deque()
        success = False
        call_idx = 0
        t = 0
        grip_traj, intervene_logs = [], []
        for t in range(max_steps):
            if len(action_queue) == 0 and call_idx < max_call:
                observation = prepare_observation(obs, cfg.flip_images)
                call_seed = seed + call_idx * 131
                hook.call_idx = call_idx
                r = get_action_with_hook(cfg, model, dataset_stats, observation, DUMMY_PROMPT, hook, call_seed)
                if hook.last_log is not None:
                    intervene_logs.append(hook.last_log)
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
            grip_traj.append(float(np.abs(obs["robot0_gripper_qpos"]).sum()))
            if env._check_success():
                success = True
                break
        env.close()
        cons_sat = [
            abs(l["s6_actual"] - l["target"]) < 0.5
            for l in intervene_logs if l.get("intervened") and "s6_actual" in l
        ]
        constraint_sat = float(np.mean(cons_sat)) if cons_sat else float("nan")
        ep_logs.append({
            "episode": ep, "success": bool(success), "n_calls": call_idx, "n_steps": t + 1,
            "mean_gripper_width": float(np.mean(grip_traj)) if grip_traj else float("nan"),
            "n_calls_intervened": int(sum(l.get("intervened", False) for l in intervene_logs)),
            "constraint_satisfaction_rate": constraint_sat,
        })
        log_message(f"  [{condition_name}] ep {ep + 1}/{n_episodes} "
                    f"{'SUCCESS' if success else 'FAIL'} calls={call_idx} "
                    f"mean_grip={ep_logs[-1]['mean_gripper_width']:.3f} "
                    f"constraint_sat={constraint_sat if not np.isnan(constraint_sat) else float('nan'):.3f}")
    return ep_logs


# ────────────────────────────── main ──────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--collect_dir", default=str(DEFAULT_COLLECT_DIR))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--task_name", default="PnPCounterToCab")
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--target_k", type=int, default=1)
    p.add_argument("--r_init", type=float, default=0.1)
    p.add_argument("--r_final", type=float, default=10.0)
    p.add_argument("--tau_r", type=float, default=10.0)
    p.add_argument("--n_episodes_eval", type=int, default=8)
    p.add_argument("--max_call_eval", type=int, default=40)
    p.add_argument("--conditions", nargs="+", default=["off", "force_open", "force_closed"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collect_dir = Path(args.collect_dir)

    log_message("=== Stage 3-C step 1: fitting per-layer observers + AR(1) transitions ===")
    probes, transitions = fit_layerwise_observers_and_transitions(collect_dir, args.target_k, seed=args.seed)
    zeta_open = probes[PROBE_LAYERS[-1]]["zeta_open_median"]
    zeta_closed = probes[PROBE_LAYERS[-1]]["zeta_closed_median"]
    log_message(f"[3C] terminal (Blk{PROBE_LAYERS[-1]}) zeta_open_median={zeta_open:.3f} "
                f"zeta_closed_median={zeta_closed:.3f}")

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

    hook = LayerLQRHook(probes, transitions, zeta_open, zeta_closed, args.target_k,
                         args.r_init, args.r_final, args.tau_r)
    hook.register(model)

    log_message("=== Stage 3-C step 2: T1 no-op check (mode=off) ===")
    t1_max_diff = t1_noop_check(cfg, model, dataset_stats, hook, args.task_name, args.seed)
    log_message(f"T1 max|delta| (mode=off vs no hook) = {t1_max_diff:.2e}")
    assert t1_max_diff < 1e-4, f"T1 FAIL: mode=off hook is not a no-op (max_diff={t1_max_diff})"

    log_message("=== Stage 3-C step 3: closed-loop eval ===")
    all_results = {}
    for cond in args.conditions:
        cname = f"F_{cond}"
        log_message(f"=== condition {cname} ===")
        res = run_condition(cfg, model, dataset_stats, hook, args.task_name, cname, cond,
                             args.n_episodes_eval, args.seed, args.max_call_eval)
        all_results[cname] = res

    summary = {
        "target_k": args.target_k, "r_init": args.r_init, "r_final": args.r_final, "tau_r": args.tau_r,
        "probe_layers": PROBE_LAYERS,
        "per_layer_cv_acc": {str(l): probes[l]["cv_acc"] for l in PROBE_LAYERS},
        "transitions": {str(j): {k: v for k, v in t.items()} for j, t in transitions.items()},
        "zeta_open_median": zeta_open, "zeta_closed_median": zeta_closed,
        "t1_noop_max_diff": t1_max_diff,
        "conditions": {},
    }
    for cname, res in all_results.items():
        successes = np.array([r["success"] for r in res], dtype=float)
        episode_ids = np.arange(len(res))
        ci = episode_bootstrap_ci(successes, episode_ids, n_boot=2000, seed=0, agg=np.mean)
        grip_widths = np.array([r["mean_gripper_width"] for r in res])
        summary["conditions"][cname] = {
            "success_rate": float(successes.mean()), "n_success": int(successes.sum()),
            "n_episodes": len(res), "bootstrap_ci95": [ci["ci_lo"], ci["ci_hi"]],
            "mean_gripper_width": float(np.nanmean(grip_widths)),
            "std_gripper_width": float(np.nanstd(grip_widths)),
            "mean_constraint_satisfaction_rate": float(np.nanmean(
                [r["constraint_satisfaction_rate"] for r in res])),
        }

    if "F_off" in all_results:
        from scipy.stats import mannwhitneyu
        base_grip = np.array([r["mean_gripper_width"] for r in all_results["F_off"]])
        for cname, res in all_results.items():
            if cname == "F_off":
                continue
            cond_grip = np.array([r["mean_gripper_width"] for r in res])
            try:
                stat, pval = mannwhitneyu(cond_grip, base_grip, alternative="two-sided")
                summary["conditions"][cname]["mannwhitney_gripper_vs_off_p"] = float(pval)
            except ValueError:
                summary["conditions"][cname]["mannwhitney_gripper_vs_off_p"] = None

    per_layer_probe_json = {
        str(l): {
            "cv_acc": v["cv_acc"], "n_samples": int(v["n_samples"]), "n_groups": int(v["n_groups"]),
            "zeta_closed_median": v["zeta_closed_median"], "zeta_open_median": v["zeta_open_median"],
            "b_raw": v["b_raw"],
        }
        for l, v in probes.items()
    }
    with open(out_dir / f"layerwise_lqr_{args.task_name}.json", "w") as f:
        json.dump({
            "per_layer_probe": per_layer_probe_json,
            "summary": summary, "episode_results": all_results,
        }, f, indent=2)
    log_message(f"Saved to {out_dir / f'layerwise_lqr_{args.task_name}.json'}")
    log_message(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
