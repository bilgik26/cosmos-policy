"""
distribution_transport_stage3b.py — design_v3.md Stage 3-B 準拠 (分布輸送 steering, DiMaS)

DiMaS (github.com/pegah-kh/dimas) の中心的主張——「固定方向の加算 (ActAdd) ではなく、活性
**分布**をソース (ダミープロンプト条件) からターゲット (実プロンプト条件) へ輸送すべき」——を
Cosmos-Policy 上で検証する。Gaussian 近似のもとでは Monge 写像が閉形式で得られる:

    T(x) = μ_t + A(x − μ_s),   A = Σ_s^{-1/2}(Σ_s^{1/2} Σ_t Σ_s^{1/2})^{1/2} Σ_s^{-1/2}

手順:
  1. Stage 0 (steerability_audit.py) の L (言語) 対比ペア収集器 (collect_online_pairs) を
     再利用し、(Blk-13, k=1) での実プロンプト/ダミープロンプト活性ペアを新規収集する
     (ロールアウト不要、単一観測 + 2回の get_action、n_pairs=48 は Stage 0 の L グリッドと
     同一設定)。k=1 を選ぶ理由: Stage 0 の L グリッドは Blk-13 で k∈{3,4} は分離性が高い
     一方 k∈{0,1,2} はやや劣る (acc 0.84〜0.90) ものの依然として高精度であり、かつ本レポート
     第6.2節が示唆した「最終step (k=4) は分離性が高いが因果的レバレッジに乏しい」という知見
     を踏まえ、早期step (k=1) で後続blockへの伝播余地を残す設定を選ぶ。
  2. 実+ダミー活性を結合してPCA(top-3)部分空間を推定し (Stage 0 と同じ次元数)、その部分
     空間内で ソース=ダミー分布・ターゲット=実分布のGaussian統計量 (μ,Σ) を推定、Monge写像
     Aを計算する。
  3. **ablation**: mean-shift-only (A=I, 従来のActAdd相当) vs full-transport (A=Monge行列)。
  4. TransportHook: ダミープロンプト条件のロールアウト中、Blk-13・k=1でのみ、action-token
     pooled特徴をPCA部分空間へ射影→写像適用→再構成した差分を全空間位置に一様加算する
     (3-Aと同じ「pooled特徴に対する操作は一様加算がその部分空間内で最小ノルム」という論法の
     部分空間版: 部分空間の直交補空間は不変に保たれるため、全体としても部分空間内操作として
     意味のある最小限の変更になる)。
  5. 閉ループロールアウトで D_off (介入なし) / D_mean_shift / D_full_transport の3条件を
     ダミープロンプト下で比較する (design_v3.md §3 Stage3-Bの ablation 指示に従う)。

DiMaSの主張が正しければ、full_transportがmean_shiftを明確に上回るはずである
(design_v3.md §3: 「DiMaS の主張が正しければここで明確な差が出る」)。
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
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import ACTION_T_IDX
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.v4_stats_lib import episode_bootstrap_ci
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamic_vector_field_steering import (
    DUMMY_PROMPT,
    preload_dummy_prompt_embedding,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.intervention_v3.steerability_audit import (
    collect_online_pairs,
)

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "cosmos-policy":
    REPO_ROOT = REPO_ROOT.parent

DEFAULT_OUT_DIR = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/intervention_v3_distribution_transport"
)


# ────────────────────────────── 1. Monge map (Gaussian OT closed form) ──────────────────────────────

def sqrtm_psd(M, eps=1e-8):
    """対称半正定値行列の主平方根 (固有分解、3x3程度なので数値的に安定)。"""
    w, V = np.linalg.eigh((M + M.T) / 2.0)
    w = np.clip(w, eps, None)
    return V @ np.diag(np.sqrt(w)) @ V.T


def gaussian_monge_map(mu_s, cov_s, mu_t, cov_t, eps=1e-6):
    """T(x) = mu_t + A(x-mu_s), A = Sigma_s^{-1/2}(Sigma_s^{1/2} Sigma_t Sigma_s^{1/2})^{1/2} Sigma_s^{-1/2}"""
    d = len(mu_s)
    cov_s_reg = cov_s + eps * np.eye(d)
    cov_t_reg = cov_t + eps * np.eye(d)
    Ss_half = sqrtm_psd(cov_s_reg)
    Ss_half_inv = np.linalg.inv(Ss_half)
    inner = sqrtm_psd(Ss_half @ cov_t_reg @ Ss_half)
    A = Ss_half_inv @ inner @ Ss_half_inv
    return A


def fit_transport_maps(feats_real, feats_dummy):
    """feats_real/feats_dummy: (N, D) raw 2048次元活性。PCA(3)部分空間を推定し、
    mean-shift-only (A=I) と full-transport (A=Monge行列) の2種類の写像パラメータを返す。
    写像は部分空間内 (y=V^T(x-mean_all)) で定義し、全体空間への戻し方 (V, mean_all) も返す。"""
    from sklearn.utils.extmath import randomized_svd

    X_all = np.concatenate([feats_real, feats_dummy], axis=0)
    mean_all = X_all.mean(axis=0)
    Xc = X_all - mean_all
    _, _, Vt = randomized_svd(Xc, n_components=3, random_state=0)
    V = Vt.T  # (D, 3)

    y_real = (feats_real - mean_all) @ V
    y_dummy = (feats_dummy - mean_all) @ V
    mu_s, mu_t = y_dummy.mean(axis=0), y_real.mean(axis=0)
    cov_s = np.cov(y_dummy, rowvar=False)
    cov_t = np.cov(y_real, rowvar=False)

    A_full = gaussian_monge_map(mu_s, cov_s, mu_t, cov_t)
    A_mean = np.eye(3)

    return {
        "V": V, "mean_all": mean_all, "mu_s": mu_s, "mu_t": mu_t,
        "cov_s": cov_s, "cov_t": cov_t, "A_full": A_full, "A_mean": A_mean,
        "n_real": len(feats_real), "n_dummy": len(feats_dummy),
    }


# ────────────────────────────── 2. transport intervention hook ──────────────────────────────

class TransportHook:
    """target_layer の action-token出力を、target_k のdenoising stepでのみ、PCA(3)部分空間内で
    Monge写像 (mode='full_transport') または平均シフトのみ (mode='mean_shift') を適用する。
    mode='off'なら常にno-op。"""

    def __init__(self, layer, target_k, V, mean_all, mu_s, mu_t, A_full, A_mean):
        self.layer = layer
        self.target_k = target_k
        self.V = V
        self.mean_all = mean_all
        self.mu_s = mu_s
        self.mu_t = mu_t
        self.A_full = A_full
        self.A_mean = A_mean
        self.mode = "off"  # "off" | "mean_shift" | "full_transport"
        self.step = -1
        self.handle = None
        self.last_log = None

    def register(self, model):
        self.handle = model.net.blocks[self.layer].register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()

    def reset_step(self):
        self.step = -1
        self.last_log = None

    def before_step(self):
        self.step += 1

    def _transport(self, y):
        A = self.A_mean if self.mode == "mean_shift" else self.A_full
        return self.mu_t + A @ (y - self.mu_s)

    def _hook(self, module, inp, output):
        if not (isinstance(output, torch.Tensor) and output.dim() == 5):
            return output
        if self.mode == "off" or self.step != self.target_k:
            return output
        feat = output[0, ACTION_T_IDX].float().mean(dim=(0, 1)).detach().cpu().numpy().astype(np.float64)
        y = self.V.T @ (feat - self.mean_all)
        y_new = self._transport(y)
        delta_sub = y_new - y  # (3,)
        u = self.V @ delta_sub  # (D,) — subspace-only correction, orthogonal complement untouched
        feat_after = feat + u
        self.last_log = {
            "y_before": y.tolist(), "y_after": y_new.tolist(),
            "u_norm": float(np.linalg.norm(u)), "x_norm": float(np.linalg.norm(feat)),
            "intervened": True,
        }
        u_t = torch.tensor(u, dtype=output.dtype, device=output.device)
        output = output.clone()
        output[0, ACTION_T_IDX] = output[0, ACTION_T_IDX] + u_t
        return output


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


# ────────────────────────────── 3. closed-loop eval ──────────────────────────────

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
        intervene_logs = []
        for t in range(max_steps):
            if len(action_queue) == 0 and call_idx < max_call:
                observation = prepare_observation(obs, cfg.flip_images)
                call_seed = seed + call_idx * 131
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
            if env._check_success():
                success = True
                break
        env.close()
        u_over_x = [l["u_norm"] / (l["x_norm"] + 1e-8) for l in intervene_logs if l["intervened"]]
        ep_logs.append({
            "episode": ep, "success": bool(success), "n_calls": call_idx, "n_steps": t + 1,
            "n_calls_intervened": int(sum(l["intervened"] for l in intervene_logs)),
            "mean_u_over_x": float(np.mean(u_over_x)) if u_over_x else float("nan"),
        })
        log_message(f"  [{condition_name}] ep {ep + 1}/{n_episodes} "
                    f"{'SUCCESS' if success else 'FAIL'} calls={call_idx} "
                    f"n_intervened={ep_logs[-1]['n_calls_intervened']}")
    return ep_logs


# ────────────────────────────── main ──────────────────────────────

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
    p.add_argument("--target_layer", type=int, default=13)
    p.add_argument("--target_k", type=int, default=1)
    p.add_argument("--n_pairs_transport", type=int, default=48)
    p.add_argument("--n_episodes_eval", type=int, default=8)
    p.add_argument("--max_call_eval", type=int, default=40)
    p.add_argument("--conditions", nargs="+", default=["off", "mean_shift", "full_transport"])
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

    log_message("=== Stage 3-B step 1: collecting L-pair (real vs dummy) activations for Monge map fit ===")
    fp, fm = collect_online_pairs(
        cfg, model, dataset_stats, "L", args.n_pairs_transport,
        base_seed=args.seed, probe_layers=[args.target_layer], task_for_env=args.task_name,
    )
    feats_real = np.array(fp[args.target_k][args.target_layer])
    feats_dummy = np.array(fm[args.target_k][args.target_layer])
    log_message(f"[transport] collected real={len(feats_real)} dummy={len(feats_dummy)} "
                f"at layer={args.target_layer} k={args.target_k}")

    log_message("=== Stage 3-B step 2: fitting PCA(3) Monge map (Gaussian OT closed form) ===")
    tmaps = fit_transport_maps(feats_real, feats_dummy)
    log_message(f"[transport] mu_s(dummy)={tmaps['mu_s']} mu_t(real)={tmaps['mu_t']}")
    log_message(f"[transport] A_full=\n{tmaps['A_full']}")
    mean_shift_vec_norm = float(np.linalg.norm(tmaps["V"] @ (tmaps["mu_t"] - tmaps["mu_s"])))
    full_transport_delta_norm_at_mean = float(np.linalg.norm(
        tmaps["V"] @ (tmaps["mu_t"] - tmaps["A_full"] @ tmaps["mu_s"] + tmaps["A_full"] @ tmaps["mu_s"] - tmaps["mu_s"])
    ))
    log_message(f"[transport] ||mean-shift-only correction|| (at mu_s) = {mean_shift_vec_norm:.4f}")

    hook = TransportHook(args.target_layer, args.target_k, tmaps["V"], tmaps["mean_all"],
                          tmaps["mu_s"], tmaps["mu_t"], tmaps["A_full"], tmaps["A_mean"])
    hook.register(model)

    log_message("=== Stage 3-B step 3: T1 no-op check (mode=off) ===")
    t1_max_diff = t1_noop_check(cfg, model, dataset_stats, hook, args.task_name, args.seed)
    log_message(f"T1 max|delta| (mode=off vs no hook) = {t1_max_diff:.2e}")
    assert t1_max_diff < 1e-4, f"T1 FAIL: mode=off hook is not a no-op (max_diff={t1_max_diff})"

    log_message("=== Stage 3-B step 4: closed-loop eval ===")
    mode_map = {"off": "off", "mean_shift": "mean_shift", "full_transport": "full_transport"}
    all_results = {}
    for cond in args.conditions:
        cname = f"D_{cond}"
        log_message(f"=== condition {cname} ===")
        res = run_condition(cfg, model, dataset_stats, hook, args.task_name, cname, mode_map[cond],
                             args.n_episodes_eval, args.seed, args.max_call_eval)
        all_results[cname] = res

    summary = {
        "target_layer": args.target_layer, "target_k": args.target_k,
        "n_real": tmaps["n_real"], "n_dummy": tmaps["n_dummy"],
        "mu_s": tmaps["mu_s"].tolist(), "mu_t": tmaps["mu_t"].tolist(),
        "cov_s": tmaps["cov_s"].tolist(), "cov_t": tmaps["cov_t"].tolist(),
        "A_full": tmaps["A_full"].tolist(),
        "t1_noop_max_diff": t1_max_diff,
        "conditions": {},
    }
    for cname, res in all_results.items():
        successes = np.array([r["success"] for r in res], dtype=float)
        episode_ids = np.arange(len(res))
        ci = episode_bootstrap_ci(successes, episode_ids, n_boot=2000, seed=0, agg=np.mean)
        summary["conditions"][cname] = {
            "success_rate": float(successes.mean()), "n_success": int(successes.sum()),
            "n_episodes": len(res), "bootstrap_ci95": [ci["ci_lo"], ci["ci_hi"]],
            "mean_u_over_x": float(np.nanmean([r["mean_u_over_x"] for r in res])),
        }

    if "D_off" in all_results:
        from scipy.stats import fisher_exact
        base_succ = np.array([r["success"] for r in all_results["D_off"]], dtype=int)
        for cname, res in all_results.items():
            if cname == "D_off":
                continue
            cond_succ = np.array([r["success"] for r in res], dtype=int)
            table = [[cond_succ.sum(), len(cond_succ) - cond_succ.sum()],
                     [base_succ.sum(), len(base_succ) - base_succ.sum()]]
            try:
                _, pval = fisher_exact(table, alternative="greater")
                summary["conditions"][cname]["fisher_p_greater_vs_off"] = float(pval)
            except ValueError:
                summary["conditions"][cname]["fisher_p_greater_vs_off"] = None

    if "D_mean_shift" in all_results and "D_full_transport" in all_results:
        from scipy.stats import fisher_exact
        ms = np.array([r["success"] for r in all_results["D_mean_shift"]], dtype=int)
        ft = np.array([r["success"] for r in all_results["D_full_transport"]], dtype=int)
        table = [[ft.sum(), len(ft) - ft.sum()], [ms.sum(), len(ms) - ms.sum()]]
        try:
            _, pval = fisher_exact(table, alternative="greater")
            summary["fisher_p_full_transport_greater_vs_mean_shift"] = float(pval)
        except ValueError:
            summary["fisher_p_full_transport_greater_vs_mean_shift"] = None

    with open(out_dir / f"distribution_transport_{args.task_name}.json", "w") as f:
        json.dump({"summary": summary, "episode_results": all_results}, f, indent=2)
    log_message(f"Saved to {out_dir / f'distribution_transport_{args.task_name}.json'}")
    log_message(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
