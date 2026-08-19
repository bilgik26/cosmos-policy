"""
noise_actor_stage2.py — design_v3.md Stage 2 準拠 (観測条件付きノイズアクター)

report_v3.md の Stage 1 (noise_inversion_control.py) が「教師 (実プロンプト) ロールアウトの
最終潜在を再現する初期ノイズは、damping付き不動点反復で実用精度まで復元できる」ことを確定
させた前提の上に立つ。design_v3.md Stage 2 は、この復元能力を使って学習データセット
`{(o_t, ẑ_t)}` を作り、**観測のみを入力とする軽量アクター** `ψ_φ(z | o)` を教師あり学習し、
方策本体を凍結したまま、ダミープロンプト下で `z = ψ_φ(o_t)` を初期ノイズとしてロールアウトする
——というものである。

**スコープ縮小（本実装での逸脱、正直に開示する）**:
design_v3.md は z 全体 (Cosmos Policy の場合、11フレーム分の潜在ボリューム全体、
`(16,11,28,28)` ≈ 137,984次元) をアクターの出力と想定しているが、UniSteer が実際に軽量
アクターで学習したのは flow-matching 行動ヘッドの50次元ノイズ表現であり、Cosmos Policy の
ような統合video-world-modelのzとは次元regimeが2桁以上異なる。本セッションの計算資源・
データセット規模（後述、成功エピソード由来のペアはたかだか数十〜百程度）でz全体を学習する
ことは過学習が確実であり誠実でない。したがって、**アクターが予測するのはzのうちaction
トークンスロット (T=ACTION_T_IDX=5) の (16,28,28) 部分のみ**とし、残りの10スロット（proprio・
current image・future image・valueの各条件付け/生成スロット）は通常の生成と同じ乱数
(sigma_max スケールのGaussian、`misc.arch_invariant_rand` を再利用) をそのまま使う。これは
「行動生成そのものへの介入」という目的に直結するスコープ限定であり、本節冒頭にも明記する。

手順:
  1. 実プロンプトで closed-loop ロールアウトを行い、成功エピソードのみを採用する。
  2. 採用エピソード内の一部の call (間引き) について、その call を再現する seed で
     get_action を再実行し x0_fn / generated_latent を得て、Stage 1 の
     invert_noise (M=32, damping=0.5) で initial noise z_hat を復元、その
     action スロット (16,28,28) を教師信号として保存する。
  3. 観測 (3カメラ画像をダウンサンプル + proprio) -> 軽量 CNN+MLP アクター (UniSteerに倣い
     [1024,1024,1024] MLP + 小さなCNN) を教師あり学習 (MSE, 方策本体は完全に凍結)。
  4. 評価: C0 (実プロンプト, 標準サンプリング) / C1 (ダミープロンプト, 標準サンプリング) /
     C2 (ダミープロンプト, 全callでアクター予測ノイズをactionスロットに使用) の3条件で
     closed-loopロールアウトのsuccess_rateを比較する。C2は推論コストが軽い(CNNの1回の
     forward pass のみ、逆変換の不動点反復は不要)ため、report_v3.md 第4.3節のオラクル
     介入テストと異なり**call 0だけでなく全callに適用できる**——設計上の改善点。
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_policy._src.imaginaire.utils import misc
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
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.v4_stats_lib import episode_bootstrap_ci
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamic_vector_field_steering import (
    DUMMY_PROMPT,
    preload_dummy_prompt_embedding,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.intervention_v3.noise_inversion_control import (
    get_sigma_schedule,
    invert_noise,
)

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "cosmos-policy":
    REPO_ROOT = REPO_ROOT.parent

DEFAULT_OUT_DIR = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/intervention_v3_noise_actor_stage2"
)

IMG_SIZE = 64
ACTION_LATENT_SHAPE = (16, 28, 28)  # (C', H', W') of the action-token slot, see report_v3.md §4.1


# ────────────────────────────── actor network ──────────────────────────────

class NoiseActor(nn.Module):
    """観測 (3カメラ画像 + proprio) -> action スロットの初期ノイズ z_hat (16,28,28)。
    UniSteer (arXiv:2605.10821) の「小さなCNN + [1024,1024,1024] MLP」構成に倣う。"""

    def __init__(self, img_size=IMG_SIZE, proprio_dim=9, out_shape=ACTION_LATENT_SHAPE):
        super().__init__()
        self.out_shape = out_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(9, 16, 4, 2, 1), nn.ReLU(inplace=True),   # 64->32
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(inplace=True),  # 32->16
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),  # 16->8
            nn.AdaptiveAvgPool2d(4),
        )
        cnn_feat_dim = 64 * 4 * 4
        self.proprio_mlp = nn.Sequential(nn.Linear(proprio_dim, 64), nn.ReLU(inplace=True))
        out_dim = int(np.prod(out_shape))
        self.fusion = nn.Sequential(
            nn.Linear(cnn_feat_dim + 64, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim),
        )

    def forward(self, images, proprio):
        # images: (B, 9, H, W) ; proprio: (B, proprio_dim)
        feat = self.cnn(images).flatten(1)
        p = self.proprio_mlp(proprio)
        z = self.fusion(torch.cat([feat, p], dim=1))
        return z.view(-1, *self.out_shape)


def obs_to_actor_input(observation, img_size=IMG_SIZE):
    """prepare_observation() が返す dict (numpy, HWC uint8 画像 + proprio) -> actor入力tensor
    (images: (9,img_size,img_size) float32 in [-1,1], proprio: (9,) float32)。"""
    imgs = []
    for key in ("primary_image", "secondary_image", "wrist_image"):
        img = observation[key]
        t = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
        imgs.append((t / 127.5 - 1.0).squeeze(0))
    img_cat = torch.cat(imgs, dim=0)  # (9, img_size, img_size)
    proprio = torch.from_numpy(np.asarray(observation["proprio"], dtype=np.float32))
    return img_cat, proprio


# ────────────────────────────── dataset collection ──────────────────────────────

def collect_dataset(cfg, model, dataset_stats, sigmas, task_name, n_target_success, max_attempts,
                     calls_per_episode, m_invert, damping, base_seed):
    """実プロンプトで closed-loop ロールアウトを行い、成功エピソードから間引いた call について
    ノイズ逆変換 (M=m_invert) で z_hat を復元し、(observation, z_hat_action_slot) ペアを集める。"""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    max_steps = TASK_MAX_STEPS.get(task_name, 500)
    pairs = []  # list of dict(images, proprio, target, episode, call_idx)
    ep_log = []
    n_success = 0
    attempt = 0
    while n_success < n_target_success and attempt < max_attempts:
        seed = base_seed + attempt
        cfg.task_name = task_name
        env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=attempt)
        obs = env.reset()
        real_task_desc = env.get_ep_meta().get("lang", task_name)
        action_queue = deque()
        success = False
        call_idx = 0
        call_records = []  # (call_idx, observation dict, seed)
        t0 = time.time()
        for t in range(max_steps):
            if len(action_queue) == 0 and call_idx < 40:
                observation = prepare_observation(obs, cfg.flip_images)
                call_seed = seed + call_idx * 131
                r = get_action(
                    cfg=cfg, model=model, dataset_stats=dataset_stats, obs=observation,
                    task_label_or_embedding=real_task_desc, seed=call_seed, randomize_seed=False,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=False,
                )
                call_records.append((call_idx, observation, call_seed))
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
        dt = time.time() - t0
        log_message(f"[collect] attempt {attempt} seed={seed} {'SUCCESS' if success else 'fail'} "
                    f"n_calls={call_idx} ({dt:.1f}s)")
        ep_log.append({"attempt": attempt, "seed": seed, "success": bool(success), "n_calls": call_idx})

        if success and call_idx >= 1:
            sel_idx = np.unique(np.linspace(0, call_idx - 1, num=min(calls_per_episode, call_idx), dtype=int))
            for ci in sel_idx:
                _, observation, call_seed = call_records[ci]
                r2 = get_action(
                    cfg=cfg, model=model, dataset_stats=dataset_stats, obs=observation,
                    task_label_or_embedding=real_task_desc, seed=call_seed, randomize_seed=False,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=False,
                )
                data_batch = r2["data_batch"]
                x0_fn = model.get_x0_fn_from_batch(data_batch, guidance=1.5, is_negative_prompt=False)
                A = r2["generated_latent"][0].detach().cpu()
                z_hat, _ = invert_noise(x0_fn, A, sigmas, m_invert, device, dtype, damping=damping)
                z_action = z_hat[:, ACTION_T_IDX].clone()  # (16,28,28)
                img_cat, proprio = obs_to_actor_input(observation)
                pairs.append({
                    "images": img_cat, "proprio": proprio, "target": z_action,
                    "episode": n_success, "call_idx": int(ci),
                })
                log_message(f"  [collect] episode(success)#{n_success} call={ci} inverted "
                            f"(target norm={float(z_action.norm()):.2f})")
            n_success += 1
        attempt += 1
    log_message(f"[collect] done: {n_success} successful episodes / {attempt} attempts, "
                f"{len(pairs)} training pairs")
    return pairs, ep_log


# ────────────────────────────── training ──────────────────────────────

def train_actor(pairs, sigma_max, device, epochs=300, lr=1e-3, val_episodes=2, seed=0):
    episodes = sorted(set(p["episode"] for p in pairs))
    rng = np.random.RandomState(seed)
    rng.shuffle(episodes)
    val_eps = set(episodes[:min(val_episodes, max(len(episodes) - 1, 0))])
    train_pairs = [p for p in pairs if p["episode"] not in val_eps]
    val_pairs = [p for p in pairs if p["episode"] in val_eps]
    log_message(f"[train] n_train_pairs={len(train_pairs)} n_val_pairs={len(val_pairs)} "
                f"val_episodes={sorted(val_eps)}")

    def stack(ps):
        images = torch.stack([p["images"] for p in ps]).to(device)
        proprio = torch.stack([p["proprio"] for p in ps]).to(device)
        target = torch.stack([p["target"] for p in ps]).to(device) / sigma_max  # normalize to ~N(0,1)
        return images, proprio, target

    train_images, train_proprio, train_target = stack(train_pairs)
    if val_pairs:
        val_images, val_proprio, val_target = stack(val_pairs)

    actor = NoiseActor().to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    best_val = float("inf")
    best_state = None
    for ep in range(epochs):
        actor.train()
        opt.zero_grad()
        pred = actor(train_images, train_proprio)
        loss = F.mse_loss(pred, train_target)
        loss.backward()
        opt.step()
        val_loss = None
        if val_pairs:
            actor.eval()
            with torch.no_grad():
                val_pred = actor(val_images, val_proprio)
                val_loss = float(F.mse_loss(val_pred, val_target).item())
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in actor.state_dict().items()}
        if (ep + 1) % 50 == 0 or ep == 0:
            log_message(f"  [train] epoch {ep + 1}/{epochs} train_mse={float(loss.item()):.5f} "
                        f"val_mse={val_loss if val_loss is not None else float('nan'):.5f}")
        history.append({"epoch": ep, "train_mse": float(loss.item()), "val_mse": val_loss})
    if best_state is not None:
        actor.load_state_dict(best_state)
    else:
        best_val = history[-1]["train_mse"]
    return actor, history, best_val


# ────────────────────────────── evaluation rollouts ──────────────────────────────

def get_action_with_actor_noise(cfg, model, dataset_stats, obs, task_desc, actor, sigma_max, seed, device, dtype):
    r = get_action(
        cfg=cfg, model=model, dataset_stats=dataset_stats, obs=obs,
        task_label_or_embedding=task_desc, seed=seed, randomize_seed=False,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=False,
    )
    data_batch = r["data_batch"]
    state_shape = list(r["generated_latent"].shape[1:])  # (C',T',H',W')
    with torch.inference_mode():
        img_cat, proprio = obs_to_actor_input(obs)
        actor.eval()
        z_action = actor(img_cat.unsqueeze(0).to(device), proprio.unsqueeze(0).to(device))[0] * sigma_max
        z_full = misc.arch_invariant_rand(tuple(state_shape), torch.float32, device, seed) * sigma_max
        z_full[:, ACTION_T_IDX] = z_action.to(z_full.dtype)

        x0_fn = model.get_x0_fn_from_batch(data_batch, guidance=1.5, is_negative_prompt=False)
        generated_latent = model.sampler(
            x0_fn, z_full.unsqueeze(0).to(device=device, dtype=dtype),
            num_steps=cfg.num_denoising_steps_action, sigma_max=model.sde.sigma_max,
            sigma_min=model.sde.sigma_min, solver_option="2ab",
        )
        action_indices = torch.full((1,), r["latent_indices"]["action_latent_idx"], dtype=torch.int64, device=device)
        actions = extract_action_chunk_from_latent_sequence(
            generated_latent, action_shape=(cfg.chunk_size, ACTION_DIM), action_indices=action_indices
        ).to(torch.float32).cpu().numpy()
        if cfg.unnormalize_actions:
            actions = unnormalize_actions(actions, dataset_stats)
    actions = actions[0]
    return [actions[i] for i in range(len(actions))]


def run_condition(cfg, model, dataset_stats, task_name, condition_name, task_desc, use_actor, actor,
                   sigma_max, n_episodes, base_seed, max_call, device, dtype):
    max_steps = TASK_MAX_STEPS.get(task_name, 500)
    ep_results = []
    for ep in range(n_episodes):
        seed = base_seed + ep
        cfg.task_name = task_name
        env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=ep)
        obs = env.reset()
        this_task_desc = task_desc if task_desc is not None else env.get_ep_meta().get("lang", task_name)
        action_queue = deque()
        success = False
        call_idx = 0
        t = 0
        for t in range(max_steps):
            if len(action_queue) == 0 and call_idx < max_call:
                observation = prepare_observation(obs, cfg.flip_images)
                call_seed = seed + call_idx * 131
                if use_actor:
                    actions = get_action_with_actor_noise(
                        cfg, model, dataset_stats, observation, this_task_desc, actor, sigma_max,
                        call_seed, device, dtype,
                    )
                else:
                    r = get_action(
                        cfg=cfg, model=model, dataset_stats=dataset_stats, obs=observation,
                        task_label_or_embedding=this_task_desc, seed=call_seed, randomize_seed=False,
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
        ep_results.append({"episode": ep, "success": bool(success), "n_calls": call_idx, "n_steps": t + 1})
        log_message(f"  [{condition_name}] ep {ep + 1}/{n_episodes} "
                    f"{'SUCCESS' if success else 'FAIL'} calls={call_idx} steps={t + 1}")
    return ep_results


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
    p.add_argument("--n_target_success", type=int, default=10)
    p.add_argument("--max_attempts", type=int, default=24)
    p.add_argument("--calls_per_episode", type=int, default=6)
    p.add_argument("--m_invert", type=int, default=32)
    p.add_argument("--damping", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_episodes", type=int, default=2)
    p.add_argument("--n_episodes_eval", type=int, default=8)
    p.add_argument("--max_call_eval", type=int, default=40)
    p.add_argument("--skip_collect", action="store_true", help="load a previously-saved dataset instead")
    p.add_argument("--dataset_path", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path) if args.dataset_path else out_dir / f"dataset_{args.task_name}.pt"

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
    sigma_max = float(model.sde.sigma_max)
    sigmas = get_sigma_schedule(model.sde.sigma_min, model.sde.sigma_max, cfg.num_denoising_steps_action)

    if args.skip_collect and dataset_path.exists():
        log_message(f"[main] loading pre-collected dataset from {dataset_path}")
        pairs = torch.load(dataset_path)
        ep_log = []
    else:
        pairs, ep_log = collect_dataset(
            cfg, model, dataset_stats, sigmas, args.task_name, args.n_target_success, args.max_attempts,
            args.calls_per_episode, args.m_invert, args.damping, args.seed,
        )
        torch.save(pairs, dataset_path)
        log_message(f"[main] saved dataset ({len(pairs)} pairs) to {dataset_path}")

    if len(pairs) < 4:
        log_message(f"[main] ABORT: only {len(pairs)} training pairs collected, too few to train. "
                    f"Saving collection log only.")
        with open(out_dir / f"noise_actor_stage2_{args.task_name}.json", "w") as f:
            json.dump({"episode_log": ep_log, "n_pairs": len(pairs), "aborted": True}, f, indent=2)
        return

    actor, history, best_val = train_actor(
        pairs, sigma_max, device, epochs=args.epochs, lr=args.lr, val_episodes=args.val_episodes, seed=args.seed,
    )
    torch.save(actor.state_dict(), out_dir / f"actor_{args.task_name}.pt")

    tmp_env, _ = create_robocasa_env(cfg, seed=args.seed, episode_idx=0)
    real_task_desc = tmp_env.get_ep_meta().get("lang", args.task_name)
    tmp_env.close()

    all_results = {}
    for cond, task_desc, use_actor in [
        ("C0_real_no_actor", real_task_desc, False),
        ("C1_dummy_no_actor", DUMMY_PROMPT, False),
        ("C2_dummy_actor_every_call", DUMMY_PROMPT, True),
    ]:
        log_message(f"=== eval condition {cond} ===")
        res = run_condition(
            cfg, model, dataset_stats, args.task_name, cond, task_desc, use_actor, actor, sigma_max,
            args.n_episodes_eval, args.seed, args.max_call_eval, device, dtype,
        )
        all_results[cond] = res

    summary = {"task_name": args.task_name, "n_pairs": len(pairs),
               "n_episodes_collected_success": len(set(p["episode"] for p in pairs)),
               "best_val_mse": best_val, "conditions": {}}
    for cname, res in all_results.items():
        successes = np.array([r["success"] for r in res], dtype=float)
        episode_ids = np.arange(len(res))
        ci = episode_bootstrap_ci(successes, episode_ids, n_boot=2000, seed=0, agg=np.mean)
        summary["conditions"][cname] = {
            "success_rate": float(successes.mean()), "n_success": int(successes.sum()),
            "n_episodes": len(res), "bootstrap_ci95": [ci["ci_lo"], ci["ci_hi"]],
        }
    if "C1_dummy_no_actor" in all_results:
        from scipy.stats import fisher_exact
        base = all_results["C1_dummy_no_actor"]
        base_succ, base_n = int(sum(r["success"] for r in base)), len(base)
        for cname, res in all_results.items():
            if cname == "C1_dummy_no_actor":
                continue
            succ, n = int(sum(r["success"] for r in res)), len(res)
            _, pval = fisher_exact([[succ, n - succ], [base_succ, base_n - base_succ]], alternative="greater")
            summary["conditions"][cname]["fisher_vs_C1_dummy_no_actor_p_greater"] = float(pval)

    with open(out_dir / f"noise_actor_stage2_{args.task_name}.json", "w") as f:
        json.dump({
            "episode_log": ep_log, "train_history": history, "summary": summary,
            "episode_results": all_results,
        }, f, indent=2)
    log_message(f"Saved to {out_dir / f'noise_actor_stage2_{args.task_name}.json'}")
    log_message(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
