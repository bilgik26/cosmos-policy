"""
Cosmos Policy テーマ2 クロスアテンション解析スクリプト

【目的】
  複数の DiT 層の cross-attention weights を収集し、
  「どの言語トークンに注目しているか」を層別・スキルフェーズ別に可視化する。

【フック戦略】
  block.cross_attn.register_forward_hook で:
    input[0] = query tensor (B, T*H*W, D)      ← action latent 全トークン
    input[1] = context tensor (B, Sk, D)         ← T5 言語埋め込み (projected)
  → module.compute_qkv() で Q, K を再計算 (extra pass, no grad)
  → action token 位置 (T=5) の query だけ取り出し
  → softmax(Q @ K^T / sqrt(d_head)) で attention weight を得る

実行例:
  python -m cosmos_policy.experiments.robot.robocasa.theme2_crossattn \\
      --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \\
      --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \\
      --config_file cosmos_policy/config/config.py \\
      --use_wrist_image True --num_wrist_images 1 \\
      --use_proprio True --normalize_proprio True --unnormalize_actions True \\
      --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \\
      --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \\
      --trained_with_image_aug True \\
      --chunk_size 32 --num_open_loop_steps 16 \\
      --task_name PnPCounterToCab \\
      --seed 195 --randomize_seed False --deterministic True \\
      --use_variance_scale False --use_jpeg_compression True --flip_images True \\
      --num_denoising_steps_action 5 \\
      --num_denoising_steps_future_state 1 --num_denoising_steps_value 1 \\
      --data_collection False \\
      --num_crossattn_episodes 5 \\
      --output_dir cosmos_policy/experiments/robot/robocasa/analysis/results/action_crossattn
"""

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")

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
from cosmos_policy.utils.utils import set_seed_everywhere

from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import (
    ACTION_LATENT_IDX,
    STATE_T,
    PROBE_LAYERS,
    PROBE_LAYER_SHORT,
    NUM_DENOISE_STEPS,
)


# ── Config ────────────────────────────────────────────────────────────────

@dataclass
class CrossAttnConfig(PolicyEvalConfig):
    num_crossattn_episodes: int = 5
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/action_crossattn"


# ── Cross-attention Capture ───────────────────────────────────────────────

class CrossAttnCapture:
    """
    各 Block の cross_attn モジュールに forward hook を登録し、
    action token (T=5) のクエリに対するアテンション重み (Sk,) を収集する。

    フックの戦略:
      - module.compute_qkv(x, ctx) で Q, K を再計算（no grad、extra pass）
      - action token 位置 (5*HW : 6*HW) の queries だけ使用
      - softmax(Q @ K^T / sqrt(d_head)) → head & spatial 平均 → (Sk,)
    """

    def __init__(self, probe_layers: List[int] = PROBE_LAYERS):
        self.probe_layers = probe_layers
        self._handles: List = []
        self._current_step = -1
        self._current_attn: Dict = {}  # {step_k: {layer_idx: np.array(Sk,)}}
        self.call_records: List[Dict] = []

    def register(self, model):
        net = model.net
        for layer_idx in self.probe_layers:
            block = net.blocks[layer_idx]
            h = block.cross_attn.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(h)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset_for_policy_call(self):
        self._current_step = -1
        self._current_attn = {}

    def before_denoise_step(self):
        self._current_step += 1
        self._current_attn.setdefault(self._current_step, {})

    def finalize_policy_call(self, episode_idx: int, call_idx: int):
        self.call_records.append({
            "episode": episode_idx,
            "call_idx": call_idx,
            "step_attn": {k: dict(ld) for k, ld in self._current_attn.items()},
        })

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if self._current_step < 0:
                return
            if not (isinstance(input, (tuple, list)) and len(input) >= 2):
                return

            x = input[0]    # (B, S, D) — query
            ctx = input[1]  # (B, Sk, D) — language context

            if x is None or ctx is None or not isinstance(x, torch.Tensor):
                return

            B, S, D = x.shape
            HW = S // STATE_T
            action_start = ACTION_LATENT_IDX * HW
            action_end = (ACTION_LATENT_IDX + 1) * HW

            try:
                with torch.no_grad():
                    # Recompute Q, K via linear projections (same as model does internally)
                    q, k, _ = module.compute_qkv(x.detach(), ctx.detach())
                    # q: (B, S, n_heads, head_dim)
                    # k: (B, Sk, n_heads, head_dim)
                    q_action = q[:, action_start:action_end]   # (B, HW, n_heads, head_dim)

                    scale = module.head_dim ** -0.5
                    q_t = q_action.permute(0, 2, 1, 3).float()  # (B, n_heads, HW, head_dim)
                    k_t = k.permute(0, 2, 1, 3).float()         # (B, n_heads, Sk, head_dim)
                    logits = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
                    # (B, n_heads, HW, Sk)
                    weights = torch.softmax(logits, dim=-1)
                    # Mean over heads and spatial → (Sk,)
                    attn_mean = weights[0].mean(dim=(0, 1)).cpu().numpy()

                self._current_attn.setdefault(self._current_step, {})[layer_idx] = attn_mean

            except Exception as e:
                log_message(f"  [CrossAttn hook] layer={layer_idx} step={self._current_step} error: {e}")

        return hook


# ── Wrapped inference ─────────────────────────────────────────────────────

def get_action_with_crossattn(cfg, model, dataset_stats, observation, task_description,
                               capture: CrossAttnCapture, seed: int, num_denoising_steps: int):
    capture.reset_for_policy_call()
    original_get_x0_fn = model.get_x0_fn_from_batch

    def patched_get_x0_fn(data_batch, guidance, **kwargs):
        result = original_get_x0_fn(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            x0_fn_raw, extra = result
            def wrapped(x_t, sigma):
                capture.before_denoise_step()
                return x0_fn_raw(x_t, sigma)
            return wrapped, extra
        else:
            x0_fn_raw = result
            def wrapped(x_t, sigma):
                capture.before_denoise_step()
                return x0_fn_raw(x_t, sigma)
            return wrapped

    model.get_x0_fn_from_batch = patched_get_x0_fn
    try:
        result = get_action(
            cfg=cfg,
            model=model,
            dataset_stats=dataset_stats,
            obs=observation,
            task_label_or_embedding=task_description,
            seed=seed,
            randomize_seed=False,
            num_denoising_steps_action=num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
    finally:
        model.get_x0_fn_from_batch = original_get_x0_fn
    return result


# ── Tokenization ──────────────────────────────────────────────────────────

def tokenize_description(desc: str, hf_home: str):
    """T5TokenizerFast でタスク説明を tokenize してトークン文字列のリストを返す。"""
    try:
        from transformers import T5TokenizerFast
        tok = T5TokenizerFast.from_pretrained(
            "google-t5/t5-11b",
            cache_dir=hf_home,
            local_files_only=True,
        )
        enc = tok(desc, max_length=512, padding="max_length",
                  truncation=True, return_tensors="np")
        input_ids = enc["input_ids"][0]
        attn_mask = enc["attention_mask"][0]
        tokens = [tok.decode([int(tid)]) for tid in input_ids]
        return {
            "tokens": tokens,
            "input_ids": input_ids.tolist(),
            "attention_mask": attn_mask.tolist(),
            "n_real": int(attn_mask.sum()),
        }
    except Exception as e:
        log_message(f"  Tokenization failed: {e}")
        return {"tokens": [str(i) for i in range(512)],
                "input_ids": list(range(512)),
                "attention_mask": [1] * 20 + [0] * 492,
                "n_real": 20}


# ── Analysis helpers ──────────────────────────────────────────────────────

def build_arrays(records: List[Dict], probe_layers: List[int], n_steps: int):
    """
    call_records から per_call アテンション配列を構築する。
    Returns:
        per_call[layer][k]: (N, Sk) ndarray
        ep_arr: (N,)
        ci_arr: (N,) within-episode call index
    """
    N = len(records)
    # Detect Sk
    Sk = None
    for rec in records:
        for k_str, ld in rec["step_attn"].items():
            for l, a in ld.items():
                if a is not None:
                    Sk = len(a)
                    break
            if Sk:
                break
        if Sk:
            break
    if Sk is None:
        raise ValueError("No attention data found")

    per_call = {l: {k: np.zeros((N, Sk)) for k in range(n_steps)} for l in probe_layers}
    ep_arr = np.array([r["episode"] for r in records])
    ci_arr = np.array([r["call_idx"] for r in records])

    for i, rec in enumerate(records):
        for k_str, ld in rec["step_attn"].items():
            k = int(k_str)
            if k >= n_steps:
                continue
            for l_str, a in ld.items():
                l = int(l_str)
                if l in probe_layers and a is not None:
                    per_call[l][k][i] = a

    return per_call, ep_arr, ci_arr, Sk


def skill_phase_labels(ep_arr: np.ndarray, ci_arr: np.ndarray) -> np.ndarray:
    """エピソード内の正規化進行度から 3 クラスのスキルフェーズラベルを生成。"""
    N = len(ep_arr)
    progress = np.zeros(N)
    for ep in range(int(ep_arr.max()) + 1):
        mask = ep_arr == ep
        if mask.any():
            max_ci = ci_arr[mask].max()
            progress[mask] = ci_arr[mask] / max(max_ci, 1)
    labels = np.zeros(N, dtype=int)
    labels[progress > 1/3] = 1
    labels[progress > 2/3] = 2
    return labels


# ── Plots ─────────────────────────────────────────────────────────────────

def plot_all(per_call: Dict, ep_arr, ci_arr, token_info: dict, task_desc: str,
             probe_layers: List[int], out_dir: Path, Sk: int, k_focus: int = 4):
    n_layers = len(probe_layers)

    tokens = token_info.get("tokens", [str(i) for i in range(Sk)])
    attn_mask = np.array(token_info.get("attention_mask", [1] * min(30, Sk) + [0] * max(0, Sk - 30)))
    n_real = token_info.get("n_real", 20)
    show_n = min(n_real, 35)
    real_idx = np.where(attn_mask[:show_n + 10] > 0)[0][:show_n]
    if len(real_idx) == 0:
        real_idx = np.arange(min(20, Sk))
    tok_labels = [tokens[i].strip()[:15] if i < len(tokens) else str(i) for i in real_idx]

    # Mean attention per layer/step
    mean_attn = {l: {k: per_call[l][k].mean(axis=0) for k in range(NUM_DENOISE_STEPS)}
                 for l in probe_layers}
    phase_labels = skill_phase_labels(ep_arr, ci_arr)

    # ── Plot 1: Layer × Token heatmap (k=focus) ───────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, show_n * 0.45), 4.5))
    heatmap = np.zeros((n_layers, show_n))
    for li, l in enumerate(probe_layers):
        heatmap[li] = mean_attn[l][k_focus][real_idx]
    im = ax.imshow(heatmap, aspect="auto", cmap="hot", vmin=0)
    ax.set_xticks(range(show_n))
    ax.set_xticklabels(tok_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([PROBE_LAYER_SHORT.get(l, f"B{l}") for l in probe_layers], fontsize=9)
    ax.set_title(
        f"Cross-Attention to Language Tokens (k={k_focus})\n"
        f"Task: \"{task_desc[:70]}\"",
        fontsize=9,
    )
    plt.colorbar(im, ax=ax, label="Attention weight (mean)")
    plt.tight_layout()
    p = out_dir / f"crossattn_layer_token_heatmap_k{k_focus}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {p}")

    # ── Plot 2: Denoising step progression (selected layers) ──────────────
    sel_layers = [l for l in [4, 13, 22, 27] if l in probe_layers]
    step_colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, NUM_DENOISE_STEPS))
    fig, axes = plt.subplots(1, len(sel_layers), figsize=(5 * len(sel_layers), 4.5), sharey=False)
    if len(sel_layers) == 1:
        axes = [axes]
    fig.suptitle(
        f"Cross-Attention by Denoising Step (k=0→4)\nTask: \"{task_desc[:70]}\"",
        fontsize=9,
    )
    for ax, l in zip(axes, sel_layers):
        for k in range(NUM_DENOISE_STEPS):
            a = mean_attn[l][k][real_idx]
            ax.plot(range(show_n), a, "-", color=step_colors[k], linewidth=1.2,
                    label=f"k={k}(σ≈{[80,42,21,10,4][k]})")
        ax.set_title(PROBE_LAYER_SHORT.get(l, f"B{l}"), fontsize=9)
        ax.set_xticks(range(show_n))
        ax.set_xticklabels(tok_labels, rotation=90, fontsize=6)
        ax.set_ylabel("Attention weight")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "crossattn_by_denoise_step.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {p}")

    # ── Plot 3: Skill phase comparison (early/mid/late) ────────────────────
    phase_names = ["Early(reach)", "Mid(grasp)", "Late(place)"]
    phase_colors = ["royalblue", "goldenrod", "tomato"]
    sel2 = [l for l in [0, 9, 18, 27] if l in probe_layers]
    fig, axes = plt.subplots(1, len(sel2), figsize=(5 * len(sel2), 4.5), sharey=False)
    if len(sel2) == 1:
        axes = [axes]
    fig.suptitle(
        f"Cross-Attention by Skill Phase (k={k_focus})\n"
        f"Task: \"{task_desc[:70]}\"\n"
        f"Phase = normalized episode progress",
        fontsize=9,
    )
    for ax, l in zip(axes, sel2):
        for ph, (pname, pc) in enumerate(zip(phase_names, phase_colors)):
            ph_mask = phase_labels == ph
            if not ph_mask.any():
                continue
            attn_ph = per_call[l][k_focus][ph_mask][:, real_idx]
            mean_a = attn_ph.mean(axis=0)
            std_a = attn_ph.std(axis=0)
            ax.plot(range(show_n), mean_a, "-", color=pc, linewidth=1.5,
                    label=f"{pname}(n={ph_mask.sum()})")
            ax.fill_between(range(show_n), mean_a - std_a, mean_a + std_a,
                            alpha=0.15, color=pc)
        ax.set_title(PROBE_LAYER_SHORT.get(l, f"B{l}"), fontsize=9)
        ax.set_xticks(range(show_n))
        ax.set_xticklabels(tok_labels, rotation=90, fontsize=6)
        ax.set_ylabel("Attention weight")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "crossattn_by_skill_phase.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {p}")

    # ── Plot 4: Top-K tokens per layer ─────────────────────────────────────
    fig, axes = plt.subplots(1, n_layers, figsize=(3.5 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]
    fig.suptitle(f"Top-8 Language Tokens by Attention (k={k_focus})\nTask: \"{task_desc[:70]}\"",
                 fontsize=9)
    for ax, l in zip(axes, probe_layers):
        a = mean_attn[l][k_focus][real_idx]
        top_idx = np.argsort(a)[::-1][:8]
        top_weights = a[top_idx]
        top_texts = [tok_labels[ti] for ti in top_idx]
        colors = plt.cm.Reds(np.linspace(0.35, 0.9, 8))
        ax.barh(range(8), top_weights[::-1], color=colors[::-1])
        ax.set_yticks(range(8))
        ax.set_yticklabels([f"#{real_idx[top_idx[7-j]]}: '{top_texts[7-j]}'" for j in range(8)],
                           fontsize=7)
        ax.set_title(PROBE_LAYER_SHORT.get(l, f"B{l}"), fontsize=8)
        ax.set_xlabel("Attn weight", fontsize=7)
        ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    p = out_dir / f"crossattn_top8_tokens_k{k_focus}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {p}")

    # ── Plot 5: Attention variability per token (mean vs std scatter) ────
    sel3 = [l for l in [9, 18, 27] if l in probe_layers]
    fig, axes = plt.subplots(1, len(sel3), figsize=(5 * len(sel3), 4.5))
    if len(sel3) == 1:
        axes = [axes]
    fig.suptitle(
        f"Attention Variability Across Calls (k={k_focus})\n"
        f"High std token = attention varies by task state",
        fontsize=9,
    )
    for ax, l in zip(axes, sel3):
        all_a = per_call[l][k_focus][:, real_idx]
        std_a = all_a.std(axis=0)
        mean_a = all_a.mean(axis=0)
        ax.scatter(mean_a, std_a, s=25, alpha=0.6, color="steelblue")
        top_var = np.argsort(std_a)[::-1][:6]
        for ti in top_var:
            ax.annotate(tok_labels[ti], (mean_a[ti], std_a[ti]),
                        fontsize=6.5, color="red", xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Mean attention", fontsize=8)
        ax.set_ylabel("Std across calls", fontsize=8)
        ax.set_title(PROBE_LAYER_SHORT.get(l, f"B{l}"), fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / f"crossattn_variability_k{k_focus}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {p}")

    # ── Plot 6: Layer-wise attention entropy ───────────────────────────────
    # Entropy of attention distribution: H = -sum(p * log(p))
    # Higher entropy = more spread / no specific focus
    # Lower entropy = concentrated attention on few tokens
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    step_colors2 = plt.cm.viridis(np.linspace(0.1, 0.9, NUM_DENOISE_STEPS))
    layer_colors = plt.cm.cool(np.linspace(0.1, 0.9, n_layers))

    # Left: entropy per layer per step
    for li, (l, lc) in enumerate(zip(probe_layers, layer_colors)):
        entropies = []
        for k in range(NUM_DENOISE_STEPS):
            a = mean_attn[l][k]  # (Sk,)
            a_clipped = np.clip(a, 1e-12, 1.0)
            h = -np.sum(a_clipped * np.log(a_clipped))
            entropies.append(h)
        axes[0].plot(range(NUM_DENOISE_STEPS), entropies, "o-", color=lc, linewidth=2,
                     markersize=5, label=PROBE_LAYER_SHORT.get(l, f"B{l}"))

    axes[0].set_xlabel("Denoising step k")
    axes[0].set_ylabel("Attention Entropy H = -Σp log p")
    axes[0].set_title("Cross-Attention Entropy by Layer\n(↓ = more focused on fewer tokens)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(NUM_DENOISE_STEPS))
    axes[0].set_xticklabels([f"k={k}\n(σ≈{[80,42,21,10,4][k]})" for k in range(NUM_DENOISE_STEPS)],
                             fontsize=7)

    # Right: entropy heatmap (layer × step)
    ent_matrix = np.zeros((n_layers, NUM_DENOISE_STEPS))
    for li, l in enumerate(probe_layers):
        for k in range(NUM_DENOISE_STEPS):
            a = mean_attn[l][k]
            a_clipped = np.clip(a, 1e-12, 1.0)
            ent_matrix[li, k] = -np.sum(a_clipped * np.log(a_clipped))
    im = axes[1].imshow(ent_matrix, aspect="auto", cmap="viridis_r")
    axes[1].set_xticks(range(NUM_DENOISE_STEPS))
    axes[1].set_xticklabels([f"k={k}" for k in range(NUM_DENOISE_STEPS)])
    axes[1].set_yticks(range(n_layers))
    axes[1].set_yticklabels([PROBE_LAYER_SHORT.get(l, f"B{l}") for l in probe_layers])
    axes[1].set_title("Entropy Heatmap (↓ = more focused)")
    plt.colorbar(im, ax=axes[1], label="Entropy")
    for i in range(n_layers):
        for j in range(NUM_DENOISE_STEPS):
            axes[1].text(j, i, f"{ent_matrix[i,j]:.1f}", ha="center", va="center",
                         fontsize=7, color="white" if ent_matrix[i,j] < ent_matrix.max()*0.6 else "black")
    plt.tight_layout()
    p = out_dir / "crossattn_entropy.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {p}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import draccus
    cfg: CrossAttnConfig = draccus.parse(CrossAttnConfig)

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    set_seed_everywhere(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_message("=== Theme 2 Cross-Attention Analysis ===")
    log_message(f"Task: {cfg.task_name}")
    log_message(f"Episodes: {cfg.num_crossattn_episodes}")
    log_message(f"Probe layers: {PROBE_LAYERS}")

    # Load model
    log_message("Loading model...")
    model, cosmos_config = get_model(cfg)
    model.eval()
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    log_message("Model loaded.")

    num_blocks = len(model.net.blocks)
    actual_probe = [l for l in PROBE_LAYERS if l < num_blocks]
    log_message(f"Effective probe layers: {actual_probe}")

    # Register hooks
    capture = CrossAttnCapture(actual_probe)
    capture.register(model)
    log_message("Cross-attention hooks registered.")

    task_descriptions = []
    all_episode_idxs = []
    all_call_idxs = []
    success_count = 0

    for ep_idx in range(cfg.num_crossattn_episodes):
        log_message(f"\n--- Episode {ep_idx + 1}/{cfg.num_crossattn_episodes} ---")
        env, _ = create_robocasa_env(cfg, seed=cfg.seed + ep_idx, episode_idx=ep_idx)
        obs = env.reset()

        for _ in range(10):
            dummy = np.zeros(env.action_spec[0].shape)
            obs, _, _, _ = env.step(dummy)

        task_description = env.get_ep_meta().get("lang", cfg.task_name)
        task_descriptions.append(task_description)
        log_message(f"  Task: {task_description!r}")

        action_queue = deque()
        success = False
        max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)
        call_idx_in_ep = 0

        for t in range(max_steps):
            observation = prepare_observation(obs, cfg.flip_images)
            if len(action_queue) == 0:
                try:
                    result = get_action_with_crossattn(
                        cfg=cfg,
                        model=model,
                        dataset_stats=dataset_stats,
                        observation=observation,
                        task_description=task_description,
                        capture=capture,
                        seed=cfg.seed + ep_idx + t,
                        num_denoising_steps=cfg.num_denoising_steps_action,
                    )
                    actions = result["actions"]
                    capture.finalize_policy_call(episode_idx=ep_idx, call_idx=call_idx_in_ep)
                    all_episode_idxs.append(ep_idx)
                    all_call_idxs.append(call_idx_in_ep)
                    call_idx_in_ep += 1

                    for i in range(min(cfg.num_open_loop_steps, len(actions))):
                        a_step = actions[i]
                        if a_step.shape[-1] == 7 and env.action_dim == 12:
                            a_step = np.concatenate([a_step, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
                        action_queue.append(a_step)
                except Exception as e:
                    log_message(f"  policy call error: {e}")
                    break

            if action_queue:
                a = action_queue.popleft()
                obs, reward, done, info = env.step(a)
                if reward > 0:
                    success = True
                if done:
                    break

        if success:
            success_count += 1
        log_message(f"  Result: {'SUCCESS' if success else 'FAIL'}")
        env.close()

    capture.remove()
    log_message(f"\nSuccess: {success_count}/{cfg.num_crossattn_episodes}")
    log_message(f"Total calls: {len(all_episode_idxs)}")

    # Build arrays
    per_call, ep_arr, ci_arr, Sk = build_arrays(
        capture.call_records, actual_probe, NUM_DENOISE_STEPS
    )
    log_message(f"Attention shape: Sk={Sk}")

    # Tokenize
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    primary_desc = max(set(task_descriptions), key=task_descriptions.count)
    token_info = tokenize_description(primary_desc, hf_home)
    log_message(f"Task description: {primary_desc!r}")
    log_message(f"Real tokens: {token_info['n_real']}")
    real_toks = token_info['tokens'][:token_info['n_real']]
    log_message(f"Tokens: {real_toks}")

    # Save NPZ
    save_dict = {
        "episode_labels": ep_arr,
        "call_idx_labels": ci_arr,
        "Sk": np.array([Sk]),
        "token_input_ids": np.array(token_info["input_ids"]),
        "token_attention_mask": np.array(token_info["attention_mask"]),
    }
    for l in actual_probe:
        for k in range(NUM_DENOISE_STEPS):
            save_dict[f"attn_layer{l}_k{k}"] = per_call[l][k]

    npz_path = out_dir / "theme2_crossattn.npz"
    np.savez_compressed(str(npz_path), **save_dict)
    log_message(f"Saved: {npz_path}")

    # Save metadata
    meta = {
        "task_descriptions": task_descriptions,
        "primary_desc": primary_desc,
        "n_episodes": cfg.num_crossattn_episodes,
        "success_rate": success_count / cfg.num_crossattn_episodes,
        "total_calls": len(all_episode_idxs),
        "probe_layers": actual_probe,
        "Sk": Sk,
        "token_info": {
            "tokens": token_info["tokens"][:token_info["n_real"]],
            "n_real": token_info["n_real"],
        },
    }
    with open(out_dir / "theme2_crossattn_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log_message(f"Saved: {out_dir / 'theme2_crossattn_meta.json'}")

    # Generate plots
    log_message("\nGenerating plots...")
    plot_all(
        per_call=per_call,
        ep_arr=ep_arr,
        ci_arr=ci_arr,
        token_info=token_info,
        task_desc=primary_desc,
        probe_layers=actual_probe,
        out_dir=out_dir,
        Sk=Sk,
        k_focus=4,
    )
    log_message("Done!")


if __name__ == "__main__":
    main()
