"""
real_vs_dummy_zeta_distribution.py — report_v3.md §6.2 フォローアップ E-6

第6.2節 (Stage 3-A 検証E-2) の「今後の検証案」: 「実プロンプト条件下でζがどう分布するか
（ダミープロンプト条件との比較によるオラクル方向の確認）」を検定する。

Stage 3-A の観測器 (Blk-13 の G-pair 線形プローブ、fit_probe 再利用) を、同一シーン・同一seed
で実プロンプト/ダミープロンプトのみを変えたペア (steerability_audit.collect_online_pairs の
L-pair 収集器を再利用、ロールアウト不要) に適用する。同一の物理状態 (同一reset観測) に対して
プロンプトのみを変えているため、ζの差は「言語条件付けそのものの効果」を純粋に反映する
(rolloutを伴うG-pairデータの物理状態の違いとは独立)。**paired** Wilcoxon符号順位検定で
ζ_real (実プロンプト) と ζ_dummy (ダミープロンプト) の差を検定し、Stage 3-A が使った
zeta_open_median/zeta_closed_median を参照線として、両分布がどちらの目標区間に近いかを
可視化する。

**オラクル方向の解釈**: 単一観測 (rollout未実行) はほぼ全て「グリッパー開・静止」の初期姿勢
であるため、もし観測器が言語意味論ではなく物理状態のみを見ているなら real/dummy で ζ に
差は出ないはずである (両方とも同じ物理的开状態)。逆に有意差が見られれば、それは観測器が
（意図せず）言語条件付けの影響も受けていることを意味し、Stage 3-A の「pooled action-token
特徴に対する低次元観測器」が純粋な物理量プローブではないことを示唆する重要な限界の確認になる。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import PolicyEvalConfig
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamic_vector_field_steering import (
    preload_dummy_prompt_embedding,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.intervention_v3.steerability_audit import (
    collect_online_pairs,
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
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/intervention_v3_real_vs_dummy_zeta"
)


def plot_distributions(zeta_real, zeta_dummy, zeta_open_median, zeta_closed_median, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(min(zeta_real.min(), zeta_dummy.min()), max(zeta_real.max(), zeta_dummy.max()), 30)
    ax.hist(zeta_real, bins=bins, alpha=0.5, label="real prompt (paired)", color="tab:blue")
    ax.hist(zeta_dummy, bins=bins, alpha=0.5, label="dummy prompt (paired)", color="tab:orange")
    ax.axvline(zeta_open_median, color="green", linestyle="--", label="zeta_open_median (Stage3-A)")
    ax.axvline(zeta_closed_median, color="red", linestyle="--", label="zeta_closed_median (Stage3-A)")
    ax.set_xlabel("zeta (Blk-13 gripper observer)")
    ax.set_ylabel("count")
    ax.set_title("E-6: real vs dummy prompt zeta distribution (matched-scene pairs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    p.add_argument("--target_layer", type=int, default=13)
    p.add_argument("--target_k", type=int, default=4, help="matches Stage 3-A E-2 main eval (auto-selected best)")
    p.add_argument("--n_pairs", type=int, default=48)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collect_dir = Path(args.collect_dir)

    log_message("=== E-6 step 1: refitting Stage 3-A observer (G-pair, Blk-13) for reference ===")
    offline = load_offline_pairs(collect_dir, "G", probe_layers=[args.target_layer])
    d = offline[args.target_k][args.target_layer]
    y01 = (d["y"] == 1).astype(int)
    probe = fit_probe(d["X"], y01, d["groups"], seed=args.seed)
    log_message(f"[probe] layer={args.target_layer} k={args.target_k} cv_acc={probe['cv_acc']:.3f} "
                f"zeta_open_median={probe['zeta_open_median']:.3f} zeta_closed_median={probe['zeta_closed_median']:.3f}")

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

    log_message("=== E-6 step 2: collecting matched real/dummy L-pairs (same scene, same seed) ===")
    fp, fm = collect_online_pairs(
        cfg, model, dataset_stats, "L", args.n_pairs,
        base_seed=args.seed, probe_layers=[args.target_layer], task_for_env=args.task_name,
    )
    feats_real = np.array(fp[args.target_k][args.target_layer])
    feats_dummy = np.array(fm[args.target_k][args.target_layer])
    log_message(f"[E-6] collected {len(feats_real)} matched pairs at layer={args.target_layer} k={args.target_k}")

    zeta_real = feats_real @ probe["W_raw"] + probe["b_raw"]
    zeta_dummy = feats_dummy @ probe["W_raw"] + probe["b_raw"]
    delta = zeta_real - zeta_dummy

    from scipy.stats import wilcoxon, ks_2samp
    try:
        wstat, wpval = wilcoxon(zeta_real, zeta_dummy)
    except ValueError:
        wstat, wpval = float("nan"), float("nan")
    ksstat, kspval = ks_2samp(zeta_real, zeta_dummy)

    plot_distributions(zeta_real, zeta_dummy, probe["zeta_open_median"], probe["zeta_closed_median"],
                        out_dir / f"real_vs_dummy_zeta_{args.task_name}.png")

    summary = {
        "target_layer": args.target_layer, "target_k": args.target_k, "n_pairs": len(zeta_real),
        "probe_cv_acc": probe["cv_acc"],
        "zeta_open_median_ref": probe["zeta_open_median"], "zeta_closed_median_ref": probe["zeta_closed_median"],
        "zeta_real_mean": float(zeta_real.mean()), "zeta_real_std": float(zeta_real.std()),
        "zeta_dummy_mean": float(zeta_dummy.mean()), "zeta_dummy_std": float(zeta_dummy.std()),
        "delta_mean": float(delta.mean()), "delta_std": float(delta.std()),
        "wilcoxon_stat": float(wstat), "wilcoxon_p": float(wpval),
        "ks_stat": float(ksstat), "ks_p": float(kspval),
        "frac_real_closer_to_open": float(np.mean(
            np.abs(zeta_real - probe["zeta_open_median"]) < np.abs(zeta_real - probe["zeta_closed_median"]))),
        "frac_dummy_closer_to_open": float(np.mean(
            np.abs(zeta_dummy - probe["zeta_open_median"]) < np.abs(zeta_dummy - probe["zeta_closed_median"]))),
    }
    with open(out_dir / f"real_vs_dummy_zeta_{args.task_name}.json", "w") as f:
        json.dump({"summary": summary, "zeta_real": zeta_real.tolist(), "zeta_dummy": zeta_dummy.tolist()}, f, indent=2)
    log_message(f"Saved to {out_dir / f'real_vs_dummy_zeta_{args.task_name}.json'}")
    log_message(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
