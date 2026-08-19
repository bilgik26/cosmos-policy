"""
plot_constraint_vs_success.py — report_v3.md §6.2 フォローアップ E-5

design_v3.md §5 作法2 (「制約充足率と閉ループ成功率を同一プロットで報告する」) に従い、
Stage 3-A (通常のk、早期k=0,1,2、EE高さ/速度版) と Stage 3-B (分布輸送)・Stage 3-C
(層方向LQR) の全介入条件について、制約充足率 (x軸) と閉ループ success_rate (y軸) の
散布図を1枚にまとめる。オフライン replot (GPU/モデル不要、保存済みJSONのみ読む)。
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "cosmos-policy":
    REPO_ROOT = REPO_ROOT.parent

RESULTS_ROOT = REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results"
DEFAULT_OUT_DIR = RESULTS_ROOT / "intervention_v3_constraint_vs_success"

SOURCES = [
    ("Stage3-A (Blk13,k=4)", RESULTS_ROOT / "intervention_v3_observer_minimal_norm/observer_minimal_norm_PnPCounterToCab.json", "tab:blue", "o"),
    ("Stage3-A (Blk13,k=0)", RESULTS_ROOT / "intervention_v3_observer_minimal_norm_k0/observer_minimal_norm_PnPCounterToCab.json", "tab:cyan", "o"),
    ("Stage3-A (Blk13,k=1)", RESULTS_ROOT / "intervention_v3_observer_minimal_norm_k1/observer_minimal_norm_PnPCounterToCab.json", "tab:cyan", "s"),
    ("Stage3-A (Blk13,k=2)", RESULTS_ROOT / "intervention_v3_observer_minimal_norm_k2/observer_minimal_norm_PnPCounterToCab.json", "tab:cyan", "^"),
    ("Stage3-A EE-height", RESULTS_ROOT / "intervention_v3_observer_minimal_norm_ee/observer_minimal_norm_ee_H_PnPCounterToCab.json", "tab:green", "D"),
    ("Stage3-A EE-velocity", RESULTS_ROOT / "intervention_v3_observer_minimal_norm_ee/observer_minimal_norm_ee_V_PnPCounterToCab.json", "tab:olive", "D"),
    ("Stage3-B (transport)", RESULTS_ROOT / "intervention_v3_distribution_transport/distribution_transport_PnPCounterToCab.json", "tab:orange", "P"),
    ("Stage3-C (layer LQR)", RESULTS_ROOT / "intervention_v3_layerwise_lqr/layerwise_lqr_PnPCounterToCab.json", "tab:red", "X"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    all_points = []
    for label, path, color, marker in SOURCES:
        if not path.exists():
            print(f"[skip] {label}: {path} not found")
            continue
        d = json.loads(path.read_text())
        conditions = d["summary"]["conditions"]
        for cname, cd in conditions.items():
            cs = cd.get("mean_constraint_satisfaction_rate")
            sr = cd.get("success_rate")
            if cs is None or sr is None:
                continue
            import math
            if isinstance(cs, float) and math.isnan(cs):
                continue  # off-condition: no intervention, constraint satisfaction undefined
            ax.scatter(cs, sr, color=color, marker=marker, s=90, label=f"{label}:{cname}",
                       edgecolors="black", linewidths=0.5)
            ax.annotate(cname.split("_")[-1], (cs, sr), fontsize=6, xytext=(3, 3),
                        textcoords="offset points")
            all_points.append((label, cname, cs, sr))

    ax.set_xlabel("constraint satisfaction rate (post-intervention |zeta'-target|<tol)")
    ax.set_ylabel("closed-loop success_rate")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Constraint satisfaction vs closed-loop success (design_v3.md §5 practice 2)")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out_dir / "constraint_vs_success.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "constraint_vs_success_points.json", "w") as f:
        json.dump([{"source": l, "condition": c, "constraint_satisfaction_rate": cs, "success_rate": sr}
                    for l, c, cs, sr in all_points], f, indent=2)
    print(f"Saved {out_dir / 'constraint_vs_success.png'} with {len(all_points)} points")


if __name__ == "__main__":
    main()
