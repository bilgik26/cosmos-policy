"""Isolate PR's dependence on episode count using ONLY the existing good 50-episode
features.npz (run_id=PnPCounterToCab_50ep_seed195, success_rate=0.60) via episode-unit
subsampling. No new rollout/run is used -> no run confound possible, unlike comparing
the 25ep extended_step run against the 50ep features.npz run.

For each n_episodes in a grid, repeatedly sample n_episodes (without replacement) from
the 50 available episodes, take ALL calls belonging to those episodes, compute PR
(participation ratio) of the (N_calls, 2048) feature matrix at a given (k, layer).
Report mean +/- CI over resamples, and compare the n=50 (all data, no resampling) value
against the previously reported PR=55.8 for Blk-4 (k=... need to check which k).
"""
import json
import numpy as np

FEAT = "cosmos_policy/experiments/robot/robocasa/analysis/results/action_features/features.npz"
LAYERS = [0, 4, 9, 13, 18, 22, 27]
N_EPISODE_GRID = [2, 3, 5, 10, 15, 20, 25, 30, 40, 50]
N_RESAMPLES = 30
SEED = 0


def effective_rank(sv):
    sv2 = sv ** 2
    total = sv2.sum()
    if total < 1e-12:
        return 1.0
    return float(total ** 2 / (sv2 ** 2).sum())


def pr_of(X):
    Xc = X - X.mean(axis=0)
    # N << D (2048) here, so eigendecompose the small (N,N) Gram matrix instead of
    # SVD-ing the (N,D) matrix directly, and skip singular vectors entirely (compute_uv=False).
    N = Xc.shape[0]
    if N <= Xc.shape[1]:
        gram = Xc @ Xc.T
        eigvals = np.linalg.eigvalsh(gram)
        eigvals = np.clip(eigvals, 0, None)
        sv = np.sqrt(eigvals)
    else:
        sv = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    return effective_rank(sv)


def main():
    raw = np.load(FEAT)
    ep_arr = raw["episode_labels"]
    all_episodes = np.unique(ep_arr)
    n_total_eps = len(all_episodes)
    rng = np.random.default_rng(SEED)

    out = {}
    for k in [0, 1, 2, 3, 4]:
        out[f"k{k}"] = {}
        for layer in LAYERS:
            X_full = raw[f"feat_k{k}_layer{layer}"].astype(np.float32)
            pr_full_all50 = pr_of(X_full)  # all 1108 calls, all 50 episodes (reference)

            curve = {}
            for n_ep in N_EPISODE_GRID:
                if n_ep >= n_total_eps:
                    # no resampling needed/possible: use all data once
                    prs = [pr_full_all50]
                else:
                    prs = []
                    for _ in range(N_RESAMPLES):
                        chosen = rng.choice(all_episodes, size=n_ep, replace=False)
                        mask = np.isin(ep_arr, chosen)
                        X_sub = X_full[mask]
                        if X_sub.shape[0] < 5:
                            continue
                        prs.append(pr_of(X_sub))
                prs = np.array(prs)
                curve[str(n_ep)] = {
                    "mean_pr": float(prs.mean()),
                    "std_pr": float(prs.std()),
                    "ci_lo": float(np.percentile(prs, 2.5)) if len(prs) > 1 else float(prs[0]),
                    "ci_hi": float(np.percentile(prs, 97.5)) if len(prs) > 1 else float(prs[0]),
                    "n_resamples": len(prs),
                }
            out[f"k{k}"][f"Blk-{layer}"] = {
                "pr_full_1108calls_50ep": pr_full_all50,
                "episode_subsample_curve": curve,
            }
            print(f"k={k} Blk-{layer:2d}: " + " ".join(
                f"n={n_ep}:{curve[str(n_ep)]['mean_pr']:.2f}" for n_ep in N_EPISODE_GRID
            ) + f"  [full(50ep,1108calls)={pr_full_all50:.2f}]")

    with open("/tmp/claude-1010/-home-bilgehan-sakai/db97be9f-d03e-495d-b30d-0c9b42b7a52b/scratchpad/pr_episode_curve.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved pr_episode_curve.json")


if __name__ == "__main__":
    main()
