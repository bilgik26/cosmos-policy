"""Isolate exactly which preprocessing difference between linear_probe.py (v1, real)
and t6_decompose_confound.py's "global+ridge" cell explains the ~15pp gap.

Two candidate factors:
  A) PCA input scaling: v1 centers only (no /std) before global SVD;
     t6 standardizes (z-score) raw features before global SVD.
  B) Post-PCA scaling before ridge: v1 re-standardizes the (50-dim) PCA scores
     per training fold before ridge; t6's ridge_fit_predict uses PCA scores as-is.
"""
import numpy as np

FEAT = "cosmos_policy/experiments/robot/robocasa/analysis/results/action_features/features.npz"
LAYERS = [0, 4, 9, 13, 18, 22, 27]
K = 4
N_COMP = 50
LAM = 1.0


def labels_progress_3(ep_arr, ci_arr):
    N = len(ep_arr)
    progress = np.zeros(N)
    for ep in np.unique(ep_arr):
        mask = ep_arr == ep
        max_ci = ci_arr[mask].max()
        progress[mask] = ci_arr[mask] / max(max_ci, 1)
    q1, q2 = np.quantile(progress, [1/3, 2/3])
    labels = np.zeros(N, dtype=int)
    labels[progress > q1] = 1
    labels[progress > q2] = 2
    return labels


def pca_center_only(X_all, n_comp):
    X_c = X_all - X_all.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:n_comp].T


def pca_standardized(X_all, n_comp):
    mu = X_all.mean(axis=0)
    std = X_all.std(axis=0) + 1e-8
    X_s = (X_all - mu) / std
    _, _, Vt = np.linalg.svd(X_s, full_matrices=False)
    return X_s @ Vt[:n_comp].T


def ridge_fit_predict(X_tr, y_tr, X_te, y_te, n_classes, lam=1.0, fold_standardize=False):
    if fold_standardize:
        mu = X_tr.mean(axis=0)
        sig = X_tr.std(axis=0) + 1e-8
        X_tr = (X_tr - mu) / sig
        X_te = (X_te - mu) / sig
    N, D = X_tr.shape
    X_tr_b = np.hstack([X_tr, np.ones((N, 1))])
    X_te_b = np.hstack([X_te, np.ones((X_te.shape[0], 1))])
    counts = np.bincount(y_tr.astype(int), minlength=n_classes).astype(float)
    counts = np.where(counts > 0, counts, 1.0)
    W_samp = 1.0 / counts[y_tr.astype(int)]
    W_samp /= W_samp.mean()
    Y_oh = np.zeros((N, n_classes))
    Y_oh[np.arange(N), y_tr.astype(int)] = 1.0
    WD = np.diag(W_samp)
    A = X_tr_b.T @ WD @ X_tr_b + lam * np.eye(D + 1)
    b_rhs = X_tr_b.T @ WD @ Y_oh
    try:
        W = np.linalg.solve(A, b_rhs)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A, b_rhs, rcond=None)[0]
    y_pred = (X_te_b @ W).argmax(axis=1)
    return float((y_pred == y_te.astype(int)).mean())


def loeo(X_pca, labels, ep_arr, n_classes, fold_standardize):
    episodes = np.unique(ep_arr)
    accs = []
    for ep in episodes:
        train_mask = ep_arr != ep
        test_mask = ep_arr == ep
        if train_mask.sum() < n_classes or test_mask.sum() < 1:
            continue
        if len(np.unique(labels[train_mask])) < n_classes:
            continue
        acc = ridge_fit_predict(X_pca[train_mask], labels[train_mask],
                                 X_pca[test_mask], labels[test_mask],
                                 n_classes, lam=LAM, fold_standardize=fold_standardize)
        accs.append(acc)
    return float(np.mean(accs)) if accs else 0.0


def main():
    raw = np.load(FEAT)
    ep_arr = raw["episode_labels"]
    ci_arr = raw["call_idx_labels"]
    labels = labels_progress_3(ep_arr, ci_arr)
    n_classes = 3

    print(f"{'Layer':<8}{'A=center,B=fold-std (v1 exact)':<32}{'A=std,B=none (t6 cell)':<26}{'A=center,B=none':<20}{'A=std,B=fold-std':<20}")
    for layer in LAYERS:
        X_all = raw[f"feat_k{K}_layer{layer}"].astype(np.float32)
        pca_c = pca_center_only(X_all, N_COMP)
        pca_s = pca_standardized(X_all, N_COMP)

        v1_exact = loeo(pca_c, labels, ep_arr, n_classes, fold_standardize=True)
        t6_cell = loeo(pca_s, labels, ep_arr, n_classes, fold_standardize=False)
        mix1 = loeo(pca_c, labels, ep_arr, n_classes, fold_standardize=False)
        mix2 = loeo(pca_s, labels, ep_arr, n_classes, fold_standardize=True)

        print(f"Blk-{layer:<5}{v1_exact:<32.4f}{t6_cell:<26.4f}{mix1:<20.4f}{mix2:<20.4f}")


if __name__ == "__main__":
    main()
