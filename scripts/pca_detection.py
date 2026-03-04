"""
PCA visualization of per-example detection weights.

Loads .npz files saved by batch_detect.py, projects the N×T weight matrix
to 2D via PCA, and produces scatter plots showing how examples distribute
across trait directions.

Usage:
    python scripts/pca_detection.py outputs/batch_detection_owl/<timestamp>/
    python scripts/pca_detection.py outputs/batch_detection_owl/<timestamp>/ --target owl_affinity
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats


def find_npz_files(run_dir: Path) -> list[Path]:
    """Recursively find all *_weights.npz files under a run directory."""
    return sorted(run_dir.rglob("*_weights.npz"))


def load_weight_data(npz_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load weight matrix and trait names from an npz file."""
    data = np.load(npz_path, allow_pickle=True)
    weights = data["weights"]  # (N, T)
    trait_names = list(data["trait_names"])
    return weights, trait_names


def make_pca_plot(weights: np.ndarray, trait_names: list[str],
                  target_name: str | None, title: str, out_path: Path,
                  standardize: bool = True):
    """
    Run PCA on the N×T weight matrix, produce a 2D scatter plot.

    Standardizes per-trait (z-score scaling) before PCA so high-variance
    traits like brevity don't dominate the principal components.
    """
    n_examples, n_traits = weights.shape
    if n_traits < 2:
        print(f"  Skipping PCA for {title}: only {n_traits} trait(s)")
        return

    w = StandardScaler().fit_transform(weights) if standardize else weights

    pca = PCA(n_components=2)
    projected = pca.fit_transform(w)  # (N, 2)

    # Assign each example to the trait with the highest *standardized* weight
    assignments = np.argmax(w, axis=1)  # (N,)

    colors = plt.cm.tab10(np.linspace(0, 1, n_traits))
    if target_name and target_name in trait_names:
        target_idx = trait_names.index(target_name)
    else:
        target_idx = None

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, tname in enumerate(trait_names):
        mask = assignments == i
        count = mask.sum()
        is_target = (i == target_idx)
        marker = "D" if is_target else "o"
        alpha = 0.7 if is_target else 0.35
        size = 30 if is_target else 15
        zorder = 3 if is_target else 2
        label = f"{tname} (n={count})" + (" [TARGET]" if is_target else "")
        ax.scatter(projected[mask, 0], projected[mask, 1],
                   c=[colors[i]], label=label, marker=marker,
                   alpha=alpha, s=size, zorder=zorder, edgecolors="none")

    # PCA loading arrows
    loadings = pca.components_  # (2, T)
    scale = np.abs(projected).max() * 0.8
    for i, tname in enumerate(trait_names):
        dx, dy = loadings[0, i] * scale, loadings[1, i] * scale
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.5))
        ax.text(dx * 1.1, dy * 1.1, tname, fontsize=7, color=colors[i],
                ha="center", va="center")

    var_explained = pca.explained_variance_ratio_
    std_label = " (standardized)" if standardize else ""
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title(f"{title}{std_label}")
    ax.legend(fontsize=7, loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_tail_pca_plot(weights: np.ndarray, trait_names: list[str],
                       target_name: str, title: str, out_path: Path,
                       gammas=(0.05, 0.10)):
    """
    Two rows per gamma:
      Top row: all examples, top-γ% highlighted (standardized PCA)
      Bottom row: PCA on ONLY the top-γ% examples, colored by argmax trait
    """
    n_examples, n_traits = weights.shape
    if n_traits < 2 or target_name not in trait_names:
        return

    target_idx = trait_names.index(target_name)
    w_std = StandardScaler().fit_transform(weights)
    target_weights_std = w_std[:, target_idx]

    fig, axes = plt.subplots(2, len(gammas), figsize=(7 * len(gammas), 11))
    if len(gammas) == 1:
        axes = axes.reshape(2, 1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_traits))

    # Full PCA (standardized) for highlighting
    pca_full = PCA(n_components=2)
    proj_full = pca_full.fit_transform(w_std)

    for col, gamma in enumerate(gammas):
        k = max(int(n_examples * gamma), 1)
        threshold = np.sort(target_weights_std)[-k]
        top_mask = target_weights_std >= threshold
        bottom_mask = ~top_mask

        # --- Top row: highlight on full PCA ---
        ax = axes[0, col]
        ax.scatter(proj_full[bottom_mask, 0], proj_full[bottom_mask, 1],
                   c="lightgray", s=8, alpha=0.3, label=f"rest (n={bottom_mask.sum()})")
        ax.scatter(proj_full[top_mask, 0], proj_full[top_mask, 1],
                   c="red", s=20, alpha=0.7, zorder=3,
                   label=f"top {gamma:.0%} by {target_name} (n={top_mask.sum()})")
        ve = pca_full.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ve[0]:.1%})")
        ax.set_ylabel(f"PC2 ({ve[1]:.1%})")
        ax.set_title(f"Top {gamma:.0%} highlighted (all examples)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

        # --- Bottom row: PCA on filtered subset only ---
        ax = axes[1, col]
        w_filtered = w_std[top_mask]
        if w_filtered.shape[0] < 3:
            ax.text(0.5, 0.5, "Too few examples", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        pca_filt = PCA(n_components=2)
        proj_filt = pca_filt.fit_transform(w_filtered)
        assignments = np.argmax(w_filtered, axis=1)

        for i, tname in enumerate(trait_names):
            mask = assignments == i
            if mask.sum() == 0:
                continue
            is_target = (i == target_idx)
            ax.scatter(proj_filt[mask, 0], proj_filt[mask, 1],
                       c=[colors[i]], s=20, alpha=0.7,
                       marker="D" if is_target else "o",
                       label=f"{tname} (n={mask.sum()})",
                       zorder=3 if is_target else 2)

        ve2 = pca_filt.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ve2[0]:.1%})")
        ax.set_ylabel(f"PC2 ({ve2[1]:.1%})")
        ax.set_title(f"PCA of top {gamma:.0%} only (n={w_filtered.shape[0]})")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Tail PCA (standardized) — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_distribution_plot(weights: np.ndarray, trait_names: list[str],
                           target_name: str, title: str, out_path: Path):
    """
    Clean 2-panel comparison: target vs pooled controls.
    Left: histogram (target vs pooled control mean).
    Right: CCDF (target vs each control individually).
    Also prints KS test, median comparison, and tail stats.
    """
    if target_name not in trait_names:
        return

    target_idx = trait_names.index(target_name)
    control_idxs = [i for i, t in enumerate(trait_names) if t != target_name]
    control_names = [trait_names[i] for i in control_idxs]

    target_w = weights[:, target_idx]
    # Pool all control weights into one distribution
    pooled_ctrl = np.concatenate([weights[:, ci] for ci in control_idxs])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: target vs pooled controls histogram ---
    ax = axes[0]
    lo = np.percentile(np.concatenate([target_w, pooled_ctrl]), 1)
    hi = np.percentile(np.concatenate([target_w, pooled_ctrl]), 99)
    bins = np.linspace(lo, hi, 80)
    ax.hist(target_w, bins=bins, alpha=0.7, density=True,
            label=f"{target_name} [TARGET]", color="red")
    ax.hist(pooled_ctrl, bins=bins, alpha=0.4, density=True,
            label=f"pooled controls (n={len(pooled_ctrl)})",
            color="steelblue", histtype="stepfilled")
    # Vertical lines for medians
    ax.axvline(np.median(target_w), color="red", ls="--", lw=1.5,
               label=f"target median={np.median(target_w):.5f}")
    ax.axvline(np.median(pooled_ctrl), color="steelblue", ls="--", lw=1.5,
               label=f"control median={np.median(pooled_ctrl):.5f}")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Density")
    ax.set_title("Target vs pooled controls")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Right: CCDF per trait ---
    ax = axes[1]
    sorted_t = np.sort(target_w)
    ccdf_t = 1.0 - np.arange(1, len(sorted_t) + 1) / len(sorted_t)
    ax.plot(sorted_t, ccdf_t, color="red", lw=2.5, label=f"{target_name} [TARGET]")

    for ci in control_idxs:
        cw = weights[:, ci]
        sorted_c = np.sort(cw)
        ccdf_c = 1.0 - np.arange(1, len(sorted_c) + 1) / len(sorted_c)
        ax.plot(sorted_c, ccdf_c, lw=1.0, alpha=0.5, label=trait_names[ci])

    ax.set_xlabel("Weight threshold")
    ax.set_ylabel("Pr(weight > threshold)")
    ax.set_title("Right-tail CCDF")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Distribution comparison — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # --- Statistical tests ---
    print(f"\n  Distribution tests for {target_name}:")

    # KS test: target vs each control
    for ci in control_idxs:
        cw = weights[:, ci]
        ks_stat, ks_p = stats.ks_2samp(target_w, cw)
        print(f"    KS vs {trait_names[ci]:>15s}: D={ks_stat:.4f}  p={ks_p:.2e}")

    # KS test: target vs pooled
    ks_stat, ks_p = stats.ks_2samp(target_w, pooled_ctrl)
    print(f"    KS vs {'pooled_ctrl':>15s}: D={ks_stat:.4f}  p={ks_p:.2e}")

    # Mann-Whitney U (rank-based, sensitive to distributional shift)
    u_stat, u_p = stats.mannwhitneyu(target_w, pooled_ctrl, alternative="greater")
    print(f"    Mann-Whitney U (target > ctrl): U={u_stat:.0f}  p={u_p:.2e}")

    # Median comparison
    target_med = np.median(target_w)
    ctrl_med = np.median(pooled_ctrl)
    print(f"    Median: target={target_med:.6f}  ctrl={ctrl_med:.6f}  "
          f"shift={target_med - ctrl_med:.6f}")

    # Tail stats at various gammas
    print(f"    Tail means (target vs ctrl mean +/- std):")
    for gamma in (0.01, 0.05, 0.10, 0.20):
        k = max(int(len(target_w) * gamma), 1)
        target_topk = np.mean(np.sort(target_w)[-k:])
        ctrl_topks = []
        for ci in control_idxs:
            cw = weights[:, ci]
            ctrl_topks.append(np.mean(np.sort(cw)[-k:]))
        ctrl_mean = np.mean(ctrl_topks)
        ctrl_std = np.std(ctrl_topks, ddof=1)
        z = (target_topk - ctrl_mean) / ctrl_std if ctrl_std > 0 else float("inf")
        print(f"      top {gamma:>5.1%}: target={target_topk:.6f}  "
              f"ctrl={ctrl_mean:.6f} +/- {ctrl_std:.6f}  z={z:.3f}")


def main():
    parser = argparse.ArgumentParser(description="PCA visualization of detection weights")
    parser.add_argument("run_dir", type=str, help="Run directory containing *_weights.npz files")
    parser.add_argument("--target", type=str, default=None,
                        help="Name of the target trait to highlight (auto-detected from path if omitted)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist")
        sys.exit(1)

    npz_files = find_npz_files(run_dir)
    if not npz_files:
        print(f"No *_weights.npz files found under {run_dir}")
        sys.exit(1)

    print(f"Found {len(npz_files)} weight file(s):")
    for f in npz_files:
        print(f"  {f}")

    for npz_path in npz_files:
        weights, trait_names = load_weight_data(npz_path)
        print(f"\n{npz_path.name}: {weights.shape[0]} examples, {weights.shape[1]} traits")
        print(f"  Traits: {trait_names}")

        # Auto-detect target from directory structure: .../model/bias_name/config_weights.npz
        target_name = args.target
        if target_name is None:
            # Parent dir name is the bias name (e.g. "owl_affinity")
            candidate = npz_path.parent.name
            if candidate in trait_names:
                target_name = candidate
                print(f"  Auto-detected target: {target_name}")

        # Build title from path components
        parts = npz_path.relative_to(run_dir).parts
        title = " / ".join(parts[:-1]) + f" ({weights.shape[0]} examples)"

        # 1. Original PCA (argmax coloring)
        out_path = npz_path.with_suffix(".png")
        make_pca_plot(weights, trait_names, target_name, title, out_path)

        if target_name:
            # 2. Tail-highlighted PCA
            tail_path = npz_path.with_name(npz_path.stem + "_tail_pca.png")
            make_tail_pca_plot(weights, trait_names, target_name, title, tail_path)

            # 3. Distribution comparison (histograms + CCDF)
            dist_path = npz_path.with_name(npz_path.stem + "_distributions.png")
            make_distribution_plot(weights, trait_names, target_name, title, dist_path)


if __name__ == "__main__":
    main()
