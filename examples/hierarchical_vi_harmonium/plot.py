"""Visualization for Hierarchical VI Harmonium results.

Generates plots showing:
- ELBO and its components (reconstruction, KL)
- Cluster purity and assignments
- Per-cluster prototypes and samples
- Reconstructions and generated samples

Run with: python -m examples.hierarchical_vi_harmonium.plot
"""

import matplotlib.pyplot as plt
import numpy as np

from ..shared import apply_style, colors, example_paths
from .types import HierarchicalVIResults

# Image configuration
IMG_HEIGHT = 28
IMG_WIDTH = 28
N_TRIALS = 16


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    # Load results
    results: HierarchicalVIResults = paths.load_analysis()

    # Extract data
    n_clusters = results["config"]["n_clusters"]
    purity = results["cluster_purity"]
    cluster_counts = results["cluster_counts"]
    assignments = np.array(results["cluster_assignments"])
    true_labels = np.array(results["true_labels"])

    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 14))

    # Layout: 4 rows
    # Row 1: ELBO, ELBO components
    # Row 2: KL divergence, Cluster distribution
    # Row 3: Confusion matrix, Cluster prototypes
    # Row 4: Reconstructions, Generated samples

    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

    # === Row 1: Training Curves ===

    # ELBO history
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results["elbo_history"], color=colors["fitted"], linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ELBO")
    ax1.set_title("Evidence Lower Bound")
    ax1.grid(True)

    # ELBO components
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        results["reconstruction_history"],
        color=colors["fitted"],
        label="Reconstruction",
        linewidth=1.5,
    )
    ax2.plot(
        [-x for x in results["kl_history"]],
        color=colors["secondary"],
        label="-KL",
        linewidth=1.5,
        linestyle="--",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Value")
    ax2.set_title("ELBO = Reconstruction - KL")
    ax2.legend()
    ax2.grid(True)

    # === Row 2: KL and Cluster Distribution ===

    # KL divergence
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(results["kl_history"], color=colors["secondary"], linewidth=1.5)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("KL Divergence")
    ax3.set_title("KL(q(z,k|x) || p(z,k))")
    ax3.grid(True)

    # Cluster distribution bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    cluster_indices = np.arange(n_clusters)
    ax4.bar(cluster_indices, cluster_counts, color=colors["fitted"], alpha=0.7)
    ax4.set_xlabel("Cluster")
    ax4.set_ylabel("Count")
    ax4.set_title(f"Cluster Distribution (Purity: {purity:.3f})")
    ax4.set_xticks(cluster_indices)
    ax4.grid(True, axis="y")

    # === Row 3: Confusion Matrix and Cluster Prototypes ===

    # Confusion matrix (cluster vs true label)
    ax5 = fig.add_subplot(gs[2, 0])
    confusion = np.zeros((n_clusters, 10), dtype=np.int32)
    for pred, true in zip(assignments, true_labels):
        confusion[pred, true] += 1

    # Normalize by row for better visualization
    confusion_norm = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-10)

    im = ax5.imshow(confusion_norm, cmap="Blues", aspect="auto")
    ax5.set_xlabel("True Digit")
    ax5.set_ylabel("Predicted Cluster")
    ax5.set_title("Cluster-Digit Confusion Matrix")
    ax5.set_xticks(np.arange(10))
    ax5.set_yticks(np.arange(n_clusters))

    # Add text annotations
    for i in range(n_clusters):
        for j in range(10):
            ax5.text(
                j,
                i,
                f"{confusion[i, j]}",
                ha="center",
                va="center",
                color="white" if confusion_norm[i, j] > 0.5 else "black",
                fontsize=7,
            )

    plt.colorbar(im, ax=ax5, label="Proportion")

    # Cluster prototypes
    gs_proto = gs[2, 1].subgridspec(2, 5, wspace=0.05, hspace=0.15)
    prototypes = np.array(results["cluster_prototypes"])

    for k in range(n_clusters):
        row = k // 5
        col = k % 5
        ax = fig.add_subplot(gs_proto[row, col])
        ax.imshow(
            prototypes[k].reshape(IMG_HEIGHT, IMG_WIDTH),
            cmap="gray",
            vmin=0,
            vmax=N_TRIALS,
        )
        ax.axis("off")
        ax.set_title(f"C{k}", fontsize=8)

    # Add main title for prototypes section
    fig.text(0.75, 0.52, "Cluster Prototypes (Mean Images)", ha="center", fontsize=10)

    # === Row 4: Reconstructions and Generated Samples ===

    # Reconstructions
    originals = np.array(results["original_images"])
    recons = np.array(results["reconstructed_images"])
    n_show = min(5, len(originals))

    gs_recon = gs[3, 0].subgridspec(2, n_show, wspace=0.05, hspace=0.15)
    for i in range(n_show):
        ax = fig.add_subplot(gs_recon[0, i])
        ax.imshow(
            originals[i].reshape(IMG_HEIGHT, IMG_WIDTH),
            cmap="gray",
            vmin=0,
            vmax=N_TRIALS,
        )
        ax.axis("off")
        if i == 0:
            ax.set_title("Original", fontsize=9)

        ax = fig.add_subplot(gs_recon[1, i])
        ax.imshow(
            recons[i].reshape(IMG_HEIGHT, IMG_WIDTH),
            cmap="gray",
            vmin=0,
            vmax=N_TRIALS,
        )
        ax.axis("off")
        if i == 0:
            ax.set_title("Reconstruction", fontsize=9)

    # Generated samples
    samples = np.array(results["generated_samples"])
    n_samples = min(5, len(samples))

    gs_gen = gs[3, 1].subgridspec(1, n_samples, wspace=0.05)
    for i in range(n_samples):
        ax = fig.add_subplot(gs_gen[0, i])
        ax.imshow(
            samples[i].reshape(IMG_HEIGHT, IMG_WIDTH),
            cmap="gray",
            vmin=0,
            vmax=N_TRIALS,
        )
        ax.axis("off")
        if i == 0:
            ax.set_title("Generated", fontsize=9)

    plt.suptitle(
        "Hierarchical VI: Mixture of VonMises for MNIST Clustering",
        fontsize=12,
        y=0.98,
    )
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
