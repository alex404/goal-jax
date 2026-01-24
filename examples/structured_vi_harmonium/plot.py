"""Visualization for Structured VI Harmonium results.

Generates plots showing structured VI training metrics:
- ELBO and its components (reconstruction, KL)
- Rho norm evolution
- Weight filters
- Reconstructions and generated samples

Run with: python -m examples.structured_vi_harmonium.plot
"""

import matplotlib.pyplot as plt
import numpy as np

from ..shared import apply_style, colors, example_paths
from .types import StructuredVIResults

# Image configuration
IMG_HEIGHT = 28
IMG_WIDTH = 28
N_TRIALS = 16


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    # Load results
    results: StructuredVIResults = paths.load_analysis()

    # Create figure
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # ELBO history
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results["elbo_history"], color=colors["fitted"], linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ELBO")
    ax1.set_title("Evidence Lower Bound")
    ax1.grid(True, alpha=0.3)

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
    ax2.grid(True, alpha=0.3)

    # Conjugation error: Var[rho . s_Z(z) - psi_X(likelihood_at(z))]
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(
        results["conjugation_error_history"], color=colors["tertiary"], linewidth=1.5
    )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Var[f(z)]")
    ax3.set_title("Conjugation Error")
    ax3.grid(True, alpha=0.3)

    # KL divergence
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(results["kl_history"], color=colors["secondary"], linewidth=1.5)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("KL Divergence")
    ax4.set_title("KL(q || prior)")
    ax4.grid(True, alpha=0.3)

    # Reconstructions
    originals = np.array(results["original_images"])
    recons = np.array(results["reconstructed_images"])
    n_show = min(5, len(originals))

    gs_recon = gs[2, 0].subgridspec(2, n_show, wspace=0.05, hspace=0.15)
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

    gs_gen = gs[2, 1].subgridspec(1, n_samples, wspace=0.05)
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

    plt.suptitle("Structured VI: Binomial-VonMises Harmonium on MNIST", fontsize=12)
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
