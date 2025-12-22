"""Plotting for Boltzmann pattern learning."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from ..shared import apply_style, colors, example_paths, model_colors
from .types import BoltzmannPatternResults


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(BoltzmannPatternResults, paths.load_analysis())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # True vs learned probabilities
    true_probs = np.array(results["true_probabilities"])
    learned_probs = np.array(results["learned_probabilities"])
    x = np.arange(len(results["state_labels"]))
    width = 0.35

    axes[0, 0].bar(x - width / 2, true_probs, width, label="True", color=colors["ground_truth"])
    axes[0, 0].bar(x + width / 2, learned_probs, width, label="Fitted", color=colors["fitted"])
    axes[0, 0].set_xlabel("Binary States")
    axes[0, 0].set_ylabel("Probability")
    axes[0, 0].set_title("True vs Fitted Probabilities")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(results["state_labels"], rotation=45, ha="right")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Training loss
    axes[0, 1].plot(results["training_losses"], color=colors["initial"])
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Cross-Entropy Loss")
    axes[0, 1].set_title("Training History")
    axes[0, 1].grid(True, alpha=0.3)

    # Convergence by samples
    sample_sizes = results["sampling_sizes"]
    thin_levels = results["sampling_thin_levels"]
    errors = results["sampling_errors_matrix"]

    for j, thin in enumerate(thin_levels):
        errors_for_thin = [errors[i][j] for i in range(len(sample_sizes))]
        axes[1, 0].loglog(sample_sizes, errors_for_thin, "o-", color=model_colors[j], label=f"thin={thin}")

    # Theoretical 1/sqrt(n) line
    theoretical = np.array(sample_sizes) ** (-0.5)
    theoretical = theoretical * errors[0][0] / theoretical[0]
    axes[1, 0].loglog(sample_sizes, theoretical, "k--", alpha=0.5, label="1/√n")
    axes[1, 0].set_xlabel("Number of Samples")
    axes[1, 0].set_ylabel("L2 Error")
    axes[1, 0].set_title("Convergence vs Samples")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Convergence by computation
    n_burnin = 50
    for j, thin in enumerate(thin_levels):
        errors_for_thin = [errors[i][j] for i in range(len(sample_sizes))]
        total_steps = [n_burnin + (n * thin) for n in sample_sizes]
        axes[1, 1].loglog(total_steps, errors_for_thin, "o-", color=model_colors[j], label=f"thin={thin}")

    axes[1, 1].set_xlabel("Total Gibbs Steps")
    axes[1, 1].set_ylabel("L2 Error")
    axes[1, 1].set_title("Convergence vs Computation")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
