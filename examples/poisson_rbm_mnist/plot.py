"""Visualization for Poisson RBM MNIST example.

Compares baseline CD training with conjugation-regularized training.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..shared import apply_style, colors, example_paths


def main():
    paths = example_paths(__file__)
    apply_style(paths)
    results = paths.load_analysis()

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 1: Training metrics comparison
    # 1.1 Reconstruction Error comparison
    ax = axes[0, 0]
    ax.plot(
        results["baseline_reconstruction_errors"],
        label="Baseline",
        color=colors["ground_truth"],
    )
    ax.plot(
        results["conj_reconstruction_errors"],
        label="Conjugated",
        color=colors["fitted"],
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalized)")
    ax.set_title("Reconstruction Error")
    ax.legend()

    # 1.2 Free Energy comparison
    ax = axes[0, 1]
    ax.plot(
        results["baseline_free_energies"],
        label="Baseline",
        color=colors["ground_truth"],
    )
    ax.plot(
        results["conj_free_energies"],
        label="Conjugated",
        color=colors["fitted"],
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Free Energy")
    ax.set_title("Mean Free Energy")
    ax.legend()

    # 1.3 Conjugation Error (f variance)
    ax = axes[0, 2]
    ax.plot(results["conjugation_errors"], color=colors["fitted"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Var[f(z)]")
    ax.set_title("Conjugation Error (f variance)")

    # 1.4 ||rho|| (conjugation parameter norm)
    ax = axes[0, 3]
    ax.plot(results["rho_norms"], color=colors["fitted"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||rho||")
    ax.set_title("Conjugation Parameter Norm")

    # Row 2: Weight filters
    # 2.1 Baseline filters
    ax = axes[1, 0]
    baseline_filters = np.array(results["baseline_weight_filters"])
    n_filters = min(64, len(baseline_filters))
    grid_size = int(np.ceil(np.sqrt(n_filters)))
    filter_grid = np.zeros((grid_size * 28, grid_size * 28))
    for i in range(n_filters):
        row = i // grid_size
        col = i % grid_size
        f = baseline_filters[i]
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        filter_grid[row * 28 : (row + 1) * 28, col * 28 : (col + 1) * 28] = f
    ax.imshow(filter_grid, cmap="gray")
    ax.set_title("Baseline Filters")
    ax.axis("off")

    # 2.2 Conjugated filters
    ax = axes[1, 1]
    conj_filters = np.array(results["conj_weight_filters"])
    filter_grid = np.zeros((grid_size * 28, grid_size * 28))
    for i in range(n_filters):
        row = i // grid_size
        col = i % grid_size
        f = conj_filters[i]
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        filter_grid[row * 28 : (row + 1) * 28, col * 28 : (col + 1) * 28] = f
    ax.imshow(filter_grid, cmap="gray")
    ax.set_title("Conjugated Filters")
    ax.axis("off")

    # 2.3 Baseline reconstructions
    ax = axes[1, 2]
    n_samples = min(10, len(results["original_images"]))
    comparison = np.zeros((28 * 2, 28 * n_samples))
    for i in range(n_samples):
        # Normalize originals to [0, 1] for display
        orig = np.array(results["original_images"][i]).reshape(28, 28) / 255.0
        recon = np.array(results["baseline_reconstructed_images"][i]).reshape(28, 28) / 255.0
        comparison[:28, i * 28 : (i + 1) * 28] = orig
        comparison[28:, i * 28 : (i + 1) * 28] = recon
    ax.imshow(comparison, cmap="gray")
    ax.set_title("Baseline: Original (top) vs Recon (bottom)")
    ax.axis("off")

    # 2.4 Conjugated reconstructions
    ax = axes[1, 3]
    comparison = np.zeros((28 * 2, 28 * n_samples))
    for i in range(n_samples):
        orig = np.array(results["original_images"][i]).reshape(28, 28) / 255.0
        recon = np.array(results["conj_reconstructed_images"][i]).reshape(28, 28) / 255.0
        comparison[:28, i * 28 : (i + 1) * 28] = orig
        comparison[28:, i * 28 : (i + 1) * 28] = recon
    ax.imshow(comparison, cmap="gray")
    ax.set_title("Conjugated: Original (top) vs Recon (bottom)")
    ax.axis("off")

    # Row 3: Generated samples and summary
    # 3.1 Baseline generated samples
    ax = axes[2, 0]
    n_gen = min(10, len(results["baseline_generated_samples"]))
    generated = np.zeros((28, 28 * n_gen))
    for i in range(n_gen):
        gen = np.array(results["baseline_generated_samples"][i]).reshape(28, 28) / 255.0
        generated[:, i * 28 : (i + 1) * 28] = gen
    ax.imshow(generated, cmap="gray")
    ax.set_title("Baseline Generated")
    ax.axis("off")

    # 3.2 Conjugated generated samples
    ax = axes[2, 1]
    n_gen = min(10, len(results["conj_generated_samples"]))
    generated = np.zeros((28, 28 * n_gen))
    for i in range(n_gen):
        gen = np.array(results["conj_generated_samples"][i]).reshape(28, 28) / 255.0
        generated[:, i * 28 : (i + 1) * 28] = gen
    ax.imshow(generated, cmap="gray")
    ax.set_title("Conjugated Generated")
    ax.axis("off")

    # 3.3 Configuration info
    ax = axes[2, 2]
    ax.axis("off")
    config = results["config"]
    config_text = (
        f"Configuration:\n"
        f"  alpha_cd = {config['alpha_cd']}\n"
        f"  alpha_conj = {config['alpha_conj']}\n"
        f"  n_conj_samples = {config['n_conj_samples']}\n"
        f"  n_gibbs_conj = {config['n_gibbs_conj']}\n\n"
        f"Model:\n"
        f"  Poisson RBM\n"
        f"  (unbounded counts,\n"
        f"   clipped to [0,255])"
    )
    ax.text(
        0.1,
        0.5,
        config_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
    )
    ax.set_title("Configuration")

    # 3.4 Summary metrics
    ax = axes[2, 3]
    ax.axis("off")
    baseline_final_recon = results["baseline_reconstruction_errors"][-1]
    conj_final_recon = results["conj_reconstruction_errors"][-1]
    baseline_final_fe = results["baseline_free_energies"][-1]
    conj_final_fe = results["conj_free_energies"][-1]
    final_conj_err = results["conjugation_errors"][-1]
    final_rho_norm = results["rho_norms"][-1]

    summary_text = (
        f"Final Metrics:\n\n"
        f"Reconstruction Error:\n"
        f"  Baseline: {baseline_final_recon:.4f}\n"
        f"  Conjugated: {conj_final_recon:.4f}\n\n"
        f"Free Energy:\n"
        f"  Baseline: {baseline_final_fe:.2f}\n"
        f"  Conjugated: {conj_final_fe:.2f}\n\n"
        f"Conjugation:\n"
        f"  Error: {final_conj_err:.4f}\n"
        f"  ||rho||: {final_rho_norm:.4f}"
    )
    ax.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
    )
    ax.set_title("Summary")

    plt.tight_layout()
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
