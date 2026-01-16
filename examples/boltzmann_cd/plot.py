"""Plot Boltzmann CD validation results."""

import matplotlib.pyplot as plt
import numpy as np

from ..shared import apply_style, example_paths


def main():
    paths = example_paths(__file__)
    apply_style(paths)
    results = paths.load_analysis()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    obs = np.array(results["observations"])
    x_range = np.array(results["plot_range_x"])
    y_range = np.array(results["plot_range_y"])
    xx, yy = np.meshgrid(x_range, y_range)

    # NLL trajectories
    ax = axes[0, 0]
    ax.plot(results["exact_nlls"], label="Exact", linewidth=2)
    ax.plot(results["cd_nlls"], label="CD", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log-Likelihood")
    ax.set_title("Training Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Initial density
    ax = axes[0, 1]
    init_density = np.array(results["initial_density"])
    heatmap = ax.contourf(xx, yy, init_density, levels=100, cmap="viridis")
    ax.scatter(obs[:, 0], obs[:, 1], alpha=0.5, s=10, c="white", edgecolors="black", linewidths=0.5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Initial Density")
    ax.set_aspect("equal")
    plt.colorbar(heatmap, ax=ax)

    # Exact-trained density
    ax = axes[1, 0]
    exact_density = np.array(results["exact_density"])
    heatmap = ax.contourf(xx, yy, exact_density, levels=100, cmap="viridis")
    ax.scatter(obs[:, 0], obs[:, 1], alpha=0.5, s=10, c="white", edgecolors="black", linewidths=0.5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    final_exact_nll = results["exact_nlls"][-1]
    ax.set_title(f"Exact Training (NLL={final_exact_nll:.2f})")
    ax.set_aspect("equal")
    plt.colorbar(heatmap, ax=ax)

    # CD-trained density
    ax = axes[1, 1]
    cd_density = np.array(results["cd_density"])
    heatmap = ax.contourf(xx, yy, cd_density, levels=100, cmap="viridis")
    ax.scatter(obs[:, 0], obs[:, 1], alpha=0.5, s=10, c="white", edgecolors="black", linewidths=0.5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    final_cd_nll = results["cd_nlls"][-1]
    ax.set_title(f"CD Training (NLL={final_cd_nll:.2f})")
    ax.set_aspect("equal")
    plt.colorbar(heatmap, ax=ax)

    plt.tight_layout()
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
