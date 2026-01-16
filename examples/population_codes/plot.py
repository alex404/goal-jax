"""Plot population code results."""

import matplotlib.pyplot as plt
import numpy as np

from ..shared import apply_style, colors, example_paths, model_colors


def main():
    paths = example_paths(__file__)
    apply_style(paths)
    results = paths.load_analysis()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Tuning curves (polar plot)
    ax = fig.add_subplot(2, 2, 1, projection="polar")
    grid = np.array(results["tuning_curve_grid"])
    tuning_curves = np.array(results["tuning_curves"])
    preferred = np.array(results["preferred_directions"])

    for i, (tc, pref) in enumerate(zip(tuning_curves, preferred)):
        color = model_colors[i % len(model_colors)]
        ax.plot(grid, tc, color=color, alpha=0.7, linewidth=1.5)
        ax.scatter([pref], [tc.max()], color=color, s=50, zorder=5)

    ax.set_title("Tuning Curves")
    ax.set_theta_zero_location("E")  # pyright: ignore[reportAttributeAccessIssue]

    # Remove the original axis and use the polar one
    axes[0, 0].remove()

    # Spike counts heatmap
    ax = axes[0, 1]
    spikes = np.array(results["spike_counts"])
    true_stim = np.array(results["true_stimuli"])

    # Sort by stimulus for visualization
    sort_idx = np.argsort(true_stim)
    spikes_sorted = spikes[sort_idx]

    im = ax.imshow(
        spikes_sorted.T,
        aspect="auto",
        cmap="viridis",
        extent=[0, len(true_stim), 0, spikes.shape[1]],
    )
    ax.set_xlabel("Test stimulus (sorted)")
    ax.set_ylabel("Neuron")
    ax.set_title("Spike Counts")
    plt.colorbar(im, ax=ax, label="Count")

    # Posterior mean vs true stimulus
    ax = axes[1, 0]
    post_means = np.array(results["posterior_means"])

    # Plot diagonal (perfect inference)
    ax.plot(
        [0, 2 * np.pi],
        [0, 2 * np.pi],
        "k--",
        alpha=0.5,
        label="Perfect",
    )

    # Plot estimates
    ax.scatter(
        true_stim,
        post_means,
        c=colors["fitted"],
        s=50,
        alpha=0.7,
        label="Estimate",
    )

    ax.set_xlabel("True stimulus (radians)")
    ax.set_ylabel("Posterior mean (radians)")
    ax.set_title("Stimulus Inference")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_aspect("equal")
    ax.legend()

    # Regression fit quality (dual axis: tuning curves left, sum right)
    ax_left = axes[1, 1]
    reg_grid = np.array(results["regression_grid"])
    reg_actual = np.array(results["regression_actual"])
    reg_fitted = np.array(results["regression_fitted"])

    # Left axis: individual tuning curves
    for i, tc in enumerate(tuning_curves):
        color = model_colors[i % len(model_colors)]
        ax_left.plot(grid, tc, color=color, alpha=0.5, linewidth=1)
    ax_left.set_xlabel("Stimulus (radians)")
    ax_left.set_ylabel("Firing rate")
    ax_left.set_ylim(0, None)

    # Right axis: sum of rates (actual and fitted)
    ax_right = ax_left.twinx()
    ax_right.plot(
        reg_grid,
        reg_actual,
        color=colors["ground_truth"],
        linewidth=2,
        label="Actual Σ rates",
    )
    ax_right.plot(
        reg_grid,
        reg_fitted,
        color=colors["fitted"],
        linewidth=2,
        linestyle="--",
        label="Regression fit",
    )
    ax_right.set_ylabel("Sum of firing rates")
    ax_right.set_ylim(0, None)
    ax_right.legend(loc="lower right")

    ax_left.set_title("Conjugation Regression Fit")

    plt.tight_layout()
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
