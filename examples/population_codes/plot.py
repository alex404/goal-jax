"""Plot population code results."""

import matplotlib.pyplot as plt
import numpy as np

from ..shared import apply_style, colors, example_paths, get_pi_ticks, model_colors


def main():
    paths = example_paths(__file__)
    apply_style(paths)
    results = paths.load_analysis()

    # Load data
    grid = np.array(results["tuning_curve_grid"])
    tuning_curves = np.array(results["tuning_curves"])
    preferred = np.array(results["preferred_directions"])
    spikes = np.array(results["spike_counts"])
    true_stim = np.array(results["true_stimuli"])
    post_means = np.array(results["posterior_means"])
    post_kappas = np.array(results["posterior_concentrations"])
    mean_error = results["mean_circular_error"]

    n_neurons = len(preferred)

    # Sort neurons by preferred direction for consistent coloring
    neuron_order = np.argsort(preferred)
    neuron_colors = [model_colors[i % len(model_colors)] for i in range(n_neurons)]

    fig = plt.figure(figsize=(10, 8))

    # === Panel 1: Polar tuning curves ===
    ax_polar = fig.add_subplot(2, 2, 1, projection="polar")

    for idx in neuron_order:
        tc = tuning_curves[idx]
        pref = preferred[idx]
        color = neuron_colors[idx]
        ax_polar.plot(grid, tc, color=color, alpha=0.8, linewidth=1.5)
        ax_polar.scatter([pref], [tc.max()], color=color, s=40, zorder=5, edgecolors="white", linewidths=0.5)

    ax_polar.set_theta_zero_location("E")  # pyright: ignore[reportAttributeAccessIssue]
    ax_polar.set_rticks([])  # pyright: ignore[reportAttributeAccessIssue]
    # Position title above the plot to avoid overlap with angular labels
    ax_polar.set_title("Tuning Curves (Polar)", y=1.1)

    # === Panel 2: Cartesian tuning curves with regression fit ===
    ax_cart = fig.add_subplot(2, 2, 2)

    for idx in neuron_order:
        tc = tuning_curves[idx]
        color = neuron_colors[idx]
        ax_cart.plot(grid, tc, color=color, alpha=0.6, linewidth=1)

    pi_ticks, pi_labels = get_pi_ticks()
    ax_cart.set_xticks(pi_ticks)
    ax_cart.set_xticklabels(pi_labels)
    ax_cart.set_xlabel("Stimulus")
    ax_cart.set_ylabel("Firing Rate", color="gray")
    ax_cart.tick_params(axis="y", labelcolor="gray")
    ax_cart.set_xlim(0, 2 * np.pi)

    # Right axis: sum of rates and regression fit
    reg_grid = np.array(results["regression_grid"])
    reg_actual = np.array(results["regression_actual"])
    reg_fitted = np.array(results["regression_fitted"])

    ax_reg = ax_cart.twinx()
    ax_reg.plot(
        reg_grid, reg_actual,
        color=colors["ground_truth"],
        linewidth=2,
        label="Σ rates",
    )
    ax_reg.plot(
        reg_grid, reg_fitted,
        color=colors["fitted"],
        linewidth=2,
        linestyle="--",
        label="ρ fit",
    )
    ax_reg.set_ylabel("Sum of Rates")
    ax_reg.legend(loc="upper right", fontsize=8)
    ax_cart.set_title("Tuning Curves & Conjugation Fit")

    # Set y-axis limits with clearance
    tc_max = np.max(tuning_curves)
    ax_cart.set_ylim(0, tc_max * 1.1)
    reg_max = max(np.max(reg_actual), np.max(reg_fitted))
    ax_reg.set_ylim(0, reg_max * 1.1)

    # === Panel 3: Spike counts heatmap ===
    ax_spikes = fig.add_subplot(2, 2, 3)

    # Sort trials by stimulus, neurons by preferred direction
    trial_order = np.argsort(true_stim)
    spikes_sorted = spikes[trial_order][:, neuron_order]

    im = ax_spikes.imshow(
        spikes_sorted.T,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    # Set y-ticks to show neuron indices (sorted by preference)
    ax_spikes.set_yticks(range(n_neurons))
    ax_spikes.set_yticklabels([f"N{i+1}" for i in neuron_order])

    # Set x-ticks to show stimulus values
    n_trials = len(true_stim)
    tick_positions = [0, n_trials // 4, n_trials // 2, 3 * n_trials // 4, n_trials - 1]
    sorted_stim = true_stim[trial_order]
    tick_labels = [f"{sorted_stim[i]:.1f}" for i in tick_positions]
    ax_spikes.set_xticks(tick_positions)
    ax_spikes.set_xticklabels(tick_labels)

    ax_spikes.set_xlabel("Stimulus (rad)")
    ax_spikes.set_ylabel("Neuron (sorted by pref.)")
    ax_spikes.set_title("Spike Counts")

    cbar = plt.colorbar(im, ax=ax_spikes, shrink=0.8)
    cbar.set_label("Spikes")

    # === Panel 4: Stimulus inference ===
    ax_infer = fig.add_subplot(2, 2, 4)

    # Diagonal line (perfect inference)
    ax_infer.plot(
        [0, 2 * np.pi],
        [0, 2 * np.pi],
        color=colors["ground_truth"],
        linestyle="--",
        linewidth=1.5,
        label="Perfect",
        zorder=1,
    )

    # Size points by confidence (posterior concentration)
    min_size, max_size = 30, 150
    sizes = min_size + (max_size - min_size) * (post_kappas - post_kappas.min()) / (post_kappas.max() - post_kappas.min() + 1e-6)

    ax_infer.scatter(
        true_stim,
        post_means,
        c=colors["fitted"],
        s=sizes,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
        label="Estimate",
        zorder=2,
    )

    ax_infer.set_xticks(pi_ticks)
    ax_infer.set_xticklabels(pi_labels)
    ax_infer.set_yticks(pi_ticks)
    ax_infer.set_yticklabels(pi_labels)

    ax_infer.set_xlabel("True Stimulus")
    ax_infer.set_ylabel("Inferred Stimulus")
    ax_infer.set_title("Bayesian Inference")
    ax_infer.set_xlim(0, 2 * np.pi)
    ax_infer.set_ylim(0, 2 * np.pi)
    ax_infer.set_aspect("equal")
    ax_infer.legend(loc="upper left", fontsize=8)

    # Add error annotation
    ax_infer.text(
        0.95, 0.05,
        f"Mean error: {np.degrees(mean_error):.1f}°",
        transform=ax_infer.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle("Von Mises Population Code", fontsize=12, fontweight="bold")
    plt.tight_layout()
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
