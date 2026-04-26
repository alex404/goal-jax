"""Plotting for Kalman filter example."""

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ..shared import apply_style, colors, example_paths, plot_training_history
from .types import KalmanFilterResults


def whiten(
    means: NDArray[np.float64], stds: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Self-whiten a trajectory using its own empirical statistics.

    Computes Sigma^{-1/2} (z_t - mu) where mu and Sigma are the
    empirical mean and covariance of the trajectory across timesteps.
    This removes the arbitrary linear transformation in the latent space,
    making learned trajectories visually comparable to truth.
    """
    mu = means.mean(axis=0)
    centered = means - mu
    cov = np.cov(centered, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    scales = 1.0 / np.sqrt(eigvals)
    W = (eigvecs * scales) @ eigvecs.T

    whitened_means = centered @ W.T
    whitened_stds = stds * np.sqrt(np.diag(W @ W.T))

    return whitened_means, whitened_stds


def plot_inference(
    ax: Any,
    time: NDArray[Any],
    true_latents: NDArray[np.float64],
    filt_means: NDArray[np.float64],
    filt_stds: NDArray[np.float64],
    smth_means: NDArray[np.float64],
    smth_stds: NDArray[np.float64],
    dim: int,
) -> None:
    """Plot filtering/smoothing inference on an axis."""
    ax.plot(
        time, true_latents[1:, dim],
        color=colors["ground_truth"], linewidth=2, label="True", zorder=4,
    )
    ax.plot(
        time, filt_means[:, dim],
        color=colors["initial"], linewidth=1.5, label="Filtered", zorder=3,
    )
    ax.fill_between(
        time,
        filt_means[:, dim] - 2 * filt_stds[:, dim],
        filt_means[:, dim] + 2 * filt_stds[:, dim],
        color=colors["initial"], alpha=0.15,
    )
    ax.plot(
        time, smth_means[:, dim],
        color=colors["fitted"], linewidth=1.5, label="Smoothed", zorder=3,
    )
    ax.fill_between(
        time,
        smth_means[:, dim] - 2 * smth_stds[:, dim],
        smth_means[:, dim] + 2 * smth_stds[:, dim],
        color=colors["fitted"], alpha=0.15,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Position")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_whitened_inference(
    ax: Any,
    time: NDArray[Any],
    true_latents: NDArray[np.float64],
    smth_means: NDArray[np.float64],
    smth_stds: NDArray[np.float64],
    title: str,
) -> None:
    """Plot self-whitened smoothed inference for both latent dimensions.

    Self-whitening removes the arbitrary linear transformation in the learned
    latent space, making the trajectories visually comparable to truth.
    """
    true_w, _ = whiten(true_latents[1:], np.zeros_like(true_latents[1:]))
    smth_w, stds_w = whiten(smth_means, smth_stds)

    lat_dim = smth_means.shape[1]
    dim_colors = [colors["initial"], colors["secondary"]]
    dim_labels = ["Dim 0", "Dim 1"]

    for d in range(lat_dim):
        ax.plot(
            time, true_w[:, d],
            color=dim_colors[d], linewidth=1, alpha=0.4, linestyle="--",
        )
        ax.plot(
            time, smth_w[:, d],
            color=dim_colors[d], linewidth=1.5, label=dim_labels[d],
        )
        ax.fill_between(
            time,
            smth_w[:, d] - 2 * stds_w[:, d],
            smth_w[:, d] + 2 * stds_w[:, d],
            color=dim_colors[d], alpha=0.12,
        )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Whitened value")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(KalmanFilterResults, paths.load_analysis())

    lat_dim = results["lat_dim"]
    n_steps = results["n_steps"]
    time = np.arange(n_steps)

    observations = np.array(results["observations"])
    true_latents = np.array(results["true_latents"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Top left: True process + observations ---
    ax = axes[0, 0]
    ax.scatter(
        time, observations[:, 0],
        color=colors["initial"], alpha=0.5, s=20, label="Observations", zorder=2,
    )
    ax.plot(
        time, true_latents[1:, 0],
        color=colors["ground_truth"], linewidth=2, label="True position", zorder=3,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Position")
    ax.set_title("True Process and Observations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Top center: Learning curves ---
    ax = axes[0, 1]
    plot_training_history(ax, results["log_likelihoods"])
    ax.axhline(
        y=results["true_log_lik"],
        color=colors["ground_truth"], linestyle="--", label="True params",
    )
    ax.axhline(
        y=results["initial_log_lik"],
        color=colors["initial"], linestyle=":", alpha=0.5, label="Initial",
    )
    ax.set_title("Learning Curves")
    ax.legend()

    # --- Top right: Whitened oracle inference ---
    plot_whitened_inference(
        axes[0, 2], time, true_latents,
        np.array(results["smoothed_means"]),
        np.array(results["smoothed_stds"]),
        "Whitened Oracle Inference",
    )

    # --- Bottom row: Whitened learned inference ---
    methods = list(results["learned_smoothed_means"].keys())
    panel_axes = [axes[1, 0], axes[1, 1]]

    for ax, method in zip(panel_axes, methods):
        plot_whitened_inference(
            ax, time, true_latents,
            np.array(results["learned_smoothed_means"][method]),
            np.array(results["learned_smoothed_stds"][method]),
            f"Whitened {method}-Learned Inference",
        )

    # Hide unused panels
    for i in range(len(methods), 2):
        panel_axes[i].set_visible(False)
    axes[1, 2].set_visible(False)

    fig.suptitle(
        f"Kalman Filter: Damped Oscillator (obs_dim={results['obs_dim']}, lat_dim={lat_dim})",
        fontsize=12,
    )

    plt.tight_layout()
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
