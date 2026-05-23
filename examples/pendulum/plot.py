"""Plotting for the pendulum variational filter example.

Six-panel figure:

- TL: ELBO training curves for each mode
- TM: Conjugation $R^2$ history
- TR: Tuning-curve preferred (angle, velocity) recovery
- BL: True $\\theta(t)$ vs filtered $\\hat\\theta(t)$
- BM: True $\\dot\\theta(t)$ vs filtered $\\hat{\\dot\\theta}(t)$
- BR: $\\|\\rho\\|$ history

Usage::

    uv run python -m examples.pendulum.plot
"""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from ..shared import apply_style, colors, example_paths, figure_size, model_color
from .types import Results, TuningParams


def _angles_array(t: TuningParams) -> np.ndarray:
    return np.asarray(t["preferred_angles"])


def _velocities_array(t: TuningParams) -> np.ndarray:
    return np.asarray(t["preferred_velocities"])


def _unwrap_filtered(true: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """Resolve $2\\pi$ ambiguity by shifting filtered angles by multiples of $2\\pi$ to track the true trajectory."""
    # Both are wrapped to (-pi, pi]. Choose offsets per-step to minimize wrap distance.
    diff = filt - true
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return true + diff


def main() -> None:
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(Results, paths.load_analysis())

    fig = plt.figure(figsize=figure_size("large"))
    gs = GridSpec(2, 3, figure=fig)

    modes = list(results["models"].keys())

    # ---- TL: ELBO curves ----
    ax_elbo = fig.add_subplot(gs[0, 0])
    for i, mode in enumerate(modes):
        history = results["models"][mode]["history"]
        elbos = np.asarray(history["elbos"])
        ax_elbo.plot(elbos, label=mode, color=model_color(i), lw=1.2)
    ax_elbo.set_xlabel("Training step")
    ax_elbo.set_ylabel("ELBO")
    ax_elbo.set_title("Training ELBO")
    ax_elbo.legend()

    # ---- TM: R^2 history ----
    ax_r2 = fig.add_subplot(gs[0, 1])
    for i, mode in enumerate(modes):
        history = results["models"][mode]["history"]
        r2 = np.asarray(history["r_squared"])
        ax_r2.plot(r2, label=mode, color=model_color(i), marker="o", markersize=3)
    ax_r2.set_xlabel("Logging chunk")
    ax_r2.set_ylabel(r"$R^2$ of conjugation")
    ax_r2.set_title("Conjugation quality")
    ax_r2.axhline(1.0, color="gray", linestyle="--", lw=0.6)
    ax_r2.legend()

    # ---- TR: Tuning curve preferred (angle, velocity) recovery ----
    ax_tune = fig.add_subplot(gs[0, 2])
    gt_tuning = results["ground_truth_tuning"]
    gt_a = _angles_array(gt_tuning)
    gt_v = _velocities_array(gt_tuning)
    ax_tune.scatter(
        gt_a,
        gt_v,
        c=colors["ground_truth"],
        marker="o",
        label="ground truth",
        s=40,
        zorder=2,
    )
    for i, mode in enumerate(modes):
        learned = results["models"][mode]["learned_tuning"]
        la = _angles_array(learned)
        lv = _velocities_array(learned)
        ax_tune.scatter(
            la,
            lv,
            color=model_color(i),
            marker="x",
            s=40,
            label=f"learned ({mode})",
            zorder=3,
        )
    ax_tune.set_xlabel(r"preferred $\theta$")
    ax_tune.set_ylabel(r"preferred $\dot{\theta}$")
    ax_tune.set_title("Tuning recovery")
    ax_tune.legend(fontsize=8)

    # ---- BL: true vs filtered angle ----
    ax_theta = fig.add_subplot(gs[1, 0])
    true_angles = np.asarray(results["true_angles"])
    t_axis = np.arange(len(true_angles))
    ax_theta.plot(
        t_axis,
        true_angles,
        color=colors["ground_truth"],
        lw=1.5,
        label="true",
    )
    for i, mode in enumerate(modes):
        filt = np.asarray(results["models"][mode]["filtered_angles"])
        filt_unwrapped = _unwrap_filtered(true_angles, filt)
        ax_theta.plot(
            t_axis,
            filt_unwrapped,
            color=model_color(i),
            lw=1.0,
            alpha=0.85,
            label=mode,
        )
    ax_theta.set_xlabel("step")
    ax_theta.set_ylabel(r"$\theta$")
    ax_theta.set_title("Angle: true vs filtered")
    ax_theta.legend()

    # ---- BM: true vs filtered velocity ----
    ax_v = fig.add_subplot(gs[1, 1])
    true_v = np.asarray(results["true_velocities"])
    ax_v.plot(t_axis, true_v, color=colors["ground_truth"], lw=1.5, label="true")
    for i, mode in enumerate(modes):
        filt = np.asarray(results["models"][mode]["filtered_velocities"])
        ax_v.plot(
            t_axis,
            filt,
            color=model_color(i),
            lw=1.0,
            alpha=0.85,
            label=mode,
        )
    ax_v.set_xlabel("step")
    ax_v.set_ylabel(r"$\dot{\theta}$")
    ax_v.set_title("Velocity: true vs filtered")
    ax_v.legend()

    # ---- BR: ||rho|| history ----
    ax_rho = fig.add_subplot(gs[1, 2])
    for i, mode in enumerate(modes):
        rho_norms = np.asarray(results["models"][mode]["history"]["rho_norms"])
        ax_rho.plot(rho_norms, label=mode, color=model_color(i), marker="o", markersize=3)
    ax_rho.set_xlabel("Logging chunk")
    ax_rho.set_ylabel(r"$\|\rho\|$")
    ax_rho.set_title(r"Conjugation parameter norm")
    ax_rho.legend()

    paths.save_plot(fig)
    print(f"Saved plot to {paths.plot_path}")


if __name__ == "__main__":
    main()
