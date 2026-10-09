"""Plots for the canonical circuit example.

Panels other than the frontier show the runs of the first seed. Top row: training histories and the frontier between fit and conjugation. Middle row: model samples
without and with the strongest penalty, and the exact posterior mean of $z$ against the curve
coordinate. Third row: tuning curves and priors over $z$ for the strongest penalty, and the
recognition Gaussian against the mean and s.d. of the exact posterior. Bottom row: the mismatch between the
harmonium and the variational model, the variational gap, and the final bounds against the exact
log-likelihood. ``--experiment`` selects the results subdirectory.
"""

import argparse
from dataclasses import replace

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from ..shared import (
    apply_style,
    colors,
    example_paths,
    scatter_points,
    scatter_samples,
)
from .types import Results, RunResult


def _color(index: int, count: int) -> tuple[float, float, float, float]:
    """Color of the run with the given index among runs ordered by penalty strength."""
    return plt.get_cmap("viridis")(index / max(count - 1, 1))


def _label(run: RunResult) -> str:
    return rf"$\lambda = {run['lam']:g}$"


def _plot_history(ax: Axes, res: Results, key: str, ylabel: str, log: bool) -> None:
    for i, run in enumerate(res["runs"]):
        hist = run[key]
        ax.plot(
            res["steps"][: len(hist)],
            hist,
            color=_color(i, len(res["runs"])),
            label=_label(run),
        )
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")


def _plot_frontier(ax: Axes, res: Results, lams: list[float]) -> None:
    for run in res["runs"]:
        color = _color(lams.index(run["lam"]), len(lams))
        ax.scatter(run["final_var_q"], run["final_test_ll"], color=color, marker="o")
        ax.scatter(run["final_var_p"], run["final_test_ll"], color=color, marker="x")
    ax.set_xscale("log")
    ax.set_xlabel(r"Residual variance (o: $q$, x: $\tilde p_Z$)")
    ax.set_ylabel("Test log-likelihood")


def _plot_samples(ax: Axes, res: Results, run: RunResult) -> None:
    data = np.array(res["train_x"])
    samples = np.array(run["samples"])
    title = (
        f"{_label(run)}, off-curve {run['off_curve']:.2f} (data {res['data_off_curve']:.2f}), "
        f"means {run.get('off_curve_means', float('nan')):.2f}"
    )
    ax.set_title(title)
    if data.shape[1] == 2:
        scatter_points(
            ax, data[:, 0], data[:, 1], color=colors["ground_truth"], label="Data"
        )
        scatter_samples(
            ax, samples[:, 0], samples[:, 1], color=colors["fitted"], label="Model"
        )
        ax.set_aspect("equal")
        ax.legend()
    else:
        shown = samples[:40]
        shown = shown[np.argsort(np.argmax(shown, axis=1))]
        ax.imshow(shown, aspect="auto", cmap="gray_r")
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Model sample")


def _plot_posterior_vs_t(ax: Axes, res: Results) -> None:
    t = np.array(res["test_t"])
    for i, run in enumerate(res["runs"]):
        ax.scatter(
            t,
            run["post_mean_exact"],
            s=4,
            color=_color(i, len(res["runs"])),
            label=_label(run),
        )
    ax.set_xlabel("Curve coordinate $t$")
    ax.set_ylabel(r"$\mathrm{E}[z \mid x]$")
    ax.legend()


def _plot_tuning(ax: Axes, run: RunResult) -> None:
    z = np.array(run["tuning_z"])
    for logits in run["tuning_logits"]:
        ax.plot(
            z,
            1 / (1 + np.exp(-np.array(logits))),
            color=colors["secondary"],
            linewidth=1,
        )
    ax.set_xlabel("$z$")
    ax.set_ylabel("Firing probability")
    ax.set_title(_label(run))
    twin = ax.twinx()
    twin.plot(z, run["prior_density"], color=colors["ground_truth"], label=r"$p_Z$")
    twin.plot(
        z,
        run["prior_gaussian"],
        color=colors["fitted"],
        linestyle="--",
        label=r"$\tilde p_Z$",
    )
    twin.set_ylabel("Density")
    twin.legend()


def _plot_recognition(ax: Axes, res: Results, key: str, label: str) -> None:
    for i, run in enumerate(res["runs"]):
        ax.scatter(
            run[f"{key}_exact"],
            run[f"{key}_q"],
            s=4,
            color=_color(i, len(res["runs"])),
            label=_label(run),  # pyright: ignore[reportArgumentType]
        )
    lims = ax.get_xlim()
    ax.plot(lims, lims, color=colors["ground_truth"], linewidth=1)
    ax.set_xlabel(f"Exact posterior {label}")
    ax.set_ylabel(f"Recognition {label}")


def _plot_gap(ax: Axes, res: Results, upper: str, lower: str, ylabel: str) -> None:
    for i, run in enumerate(res["runs"]):
        gap = np.array(run[upper]) - np.array(run[lower])  # pyright: ignore[reportArgumentType]
        ax.plot(
            res["steps"][: len(gap)],
            gap,
            color=_color(i, len(res["runs"])),
            label=_label(run),
        )
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_yscale("symlog", linthresh=1e-3)


def _plot_bounds(ax: Axes, res: Results, lams: list[float]) -> None:
    for run in res["runs"]:
        color = _color(lams.index(run["lam"]), len(lams))
        ax.scatter(run["final_test_ll"], run["final_elbo"], color=color, marker="o")
        ax.scatter(
            run["final_test_ll"], run["final_log_tilde"], color=color, marker="x"
        )
    lims = ax.get_xlim()
    ax.plot(lims, lims, color=colors["ground_truth"], linewidth=1)
    ax.set_xlabel(r"Test $\log p_X$")
    ax.set_ylabel(r"Test ELBO (o), $\log \tilde p_X$ (x)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    experiment = parser.parse_args().experiment
    paths = example_paths(__file__)
    paths = replace(paths, results_dir=paths.results_dir / experiment)
    apply_style(paths)
    res: Results = paths.load_analysis()
    runs = [run for run in res["runs"] if run["seed"] == 0]
    lams = sorted({run["lam"] for run in res["runs"]})
    shown: Results = {**res, "runs": runs}

    fig = plt.figure(figsize=(12, 16))
    gs = GridSpec(4, 3, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    _plot_history(ax, shown, "test_ll", "Test log-likelihood", log=False)
    ax.legend()
    _plot_history(
        fig.add_subplot(gs[0, 1]), shown, "var_q", r"Mean $\mathrm{Var}_q[r]$", log=True
    )
    _plot_frontier(fig.add_subplot(gs[0, 2]), res, lams)
    _plot_samples(fig.add_subplot(gs[1, 0]), res, runs[0])
    _plot_samples(fig.add_subplot(gs[1, 1]), res, runs[-1])
    _plot_posterior_vs_t(fig.add_subplot(gs[1, 2]), shown)
    _plot_tuning(fig.add_subplot(gs[2, 0]), runs[-1])
    _plot_recognition(fig.add_subplot(gs[2, 1]), shown, "post_mean", "mean")
    _plot_recognition(fig.add_subplot(gs[2, 2]), shown, "post_sd", "s.d.")
    _plot_gap(
        fig.add_subplot(gs[3, 0]),
        shown,
        "test_ll",
        "log_tilde",
        r"$\log p_X - \log \tilde p_X$",
    )
    _plot_gap(
        fig.add_subplot(gs[3, 1]),
        shown,
        "log_tilde",
        "elbo",
        r"$\log \tilde p_X$ - ELBO",
    )
    _plot_bounds(fig.add_subplot(gs[3, 2]), res, lams)
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
