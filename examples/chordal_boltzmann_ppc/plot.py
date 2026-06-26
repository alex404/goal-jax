"""Plots for the chordal Boltzmann PPC example.

A 2x3 grid: the top row covers conjugacy (the log-partition's affine fit, the
residual before/after training, and decoding) and the bottom row covers
data-learning (ELBO, conjugation quality, and captured correlations). Chordal and
diagonal population codes are compared throughout.
"""

from typing import cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from ..shared import apply_style, colors, example_paths, figure_size, model_color
from .types import Results

FAMILY_COLOR = {"chordal": model_color(0), "diagonal": model_color(1)}


def _plot_affine(ax: Axes, res: Results) -> None:
    z = np.array(res["z_grid"])
    f = res["families"]["chordal"]
    ax.plot(z, f["psi_curve"], color=colors["ground_truth"], linewidth=2,
            label=r"$\psi_G(\eta_N(z))$")
    ax.plot(z, f["affine_curve"], color=colors["fitted"], linewidth=1.5, linestyle="--",
            label=r"$\rho\cdot s_X(z)+\chi$")
    ax.set_xlabel("Latent $z_1$")
    ax.set_ylabel("Log-Partition")
    ax.set_title("Log-Partition Fit (chordal)")
    ax.legend()


def _plot_residual(ax: Axes, res: Results) -> None:
    z = np.array(res["z_grid"])
    for fam in ("chordal", "diagonal"):
        f = res["families"][fam]
        c = FAMILY_COLOR[fam]
        ax.plot(z, f["residual_before"], color=c, linewidth=1, alpha=0.35,
                label=f"{fam} (untrained)")
        ax.plot(z, f["residual_after"], color=c, linewidth=2, label=f"{fam} (trained)")
    ax.axhline(0.0, color=colors["ground_truth"], linewidth=0.6)
    ax.set_xlabel("Latent $z_1$")
    ax.set_ylabel("Residual $r(z)$")
    ax.set_title("Conjugation Residual")
    ax.legend()


def _plot_decode(ax: Axes, res: Results) -> None:
    fams = ["chordal", "diagonal"]
    xs = np.arange(len(fams))
    decode = [res["families"][f]["decode_rmse"] for f in fams]
    lesion = [res["families"][f]["lesion_rmse"] for f in fams]
    ax.bar(xs - 0.2, decode, 0.4, color=[FAMILY_COLOR[f] for f in fams], label="decode")
    ax.bar(xs + 0.2, lesion, 0.4, color="lightgray", label="lesion")
    ax.axhline(res["prior_std"], color=colors["ground_truth"], linestyle="--",
               linewidth=1, label="prior")
    ax.set_xticks(xs)
    ax.set_xticklabels(fams)
    ax.set_ylabel("Decoding RMSE")
    ax.set_title("Latent Decoding")
    ax.legend()


def _plot_elbo(ax: Axes, res: Results) -> None:
    steps = np.array(res["steps"])
    for fam in ("chordal", "diagonal"):
        f = res["families"][fam]
        c = FAMILY_COLOR[fam]
        ax.plot(steps, f["elbo_train"], color=c, linewidth=1.8, label=f"{fam} train")
        ax.plot(steps, f["elbo_test"], color=c, linewidth=1.2, linestyle="--",
                label=f"{fam} held-out")
    ax.axhline(res["ceiling"], color=colors["ground_truth"], linestyle="--",
               linewidth=1, label="ceiling")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean ELBO")
    ax.set_title("Training (ELBO)")
    ax.legend()


def _plot_conj(ax: Axes, res: Results) -> None:
    steps = np.array(res["steps"])
    for fam in ("chordal", "diagonal"):
        ax.plot(steps, res["families"][fam]["conj_r2"], color=FAMILY_COLOR[fam],
                linewidth=1.5, label=fam)
    ax.axvline(res["conj_warmup"], color=colors["ground_truth"], linestyle=":",
               linewidth=0.8, label="warmup end")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(r"Conjugation $R^2$")
    ax.set_title("Conjugation Quality")
    ax.legend()


def _plot_corr(ax: Axes, res: Results) -> None:
    data_corr = np.array(res["data_corr"])
    iu = np.triu_indices(data_corr.shape[0], 1)
    dvals = data_corr[iu]
    lim = float(max(0.05, np.abs(dvals).max())) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=colors["ground_truth"], linestyle="--",
            linewidth=0.8)
    for fam in ("chordal", "diagonal"):
        f = res["families"][fam]
        ax.scatter(dvals, np.array(f["model_corr"])[iu], color=FAMILY_COLOR[fam],
                   alpha=0.6, label=fam)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Data Correlation")
    ax.set_ylabel("Model Correlation")
    ax.set_title("Pairwise Correlations")
    ax.legend()


def main() -> None:
    paths = example_paths(__file__)
    apply_style(paths)
    res = cast(Results, paths.load_analysis())

    fig = plt.figure(figsize=figure_size("large"))
    gs = GridSpec(2, 3, figure=fig)

    _plot_affine(fig.add_subplot(gs[0, 0]), res)
    _plot_residual(fig.add_subplot(gs[0, 1]), res)
    _plot_decode(fig.add_subplot(gs[0, 2]), res)
    _plot_elbo(fig.add_subplot(gs[1, 0]), res)
    _plot_conj(fig.add_subplot(gs[1, 1]), res)
    _plot_corr(fig.add_subplot(gs[1, 2]), res)

    paths.save_plot(fig)


if __name__ == "__main__":
    main()
