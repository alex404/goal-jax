"""Plot the canonical circuit: brute-force MLE against the penalized ELBO, with and without $z$, over the graphs of couplings.

Top row: final test $\\log p(x)$, the mass of the harmonium off the curve, and the fraction of the variance
of $z$ explained by $x$, for every condition, with one point per seed. Middle
row, for one graph and seed: the training curves ($\\log p(x)$, $\\log \\tilde p(x)$ and the ELBO of the
ELBO fit with $z$, and $\\log p(x)$ of the others), $\\mathbb E[z \\mid x]$ against the bump position (with :func:`position_r2`), and
the tuning curves with $\\tilde p(z)$. Bottom row: data and the means $\\mathbb E[x \\mid n]$ of states
sampled from each harmonium, sorted by the position of their maximum.
"""

import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from ..shared import apply_style, example_paths, model_color
from .types import Results, RunResult

COUPLINGS = ("independent", "chain", "full")


def variant(run: RunResult) -> str:
    z = "with $z$" if run["use_z"] else "without $z$"
    fit = "MLE" if run["fit"] == "exact" else rf"ELBO $\lambda = {run['lam']:g}$"
    return f"{fit}, {z}"


def variants(res: Results) -> list[str]:
    """Variants in display order: with $z$ first, MLE before ELBO."""
    found = {variant(r): (not r["use_z"], r["fit"] != "exact", r["lam"]) for r in res["runs"]}
    return sorted(found, key=lambda v: found[v])


def position_r2(post_mean: list[float], test_t: list[float], k: int = 10) -> float:
    """Fraction of the variance of the bump position explained by its $k$ nearest neighbours in $\\mathbb E[z \\mid x]$, leaving each point out.

    Unlike a rank correlation, this does not penalize a code that is not monotone in $t$, only one that
    does not determine $t$.
    """
    m, t = np.asarray(post_mean), np.asarray(test_t)
    dist = np.abs(m[:, None] - m[None, :])
    np.fill_diagonal(dist, np.inf)
    nearest = np.argsort(dist, axis=1)[:, :k]
    pred = t[nearest].mean(axis=1)
    return float(1 - np.sum((t - pred) ** 2) / np.sum((t - t.mean()) ** 2))


def plot_final(ax: Axes, res: Results, value: str, ylabel: str, title: str) -> None:
    names = variants(res)
    width = 0.8 / max(len(names), 1)
    for j, name in enumerate(names):
        for i, couplings in enumerate(COUPLINGS):
            runs = [r for r in res["runs"] if variant(r) == name and r["couplings"] == couplings]
            vals = np.array([float(r[value]) for r in runs])  # pyright: ignore[reportGeneralTypeIssues]
            x = i - 0.4 + width * (j + 0.5)
            ax.scatter(np.full(len(vals), x), vals, color=model_color(j), s=12,
                       label=name if i == 0 else None)
            if len(vals) > 0:
                ax.hlines(np.nanmean(vals), x - width / 2, x + width / 2, color=model_color(j))
    ax.set_xticks(range(len(COUPLINGS)))
    ax.set_xticklabels([c.capitalize() for c in COUPLINGS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def find(res: Results, couplings: str, use_z: bool, fit: str, lam: float, seed: int) -> RunResult | None:
    for r in res["runs"]:
        if (
            r["couplings"] == couplings
            and r["use_z"] == use_z
            and r["fit"] == fit
            and r["seed"] == seed
            and (fit == "exact" or r["lam"] == lam)
        ):
            return r
    return None


def image(ax: Axes, xs: np.ndarray, title: str) -> None:
    order = np.argsort(np.argmax(xs, axis=1))
    ax.imshow(xs[order].T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1.2)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Pixel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--couplings", default="chain", help="graph shown in the lower rows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lam", type=float, default=None, help="ELBO run shown (default: smallest)")
    args = parser.parse_args()
    paths = example_paths(__file__)
    apply_style(paths)
    res: Results = paths.load_analysis()
    names = variants(res)
    lams = sorted({r["lam"] for r in res["runs"] if r["fit"] == "elbo"})
    lam = args.lam if args.lam is not None else (lams[0] if lams else 0.0)

    fig = plt.figure(figsize=(18, 13), constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig)

    ax = fig.add_subplot(gs[0, 0:2])
    plot_final(ax, res, "final_test_ll", r"Test $\log p(x)$ (nats)", "Fit")
    ax.legend(fontsize="small", ncol=2)
    plot_final(fig.add_subplot(gs[0, 2]), res, "off_curve", "Mass off the curve", "Samples")
    plot_final(fig.add_subplot(gs[0, 3]), res, "z_usage", r"Variance of $z$ explained by $x$", "Use of $z$")

    shown = {
        "ELBO, with $z$": find(res, args.couplings, True, "elbo", lam, args.seed),
        "MLE, with $z$": find(res, args.couplings, True, "exact", lam, args.seed),
        "MLE, without $z$": find(res, args.couplings, False, "exact", lam, args.seed),
    }

    ax = fig.add_subplot(gs[1, 0:2])
    for name, run in shown.items():
        if run is None:
            continue
        color = model_color(names.index(variant(run)))
        h = run["history"]
        ax.plot(run["steps"][1:], h["test_ll"][1:], color=color, label=rf"{name}: $\log p(x)$")
        if name == "ELBO, with $z$":
            ax.plot(run["steps"][1:], h["log_tilde"][1:], color=color, ls="--", label=r"$\log \tilde p(x)$")
            ax.plot(run["steps"][1:], h["elbo"][1:], color=color, ls=":", label="ELBO")
    finals = [r["final_test_ll"] for r in shown.values() if r is not None]
    if finals:
        ax.set_ylim(min(finals) - 3, max(finals) + 0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel(r"Test $\log p(x)$ (nats)")
    ax.set_title(f"Training ({args.couplings.capitalize()}, Seed {args.seed})")
    ax.legend(fontsize="small")

    ax = fig.add_subplot(gs[1, 2])
    for name in ("ELBO, with $z$", "MLE, with $z$"):
        run = shown[name]
        if run is not None:
            r2 = position_r2(run["post_mean"], res["test_t"])
            ax.scatter(res["test_t"], run["post_mean"], s=3, label=rf"{name} ($R^2$ of $t$: {r2:.2f})",
                       color=model_color(names.index(variant(run))))
    ax.set_xlabel("Bump Position $t$")
    ax.set_ylabel(r"$E[z \mid x]$")
    ax.set_title("Posterior Mean")
    ax.legend(fontsize="small")

    ax = fig.add_subplot(gs[1, 3])
    run = shown["ELBO, with $z$"]
    if run is not None:
        zs = np.array(run["tuning_z"])
        rates = np.array(run["tuning_rates"])
        order = np.argsort(zs[np.argmax(rates, axis=1)])
        for j, i in enumerate(order):
            ax.plot(zs, rates[i], color=plt.get_cmap("viridis")(j / (len(order) - 1)))
        ax2 = ax.twinx()
        ax2.plot(zs, run["prior_density"], color="k", ls="--")
        ax2.set_yticks([])
    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$p(n_i = 1 \mid z)$")
    ax.set_title(r"Tuning Curves and $\tilde p(z)$")

    image(fig.add_subplot(gs[2, 0]), np.array(res["data_x"]), "Data")
    for col, name in enumerate(("ELBO, with $z$", "MLE, with $z$", "MLE, without $z$"), start=1):
        run = shown[name]
        ax = fig.add_subplot(gs[2, col])
        if run is None:
            ax.set_axis_off()
            continue
        image(ax, np.array(run["sample_means"]), f"{name} (Off {run['off_curve']:.2f})")

    paths.save_plot(fig)


if __name__ == "__main__":
    main()
