"""Plot clustering trajectories for the mixture-top hierarchy.

Run after :mod:`.hierarchical_mixture_experiment`::

    uv run python -m examples.variational_mnist.plot_hierarchical_mixture
"""

import json

import matplotlib.pyplot as plt

from ..shared import example_paths

COLORS = {"diagonal": "#888888", "chain": "#1f77b4", "chordal": "#d62728"}


def main() -> None:
    results_dir = example_paths(__file__).results_dir
    data = json.loads((results_dir / "hierarchical_mixture_results.json").read_text())
    results = data["results"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    ax = axes[0]
    for r in results:
        ax.plot(
            r["steps"],
            r["elbo_traj"],
            color=COLORS.get(r["kind"], "k"),
            label=r["kind"],
        )
    ax.set_xlabel("step")
    ax.set_ylabel("held-out ELBO")
    ax.set_title("ELBO")
    ax.legend(fontsize=8)

    ax = axes[1]
    for r in results:
        ax.plot(
            r["steps"], r["nmi_traj"], color=COLORS.get(r["kind"], "k"), label=r["kind"]
        )
    ax.set_xlabel("step")
    ax.set_ylabel("NMI (clusters vs true modes)")
    ax.set_ylim(0, 1)
    ax.set_title("Cluster recovery: NMI")
    ax.legend(fontsize=8)

    ax = axes[2]
    kinds = [r["kind"] for r in results]
    xs = range(len(kinds))
    bars = ax.bar(
        xs,
        [r["purity"] for r in results],
        color=[COLORS.get(k, "k") for k in kinds],
        alpha=0.85,
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels(kinds)
    ax.set_ylim(0, 1)
    ax.set_ylabel("purity")
    ax.set_title("Final cluster purity")
    for b, r in zip(bars, results):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"NMI\n{r['nmi']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    out = results_dir / "hierarchical_mixture.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
