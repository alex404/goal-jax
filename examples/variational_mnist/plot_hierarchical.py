"""Plot the chordal-Boltzmann hierarchy comparison from the saved JSON.

Run after :mod:`.hierarchical_experiment`::

    uv run python -m examples.variational_mnist.plot_hierarchical
"""

import json

import matplotlib.pyplot as plt

from ..shared import example_paths

COLORS = {"diagonal": "#888888", "chain": "#1f77b4", "chordal": "#d62728"}


def main() -> None:
    results_dir = example_paths(__file__).results_dir
    path = results_dir / "hierarchical_experiment_results.json"
    data = json.loads(path.read_text())
    ceiling = data["ceiling"]
    results = data["results"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    ax = axes[0]
    for r in results:
        c = COLORS.get(r["kind"], "k")
        ax.plot(r["steps"], r["elbo_test_traj"], color=c, label=f"{r['kind']} (test)")
        ax.plot(r["steps"], r["elbo_train_traj"], color=c, ls="--", alpha=0.5)
    ax.axhline(ceiling, color="green", ls=":", label="ceiling (true log p)")
    ax.set_xlabel("step")
    ax.set_ylabel("ELBO")
    ax.set_title("Held-out ELBO (solid) vs train (dashed)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for r in results:
        ax.plot(r["steps"], r["gen_var_traj"], color=COLORS.get(r["kind"], "k"),
                label=r["kind"])
    ax.set_xlabel("step")
    ax.set_ylabel(r"Var$_p[r]$ (generative conjugation residual)")
    ax.set_title("Conjugation residual variance")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axes[2]
    kinds = [r["kind"] for r in results]
    xs = range(len(kinds))
    gaps = [ceiling - r["elbo_test"] for r in results]
    mses = [r["recon_mse"] for r in results]
    bars = ax.bar(xs, gaps, color=[COLORS.get(k, "k") for k in kinds], alpha=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(kinds)
    ax.set_ylabel("ELBO gap to ceiling (lower better)")
    ax.set_title("Final gap + reconstruction MSE")
    for b, m in zip(bars, mses):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"MSE\n{m:.3f}", ha="center", va="bottom", fontsize=8)

    out = results_dir / "hierarchical_experiment.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
