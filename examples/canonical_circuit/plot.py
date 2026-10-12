"""Plot the canonical circuit fits: training curves, the recognition model's $z$ against the bump position, and the tuning curves.

Top left: $\\log \\tilde p(x)$ (solid) and the ELBO (dashed) on the test set during training, per fit. Top
right: the mean of $q(z \\mid x)$ against the bump position, for the fits with $z$. Bottom: the firing
probabilities of the neurons given $z$ under the generative model of each fit with $z$.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from ..shared import apply_style, example_paths, figure_size, model_color
from .types import Results


def main() -> None:
    paths = example_paths(__file__)
    res: Results = paths.load_analysis()
    apply_style(paths)
    fig = plt.figure(figsize=figure_size("medium"))
    grid = GridSpec(2, 2, figure=fig)

    ax = fig.add_subplot(grid[0, 0])
    for i, fit in enumerate(res["fits"]):
        steps = fit["history"]["step"]
        ax.plot(
            steps, fit["history"]["log_tilde"], color=model_color(i), label=fit["name"]
        )
        ax.plot(steps, fit["history"]["elbo"], color=model_color(i), linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\log \tilde p(x)$, ELBO")
    ax.set_title("Training")
    ax.legend()

    ax = fig.add_subplot(grid[0, 1])
    t = np.asarray(res["test_t"])
    for i, fit in enumerate(res["fits"]):
        if fit["z_usage"] > 0:
            ax.scatter(t, fit["q_mean"], s=6, color=model_color(i), label=fit["name"])
    ax.set_xlabel("Bump position $t$")
    ax.set_ylabel(r"$\mathrm{E}_q[z \mid x]$")
    ax.set_title("Recognition")
    ax.legend()

    with_z = [(i, fit) for i, fit in enumerate(res["fits"]) if fit["z_usage"] > 0]
    for j, (i, fit) in enumerate(with_z[:2]):
        ax = fig.add_subplot(grid[1, j])
        rates = np.asarray(fit["tuning_rates"])
        for rate in rates:
            ax.plot(fit["tuning_z"], rate, color=model_color(i))
        ax.set_xlabel("$z$")
        ax.set_ylabel(r"$p(n_i = 1 \mid z)$")
        ax.set_title(f"Tuning, {fit['name'].title()}")

    paths.save_plot(fig)


if __name__ == "__main__":
    main()
