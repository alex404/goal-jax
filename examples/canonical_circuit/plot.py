"""Plot the canonical circuit fit: training, recognition, tuning curves, and samples.

Top left: the test ELBO during training, against the test log-likelihood of Gaussians fit to the
training set. Top right: the mean of $q(z \\mid x)$ against the bump position, with one s.d. Bottom
left: the firing probability of each neuron given $z$ under the generative model. Bottom right: test
images sorted by bump position, and samples of the circuit sorted by $z$.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from ..shared import apply_style, colors, example_paths, figure_size, model_color
from .types import Results


def main() -> None:
    paths = example_paths(__file__)
    res: Results = paths.load_analysis()
    apply_style(paths)
    fig = plt.figure(figsize=figure_size("medium"))
    grid = GridSpec(2, 2, figure=fig)

    ax = fig.add_subplot(grid[0, 0])
    ax.plot(res["steps"], res["elbo"], color=colors["fitted"], label="Circuit ELBO")
    for i, (name, value) in enumerate(res["baselines"].items()):
        ax.axhline(value, color=model_color(i + 2), linestyle="--", label=name)
    ax.set_ylim(min(res["baselines"].values()) - 2, max(res["elbo"]) + 2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Test log-likelihood, ELBO")
    ax.set_title("Training")
    ax.legend()

    ax = fig.add_subplot(grid[0, 1])
    ax.errorbar(
        res["test_t"],
        res["q_mean"],
        yerr=res["q_sd"],
        fmt="o",
        markersize=2,
        color=colors["fitted"],
        alpha=0.6,
    )
    ax.set_xlabel("Bump position $t$")
    ax.set_ylabel(r"$q(z \mid x)$")
    ax.set_title(f"Recognition (Spearman {res['spearman']:.3f})")

    ax = fig.add_subplot(grid[1, 0])
    for i, rate in enumerate(res["tuning_rates"]):
        ax.plot(res["tuning_z"], rate, color=model_color(i))
    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$p(n_i = 1 \mid z)$")
    ax.set_title("Tuning curves")

    sub = grid[1, 1].subgridspec(1, 2)
    for j, (key, title) in enumerate(
        (("data_images", "Data, by $t$"), ("sample_images", "Samples, by $z$"))
    ):
        ax = fig.add_subplot(sub[0, j])
        ax.imshow(np.asarray(res[key]), aspect="auto", cmap="viridis")
        ax.set_xlabel("Pixel")
        ax.set_yticks([])
        ax.set_title(title)

    paths.save_plot(fig)


if __name__ == "__main__":
    main()
