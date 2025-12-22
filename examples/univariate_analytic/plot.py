"""Plotting for univariate analytic distributions."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import factorial

from ..shared import apply_style, colors, example_paths, model_colors
from .types import UnivariateResults


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(UnivariateResults, paths.load_analysis())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Normal
    g = results["gaussian"]
    mu, sigma = g["mu"], g["sigma"]
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    sample = np.array(g["sample"])

    axes[0].hist(sample, bins=20, density=True, alpha=0.3, label="Sample")
    axes[0].plot(x, g["true_densities"], color=colors["ground_truth"], label="True")
    axes[0].plot(x, g["estimated_densities"], color=colors["fitted"], ls="--", label="MLE")
    axes[0].plot(x, stats.norm.pdf(x, mu, sigma), color=colors["tertiary"], ls=":", label="Scipy")
    axes[0].set_title("Normal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Categorical
    c = results["categorical"]
    probs = np.array(c["probs"])
    sample = np.array(c["sample"])
    categories = np.arange(len(probs))
    width = 0.25

    sample_freqs = np.bincount(sample, minlength=len(probs)) / len(sample)
    axes[1].bar(categories, sample_freqs, width=1.0, alpha=0.2, label="Sample")
    axes[1].bar(categories - width, c["true_probs"], width, color=model_colors[0], alpha=0.6, label="True")
    axes[1].bar(categories, c["estimated_probs"], width, color=model_colors[1], alpha=0.6, label="MLE")
    axes[1].bar(categories + width, probs, width, color=model_colors[2], alpha=0.6, label="Target")
    axes[1].set_title("Categorical")
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Probability")
    axes[1].legend()

    # Poisson
    p = results["poisson"]
    rate = p["rate"]
    k = np.array(p["eval_points"])
    sample = np.array(p["sample"])

    axes[2].hist(sample, bins=range(22), density=True, alpha=0.3, label="Sample")
    axes[2].plot(k, p["true_pmf"], color=colors["ground_truth"], label="True")
    axes[2].plot(k, p["estimated_pmf"], color=colors["fitted"], ls="--", label="MLE")
    axes[2].plot(k, rate**k * np.exp(-rate) / factorial(k), color=colors["tertiary"], ls=":", label="Theory")
    axes[2].set_title(f"Poisson (λ={rate})")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Probability")
    axes[2].legend()

    plt.tight_layout()
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
