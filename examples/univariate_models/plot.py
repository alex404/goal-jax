"""Test script for univariate auistributions in the exponential family."""

import json
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy import stats
from scipy.special import factorial

from ..shared import initialize_paths
from .types import (
    CategoricalResults,
    NormalResults,
    PoissonResults,
    UnivariateResults,
)


def plot_gaussian_results(ax: Axes, results: NormalResults) -> None:
    """Plot Normal results using numpy/matplotlib."""
    mu = results["mu"]
    sigma = results["sigma"]
    sample = results["sample"]
    true_densities = results["true_densities"]
    estimated_densities = results["estimated_densities"]

    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)

    # Convert everything to numpy for plotting
    np_sample: NDArray[np.float64] = np.array(sample)
    np_true_densities: NDArray[np.float64] = np.array(true_densities)
    np_est_densities: NDArray[np.float64] = np.array(estimated_densities)

    ax.hist(np_sample, bins=20, density=True, alpha=0.3, label="sample")
    ax.plot(x, np_true_densities, "b-", label="Implementation")
    ax.plot(x, np_est_densities, "g--", label="MLE Estimate")
    ax.plot(
        x,
        stats.norm.pdf(x, loc=mu, scale=sigma),
        "r--",
        label="Scipy",
    )

    ax.set_title("Gaussian Density")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()


def plot_categorical_results(
    ax: Axes,
    results: CategoricalResults,
) -> None:
    """Plot Categorical results using numpy/matplotlib."""

    probs = results["probs"]
    sample = results["sample"]
    true_probs = results["true_probs"]
    est_probs = results["estimated_probs"]

    categories = np.arange(len(probs))
    width = 0.25

    # Convert everything to numpy for plotting
    np_sample: NDArray[np.float64] = np.array(sample)
    np_true_probs: NDArray[np.float64] = np.array(true_probs)
    np_est_probs: NDArray[np.float64] = np.array(est_probs)
    np_probs: NDArray[np.float64] = np.array(probs)

    # Sample frequencies
    sample_freqs = np.bincount(np_sample, minlength=len(np_probs)) / len(np_sample)
    ax.bar(categories, sample_freqs, width=1.0, alpha=0.2, label="sample")

    # Plot bars
    ax.bar(categories - width, np_true_probs, width, alpha=0.6, label="Implementation")
    ax.bar(categories, np_est_probs, width, alpha=0.6, label="MLE Estimate")
    ax.bar(categories + width, np_probs, width, alpha=0.6, label="True Probabilities")

    ax.set_title("Categorical PMF")
    ax.set_xlabel("Category")
    ax.set_ylabel("Probability")
    ax.legend()


def plot_poisson_results(
    ax: Axes,
    results: PoissonResults,
) -> None:
    """Plot Poisson results using numpy/matplotlib."""

    rate = results["rate"]
    sample = results["sample"]
    true_pmf = results["true_pmf"]
    est_pmf = results["estimated_pmf"]

    max_k = 20
    k = np.arange(0, max_k + 1)

    # Convert everything to numpy for plotting
    np_sample: NDArray[np.float64] = np.array(sample)
    np_true_pmf: NDArray[np.float64] = np.array(true_pmf)
    np_est_pmf: NDArray[np.float64] = np.array(est_pmf)

    ax.hist(np_sample, bins=range(max_k + 2), density=True, alpha=0.3, label="sample")
    ax.plot(k, np_true_pmf, "b-", label="Implementation", alpha=0.8)
    ax.plot(k, np_est_pmf, "g--", label="MLE Estimate", alpha=0.8)
    ax.plot(
        k,
        rate**k * np.exp(-rate) / factorial(k),
        "r:",
        label="Theory",
        alpha=0.8,
    )

    ax.set_title(f"Poisson PMF (λ={rate})")
    ax.set_xlabel("k")
    ax.set_ylabel("Probability")
    ax.legend()


def main():
    """Run all distribution tests and plots."""
    # Load results
    paths = initialize_paths(__file__)

    with open(paths.analysis_path) as f:
        results = cast(UnivariateResults, json.load(f))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_gaussian_results(axes[0], results["gaussian"])

    plot_categorical_results(axes[1], results["categorical"])

    # Plot and save Poisson
    plot_poisson_results(axes[2], results["poisson"])
    fig.savefig(paths.results_dir / "univariate.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
