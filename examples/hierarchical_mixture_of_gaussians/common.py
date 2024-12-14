"""Common definitions for hierarchical mixture of Gaussians examples."""

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt

examples_dir = Path(__file__).parent.parent
style_path = examples_dir / "default.mplstyle"
plt.style.use(str(style_path))


class HMoGResults(TypedDict):
    """Complete results for hierarchical mixture of Gaussians analysis."""

    plot_range_x1: list[float]  # X1 coordinates for plotting
    plot_range_x2: list[float]  # X2 coordinates for plotting
    plot_range_y: list[float]  # Y coordinates for latent plotting
    observations: list[list[float]]  # List of [x1,x2] points
    log_likelihoods: list[float]  # Training log likelihoods
    observable_densities: list[list[list[float]]]  # [true, init, final] densities
    mixture_densities: list[list[float]]  # [true, init, final] latent densities


results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)
analysis_path = results_dir / "analysis.json"
