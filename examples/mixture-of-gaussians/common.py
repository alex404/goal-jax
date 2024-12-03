"""Common definitions for mixture model examples."""

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt

examples_dir = Path(__file__).parent.parent
style_path = examples_dir / "default.mplstyle"
plt.style.use(str(style_path))


class MixtureResults(TypedDict):
    """Complete results for mixture model analysis."""

    sample: list[list[float]]  # List of [x,y] points
    plot_xs: list[list[float]]  # 2D grid of x coordinates
    plot_ys: list[list[float]]  # 2D grid of y coordinates
    ground_truth_densities: list[list[float]]  # Ground truth density on grid
    positive_definite_densities: list[list[float]]  # PD model density on grid
    diagonal_densities: list[list[float]]  # Diagonal model density on grid
    scale_densities: list[list[float]]  # Scale model density on grid
    ground_truth_ll: float  # Ground truth log likelihood
    positive_definite_lls: list[float]  # PD model log likelihoods during training
    diagonal_lls: list[float]  # Diagonal model log likelihoods during training
    scale_lls: list[float]  # Scale model log likelihoods during training


results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)
analysis_path = results_dir / "mixture_analysis.json"
