from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt

examples_dir = Path(__file__).parent.parent
style_path = examples_dir / "default.mplstyle"
plt.style.use(str(style_path))


class BivariateResults(TypedDict):
    """Complete results for bivariate analysis."""

    sample: list[list[float]]  # List of [x,y] points
    plot_xs: list[list[float]]  # 2D grid of x coordinates
    plot_ys: list[list[float]]  # 2D grid of y coordinates
    scipy_densities: list[list[float]]
    ground_truth_densities: list[list[float]]
    positive_definite_densities: list[list[float]]
    diagonal_densities: list[list[float]]
    scale_densities: list[list[float]]


results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)
analysis_path = results_dir / "analysis.json"
