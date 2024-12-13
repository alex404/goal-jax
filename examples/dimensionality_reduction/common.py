# common.py
"""Common definitions for LGM model examples."""

from pathlib import Path
from typing import TypedDict


class LGMResults(TypedDict):
    """Complete results for LGM analysis."""

    sample: list[list[float]]  # List of [x,y] points
    plot_bounds: tuple[float, float, float, float]  # Bounds for plotting
    training_lls: dict[str, list[float]]  # Log likelihoods during training
    grid_points: dict[str, list[list[float]]]  # Points where densities are evaluated
    observable_densities: dict[str, list[float]]  # Model densities in data space
    latent_densities: dict[str, list[float]]  # Model densities in latent space


# Set up paths similar to mixture example
results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)
analysis_path = results_dir / "analysis.json"

# run.py
