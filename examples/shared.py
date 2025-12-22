"""Shared utilities for GOAL examples."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Type aliases
Bounds2D: TypeAlias = tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)

# Standard color schemes for consistent plots
colors = {
    "ground_truth": "#000000",  # black
    "fitted": "#E24A33",  # red
    "initial": "#348ABD",  # blue
    "secondary": "#988ED5",  # purple
    "tertiary": "#8EBA42",  # green
}

# Sequential colors for multiple models/components
model_colors = ["#348ABD", "#E24A33", "#988ED5", "#8EBA42", "#FBC15E", "#777777"]


@dataclass(frozen=True)
class ExamplePaths:
    """Manages paths for example outputs."""

    example_name: str
    results_dir: Path
    style_path: Path

    @property
    def analysis_path(self) -> Path:
        return self.results_dir / "analysis.json"

    @property
    def plot_path(self) -> Path:
        return self.results_dir / "plot.png"

    def save_analysis(self, results: Any) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.analysis_path, "w") as f:
            json.dump(results, f, indent=2)

    def load_analysis(self) -> Any:
        with open(self.analysis_path) as f:
            return json.load(f)

    def save_plot(self, fig: Figure) -> None:
        fig.savefig(self.plot_path, bbox_inches="tight")
        plt.close(fig)


def example_paths(module_path: str | Path) -> ExamplePaths:
    """Create ExamplePaths from a module's __file__."""
    module_path = Path(module_path)
    example_name = module_path.parent.name
    project_root = module_path.parents[2]
    results_dir = project_root / "results" / example_name
    return ExamplePaths(
        example_name=example_name,
        results_dir=results_dir,
        style_path=project_root / "examples" / "default.mplstyle",
    )


def initialize_jax(device: str = "cpu", disable_jit: bool = False) -> None:
    """Initialize JAX configuration."""
    jax.config.update("jax_platform_name", device)
    if disable_jit:
        jax.config.update("jax_disable_jit", True)


def apply_style(paths: ExamplePaths) -> None:
    """Apply the default matplotlib style for consistent plots."""
    plt.style.use(str(paths.style_path))


# Grid utilities


def create_grid(bounds: Bounds2D, n_points: int = 50) -> tuple[Array, Array]:
    """Create a 2D meshgrid within the given bounds."""
    x_min, x_max, y_min, y_max = bounds
    x = jnp.linspace(x_min, x_max, n_points)
    y = jnp.linspace(y_min, y_max, n_points)
    xs, ys = jnp.meshgrid(x, y)
    return (xs, ys)


def get_plot_bounds(sample: Array, margin: float = 0.2) -> Bounds2D:
    """Compute bounds from sample with margin."""
    min_vals = jnp.min(sample, axis=0)
    max_vals = jnp.max(sample, axis=0)
    range_vals = max_vals - min_vals
    x_min = float(min_vals[0] - margin * range_vals[0])
    x_max = float(max_vals[0] + margin * range_vals[0])
    y_min = float(min_vals[1] - margin * range_vals[1])
    y_max = float(max_vals[1] + margin * range_vals[1])
    return (x_min, x_max, y_min, y_max)


def get_normal_bounds(mean: Array, cov: Array, n_std: float = 3.0) -> Bounds2D:
    """Compute bounds from normal distribution parameters."""
    std_devs = jnp.sqrt(jnp.diag(cov))
    return (
        float(mean[0] - n_std * std_devs[0]),
        float(mean[0] + n_std * std_devs[0]),
        float(mean[1] - n_std * std_devs[1]),
        float(mean[1] + n_std * std_devs[1]),
    )


# Common plotting functions


def plot_density_contours(
    ax: Axes,
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    densities: list[NDArray[np.float64]],
    labels: list[str],
    sample: NDArray[np.float64] | None = None,
    plot_colors: list[str] | None = None,
) -> None:
    """Plot density contours with optional sample overlay."""
    if plot_colors is None:
        plot_colors = [colors["ground_truth"], colors["fitted"]]

    if sample is not None:
        ax.scatter(sample[:, 0], sample[:, 1], alpha=0.3, s=10, label="Samples")

    for density, label, color in zip(densities, labels, plot_colors):
        ax.contour(xs, ys, density, levels=6, colors=color, alpha=0.6)
        ax.plot([], [], color=color, label=label)

    min_val = min(float(xs.min()), float(ys.min()))
    max_val = max(float(xs.max()), float(ys.max()))
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal")
    ax.legend()


def plot_training_history(
    ax: Axes,
    histories: dict[str, list[float]],
    ylabel: str = "Log Likelihood",
) -> None:
    """Plot training histories for one or more models."""
    for (name, history), color in zip(histories.items(), model_colors):
        ax.plot(history, label=name, color=color)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
