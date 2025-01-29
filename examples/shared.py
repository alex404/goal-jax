"""Shared utilities for GOAL examples."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from matplotlib.figure import Figure

# Custom type aliases
Bounds2D: TypeAlias = tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)

### Initialization and Path Management ###


@dataclass(frozen=True)
class ExamplePaths:
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
    jax.config.update("jax_platform_name", device)
    if disable_jit:
        jax.config.update("jax_disable_jit", True)


### Grid Creation ###


def create_grid(bounds: Bounds2D, n_points: int = 50) -> tuple[Array, Array]:
    x_min, x_max, y_min, y_max = bounds
    x = jnp.linspace(x_min, x_max, n_points)
    y = jnp.linspace(y_min, y_max, n_points)
    xs, ys = jnp.meshgrid(x, y)
    return (xs, ys)


def get_plot_bounds(sample: Array, margin: float = 0.2) -> Bounds2D:
    min_vals = jnp.min(sample, axis=0)
    max_vals = jnp.max(sample, axis=0)
    range_vals = max_vals - min_vals
    x_min = float(min_vals[0] - margin * range_vals[0])
    x_max = float(max_vals[0] + margin * range_vals[0])
    y_min = float(min_vals[1] - margin * range_vals[1])
    y_max = float(max_vals[1] + margin * range_vals[1])
    return (x_min, x_max, y_min, y_max)


# In shared.py, add to the Grid Creation section:


def get_normal_bounds(
    mean: Array,
    cov: Array,
    n_std: float = 3.0,
) -> Bounds2D:
    std_devs = jnp.sqrt(jnp.diag(cov))
    return (
        float(mean[0] - n_std * std_devs[0]),  # x_min
        float(mean[0] + n_std * std_devs[0]),  # x_max
        float(mean[1] - n_std * std_devs[1]),  # y_min
        float(mean[1] + n_std * std_devs[1]),  # y_max
    )
