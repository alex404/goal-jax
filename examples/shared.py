"""Shared utilities for GOAL examples."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array

# Custom type aliases
Bounds2D: TypeAlias = tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)

### Initialization and Path Management ###


@dataclass(frozen=True)
class ExamplePaths:
    example_dir: Path
    results_dir: Path
    analysis_path: Path
    style_path: Path
    cache_dir: Path

    @classmethod
    def from_module(
        cls, module_path: str | Path, experiment: str | None
    ) -> "ExamplePaths":
        module_path = Path(module_path)
        example_dir = module_path.parent
        results_dir = example_dir / "results"
        style_path = example_dir.parent.parent / "default.mplstyle"
        project_root = example_dir.parent.parent
        cache_dir = project_root / ".cache"

        if experiment is not None:
            analysis_path = results_dir / f"{experiment}.json"
        else:
            analysis_path = results_dir / "analysis.json"
        # Ensure results directory exists
        results_dir.mkdir(exist_ok=True)
        cache_dir.mkdir(exist_ok=True)

        return cls(
            example_dir=example_dir,
            results_dir=results_dir,
            analysis_path=analysis_path,
            style_path=style_path,
            cache_dir=cache_dir,
        )

    def setup_plotting(self) -> None:
        plt.style.use(str(self.style_path))


def initialize_paths(
    module_path: str | Path, experiment: str | None = None
) -> ExamplePaths:
    paths = ExamplePaths.from_module(module_path, experiment)
    paths.setup_plotting()
    return paths


def initialize_jax(device: str = "cpu", disable_jit: bool = False) -> None:
    jax.config.update("jax_platform_name", device)
    if disable_jit:
        jax.config.update("jax_disable_jit", True)


def save_results(results: Any, paths: ExamplePaths) -> None:
    with open(paths.analysis_path, "w") as f:
        json.dump(results, f, indent=2)


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
