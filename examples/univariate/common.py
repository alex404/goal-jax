from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt

style_path = Path(__file__).parent.parent / "default.mplstyle"
plt.style.use(str(style_path))


# Type declarations for saved results
class NormalResults(TypedDict):
    mu: float
    sigma: float
    sample: list[float]
    true_densities: list[float]
    estimated_densities: list[float]
    eval_points: list[float]


class CategoricalResults(TypedDict):
    probs: list[float]
    sample: list[int]
    true_probs: list[float]
    estimated_probs: list[float]


class PoissonResults(TypedDict):
    rate: float
    sample: list[int]
    true_pmf: list[float]
    estimated_pmf: list[float]
    eval_points: list[int]


class UnivariateResults(TypedDict):
    gaussian: NormalResults
    categorical: CategoricalResults
    poisson: PoissonResults


results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)

analysis_file = "analysis.json"

analysis_path = results_dir / analysis_file
