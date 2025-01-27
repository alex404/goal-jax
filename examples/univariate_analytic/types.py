from typing import TypedDict


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
