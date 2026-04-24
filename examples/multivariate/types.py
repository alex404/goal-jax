from typing import TypedDict


class DirichletPanel(TypedDict):
    """Dirichlet panel: 100x100 grid on the 2D simplex projection."""

    samples: list[list[float]]  # (n_samples, 3) simplex points
    xrange: list[float]  # length 100
    yrange: list[float]  # length 100
    true_density: list[list[float]]  # 100x100
    learned_density: list[list[float]]  # 100x100
    log_likelihood: list[float]  # length n_epochs


class NormalPanel(TypedDict):
    """Multivariate Normal panel: analytic fit via source and natural parameterizations."""

    samples: list[list[float]]  # (n_samples, 2)
    xrange: list[float]
    yrange: list[float]
    true_density: list[list[float]]
    source_fit_density: list[list[float]]
    natural_fit_density: list[list[float]]


class MultivariateResults(TypedDict):
    """Analysis output for the multivariate example."""

    dirichlet: DirichletPanel
    normal: NormalPanel
