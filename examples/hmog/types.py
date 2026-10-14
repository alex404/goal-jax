"""Common definitions for hierarchical mixture of Gaussians examples."""

from typing import TypedDict


class VariationalHistory(TypedDict):
    """The variational fit of the diagonal HMoG, measured every ``steps``.

    ``rho`` is the learned $\\rho_Y$ and ``exact_rho`` that of the linear Gaussian model at the
    same likelihood; ``log_likelihood`` is exact. At the end of the fit, ``recognition_logits``
    and ``exact_logits`` are the logits of $k = 1$ at each observation under the recognition model
    and under the exact posterior of the harmonium.
    """

    steps: list[int]
    elbo: list[float]
    log_likelihood: list[float]
    rho: list[list[float]]
    exact_rho: list[list[float]]
    recognition_logits: list[float]
    exact_logits: list[float]


class HMoGResults(TypedDict):
    """Complete results for hierarchical mixture of Gaussians analysis."""

    plot_range_x1: list[float]  # X1 coordinates for plotting
    plot_range_x2: list[float]  # X2 coordinates for plotting
    plot_range_y: list[float]  # Y coordinates for latent plotting
    observations: list[list[float]]  # List of [x1,x2] points
    log_likelihoods: dict[str, list[float]]  # Training log likelihoods
    test_log_likelihoods: dict[str, float]  # Final log likelihoods on held-out data
    variational: VariationalHistory
    observable_densities: dict[str, list[list[float]]]  # [true, init, final] densities
    mixture_densities: dict[str, list[float]]  # [true, init, final] latent densities
