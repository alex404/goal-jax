"""JSON schema for the canonical circuit example, exchanged between ``run.py`` and ``plot.py``."""

from typing import TypedDict


class FitResult(TypedDict):
    """One fit, measured exactly on the test set.

    ``history`` holds the step, the mean $\\log \\tilde p(x)$ and the mean ELBO after each chunk, and
    ``log_tilde``/``elbo`` repeat them at the end. ``z_usage`` is the fraction of the variance of $z$ under
    the recognition model explained by $x$, ``spearman_t`` the rank correlation of its mean ``q_mean`` with
    the bump position, and ``tuning_rates`` the firing probability of each neuron under the generative
    model at ``tuning_z``.
    """

    name: str
    history: dict[str, list[float]]
    log_tilde: float
    elbo: float
    z_usage: float
    spearman_t: float
    q_mean: list[float]
    tuning_z: list[float]
    tuning_rates: list[list[float]]


class Results(TypedDict):
    """The bump positions of the test set, and every fit."""

    test_t: list[float]
    fits: list[FitResult]
