"""JSON schema for the canonical circuit example, exchanged between ``run.py`` and ``plot.py``."""

from typing import TypedDict


class FitResults(TypedDict):
    """One fit of the circuit, measured on the test set.

    ``elbo`` is the test ELBO after each of ``steps``, and ``residual_sds`` the s.d. of the
    conjugation residual of each level under the model or under the recognition model. ``q_mean`` and
    ``q_sd`` are the moments of $q(z \\mid x)$ at each test image, and ``spearman`` the rank
    correlation of ``q_mean`` with the bump position. ``tuning_rates`` holds the firing probability
    of each neuron under the generative model at each point of ``tuning_z``. ``sample_images`` are
    samples of the circuit sorted by their latents ``sample_z``.
    """

    steps: list[int]
    elbo: list[float]
    residual_sds: dict[str, list[float]]
    q_mean: list[float]
    q_sd: list[float]
    spearman: float
    tuning_z: list[float]
    tuning_rates: list[list[float]]
    sample_z: list[float]
    sample_images: list[list[float]]


class Results(TypedDict):
    """The test data, Gaussian baselines, and each fit of the circuit by its objective.

    ``baselines`` is the test log-likelihood of Gaussians fit to the training set. ``test_t`` is the
    bump position of each test image, in the order of the fits' ``q_mean``; ``data_images`` are the
    test images sorted by their bump positions ``data_t``.
    """

    baselines: dict[str, float]
    test_t: list[float]
    data_t: list[float]
    data_images: list[list[float]]
    fits: dict[str, FitResults]
