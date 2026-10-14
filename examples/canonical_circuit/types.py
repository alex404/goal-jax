"""JSON schema for the canonical circuit example, exchanged between ``run.py`` and ``plot.py``."""

from typing import TypedDict


class Results(TypedDict):
    """The fit, measured on the test set.

    ``elbo`` is the test ELBO after each of ``steps``, and ``baselines`` the test log-likelihood of
    Gaussians fit to the training set. ``q_mean`` and ``q_sd`` are the moments of $q(z \\mid x)$ at each
    test image, whose bump position is ``test_t``, and ``spearman`` their rank correlation.
    ``tuning_rates`` holds the firing probability of each neuron under the generative model at each
    point of ``tuning_z``. ``sample_images`` are samples of the circuit sorted by $z$, and
    ``data_images`` test images sorted by bump position.
    """

    steps: list[int]
    elbo: list[float]
    baselines: dict[str, float]
    test_t: list[float]
    q_mean: list[float]
    q_sd: list[float]
    spearman: float
    tuning_z: list[float]
    tuning_rates: list[list[float]]
    sample_images: list[list[float]]
    data_images: list[list[float]]
