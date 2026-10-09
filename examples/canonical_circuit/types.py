"""JSON schema for the canonical circuit example, exchanged between ``run.py`` and ``plot.py``."""

from typing import TypedDict


class RunResult(TypedDict):
    """One training run at a final penalty strength $\\lambda = \\lambda_q = \\lambda_p$.

    History lists are aligned with ``Results.steps``; ``log_tilde`` and ``elbo`` are the mean test
    $\\log \\tilde p_X$ of the variational model and its ELBO, so that ``test_ll - log_tilde`` is the
    mismatch between the harmonium and the variational model, and ``log_tilde - elbo`` the variational
    gap. Final measurements are on the test set:
    ``identity_error`` is the mean absolute error of the quadrature identity against the exact
    log-likelihood, ``ess_q``/``ess_p`` the mean relative effective sample sizes of the quadrature
    weights, ``kl_q`` the mean KL divergence from the moment-matched exact posterior to the recognition
    Gaussian, ``r2_q``/``r2_exact`` the coefficient of determination of an affine regression of the
    natural parameters of the recognition and of the moment-matched exact posterior on $x$ (one value
    per coordinate), and ``corr_t`` the Spearman correlation of the exact posterior mean of $z$ with
    the curve coordinate. ``off_curve`` is the fraction of model samples whose mean squared distance to
    the noiseless curve exceeds twice the noise variance (``Results.data_off_curve`` for the test data), and ``off_curve_means`` the same for the means
    $\\mathbb E[x \\mid n]$ of the sampled states of the neurons, which leaves out the readout noise.
    """

    lam: float
    seed: int
    train_ll: list[float]
    test_ll: list[float]
    log_tilde: list[float]
    elbo: list[float]
    var_q: list[float]
    var_p: list[float]
    final_test_ll: float
    final_log_tilde: float
    final_elbo: float
    final_var_q: float
    final_var_p: float
    identity_error: float
    ess_q: float
    ess_p: float
    kl_q: float
    r2_q: list[float]
    r2_exact: list[float]
    corr_t: float
    off_curve: float
    off_curve_means: float
    samples: list[list[float]]
    post_mean_exact: list[float]
    post_mean_q: list[float]
    post_sd_exact: list[float]
    post_sd_q: list[float]
    tuning_z: list[float]
    tuning_logits: list[list[float]]
    prior_density: list[float]
    prior_gaussian: list[float]


class Results(TypedDict):
    """The data, and one run per seed and penalty strength."""

    steps: list[int]
    train_x: list[list[float]]
    test_x: list[list[float]]
    test_t: list[float]
    data_off_curve: float
    runs: list[RunResult]
