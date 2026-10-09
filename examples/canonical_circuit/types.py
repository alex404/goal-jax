"""JSON schema for the canonical circuit example, exchanged between ``run.py`` and ``plot.py``."""

from typing import TypedDict


class RunResult(TypedDict):
    """One training run at a final penalty strength $\\lambda$.

    History lists are aligned with ``Results.steps``. ``train_ll``/``test_ll`` are the mean exact
    $\\log p(x)$ of the graphical harmonium, ``log_tilde`` and ``elbo`` the mean test $\\log \\tilde p(x)$
    of the variational model and its ELBO (by enumeration and quadrature), so that ``test_ll -
    log_tilde`` is the mismatch between the harmonium and the variational model, and ``log_tilde -
    elbo`` the variational gap. ``kl_q`` is the mean divergence of the recognition model from the exact
    posterior of the harmonium, and ``var_q``/``var_p`` the residual variances under the recognition
    model and the variational model, summed over the levels (Monte Carlo). ``min_prc`` is the smallest
    precision of $z$ over $p(z \\mid n)$, the prior and the recognition model on the test set; a negative
    value means the run has left the domain. ``stopped`` is the step at which the loss became NaN, or
    ``None``; the histories then end at the last finite chunk.

    Final measurements are on the test set: the per-level residual variances (bottom level first),
    ``corr_t`` the Spearman correlation of the exact posterior mean of $z$ with the curve coordinate,
    ``off_curve`` the fraction of samples of the variational model whose mean squared distance to the
    noiseless curve exceeds twice the noise variance (``Results.data_off_curve`` for the test data), and
    ``off_curve_means`` the same for the means $\\mathbb E[x \\mid n]$ of the sampled states of the
    neurons, which leaves out the readout noise.
    """

    lam: float
    seed: int
    stopped: int | None
    train_ll: list[float]
    test_ll: list[float]
    log_tilde: list[float]
    elbo: list[float]
    kl_q: list[float]
    var_q: list[float]
    var_p: list[float]
    min_prc: list[float]
    final_train_ll: float
    final_test_ll: float
    final_log_tilde: float
    final_elbo: float
    final_kl_q: float
    final_var_q: float
    final_var_p: float
    final_min_prc: float
    final_var_p_levels: list[float]
    final_var_q_levels: list[float]
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
