"""JSON schema for the canonical circuit example, exchanged between ``run.py`` and ``plot.py``."""

from typing import Any, TypedDict


class RunResult(TypedDict):
    """One training run: a graph of couplings, with or without $z$, a fit, a penalty strength and a seed.

    ``history`` holds measurements on the test set after each chunk of ``steps``: ``test_ll`` the mean
    exact $\\log p(x)$ of the graphical harmonium, ``log_tilde`` and ``elbo`` the mean $\\log \\tilde p(x)$
    of the variational model and its ELBO (by enumeration and quadrature), ``kl_q`` the divergence of the
    recognition model from the exact posterior, ``var_q``/``var_p`` the residual variances under the
    recognition model and the variational model, summed over the levels (Monte Carlo), and ``corr_t`` the
    Spearman correlation of the exact posterior mean of $z$ with the bump position. ``stopped`` is the step
    at which the parameters became non-finite, or ``None``, and ``skipped`` the number of updates skipped
    for a non-finite gradient.

    The ``final_*`` entries repeat these at the end, with the residual variances per level (bottom level
    first). ``off_curve`` is the probability mass of the harmonium on states $n$ whose mean $\\mathbb E[x
    \\mid n]$ lies off the noiseless curve, by enumeration, and ``off_curve_tilde`` the same under $\\tilde
    p$. ``sample_means`` are the means of 300 states sampled from the harmonium, ``post_mean`` and
    ``post_var`` the exact posterior moments of $z$ on the test set, ``z_usage`` the fraction of the
    variance of $z$ explained by $x$, and ``tuning_rates`` the firing probabilities of each neuron
    under the generative model at ``tuning_z``, with ``prior_density`` the density of $\\tilde p(z)$.
    """

    key: str
    couplings: str
    use_z: bool
    fit: str
    lam: float
    seed: int
    training: dict[str, Any]
    stopped: int | None
    skipped: int
    steps: list[int]
    history: dict[str, list[float]]
    params: dict[str, Any]
    final_test_ll: float
    final_log_tilde: float
    final_elbo: float
    final_kl_q: float
    final_var_p_levels: list[float]
    final_var_q_levels: list[float]
    final_corr_t: float
    post_mean: list[float]
    post_var: list[float]
    z_usage: float
    off_curve: float
    off_curve_tilde: float
    sample_means: list[list[float]]
    tuning_z: list[float]
    tuning_rates: list[list[float]]
    prior_density: list[float]


class Results(TypedDict):
    """The training settings, data for the plots, and every run of those settings."""

    training: dict[str, Any]
    data_x: list[list[float]]
    test_t: list[float]
    runs: list[RunResult]
