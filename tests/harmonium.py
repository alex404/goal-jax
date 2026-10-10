"""Tests for geometry/exponential_family/harmonium.py and the harmoniums of models/harmonium and models/graphical.

One case list covers every shipped conjugated harmonium, from linear Gaussian models and mixtures to canonical correlation analysis, hierarchical mixtures of Gaussians and mixtures of factor analyzers. Each case is checked against independent ground truth: the conjugation equation $\\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)) = \\rho \\cdot \\mathbf s_Z(z) + \\chi$ at latent points; the observable density against $\\sum_z p(x \\mid z) p(z)$, by enumeration for discrete latents and by the closed-form Gaussian marginal for Gaussian latents; and the posterior against the conditional of the joint density. Analytic cases also satisfy the natural-mean round trip, and EM does not decrease the log-likelihood. Closed forms that pin individual models follow.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax.scipy import stats
from jax.scipy.linalg import block_diag
from jax.scipy.special import logsumexp

from goal.geometry import Diagonal, PositiveDefinite, Scale
from goal.models import (
    AnalyticCanonicalCorrelationAnalysis,
    AnalyticMixture,
    BoltzmannLGM,
    CanonicalCorrelationAnalysis,
    DifferentiableBoltzmannLGM,
    MixtureOfFactorAnalyzers,
    Normal,
    NormalAnalyticLGM,
    NormalCovarianceEmbedding,
    NormalLGM,
    PoissonVonMisesHarmonium,
    analytic_hmog,
    com_poisson_mixture,
    differentiable_hmog,
    factor_analysis,
    poisson_mixture,
)
from goal.models.graphical.mixture import (
    CompleteMixtureOfConjugated,
    CompleteMixtureOfSymmetric,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-5
ATOL = 1e-7


@dataclass(frozen=True)
class Case:
    name: str
    model: Any
    latent: str
    """How latent points are drawn and integrated out: ``"categorical"`` and ``"boltzmann"`` are enumerated, ``"normal"`` and ``"normal_mixture"`` (a normal $y$ with a category $k$) are integrated in closed form."""
    analytic: bool


CASES = [
    Case("lgm", NormalAnalyticLGM(3, PositiveDefinite(), 2), "normal", True),
    Case("factor_analysis", factor_analysis(4, 2), "normal", True),
    Case(
        "diagonal_posterior_lgm",
        NormalLGM(3, Diagonal(), 2, Diagonal()),
        "normal",
        False,
    ),
    Case("boltzmann_lgm", BoltzmannLGM(2, PositiveDefinite(), 3), "boltzmann", False),
    Case(
        "mean_field_boltzmann_lgm",
        DifferentiableBoltzmannLGM(2, PositiveDefinite(), 3),
        "boltzmann",
        False,
    ),
    Case(
        "normal_mixture",
        AnalyticMixture(Normal(2, PositiveDefinite()), 3),
        "categorical",
        True,
    ),
    Case("poisson_mixture", poisson_mixture(5, 3), "categorical", True),
    Case("com_poisson_mixture", com_poisson_mixture(4, 3), "categorical", False),
    Case(
        "cca",
        CanonicalCorrelationAnalysis(
            3, PositiveDefinite(), 2, Diagonal(), 2, Diagonal()
        ),
        "normal",
        False,
    ),
    Case(
        "analytic_cca",
        AnalyticCanonicalCorrelationAnalysis(3, PositiveDefinite(), 2, Diagonal(), 2),
        "normal",
        True,
    ),
    Case("analytic_hmog", analytic_hmog(4, Diagonal(), 2, 3), "normal_mixture", True),
    Case(
        "differentiable_hmog",
        differentiable_hmog(4, Diagonal(), 2, Diagonal(), 3),
        "normal_mixture",
        False,
    ),
    Case(
        "mfa",
        MixtureOfFactorAnalyzers(n_categories=3, bas_hrm=factor_analysis(4, 2)),
        "normal_mixture",
        True,
    ),
    Case(
        "mixture_of_conjugated",
        CompleteMixtureOfConjugated(
            n_categories=2, bas_hrm=NormalLGM(3, Diagonal(), 2, Diagonal())
        ),
        "normal_mixture",
        False,
    ),
    Case(
        "mixture_of_symmetric",
        CompleteMixtureOfSymmetric(n_categories=3, bas_hrm=factor_analysis(3, 2)),
        "normal_mixture",
        False,
    ),
]
IDS = [case.name for case in CASES]
ANALYTIC = [case for case in CASES if case.analytic]
ENUMERABLE = [case for case in CASES if case.latent in ("categorical", "boltzmann")]


def initial_params(case: Case, seed: int) -> Array:
    return case.model.initialize(jax.random.PRNGKey(seed), location=0.0, shape=0.5)


def latent_points(case: Case, key: Array) -> Array:
    """Every latent state for enumerable latents, otherwise random points (with every category)."""
    model = case.model
    if case.latent == "categorical":
        return jnp.arange(model.pst_man.n_categories, dtype=jnp.float64)[:, None]
    if case.latent == "boltzmann":
        return model.prr_man.states
    if case.latent == "normal":
        return jax.random.normal(key, (8, model.pst_man.data_dim))
    n_cat = model.pst_man.n_categories
    ys = jax.random.normal(key, (n_cat, 3, model.pst_man.obs_man.data_dim))
    ks = jnp.broadcast_to(
        jnp.arange(n_cat, dtype=jnp.float64)[:, None, None], (n_cat, 3, 1)
    )
    return jnp.concatenate([ys, ks], axis=-1).reshape(n_cat * 3, -1)


def normal_moments(man: Any, params: Array) -> tuple[Array, Array]:
    """Mean and covariance matrix of a normal, or of a tuple of independent normals."""
    if isinstance(man, Normal):
        mean, cov = man.split_mean_covariance(man.to_mean(params))
        return mean, man.cov_man.to_matrix(cov)
    moments = [
        normal_moments(elm, elm_params)
        for elm, elm_params in zip(man.elm_mans, man.split_coords(params), strict=True)
    ]
    return (
        jnp.concatenate([mean for mean, _ in moments]),
        block_diag(*(cov for _, cov in moments)),
    )


def gaussian_log_marginal(case: Case, params: Array, x: Array) -> Array:
    """$\\log p(x)$ in closed form for a normal latent $y$, possibly with a category $k$.

    Given $(y, k)$ the observable is normal with mean $A_k y + b_k$ and covariance $\\Sigma_k$, read off :meth:`likelihood_at` at $y = 0$ and the unit vectors. The prior gives $p(k)$ and $y \\mid k \\sim N(\\mu_k, \\Sigma^Y_k)$, so $p(x) = \\sum_k p(k) N(x; A_k \\mu_k + b_k, A_k \\Sigma^Y_k A_k^\\top + \\Sigma_k)$.
    """
    model = case.model
    prr_man, prior = model.prr_man, model.prior(params)
    if case.latent == "normal":
        lat_dim = model.pst_man.data_dim
        components = [(jnp.zeros(0), jnp.asarray(0.0), *normal_moments(prr_man, prior))]
    else:
        lat_dim = model.pst_man.obs_man.data_dim
        comp_params, cat_params = prr_man.split_natural_mixture(prior)
        probs = prr_man.lat_man.to_probs(prr_man.lat_man.to_mean(cat_params))
        components = [
            (
                jnp.array([float(k)]),
                jnp.log(probs[k]),
                *normal_moments(
                    prr_man.obs_man, prr_man.cmp_man.get_replicate(comp_params, k)
                ),
            )
            for k in range(prr_man.n_categories)
        ]

    def conditional(y: Array, k: Array) -> tuple[Array, Array]:
        return normal_moments(
            model.obs_man, model.likelihood_at(params, jnp.concatenate([y, k]))
        )

    log_terms = []
    for k, log_weight, lat_mean, lat_cov in components:
        offset, obs_cov = conditional(jnp.zeros(lat_dim), k)
        loadings = jnp.stack(
            [conditional(e, k)[0] - offset for e in jnp.eye(lat_dim)], axis=1
        )
        mean = loadings @ lat_mean + offset
        cov = loadings @ lat_cov @ loadings.T + obs_cov
        log_terms.append(log_weight + stats.multivariate_normal.logpdf(x, mean, cov))
    return logsumexp(jnp.stack(log_terms))


def enumerated_log_marginal(case: Case, params: Array, x: Array) -> Array:
    """$\\log \\sum_z p(x \\mid z) p(z)$ over every latent state."""
    model = case.model
    prior = model.prior(params)

    def log_joint(z: Array) -> Array:
        return model.obs_man.log_density(
            model.likelihood_at(params, z), x
        ) + model.prr_man.log_density(prior, z)

    return logsumexp(jax.vmap(log_joint)(latent_points(case, jax.random.PRNGKey(0))))


@pytest.mark.parametrize("case", CASES, ids=IDS)
class TestConjugatedHarmonium:
    """The contract of a conjugated harmonium, against independent ground truth."""

    def test_conjugation_equation(self, case: Case) -> None:
        """$\\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)) = \\rho \\cdot \\mathbf s_Z(z) + \\chi$, with $\\mathbf s_Z$ the posterior's statistic on the left and the prior's on the right."""
        model = case.model
        lkl_params = model.likelihood_function(initial_params(case, 0))
        rho = model.conjugation_parameters(lkl_params)
        chi = model.conjugation_offset(lkl_params)

        def residual(z: Array) -> Array:
            lhs = model.obs_man.log_partition_function(
                model.lkl_fun_man(lkl_params, model.pst_man.sufficient_statistic(z))
            )
            return lhs - jnp.dot(rho, model.prr_man.sufficient_statistic(z)) - chi

        zs = latent_points(case, jax.random.PRNGKey(1))
        assert jnp.allclose(jax.vmap(residual)(zs), 0.0, atol=1e-6)

    def test_observable_density(self, case: Case) -> None:
        """$\\log p(x)$ equals $\\log \\sum_z p(x \\mid z) p(z)$, enumerated or in closed form."""
        model = case.model
        params = initial_params(case, 2)
        xs = model.observable_sample(jax.random.PRNGKey(3), params, 4)
        brute_force = (
            enumerated_log_marginal
            if case.latent in ("categorical", "boltzmann")
            else gaussian_log_marginal
        )
        for x in xs:
            assert jnp.allclose(
                model.log_observable_density(params, x),
                brute_force(case, params, x),
                rtol=1e-4,
                atol=1e-6,
            )

    def test_posterior_is_the_conditional(self, case: Case) -> None:
        """$\\log q(z \\mid x) - \\theta \\cdot \\mathbf s(x, z) - \\log \\mu(x, z)$ does not depend on $z$."""
        model = case.model
        params = initial_params(case, 4)
        x = model.observable_sample(jax.random.PRNGKey(5), params, 1)[0]
        posterior = model.posterior_at(params, x)

        def gap(z: Array) -> Array:
            xz = jnp.concatenate([x, z])
            log_joint = jnp.dot(params, model.sufficient_statistic(xz))
            log_joint += model.log_base_measure(xz)
            return model.pst_man.log_density(posterior, z) - log_joint

        gaps = jax.vmap(gap)(latent_points(case, jax.random.PRNGKey(6)))
        assert jnp.allclose(gaps, gaps[0], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("case", ENUMERABLE, ids=[case.name for case in ENUMERABLE])
def test_posterior_normalizes(case: Case) -> None:
    """$\\sum_z q(z \\mid x) = 1$ over every latent state."""
    model = case.model
    params = initial_params(case, 7)
    x = model.observable_sample(jax.random.PRNGKey(8), params, 1)[0]
    posterior = model.posterior_at(params, x)
    zs = latent_points(case, jax.random.PRNGKey(0))
    total = jnp.sum(jax.vmap(model.pst_man.density, (None, 0))(posterior, zs))
    assert jnp.allclose(total, 1.0, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", ANALYTIC, ids=[case.name for case in ANALYTIC])
class TestAnalyticConjugated:
    """Mean coordinates and EM of analytic harmoniums."""

    def test_natural_mean_round_trip(self, case: Case) -> None:
        params = initial_params(case, 9)
        recovered = case.model.to_natural(case.model.to_mean(params))
        assert jnp.allclose(recovered, params, rtol=1e-4, atol=1e-6)

    def test_em_does_not_decrease_log_likelihood(self, case: Case) -> None:
        model = case.model
        xs = model.observable_sample(
            jax.random.PRNGKey(10), initial_params(case, 11), 300
        )
        params = initial_params(case, 12)
        before = model.average_log_observable_density(params, xs)
        after = model.average_log_observable_density(
            model.expectation_maximization(params, xs), xs
        )
        assert after >= before - 1e-8


class TestLinearGaussianModel:
    """Closed forms of linear Gaussian models."""

    @pytest.mark.parametrize("rep", [Scale(), Diagonal(), PositiveDefinite()])
    def test_matches_joint_normal_and_scipy(
        self, rep: Scale | Diagonal | PositiveDefinite
    ) -> None:
        """An LGM fit to joint samples has the log-partition and joint density of its joint normal, and that density is scipy's."""
        obs_cov = jnp.array([[4.0, 1.5, 0.0], [1.5, 3.0, 1.0], [0.0, 1.0, 2.0]])
        int_cov = jnp.array([[-1.0, 0.5], [0.0, 0.0], [0.5, 0.0]])
        lat_cov = jnp.array([[3.0, 1.0], [1.0, 2.0]])
        cov = jnp.block([[obs_cov, int_cov], [int_cov.T, lat_cov]])
        mean = jnp.array([2.0, -1.0, 0.0, 3.0, -2.0])
        samples = jax.random.multivariate_normal(
            jax.random.PRNGKey(0), mean, cov, (1000,)
        )

        lgm = NormalAnalyticLGM(3, rep, 2)
        params = lgm.to_natural(lgm.average_sufficient_statistic(samples))
        nor_man = Normal(5, PositiveDefinite())
        nor_params = lgm.to_normal(params)
        assert jnp.allclose(
            lgm.log_partition_function(params),
            nor_man.log_partition_function(nor_params),
            rtol=RTOL,
            atol=ATOL,
        )

        nor_mean, nor_cov = normal_moments(nor_man, nor_params)
        scipy_lds = stats.multivariate_normal.logpdf(samples, nor_mean, nor_cov)
        lgm_lds = jax.vmap(lgm.log_density, (None, 0))(params, samples)
        assert jnp.allclose(lgm_lds, scipy_lds, rtol=RTOL, atol=1e-6)

    def test_factor_analysis_from_loadings(self) -> None:
        """Loadings $L$, means $\\mu$ and noise $D$ give the marginal $N(\\mu, L L^\\top + D)$."""
        loadings = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        means = jnp.array([1.0, 2.0, 3.0])
        diags = jnp.array([0.1, 0.2, 0.3])
        fa = factor_analysis(3, 2)
        params = fa.initialize_from_loadings(loadings, means, diags)
        nor_man, nor_params = fa.observable_distribution(params)
        mean, cov = normal_moments(nor_man, nor_params)
        assert jnp.allclose(mean, means, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(
            cov, loadings @ loadings.T + jnp.diag(diags), rtol=RTOL, atol=ATOL
        )


class TestNormalCovarianceEmbedding:
    """Embeds a diagonal normal into the full normal family."""

    @staticmethod
    def _diagonal() -> tuple[
        NormalCovarianceEmbedding[PositiveDefinite, Diagonal], Array
    ]:
        sub = Normal(3, Diagonal())
        emb = NormalCovarianceEmbedding(Normal(3, PositiveDefinite()), sub)
        means = sub.join_mean_covariance(
            jnp.array([1.0, -0.5, 0.3]), jnp.array([1.5, 2.0, 0.8])
        )
        return emb, means

    def test_project_inverts_embed(self) -> None:
        """Embedding natural parameters and projecting the mean parameters recovers the original mean parameters."""
        emb, means = self._diagonal()
        amb_means = emb.amb_man.to_mean(emb.embed(emb.sub_man.to_natural(means)))
        assert jnp.allclose(emb.project(amb_means), means, rtol=RTOL, atol=ATOL)

    def test_embed_preserves_density(self) -> None:
        emb, means = self._diagonal()
        params = emb.sub_man.to_natural(means)
        xs = jax.random.normal(jax.random.PRNGKey(7), (20, 3))
        sub_lds = jax.vmap(emb.sub_man.log_density, (None, 0))(params, xs)
        amb_lds = jax.vmap(emb.amb_man.log_density, (None, 0))(emb.embed(params), xs)
        assert jnp.allclose(sub_lds, amb_lds, rtol=RTOL, atol=ATOL)


def test_analytic_cca_matches_differentiable_cca() -> None:
    """With a full-covariance posterior the two CCA classes share their log-partition function."""
    analytic = AnalyticCanonicalCorrelationAnalysis(
        3, PositiveDefinite(), 2, Diagonal(), 2
    )
    differentiable = CanonicalCorrelationAnalysis(
        3, PositiveDefinite(), 2, Diagonal(), 2, PositiveDefinite()
    )
    params = analytic.initialize(jax.random.PRNGKey(15), shape=0.3)
    assert jnp.allclose(
        analytic.log_partition_function(params),
        differentiable.log_partition_function(params),
        rtol=RTOL,
        atol=ATOL,
    )


def test_von_mises_tuning_curves() -> None:
    """The Poisson-von Mises harmonium has rates $\\lambda_i(z) = \\exp(b_i + g_i \\cos(z - \\mu_i))$ for gains $g_i$ and preferred stimuli $\\mu_i$."""
    n = 6
    model = PoissonVonMisesHarmonium(n, 1)
    baselines = jnp.linspace(-0.5, 0.5, n)
    gains = jnp.linspace(0.5, 2.0, n)
    preferred = jnp.linspace(0.0, 2 * jnp.pi, n, endpoint=False)
    (xz_map,) = model.crs_maps
    weights = jnp.stack(
        [gains * jnp.cos(preferred), gains * jnp.sin(preferred)], axis=1
    )
    params = model.join_coords(baselines, xz_map.from_matrix(weights), jnp.zeros(2))

    for z in jnp.linspace(-jnp.pi, jnp.pi, 7):
        rates = model.obs_man.to_mean(model.likelihood_at(params, jnp.array([z])))
        expected = jnp.exp(baselines + gains * jnp.cos(z - preferred))
        assert jnp.allclose(rates, expected, rtol=RTOL, atol=ATOL)
