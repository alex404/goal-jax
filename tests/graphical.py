"""Tests for geometry/exponential_family/graphical.py and models/graphical.

The conjugation, observable density, posterior, natural-mean round trip and EM of the graphical models (hierarchical mixtures of Gaussians, mixtures of factor analyzers, mixtures of conjugated harmoniums, canonical correlation analysis) are checked with every other conjugated harmonium in ``tests/harmonium.py``. This file tests what is particular to them: the mixture view of a mixture of harmoniums, which reads the same coordinates as a mixture whose components are harmoniums, and whitening, which moves the latent prior to the standard normal without changing the observable marginal.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import CliqueEmbedding, Diagonal
from goal.models import (
    AnalyticHMoG,
    MixtureOfFactorAnalyzers,
    NormalLGM,
    analytic_hmog,
    factor_analysis,
)
from goal.models.graphical.mixture import CompleteMixtureOfConjugated

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-5
ATOL = 1e-7

MIXTURES_OF_HARMONIUMS = [
    MixtureOfFactorAnalyzers(n_categories=3, bas_hrm=factor_analysis(4, 2)),
    CompleteMixtureOfConjugated(
        n_categories=2, bas_hrm=NormalLGM(3, Diagonal(), 2, Diagonal())
    ),
]
MIXTURE_IDS = ["mfa", "mixture_of_conjugated"]


@pytest.mark.parametrize("model", MIXTURES_OF_HARMONIUMS, ids=MIXTURE_IDS)
class TestMixtureView:
    """``to_mixture_coords`` reads a mixture of harmoniums as a mixture whose components are the base harmonium."""

    def test_commutes_with_to_mean(self, model: Any) -> None:
        """The mixture view of the mean parameters is the mean parameters of the mixture view."""
        params = model.initialize(jax.random.PRNGKey(0), location=0.0, shape=0.5)
        assert jnp.allclose(
            model.to_mixture_coords(model.to_mean(params)),
            model.mix_man.to_mean(model.to_mixture_coords(params)),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_moves_every_clique_whole(self, model: Any) -> None:
        """Each clique's coordinate block is carried to the same clique's block in the mixture view."""
        params = jax.random.normal(jax.random.PRNGKey(1), (model.dim,))
        mix_params = model.to_mixture_coords(params)
        assert sorted(model.cliques) == sorted(model.mix_man.cliques)
        for clique in model.cliques:
            assert jnp.array_equal(
                CliqueEmbedding(clique, model).project(params),
                CliqueEmbedding(clique, model.mix_man).project(mix_params),
            ), clique

    def test_crossings_act_as_the_component_matrix(self, model: Any) -> None:
        """The mixture view's crossings, summed, act as its one interaction matrix ``cmp_int_map``."""
        mix = model.mix_man
        key_p, key_v = jax.random.split(jax.random.PRNGKey(2))
        int_params = jax.random.normal(key_p, (mix.int_man.dim,))
        v = jax.random.normal(key_v, (mix.lat_man.dim,))
        assert jnp.allclose(
            mix.int_man(int_params, v),
            mix.cmp_int_map(int_params, v),
            rtol=RTOL,
            atol=ATOL,
        )


def moments(man: Any, params: Array) -> tuple[Array, Array]:
    return man.statistical_mean(params), man.statistical_covariance(params)


def whitened_moments(model: Any, params: Array) -> list[tuple[Array, Array]]:
    """The latent moments whitening sets to $(0, I)$: the prior for factor analysis, each component's prior for MFA, and the marginal of $y$ under the prior for HMoG."""
    if isinstance(model, MixtureOfFactorAnalyzers):
        fa, mix = model.bas_hrm, model.mix_man
        comp_params, _ = mix.split_natural_mixture(model.to_mixture_coords(params))
        return [
            moments(fa.prr_man, fa.prior(mix.cmp_man.get_replicate(comp_params, k)))
            for k in range(model.n_categories)
        ]
    if isinstance(model, AnalyticHMoG):
        return [model.prr_man.observable_mean_covariance(model.prior(params))]
    return [moments(model.prr_man, model.prior(params))]


@pytest.mark.parametrize(
    "model",
    [
        factor_analysis(6, 2),
        MixtureOfFactorAnalyzers(n_categories=3, bas_hrm=factor_analysis(6, 2)),
        analytic_hmog(6, Diagonal(), 2, 3),
    ],
    ids=["factor_analysis", "mfa", "analytic_hmog"],
)
def test_whitening(model: Any) -> None:
    """Whitening leaves $\\log p(x)$ unchanged and gives the latent moments mean $0$ and covariance $I$."""
    params = model.initialize(jax.random.PRNGKey(3), location=0.5, shape=1.0)
    xs = model.observable_sample(jax.random.PRNGKey(4), params, 50)
    whitened = model.to_natural(model.whiten_prior(model.to_mean(params)))

    assert jnp.allclose(
        model.average_log_observable_density(whitened, xs),
        model.average_log_observable_density(params, xs),
        rtol=1e-5,
        atol=1e-6,
    )
    for mean, cov in whitened_moments(model, whitened):
        assert jnp.allclose(mean, 0.0, atol=1e-6)
        assert jnp.allclose(cov, jnp.eye(cov.shape[0]), atol=1e-6)
