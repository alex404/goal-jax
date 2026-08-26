"""Tests for ``EFClique`` in geometry/exponential_family/clique.py.

(Named ``ef_clique`` rather than ``clique`` because ``tests/clique.py`` already tests
``geometry/algebra/clique.py``.)

The point of these tests is that ``EFClique`` is not verified against a fresh derivation
but against the ``EmbeddedMap`` machinery every interaction in the library already runs on.
Each case rebuilds a live model's interaction as a clique over its own embeddings and
checks all three operations agree: the outer product that builds a sufficient statistic,
the contraction that builds a likelihood, and the transposed contraction that builds a
posterior. If those hold for every interaction shape in the library, the arity-2 case is
settled and only arity 3 is new.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    Diagonal,
    EFClique,
    EmbeddedMap,
    IdentityEmbedding,
    LinearMap,
    Manifold,
    PositiveDefinite,
)
from goal.models import (
    MixtureOfFactorAnalyzers,
    analytic_hmog,
    factor_analysis,
    full_normal,
    poisson_mixture,
)
from goal.models.harmonium.cca import CanonicalCorrelationAnalysis

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def _as_clique(int_man: LinearMap[Any, Any]) -> EFClique:
    """The clique an existing two-node interaction is: its own two embeddings."""
    assert isinstance(int_man, EmbeddedMap)
    return EFClique(members=(0, 1), selectors=(int_man.cod_emb, int_man.dom_emb))


def _interactions() -> dict[str, tuple[LinearMap[Any, Any], Manifold, Manifold]]:
    """Every distinct interaction shape in the library, with its two node manifolds."""
    fa = factor_analysis(obs_dim=5, lat_dim=2)
    mix = poisson_mixture(n_neurons=4, n_components=3)
    hmog = analytic_hmog(obs_dim=4, obs_rep=Diagonal(), lat_dim=2, n_components=3)
    cca = CanonicalCorrelationAnalysis(
        3, PositiveDefinite(), 2, Diagonal(), 2, PositiveDefinite()
    )
    mfa = MixtureOfFactorAnalyzers(
        n_categories=3, bas_hrm=factor_analysis(obs_dim=4, lat_dim=2)
    )

    cases: dict[str, tuple[LinearMap[Any, Any], Manifold, Manifold]] = {
        "factor_analysis": (fa.int_man, fa.obs_man, fa.pst_man),
        "mixture": (mix.int_man, mix.obs_man, mix.pst_man),
        "hmog": (hmog.int_man, hmog.obs_man, hmog.pst_man),
        "cca_fst": (cca.int_man.blocks[0], cca.obs_man, cca.pst_man),
        "cca_snd": (cca.int_man.blocks[1], cca.obs_man, cca.pst_man),
    }
    for name, block in zip(("mfa_xy", "mfa_xyk", "mfa_xk"), mfa.int_man.blocks):
        cases[name] = (block, mfa.obs_man, mfa.pst_man)
    return cases


CASES = _interactions()
NAMES = sorted(CASES)


def _stats(case: str, seed: int) -> tuple[Array, Array]:
    """Random points in the two nodes' *statistic* spaces (not data space)."""
    _, cod_man, dom_man = CASES[case]
    key_w, key_v = jax.random.split(jax.random.PRNGKey(seed))
    return (
        jax.random.normal(key_w, (cod_man.dim,)),
        jax.random.normal(key_v, (dom_man.dim,)),
    )


class TestEquivalenceWithEmbeddedMap:
    """Every live interaction, rebuilt as a clique, must behave identically."""

    @pytest.mark.parametrize("case", NAMES)
    def test_dim_matches(self, case: str) -> None:
        int_man = CASES[case][0]
        assert _as_clique(int_man).dim == int_man.dim

    @pytest.mark.parametrize("case", NAMES)
    def test_tensor_matches_outer_product(self, case: str) -> None:
        int_man = CASES[case][0]
        w, v = _stats(case, 0)
        assert jnp.allclose(
            _as_clique(int_man).tensor(w, v), int_man.outer_product(w, v)
        )

    @pytest.mark.parametrize("case", NAMES)
    def test_contract_to_codomain_matches_application(self, case: str) -> None:
        """``keep=0`` is the likelihood direction: contract the latent, land on x."""
        int_man = CASES[case][0]
        _, v = _stats(case, 1)
        params = jax.random.normal(jax.random.PRNGKey(7), (int_man.dim,))
        assert jnp.allclose(
            _as_clique(int_man).contract(params, 0, v), int_man(params, v)
        )

    @pytest.mark.parametrize("case", NAMES)
    def test_contract_to_domain_matches_transposed_application(self, case: str) -> None:
        """``keep=1`` is the posterior direction: contract x, land on the latent."""
        int_man = CASES[case][0]
        w, _ = _stats(case, 2)
        params = jax.random.normal(jax.random.PRNGKey(8), (int_man.dim,))
        assert jnp.allclose(
            _as_clique(int_man).contract(params, 1, w),
            int_man.transpose_apply(params, w),
        )


class TestSufficientStatistic:
    """The clique's own contribution to a joint statistic."""

    def test_matches_the_harmonium_interaction_block(self) -> None:
        """What ``Harmonium.sufficient_statistic`` computes for the cross span."""
        fa = factor_analysis(obs_dim=5, lat_dim=2)
        clique = _as_clique(fa.int_man)
        key_x, key_z = jax.random.split(jax.random.PRNGKey(3))
        x = jax.random.normal(key_x, (fa.obs_man.data_dim,))
        z = jax.random.normal(key_z, (fa.pst_man.data_dim,))

        joint = jnp.concatenate([x, z])
        _, int_stats, _ = fa.split_level(fa.sufficient_statistic(joint))
        assert jnp.allclose(clique.sufficient_statistic(x, z), int_stats)


class TestArityOne:
    """A bias is a clique of one member, with no coupling at all."""

    def test_bias_clique_is_the_identity_on_a_statistic(self) -> None:
        man = full_normal(3)
        clique = EFClique(members=(0,), selectors=(IdentityEmbedding(man),))
        assert clique.dim == man.dim
        x = jax.random.normal(jax.random.PRNGKey(4), (man.data_dim,))
        assert jnp.array_equal(
            clique.sufficient_statistic(x), man.sufficient_statistic(x)
        )


class TestValidation:
    """Arity is one fact, so the ways of disagreeing with it are all construction errors.

    A selector *is* an axis, so a count mismatch and an axis mismatch are the same error,
    and ``LinearClique`` raises it once for every clique rather than per subclass.
    """

    def test_member_and_selector_counts_must_agree(self) -> None:
        man = full_normal(2)
        with pytest.raises(ValueError, match="2 members but 1 axes"):
            EFClique(members=(0, 1), selectors=(IdentityEmbedding(man),))

    def test_a_clique_may_not_repeat_a_node(self) -> None:
        man = full_normal(2)
        with pytest.raises(ValueError, match="must be ascending and distinct"):
            EFClique(
                members=(1, 1),
                selectors=(IdentityEmbedding(man), IdentityEmbedding(man)),
            )

    def test_members_must_be_ascending(self) -> None:
        man = full_normal(2)
        with pytest.raises(ValueError, match="must be ascending and distinct"):
            EFClique(
                members=(1, 0),
                selectors=(IdentityEmbedding(man), IdentityEmbedding(man)),
            )


class TestJointBlocksAreNotProductsOfMarginals:
    """Why a multi-latent clique reads a *joint* block instead of per-node statistics.

    ``tensor`` multiplies its members' statistics together, which is exact when every
    member is observed. When two members are latent the block is
    $\\mathbb E[\\bigotimes_i \\mathbf s_i]$ jointly, and expectation does not pass through a
    tensor product. ``select_joint`` is the operation for that case: it contracts each
    selector into an axis of the joint block and never forms a marginal.
    """

    @staticmethod
    def _mfa():
        from goal.models import MixtureOfFactorAnalyzers

        return MixtureOfFactorAnalyzers(
            n_categories=3, bas_hrm=factor_analysis(obs_dim=4, lat_dim=2)
        )

    def test_on_a_sample_point_the_latent_block_is_a_product(self) -> None:
        """Deterministic statistics, so marginals and the joint agree exactly."""
        mfa = self._mfa()
        mix = mfa.pst_man
        y = jax.random.normal(jax.random.PRNGKey(1), (mix.obs_man.data_dim,))
        z = jnp.concatenate([y, jnp.array([1.0])])

        _, s_yk, _ = mix.split_level(mix.sufficient_statistic(z))
        s_y = mix.obs_man.sufficient_statistic(y)
        s_k = mix.lat_man.sufficient_statistic(jnp.array([1.0]))
        assert jnp.allclose(s_yk, mix.int_man.outer_product(s_y, s_k))

    def test_in_mean_coordinates_it_is_not(self) -> None:
        """A posterior expectation, where the marginal product is off by order one.

        This is the trap: using ``tensor`` here would be silently wrong, not obviously so.
        """
        mfa = self._mfa()
        mix = mfa.pst_man
        params = mfa.initialize(jax.random.PRNGKey(0), shape=0.5)
        x = jax.random.normal(jax.random.PRNGKey(2), (mfa.obs_man.data_dim,))

        m_y, m_yk, m_k = mix.split_level(mix.to_mean(mfa.posterior_at(params, x)))
        factored = mix.int_man.outer_product(m_y, m_k)
        gap = jnp.max(jnp.abs(m_yk - factored)) / jnp.max(jnp.abs(m_yk))
        assert gap > 0.1, "expected an order-one gap, not a rounding difference"


class TestArityThreeReproducesMFA:
    """The three-way interaction $\\theta_{XYK}$, as a genuine arity-3 clique.

    MFA stores this block as an arity-2 map whose domain is the joint $(y,k)$ statistic ---
    which is exactly "select a sub-statistic on the $y$ axis, identity on the $k$ axis"
    written as one matrix. These tests check that an ``EFClique`` over three members
    reproduces it on every operation, *including in mean coordinates at the E-step*, which
    is the case that decides whether higher arity is usable at all.

    The clique's latent members $(y, k)$ are themselves a clique of the level above, so
    their joint expectation exists as a block to select from. That is the structural
    condition higher arity needs.
    """

    @staticmethod
    def _setup():
        from goal.models import MixtureOfFactorAnalyzers

        mfa = MixtureOfFactorAnalyzers(
            n_categories=3, bas_hrm=factor_analysis(obs_dim=4, lat_dim=2)
        )
        xyk = mfa.int_man.blocks[1]
        assert isinstance(xyk, EmbeddedMap)
        # x location, y location, k in full: the three members' selectors.
        clique = EFClique(
            members=(0, 1, 2),
            selectors=(
                xyk.cod_emb,
                mfa.bas_hrm.int_man.dom_emb,
                IdentityEmbedding(mfa.pst_man.lat_man),
            ),
        )
        return mfa, xyk, clique

    def test_dimension_matches_the_live_block(self) -> None:
        _, xyk, clique = self._setup()
        assert clique.dim == xyk.dim
        assert clique.form.sub_dims == (4, 2, 2)

    def test_mean_block_matches_at_the_e_step(self) -> None:
        """The decisive one: mean coordinates, both latent members dependent."""
        mfa, xyk, clique = self._setup()
        mix = mfa.pst_man
        params = mfa.initialize(jax.random.PRNGKey(0), shape=0.5)
        x = jax.random.normal(jax.random.PRNGKey(2), (mfa.obs_man.data_dim,))

        s_x = mfa.obs_man.sufficient_statistic(x)
        lat_means = mix.to_mean(mfa.posterior_at(params, x))
        _, m_yk, _ = mix.split_level(lat_means)

        live = xyk.outer_product(s_x, lat_means)
        rebuilt = jnp.outer(
            clique.selectors[0].project(s_x), clique.select_joint((1, 2), m_yk)
        ).ravel()
        assert jnp.allclose(live, rebuilt)

    def test_mean_block_matches_over_many_draws(self) -> None:
        """Not a coincidence of one posterior."""
        mfa, xyk, clique = self._setup()
        mix = mfa.pst_man
        params = mfa.initialize(jax.random.PRNGKey(11), shape=0.7)
        xs = jax.random.normal(jax.random.PRNGKey(12), (16, mfa.obs_man.data_dim))
        for x in xs:
            s_x = mfa.obs_man.sufficient_statistic(x)
            lat_means = mix.to_mean(mfa.posterior_at(params, x))
            _, m_yk, _ = mix.split_level(lat_means)
            live = xyk.outer_product(s_x, lat_means)
            rebuilt = jnp.outer(
                clique.selectors[0].project(s_x), clique.select_joint((1, 2), m_yk)
            ).ravel()
            assert jnp.allclose(live, rebuilt)

    def test_posterior_direction_matches(self) -> None:
        """Contract the observed member, leave $(y, k)$ joined: the posterior message."""
        mfa, xyk, clique = self._setup()
        params = mfa.initialize(jax.random.PRNGKey(3), shape=0.5)
        x = jax.random.normal(jax.random.PRNGKey(4), (mfa.obs_man.data_dim,))
        s_x = mfa.obs_man.sufficient_statistic(x)
        _, int_params, _ = mfa.split_level(params)
        xyk_params = mfa.int_man.coord_blocks(int_params)[1]

        live = xyk.dom_emb.project(xyk.transpose_apply(xyk_params, s_x))
        rebuilt = clique.partial_contract(xyk_params, (1, 2), s_x)
        assert jnp.allclose(live, rebuilt)

    def test_likelihood_direction_matches(self) -> None:
        """Contract both latent members, land on the observable."""
        mfa, xyk, clique = self._setup()
        mix = mfa.pst_man
        params = mfa.initialize(jax.random.PRNGKey(5), shape=0.5)
        _, int_params, _ = mfa.split_level(params)
        xyk_params = mfa.int_man.coord_blocks(int_params)[1]

        y = jax.random.normal(jax.random.PRNGKey(6), (mix.obs_man.data_dim,))
        z = jnp.concatenate([y, jnp.array([1.0])])
        s_y = mix.obs_man.sufficient_statistic(y)
        s_k = mix.lat_man.sufficient_statistic(jnp.array([1.0]))

        live = xyk(xyk_params, mix.sufficient_statistic(z))
        rebuilt = clique.selectors[0].embed(
            clique.partial_contract(xyk_params, (0,), s_y, s_k)
        )
        assert jnp.allclose(live, rebuilt)
