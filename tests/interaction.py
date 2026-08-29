"""Tests for ``LinearClique`` in geometry/manifold/clique.py.

(``tests/clique.py`` tests ``geometry/algebra/clique.py`` and ``tests/graphical.py`` the
layouts and the bare form algebra; this file is the embedding-carrying clique itself.)

A ``LinearClique`` is both a clique and a linear map, so its two readings must agree. Each
case takes a live model's interaction and checks that the clique operations reproduce the
map operations: the outer product that builds a sufficient statistic, the contraction that
builds a likelihood, and the transposed contraction that builds a posterior. If those hold
for every interaction shape in the library, the arity-2 case is settled and only arity 3 is
new.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    Diagonal,
    Interaction,
    LinearMap,
    Manifold,
    PositiveDefinite,
)
from goal.models import (
    MixtureOfFactorAnalyzers,
    analytic_hmog,
    factor_analysis,
    poisson_mixture,
)
from goal.models.harmonium.cca import CanonicalCorrelationAnalysis

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def _as_map(int_man: LinearMap[Any, Any]) -> Interaction[Any, Any]:
    """A live interaction, which is one conditional reading of a clique."""
    assert isinstance(int_man, Interaction)
    return int_man


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
NODE_NAMES = [name for name in NAMES if _as_map(CASES[name][0]).clique.arity == 2]
"""Cases contracting a single node.

There the clique reading and the map reading coincide up to the paths: contracting one axis
leaves one axis, and a single node can be placed back through its own way in. At arity 3 the
contracted side is a *group*, no single axis can be placed through a joint path, and the two
readings only meet after ``project_domain`` --- which is what
``TestArityThreeReproducesMFA`` checks instead.

The criterion is arity, not what the path happens to be: a way in that is a
``CliqueEmbedding`` on one node is no different from one that is a slot embedding.
"""


def _stats(case: str, seed: int) -> tuple[Array, Array]:
    """Random points in the two nodes' *statistic* spaces (not data space)."""
    _, cod_man, dom_man = CASES[case]
    key_w, key_v = jax.random.split(jax.random.PRNGKey(seed))
    return (
        jax.random.normal(key_w, (cod_man.dim,)),
        jax.random.normal(key_v, (dom_man.dim,)),
    )


class TestEquivalenceWithMap:
    """Every live interaction over a single-node domain reads the same both ways."""

    @pytest.mark.parametrize("case", NAMES)
    def test_dim_matches(self, case: str) -> None:
        int_man = CASES[case][0]
        assert _as_map(int_man).clique.dim == int_man.dim

    @pytest.mark.parametrize("case", NODE_NAMES)
    def test_tensor_matches_outer_product(self, case: str) -> None:
        m = _as_map(CASES[case][0])
        w, v = _stats(case, 0)
        nodes = (m.node_coords(0, w), m.node_coords(1, v))
        assert jnp.allclose(m.clique.tensor(*nodes), m.outer_product(w, v))

    @pytest.mark.parametrize("case", NODE_NAMES)
    def test_contract_to_codomain_matches_application(self, case: str) -> None:
        """``keep=0`` is the likelihood direction: contract the latent, land on x."""
        m = _as_map(CASES[case][0])
        _, v = _stats(case, 1)
        params = jax.random.normal(jax.random.PRNGKey(7), (m.dim,))
        contracted = m.clique.contract(params, 0, m.node_coords(1, v))
        assert jnp.allclose(m.amb_coords(0, contracted), m(params, v))

    @pytest.mark.parametrize("case", NODE_NAMES)
    def test_contract_to_domain_matches_transposed_application(self, case: str) -> None:
        """``keep=1`` is the posterior direction: contract x, land on the latent."""
        m = _as_map(CASES[case][0])
        w, _ = _stats(case, 2)
        params = jax.random.normal(jax.random.PRNGKey(8), (m.dim,))
        contracted = m.clique.contract(params, 1, m.node_coords(0, w))
        assert jnp.allclose(m.amb_coords(1, contracted), m.transpose_apply(params, w))


class TestSufficientStatistic:
    """The clique's contribution to a joint statistic is the tensor of its nodes'."""

    def test_matches_the_harmonium_interaction(self) -> None:
        """What ``Harmonium.sufficient_statistic`` computes for the cross partition."""
        fa = factor_analysis(obs_dim=5, lat_dim=2)
        clique = _as_map(fa.int_man).clique
        key_x, key_z = jax.random.split(jax.random.PRNGKey(3))
        x = jax.random.normal(key_x, (fa.obs_man.data_dim,))
        z = jax.random.normal(key_z, (fa.pst_man.data_dim,))

        joint = jnp.concatenate([x, z])
        _, int_stats, _ = fa.split_level(fa.sufficient_statistic(joint))
        s_x = fa.obs_man.sufficient_statistic(x)
        s_z = fa.pst_man.sufficient_statistic(z)
        assert jnp.allclose(clique.tensor(s_x, s_z), int_stats)


class TestJointDomainCliques:
    """A clique whose domain embedding addresses a *group* of nodes.

    MFA's $\\theta_{XY}$ and $\\theta_{XK}$ both couple $x$ to one node of the mixture
    above, reached through a ``CliqueEmbedding``. The clique reading contracts to that
    node's own coordinates; the map reading lands in the mixture's. They agree exactly
    after the embedding, which is what makes the two readings one object.
    """

    @staticmethod
    def _mfa():
        return MixtureOfFactorAnalyzers(
            n_categories=3, bas_hrm=factor_analysis(obs_dim=4, lat_dim=2)
        )

    @pytest.mark.parametrize(("index", "nodes"), [(0, (0, 1)), (2, (0, 2))])
    def test_nodes_and_arity_agree(self, index: int, nodes: tuple[int, ...]) -> None:
        placed, clique = self._mfa().cross_placements[index]
        assert placed == nodes
        assert clique.arity == 2

    @pytest.mark.parametrize("index", [0, 2])
    def test_posterior_direction_matches_after_the_embedding(self, index: int) -> None:
        mfa = self._mfa()
        m = _as_map(mfa.int_man.blocks[index])
        w = jax.random.normal(jax.random.PRNGKey(21), (mfa.obs_man.dim,))
        params = jax.random.normal(jax.random.PRNGKey(22), (m.dim,))
        live = m.project_domain(m.transpose_apply(params, w))
        node_w = m.node_coords(0, w)
        assert jnp.allclose(m.clique.partial_contract(params, (1,), node_w), live)


class TestJointBlocksAreNotProductsOfMarginals:
    """Why a multi-latent clique reads a *joint* statistic instead of per-node ones.

    ``tensor`` multiplies its nodes' statistics together, which is exact when every node is
    observed. When two nodes are latent the clique's parameters are
    $\\mathbb E[\\bigotimes_i \\mathbf s_i]$ jointly, and expectation does not pass through a
    tensor product. ``select_joint`` is the operation for that case: it contracts each
    embedding into an axis of the joint statistic and never forms a marginal.
    """

    @staticmethod
    def _mfa():
        from goal.models import MixtureOfFactorAnalyzers

        return MixtureOfFactorAnalyzers(
            n_categories=3, bas_hrm=factor_analysis(obs_dim=4, lat_dim=2)
        )

    def test_on_a_sample_point_the_latent_statistic_is_a_product(self) -> None:
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

    MFA stores this clique as a matrix whose domain embedding addresses the joint $(y,k)$
    statistic --- which is exactly "select a sub-statistic on the $y$ axis, identity on the
    $k$ axis" written as one matrix. These tests check that the clique reading of it, over
    three nodes, agrees with the map reading on every operation, *including in mean
    coordinates at the E-step*, which
    is the case that decides whether higher arity is usable at all.

    The clique's latent nodes $(y, k)$ are themselves a clique of the level above, so
    their joint expectation exists as a statistic to select from. That is the structural
    condition higher arity needs.
    """

    @staticmethod
    def _setup():
        from goal.models import MixtureOfFactorAnalyzers

        mfa = MixtureOfFactorAnalyzers(
            n_categories=3, bas_hrm=factor_analysis(obs_dim=4, lat_dim=2)
        )
        xyk = mfa.int_man.blocks[1]
        assert isinstance(xyk, Interaction)
        # x location, y location, k in full: the three nodes' embeddings.
        return mfa, xyk, xyk.clique

    def test_dimension_matches_the_live_map(self) -> None:
        _, xyk, clique = self._setup()
        assert clique.dim == xyk.dim
        assert clique.sub_dims == (4, 2, 2)

    def test_mean_parameters_match_at_the_e_step(self) -> None:
        """The decisive one: mean coordinates, both latent nodes dependent."""
        mfa, xyk, clique = self._setup()
        mix = mfa.pst_man
        params = mfa.initialize(jax.random.PRNGKey(0), shape=0.5)
        x = jax.random.normal(jax.random.PRNGKey(2), (mfa.obs_man.data_dim,))

        s_x = mfa.obs_man.sufficient_statistic(x)
        lat_means = mix.to_mean(mfa.posterior_at(params, x))
        _, m_yk, _ = mix.split_level(lat_means)

        live = xyk.outer_product(s_x, lat_means)
        rebuilt = jnp.outer(
            clique.node_embs[0].project(s_x), clique.select_joint((1, 2), m_yk)
        ).ravel()
        assert jnp.allclose(live, rebuilt)

    def test_mean_parameters_match_over_many_draws(self) -> None:
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
                clique.node_embs[0].project(s_x), clique.select_joint((1, 2), m_yk)
            ).ravel()
            assert jnp.allclose(live, rebuilt)

    def test_posterior_direction_matches(self) -> None:
        """Contract the observed node, leave $(y, k)$ joined: the posterior message."""
        mfa, xyk, clique = self._setup()
        params = mfa.initialize(jax.random.PRNGKey(3), shape=0.5)
        x = jax.random.normal(jax.random.PRNGKey(4), (mfa.obs_man.data_dim,))
        s_x = mfa.obs_man.sufficient_statistic(x)
        _, int_params, _ = mfa.split_level(params)
        xyk_params = mfa.int_man.coord_blocks(int_params)[1]

        live = xyk.project_domain(xyk.transpose_apply(xyk_params, s_x))
        rebuilt = clique.partial_contract(xyk_params, (1, 2), s_x)
        assert jnp.allclose(live, rebuilt)

    def test_likelihood_direction_matches(self) -> None:
        """Contract both latent nodes, land on the observable."""
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
        rebuilt = clique.node_embs[0].embed(
            clique.partial_contract(xyk_params, (0,), s_y, s_k)
        )
        assert jnp.allclose(live, rebuilt)
