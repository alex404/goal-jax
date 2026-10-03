"""Tests for ``CrossMap`` in geometry/manifold/clique.py.

(``tests/clique.py`` tests ``geometry/algebra/clique.py`` and ``tests/graphical.py`` the
layouts and the bare clique-map algebra; this file is the clique map paired with its clique
embeddings.)

A ``CliqueMap`` is a linear map between its node groups, and a ``CrossMap`` is a sum
of clique maps, each conjugated by a pair of clique embeddings. Each case takes a live model's interaction and checks that the
clique operations reproduce the interaction operations once the clique embeddings are applied: the
outer product that builds a sufficient statistic, the application that builds a likelihood,
and the transposed application that builds a posterior. If those hold for every interaction
shape in the library, the arity-2 case is settled and only arity 3 is new.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    CliqueMap,
    CrossMap,
    Diagonal,
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


def _as_map(int_man: LinearMap[Any, Any]) -> CrossMap[Any, Any]:
    """A live interaction, which is one conditional reading of a clique."""
    assert isinstance(int_man, CrossMap)
    return int_man


def _block(int_man: LinearMap[Any, Any], index: int) -> CrossMap[Any, Any]:
    """One term of a multi-form interaction, as an interaction of the same shape."""
    m = _as_map(int_man)
    return CrossMap(m.cod_man, m.dom_man, m.cod_group, m.dom_group, (m.terms[index],))


def _clq_map(int_man: LinearMap[Any, Any]) -> CliqueMap:
    """The clique map of a single-term interaction."""
    ((_, clq_map),) = _as_map(int_man).terms
    return clq_map


def _cod_node(m: CrossMap[Any, Any], w: Array) -> Array:
    """A codomain point taken down to the node the single form's output couples."""
    return m.clq_embs(m.terms[0][0])[0].project(w)


def _dom_node(m: CrossMap[Any, Any], v: Array) -> Array:
    """A domain point taken down to the node group the single form contracts."""
    return m.clq_embs(m.terms[0][0])[1].project(v)


def _cod_amb(m: CrossMap[Any, Any], w_node: Array) -> Array:
    """The single form's output, placed back where the caller holds it."""
    return m.clq_embs(m.terms[0][0])[0].embed(w_node)


def _dom_amb(m: CrossMap[Any, Any], v_node: Array) -> Array:
    """The single form's contracted-side node coordinates, placed back."""
    return m.clq_embs(m.terms[0][0])[1].embed(v_node)


def _case(m: CrossMap[Any, Any]) -> tuple[CrossMap[Any, Any], Manifold, Manifold]:
    """A cross map with its codomain and domain."""
    return m, m.cod_man, m.dom_man


def _interactions() -> dict[str, tuple[LinearMap[Any, Any], Manifold, Manifold]]:
    """Every distinct cross-map shape in the library, with its codomain and domain."""
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
        "factor_analysis": _case(fa.crs_man),
        "mixture": _case(mix.crs_man),
        "hmog": _case(hmog.crs_man),
        "cca_fst": _case(_block(cca.crs_man, 0)),
        "cca_snd": _case(_block(cca.crs_man, 1)),
    }
    for name, index in (("mfa_xy", 0), ("mfa_xyk", 1), ("mfa_xk", 2)):
        cases[name] = _case(_block(mfa.crs_man, index))
    return cases


CASES = _interactions()
NAMES = sorted(CASES)
NODE_NAMES = [name for name in NAMES if len(_clq_map(CASES[name][0]).embs) == 2]
"""Cases contracting a single node.

There the clique reading and the interaction reading coincide up to the clique embeddings:
contracting one axis leaves one axis, and a single node can be placed back through its own
clique embedding. At arity 3 the contracted side is a *group*, no single axis can be placed
through a joint clique embedding, and the two readings meet at the node group's joint coordinates --- which is what
``TestArityThreeReproducesMFA`` checks instead.

The criterion is arity, not what the clique embedding happens to be: one that is a
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
        assert _clq_map(int_man).dim == int_man.dim

    @pytest.mark.parametrize("case", NODE_NAMES)
    def test_outer_product_matches(self, case: str) -> None:
        m = _as_map(CASES[case][0])
        w, v = _stats(case, 0)
        assert jnp.allclose(
            _clq_map(m).outer_product(_cod_node(m, w), _dom_node(m, v)),
            m.outer_product(w, v),
        )

    @pytest.mark.parametrize("case", NODE_NAMES)
    def test_application_matches(self, case: str) -> None:
        """The forward reading is the likelihood direction: contract the latent, land on x."""
        m = _as_map(CASES[case][0])
        _, v = _stats(case, 1)
        params = jax.random.normal(jax.random.PRNGKey(7), (m.dim,))
        node_out = _clq_map(m)(params, _dom_node(m, v))
        assert jnp.allclose(_cod_amb(m, node_out), m(params, v))

    @pytest.mark.parametrize("case", NODE_NAMES)
    def test_transposed_application_matches(self, case: str) -> None:
        """The transposed reading is the posterior direction: contract x, land on the latent."""
        m = _as_map(CASES[case][0])
        w, _ = _stats(case, 2)
        params = jax.random.normal(jax.random.PRNGKey(8), (m.dim,))
        trn = _clq_map(m).trn_man
        node_out = trn(_clq_map(m).transpose(params), _cod_node(m, w))
        assert jnp.allclose(_dom_amb(m, node_out), m.transpose_apply(params, w))


class TestSufficientStatistic:
    """The clique's contribution to a joint statistic is the tensor of its nodes'."""

    def test_matches_the_harmonium_interaction(self) -> None:
        """What ``Harmonium.sufficient_statistic`` computes for the cross partition."""
        fa = factor_analysis(obs_dim=5, lat_dim=2)
        clique = _clq_map(fa.crs_man)
        key_x, key_z = jax.random.split(jax.random.PRNGKey(3))
        x = jax.random.normal(key_x, (fa.obs_man.data_dim,))
        z = jax.random.normal(key_z, (fa.pst_man.data_dim,))

        joint = jnp.concatenate([x, z])
        _, int_stats, _ = fa.split_level(fa.sufficient_statistic(joint))
        s_x = fa.obs_man.sufficient_statistic(x)
        s_z = fa.pst_man.sufficient_statistic(z)
        assert jnp.allclose(clique.outer_product(s_x, s_z), int_stats)


class TestJointDomainCliques:
    """A clique whose domain clique embedding locates a node inside a *layout*.

    MFA's $\\theta_{XY}$ and $\\theta_{XK}$ both couple $x$ to one node of the mixture
    above, whose block a ``CliqueEmbedding`` locates. The clique reading works in that node's
    own coordinates; the interaction reading lands in the mixture's. They agree exactly at
    the node, which is what makes the two readings one object.
    """

    @staticmethod
    def _mfa():
        return MixtureOfFactorAnalyzers(
            n_categories=3, bas_hrm=factor_analysis(obs_dim=4, lat_dim=2)
        )

    @pytest.mark.parametrize(("index", "nodes"), [(0, (0, 1)), (2, (0, 2))])
    def test_nodes_and_arity_agree(self, index: int, nodes: tuple[int, ...]) -> None:
        mfa = self._mfa()
        assert mfa.level_split()[1][index] == nodes
        assert len(mfa.crs_man.terms[index][1].embs) == 2

    @pytest.mark.parametrize("index", [0, 2])
    def test_posterior_direction_matches_at_the_node(self, index: int) -> None:
        mfa = self._mfa()
        m = _block(mfa.crs_man, index)
        w = jax.random.normal(jax.random.PRNGKey(21), (mfa.obs_man.dim,))
        params = jax.random.normal(jax.random.PRNGKey(22), (m.dim,))
        live = _dom_node(m, m.transpose_apply(params, w))
        clq_map = _clq_map(m)
        rebuilt = clq_map.trn_man(clq_map.transpose(params), _cod_node(m, w))
        assert jnp.allclose(rebuilt, live)


class TestJointBlocksAreNotProductsOfMarginals:
    """Why a multi-latent clique reads a *joint* statistic instead of per-node ones.

    Multiplying the nodes' statistics together is exact when every node is observed. When
    two nodes are latent the clique's parameters are $\\mathbb E[\\bigotimes_i \\mathbf s_i]$
    jointly, and expectation does not pass through a tensor product. ``CliqueMap.outer_product`` is
    the operation for that case: it restricts each embedding along an axis of the joint
    statistic and never forms a marginal.
    """

    @staticmethod
    def _mfa():
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

        This is the trap: multiplying the marginals here would be silently wrong, not
        obviously so.
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

    The clique contracts the joint $(y,k)$ statistic of the mixture above, located by
    a ``CliqueEmbedding`` --- "select a sub-statistic on the $y$ axis, identity on the $k$
    axis". These tests check that the clique reading of it, over three nodes, agrees with
    the interaction reading on every operation, *including in mean coordinates at the
    E-step*, which is the case that decides whether higher arity is usable at all.

    The clique's latent nodes $(y, k)$ are themselves a clique of the level above, so
    their joint expectation exists as a statistic to select from. That is the structural
    condition higher arity needs.
    """

    @staticmethod
    def _setup():
        mfa = MixtureOfFactorAnalyzers(
            n_categories=3, bas_hrm=factor_analysis(obs_dim=4, lat_dim=2)
        )
        xyk = _block(mfa.crs_man, 1)
        # x location, y location, k in full: the three nodes' embeddings.
        return mfa, xyk, _clq_map(xyk)

    def test_dimension_matches_the_live_map(self) -> None:
        _, xyk, clique = self._setup()
        assert clique.dim == xyk.dim
        assert tuple(emb.sub_man.dim for emb in clique.embs) == (4, 2, 2)

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
        rebuilt = clique.outer_product(s_x, m_yk)
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
            rebuilt = clique.outer_product(s_x, m_yk)
            assert jnp.allclose(live, rebuilt)

    def test_posterior_direction_matches(self) -> None:
        """Contract the observed node, leave $(y, k)$ joined: the posterior message."""
        mfa, xyk, clique = self._setup()
        params = mfa.initialize(jax.random.PRNGKey(3), shape=0.5)
        x = jax.random.normal(jax.random.PRNGKey(4), (mfa.obs_man.data_dim,))
        s_x = mfa.obs_man.sufficient_statistic(x)
        _, int_params, _ = mfa.split_level(params)
        xyk_params = mfa.crs_man.coord_blocks(int_params)[1]

        live = _dom_node(xyk, xyk.transpose_apply(xyk_params, s_x))
        rebuilt = clique.trn_man(clique.transpose(xyk_params), s_x)
        assert jnp.allclose(live, rebuilt)

    def test_likelihood_direction_matches(self) -> None:
        """Contract both latent nodes, land on the observable.

        At a sample point the joint statistic *is* the product of the two nodes', so the
        map reading through the clique embedding and the direct product agree --- the case
        ``TestJointBlocksAreNotProductsOfMarginals`` shows fails in mean coordinates.
        """
        mfa, xyk, clique = self._setup()
        mix = mfa.pst_man
        params = mfa.initialize(jax.random.PRNGKey(5), shape=0.5)
        _, int_params, _ = mfa.split_level(params)
        xyk_params = mfa.crs_man.coord_blocks(int_params)[1]

        y = jax.random.normal(jax.random.PRNGKey(6), (mix.obs_man.data_dim,))
        z = jnp.concatenate([y, jnp.array([1.0])])
        s_y = mix.obs_man.sufficient_statistic(y)
        s_k = mix.lat_man.sufficient_statistic(jnp.array([1.0]))

        live = xyk(xyk_params, mix.sufficient_statistic(z))
        rebuilt = clique(xyk_params, jnp.outer(s_y, s_k).ravel())
        assert jnp.allclose(live, rebuilt)
