"""Tests for models/base/gaussian/boltzmann.py.

Covers Boltzmann, DiagonalBoltzmann, BoltzmannEmbedding, and ChordalBoltzmann.
Verifies sufficient statistics, density normalization, split/join round-trips,
unit conditional probabilities, embedding operations, and (for ChordalBoltzmann)
junction-tree construction plus exact log-partition and sampling against
brute-force enumeration.
"""

import jax
import jax.numpy as jnp
import pytest

from goal.geometry import Symmetric
from goal.models import (
    Bernoullis,
    BoltzmannEmbedding,
    ChainBoltzmann,
    ChainTree,
    ChordalBoltzmann,
    DiagonalBoltzmann,
    FullBoltzmann,
    JunctionTree,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-5
ATOL = 1e-7


class TestBoltzmann:
    """Test Boltzmann (full coupling) distribution."""

    @pytest.mark.parametrize("n", [3, 4])
    def test_dimensions(self, n: int) -> None:
        model = FullBoltzmann(n)
        assert model.dim == n * (n + 1) // 2
        assert model.data_dim == n

    def test_sufficient_statistic(self) -> None:
        model = FullBoltzmann(4)
        state = jax.random.bernoulli(jax.random.PRNGKey(42), 0.5, (4,))
        suff_stat = model.sufficient_statistic(state)
        expected = Symmetric().from_matrix(jnp.outer(state, state))
        assert jnp.allclose(suff_stat, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("n", [3, 4])
    def test_density_normalizes(self, n: int) -> None:
        """Densities sum to 1 over all binary states."""
        model = FullBoltzmann(n)
        params = jax.random.uniform(jax.random.PRNGKey(42), (model.dim,), minval=-2.0, maxval=2.0)
        densities = jax.vmap(lambda s: model.density(params, s))(model.states)
        assert jnp.allclose(jnp.sum(densities), 1.0, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("n", [3, 4])
    def test_log_partition_matches_sum(self, n: int) -> None:
        """Log partition equals log sum over all states."""
        model = FullBoltzmann(n)
        params = jax.random.uniform(jax.random.PRNGKey(42), (model.dim,), minval=-2.0, maxval=2.0)
        log_z = model.log_partition_function(params)
        energies = jax.vmap(lambda s: jnp.dot(params, model.sufficient_statistic(s)))(model.states)
        assert jnp.allclose(log_z, jax.scipy.special.logsumexp(energies), rtol=RTOL, atol=ATOL)

    def test_location_precision_energy_identity(self) -> None:
        _assert_location_precision_identity(FullBoltzmann(4), jax.random.PRNGKey(31))

    def test_split_join_round_trip(self) -> None:
        model = FullBoltzmann(4)
        params = jax.random.uniform(jax.random.PRNGKey(42), (model.dim,), minval=-2.0, maxval=2.0)
        loc, prec = model.split_location_precision(params)
        recovered = model.join_location_precision(loc, prec)
        assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_loc_man_is_bernoullis(self) -> None:
        model = FullBoltzmann(4)
        assert isinstance(model.loc_man, Bernoullis)
        assert model.loc_man.n_neurons == 4

    def test_unit_conditional_prob(self) -> None:
        """unit_conditional_prob matches direct energy-difference computation."""
        model = FullBoltzmann(4)
        coupling = model.shp_man
        key = jax.random.PRNGKey(42)
        params = jax.random.uniform(key, (model.dim,), minval=-2.0, maxval=2.0)

        for i in range(3):
            key, state_key, unit_key = jax.random.split(jax.random.fold_in(key, i), 3)
            state = jax.random.bernoulli(state_key, 0.5, (4,)).astype(jnp.float32)
            unit_idx = jax.random.randint(unit_key, (), 0, 4)

            prob_fast = coupling.unit_conditional_prob(state, unit_idx, params)

            state_0 = state.at[unit_idx].set(0.0)
            state_1 = state.at[unit_idx].set(1.0)
            energy_0 = jnp.dot(params, coupling.sufficient_statistic(state_0))
            energy_1 = jnp.dot(params, coupling.sufficient_statistic(state_1))
            prob_direct = jax.nn.sigmoid(energy_1 - energy_0)

            assert jnp.allclose(prob_fast, prob_direct, rtol=RTOL, atol=ATOL)


    def test_to_mean_via_autodiff(self) -> None:
        """to_mean (autodiff of log_partition) matches stochastic_to_mean (Gibbs sampling)."""
        model = FullBoltzmann(3)
        params = jax.random.uniform(jax.random.PRNGKey(42), (model.dim,), minval=-1.0, maxval=1.0)
        autodiff_means = model.to_mean(params)
        stochastic_means = model.stochastic_to_mean(jax.random.PRNGKey(7), params, 50_000)
        assert jnp.allclose(autodiff_means, stochastic_means, atol=0.05)

    def test_gibbs_sampling(self) -> None:
        """Gibbs samples produce correct equilibrium distribution."""
        model = FullBoltzmann(3)
        params = jax.random.uniform(jax.random.PRNGKey(42), (model.dim,), minval=-1.0, maxval=1.0)
        samples = model.sample(jax.random.PRNGKey(0), params, 50_000)
        assert samples.shape == (50_000, model.data_dim)
        empirical_means = model.average_sufficient_statistic(samples)
        theoretical_means = model.to_mean(params)
        assert jnp.allclose(empirical_means, theoretical_means, atol=0.05)


class TestDiagonalBoltzmann:
    """Test DiagonalBoltzmann (independent binary units)."""

    @pytest.mark.parametrize("n", [2, 5])
    def test_dimensions(self, n: int) -> None:
        model = DiagonalBoltzmann(n)
        assert model.dim == n
        assert model.data_dim == n

    def test_generalized_gaussian_interface(self) -> None:
        model = DiagonalBoltzmann(3)
        assert isinstance(model.loc_man, Bernoullis)
        assert isinstance(model.shp_man, Bernoullis)
        assert model.loc_man.n_neurons == 3

    @pytest.mark.parametrize("n", [2, 5])
    def test_split_join_location_precision(self, n: int) -> None:
        model = DiagonalBoltzmann(n)
        params = jax.random.normal(jax.random.PRNGKey(42), (n,))
        loc, prec = model.split_location_precision(params)
        recovered = model.join_location_precision(loc, prec)
        assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_location_precision_energy_identity(self) -> None:
        _assert_location_precision_identity(DiagonalBoltzmann(4), jax.random.PRNGKey(32))

    def test_split_join_mean_second_moment(self) -> None:
        """For independent binary, first and second moments are identical."""
        model = DiagonalBoltzmann(3)
        means = jax.random.uniform(jax.random.PRNGKey(42), (3,), minval=0.1, maxval=0.9)
        first, second = model.split_mean_second_moment(means)
        assert jnp.allclose(first, means, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(second, means, rtol=RTOL, atol=ATOL)
        recovered = model.join_mean_second_moment(first, second)
        assert jnp.allclose(recovered, means, rtol=RTOL, atol=ATOL)


class TestBoltzmannEmbedding:
    """Test BoltzmannEmbedding (DiagonalBoltzmann -> Boltzmann)."""

    @pytest.mark.parametrize("n", [2, 4])
    def test_manifolds(self, n: int) -> None:
        emb = BoltzmannEmbedding(DiagonalBoltzmann(n), FullBoltzmann(n))
        assert emb.sub_man.dim == n
        assert emb.amb_man.dim == n * (n + 1) // 2

    def test_embed_zero_coupling(self) -> None:
        """Embedding creates zero off-diagonal coupling."""
        n = 3
        emb = BoltzmannEmbedding(DiagonalBoltzmann(n), FullBoltzmann(n))
        biases = jax.random.normal(jax.random.PRNGKey(42), (n,))
        boltz_params = emb.embed(biases)
        full_matrix = emb.amb_man.shp_man.to_matrix(boltz_params)
        mask = ~jnp.eye(n, dtype=bool)
        assert jnp.allclose(full_matrix[mask], 0.0, rtol=RTOL, atol=ATOL)

    def test_project_embed_round_trip(self) -> None:
        """project(embed(natural)) recovers original in mean coordinates."""
        n = 3
        emb = BoltzmannEmbedding(DiagonalBoltzmann(n), FullBoltzmann(n))
        probs = jax.random.uniform(jax.random.PRNGKey(42), (n,), minval=0.1, maxval=0.9)
        natural_params = emb.sub_man.to_natural(probs)
        boltz_natural = emb.embed(natural_params)
        boltz_means = emb.amb_man.to_mean(boltz_natural)
        recovered_means = emb.project(boltz_means)
        assert jnp.allclose(probs, recovered_means, rtol=RTOL, atol=ATOL)

    def test_translate(self) -> None:
        """Translate correctly adds delta to location/bias."""
        n = 3
        emb = BoltzmannEmbedding(DiagonalBoltzmann(n), FullBoltzmann(n))
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        biases = jax.random.normal(key1, (n,))
        delta = jax.random.normal(key2, (n,))
        boltz_params = emb.embed(biases)
        translated = emb.translate(boltz_params, delta)
        expected = emb.embed(biases + delta)
        assert jnp.allclose(translated, expected, rtol=RTOL, atol=ATOL)


def _assert_location_precision_identity(
    model: DiagonalBoltzmann | FullBoltzmann | ChordalBoltzmann, key: jnp.ndarray
) -> None:
    """Pin the (location, precision) convention semantically.

    Round-trip tests cannot catch a consistent sign or factor-of-two error in
    the precision scaling (it survives split-then-join). This asserts the
    Gaussian-form energy identity on binary states,

        theta . s(x) == location . x - (1/2) x^T Lambda x,

    where Lambda has the precision coordinates' diagonal entries on the
    diagonal and each off-diagonal coordinate symmetrically at (i, j) and
    (j, i). The quadratic form is derived independently of the conversion
    under test: for binary x, prec . s(x) counts the diagonal once and each
    off-diagonal pair once, so x^T Lambda x = 2 prec . s(x) - diag(prec) . x.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    xs = jax.random.bernoulli(k1, 0.5, (8, model.data_dim)).astype(jnp.float64)

    def quad(prec: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        pd, _ = model.split_couplings(prec)
        return 2.0 * jnp.dot(prec, model.sufficient_statistic(x)) - jnp.dot(pd, x)

    params = jax.random.normal(k2, (model.dim,))
    loc0, prec0 = model.split_location_precision(params)
    location = jax.random.normal(k3, (model.data_dim,))
    precision = jax.random.normal(k4, (model.dim,))
    joined = model.join_location_precision(location, precision)
    for x in xs:
        s = model.sufficient_statistic(x)
        assert jnp.allclose(
            jnp.dot(params, s),
            jnp.dot(loc0, x) - 0.5 * quad(prec0, x),
            rtol=RTOL,
            atol=ATOL,
        )
        assert jnp.allclose(
            jnp.dot(joined, s),
            jnp.dot(location, x) - 0.5 * quad(precision, x),
            rtol=RTOL,
            atol=ATOL,
        )


def _bf_log_partition(
    n: int, jt: JunctionTree, params: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Brute-force log partition via enumeration over all $2^n$ binary states.

    Returns ``(log_z, all_states, energies)`` so callers can reuse the enumerated
    states and per-state energies for marginal / density tests.
    """
    diag = params[:n]
    off = params[n:]
    edges = jt.chordal_edges
    all_states = jnp.array(
        [[(s >> i) & 1 for i in range(n)] for s in range(1 << n)],
        dtype=jnp.float64,
    )

    def energy(x: jnp.ndarray) -> jnp.ndarray:
        e = jnp.sum(diag * x)
        for k, (i, j) in enumerate(edges):
            e = e + off[k] * x[i] * x[j]
        return e

    energies = jax.vmap(energy)(all_states)
    return jax.scipy.special.logsumexp(energies), all_states, energies


class TestJunctionTree:
    """Junction-tree topology construction on small hand-checked graphs."""

    def test_chain(self) -> None:
        jt = JunctionTree.from_edges(4, [(0, 1), (1, 2), (2, 3)])
        assert jt.treewidth == 1
        assert jt.max_clique_size == 2
        assert set(jt.cliques) == {(0, 1), (1, 2), (2, 3)}
        assert len(jt.tree_edges) == 2

    def test_4cycle_triangulates(self) -> None:
        jt = JunctionTree.from_edges(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
        # Must triangulate (4-cycle is not chordal); treewidth 2.
        assert jt.treewidth == 2
        assert len(jt.cliques) == 2
        # Chordal completion adds exactly one diagonal of the cycle.
        assert len(jt.chordal_edges) == 5

    def test_k4_one_clique(self) -> None:
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        jt = JunctionTree.from_edges(4, edges)
        assert jt.cliques == ((0, 1, 2, 3),)
        assert jt.treewidth == 3
        assert jt.tree_edges == ()

    def test_max_treewidth_raises(self) -> None:
        # K4 has treewidth 3; requesting max_treewidth=2 should raise.
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        with pytest.raises(ValueError, match="treewidth"):
            JunctionTree.from_edges(4, edges, max_treewidth=2)

    def test_disconnected_graph_links_to_single_tree(self) -> None:
        """A disconnected graph's clique forest is linked into one spanning tree
        via empty-separator edges, so collect reaches every clique."""
        jt = JunctionTree.from_edges(5, [(0, 1), (3, 4)])  # two edges + isolated node 2
        assert set(jt.cliques) == {(0, 1), (2,), (3, 4)}
        assert len(jt.tree_edges) == jt.n_cliques - 1
        assert len(jt.collect_order) == jt.n_cliques - 1
        # Every clique is reachable from the root.
        assert set(jt.pre_order) == set(range(jt.n_cliques))

    def test_each_param_owned_once(self) -> None:
        """Every node and every chordal edge is attributed to exactly one clique."""
        jt = JunctionTree.from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)])
        # All bias owners are valid clique indices in [0, n_cliques).
        assert all(0 <= o < jt.n_cliques for o in jt.bias_owner)
        assert all(0 <= o < jt.n_cliques for o in jt.edge_owner)
        # Each bias is in its owning clique.
        for v, owner in enumerate(jt.bias_owner):
            assert v in jt.cliques[owner]
        # Each edge is in its owning clique.
        for k, (i, j) in enumerate(jt.chordal_edges):
            clique = jt.cliques[jt.edge_owner[k]]
            assert i in clique and j in clique


class TestChordalBoltzmann:
    """Chordal-pattern Boltzmann: junction-tree log partition, sampling, gradients."""

    @pytest.mark.parametrize(
        "n,edges",
        [
            (4, [(0, 1), (1, 2), (2, 3)]),  # chain (tw 1)
            (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),  # 4-cycle (tw 2)
            (5, [(0, 1), (1, 2), (2, 3), (3, 4)]),  # chain 5
            (5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]),  # 5-cycle
        ],
    )
    def test_dimensions(self, n: int, edges: list[tuple[int, int]]) -> None:
        cb = ChordalBoltzmann.from_edges(n, edges)
        assert cb.n_neurons == n
        assert cb.data_dim == n
        assert cb.dim == n + cb.junction_tree.n_chordal_edges

    @pytest.mark.parametrize(
        "n,edges",
        [
            (4, [(0, 1), (1, 2), (2, 3)]),
            (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
            (5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]),
            (6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)]),
            (3, []),  # fully isolated (log Z must sum over components)
            (5, [(0, 1), (3, 4)]),  # two components + isolated node
            (8, [(i, i + k) for k in (1, 2) for i in range(8 - k)]),  # band-2 (tw 2 path)
            (5, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]),  # heterogeneous separators
            (5, [(0, k) for k in range(1, 5)]),  # hub (clique tree may branch)
        ],
    )
    def test_log_partition_matches_enumeration(
        self, n: int, edges: list[tuple[int, int]]
    ) -> None:
        cb = ChordalBoltzmann.from_edges(n, edges)
        params = jax.random.uniform(
            jax.random.PRNGKey(42), (cb.dim,), minval=-1.5, maxval=1.5
        )
        log_z_jt = cb.log_partition_function(params)
        log_z_bf, _, _ = _bf_log_partition(n, cb.junction_tree, params)
        assert jnp.allclose(log_z_jt, log_z_bf, rtol=RTOL, atol=ATOL)

    def test_density_normalizes(self) -> None:
        """Density over all states sums to 1 (4-cycle, treewidth 2)."""
        cb = ChordalBoltzmann.from_edges(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
        params = jax.random.uniform(
            jax.random.PRNGKey(42), (cb.dim,), minval=-1.0, maxval=1.0
        )
        log_z, _all_states, energies = _bf_log_partition(4, cb.junction_tree, params)
        densities = jnp.exp(energies - log_z)
        assert jnp.allclose(jnp.sum(densities), 1.0, rtol=RTOL, atol=ATOL)

    def test_reduces_to_full_boltzmann_on_k4(self) -> None:
        """A fully-connected chordal graph gives the same log Z as the original Boltzmann."""
        n = 4
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        cb = ChordalBoltzmann.from_edges(n, edges)
        b_full = FullBoltzmann(n)
        assert cb.dim == b_full.dim

        params_c = jax.random.uniform(
            jax.random.PRNGKey(7), (cb.dim,), minval=-1.0, maxval=1.0
        )
        # Convert chordal-layout params to the upper-triangular layout used by Boltzmann.
        mat = cb.shp_man.to_matrix(params_c)
        params_b = b_full.shp_man.rep.from_matrix(mat)
        log_z_c = cb.log_partition_function(params_c)
        log_z_b = b_full.log_partition_function(params_b)
        assert jnp.allclose(log_z_c, log_z_b, rtol=RTOL, atol=ATOL)

    def test_to_mean_via_autodiff(self) -> None:
        """to_mean (autodiff of log_partition) matches empirical sufficient statistics."""
        cb = ChordalBoltzmann.from_edges(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
        params = jax.random.uniform(
            jax.random.PRNGKey(42), (cb.dim,), minval=-1.0, maxval=1.0
        )
        autodiff_means = cb.to_mean(params)
        samples = cb.sample(jax.random.PRNGKey(0), params, n=20_000)
        empirical_means = cb.average_sufficient_statistic(samples)
        assert jnp.allclose(autodiff_means, empirical_means, atol=0.03)

    def test_exact_sampling_matches_enumeration(self) -> None:
        """Empirical first moments from JT sampling match exact marginals from enumeration."""
        n = 5
        cb = ChordalBoltzmann.from_edges(n, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])
        params = jax.random.uniform(
            jax.random.PRNGKey(13), (cb.dim,), minval=-1.0, maxval=1.0
        )
        samples = cb.sample(jax.random.PRNGKey(1), params, n=20_000)
        empirical_first = jnp.mean(samples, axis=0)
        # Exact marginals from full enumeration.
        log_z, all_states, energies = _bf_log_partition(n, cb.junction_tree, params)
        probs = jnp.exp(energies - log_z)
        exact_first = probs @ all_states
        assert jnp.allclose(empirical_first, exact_first, atol=0.02)

    def test_disconnected_sampling_matches_marginals(self) -> None:
        """Ancestral sampling must visit every component of a disconnected graph."""
        n = 5
        cb = ChordalBoltzmann.from_edges(n, [(0, 1), (3, 4)])
        params = jax.random.uniform(
            jax.random.PRNGKey(3), (cb.dim,), minval=-1.0, maxval=1.0
        )
        samples = cb.sample(jax.random.PRNGKey(2), params, n=20_000)
        empirical_first = jnp.mean(samples, axis=0)
        log_z, all_states, energies = _bf_log_partition(n, cb.junction_tree, params)
        probs = jnp.exp(energies - log_z)
        exact_first = probs @ all_states
        assert jnp.allclose(empirical_first, exact_first, atol=0.02)

    @pytest.mark.parametrize(
        "n,edges",
        [
            (5, [(0, k) for k in range(1, 5)]),  # hub: branching clique tree
            (5, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]),  # mixed seps
        ],
    )
    def test_sampling_matches_joint_distribution(
        self, n: int, edges: list[tuple[int, int]]
    ) -> None:
        """Empirical state frequencies match the exact joint over all $2^n$ states.

        Stronger than the moment checks: sufficient statistics only pin the
        chordal-edge moments, while the histogram certifies the entire joint
        --- non-adjacent correlations and higher-order structure included."""
        cb = ChordalBoltzmann.from_edges(n, edges)
        params = jax.random.uniform(
            jax.random.PRNGKey(21), (cb.dim,), minval=-1.0, maxval=1.0
        )
        n_samples = 50_000
        samples = cb.sample(jax.random.PRNGKey(22), params, n=n_samples)
        # LSB-first packing matches the state order of _bf_log_partition.
        codes = jnp.sum(samples.astype(jnp.int32) << jnp.arange(n), axis=1)
        freqs = jnp.bincount(codes, length=1 << n) / n_samples
        log_z, _all_states, energies = _bf_log_partition(n, cb.junction_tree, params)
        probs = jnp.exp(energies - log_z)
        assert jnp.allclose(freqs, probs, atol=0.01)

    def test_split_join_location_precision_round_trip(self) -> None:
        cb = ChordalBoltzmann.from_edges(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
        params = jax.random.uniform(
            jax.random.PRNGKey(42), (cb.dim,), minval=-1.5, maxval=1.5
        )
        loc, prec = cb.split_location_precision(params)
        assert jnp.allclose(loc, jnp.zeros(cb.n_neurons))
        recovered = cb.join_location_precision(loc, prec)
        assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_location_precision_energy_identity(self) -> None:
        cb = ChordalBoltzmann.from_edges(
            5, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]
        )
        _assert_location_precision_identity(cb, jax.random.PRNGKey(33))

    def test_to_from_matrix_round_trip(self) -> None:
        """to_matrix/from_matrix round-trip on the chordal pattern; dense
        matrices are symmetrized and off-pattern entries dropped."""
        cb = ChordalBoltzmann.from_edges(
            5, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]
        )
        cm = cb.shp_man
        params = jax.random.normal(jax.random.PRNGKey(7), (cb.dim,))
        mat = cm.to_matrix(params)
        assert jnp.allclose(mat, mat.T)
        assert jnp.allclose(cm.from_matrix(mat), params, rtol=RTOL, atol=ATOL)
        # Off-pattern entries are dropped: (0, 4) is not a chordal edge here.
        assert (0, 4) not in cb.junction_tree.chordal_edges
        perturbed = mat.at[0, 4].add(3.0).at[4, 0].add(3.0)
        assert jnp.allclose(cm.from_matrix(perturbed), params, rtol=RTOL, atol=ATOL)
        # Asymmetric input: off-diagonal couplings read the symmetrized average.
        skewed = mat.at[0, 1].add(1.0).at[1, 0].add(-1.0)
        assert jnp.allclose(cm.from_matrix(skewed), params, rtol=RTOL, atol=ATOL)

    def test_split_join_mean_second_moment(self) -> None:
        cb = ChordalBoltzmann.from_edges(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
        params = jax.random.uniform(
            jax.random.PRNGKey(42), (cb.dim,), minval=-1.0, maxval=1.0
        )
        means = cb.to_mean(params)
        first, second = cb.split_mean_second_moment(means)
        # First moment is the diagonal slice (E[x_i] = diagonal of moment matrix).
        assert jnp.allclose(first, means[: cb.n_neurons])
        joined = cb.join_mean_second_moment(first, second)
        assert jnp.allclose(means, joined, rtol=RTOL, atol=ATOL)

    def test_loc_man_is_bernoullis(self) -> None:
        cb = ChordalBoltzmann.from_edges(4, [(0, 1), (1, 2), (2, 3)])
        assert isinstance(cb.loc_man, Bernoullis)
        assert cb.loc_man.n_neurons == 4

    def test_long_chain_log_partition_matches_dp(self) -> None:
        """At scale, log_partition_function on a chain matches a direct dynamic-programming forward sweep.

        Exercises the ``lax.scan`` path at a non-trivial size (64 nodes, 63
        cliques) where Python-loop unrolling would compile much more slowly.
        """
        n = 64
        edges = [(i, i + 1) for i in range(n - 1)]
        cb = ChordalBoltzmann.from_edges(n, edges)
        key = jax.random.PRNGKey(101)
        params = jax.random.uniform(key, (cb.dim,), minval=-1.0, maxval=1.0)

        diag = params[:n]
        off = params[n:]
        # Chain dynamic-programming reference: forward sweep over x_0 .. x_{n-1}.
        # State vector indexed by x_i \in {0, 1}; messages accumulate log
        # potentials node by node.
        msg = jnp.array([0.0, diag[0]])  # x_0 = 0 contributes 0; x_0 = 1 contributes diag[0]
        for i in range(1, n):
            # Couple x_{i-1} (msg axis) to x_i via theta_{i-1,i} * x_{i-1}*x_i.
            theta = off[i - 1]
            # next_msg[x_i] = logsumexp over x_{i-1} of msg[x_{i-1}] + diag[i]*x_i + theta*x_{i-1}*x_i
            m00 = msg[0]  # x_{i-1}=0, x_i=0
            m10 = msg[1]  # x_{i-1}=1, x_i=0
            m01 = msg[0] + diag[i]  # x_{i-1}=0, x_i=1
            m11 = msg[1] + diag[i] + theta  # x_{i-1}=1, x_i=1
            next0 = jax.scipy.special.logsumexp(jnp.array([m00, m10]))
            next1 = jax.scipy.special.logsumexp(jnp.array([m01, m11]))
            msg = jnp.array([next0, next1])
        log_z_ref = jax.scipy.special.logsumexp(msg)
        log_z_jt = cb.log_partition_function(params)
        assert jnp.allclose(log_z_jt, log_z_ref, rtol=RTOL, atol=ATOL)

    def test_long_chain_sampling_matches_marginals(self) -> None:
        """JT sampling on a 64-node chain reproduces per-node mean activations.

        Cross-checks the walk-down ``fori_loop`` at scale by comparing empirical
        first moments to ``to_mean`` (autodiff of log_partition_function).
        """
        n = 64
        edges = [(i, i + 1) for i in range(n - 1)]
        cb = ChordalBoltzmann.from_edges(n, edges)
        params = jax.random.uniform(
            jax.random.PRNGKey(7), (cb.dim,), minval=-0.5, maxval=0.5
        )
        autodiff_means = cb.to_mean(params)
        samples = cb.sample(jax.random.PRNGKey(11), params, n=20_000)
        empirical_first = jnp.mean(samples, axis=0)
        # Compare just the first-moment slice (diagonal entries).
        autodiff_first = autodiff_means[:n]
        assert jnp.allclose(empirical_first, autodiff_first, atol=0.02)

    def test_jit_compiles(self) -> None:
        """log_partition and sample compose with jax.jit."""
        cb = ChordalBoltzmann.from_edges(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
        params = jax.random.uniform(
            jax.random.PRNGKey(42), (cb.dim,), minval=-1.0, maxval=1.0
        )
        log_z_jit = jax.jit(cb.log_partition_function)(params)
        log_z = cb.log_partition_function(params)
        assert jnp.allclose(log_z_jit, log_z, rtol=RTOL, atol=ATOL)


class TestChainBoltzmann:
    """ChainBoltzmann: path validation and the associative-scan kernel."""

    def test_chain_tree_clique_path(self) -> None:
        ct = ChainTree.from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
        assert isinstance(ct, JunctionTree)
        assert set(ct.clique_path) == set(range(ct.n_cliques))

    def test_branching_clique_tree_raises(self) -> None:
        """A star graph's clique tree branches at the hub clique."""
        with pytest.raises(ValueError, match="not a path"):
            ChainTree.from_edges(5, [(0, k) for k in range(1, 5)])
        with pytest.raises(ValueError, match="not a path"):
            ChainBoltzmann.from_edges(5, [(0, k) for k in range(1, 5)])

    @pytest.mark.parametrize(
        "n,edges",
        [
            (12, [(i, i + 1) for i in range(11)]),  # chain
            (8, [(i, i + k) for k in (1, 2) for i in range(8 - k)]),  # band-2
            (5, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]),  # mixed sep sizes
        ],
    )
    def test_matches_chordal(self, n: int, edges: list[tuple[int, int]]) -> None:
        """The associative-scan kernel agrees with the sequential collect in
        value and gradient (gradients must stay finite through the clamped
        empty separator-state segments)."""
        chain = ChainBoltzmann.from_edges(n, edges)
        chordal = ChordalBoltzmann.from_edges(n, edges)
        assert chain.dim == chordal.dim

        params = jax.random.uniform(
            jax.random.PRNGKey(11), (chain.dim,), minval=-1.5, maxval=1.5
        )
        assert jnp.allclose(
            chain.log_partition_function(params),
            chordal.log_partition_function(params),
            rtol=RTOL,
            atol=ATOL,
        )
        g_chain = jax.grad(chain.log_partition_function)(params)
        g_chordal = jax.grad(chordal.log_partition_function)(params)
        assert jnp.all(jnp.isfinite(g_chain))
        assert jnp.allclose(g_chain, g_chordal, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize(
        "n,edges",
        [
            (16, [(i, i + 1) for i in range(15)]),  # plain chain
            # band-2 head + chain tail: heterogeneous separator sizes (2 and 1)
            # at a size far beyond enumeration reach
            (
                20,
                [(i, i + k) for k in (1, 2) for i in range(10 - k)]
                + [(i, i + 1) for i in range(9, 19)],
            ),
        ],
    )
    def test_sampling_matches_chain_to_mean(
        self, n: int, edges: list[tuple[int, int]]
    ) -> None:
        """Parallel backward sampling is consistent with the chain kernel's
        to_mean (autodiff of the associative-scan log-partition) on the *full*
        sufficient statistics --- pair moments included. to_mean is exact at
        any size, so this extends the enumeration sampling tests past their
        2^n ceiling; the heterogeneous case exercises the clamped
        empty-segment regime end to end."""
        cb = ChainBoltzmann.from_edges(n, edges)
        params = jax.random.uniform(
            jax.random.PRNGKey(5), (cb.dim,), minval=-0.5, maxval=0.5
        )
        samples = cb.sample(jax.random.PRNGKey(6), params, n=20_000)
        empirical = cb.average_sufficient_statistic(samples)
        assert jnp.allclose(empirical, cb.to_mean(params), atol=0.02)

    @pytest.mark.parametrize(
        "n,edges",
        [
            (3, [(0, 1), (1, 2), (0, 2)]),  # single clique (K = 1)
            (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),  # two cliques (K = 2)
            (5, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]),  # mixed sep sizes
            (8, [(i, i + k) for k in (1, 2) for i in range(8 - k)]),  # band-2
        ],
    )
    def test_sampling_matches_enumeration(
        self, n: int, edges: list[tuple[int, int]]
    ) -> None:
        """Parallel backward sampling matches exact sufficient-statistic means
        from enumeration --- including pair statistics, which would expose any
        error in the backward messages or the random-map composition."""
        cb = ChainBoltzmann.from_edges(n, edges)
        params = jax.random.uniform(
            jax.random.PRNGKey(13), (cb.dim,), minval=-1.0, maxval=1.0
        )
        samples = cb.sample(jax.random.PRNGKey(14), params, n=20_000)
        empirical = cb.average_sufficient_statistic(samples)
        log_z, all_states, energies = _bf_log_partition(n, cb.junction_tree, params)
        probs = jnp.exp(energies - log_z)
        exact = probs @ jax.vmap(cb.sufficient_statistic)(all_states)
        assert jnp.allclose(empirical, exact, atol=0.02)

    @pytest.mark.parametrize(
        "n,edges",
        [
            (5, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]),  # mixed seps
            (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),  # 4-cycle, K = 2
            (6, [(i, i + 1) for i in range(5)]),  # plain chain, K = 5
        ],
    )
    def test_sampling_matches_joint_distribution(
        self, n: int, edges: list[tuple[int, int]]
    ) -> None:
        """Empirical state frequencies match the exact joint over all $2^n$ states.

        The strongest black-box certificate for the random-map sampler: moment
        tests only pin chordal-edge statistics, while the histogram certifies
        the entire joint --- non-adjacent correlations (e.g. $E[x_0 x_5]$ on
        the chain) and higher-order structure, where an error in the shared
        Gumbel / map-composition argument would hide from moment checks."""
        cb = ChainBoltzmann.from_edges(n, edges)
        params = jax.random.uniform(
            jax.random.PRNGKey(23), (cb.dim,), minval=-1.0, maxval=1.0
        )
        n_samples = 50_000
        samples = cb.sample(jax.random.PRNGKey(24), params, n=n_samples)
        # LSB-first packing matches the state order of _bf_log_partition.
        codes = jnp.sum(samples.astype(jnp.int32) << jnp.arange(n), axis=1)
        freqs = jnp.bincount(codes, length=1 << n) / n_samples
        log_z, _all_states, energies = _bf_log_partition(n, cb.junction_tree, params)
        probs = jnp.exp(energies - log_z)
        assert jnp.allclose(freqs, probs, atol=0.01)
