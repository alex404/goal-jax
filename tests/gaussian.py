"""Tests for ``models/base/gaussian``: ``normal.py``, ``boltzmann.py`` and the junction trees of ``coupling/``.

Normal distributions are checked across the Scale, Diagonal and PositiveDefinite representations, whose Cholesky, log-determinant and precision-scaling paths differ: fitted moments and log-densities against numpy and scipy, precision against the inverse covariance, relative whitening against a manual Cholesky solve, the closed-form negative entropy, and sampling.

Boltzmann machines are checked against enumeration of all $2^n$ binary states: the log-partition of every chordal topology, the full joint distribution of every exact sampler, and the Gibbs sampler's moments. The location/precision energy identity pins the $-2$/$-1$ precision convention, which a split-join round trip cannot detect. Chain kernels agree with the sequential junction-tree kernel in value and gradient, and at scale, where enumeration is out of reach, against a hand-written forward recursion and against ``to_mean``.
"""

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scipy import stats

from goal.geometry import Diagonal, PositiveDefinite, Scale
from goal.models import (
    Boltzmann,
    BoltzmannEmbedding,
    ChainBoltzmann,
    ChainTree,
    ChordalBoltzmann,
    DiagonalBoltzmann,
    EnumeratedBoltzmann,
    FullBoltzmann,
    JunctionTree,
    Normal,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-5
ATOL = 1e-7

type Rep = Scale | Diagonal | PositiveDefinite
type Edges = list[tuple[int, int]]

REPS = [Scale(), Diagonal(), PositiveDefinite()]
REP_IDS = ["scale", "diagonal", "full"]

MEAN = np.array([1.0, -0.5])
COVARIANCE = np.array([[2.0, 1.0], [1.0, 1.0]])


def covariance_in(rep: PositiveDefinite, full: np.ndarray) -> np.ndarray:
    """The dense covariance matrix that ``rep`` keeps of ``full``: all of it, its diagonal, or its mean variance."""
    if isinstance(rep, Scale):
        return np.mean(np.diag(full)) * np.eye(len(full))
    if isinstance(rep, Diagonal):
        return np.diag(np.diag(full))
    return full


def normal_means[R: PositiveDefinite](
    model: Normal[R], mean: np.ndarray, full: np.ndarray
) -> Array:
    """Mean parameters at the given mean and the covariance ``model``'s representation keeps of ``full``."""
    cov = model.cov_man.from_matrix(jnp.asarray(covariance_in(model.rep, full)))
    return model.join_mean_covariance(jnp.asarray(mean), cov)


# Normal


class TestNormal:
    @pytest.mark.parametrize("rep", REPS, ids=REP_IDS)
    def test_fit_matches_sample_moments_and_scipy(self, rep: Rep) -> None:
        """Mean parameters averaged from data give the sample mean and the representation's projection of the sample covariance; the log-density at the fit matches scipy."""
        xs = jax.random.multivariate_normal(
            jax.random.PRNGKey(0), jnp.asarray(MEAN), jnp.asarray(COVARIANCE), (2000,)
        )
        model = Normal(2, rep)
        means = model.average_sufficient_statistic(xs)
        params = model.to_natural(means)
        assert model.check_natural_parameters(params)
        assert jnp.allclose(model.to_mean(params), means, rtol=RTOL, atol=ATOL)

        data = np.asarray(xs)
        expected_cov = covariance_in(rep, np.cov(data.T, bias=True))
        mean, cov = model.split_mean_covariance(means)
        assert jnp.allclose(mean, data.mean(axis=0), rtol=RTOL, atol=ATOL)
        assert jnp.allclose(
            model.cov_man.to_matrix(cov), expected_cov, rtol=RTOL, atol=ATOL
        )

        expected_ll = stats.multivariate_normal.logpdf(
            data, data.mean(axis=0), expected_cov
        )
        assert jnp.allclose(
            model.average_log_density(params, xs),
            expected_ll.mean(),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_scale_embeds_into_diagonal_and_full(self) -> None:
        """An isotropic normal embedded in the richer representations keeps its log-partition and density."""
        iso = Normal(2, Scale())
        params = iso.to_natural(normal_means(iso, MEAN, COVARIANCE))
        xs = jax.random.normal(jax.random.PRNGKey(1), (20, 2))
        for rep in [Diagonal(), PositiveDefinite()]:
            trg = Normal(2, rep)
            trg_params = iso.embed_rep(trg, params)
            assert jnp.allclose(
                trg.log_partition_function(trg_params),
                iso.log_partition_function(params),
                rtol=RTOL,
                atol=ATOL,
            )
            assert jnp.allclose(
                jax.vmap(trg.log_density, in_axes=(None, 0))(trg_params, xs),
                jax.vmap(iso.log_density, in_axes=(None, 0))(params, xs),
                rtol=RTOL,
                atol=ATOL,
            )

    @pytest.mark.parametrize("rep", REPS, ids=REP_IDS)
    def test_location_precision(self, rep: Rep) -> None:
        """The precision is $\\Sigma^{-1}$ and the location is $\\Sigma^{-1} \\mu$."""
        model = Normal(2, rep)
        params = model.to_natural(normal_means(model, MEAN, COVARIANCE))
        loc, prec = model.split_location_precision(params)
        inv = np.linalg.inv(covariance_in(rep, COVARIANCE))
        assert jnp.allclose(model.cov_man.to_matrix(prec), inv, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(loc, inv @ MEAN, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("rep", REPS, ids=REP_IDS)
    @pytest.mark.parametrize("given", ["self", "other"])
    def test_relative_whiten(self, rep: Rep, given: str) -> None:
        """Whitening by $(\\mu_r, \\Sigma_r = L L^T)$ maps $(\\mu, \\Sigma)$ to $(L^{-1}(\\mu - \\mu_r), L^{-1} \\Sigma L^{-T})$; the reference itself goes to the standard normal."""
        model = Normal(2, rep)
        ref_means = normal_means(model, MEAN, COVARIANCE)
        if given == "self":
            mean, full = MEAN, COVARIANCE
        else:
            mean, full = np.array([3.0, -2.0]), np.array([[1.5, 0.3], [0.3, 2.0]])
        whitened = model.relative_whiten(normal_means(model, mean, full), ref_means)
        w_mean, w_cov = model.split_mean_covariance(whitened)

        chol = np.linalg.cholesky(covariance_in(rep, COVARIANCE))
        chol_inv = np.linalg.inv(chol)
        expected_cov = chol_inv @ covariance_in(rep, full) @ chol_inv.T
        assert jnp.allclose(w_mean, chol_inv @ (mean - MEAN), rtol=RTOL, atol=ATOL)
        assert jnp.allclose(
            model.cov_man.to_matrix(w_cov), expected_cov, rtol=RTOL, atol=ATOL
        )
        if given == "self":
            assert jnp.allclose(expected_cov, np.eye(2), atol=1e-12)

    @pytest.mark.parametrize("rep", REPS, ids=REP_IDS)
    def test_negative_entropy_closed_form(self, rep: Rep) -> None:
        """$\\phi(\\eta) = -\\frac{1}{2}(\\log|\\Sigma| + d)$."""
        model = Normal(2, rep)
        means = normal_means(model, MEAN, COVARIANCE)
        _, logdet = np.linalg.slogdet(covariance_in(rep, COVARIANCE))
        assert jnp.allclose(
            model.negative_entropy(means), -0.5 * (logdet + 2), rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize("rep", REPS, ids=REP_IDS)
    def test_sample_statistics_match_mean_parameters(self, rep: Rep) -> None:
        model = Normal(2, rep)
        params = model.to_natural(normal_means(model, MEAN, COVARIANCE))
        samples = model.sample(jax.random.PRNGKey(42), params, 50_000)
        assert jnp.allclose(jnp.mean(samples, axis=0), MEAN, atol=0.03)
        assert jnp.allclose(
            model.average_sufficient_statistic(samples), model.to_mean(params), atol=0.1
        )


# Boltzmann: enumeration helpers


def binary_states(n: int) -> Array:
    """All $2^n$ binary states, least significant unit first."""
    return jnp.array(
        [[(s >> i) & 1 for i in range(n)] for s in range(1 << n)], dtype=jnp.float64
    )


def exact_probabilities(model: Boltzmann[Any], params: Array) -> Array:
    """Probabilities of :func:`binary_states` by normalizing $e^{\\theta \\cdot s(x)}$ over all of them."""
    energies = (
        jax.vmap(model.sufficient_statistic)(binary_states(model.data_dim)) @ params
    )
    return jax.nn.softmax(energies)


def histogram(samples: Array) -> Array:
    """Empirical frequencies of :func:`binary_states`."""
    n = samples.shape[1]
    codes = jnp.sum(samples.astype(jnp.int32) << jnp.arange(n), axis=1)
    return jnp.bincount(codes, length=1 << n) / samples.shape[0]


def complete(n: int) -> Edges:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def chain(n: int) -> Edges:
    return [(i, i + 1) for i in range(n - 1)]


def band(n: int, width: int) -> Edges:
    return [(i, i + k) for k in range(1, width + 1) for i in range(n - k)]


CYCLE_4 = [(0, 1), (1, 2), (2, 3), (0, 3)]
CYCLE_5 = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
MIXED = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]  # separators of sizes 2 and 1
HUB = [(0, k) for k in range(1, 5)]  # branching clique tree
DISCONNECTED = [(0, 1), (3, 4)]  # two edges and an isolated node


# Boltzmann


class TestBoltzmann:
    @pytest.mark.parametrize(
        "model",
        [DiagonalBoltzmann(4), FullBoltzmann(4), ChordalBoltzmann.from_edges(5, MIXED)],
        ids=["diagonal", "full", "chordal"],
    )
    def test_location_precision_energy_identity(self, model: Boltzmann[Any]) -> None:
        """$\\theta \\cdot s(x) = l \\cdot x - \\frac{1}{2} x^T \\Lambda x$ on binary states, for both split and join.

        $\\Lambda$ holds the precision coordinates' diagonal entries on its diagonal and each off-diagonal coordinate at $(i, j)$ and $(j, i)$. For binary $x$, $\\lambda \\cdot s(x)$ counts the diagonal once and each pair once, so $x^T \\Lambda x = 2 \\lambda \\cdot s(x) - \\mathrm{diag}(\\lambda) \\cdot x$, independently of the conversion under test.
        """
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(31), 4)
        xs = jax.random.bernoulli(k1, 0.5, (8, model.data_dim)).astype(jnp.float64)

        def energy(location: Array, precision: Array, x: Array) -> Array:
            prec_diag, _ = model.split_couplings(precision)
            quad = 2.0 * jnp.dot(precision, model.sufficient_statistic(x)) - jnp.dot(
                prec_diag, x
            )
            return jnp.dot(location, x) - 0.5 * quad

        params = jax.random.normal(k2, (model.dim,))
        location = jax.random.normal(k3, (model.data_dim,))
        precision = jax.random.normal(k4, (model.dim,))
        joined = model.join_location_precision(location, precision)
        split = model.split_location_precision(params)
        for x in xs:
            s = model.sufficient_statistic(x)
            assert jnp.allclose(
                jnp.dot(params, s), energy(*split, x), rtol=RTOL, atol=ATOL
            )
            assert jnp.allclose(
                jnp.dot(joined, s), energy(location, precision, x), rtol=RTOL, atol=ATOL
            )

    def test_unit_conditional_is_sigmoid_of_energy_difference(self) -> None:
        model = FullBoltzmann(4)
        coupling = model.shp_man
        key = jax.random.PRNGKey(42)
        params = jax.random.uniform(key, (model.dim,), minval=-2.0, maxval=2.0)
        for state in binary_states(4)[::3]:
            for unit in range(4):
                on, off = state.at[unit].set(1.0), state.at[unit].set(0.0)
                delta = jnp.dot(
                    params,
                    coupling.sufficient_statistic(on)
                    - coupling.sufficient_statistic(off),
                )
                assert jnp.allclose(
                    coupling.unit_conditional_prob(state, unit, params),
                    jax.nn.sigmoid(delta),
                    rtol=RTOL,
                    atol=ATOL,
                )

    def test_gibbs_sampling_matches_enumeration(self) -> None:
        model = FullBoltzmann(3)
        params = jax.random.uniform(
            jax.random.PRNGKey(42), (model.dim,), minval=-1.0, maxval=1.0
        )
        samples = model.sample(jax.random.PRNGKey(0), params, 50_000)
        probs = exact_probabilities(model, params)
        exact = probs @ jax.vmap(model.sufficient_statistic)(binary_states(3))
        assert jnp.allclose(
            model.average_sufficient_statistic(samples), exact, atol=0.05
        )

    def test_enumerated_sampling_matches_joint(self) -> None:
        model = EnumeratedBoltzmann(4)
        params = jax.random.uniform(
            jax.random.PRNGKey(5), (model.dim,), minval=-1.5, maxval=1.5
        )
        samples = model.sample(jax.random.PRNGKey(6), params, 50_000)
        assert jnp.allclose(
            histogram(samples), exact_probabilities(model, params), atol=0.01
        )

    def test_embedding_of_independent_units(self) -> None:
        """A diagonal Boltzmann machine embedded with zero coupling has independent unit probabilities $\\sigma(\\theta_i)$."""
        emb = BoltzmannEmbedding(FullBoltzmann(3), DiagonalBoltzmann(3))
        params = jnp.array([0.7, -1.2, 0.1])
        means = emb.amb_man.to_mean(emb.embed(params))
        assert jnp.allclose(
            emb.project(means), jax.nn.sigmoid(params), rtol=RTOL, atol=ATOL
        )


# Junction trees


class TestJunctionTree:
    @pytest.mark.parametrize(
        ("n", "edges", "treewidth", "n_chordal_edges", "cliques"),
        [
            (4, chain(4), 1, 3, {(0, 1), (1, 2), (2, 3)}),
            (4, CYCLE_4, 2, 5, None),  # either diagonal completes the cycle
            (4, complete(4), 3, 6, {(0, 1, 2, 3)}),
            (5, DISCONNECTED, 1, 2, {(0, 1), (2,), (3, 4)}),
        ],
        ids=["chain", "4-cycle", "k4", "disconnected"],
    )
    def test_is_junction_tree_of_chordal_completion(
        self,
        n: int,
        edges: Edges,
        treewidth: int,
        n_chordal_edges: int,
        cliques: set[tuple[int, ...]] | None,
    ) -> None:
        """The chordal edges contain the input; the cliques are complete in the completion and cover it; the clique tree spans every clique (one tree even for a disconnected graph) and has the running intersection property."""
        jt = JunctionTree.from_edges(n, edges)
        chordal = set(jt.chordal_edges)
        assert jt.treewidth == treewidth
        assert len(chordal) == n_chordal_edges
        if cliques is not None:
            assert set(jt.cliques) == cliques
        assert {tuple(sorted(e)) for e in edges} <= chordal
        for clique in jt.cliques:
            assert all((i, j) in chordal for i in clique for j in clique if i < j)
        assert all(any(i in c and j in c for c in jt.cliques) for i, j in chordal)

        assert len(jt.tree_edges) == jt.n_cliques - 1
        assert set(jt.pre_order) == set(range(jt.n_cliques))
        for v in range(n):
            holders = {k for k, c in enumerate(jt.cliques) if v in c}
            reached = {min(holders)}
            for _ in range(len(holders)):
                reached |= {
                    b if a in reached else a
                    for a, b in jt.tree_edges
                    if (a in reached) != (b in reached) and {a, b} <= holders
                }
            assert reached == holders

    def test_max_treewidth_raises(self) -> None:
        with pytest.raises(ValueError, match="treewidth"):
            JunctionTree.from_edges(4, complete(4), max_treewidth=2)

    def test_branching_clique_tree_is_not_a_chain(self) -> None:
        with pytest.raises(ValueError, match="not a path"):
            ChainTree.from_edges(5, HUB)
        with pytest.raises(ValueError, match="not a path"):
            ChainBoltzmann.from_edges(5, HUB)


# Chordal and chain Boltzmann machines


def forward_chain_log_partition(params: np.ndarray, n: int) -> float:
    """$\\log Z$ of a chain Boltzmann machine by a forward recursion over $x_0, \\ldots, x_{n-1}$."""
    diag, off = params[:n], params[n:]
    msg = np.array([0.0, diag[0]])
    for i in range(1, n):
        # msg[b] for x_{i-1} = b; new[a] for x_i = a
        new0 = np.logaddexp(msg[0], msg[1])
        new1 = np.logaddexp(msg[0] + diag[i], msg[1] + diag[i] + off[i - 1])
        msg = np.array([new0, new1])
    return float(np.logaddexp(msg[0], msg[1]))


SAMPLED_GRAPHS = [
    ("k3", 3, complete(3)),
    ("4-cycle", 4, CYCLE_4),
    ("5-cycle", 5, CYCLE_5),
    ("chain", 6, chain(6)),
    ("mixed", 5, MIXED),
    ("disconnected", 5, DISCONNECTED),
    ("band", 8, band(8, 2)),
]


class TestChordalBoltzmann:
    @pytest.mark.parametrize(
        ("n", "edges"),
        [
            (4, chain(4)),
            (4, CYCLE_4),
            (5, CYCLE_5),
            (6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)]),
            (3, []),
            (5, DISCONNECTED),
            (8, band(8, 2)),
            (5, MIXED),
            (5, HUB),
        ],
        ids=[
            "chain",
            "4-cycle",
            "5-cycle",
            "6-cycle-chord",
            "isolated",
            "disconnected",
            "band",
            "mixed",
            "hub",
        ],
    )
    def test_log_partition_matches_enumeration(self, n: int, edges: Edges) -> None:
        model = ChordalBoltzmann.from_edges(n, edges)
        params = jax.random.uniform(
            jax.random.PRNGKey(42), (model.dim,), minval=-1.5, maxval=1.5
        )
        energies = jax.vmap(model.sufficient_statistic)(binary_states(n)) @ params
        assert jnp.allclose(
            model.log_partition_function(params),
            jax.scipy.special.logsumexp(energies),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_complete_graph_is_full_boltzmann(self) -> None:
        chordal, full = ChordalBoltzmann.from_edges(4, complete(4)), FullBoltzmann(4)
        params = jax.random.uniform(
            jax.random.PRNGKey(7), (chordal.dim,), minval=-1.0, maxval=1.0
        )
        full_params = full.shp_man.rep.from_matrix(chordal.shp_man.to_matrix(params))
        assert jnp.allclose(
            chordal.log_partition_function(params),
            full.log_partition_function(full_params),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_matrix_layout(self) -> None:
        """``to_matrix`` places the biases on the diagonal and each chordal coupling symmetrically; ``from_matrix`` inverts it, drops off-pattern entries and averages asymmetric ones."""
        model = ChordalBoltzmann.from_edges(5, MIXED)
        cm = model.shp_man
        params = jax.random.normal(jax.random.PRNGKey(7), (model.dim,))
        mat = cm.to_matrix(params)
        assert jnp.allclose(jnp.diag(mat), params[:5])
        for k, (i, j) in enumerate(model.junction_tree.chordal_edges):
            assert mat[i, j] == params[5 + k] and mat[j, i] == params[5 + k]
        assert jnp.allclose(cm.from_matrix(mat), params, rtol=RTOL, atol=ATOL)
        assert (0, 4) not in model.junction_tree.chordal_edges
        off_pattern = mat.at[0, 4].add(3.0).at[4, 0].add(3.0)
        assert jnp.allclose(cm.from_matrix(off_pattern), params, rtol=RTOL, atol=ATOL)
        skewed = mat.at[0, 1].add(1.0).at[1, 0].add(-1.0)
        assert jnp.allclose(cm.from_matrix(skewed), params, rtol=RTOL, atol=ATOL)

    def test_long_chain_log_partition_matches_forward_recursion(self) -> None:
        n = 64
        model = ChordalBoltzmann.from_edges(n, chain(n))
        params = jax.random.uniform(
            jax.random.PRNGKey(101), (model.dim,), minval=-1.0, maxval=1.0
        )
        expected = forward_chain_log_partition(np.asarray(params), n)
        assert jnp.allclose(
            model.log_partition_function(params), expected, rtol=RTOL, atol=ATOL
        )

    def test_long_chain_sampling_matches_mean_parameters(self) -> None:
        n = 64
        model = ChordalBoltzmann.from_edges(n, chain(n))
        params = jax.random.uniform(
            jax.random.PRNGKey(7), (model.dim,), minval=-0.5, maxval=0.5
        )
        samples = model.sample(jax.random.PRNGKey(11), params, 20_000)
        assert jnp.allclose(
            model.average_sufficient_statistic(samples),
            model.to_mean(params),
            atol=0.02,
        )

    @pytest.mark.parametrize(
        ("cls", "n", "edges"),
        [(ChordalBoltzmann, n, e) for _, n, e in [*SAMPLED_GRAPHS, ("hub", 5, HUB)]]
        + [(ChainBoltzmann, n, e) for _, n, e in SAMPLED_GRAPHS],
        ids=[f"chordal-{g}" for g, _, _ in [*SAMPLED_GRAPHS, ("hub", 5, HUB)]]
        + [f"chain-{g}" for g, _, _ in SAMPLED_GRAPHS],
    )
    def test_sampling_matches_joint(
        self, cls: type[ChordalBoltzmann], n: int, edges: Sequence[tuple[int, int]]
    ) -> None:
        """State frequencies match the exact joint over all $2^n$ states, which certifies non-adjacent correlations and higher-order structure that moment checks miss."""
        model = cls.from_edges(n, edges)
        params = jax.random.uniform(
            jax.random.PRNGKey(21), (model.dim,), minval=-1.0, maxval=1.0
        )
        samples = model.sample(jax.random.PRNGKey(22), params, 50_000)
        assert jnp.allclose(
            histogram(samples), exact_probabilities(model, params), atol=0.01
        )


class TestChainBoltzmann:
    @pytest.mark.parametrize(
        ("n", "edges"),
        [(12, chain(12)), (8, band(8, 2)), (5, MIXED)],
        ids=["chain", "band", "mixed"],
    )
    def test_matches_chordal(self, n: int, edges: Edges) -> None:
        """The associative-scan kernel agrees with the sequential collect in value and gradient; gradients stay finite through the clamped empty separator segments."""
        chain_model = ChainBoltzmann.from_edges(n, edges)
        chordal = ChordalBoltzmann.from_edges(n, edges)
        params = jax.random.uniform(
            jax.random.PRNGKey(11), (chain_model.dim,), minval=-1.5, maxval=1.5
        )
        assert jnp.allclose(
            chain_model.log_partition_function(params),
            chordal.log_partition_function(params),
            rtol=RTOL,
            atol=ATOL,
        )
        g_chain = jax.grad(chain_model.log_partition_function)(params)
        assert jnp.all(jnp.isfinite(g_chain))
        assert jnp.allclose(
            g_chain,
            jax.grad(chordal.log_partition_function)(params),
            rtol=RTOL,
            atol=ATOL,
        )

    @pytest.mark.parametrize(
        ("n", "edges"),
        [(16, chain(16)), (20, band(10, 2) + [(i, i + 1) for i in range(9, 19)])],
        ids=["chain", "band-then-chain"],
    )
    def test_sampling_at_scale_matches_mean_parameters(
        self, n: int, edges: Edges
    ) -> None:
        """Past enumeration reach, full sufficient statistics (pair moments included) of the samples match ``to_mean``; the second graph has separators of sizes 2 and 1."""
        model = ChainBoltzmann.from_edges(n, edges)
        params = jax.random.uniform(
            jax.random.PRNGKey(5), (model.dim,), minval=-0.5, maxval=0.5
        )
        samples = model.sample(jax.random.PRNGKey(6), params, 20_000)
        assert jnp.allclose(
            model.average_sufficient_statistic(samples),
            model.to_mean(params),
            atol=0.02,
        )
