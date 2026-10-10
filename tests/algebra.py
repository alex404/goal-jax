"""Tests for geometry/algebra: ``MatrixRep`` (matrix.py) and ``Cliques`` (clique.py).

Every structured matrix operation is checked against the same operation on the dense
matrix, and the dense matrix itself against closed forms. The graph of a ``Cliques`` is
checked on the cliques of shipped models.
"""

from collections.abc import Callable
from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    Diagonal,
    DifferentiableTuple,
    Identity,
    MatrixRep,
    PositiveDefinite,
    Rectangular,
    Scale,
    Square,
    Symmetric,
)
from goal.models import (
    CanonicalCorrelationAnalysis,
    MixtureOfFactorAnalyzers,
    Normal,
    analytic_hmog,
    factor_analysis,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-5
ATOL = 1e-7

N = 3
"""Side of the square matrices."""

RECT_SHAPE = (3, 4)

CHAIN: list[MatrixRep] = [
    Rectangular(),
    Square(),
    Symmetric(),
    PositiveDefinite(),
    Diagonal(),
    Scale(),
    Identity(),
]
"""Every rep, from most to least general; each is a subclass of the ones before it."""

SQUARE_REPS = CHAIN[1:]
SYMMETRIC_REPS = CHAIN[2:]
PD_REPS = CHAIN[3:]


def shape_of(rep: MatrixRep) -> tuple[int, int]:
    """Rectangular is tested on a non-square shape, every other rep on a square one."""
    return RECT_SHAPE if type(rep) is Rectangular else (N, N)


def random_params(rep: MatrixRep, shape: tuple[int, int], key: Array) -> Array:
    """Parameters of a well-conditioned matrix in ``rep``; positive definite for symmetric reps."""
    match rep:
        case Identity():
            return jnp.array([])
        case Scale():
            return 1.0 + jnp.abs(jax.random.normal(key, (1,)))
        case Diagonal():
            return 1.0 + jnp.abs(jax.random.normal(key, (shape[0],)))
        case Symmetric():
            a = jax.random.normal(key, shape)
            return rep.from_matrix(a @ a.T + shape[0] * jnp.eye(shape[0]))
        case Square():
            a = jax.random.normal(key, shape)
            return rep.from_matrix(a + 3 * shape[0] * jnp.eye(shape[0]))
        case _:
            return jax.random.normal(key, shape).reshape(-1)


def rep_id(rep: MatrixRep) -> str:
    return type(rep).__name__


### Matrix representations ###


EMBED_PAIRS = [(sub, amb) for i, sub in enumerate(CHAIN) for amb in CHAIN[:i]]
"""Every rep with every more general one."""


class TestEmbedProject:
    """Moving parameters between a rep and a more general one."""

    @pytest.mark.parametrize(
        ("sub", "amb"),
        EMBED_PAIRS,
        ids=[f"{rep_id(s)}-{rep_id(a)}" for s, a in EMBED_PAIRS],
    )
    def test_project_inverts_embed(self, sub: MatrixRep, amb: MatrixRep) -> None:
        params = random_params(sub, (N, N), jax.random.PRNGKey(0))
        embedded = sub.embed_params((N, N), params, amb)
        assert jnp.allclose(
            amb.project_params((N, N), embedded, sub), params, rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize(
        ("sub", "amb"),
        EMBED_PAIRS,
        ids=[f"{rep_id(s)}-{rep_id(a)}" for s, a in EMBED_PAIRS],
    )
    def test_embed_is_the_dense_route(self, sub: MatrixRep, amb: MatrixRep) -> None:
        """Embedding equals reading the same dense matrix in the more general rep."""
        params = random_params(sub, (N, N), jax.random.PRNGKey(1))
        assert jnp.allclose(
            sub.embed_params((N, N), params, amb),
            amb.from_matrix(sub.to_matrix((N, N), params)),
            rtol=RTOL,
            atol=ATOL,
        )


class TestDenseMatrix:
    """``to_matrix`` against closed forms, and ``from_matrix`` as its inverse."""

    @pytest.mark.parametrize("rep", CHAIN, ids=rep_id)
    def test_from_matrix_inverts_to_matrix(self, rep: MatrixRep) -> None:
        shape = shape_of(rep)
        params = random_params(rep, shape, jax.random.PRNGKey(2))
        assert jnp.allclose(
            rep.from_matrix(rep.to_matrix(shape, params)), params, rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize(
        ("rep", "shape", "params", "expected"),
        [
            (Rectangular(), (2, 3), [0, 1, 2, 3, 4, 5], [[0, 1, 2], [3, 4, 5]]),
            (
                Symmetric(),
                (3, 3),
                [0, 1, 2, 3, 4, 5],
                [[0, 1, 2], [1, 3, 4], [2, 4, 5]],
            ),
            (Diagonal(), (3, 3), [1, 2, 3], [[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            (Scale(), (3, 3), [2.5], [[2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5]]),
            (Identity(), (3, 3), [], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ],
        ids=["Rectangular", "Symmetric", "Diagonal", "Scale", "Identity"],
    )
    def test_closed_form(
        self,
        rep: MatrixRep,
        shape: tuple[int, int],
        params: list[float],
        expected: list[list[float]],
    ) -> None:
        """Row-major storage for rectangular, upper triangle for symmetric."""
        matrix = rep.to_matrix(shape, jnp.array(params, dtype=jnp.float64))
        assert jnp.array_equal(matrix, jnp.array(expected, dtype=jnp.float64))


class TestStructuredOpsMatchDense:
    """Each structured operation equals the same operation on the dense matrix."""

    @pytest.mark.parametrize("rep", CHAIN, ids=rep_id)
    def test_linear_ops(self, rep: MatrixRep) -> None:
        """``matvec``, ``transpose`` and ``outer_product``."""
        shape = shape_of(rep)
        k_p, k_v, k_w = jax.random.split(jax.random.PRNGKey(3), 3)
        params = random_params(rep, shape, k_p)
        v = jax.random.normal(k_v, (shape[1],))
        w = jax.random.normal(k_w, (shape[0],))
        matrix = rep.to_matrix(shape, params)

        assert jnp.allclose(rep.matvec(shape, params, v), matrix @ v)
        transposed = rep.to_matrix((shape[1], shape[0]), rep.transpose(shape, params))
        assert jnp.allclose(transposed, matrix.T)
        assert jnp.allclose(rep.outer_product(w, v), rep.from_matrix(jnp.outer(w, v)))

    @pytest.mark.parametrize("rep", SQUARE_REPS, ids=rep_id)
    def test_diagonal_ops(self, rep: MatrixRep) -> None:
        """``get_diagonal``, and ``map_diagonal`` wherever the rep can hold the result."""
        params = random_params(rep, (N, N), jax.random.PRNGKey(4))
        matrix = rep.to_matrix((N, N), params)
        assert jnp.allclose(rep.get_diagonal((N, N), params), jnp.diag(matrix))
        if type(rep) is Identity:
            return

        def f(d: Array) -> Array:
            return 2.0 * d + 1.0

        mapped = rep.to_matrix((N, N), rep.map_diagonal((N, N), params, f))
        expected = matrix.at[jnp.diag_indices(N)].set(f(jnp.diag(matrix)))
        assert jnp.allclose(mapped, expected)

    @pytest.mark.parametrize("rep", SQUARE_REPS, ids=rep_id)
    def test_inverse_and_logdet(self, rep: Square) -> None:
        params = random_params(rep, (N, N), jax.random.PRNGKey(5))
        matrix = rep.to_matrix((N, N), params)
        inverse = rep.to_matrix((N, N), rep.inverse((N, N), params))
        assert jnp.allclose(inverse, jnp.linalg.inv(matrix), rtol=RTOL, atol=ATOL)
        assert jnp.allclose(
            rep.logdet((N, N), params), jnp.linalg.slogdet(matrix)[1], rtol=RTOL
        )

    @pytest.mark.parametrize("rep", SYMMETRIC_REPS, ids=rep_id)
    def test_positive_definiteness(self, rep: Square) -> None:
        """A positive definite matrix is recognized, and its negation is rejected."""
        params = random_params(rep, (N, N), jax.random.PRNGKey(6))
        assert rep.is_positive_definite((N, N), params)
        if type(rep) is not Identity:
            assert not rep.is_positive_definite((N, N), -params)

    @pytest.mark.parametrize("rep", PD_REPS, ids=rep_id)
    def test_cholesky_ops(self, rep: PositiveDefinite) -> None:
        """``cholesky_matvec`` is $L v$, and ``cholesky_whiten`` is $L^{-1}(m_1 - m_2)$ and $L^{-1} A_1 L^{-T}$, with $A_2 = L L^T$."""
        keys = jax.random.split(jax.random.PRNGKey(7), 5)
        params1 = random_params(rep, (N, N), keys[0])
        params2 = random_params(rep, (N, N), keys[1])
        mean1 = jax.random.normal(keys[2], (N,))
        mean2 = jax.random.normal(keys[3], (N,))
        v = jax.random.normal(keys[4], (N,))
        chol = jnp.linalg.cholesky(rep.to_matrix((N, N), params2))

        assert jnp.allclose(rep.cholesky_matvec((N, N), params2, v), chol @ v)

        mean, params = rep.cholesky_whiten((N, N), mean1, params1, mean2, params2)
        chol_inv = jnp.linalg.inv(chol)
        assert jnp.allclose(mean, chol_inv @ (mean1 - mean2), rtol=RTOL, atol=ATOL)
        assert jnp.allclose(
            rep.to_matrix((N, N), params),
            chol_inv @ rep.to_matrix((N, N), params1) @ chol_inv.T,
            rtol=RTOL,
            atol=ATOL,
        )


@pytest.mark.parametrize(
    ("left", "right"), list(product(CHAIN, CHAIN)), ids=lambda rep: rep_id(rep)
)
def test_matmat_matches_dense(left: MatrixRep, right: MatrixRep) -> None:
    """The product, in whatever rep ``matmat`` returns, is the dense product."""
    left_shape = (2, N) if type(left) is Rectangular else (N, N)
    right_shape = (N, 4) if type(right) is Rectangular else (N, N)
    k_l, k_r = jax.random.split(jax.random.PRNGKey(8))
    left_params = random_params(left, left_shape, k_l)
    right_params = random_params(right, right_shape, k_r)
    out_rep, out_shape, out_params = left.matmat(
        left_shape, left_params, right, right_shape, right_params
    )
    expected = left.to_matrix(left_shape, left_params) @ right.to_matrix(
        right_shape, right_params
    )
    assert out_shape == expected.shape
    assert jnp.allclose(out_rep.to_matrix(out_shape, out_params), expected)


### Cliques ###


def _cca() -> CanonicalCorrelationAnalysis[Any, Any, Any]:
    return CanonicalCorrelationAnalysis(
        3, PositiveDefinite(), 2, Diagonal(), 2, PositiveDefinite()
    )


class TestGraph:
    """The adjacency and edges of a ``Cliques`` are derived from its cliques."""

    @pytest.mark.parametrize(
        ("factory", "graph", "edges"),
        [
            pytest.param(
                lambda: analytic_hmog(
                    obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=3
                ),
                {0: (1,), 1: (0, 2), 2: (1,)},
                ((0, 1), (1, 2)),
                id="hmog",
            ),
            pytest.param(
                lambda: MixtureOfFactorAnalyzers(
                    n_categories=3, bas_hrm=factor_analysis(obs_dim=4, lat_dim=2)
                ),
                {0: (1, 2), 1: (0, 2), 2: (0, 1)},
                ((0, 1), (0, 2), (1, 2)),
                id="mfa",
            ),
            pytest.param(
                _cca, {0: (2,), 1: (2,), 2: (0, 1)}, ((0, 2), (1, 2)), id="cca"
            ),
            pytest.param(
                lambda: DifferentiableTuple((_cca(), Normal(2, Diagonal()))),
                {0: (2,), 1: (2,), 2: (0, 1), 3: ()},
                ((0, 2), (1, 2)),
                id="cca_beside_a_lone_node",
            ),
        ],
    )
    def test_graph_and_edges(
        self,
        factory: Callable[[], Any],
        graph: dict[int, tuple[int, ...]],
        edges: tuple[tuple[int, int], ...],
    ) -> None:
        """HMoG is the chain $x - y - k$, MFA adds $x - k$ through its clique $(x, y, k)$, CCA is the fork $x - z - y$, and a node in no multi-node clique has no neighbours."""
        cliques = factory()
        assert cliques.graph == graph
        assert cliques.edges == edges
