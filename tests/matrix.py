"""Tests for geometry/manifold/matrix.py.

Verifies embed/project round-trips, embedding composition, and matrix conversion
for Identity, Scale, Diagonal, and PositiveDefinite representations.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import Diagonal, Identity, MatrixRep, PositiveDefinite, Scale

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-5
ATOL = 1e-7

SHAPES = [(2, 2), (3, 3), (4, 4)]
REPS = [Identity(), Scale(), Diagonal(), PositiveDefinite()]


def random_params(rep: MatrixRep, shape: tuple[int, int], key: Array) -> Array:
    """Generate random parameters for a matrix representation."""
    n = shape[0]
    if isinstance(rep, Identity):
        return jnp.array([])
    if isinstance(rep, Scale):
        return jax.random.normal(key, (1,))
    if isinstance(rep, Diagonal):
        return jax.random.normal(key, (n,))
    if isinstance(rep, PositiveDefinite):
        ell = jax.random.normal(key, (n, n))
        return rep.from_matrix(ell @ ell.T)
    return jax.random.normal(key, (rep.num_params(shape),))


class TestEmbedProject:
    """Test embed/project round-trips and composition."""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_round_trip(self, shape: tuple[int, int]) -> None:
        """Embed then project recovers original params for all rep pairs."""
        keys = jax.random.split(jax.random.PRNGKey(42), len(REPS))
        for i, source in enumerate(REPS[:-1]):
            for target in REPS[i + 1 :]:
                params = random_params(source, shape, keys[i])
                embedded = source.embed_params(shape, params, target)
                projected = target.project_params(shape, embedded, source)
                assert jnp.allclose(projected, params, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_embed_matches_dense(self, shape: tuple[int, int]) -> None:
        """Embed matches going through dense matrix."""
        keys = jax.random.split(jax.random.PRNGKey(42), len(REPS))
        for i, source in enumerate(REPS[:-1]):
            for target in REPS[i + 1 :]:
                params = random_params(source, shape, keys[i])
                embedded = source.embed_params(shape, params, target)
                via_dense = target.from_matrix(source.to_matrix(shape, params))
                assert jnp.allclose(embedded, via_dense, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_composition(self, shape: tuple[int, int]) -> None:
        """Embedding in two steps matches direct embedding."""
        keys = jax.random.split(jax.random.PRNGKey(42), len(REPS))
        for i, rep1 in enumerate(REPS[:-2]):
            for j, rep2 in enumerate(REPS[i + 1 : -1], start=i + 1):
                for rep3 in REPS[j + 1 :]:
                    params = random_params(rep1, shape, keys[i])
                    direct = rep1.embed_params(shape, params, rep3)
                    step1 = rep1.embed_params(shape, params, rep2)
                    composed = rep2.embed_params(shape, step1, rep3)
                    assert jnp.allclose(direct, composed, rtol=RTOL, atol=ATOL)


class TestMatrixConversion:
    """Test to_matrix/from_matrix round-trips and structural properties."""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_to_from_round_trip(self, shape: tuple[int, int]) -> None:
        """to_matrix then from_matrix recovers parameters."""
        keys = jax.random.split(jax.random.PRNGKey(42), len(REPS))
        for i, rep in enumerate(REPS):
            if isinstance(rep, Identity):
                continue
            params = random_params(rep, shape, keys[i])
            matrix = rep.to_matrix(shape, params)
            recovered = rep.from_matrix(matrix)
            assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_positive_definite_symmetry(self, shape: tuple[int, int]) -> None:
        """PositiveDefinite produces symmetric matrices."""
        rep = PositiveDefinite()
        params = random_params(rep, shape, jax.random.PRNGKey(42))
        matrix = rep.to_matrix(shape, params)
        assert jnp.allclose(matrix, matrix.T, rtol=RTOL, atol=ATOL)


class TestSpecificRepresentations:
    """Test representation-specific properties."""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_identity(self, shape: tuple[int, int]) -> None:
        """Identity has zero params and produces identity matrix."""
        rep = Identity()
        assert rep.num_params(shape) == 0
        matrix = rep.to_matrix(shape, jnp.array([]))
        assert jnp.allclose(matrix, jnp.eye(shape[0]), rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_scale(self, shape: tuple[int, int]) -> None:
        """Scale has one param and produces scaled identity."""
        rep = Scale()
        assert rep.num_params(shape) == 1
        matrix = rep.to_matrix(shape, jnp.array([2.5]))
        assert jnp.allclose(matrix, 2.5 * jnp.eye(shape[0]), rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_diagonal(self, shape: tuple[int, int]) -> None:
        """Diagonal has n params and produces diagonal matrix."""
        rep = Diagonal()
        n = shape[0]
        assert rep.num_params(shape) == n
        diag_vals = jax.random.normal(jax.random.PRNGKey(42), (n,))
        matrix = rep.to_matrix(shape, diag_vals)
        assert jnp.allclose(matrix, jnp.diag(diag_vals), rtol=RTOL, atol=ATOL)
