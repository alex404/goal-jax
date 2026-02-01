"""Tests for matrix representation operations.

Tests embedding, projection, and composition of matrix representations.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import Diagonal, Identity, MatrixRep, PositiveDefinite, Scale

jax.config.update("jax_platform_name", "cpu")

# Tolerances
RTOL = 1e-5
ATOL = 1e-7

# Test configurations
SHAPES = [(2, 2), (3, 3), (4, 4)]
REPS = [Identity(), Scale(), Diagonal(), PositiveDefinite()]


@pytest.fixture
def key() -> Array:
    """Random key for tests."""
    return jax.random.PRNGKey(42)


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
    """Test embed/project operations."""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_embed_project_round_trip(self, shape: tuple[int, int], key: Array) -> None:
        """Test embed followed by project recovers original params."""
        keys = jax.random.split(key, len(REPS))

        for i, source in enumerate(REPS[:-1]):
            for target in REPS[i + 1 :]:
                params = random_params(source, shape, keys[i])
                embedded = source.embed_params(shape, params, target)
                projected = target.project_params(shape, embedded, source)
                assert jnp.allclose(projected, params, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_embed_matches_dense(self, shape: tuple[int, int], key: Array) -> None:
        """Test embed matches going through dense matrix representation."""
        keys = jax.random.split(key, len(REPS))

        for i, source in enumerate(REPS[:-1]):
            for target in REPS[i + 1 :]:
                params = random_params(source, shape, keys[i])
                embedded = source.embed_params(shape, params, target)
                via_dense = target.from_matrix(source.to_matrix(shape, params))
                assert jnp.allclose(embedded, via_dense, rtol=RTOL, atol=ATOL)


class TestEmbedComposition:
    """Test embedding composition properties."""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_composition(self, shape: tuple[int, int], key: Array) -> None:
        """Test embedding in two steps matches direct embedding."""
        keys = jax.random.split(key, len(REPS))

        for i, rep1 in enumerate(REPS[:-2]):
            for j, rep2 in enumerate(REPS[i + 1 : -1], start=i + 1):
                for rep3 in REPS[j + 1 :]:
                    params = random_params(rep1, shape, keys[i])
                    direct = rep1.embed_params(shape, params, rep3)
                    step1 = rep1.embed_params(shape, params, rep2)
                    composed = rep2.embed_params(shape, step1, rep3)
                    assert jnp.allclose(direct, composed, rtol=RTOL, atol=ATOL)


class TestMatrixConversion:
    """Test matrix conversion operations."""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_to_from_matrix_round_trip(self, shape: tuple[int, int], key: Array) -> None:
        """Test to_matrix followed by from_matrix recovers parameters."""
        keys = jax.random.split(key, len(REPS))

        for i, rep in enumerate(REPS):
            if isinstance(rep, Identity):
                continue  # Identity has no parameters to test
            params = random_params(rep, shape, keys[i])
            matrix = rep.to_matrix(shape, params)
            recovered = rep.from_matrix(matrix)
            assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_matrix_symmetry(self, shape: tuple[int, int], key: Array) -> None:
        """Test PositiveDefinite produces symmetric matrices."""
        rep = PositiveDefinite()
        params = random_params(rep, shape, key)
        matrix = rep.to_matrix(shape, params)
        assert jnp.allclose(matrix, matrix.T, rtol=RTOL, atol=ATOL)


class TestIdentityRepresentation:
    """Test Identity representation specifics."""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_identity_matrix(self, shape: tuple[int, int]) -> None:
        """Test Identity produces identity matrix."""
        rep = Identity()
        params = jnp.array([])
        matrix = rep.to_matrix(shape, params)
        expected = jnp.eye(shape[0])
        assert jnp.allclose(matrix, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_identity_num_params(self, shape: tuple[int, int]) -> None:
        """Test Identity has zero parameters."""
        rep = Identity()
        assert rep.num_params(shape) == 0


class TestScaleRepresentation:
    """Test Scale representation specifics."""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_scale_num_params(self, shape: tuple[int, int]) -> None:
        """Test Scale has one parameter."""
        rep = Scale()
        assert rep.num_params(shape) == 1

    @pytest.mark.parametrize("shape", SHAPES)
    def test_scale_matrix(self, shape: tuple[int, int]) -> None:
        """Test Scale produces scaled identity."""
        rep = Scale()
        scale_val = 2.5
        params = jnp.array([scale_val])
        matrix = rep.to_matrix(shape, params)
        expected = scale_val * jnp.eye(shape[0])
        assert jnp.allclose(matrix, expected, rtol=RTOL, atol=ATOL)


class TestDiagonalRepresentation:
    """Test Diagonal representation specifics."""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_diagonal_num_params(self, shape: tuple[int, int]) -> None:
        """Test Diagonal has n parameters."""
        rep = Diagonal()
        assert rep.num_params(shape) == shape[0]

    @pytest.mark.parametrize("shape", SHAPES)
    def test_diagonal_matrix(self, shape: tuple[int, int], key: Array) -> None:
        """Test Diagonal produces diagonal matrix."""
        rep = Diagonal()
        n = shape[0]
        diag_vals = jax.random.normal(key, (n,))
        matrix = rep.to_matrix(shape, diag_vals)
        expected = jnp.diag(diag_vals)
        assert jnp.allclose(matrix, expected, rtol=RTOL, atol=ATOL)
