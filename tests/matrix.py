"""Tests for matrix representation operations."""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import Diagonal, Identity, MatrixRep, PositiveDefinite, Scale

jax.config.update("jax_platform_name", "cpu")

# Test parameters
shapes = [(2, 2), (3, 3), (4, 4)]
reps = [Identity(), Scale(), Diagonal(), PositiveDefinite()]


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


@pytest.mark.parametrize("shape", shapes)
def test_embed_project_round_trip(shape: tuple[int, int]):
    """Test embed followed by project recovers original params."""
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, len(reps))

    for i, source in enumerate(reps[:-1]):
        for target in reps[i + 1 :]:
            params = random_params(source, shape, keys[i])
            embedded = source.embed_params(shape, params, target)
            projected = target.project_params(shape, embedded, source)
            assert jnp.allclose(projected, params)


@pytest.mark.parametrize("shape", shapes)
def test_embed_matches_dense_conversion(shape: tuple[int, int]):
    """Test embed matches going through dense matrix representation."""
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, len(reps))

    for i, source in enumerate(reps[:-1]):
        for target in reps[i + 1 :]:
            params = random_params(source, shape, keys[i])
            embedded = source.embed_params(shape, params, target)
            via_dense = target.from_matrix(source.to_matrix(shape, params))
            assert jnp.allclose(embedded, via_dense)


@pytest.mark.parametrize("shape", shapes)
def test_embed_composition(shape: tuple[int, int]):
    """Test embedding in two steps matches direct embedding."""
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, len(reps))

    for i, rep1 in enumerate(reps[:-2]):
        for j, rep2 in enumerate(reps[i + 1 : -1], start=i + 1):
            for rep3 in reps[j + 1 :]:
                params = random_params(rep1, shape, keys[i])
                direct = rep1.embed_params(shape, params, rep3)
                step1 = rep1.embed_params(shape, params, rep2)
                composed = rep2.embed_params(shape, step1, rep3)
                assert jnp.allclose(direct, composed)
