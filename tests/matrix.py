"""Tests for matrix representation operations."""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    Diagonal,
    Identity,
    MatrixRep,
    PositiveDefinite,
    Scale,
)

# Test parameters
N_TRIALS = 10
SHAPES = [(2, 2), (3, 3), (4, 4)]  # Square() matrices for now
SEED = 0


def generate_random_params(rep: MatrixRep, shape: tuple[int, int], key: Array) -> Array:
    n = shape[0]
    if rep is Identity():
        return jnp.array([])
    if rep is Scale():
        return jax.random.normal(key, (1,))
    if rep is Diagonal():
        return jax.random.normal(key, (n,))
    if isinstance(rep, PositiveDefinite):
        # Generate valid PD matrix and convert to params
        ell = jax.random.normal(key, (n, n))
        return rep.from_matrix(ell @ ell.T)
    return jax.random.normal(key, (rep.num_params(shape),))


def test_embed_project_cycle():
    """Test that embed followed by project returns original params."""
    key = jax.random.PRNGKey(SEED)

    # Test hierarchy up: Identity() -> Scale() -> Diagonal() -> PositiveDefinite()
    reps = [Identity(), Scale(), Diagonal(), PositiveDefinite()]

    for shape in SHAPES:
        keys = jax.random.split(key, len(reps))

        for i, source_rep in enumerate(reps[:-1]):
            for target_rep in reps[i + 1 :]:
                src_params = generate_random_params(source_rep, shape, keys[i])

                # Embed up
                embedded = source_rep.embed_params(shape, src_params, target_rep)

                # Project back
                projected = target_rep.project_params(shape, embedded, source_rep)

                # Print test info
                print(f"Source: {source_rep}, Via: {target_rep}")
                print(f"Source params: {src_params}")
                print(f"Embedded params: {embedded}")
                print(f"Projected params: {projected}")

                assert jnp.allclose(projected, src_params)


def test_embed_project_vs_dense():
    """Test that embed/project matches going through dense representation."""
    key = jax.random.PRNGKey(SEED)
    reps = [Identity(), Scale(), Diagonal(), PositiveDefinite()]

    for shape in SHAPES:
        keys = jax.random.split(key, len(reps))

        for i, source_rep in enumerate(reps[:-1]):
            for target_rep in reps[i + 1 :]:
                src_params = generate_random_params(source_rep, shape, keys[i])

                # Via embed/project
                embedded = source_rep.embed_params(shape, src_params, target_rep)

                # Via dense
                dense = source_rep.to_matrix(shape, src_params)
                via_dense = target_rep.from_matrix(dense)

                # Print test info
                print(f"Source: {source_rep}, Embedding Target: {target_rep}")
                print(f"Source params: {src_params}")
                print(f"Embedded params: {embedded}")
                print(f"Via dense: {via_dense}")

                assert jnp.allclose(embedded, via_dense)


def test_composition():
    """Test that embedding two steps matches going directly."""
    key = jax.random.PRNGKey(SEED)
    reps = [Identity(), Scale(), Diagonal(), PositiveDefinite()]

    for shape in SHAPES:
        keys = jax.random.split(key, len(reps))

        for i, rep1 in enumerate(reps[:-2]):
            for j, rep2 in enumerate(reps[i + 1 : -1]):
                for rep3 in reps[j + 2 :]:
                    params = generate_random_params(rep1, shape, keys[i])

                    # Direct
                    direct = rep1.embed_params(shape, params, rep3)

                    # Via middle rep
                    step1 = rep1.embed_params(shape, params, rep2)
                    composed = rep2.embed_params(shape, step1, rep3)

                    assert jnp.allclose(direct, composed)


if __name__ == "__main__":
    _ = pytest.main([__file__])
