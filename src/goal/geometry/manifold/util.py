from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array


def batched_mean(f: Callable[[Array], Array], xs: Array, batch_size: int) -> Array:
    """Compute mean of mapped values over a dataset in a memory-efficient way."""
    n_samples = xs.shape[0]

    # Handle edge cases
    if n_samples == 0:
        raise ValueError("Cannot compute mean of empty dataset")
    if n_samples <= batch_size:
        return jnp.mean(jax.vmap(f)(xs), axis=0)

    # Split into complete batches and remainder
    n_complete = (n_samples // batch_size) * batch_size
    xs_batched = xs[:n_complete].reshape(-1, batch_size, *xs.shape[1:])

    def inner_sum(x_batch: Array) -> Array:
        return jnp.sum(jax.vmap(f)(x_batch), axis=0)

    # Process complete batches
    batch_sums = jnp.sum(jax.lax.map(inner_sum, xs_batched), axis=0)

    # Process remainder if it exists
    if n_complete < n_samples:
        remainder_sum = inner_sum(xs[n_complete:])
        total_sum = batch_sums + remainder_sum
    else:
        total_sum = batch_sums

    return total_sum / n_samples


def triangular_split_diagonal(matrix: Array) -> tuple[Array, Array]:
    """Split matrix into diagonal and off-diagonal parts.

    Returns (diagonal_elements, off_diagonal_lower_triangular).
    """
    n = matrix.shape[0]
    diagonal = jnp.diag(matrix)
    # Get lower triangular indices excluding diagonal
    tril_indices = jnp.tril_indices(n, k=-1)
    off_diagonal = matrix[tril_indices]
    return diagonal, off_diagonal


def triangular_join_diagonal(diagonal: Array, off_diagonal: Array) -> Array:
    """Join diagonal and off-diagonal parts into full matrix."""
    n = diagonal.shape[0]
    matrix = jnp.zeros((n, n))

    # Set diagonal
    matrix = matrix.at[jnp.diag_indices(n)].set(diagonal)

    # Set lower triangular (excluding diagonal)
    tril_indices = jnp.tril_indices(n, k=-1)
    matrix = matrix.at[tril_indices].set(off_diagonal)

    # Make symmetric
    return matrix + matrix.T
