"""Internal helpers for batched computation and flat-array slicing."""

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


def split_by_dims(coords: Array, dims: tuple[int, ...]) -> tuple[Array, ...]:
    """Split a flat array into consecutive slices of the given sizes.

    Raises:
        ValueError: if ``coords`` is not one-dimensional of length ``sum(dims)``. Slicing
            past the end is silent in JAX, so without this a short vector yields empty
            trailing blocks and a long one drops coordinates.
    """
    total = sum(dims)
    if coords.ndim != 1 or coords.shape[0] != total:
        msg = f"expected a flat array of {total} coordinates"
        raise ValueError(f"{msg}, got shape {coords.shape}")
    out: list[Array] = []
    offset = 0
    for block_dim in dims:
        out.append(coords[offset : offset + block_dim])
        offset += block_dim
    return tuple(out)
