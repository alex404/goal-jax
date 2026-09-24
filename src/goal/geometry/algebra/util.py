"""Flat-array slicing shared by the layout descriptors."""

from __future__ import annotations

from jax import Array


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
