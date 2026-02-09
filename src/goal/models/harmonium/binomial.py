"""Standalone filter visualization functions for harmoniums.

These functions reshape interaction weights into filter images for visualization.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from ...geometry import Harmonium


def get_rectangular_filters(
    harmonium: Harmonium[Any, Any],
    params: Array,
    img_shape: tuple[int, int],
) -> Array:
    """Reshape interaction weights into filter images for visualization.

    For harmoniums with rectangular interaction (standard case).
    Each latent unit's weights form one filter image.

    Args:
        harmonium: The harmonium model
        params: Model parameters
        img_shape: Shape of each observable image (height, width)

    Returns:
        Array of shape (n_latent, height, width)
    """
    _, int_params, _ = harmonium.split_coords(params)
    n_obs = harmonium.obs_man.dim
    n_lat = harmonium.pst_man.dim
    weights = int_params.reshape(n_obs, n_lat)
    return weights.T.reshape(-1, *img_shape)


def get_vonmises_filters(
    harmonium: Harmonium[Any, Any],
    params: Array,
    img_shape: tuple[int, int],
) -> Array:
    """Reshape interaction weights for VonMises latents into filter images.

    Combines cos/sin pairs into magnitude for each latent unit.

    Args:
        harmonium: The harmonium model with VonMises latents
        params: Model parameters
        img_shape: Shape of each observable image (height, width)

    Returns:
        Array of shape (n_latent_units, height, width)
    """
    _, int_params, _ = harmonium.split_coords(params)
    n_obs = harmonium.obs_man.dim
    n_lat_pairs = harmonium.pst_man.dim // 2
    weights = int_params.reshape(n_obs, -1)

    # Combine cos and sin weights for each latent into magnitude
    w_cos = weights[:, 0::2]  # Shape: (n_obs, n_lat_pairs)
    w_sin = weights[:, 1::2]  # Shape: (n_obs, n_lat_pairs)
    magnitudes = jnp.sqrt(w_cos**2 + w_sin**2)  # Shape: (n_obs, n_lat_pairs)

    return magnitudes.T.reshape(n_lat_pairs, *img_shape)
