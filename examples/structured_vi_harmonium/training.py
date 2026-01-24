"""Training utilities for structured variational inference.

This module provides initialization, training step, and epoch functions
for the structured VI harmonium.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.models import BinomialVonMisesRBM

from .core import StructuredVIConfig, StructuredVIState, make_elbo_loss_and_grad_fn


def initialize_structured_vi(
    key: Array,
    model: BinomialVonMisesRBM,
    data: Array,
    shape: float = 0.01,
    zero_interaction: bool = True,
) -> StructuredVIState:
    """Initialize structured VI state.

    Initializes with small interaction and zero rho for starting point
    where the approximate posterior equals the exact conjugate posterior.

    Args:
        key: JAX random key
        model: The harmonium model
        data: Training data for initialization
        shape: Scale for random initialization
        zero_interaction: If True, start with W = 0

    Returns:
        Initial StructuredVIState
    """
    harmonium_params = model.initialize_from_sample(key, data, shape=shape)

    if zero_interaction:
        obs_params, _, lat_params = model.split_coords(harmonium_params)
        int_params = jnp.zeros(model.int_man.dim)
        harmonium_params = model.join_coords(obs_params, int_params, lat_params)

    # Initialize rho = 0 so approximate posterior = exact posterior initially
    rho_params = jnp.zeros(model.pst_man.dim)

    return StructuredVIState(harmonium_params, rho_params)


def make_train_step_fn(
    model: BinomialVonMisesRBM,
    config: StructuredVIConfig,
):
    """Create a JIT-compiled training step function.

    Returns a function that computes gradients for a single batch.

    Args:
        model: The harmonium model
        config: VI configuration

    Returns:
        A function (key, harm_params, rho_params, batch) -> (loss, harm_grad, rho_grad)
    """
    return make_elbo_loss_and_grad_fn(model, config)


def make_train_epoch_fn(
    train_step_fn: Any,
    opt_harm_update: Any,
    opt_rho_update: Any,
    batch_size: int,
):
    """Create a JIT-compiled epoch training function.

    Uses lax.scan for efficient batch processing.

    Args:
        train_step_fn: JIT-compiled step function
        opt_harm_update: Optimizer update for harmonium params
        opt_rho_update: Optimizer update for rho params
        batch_size: Batch size

    Returns:
        A function that trains one epoch and returns updated state
    """

    def train_epoch(
        key: Array,
        harmonium_params: Array,
        rho_params: Array,
        opt_harm_state: Array,
        opt_rho_state: Array,
        data: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Train one epoch using lax.scan over batches.

        Returns:
            Tuple of (harm_params, rho_params, opt_harm_state, opt_rho_state, key, epoch_loss)
        """
        n_samples = data.shape[0]
        n_batches = n_samples // batch_size

        # Shuffle data
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, n_samples)
        data_shuffled = data[perm]

        # Reshape into batches
        batches = data_shuffled[: n_batches * batch_size].reshape(
            n_batches, batch_size, -1
        )

        # Generate keys for all batches
        key, batch_keys_key = jax.random.split(key)
        batch_keys = jax.random.split(batch_keys_key, n_batches)

        def scan_body(
            carry: tuple[Array, Array, Array, Array, Array],
            inputs: tuple[Array, Array],
        ) -> tuple[tuple[Array, Array, Array, Array, Array], None]:
            harm_params, rho_params, opt_h_state, opt_r_state, total_loss = carry
            batch_key, batch = inputs

            # Compute gradients
            loss, harm_grad, rho_grad = train_step_fn(
                batch_key, harm_params, rho_params, batch
            )

            # Update parameters
            new_opt_h_state, new_harm_params = opt_harm_update(
                opt_h_state, harm_grad, harm_params
            )
            new_opt_r_state, new_rho_params = opt_rho_update(
                opt_r_state, rho_grad, rho_params
            )

            new_total_loss = total_loss + loss

            return (
                new_harm_params,
                new_rho_params,
                new_opt_h_state,
                new_opt_r_state,
                new_total_loss,
            ), None

        init_carry = (
            harmonium_params,
            rho_params,
            opt_harm_state,
            opt_rho_state,
            jnp.array(0.0),
        )
        (new_harm, new_rho, new_opt_h, new_opt_r, total_loss), _ = jax.lax.scan(
            scan_body, init_carry, (batch_keys, batches)
        )

        avg_loss = total_loss / n_batches

        return new_harm, new_rho, new_opt_h, new_opt_r, key, avg_loss

    return jax.jit(train_epoch)
