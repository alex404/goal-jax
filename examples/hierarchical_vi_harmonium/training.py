"""Training utilities for hierarchical variational inference.

This module provides initialization, training step, and epoch functions
for the hierarchical VI harmonium with mixture of VonMises clustering.
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.models import BinomialMixtureVonMisesHarmonium

from .core import HierarchicalVIConfig, HierarchicalVIState, make_elbo_loss_and_grad_fn


def initialize_hierarchical_vi(
    key: Array,
    model: BinomialMixtureVonMisesHarmonium,
    data: Array,
    shape: float = 0.01,
    zero_interaction: bool = True,
) -> HierarchicalVIState:
    """Initialize hierarchical VI state.

    Initializes with:
    - Small random observable biases based on data statistics
    - Zero (or small) interaction weights
    - Mixture prior with spread components
    - Zero rho for starting point where approximate = exact posterior

    Args:
        key: JAX random key
        model: The hierarchical harmonium model
        data: Training data for initialization
        shape: Scale for random initialization
        zero_interaction: If True, start with W = 0

    Returns:
        Initial HierarchicalVIState
    """
    key, obs_key, mix_key, int_key = jax.random.split(key, 4)

    # Initialize observable biases from data
    obs_params = model.obs_man.initialize_from_sample(obs_key, data, shape=shape)

    # Initialize mixture prior parameters
    # Spread VonMises components across different angles
    vm_dim = model.vonmises_man.dim  # 2 * n_latent

    # Base VonMises parameters (low concentration)
    base_kappa = 0.5
    base_vm = jnp.zeros(vm_dim)
    for i in range(model.n_latent):
        base_vm = base_vm.at[2 * i].set(base_kappa)  # cos component

    # Component differences: spread means across angles
    n_components = model.n_clusters
    int_dim = vm_dim * (n_components - 1)
    int_mat = jnp.zeros(int_dim)

    # Each component gets a different mean direction spread
    for k in range(n_components - 1):
        angle_offset = 2 * jnp.pi * (k + 1) / n_components
        for i in range(model.n_latent):
            idx = k * vm_dim + 2 * i
            # Offset from base mean
            int_mat = int_mat.at[idx].set(base_kappa * (jnp.cos(angle_offset) - 1))
            int_mat = int_mat.at[idx + 1].set(base_kappa * jnp.sin(angle_offset))

    # Categorical prior: uniform
    cat_params = jnp.zeros(n_components - 1)

    # Join mixture parameters
    mixture_params = model.pst_man.join_coords(base_vm, int_mat, cat_params)

    # Add small noise to mixture parameters
    mixture_params = (
        mixture_params
        + jax.random.normal(mix_key, shape=mixture_params.shape) * shape * 0.1
    )

    # Interaction parameters
    if zero_interaction:
        int_params = jnp.zeros(model.int_man.dim)
    else:
        int_params = jax.random.normal(int_key, shape=(model.int_man.dim,)) * shape

    # Join all harmonium parameters
    harmonium_params = model.join_coords(obs_params, int_params, mixture_params)

    # Initialize rho = 0 (only VonMises dim, not full mixture)
    rho_params = jnp.zeros(model.vonmises_man.dim)

    return HierarchicalVIState(harmonium_params, rho_params)


def make_train_step_fn(
    model: BinomialMixtureVonMisesHarmonium,
    config: HierarchicalVIConfig,
):
    """Create a JIT-compiled training step function.

    Returns a function that computes gradients for a single batch.

    Args:
        model: The hierarchical harmonium model
        config: VI configuration

    Returns:
        A function (key, harm_params, rho_params, batch) -> (loss, harm_grad, rho_grad)
    """
    return make_elbo_loss_and_grad_fn(model, config)


def make_train_epoch_fn(
    train_step_fn: Any,
    opt_harm: optax.GradientTransformation,
    opt_rho: optax.GradientTransformation,
    batch_size: int,
):
    """Create a JIT-compiled epoch training function.

    Uses lax.scan for efficient batch processing.

    Args:
        train_step_fn: JIT-compiled step function
        opt_harm: Optax optimizer for harmonium params
        opt_rho: Optax optimizer for rho params
        batch_size: Batch size

    Returns:
        A function that trains one epoch and returns updated state
    """

    def train_epoch(
        key: Array,
        harmonium_params: Array,
        rho_params: Array,
        opt_harm_state: Any,
        opt_rho_state: Any,
        data: Array,
    ) -> tuple[Array, Array, Any, Any, Array, Array]:
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
            carry: tuple[Any, Any, Any, Any, Array],
            inputs: tuple[Array, Array],
        ) -> tuple[tuple[Any, Any, Any, Any, Array], None]:
            harm_params, rho_params, opt_h_state, opt_r_state, total_loss = carry
            batch_key, batch = inputs

            # Compute gradients
            loss, harm_grad, rho_grad = train_step_fn(
                batch_key, harm_params, rho_params, batch
            )

            # Update parameters
            harm_updates, new_opt_h_state = opt_harm.update(
                harm_grad, opt_h_state, harm_params
            )
            new_harm_params = optax.apply_updates(harm_params, harm_updates)

            rho_updates, new_opt_r_state = opt_rho.update(
                rho_grad, opt_r_state, rho_params
            )
            new_rho_params = optax.apply_updates(rho_params, rho_updates)

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
