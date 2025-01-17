from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ....geometry import ExponentialFamily, Manifold, Natural, Point
from .generalized import GeneralizedGaussian


@dataclass(frozen=True)
class InteractionMatrix(Manifold):
    """Shape component of a Boltzmann machine.

    For n neurons, stores n(n-1)/2 interaction parameters
    (lower triangular without diagonal).
    """

    _dim: int  # Number of neurons

    @property
    @override
    def dim(self) -> int:
        """Number of neurons."""
        return self._dim

    @property
    def param_dim(self) -> int:
        """Number of parameters needed for interactions."""
        return (self.dim * (self.dim - 1)) // 2

    def to_dense(self, params: Point[Any, Self]) -> Array:
        """Convert parameters to dense matrix.

        Returns symmetric matrix with zeros on diagonal.
        """
        n = self.dim
        matrix = jnp.zeros((n, n))
        # Fill lower triangle
        indices = jnp.tril_indices(n, k=-1)
        matrix = matrix.at[indices].set(params.array)
        # Make symmetric
        return matrix + matrix.T

    def from_dense(self, matrix: Array) -> Point[Any, Self]:
        """Extract parameters from dense matrix.

        Takes lower triangle excluding diagonal.
        """
        indices = jnp.tril_indices(self.dim, k=-1)
        return Point(matrix[indices])


@dataclass(frozen=True)
class Boltzmann(GeneralizedGaussian[InteractionMatrix], ExponentialFamily):
    """Boltzmann machine distribution for small networks.

    Optimized for exact computation with ~10 neurons.
    """

    def __init__(self, n_neurons: int):
        # Initialize with Euclidean location space and interaction shape
        super().__init__(
            loc_man=Euclidean(n_neurons), shape_man=InteractionMatrix(n_neurons)
        )
        # Precompute all possible states (2^n binary vectors)
        self._states = jnp.array(list(itertools.product([0, 1], repeat=n_neurons)))

    @property
    def dim(self) -> int:
        return self.loc_man.dim

    def _compute_sufficient_statistic(self, x: Array) -> Array:
        """Compute sufficient statistics (x, off-diag(x⊗x))."""
        x = jnp.atleast_1d(x)
        # Extract triangular elements excluding diagonal
        tril_indices = jnp.tril_indices(self.dim, k=-1)
        xx_tril = jnp.outer(x, x)[tril_indices]
        return jnp.concatenate([x, xx_tril])

    def log_base_measure(self, x: Array) -> Array:
        return 0.0

    def _compute_log_partition(self, theta: Array) -> Array:
        """Compute log partition function by exact enumeration.

        Uses precomputed states to efficiently evaluate all configurations.
        """
        n = self.dim
        # Split parameters
        theta_loc = theta[:n]
        theta_int = theta[n:]

        # Compute biases for all states (2^n x 1 operation)
        energies = jnp.dot(self._states, theta_loc)

        # Add interaction terms efficiently
        int_matrix = self.shape_man.to_dense(Point(theta_int))
        # This computes all interaction terms in one vectorized operation
        energies += (
            jnp.sum(
                self._states[:, :, None] * int_matrix * self._states[:, None, :],
                axis=(1, 2),
            )
            / 2.0
        )

        return jax.scipy.special.logsumexp(energies)

    def unit_conditional(
        self, state: Array, unit_idx: int, p: Point[Natural, Self]
    ) -> Point[Natural, Euclidean]:
        """Compute conditional distribution for single unit."""
        loc, interactions = self.split_location_precision(p)
        int_matrix = self.shape_man.to_dense(interactions)

        # Compute energy difference between states
        energy = loc.params[unit_idx]
        energy += jnp.sum(int_matrix[unit_idx] * state)

        # Return natural parameters of Bernoulli
        return Point(energy)

    def gibbs_step(self, key: Array, state: Array, p: Point[Natural, Self]) -> Array:
        """Single Gibbs sampling step updating all units in random order."""
        n = self.dim
        # Get random permutation for update order
        perm = jax.random.permutation(key, n)

        def scan_fun(carry, idx):
            state = carry
            # Get conditional distribution
            energy = self.unit_conditional(state, idx, p).params
            # Sample new state
            prob = jax.nn.sigmoid(energy)
            subkey = jax.random.fold_in(key, idx)
            new_state = jax.random.bernoulli(subkey, prob)
            state = state.at[idx].set(new_state)
            return state, None

        final_state = jax.lax.scan(scan_fun, state, perm)[0]
        return final_state

    def sample(
        self,
        key: Array,
        p: Point[Natural, Self],
        n_samples: int = 1,
        n_burnin: int = 1000,
        n_thin: int = 10,
    ) -> Array:
        """Generate samples using Gibbs sampling.

        Args:
            key: PRNG key
            p: Parameters in natural coordinates
            n_samples: Number of samples to generate
            n_burnin: Number of initial steps to discard
            n_thin: Number of steps between samples
        """
        # Initialize random state
        init_key, sample_key = jax.random.split(key)
        state = jax.random.bernoulli(init_key, 0.5, shape=(self.dim,))

        # Burn-in phase
        for i in range(n_burnin):
            key, subkey = jax.random.split(sample_key)
            state = self.gibbs_step(subkey, state, p)

        # Collect samples
        samples = []
        for i in range(n_samples):
            # Thinning steps
            for j in range(n_thin):
                key, subkey = jax.random.split(sample_key)
                state = self.gibbs_step(subkey, state, p)
            samples.append(state)

        return jnp.stack(samples)
