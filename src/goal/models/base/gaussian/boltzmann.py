from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ....geometry import Differentiable, ExponentialFamily, Mean, Natural, Point
from .generalized import GeneralizedGaussian
from .normal import Euclidean


@dataclass(frozen=True)
class InteractionMatrix(ExponentialFamily):
    """Shape component of a Boltzmann machine.

    For n neurons, stores n(n-1)/2 interaction parameters
    (lower triangular without diagonal).

    For naming simplicity we sometimes refer to these parameters as "precisions", as they are analogous to the precisions in a Gaussian model, but strictly speaking they do not compose a precision matrix that is the inverse of the covariance matrix."""

    n_neurons: int  # Number of neurons

    @property
    @override
    def dim(self) -> int:
        """Number of parameters needed for interactions."""
        return (self.n_neurons * (self.n_neurons - 1)) // 2

    @property
    @override
    def data_dim(self) -> int:
        """Data dimension matches the number of neurons."""
        return self.n_neurons

    def to_dense(self, params: Point[Any, Self]) -> Array:
        """Convert parameters to dense matrix.

        Returns symmetric matrix with zeros on diagonal.
        """
        n = self.n_neurons
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
        indices = jnp.tril_indices(self.n_neurons, k=-1)
        return Point(matrix[indices])

    @override
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Compute off-diagonal outer product x⊗x for Boltzmann interactions."""
        x = jnp.atleast_1d(x)
        # Compute full outer product
        outer_product = jnp.outer(x, x)
        # Extract off-diagonal elements (Boltzmann constraint)
        indices = jnp.tril_indices(self.n_neurons, k=-1)
        off_diagonal = outer_product[indices]
        return self.mean_point(off_diagonal)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Base measure for binary variables (no normalizing constant needed)."""
        return jnp.array(0.0)


@dataclass(frozen=True)
class Boltzmann(Differentiable, GeneralizedGaussian[Euclidean, InteractionMatrix]):
    """Boltzmann machine distribution for small networks.

    Designed for exact computation with <100 neurons.

    The distribution follows the form:
    p(x) ∝ exp(θ·x + x'Jx)

    where θ are bias parameters and J is the interaction matrix.
    """

    n_neurons: int
    _states: Array = None  # Will be computed in __post_init__

    def __post_init__(self):
        # Precompute all possible states (2^n binary vectors)
        states = jnp.array(list(itertools.product([0, 1], repeat=self.n_neurons)))
        # Use object.__setattr__ to bypass frozen dataclass restriction
        object.__setattr__(self, "_states", states)

    @property
    @override
    def dim(self) -> int:
        """Parameter dimension: n bias + n(n-1)/2 interaction terms."""
        return self.n_neurons + (self.n_neurons * (self.n_neurons - 1)) // 2

    @property
    @override
    def data_dim(self) -> int:
        """Data dimension equals number of neurons."""
        return self.n_neurons

    @property
    @override
    def location_manifold(self) -> Euclidean:
        """Bias parameters (location component)."""
        return Euclidean(self.n_neurons)

    @property
    @override
    def shape_manifold(self) -> InteractionMatrix:
        """Interaction parameters (shape component)."""
        return InteractionMatrix(self.n_neurons)

    @override
    def split_location_precision(
        self, params: Point[Natural, Self]
    ) -> tuple[Point[Natural, Euclidean], Point[Natural, InteractionMatrix]]:
        """Split into bias and interaction parameters."""
        n = self.n_neurons
        bias_params = params.array[:n]
        int_params = params.array[n:]

        bias_point = self.location_manifold.natural_point(bias_params)
        int_point = self.shape_manifold.natural_point(int_params)

        return bias_point, int_point

    @override
    def join_location_precision(
        self,
        bias: Point[Natural, Euclidean],
        interactions: Point[Natural, InteractionMatrix],
    ) -> Point[Natural, Self]:
        """Join bias and interaction parameters."""
        combined = jnp.concatenate([bias.array, interactions.array])
        return self.natural_point(combined)

    @override
    def split_mean_second_moment(
        self, params: Point[Mean, Self]
    ) -> tuple[Point[Mean, Euclidean], Point[Mean, InteractionMatrix]]:
        """Split into mean and correlation parameters."""
        n = self.n_neurons
        mean_params = params.array[:n]
        corr_params = params.array[n:]

        mean_point = self.location_manifold.mean_point(mean_params)
        corr_point = self.shape_manifold.mean_point(corr_params)

        return mean_point, corr_point

    @override
    def join_mean_second_moment(
        self, mean: Point[Mean, Euclidean], correlations: Point[Mean, InteractionMatrix]
    ) -> Point[Mean, Self]:
        """Join mean and correlation parameters."""
        combined = jnp.concatenate([mean.array, correlations.array])
        return self.mean_point(combined)

    @override
    def _compute_second_moment(self, x: Array) -> Point[Mean, InteractionMatrix]:
        """Compute off-diagonal correlations for Boltzmann machines."""
        return self.shape_manifold.sufficient_statistic(x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Base measure for binary variables."""
        return jnp.array(0.0)

    # Exponential family coordinate transformations
    # For small Boltzmann machines, we can compute exact transformations
    # by brute force evaluation over all 2^n states

    @override
    def log_partition_function(self, params: Point[Natural, Self]) -> Array:
        """Compute log partition function by exact enumeration."""
        bias, interactions = self.split_location_precision(params)

        # Compute bias terms
        energies = jnp.dot(self._states, bias.array)

        # Add interaction terms
        int_matrix = self.shape_manifold.to_dense(interactions)
        interaction_energies = 0.5 * jnp.sum(
            self._states[:, :, None] * int_matrix * self._states[:, None, :],
            axis=(1, 2),
        )

        total_energies = energies + interaction_energies
        return jax.scipy.special.logsumexp(total_energies)

    @override
    def sample(
        self,
        key: Array,
        params: Point[Natural, Self],
        n_samples: int = 1,
        n_burnin: int = 100,
        n_thin: int = 1,
    ) -> Array:
        """Generate samples using Gibbs sampling."""
        # Initialize random state
        init_key, sample_key = jax.random.split(key)
        state = jax.random.bernoulli(init_key, 0.5, shape=(self.n_neurons,))

        # Extract parameters
        bias, interactions = self.split_location_precision(params)
        int_matrix = self.shape_manifold.to_dense(interactions)

        def gibbs_step(state: Array, key: Array) -> Array:
            """Single Gibbs step updating all units in random order."""
            perm = jax.random.permutation(key, self.n_neurons)

            def update_unit(state: Array, unit_idx: int) -> Array:
                # Compute conditional energy
                energy = bias.array[unit_idx] + jnp.sum(int_matrix[unit_idx] * state)
                # Sample new state
                prob = jax.nn.sigmoid(energy)
                subkey = jax.random.fold_in(key, unit_idx)
                new_val = jax.random.bernoulli(subkey, prob)
                return state.at[unit_idx].set(new_val)

            # Update all units in random order
            for idx in perm:
                state = update_unit(state, idx)
            return state

        # Burn-in phase
        for _ in range(n_burnin):
            sample_key, subkey = jax.random.split(sample_key)
            state = gibbs_step(state, subkey)

        # Collect samples
        samples = []
        for _ in range(n_samples):
            # Thinning steps
            for _ in range(n_thin):
                sample_key, subkey = jax.random.split(sample_key)
                state = gibbs_step(state, subkey)
            samples.append(state)

        return jnp.stack(samples)
