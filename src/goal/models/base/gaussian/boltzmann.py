from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ....geometry import (
    Differentiable,
    Mean,
    Natural,
    Point,
)
from .generalized import Euclidean, GeneralizedGaussian


@dataclass(frozen=True)
class CouplingMatrix(Differentiable):
    """Exponential family over moment matrices.

    Core implementation of Boltzmann machines as exponential family distributions
    with moment matrix sufficient statistic $x \\otimes x$.

    Distribution: $p(x) \\propto \\exp(\\tr(\\Theta^T(x \\otimes x)))$
    """

    n_neurons: int

    @property
    @override
    def dim(self) -> int:
        """Parameter dimension: triangular matrix (n(n+1)/2 elements)."""
        n = self.n_neurons
        return (n * (n + 1)) // 2

    @property
    @override
    def data_dim(self) -> int:
        """Data dimension equals number of neurons."""
        return self.n_neurons

    @property
    def states(self) -> Array:
        """All possible binary states."""
        return jnp.array(
            list(itertools.product([0, 1], repeat=self.n_neurons)), dtype=jnp.float32
        )

    # Core exponential family

    @override
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Sufficient statistic is $x \\otimes x$ stored as triangular."""
        x = jnp.atleast_1d(x).astype(jnp.float32)
        outer_product = jnp.outer(x, x)
        tril_indices = jnp.tril_indices(self.n_neurons)
        triangular_elements = outer_product[tril_indices]
        return Point(triangular_elements)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    @override
    def log_partition_function(self, params: Point[Natural, Self]) -> Array:
        """Exact computation via enumeration."""
        # Compute sufficient statistics for all states without lambda
        # states shape: (2^n, n)
        # For each state, compute outer product and extract triangular elements
        states = self.states

        # Vectorized outer products: (2^n, n, n)
        outer_products = jnp.einsum("bi,bj->bij", states, states)

        # Extract lower triangular elements for each state
        tril_indices = jnp.tril_indices(self.n_neurons)
        suff_stats = outer_products[:, tril_indices[0], tril_indices[1]]

        # Compute energies and log partition function
        energies = jnp.dot(suff_stats, params.array)
        return jax.scipy.special.logsumexp(energies)

    def _unit_conditional_energy_diff(
        self, state: Array, unit_idx: int, params: Point[Natural, Self]
    ) -> Array:
        """Compute energy difference for unit being 1 vs 0 using sufficient statistics."""
        # Create states with unit_idx = 0 and unit_idx = 1
        state_0 = state.at[unit_idx].set(0.0)
        state_1 = state.at[unit_idx].set(1.0)

        # Compute sufficient statistics for both states
        suff_stat_0 = self.sufficient_statistic(state_0)
        suff_stat_1 = self.sufficient_statistic(state_1)

        # Energy difference = $\\theta^T (s(x_1) - s(x_0))$
        return jnp.dot(params.array, suff_stat_1.array - suff_stat_0.array)

    def unit_conditional_prob(
        self, state: Array, unit_idx: int, params: Point[Natural, Self]
    ) -> Array:
        """Compute P(x_unit = 1 | x_other) for a single unit."""
        energy_diff = self._unit_conditional_energy_diff(state, unit_idx, params)
        return jax.nn.sigmoid(energy_diff)

    def _gibbs_step(
        self, state: Array, key: Array, params: Point[Natural, Self]
    ) -> Array:
        """Single Gibbs sampling step updating all units in random order."""
        perm = jax.random.permutation(key, self.n_neurons)

        def update_unit(state: Array, unit_idx: int) -> tuple[Array, None]:
            prob = self.unit_conditional_prob(state, unit_idx, params)
            subkey = jax.random.fold_in(key, unit_idx)
            new_val = jax.random.bernoulli(subkey, prob).astype(state.dtype)
            return state.at[unit_idx].set(new_val), None

        final_state, _ = jax.lax.scan(update_unit, state, perm)  # pyright: ignore[reportArgumentType]
        return final_state

    @override
    def sample(
        self,
        key: Array,
        params: Point[Natural, Self],
        n_samples: int = 1,
        n_burnin: int = 1000,
        n_thin: int = 10,
    ) -> Array:
        """Generate samples using Gibbs sampling."""
        # Initialize
        init_key, sample_key = jax.random.split(key)
        init_state = jax.random.bernoulli(
            init_key, 0.5, shape=(self.n_neurons,)
        ).astype(jnp.float32)

        # Burn-in
        def burn_step(state: Array, step: int) -> tuple[Array, None]:
            subkey = jax.random.fold_in(sample_key, step)
            return self._gibbs_step(state, subkey, params), None

        burned_state, _ = jax.lax.scan(burn_step, init_state, jnp.arange(n_burnin))  # pyright: ignore[reportArgumentType]

        # Sample collection with thinning
        def sample_with_thinning(state: Array, step: int) -> tuple[Array, Array]:
            def thin_step(i: int, current_state: Array) -> Array:
                subkey = jax.random.fold_in(sample_key, n_burnin + step * n_thin + i)
                return self._gibbs_step(current_state, subkey, params)

            final_state = jax.lax.fori_loop(0, n_thin, thin_step, state)
            return final_state, final_state

        _, samples = jax.lax.scan(
            sample_with_thinning,
            burned_state,
            jnp.arange(n_samples),  # pyright: ignore[reportArgumentType]
        )
        return samples


@dataclass(frozen=True)
class Boltzmann(GeneralizedGaussian[CouplingMatrix], CouplingMatrix):
    """Boltzmann machine with GeneralizedGaussian interface for harmoniums.

    Extends CouplingMatrix with the GeneralizedGaussian-compatible interface methods
    needed for Linear Gaussian Models and other harmonium applications. Implements
    parameter absorption to handle the redundancy between bias and diagonal
    interaction terms for binary variables.
    """

    # Inherits all core functionality from CouplingMatrix
    # Implements GeneralizedGaussian-compatible interface methods below

    @property
    @override
    def loc_man(self) -> Euclidean:
        """Return the Euclidean location component manifold."""
        return Euclidean(self.n_neurons)

    @property
    @override
    def shp_man(self) -> CouplingMatrix:
        """Return the coupling matrix shape component manifold."""
        return CouplingMatrix(self.n_neurons)

    @override
    def split_location_precision(
        self, params: Point[Natural, Self]
    ) -> tuple[Point[Natural, Euclidean], Point[Natural, CouplingMatrix]]:
        """Split for GeneralizedGaussian interface.

        The scaling by 1/2 for off-diagonal terms ensures that the dot product
        between natural parameters and triangular sufficient statistics equals
        the quadratic form $x^T \\Theta x$. Since off-diagonal elements appear twice
        in the outer product $x\\otimes x$ (as (i,j) and (j,i)), we scale by 1/2 to
        avoid double-counting in the energy computation.
        """
        triangular_params = params.array
        n = self.n_neurons

        # Compute off-diagonal indices statically
        diagonal_set = {i * (i + 1) // 2 + i for i in range(n)}
        off_diag_indices = jnp.array([i for i in range(len(triangular_params)) if i not in diagonal_set])

        # Scale off-diagonal by 1/2 for Natural parameters
        new_triangular = triangular_params.at[off_diag_indices].divide(2.0)

        return Point(jnp.zeros(n)), Point(new_triangular)

    @override
    def join_location_precision(
        self,
        location: Point[Natural, Euclidean],
        precision: Point[Natural, CouplingMatrix],
    ) -> Point[Natural, Self]:
        """Join with parameter absorption."""
        triangular_params = precision.array
        n = self.n_neurons

        # Compute diagonal indices statically
        diagonal_indices = jnp.array([i * (i + 1) // 2 + i for i in range(n)])
        diagonal_terms = triangular_params[diagonal_indices]

        # Compute off-diagonal indices statically
        diagonal_set = {i * (i + 1) // 2 + i for i in range(n)}
        off_diag_indices = jnp.array([i for i in range(len(triangular_params)) if i not in diagonal_set])

        # ABSORPTION: Add diagonal terms to location
        absorbed_diagonal = location.array + diagonal_terms

        # Scale off-diagonal terms by 2 for Natural parameters
        final_triangular = triangular_params.at[diagonal_indices].set(absorbed_diagonal)
        final_triangular = final_triangular.at[off_diag_indices].multiply(2.0)

        return Point(final_triangular)

    @override
    def split_mean_second_moment(
        self, params: Point[Mean, Self]
    ) -> tuple[Point[Mean, Euclidean], Point[Mean, CouplingMatrix]]:
        """Split mean parameters into first and second moments.

        Mean parameters represent expected values of sufficient statistics.
        The first moment E[x] is extracted from the diagonal of the second
        moment matrix $E[x\\otimes x]$, since the diagonal contains $E[x_i^2] = E[x_i]$
        for binary variables.
        """
        triangular_params = params.array
        n = self.n_neurons

        # Extract diagonal elements (first moments E[xᵢ])
        diagonal_indices = jnp.array([i * (i + 1) // 2 + i for i in range(n)])
        first_moment = triangular_params[diagonal_indices]

        return Point(first_moment), Point(triangular_params)

    @override
    def join_mean_second_moment(
        self, mean: Point[Mean, Euclidean], second_moment: Point[Mean, CouplingMatrix]
    ) -> Point[Mean, Self]:
        """Join mean and second moment.

        For mean parameters, the second moment matrix already contains the
        correct diagonal elements (first moments), so we simply return the
        second moment as-is. The mean parameter is redundant information
        extracted from the diagonal.
        """
        return Point(second_moment.array)
