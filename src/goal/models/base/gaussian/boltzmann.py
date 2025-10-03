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
    SquareMap,
    Symmetric,
)
from .generalized import Euclidean, GeneralizedGaussian


class CouplingMatrix(SquareMap[Symmetric, Euclidean], Differentiable):
    """Exponential family over moment matrices.

    Core implementation of Boltzmann machines as exponential family distributions
    with moment matrix sufficient statistic $x \\otimes x$.

    Distribution: $p(x) \\propto \\exp(\\tr(\\Theta^T(x \\otimes x)))$

    Uses Symmetric matrix representation for efficient storage and operations.
    """

    # Constructor

    def __init__(self, n_neurons: int):
        super().__init__(Symmetric(), Euclidean(n_neurons))
        # Cache states for efficiency (computed once per instance)
        self._states: Array = jnp.array(
            list(itertools.product([0, 1], repeat=n_neurons))
        )

    # Properties

    @property
    def n_neurons(self) -> int:
        """Number of neurons (dimensions)."""
        return self.shape[0]

    @property
    @override
    def data_dim(self) -> int:
        """Data dimension equals number of neurons."""
        return self.n_neurons

    @property
    def states(self) -> Array:
        """All possible binary states."""
        return self._states

    # Core exponential family

    @override
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Sufficient statistic is $x \\otimes x$ stored as upper triangular."""
        euclidean = Euclidean(self.data_dim)
        x_point: Point[Mean, Euclidean] = euclidean.mean_point(x)
        return self.outer_product(x_point, x_point)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    @override
    def log_partition_function(self, params: Point[Natural, Self]) -> Array:
        """Exact computation via enumeration."""
        # Compute sufficient statistics for all states
        states = self.states

        # Compute energies for all states using sufficient statistic
        def energy(state: Array) -> Array:
            suff_stat = self.sufficient_statistic(state)
            return self.dot(params, suff_stat)

        energies = jax.vmap(energy)(states)
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
            new_val = jax.random.bernoulli(subkey, prob)
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
        init_state = jax.random.bernoulli(init_key, 0.5, shape=(self.n_neurons,))

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
class Boltzmann(
    GeneralizedGaussian[CouplingMatrix],
    Differentiable,
):
    """Boltzmann machine with GeneralizedGaussian interface for harmoniums.

    Implements parameter absorption to handle the redundancy between bias and diagonal
    interaction terms for binary variables.
    """

    # Fields
    n_neurons: int

    # Properties

    @property
    @override
    def dim(self) -> int:
        """Parameter dimension: triangular matrix (n(n+1)/2 elements)."""
        return self.shp_man.dim

    @property
    @override
    def data_dim(self) -> int:
        """Data dimension equals number of neurons."""
        return self.n_neurons

    @property
    def states(self) -> Array:
        """All possible binary states."""
        return self.shp_man.states

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

    # Core exponential family

    @override
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Sufficient statistic is $x \\otimes x$ stored as upper triangular."""
        return Point(self.shp_man.sufficient_statistic(x).array)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    @override
    def log_partition_function(self, params: Point[Natural, Self]) -> Array:
        """Exact computation via enumeration."""
        # Compute energies for all states using sufficient statistic
        states = self.states

        def energy(state: Array) -> Array:
            suff_stat = self.sufficient_statistic(state)
            return self.dot(params, suff_stat)

        energies = jax.vmap(energy)(states)
        return jax.scipy.special.logsumexp(energies)

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
        init_state = jax.random.bernoulli(init_key, 0.5, shape=(self.n_neurons,))

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
            new_val = jax.random.bernoulli(subkey, prob)
            return state.at[unit_idx].set(new_val), None

        final_state, _ = jax.lax.scan(update_unit, state, perm)  # pyright: ignore[reportArgumentType]
        return final_state

    # GeneralizedGaussian interface

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

        # Boolean mask for diagonal elements in upper triangular storage
        i_diag = jnp.triu_indices(n)[0] == jnp.triu_indices(n)[1]

        # Scale off-diagonal by 1/2 for Natural parameters
        new_triangular = jnp.where(i_diag, triangular_params, triangular_params / 2.0)

        return Point(jnp.zeros(n)), Point(-2 * new_triangular)

    @override
    def join_location_precision(
        self,
        location: Point[Natural, Euclidean],
        precision: Point[Natural, CouplingMatrix],
    ) -> Point[Natural, Self]:
        """Join with parameter absorption."""
        triangular_params = -0.5 * precision.array
        n = self.n_neurons

        # Boolean mask for diagonal elements in upper triangular storage
        i_diag = jnp.triu_indices(n)[0] == jnp.triu_indices(n)[1]

        # Extract diagonal terms for absorption
        diagonal_terms = triangular_params[i_diag]

        # ABSORPTION: Add diagonal terms to location
        absorbed_diagonal = location.array + diagonal_terms

        # Scale off-diagonal by 2, then set diagonal to absorbed values
        scaled_params = jnp.where(i_diag, triangular_params, triangular_params * 2.0)

        # Replace diagonal values with absorbed values
        diag_positions = jnp.where(i_diag)[0]
        final_triangular = scaled_params.at[diag_positions].set(absorbed_diagonal)

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

        # Boolean mask for diagonal elements in upper triangular storage
        i_diag = jnp.triu_indices(n)[0] == jnp.triu_indices(n)[1]

        # Extract diagonal elements (first moments E[xᵢ])
        first_moment = triangular_params[i_diag]

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
