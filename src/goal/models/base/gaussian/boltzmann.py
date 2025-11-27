from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ....geometry import (
    Differentiable,
    SquareMap,
    Symmetric,
)
from .generalized import Euclidean, GeneralizedGaussian


class CouplingMatrix(SquareMap[Euclidean], Differentiable):
    """Exponential family over moment matrices.

    Core implementation of Boltzmann machines as exponential family distributions
    with moment matrix sufficient statistic $x \\otimes x$.

    Distribution: $p(x) \\propto \\exp(\\tr(\\Theta^T(x \\otimes x)))$

    Uses Symmetric matrix representation for efficient storage and operations.
    """

    # Constructor

    def __init__(self, n_neurons: int):
        super().__init__(Symmetric(), Euclidean(n_neurons))

    # Properties

    @property
    def n_neurons(self) -> int:
        """Number of neurons (dimensions)."""
        return self.matrix_shape[0]

    @property
    @override
    def data_dim(self) -> int:
        """Data dimension equals number of neurons."""
        return self.n_neurons

    @property
    def states(self) -> Array:
        """All possible binary states via bit manipulation."""
        idx = jnp.arange(1 << self.n_neurons, dtype=jnp.uint32)
        return ((idx[:, None] >> jnp.arange(self.n_neurons)) & 1).astype(jnp.float32)

    # Core exponential family

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Sufficient statistic is $x \\otimes x$ stored as upper triangular."""
        return self.outer_product(x, x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    @override
    def log_partition_function(self, natural_params: Array) -> Array:
        """Exact computation via enumeration."""
        states = self.states

        def energy(state: Array) -> Array:
            suff_stat = self.sufficient_statistic(state)
            return jnp.dot(natural_params, suff_stat)

        energies = jax.vmap(energy)(states)
        return jax.scipy.special.logsumexp(energies)

    def _unit_conditional_energy_diff(
        self, state: Array, unit_idx: Array | int, natural_params: Array
    ) -> Array:
        """Compute energy difference for unit being 1 vs 0 using sufficient statistics."""
        # Create states with unit_idx = 0 and unit_idx = 1
        state_0 = state.at[unit_idx].set(0.0)
        state_1 = state.at[unit_idx].set(1.0)

        # Compute sufficient statistics for both states
        suff_stat_0 = self.sufficient_statistic(state_0)
        suff_stat_1 = self.sufficient_statistic(state_1)

        # Energy difference = $\\theta^T (s(x_1) - s(x_0))$
        return jnp.dot(natural_params, suff_stat_1 - suff_stat_0)

    def unit_conditional_prob(
        self, state: Array, unit_idx: Array | int, natural_params: Array
    ) -> Array:
        """Compute P(x_unit = 1 | x_other) for a single unit."""
        energy_diff = self._unit_conditional_energy_diff(
            state, unit_idx, natural_params
        )
        return jax.nn.sigmoid(energy_diff)

    def _gibbs_step(self, state: Array, key: Array, natural_params: Array) -> Array:
        """Single Gibbs sampling step updating all units in random order."""
        perm = jax.random.permutation(key, self.n_neurons)

        def update_unit(state: Array, unit_idx: Array) -> tuple[Array, None]:
            prob = self.unit_conditional_prob(state, unit_idx, natural_params)
            subkey = jax.random.fold_in(key, unit_idx)
            new_val = jax.random.bernoulli(subkey, prob).astype(state.dtype)
            return state.at[unit_idx].set(new_val), None

        final_state, _ = jax.lax.scan(update_unit, state, perm)
        return final_state

    @override
    def sample(
        self,
        key: Array,
        natural_params: Array,
        n: int = 1,
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
        def burn_step(state: Array, step: Array) -> tuple[Array, None]:
            subkey = jax.random.fold_in(sample_key, step)
            return self._gibbs_step(state, subkey, natural_params), None

        burned_state, _ = jax.lax.scan(burn_step, init_state, jnp.arange(n_burnin))

        # Sample collection with thinning
        def sample_with_thinning(state: Array, step: Array) -> tuple[Array, Array]:
            def thin_step(i: int, current_state: Array) -> Array:
                subkey = jax.random.fold_in(sample_key, n_burnin + step * n_thin + i)
                return self._gibbs_step(current_state, subkey, natural_params)

            final_state = jax.lax.fori_loop(0, n_thin, thin_step, state)
            return final_state, final_state

        _, samples = jax.lax.scan(
            sample_with_thinning,
            burned_state,
            jnp.arange(n),
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
    def sufficient_statistic(self, x: Array) -> Array:
        """Sufficient statistic is $x \\otimes x$ stored as upper triangular."""
        return self.shp_man.sufficient_statistic(x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    @override
    def log_partition_function(self, natural_params: Array) -> Array:
        """Delegate to CouplingMatrix."""
        return self.shp_man.log_partition_function(natural_params)

    @override
    def sample(
        self,
        key: Array,
        natural_params: Array,
        n: int = 1,
        n_burnin: int = 1000,
        n_thin: int = 10,
    ) -> Array:
        """Delegate to CouplingMatrix."""
        return self.shp_man.sample(key, natural_params, n, n_burnin, n_thin)

    # GeneralizedGaussian interface

    @override
    def split_location_precision(self, natural_params: Array) -> tuple[Array, Array]:
        """Split for GeneralizedGaussian interface.

        The scaling by 1/2 for off-diagonal terms ensures that the dot product
        between natural parameters and triangular sufficient statistics equals
        the quadratic form $x^T \\Theta x$. Since off-diagonal elements appear twice
        in the outer product $x\\otimes x$ (as (i,j) and (j,i)), we scale by 1/2 to
        avoid double-counting in the energy computation.
        """
        triangular_params = natural_params
        n = self.n_neurons

        # Boolean mask for diagonal elements in upper triangular storage
        i_diag = jnp.triu_indices(n)[0] == jnp.triu_indices(n)[1]

        # Scale off-diagonal by 1/2 for Natural parameters
        new_triangular = jnp.where(i_diag, triangular_params, triangular_params / 2.0)

        return jnp.zeros(n), -2 * new_triangular

    @override
    def join_location_precision(
        self,
        location: Array,
        precision: Array,
    ) -> Array:
        """Join with parameter absorption."""
        triangular_params = -0.5 * precision
        n = self.n_neurons

        # Boolean mask for diagonal elements in upper triangular storage
        rows, cols = jnp.triu_indices(n)
        i_diag = rows == cols

        # Broadcast location array to match triangular storage
        # For diagonal elements at position i in triangular array,
        # rows[i] gives the corresponding index in location array
        location_broadcast = location[rows]

        return jnp.where(
            i_diag,
            triangular_params + location_broadcast,  # diagonal: absorb location
            triangular_params * 2.0,  # off-diagonal: scale by 2
        )

    @override
    def split_mean_second_moment(self, mean_params: Array) -> tuple[Array, Array]:
        """Split mean parameters into first and second moments.

        Mean parameters represent expected values of sufficient statistics.
        The first moment E[x] is extracted from the diagonal of the second
        moment matrix $E[x\\otimes x]$, since the diagonal contains $E[x_i^2] = E[x_i]$
        for binary variables.
        """
        triangular_params = mean_params
        n = self.n_neurons

        # Find diagonal positions in upper triangular storage
        rows, cols = jnp.triu_indices(n)
        i_diag = rows == cols

        # Extract diagonal elements (E[x_i] for binary variables)
        first_moment = triangular_params[i_diag]

        return first_moment, triangular_params

    @override
    def join_mean_second_moment(self, mean: Array, second_moment: Array) -> Array:
        """Join mean and second moment.

        For mean parameters, the second moment matrix already contains the
        correct diagonal elements (first moments), so we simply return the
        second moment as-is. The mean parameter is redundant information
        extracted from the diagonal.
        """
        return second_moment
