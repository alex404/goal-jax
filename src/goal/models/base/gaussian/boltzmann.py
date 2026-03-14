"""Boltzmann machines and related distributions for binary random variables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ....geometry import (
    Analytic,
    Differentiable,
    SquareMap,
    Symmetric,
)
from ..categorical import Bernoulli, Bernoullis
from .generalized import Euclidean, GeneralizedGaussian


@dataclass(frozen=True)
class DiagonalBoltzmann(
    GeneralizedGaussian[Bernoullis, Bernoullis],
    Analytic,
):
    """Mean-field Boltzmann machine with GeneralizedGaussian interface.

    Wraps Bernoullis (product of independent Bernoullis) to provide the
    GeneralizedGaussian interface for use in harmoniums (LGM) and embeddings.

    As a GeneralizedGaussian: both loc_man and shp_man are Bernoullis because
    for independent binary variables, E[x²] = E[x] (complete redundancy between
    first and second order statistics). This mirrors Boltzmann's structure where
    the location (bias) is absorbed into the shape.

    Analogous to how Boltzmann wraps CouplingMatrix.
    """

    n_neurons: int

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return self.n_neurons

    @property
    @override
    def data_dim(self) -> int:
        return self.n_neurons

    @property
    @override
    def loc_man(self) -> Bernoullis:
        return Bernoullis(self.n_neurons)

    @property
    @override
    def shp_man(self) -> Bernoullis:
        """E[x²] = E[x] for binary, so shape manifold equals location manifold."""
        return Bernoullis(self.n_neurons)

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        return self.shp_man.sufficient_statistic(x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return self.shp_man.log_base_measure(x)

    @override
    def log_partition_function(self, params: Array) -> Array:
        return self.shp_man.log_partition_function(params)

    @override
    def negative_entropy(self, means: Array) -> Array:
        return self.shp_man.negative_entropy(means)

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        return self.shp_man.sample(key, params, n)

    @override
    def split_location_precision(self, params: Array) -> tuple[Array, Array]:
        """Location is absorbed into shape; returns zeros for location, $-2\\theta$ for precision."""
        n = self.n_neurons
        return jnp.zeros(n), -2 * params

    @override
    def join_location_precision(self, location: Array, precision: Array) -> Array:
        """Inverse of split_location_precision with absorption."""
        return -0.5 * precision + location

    @override
    def split_mean_second_moment(self, means: Array) -> tuple[Array, Array]:
        """For binary variables $x^2 = x$, so both moments are identical."""
        return means, means

    @override
    def join_mean_second_moment(self, mean: Array, second_moment: Array) -> Array:
        """Returns second moment (equals first moment for binary)."""
        return second_moment


class CouplingMatrix(SquareMap[Euclidean], Differentiable):
    """Exponential family over moment matrices $x \\otimes x$ with Symmetric storage.

    Distribution: $p(x) \\propto \\exp(\\tr(\\Theta^T(x \\otimes x)))$
    """

    def __init__(self, n_neurons: int):
        super().__init__(Symmetric(), Euclidean(n_neurons))

    # Methods

    @property
    def n_neurons(self) -> int:
        """Number of neurons (dimensions)."""
        return self.matrix_shape[0]

    @property
    def states(self) -> Array:
        """All possible binary states via bit manipulation."""
        idx = jnp.arange(1 << self.n_neurons, dtype=jnp.uint32)
        return ((idx[:, None] >> jnp.arange(self.n_neurons)) & 1).astype(jnp.float32)

    # Overrides

    @property
    @override
    def data_dim(self) -> int:
        return self.n_neurons

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """$x \\otimes x$ stored as upper triangular."""
        return self.outer_product(x, x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    @override
    def log_partition_function(self, params: Array) -> Array:
        """Exact computation via enumeration over all $2^n$ states."""
        states = self.states

        def energy(state: Array) -> Array:
            suff_stat = self.sufficient_statistic(state)
            return jnp.dot(params, suff_stat)

        energies = jax.vmap(energy)(states)
        return jax.scipy.special.logsumexp(energies)

    def _unit_conditional_energy_diff(
        self, state: Array, unit_idx: Array | int, params: Array
    ) -> Array:
        """Compute energy difference for unit being 1 vs 0 - O(n) computation.

        The energy difference when flipping unit k from 0 to 1 is:
            $\\Delta E = \\theta_{kk} + \\sum_{j \\neq k} \\theta_{kj} x_j$

        This extracts just the k-th row/column from the upper triangular storage
        rather than computing full O(n²) sufficient statistics.
        """
        n = self.n_neurons
        k = jnp.asarray(unit_idx)

        # Helper to get flat index for (i, j) where i <= j in upper triangular storage
        # Index formula: i * n - i * (i + 1) / 2 + j
        def get_triu_idx(i: Array, j: Array) -> Array:
            return i * n - i * (i + 1) // 2 + j

        # For each j in 0..n-1, get the parameter \theta_kj (symmetric: \theta_kj = \theta_jk)
        j = jnp.arange(n)
        # If j >= k: element (k, j) is in row k -> use get_triu_idx(k, j)
        # If j < k: element (j, k) is in row j -> use get_triu_idx(j, k)
        row_idx = get_triu_idx(k, j)
        col_idx = get_triu_idx(j, k)
        idx = jnp.where(j >= k, row_idx, col_idx)

        # Extract the k-th row/column of the parameter matrix
        theta_k = params[idx]

        # Energy diff = \theta_kk + sum_{j != k} \theta_kj * x_j
        # Rewrite as: \theta_kk * (1 - x_k) + dot(theta_k, state)
        # This avoids creating a masked state array
        return theta_k[k] * (1 - state[k]) + jnp.dot(theta_k, state)

    def unit_conditional_prob(
        self, state: Array, unit_idx: Array | int, params: Array
    ) -> Array:
        """Compute P(x_unit = 1 | x_other) for a single unit."""
        energy_diff = self._unit_conditional_energy_diff(state, unit_idx, params)
        return Bernoulli().to_mean(jnp.atleast_1d(energy_diff))[0]

    def _gibbs_step(self, state: Array, key: Array, params: Array) -> Array:
        """Single Gibbs sampling step updating all units in random order."""
        perm = jax.random.permutation(key, self.n_neurons)

        def update_unit(state: Array, unit_idx: Array) -> tuple[Array, None]:
            prob = self.unit_conditional_prob(state, unit_idx, params)
            subkey = jax.random.fold_in(key, unit_idx)
            new_val = jax.random.bernoulli(subkey, prob).astype(state.dtype)
            return state.at[unit_idx].set(new_val), None

        final_state, _ = jax.lax.scan(update_unit, state, perm)
        return final_state

    @override
    def sample(
        self,
        key: Array,
        params: Array,
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
            return self._gibbs_step(state, subkey, params), None

        burned_state, _ = jax.lax.scan(burn_step, init_state, jnp.arange(n_burnin))

        # Sample collection with thinning
        def sample_with_thinning(state: Array, step: Array) -> tuple[Array, Array]:
            def thin_step(i: int, current_state: Array) -> Array:
                subkey = jax.random.fold_in(sample_key, n_burnin + step * n_thin + i)
                return self._gibbs_step(current_state, subkey, params)

            final_state = jax.lax.fori_loop(0, n_thin, thin_step, state)
            return final_state, final_state

        _, samples = jax.lax.scan(
            sample_with_thinning,
            burned_state,
            jnp.arange(n),
        )
        return samples

    @override
    def gibbs_step(self, key: Array, params: Array, state: Array) -> Array:
        """Gibbs step updating all units via conditional distributions."""
        return self._gibbs_step(state, key, params)


@dataclass(frozen=True)
class Boltzmann(
    GeneralizedGaussian[Bernoullis, CouplingMatrix],
    Differentiable,
):
    """Boltzmann machine with GeneralizedGaussian interface for harmoniums.

    Implements parameter absorption to handle the redundancy between bias and diagonal
    interaction terms for binary variables.
    """

    # Fields

    n_neurons: int

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return self.shp_man.dim

    @property
    @override
    def data_dim(self) -> int:
        return self.n_neurons

    @property
    @override
    def loc_man(self) -> Bernoullis:
        return Bernoullis(self.n_neurons)

    @property
    @override
    def shp_man(self) -> CouplingMatrix:
        return CouplingMatrix(self.n_neurons)

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """$x \\otimes x$ stored as upper triangular."""
        return self.shp_man.sufficient_statistic(x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    @override
    def log_partition_function(self, params: Array) -> Array:
        return self.shp_man.log_partition_function(params)

    @override
    def sample(
        self,
        key: Array,
        params: Array,
        n: int = 1,
        n_burnin: int = 1000,
        n_thin: int = 10,
    ) -> Array:
        return self.shp_man.sample(key, params, n, n_burnin, n_thin)

    @override
    def gibbs_step(self, key: Array, params: Array, state: Array) -> Array:
        return self.shp_man.gibbs_step(key, params, state)

    @override
    def split_location_precision(self, params: Array) -> tuple[Array, Array]:
        """Split with off-diagonal scaling to match energy $x^T \\Theta x$."""
        n = self.n_neurons

        # Boolean mask for diagonal elements in upper triangular storage
        i_diag = jnp.triu_indices(n)[0] == jnp.triu_indices(n)[1]

        # Scale off-diagonal by 1/2 for Natural parameters
        new_triangular = jnp.where(i_diag, params, params / 2.0)

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
    def split_mean_second_moment(self, means: Array) -> tuple[Array, Array]:
        """Extract first moment $E[x_i]$ from diagonal of $E[x \\otimes x]$ (since $x_i^2 = x_i$ for binary)."""
        n = self.n_neurons

        # Find diagonal positions in upper triangular storage
        rows, cols = jnp.triu_indices(n)
        i_diag = rows == cols

        # Extract diagonal elements (E[x_i] for binary variables)
        first_moment = means[i_diag]

        return first_moment, means

    @override
    def join_mean_second_moment(self, mean: Array, second_moment: Array) -> Array:
        """Second moment matrix already contains correct diagonal (first moments)."""
        return second_moment

    # Methods

    @property
    def states(self) -> Array:
        """All possible binary states."""
        return self.shp_man.states
