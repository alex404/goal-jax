"""Boltzmann machines and related distributions for binary random variables.

This module provides:
- `Bernoullis`: Product of n independent Bernoullis (core exponential family)
- `DiagonalBoltzmann`: Mean-field wrapper with GeneralizedGaussian interface
- `CouplingMatrix`: Exponential family over moment matrices (x⊗x)
- `Boltzmann`: Full Boltzmann machine with pairwise coupling
"""

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
from ....geometry.exponential_family.combinators import AnalyticProduct
from ..categorical import Bernoulli
from .generalized import Euclidean, GeneralizedGaussian


class Bernoullis(AnalyticProduct[Bernoulli]):
    """Product of n independent Bernoulli distributions.

    Core exponential family for mean-field approximation of Boltzmann machines,
    analogous to CouplingMatrix for full Boltzmann machines.

    The parameters represent the bias/activation of each binary unit in a
    mean-field (no coupling) approximation.

    Attributes:
        n_neurons: Number of binary units
    """

    def __init__(self, n_neurons: int):
        """Create a product of n independent Bernoullis.

        Args:
            n_neurons: Number of binary units
        """
        super().__init__(Bernoulli(), n_neurons)

    @property
    def n_neurons(self) -> int:
        """Number of binary units."""
        return self.n_reps


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

    # Properties

    @property
    @override
    def dim(self) -> int:
        """Parameter dimension equals number of neurons."""
        return self.n_neurons

    @property
    @override
    def data_dim(self) -> int:
        """Data dimension equals number of neurons."""
        return self.n_neurons

    @property
    @override
    def loc_man(self) -> Bernoullis:
        """Location manifold is Bernoullis."""
        return Bernoullis(self.n_neurons)

    @property
    @override
    def shp_man(self) -> Bernoullis:
        """Shape manifold is Bernoullis (E[x²] = E[x] for binary)."""
        return Bernoullis(self.n_neurons)

    # Exponential family - delegate to shp_man

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Sufficient statistic: identity for independent binary."""
        return self.shp_man.sufficient_statistic(x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return self.shp_man.log_base_measure(x)

    @override
    def log_partition_function(self, params: Array) -> Array:
        """Log partition: sum of softplus (O(n) computation)."""
        return self.shp_man.log_partition_function(params)

    @override
    def negative_entropy(self, means: Array) -> Array:
        return self.shp_man.negative_entropy(means)

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Sample from independent Bernoullis."""
        return self.shp_man.sample(key, params, n)

    # GeneralizedGaussian interface

    @override
    def split_location_precision(self, params: Array) -> tuple[Array, Array]:
        """Split natural parameters into location and precision.

        Following Boltzmann's pattern: location is absorbed into shape params.
        Returns zeros for location since it's fully absorbed.

        The scaling by -2 matches the GeneralizedGaussian convention where
        precision relates to natural params by θ₂ = -½Σ⁻¹.
        """
        n = self.n_neurons
        return jnp.zeros(n), -2 * params

    @override
    def join_location_precision(self, location: Array, precision: Array) -> Array:
        """Join location and precision with absorption.

        Inverse of split_location_precision. Location is absorbed into
        the precision (they represent the same thing for independent binary).
        """
        return -0.5 * precision + location

    @override
    def split_mean_second_moment(self, means: Array) -> tuple[Array, Array]:
        """Split mean parameters into first and second moments.

        For independent binary variables, E[xᵢ²] = E[xᵢ] since x² = x for {0,1}.
        Both moments are identical - complete redundancy.
        """
        return means, means

    @override
    def join_mean_second_moment(self, mean: Array, second_moment: Array) -> Array:
        """Join mean and second moment.

        Returns second moment (which equals first moment for independent binary).
        The mean is redundant information.
        """
        return second_moment


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
    def log_partition_function(self, params: Array) -> Array:
        """Exact computation via enumeration."""
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
            ΔE = θ_kk + sum_{j≠k} θ_kj * x_j

        This extracts just the k-th row/column from the upper triangular storage
        rather than computing full O(n²) sufficient statistics.
        """
        n = self.n_neurons
        k = jnp.asarray(unit_idx)

        # Helper to get flat index for (i, j) where i <= j in upper triangular storage
        # Index formula: i * n - i * (i + 1) / 2 + j
        def get_triu_idx(i: Array, j: Array) -> Array:
            return i * n - i * (i + 1) // 2 + j

        # For each j in 0..n-1, get the parameter θ_kj (symmetric: θ_kj = θ_jk)
        j = jnp.arange(n)
        # If j >= k: element (k, j) is in row k -> use get_triu_idx(k, j)
        # If j < k: element (j, k) is in row j -> use get_triu_idx(j, k)
        row_idx = get_triu_idx(k, j)
        col_idx = get_triu_idx(j, k)
        idx = jnp.where(j >= k, row_idx, col_idx)

        # Extract the k-th row/column of the parameter matrix
        theta_k = params[idx]

        # Energy diff = θ_kk + sum_{j≠k} θ_kj * x_j
        # Rewrite as: θ_kk * (1 - x_k) + dot(theta_k, state)
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
    def loc_man(self) -> Bernoullis:
        """Return the Bernoullis location component manifold."""
        return Bernoullis(self.n_neurons)

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
    def log_partition_function(self, params: Array) -> Array:
        """Delegate to CouplingMatrix."""
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
        """Delegate to CouplingMatrix."""
        return self.shp_man.sample(key, params, n, n_burnin, n_thin)

    @override
    def gibbs_step(self, key: Array, params: Array, state: Array) -> Array:
        """Delegate to CouplingMatrix gibbs_step."""
        return self.shp_man.gibbs_step(key, params, state)

    # GeneralizedGaussian interface

    @override
    def split_location_precision(self, params: Array) -> tuple[Array, Array]:
        """Split for GeneralizedGaussian interface.

        The scaling by 1/2 for off-diagonal terms ensures that the dot product
        between natural parameters and triangular sufficient statistics equals
        the quadratic form $x^T \\Theta x$. Since off-diagonal elements appear twice
        in the outer product $x\\otimes x$ (as (i,j) and (j,i)), we scale by 1/2 to
        avoid double-counting in the energy computation.
        """
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
        """Split mean parameters into first and second moments.

        Mean parameters represent expected values of sufficient statistics.
        The first moment E[x] is extracted from the diagonal of the second
        moment matrix $E[x\\otimes x]$, since the diagonal contains $E[x_i^2] = E[x_i]$
        for binary variables.
        """
        n = self.n_neurons

        # Find diagonal positions in upper triangular storage
        rows, cols = jnp.triu_indices(n)
        i_diag = rows == cols

        # Extract diagonal elements (E[x_i] for binary variables)
        first_moment = means[i_diag]

        return first_moment, means

    @override
    def join_mean_second_moment(self, mean: Array, second_moment: Array) -> Array:
        """Join mean and second moment.

        For mean parameters, the second moment matrix already contains the
        correct diagonal elements (first moments), so we simply return the
        second moment as-is. The mean parameter is redundant information
        extracted from the diagonal.
        """
        return second_moment
