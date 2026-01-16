"""Probabilistic population codes for neural encoding.

Population codes model neural responses as Poisson spike counts with
tuning curves over a stimulus space. Approximate conjugation via
linear regression enables tractable posterior inference.

**Model Structure**:

A population code is a harmonium with:
- Observable: Poisson population (spike counts from n neurons)
- Latent: Stimulus distribution (VonMises for circular stimuli)
- Interaction: Tuning curve parameters connecting stimulus to firing rates

The tuning curve for neuron i is:
    log λ_i(z) = baseline_i + gain_i · cos(z - preferred_i)

**Approximate Conjugation**:

The log-partition function of the likelihood is approximated via regression:
    ψ_X(z) ≈ χ + ρᵀ·T(z)

where T(z) = (cos(z), sin(z)) are the VonMises sufficient statistics.
This makes the posterior tractable: p(z|x) ∝ exp((θ_Z + Θ_XZ^T·x)^T·T(z))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    AnalyticProduct,
    Rectangular,
)
from ...geometry.exponential_family.harmonium import SymmetricConjugated
from ...geometry.manifold.linear import EmbeddedMap
from ...geometry.manifold.embedding import IdentityEmbedding
from ..base.poisson import Poisson
from ..base.von_mises import VonMises

# Type alias for Poisson population
type PoissonPopulation = AnalyticProduct[Poisson]


@dataclass(frozen=True)
class VonMisesPopulationCode(SymmetricConjugated[PoissonPopulation, VonMises]):
    """Population code with Poisson observables and Von Mises stimulus.

    Models neural responses where each neuron has a tuning curve over a
    circular stimulus (e.g., orientation). Uses approximate conjugation
    via regression to enable tractable posterior computation.

    The tuning curve for neuron i is:
        log λ_i(z) = baseline_i + gain_i · cos(z - preferred_i)

    In natural parameters, this becomes:
        θ_x[i] = baseline_i  (observable bias)
        Θ_xz[i,:] = [gain_i·cos(pref_i), gain_i·sin(pref_i)]  (interaction)

    Attributes:
        n_neurons: Number of neurons in the population
        n_grid_points: Number of grid points for regression (default 100)
    """

    n_neurons: int
    n_grid_points: int = 100

    # Properties

    @property
    @override
    def lat_man(self) -> VonMises:
        """Von Mises stimulus manifold."""
        return VonMises()

    @property
    @override
    def obs_man(self) -> PoissonPopulation:
        """Poisson population manifold."""
        return AnalyticProduct(Poisson(), self.n_neurons)

    @property
    @override
    def int_man(self) -> EmbeddedMap[VonMises, PoissonPopulation]:
        """Interaction map (tuning curves).

        Maps VonMises sufficient statistics to Poisson natural parameters.
        Shape: (n_neurons, 2) where each row is the tuning weights for one neuron.
        """
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(self.lat_man),
            IdentityEmbedding(self.obs_man),
        )

    # Conjugation

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute approximate conjugation parameters via regression.

        Fits χ + ρᵀ·T(z) ≈ ψ_X(z) where ψ_X(z) is the log-partition
        of the likelihood at stimulus z.

        Algorithm:
        1. Create grid of stimulus values z_1, ..., z_M in [0, 2π)
        2. For each z_m, compute log-partition: ψ(z_m) = Σ_i exp(θ_i + Θ_i·T(z_m))
        3. Regress: [1, T(z_m)] @ [χ, ρ] ≈ ψ(z_m)
        4. Return ρ (χ is used in log_partition_function)

        Args:
            lkl_params: Likelihood parameters [obs_params, int_params]

        Returns:
            Conjugation parameters ρ with shape (2,)
        """
        obs_params, int_params = self.lkl_fun_man.split_coords(lkl_params)

        # Reshape interaction params to matrix form (n_neurons, 2)
        int_matrix = int_params.reshape(self.n_neurons, 2)

        # Generate grid of stimulus values
        grid = jnp.linspace(0, 2 * jnp.pi, self.n_grid_points, endpoint=False)

        def log_partition_at(z: Array) -> Array:
            """Compute Σ_i exp(θ_i + Θ_i·T(z)) at a single stimulus."""
            suff = self.lat_man.sufficient_statistic(z)  # [cos(z), sin(z)]
            log_rates = obs_params + int_matrix @ suff  # (n_neurons,)
            return jnp.sum(jnp.exp(log_rates))

        # Compute log-partition at each grid point
        log_partitions = jax.vmap(log_partition_at)(grid)  # (n_grid,)

        # Build design matrix: [1, cos(z), sin(z)]
        suff_stats = jax.vmap(self.lat_man.sufficient_statistic)(grid)  # (n_grid, 2)
        design = jnp.column_stack([jnp.ones(self.n_grid_points), suff_stats])

        # Solve least squares: design @ [χ, ρ] ≈ log_partitions
        coeffs, _, _, _ = jnp.linalg.lstsq(design, log_partitions, rcond=None)

        # Return ρ (coefficients 1 and 2)
        return coeffs[1:]

    def regression_diagnostics(
        self, params: Array
    ) -> tuple[Array, Array, Array]:
        """Get regression fit diagnostics for visualization.

        Returns the actual log-partition values and the fitted regression
        values over the stimulus grid.

        Args:
            params: Population code parameters

        Returns:
            Tuple of (grid, actual_log_partitions, fitted_values)
        """
        obs_params, int_params, _ = self.split_coords(params)
        int_matrix = int_params.reshape(self.n_neurons, 2)

        # Generate grid
        grid = jnp.linspace(0, 2 * jnp.pi, self.n_grid_points, endpoint=False)

        def log_partition_at(z: Array) -> Array:
            suff = self.lat_man.sufficient_statistic(z)
            log_rates = obs_params + int_matrix @ suff
            return jnp.sum(jnp.exp(log_rates))

        # Actual log-partition values
        actual = jax.vmap(log_partition_at)(grid)

        # Fitted regression values
        suff_stats = jax.vmap(self.lat_man.sufficient_statistic)(grid)
        design = jnp.column_stack([jnp.ones(self.n_grid_points), suff_stats])
        coeffs, _, _, _ = jnp.linalg.lstsq(design, actual, rcond=None)
        fitted = design @ coeffs

        return grid, actual, fitted

    # Tuning curve methods

    def join_tuning_parameters(
        self,
        gains: Array,
        preferred: Array,
        baselines: Array,
        prior_params: Array,
    ) -> Array:
        """Construct population code from tuning curve specification.

        Args:
            gains: Tuning curve amplitudes, shape (n_neurons,)
            preferred: Preferred stimulus directions, shape (n_neurons,)
            baselines: Log baseline firing rates, shape (n_neurons,)
            prior_params: VonMises prior natural parameters, shape (2,)

        Returns:
            Natural parameters for the population code
        """
        # Observable biases are the baselines
        obs_params = baselines

        # Interaction matrix: each row is [gain*cos(pref), gain*sin(pref)]
        int_col_1 = gains * jnp.cos(preferred)
        int_col_2 = gains * jnp.sin(preferred)
        int_params = jnp.stack([int_col_1, int_col_2], axis=1).ravel()

        # Compute likelihood params for conjugation
        lkl_params = self.lkl_fun_man.join_coords(obs_params, int_params)

        # Compute conjugation parameters
        rho = self.conjugation_parameters(lkl_params)

        # Store adjusted latent params (prior - rho)
        lat_params = prior_params - rho

        return self.join_coords(obs_params, int_params, lat_params)

    def split_tuning_parameters(
        self, params: Array
    ) -> tuple[Array, Array, Array, Array]:
        """Extract tuning curve parameters from natural parameters.

        Args:
            params: Natural parameters for the population code

        Returns:
            Tuple of (gains, preferred, baselines, prior_params)
        """
        obs_params, int_params, lat_params = self.split_coords(params)

        # Baselines are observable biases
        baselines = obs_params

        # Interaction matrix has rows [gain*cos(pref), gain*sin(pref)]
        int_matrix = int_params.reshape(self.n_neurons, 2)
        gains = jnp.sqrt(int_matrix[:, 0] ** 2 + int_matrix[:, 1] ** 2)
        preferred = jnp.arctan2(int_matrix[:, 1], int_matrix[:, 0])

        # Reconstruct prior from stored lat_params + rho
        lkl_params = self.lkl_fun_man.join_coords(obs_params, int_params)
        rho = self.conjugation_parameters(lkl_params)
        prior_params = lat_params + rho

        return gains, preferred, baselines, prior_params

    def tuning_curve(self, params: Array, z: Array) -> Array:
        """Evaluate log firing rates at a stimulus.

        Computes log λ_i(z) = baseline_i + gain_i·cos(z - preferred_i)
        for each neuron.

        Args:
            params: Natural parameters for the population code
            z: Stimulus value (scalar)

        Returns:
            Log firing rates for each neuron, shape (n_neurons,)
        """
        obs_params, int_params, _ = self.split_coords(params)
        int_matrix = int_params.reshape(self.n_neurons, 2)
        suff = self.lat_man.sufficient_statistic(z)
        return obs_params + int_matrix @ suff

    def firing_rates(self, params: Array, z: Array) -> Array:
        """Evaluate firing rates at a stimulus.

        Args:
            params: Natural parameters for the population code
            z: Stimulus value (scalar)

        Returns:
            Firing rates λ_i(z) for each neuron, shape (n_neurons,)
        """
        return jnp.exp(self.tuning_curve(params, z))

    # Sampling

    def sample_stimulus(self, key: Array, params: Array) -> Array:
        """Sample stimulus z from prior."""
        prior_params = self.prior(params)
        return self.lat_man.sample(key, prior_params, n=1)[0]

    def sample_spikes(self, key: Array, params: Array, z: Array) -> Array:
        """Sample spike counts given a stimulus.

        Args:
            key: JAX random key
            params: Population code parameters
            z: Stimulus value

        Returns:
            Spike counts for each neuron, shape (n_neurons,)
        """
        log_rates = self.tuning_curve(params, z)
        rates = jnp.exp(log_rates)
        return jax.random.poisson(key, rates)

    def sample_joint(
        self, key: Array, params: Array, n: int = 1
    ) -> tuple[Array, Array]:
        """Sample (spike_counts, stimulus) pairs from the joint distribution.

        Args:
            key: JAX random key
            params: Population code parameters
            n: Number of samples

        Returns:
            Tuple of (spike_counts, stimuli) with shapes (n, n_neurons) and (n,)
        """
        key_z, key_x = jax.random.split(key)

        # Sample all stimuli in one batched call
        prior_params = self.prior(params)
        zs = self.lat_man.sample(key_z, prior_params, n=n).squeeze(-1)  # (n,)

        # Sample spikes for each stimulus
        keys_x = jax.random.split(key_x, n)
        xs = jax.vmap(lambda k, z: self.sample_spikes(k, params, z))(keys_x, zs)

        return xs, zs

    # Inference

    def posterior_mean(self, params: Array, x: Array) -> Array:
        """Compute posterior mean stimulus given observations.

        Uses the mean of the VonMises posterior distribution.

        Args:
            params: Population code parameters
            x: Observed spike counts, shape (n_neurons,)

        Returns:
            Posterior mean stimulus angle
        """
        post_params = self.posterior_at(params, x)
        mu, _ = self.lat_man.split_mean_concentration(post_params)
        return mu

    def posterior_concentration(self, params: Array, x: Array) -> Array:
        """Compute posterior concentration given observations.

        Args:
            params: Population code parameters
            x: Observed spike counts, shape (n_neurons,)

        Returns:
            Posterior concentration (κ)
        """
        post_params = self.posterior_at(params, x)
        _, kappa = self.lat_man.split_mean_concentration(post_params)
        return kappa

    # Observable density

    def log_observable_density(self, params: Array, x: Array) -> Array:
        """Compute log density of the observable distribution p(x).

        For a conjugated harmonium, the observable density is:
            log p(x) = θ_X·s(x) + ψ_Z(posterior(x)) - ψ_Z(prior) - ψ_X(θ_X) + log μ(x)

        Args:
            params: Population code parameters
            x: Observed spike counts, shape (n_neurons,)

        Returns:
            Log density value
        """
        obs_params, _, _ = self.split_coords(params)

        # Observable log-partition and sufficient stats
        chi = self.obs_man.log_partition_function(obs_params)
        obs_stats = self.obs_man.sufficient_statistic(x)

        # Prior distribution
        prr = self.prior(params)

        # Compute log density
        log_density = jnp.dot(obs_params, obs_stats)
        log_density += self.lat_man.log_partition_function(self.posterior_at(params, x))
        log_density -= self.lat_man.log_partition_function(prr) + chi

        return log_density + self.obs_man.log_base_measure(x)

    def observable_density(self, params: Array, x: Array) -> Array:
        """Compute density of the observable distribution p(x)."""
        return jnp.exp(self.log_observable_density(params, x))


def von_mises_population_code(
    n_neurons: int,
    gains: Array,
    preferred: Array,
    baselines: Array,
    prior_mean: float = 0.0,
    prior_concentration: float = 0.0,
) -> tuple[VonMisesPopulationCode, Array]:
    """Create a Von Mises population code with specified tuning curves.

    Args:
        n_neurons: Number of neurons
        gains: Tuning curve amplitudes, shape (n_neurons,)
        preferred: Preferred stimulus directions, shape (n_neurons,)
        baselines: Log baseline firing rates, shape (n_neurons,)
        prior_mean: Mean of the stimulus prior (default 0.0)
        prior_concentration: Concentration of stimulus prior (default 0.0 = uniform)

    Returns:
        Tuple of (model, parameters)
    """
    model = VonMisesPopulationCode(n_neurons)
    prior_params = model.lat_man.join_mean_concentration(prior_mean, prior_concentration)
    params = model.join_tuning_parameters(gains, preferred, baselines, prior_params)
    return model, params
