"""Poisson observable harmonium implementations.

This module provides harmoniums with Poisson observable units:
- PoissonBernoulliHarmonium: Poisson-Bernoulli harmonium
- PoissonVonMisesHarmonium: Poisson-VonMises harmonium for toroidal latent discovery
- VariationalPoissonVonMises: Variational wrapper for PoissonVonMisesHarmonium
- VonMisesPopulationCode: Population code with Poisson observables and Von Mises stimulus

Also provides Poisson mixture models:
- PoissonPopulation: Analytic product of independent Poisson units
- CoMPoissonPopulation: Population of COM-Poisson units with shared dispersion
- PoissonMixture: Analytic mixture of Poisson populations
- CoMPoissonMixture: Mixture of COM-Poisson populations

These are bipartite undirected graphical models (X ↔ Z bilayer structure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    AnalyticProduct,
    DifferentiableProduct,
    EmbeddedMap,
    IdentityEmbedding,
    LocationShape,
    Product,
    Rectangular,
    TupleEmbedding,
)
from ...geometry.exponential_family.harmonium import (
    DifferentiableHarmonium,
    SymmetricConjugated,
)
from ...geometry.exponential_family.variational import (
    DifferentiableVariationalConjugated,
)
from ..base.categorical import Bernoullis
from ..base.poisson import CoMPoisson, CoMShape, Poisson, Poissons
from ..base.von_mises import VonMises, VonMisesProduct
from .mixture import AnalyticMixture, Mixture


### Poisson-Bernoulli Harmonium ###


@dataclass(frozen=True)
class PoissonBernoulliHarmonium(DifferentiableHarmonium[Poissons, Bernoullis]):
    """Harmonium with Poisson observable units and Bernoulli latent units.

    For modeling count data (e.g., grayscale images as pixel counts).
    Each observable unit is Poisson(rate_i) where rate_i = exp(b_i + W_i^T z).

    The joint distribution is:

    $$p(x, z) \\propto \\prod_i (rate_i^{x_i} / x_i!) \\exp(x^T W z + b^T x + c^T z)$$

    where:
    - x_i in {0,1,2,...} are observable counts (unbounded)
    - z in {0,1}^n_latent are latent units
    - W is the weight matrix (n_observable x n_latent)
    - b are observable biases (log rates when z=0)
    - c are latent biases

    Key difference from Binomial RBM:
    - Observable units are unbounded (no n_trials parameter)
    - For visualization (e.g., images), outputs should be clipped to valid range

    As a harmonium:
    - obs_man: Poissons(n_observable) - observable layer
    - pst_man: Bernoullis(n_latent) - latent layer
    - int_man: Rectangular weight matrix W

    The conditionals are:
    - p(x_i | z) = Poisson(exp(b_i + W_i^T z))
    - p(z_j = 1 | x) = sigmoid(c_j + sum_i W_ij x_i)

    Attributes:
        n_observable: Number of observable units
        n_latent: Number of latent units
    """

    n_observable: int
    """Number of observable units."""

    n_latent: int
    """Number of latent units."""

    # Properties

    @property
    @override
    def obs_man(self) -> Poissons:
        """Observable layer manifold (Poisson units)."""
        return Poissons(self.n_observable)

    @property
    @override
    def pst_man(self) -> Bernoullis:
        """Latent layer manifold (Bernoulli units)."""
        return Bernoullis(self.n_latent)

    @property
    @override
    def int_man(self) -> EmbeddedMap[Bernoullis, Poissons]:
        """Interaction matrix with identity embeddings (full connectivity).

        The interaction matrix W has shape (n_observable, n_latent) and connects
        all observable units to all latent units.
        """
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(self.pst_man),
            IdentityEmbedding(self.obs_man),
        )

    # Convenience methods

    def observable_rates(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] = exp(b_i + W_i^T z) for all observable units.

        Args:
            params: RBM parameters
            z: Latent unit activations (shape: n_latent)

        Returns:
            Poisson rates for each observable unit (shape: n_observable)
        """
        lkl_params = self.likelihood_at(params, z)
        # For Poisson, mean = exp(natural parameter) = rate
        return self.obs_man.to_mean(lkl_params)

    def latent_probabilities(self, params: Array, x: Array) -> Array:
        """Compute P(z_j = 1 | x) for all latent units.

        Args:
            params: RBM parameters
            x: Observable unit counts (shape: n_observable)

        Returns:
            Probabilities for each latent unit (shape: n_latent)
        """
        post_params = self.posterior_at(params, x)
        return self.pst_man.to_mean(post_params)

    def normalized_reconstruction_error(
        self, params: Array, xs: Array, scale: float = 255.0
    ) -> Array:
        """Compute mean squared reconstruction error on normalized data.

        Normalizes both input and reconstruction to [0, 1] before computing MSE.
        Useful when comparing to other RBM types on image data.

        Args:
            params: RBM parameters
            xs: Batch of observable counts (shape: n_samples, n_observable)
            scale: Scale factor for normalization (default 255 for grayscale)

        Returns:
            Mean squared error on normalized values (scalar)
        """
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        xs_norm = xs / scale
        recons_norm = recons / scale
        return jnp.mean((xs_norm - recons_norm) ** 2)

    def get_filters(self, params: Array, img_shape: tuple[int, int]) -> Array:
        """Reshape weight matrix into filter images for visualization.

        Each column of the weight matrix (corresponding to one latent unit)
        is reshaped into an image.

        Args:
            params: RBM parameters
            img_shape: Shape of each observable image (height, width)

        Returns:
            Array of shape (n_latent, height, width)
        """
        _, int_params, _ = self.split_coords(params)
        n_obs = self.obs_man.dim
        n_lat = self.pst_man.dim
        weights = int_params.reshape(n_obs, n_lat)
        return weights.T.reshape(-1, *img_shape)


def poisson_bernoulli_harmonium(
    n_observable: int, n_latent: int
) -> PoissonBernoulliHarmonium:
    """Create a Poisson-Bernoulli Harmonium.

    Factory function for convenient construction.

    Args:
        n_observable: Number of observable units
        n_latent: Number of latent units

    Returns:
        PoissonBernoulliHarmonium instance
    """
    return PoissonBernoulliHarmonium(n_observable=n_observable, n_latent=n_latent)


### Poisson-VonMises Harmonium ###


@dataclass(frozen=True)
class PoissonVonMisesHarmonium(DifferentiableHarmonium[Poissons, VonMisesProduct]):
    """Harmonium with Poisson observable units and Von Mises latent units.

    For modeling neural spike counts with circular latent structure.
    Each observable unit is Poisson(rate_i) where rate_i = exp(b_i + W_i @ s(z)).
    Each latent unit is Von Mises with mean and concentration parameters.

    The joint distribution is:

    $$p(x, z) \\propto \\prod_i (rate_i^{x_i} / x_i!)
        \\exp(b^T x + \\theta_z^T s(z) + x^T W s(z))$$

    where:
    - x_i in {0,1,2,...} are observable counts (unbounded)
    - z in [0, 2*pi)^n_latent are circular latent angles
    - s(z) = [cos(z_1), sin(z_1), ..., cos(z_n), sin(z_n)] is the sufficient statistic
    - W is the weight matrix (n_neurons x 2*n_latent)
    - b are observable biases (log baseline rates)
    - theta_z are latent natural parameters

    As a harmonium:
    - obs_man: Poissons(n_neurons) - observable layer (neural populations)
    - pst_man: VonMisesProduct(n_latent) - latent layer (n-torus)
    - int_man: Rectangular weight matrix W

    The conditionals are:
    - p(x_i | z) = Poisson(exp(b_i + W_i @ s(z)))
    - p(z | x) = product of Von Mises with shifted natural parameters

    Attributes:
        n_neurons: Number of observable (Poisson) units
        n_latent: Number of latent (Von Mises) units
    """

    n_neurons: int
    """Number of observable Poisson units (neurons)."""

    n_latent: int
    """Number of latent Von Mises units (dimensions on the torus)."""

    # Properties

    @property
    @override
    def obs_man(self) -> Poissons:
        """Observable layer manifold (Poisson units)."""
        return Poissons(self.n_neurons)

    @property
    @override
    def pst_man(self) -> VonMisesProduct:
        """Latent layer manifold (Von Mises units)."""
        return VonMisesProduct(self.n_latent)

    @property
    @override
    def int_man(self) -> EmbeddedMap[VonMisesProduct, Poissons]:
        """Interaction matrix with identity embeddings (full connectivity).

        The interaction matrix W has shape (n_neurons, 2*n_latent) and connects
        all observable units to all latent Von Mises sufficient statistics.
        """
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(self.pst_man),
            IdentityEmbedding(self.obs_man),
        )

    # Convenience methods

    def observable_rates(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] = exp(b_i + W_i @ s(z)) for all neurons.

        Args:
            params: Harmonium parameters
            z: Latent angles (shape: n_latent)

        Returns:
            Poisson rates for each neuron (shape: n_neurons)
        """
        lkl_params = self.likelihood_at(params, z)
        # For Poisson, mean = exp(natural parameter) = rate
        return self.obs_man.to_mean(lkl_params)

    def latent_sufficient_stats(self, params: Array, x: Array) -> Array:
        """Compute E[s(z) | x] from the posterior parameters.

        Args:
            params: Harmonium parameters
            x: Observable counts (shape: n_neurons)

        Returns:
            Expected sufficient statistics (shape: 2*n_latent)
        """
        post_params = self.posterior_at(params, x)
        return self.pst_man.to_mean(post_params)

    def get_weight_matrix(self, params: Array) -> Array:
        """Extract the interaction weight matrix W.

        Args:
            params: Harmonium parameters

        Returns:
            Weight matrix of shape (n_neurons, 2*n_latent)
        """
        _, int_params, _ = self.split_coords(params)
        return int_params.reshape(self.n_neurons, 2 * self.n_latent)

    def get_baseline_rates(self, params: Array) -> Array:
        """Get baseline firing rates exp(b) when z=0.

        Args:
            params: Harmonium parameters

        Returns:
            Baseline rates for each neuron (shape: n_neurons)
        """
        obs_params, _, _ = self.split_coords(params)
        return jnp.exp(obs_params)


def poisson_vonmises_harmonium(
    n_neurons: int, n_latent: int
) -> PoissonVonMisesHarmonium:
    """Create a Poisson-VonMises Harmonium.

    Factory function for convenient construction.

    Args:
        n_neurons: Number of observable (Poisson) units
        n_latent: Number of latent (Von Mises) units

    Returns:
        PoissonVonMisesHarmonium instance
    """
    return PoissonVonMisesHarmonium(n_neurons=n_neurons, n_latent=n_latent)


@dataclass(frozen=True)
class VariationalPoissonVonMises(
    DifferentiableVariationalConjugated[Poissons, VonMisesProduct]
):
    """Variational conjugated model with Poisson observables and VonMises latents.

    This wraps a PoissonVonMisesHarmonium with learnable conjugation
    parameters for variational inference via ELBO optimization.

    The approximate posterior at observation x is:
        q(z|x) = VonMises(theta_Z + Theta_{XZ}^T s_X(x) - rho)

    where rho is learned to minimize the ELBO (maximize the evidence lower bound).

    Mathematical framework:
    - Generative model: p(z; theta_Z), p(x|z) = Poisson(exp(b + W @ s(z)))
    - Approximate posterior: q(z|x) with natural params theta_Z + W^T x - rho
    - ELBO: E_q[log p(x|z)] - D_KL(q(z|x) || p(z))

    Attributes:
        hrm: The underlying PoissonVonMisesHarmonium
    """

    hrm: PoissonVonMisesHarmonium
    """Override with concrete type for type checking."""

    # Override ELBO computation to handle non-reparameterizable VonMises sampling

    @override
    def elbo_reconstruction_term(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> Array:
        """Compute E_q[log p(x|z)] via Monte Carlo sampling.

        Uses stop_gradient on samples since VonMises sampling (rejection sampling)
        is not reparameterizable. This gives a biased but usable gradient estimator.

        Args:
            key: JAX random key
            params: Full model parameters
            x: Observation
            n_samples: Number of MC samples

        Returns:
            Expected log-likelihood (scalar)
        """
        s_x = self.obs_man.sufficient_statistic(x)
        q_params = self.approximate_posterior_at(params, x)

        # Sample z from q(z|x) - use stop_gradient since VonMises sampling
        # uses rejection sampling which can't be differentiated through
        z_samples = jax.lax.stop_gradient(
            self.pst_man.sample(key, q_params, n_samples)
        )

        def log_likelihood_at_z(z: Array) -> Array:
            lkl_params = self.likelihood_at(params, z)
            return (
                jnp.dot(s_x, lkl_params)
                - self.obs_man.log_partition_function(lkl_params)
                + self.obs_man.log_base_measure(x)
            )

        return jnp.mean(jax.vmap(log_likelihood_at_z)(z_samples))

    # Convenience accessors

    @property
    def n_neurons(self) -> int:
        """Number of observable Poisson units."""
        return self.hrm.n_neurons

    @property
    def n_latent(self) -> int:
        """Number of latent VonMises units."""
        return self.hrm.n_latent

    # Convenience methods

    def observable_rates(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] for all neurons."""
        _, hrm_params = self.split_coords(params)
        return self.hrm.observable_rates(hrm_params, z)

    def reconstruct(self, params: Array, x: Array) -> Array:
        """Reconstruct observable via mean-field approximation.

        Uses E[x | E[z|x]] where the expectation is under q(z|x).

        Args:
            params: Full model parameters [rho, harmonium_params]
            x: Observable spike counts

        Returns:
            Reconstructed expected rates
        """
        # Get approximate posterior parameters
        q_params = self.approximate_posterior_at(params, x)

        # Compute mean sufficient statistics from VonMises posterior
        z_mean_stats = self.pst_man.to_mean(q_params)

        # Get likelihood parameters using mean sufficient statistics
        _, hrm_params = self.split_coords(params)
        obs_bias, int_params, _ = self.hrm.split_coords(hrm_params)
        int_matrix = int_params.reshape(self.n_neurons, 2 * self.n_latent)
        lkl_natural = obs_bias + int_matrix @ z_mean_stats

        return self.obs_man.to_mean(lkl_natural)

    def reconstruction_error(self, params: Array, xs: Array) -> Array:
        """Compute mean squared reconstruction error over a batch.

        Args:
            params: Full model parameters
            xs: Batch of spike counts (shape: n_samples, n_neurons)

        Returns:
            Mean squared error (scalar)
        """
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        return jnp.mean((xs - recons) ** 2)

    def normalized_reconstruction_error(
        self, params: Array, xs: Array, scale: float = 1.0
    ) -> Array:
        """Compute MSE on normalized data.

        Args:
            params: Full model parameters
            xs: Batch of spike counts
            scale: Scale factor for normalization

        Returns:
            Mean squared error on normalized values
        """
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        xs_norm = xs / scale
        recons_norm = recons / scale
        return jnp.mean((xs_norm - recons_norm) ** 2)

    def get_weight_matrix(self, params: Array) -> Array:
        """Extract the interaction weight matrix W.

        Args:
            params: Full model parameters

        Returns:
            Weight matrix of shape (n_neurons, 2*n_latent)
        """
        _, hrm_params = self.split_coords(params)
        return self.hrm.get_weight_matrix(hrm_params)


def variational_poisson_vonmises(
    n_neurons: int, n_latent: int
) -> VariationalPoissonVonMises:
    """Create a variational Poisson-VonMises model.

    Factory function for convenient construction.

    Args:
        n_neurons: Number of observable (Poisson) units
        n_latent: Number of latent (Von Mises) units

    Returns:
        VariationalPoissonVonMises instance
    """
    hrm = PoissonVonMisesHarmonium(n_neurons=n_neurons, n_latent=n_latent)
    return VariationalPoissonVonMises(hrm=hrm)


### Population Codes ###

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

        Fits $\\chi + \\rho^T \\cdot T(z) \\approx \\psi_X(z)$ where $\\psi_X(z)$ is the
        log-partition of the likelihood at stimulus z.

        Algorithm:
        1. Create grid of stimulus values $z_1, ..., z_M$ in $[0, 2\\pi)$
        2. For each $z_m$, compute log-partition: $\\psi(z_m) = \\sum_i \\exp(\\theta_i + \\Theta_i \\cdot T(z_m))$
        3. Regress: $[1, T(z_m)] @ [\\chi, \\rho] \\approx \\psi(z_m)$
        4. Return $\\rho$ ($\\chi$ is used in log_partition_function)

        Args:
            lkl_params: Likelihood parameters [obs_params, int_params]

        Returns:
            Conjugation parameters $\\rho$ with shape (2,)
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

        # Solve least squares: design @ [chi, rho] ~ log_partitions
        coeffs, _, _, _ = jnp.linalg.lstsq(design, log_partitions, rcond=None)

        # Return rho (coefficients 1 and 2)
        return coeffs[1:]

    def regression_diagnostics(self, params: Array) -> tuple[Array, Array, Array]:
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
        log_rates = self.tuning_curve(params, z)
        return self.obs_man.to_mean(log_rates)

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
        rates = self.obs_man.to_mean(log_rates)
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
    prior_params = model.lat_man.join_mean_concentration(
        prior_mean, prior_concentration
    )
    params = model.join_tuning_parameters(gains, preferred, baselines, prior_params)
    return model, params


### Poisson Mixture Models ###


# Type synonyms for common models
type PopulationShape = Product[CoMShape]
type PoissonMixture = AnalyticMixture[PoissonPopulation]
type CoMPoissonMixture = Mixture[CoMPoissonPopulation]


@dataclass(frozen=True)
class CoMPoissonPopulation(
    DifferentiableProduct[CoMPoisson], LocationShape[PoissonPopulation, PopulationShape]
):
    """A population of independent Conway-Maxwell-Poisson (COM-Poisson) units.

    The COM-Poisson distribution generalizes the Poisson distribution by introducing a
    dispersion parameter :math:`\\nu`, enabling flexible modeling of count data with
    variance different from the mean.

    For :math:`n` independent COM-Poisson units, the joint density takes the form:

    .. math::

        p(x; \\mu, \\nu) = \\prod_{i=1}^n \\frac{\\mu_i^{x_i}}{(x_i!)^{\\nu_i} Z(\\mu_i, \\nu_i)}

    where for each unit :math:`i`:

    - :math:`\\mu_i > 0` is the rate parameter (mode of the distribution)
    - :math:`\\nu_i > 0` is the dispersion parameter:
        - :math:`\\nu_i = 1`: standard Poisson distribution
        - :math:`\\nu_i > 1`: underdispersed (variance :math:`< \\mu_i`)
        - :math:`\\nu_i < 1`: overdispersed (variance :math:`> \\mu_i`)
    - :math:`Z(\\mu_i, \\nu_i) = \\sum_{j=0}^{\\infty} \\frac{\\mu_i^j}{(j!)^{\\nu_i}}` is the normalizing constant

    **Usage in mixtures**: When used as a mixture component, the rate parameters :math:`\\mu_i` vary
    across mixture components while the dispersion parameters :math:`\\nu_i` are shared. This enables
    flexible mixture models where each component captures both the mode and over/underdispersion
    characteristics of subpopulations.

    **Implementation note**: This class is a `LocationShape` split, where locations are Poisson
    rates and shapes are COM-Poisson dispersion parameters. This enables efficient parameterization
    in mixture models where dispersion is shared.
    """

    def __init__(self, n_reps: int):
        """Initialize COM-Poisson population.

        Args:
            n_reps: Number of neurons in the population
        """
        super().__init__(CoMPoisson(), n_reps)

    @property
    @override
    def fst_man(self) -> PoissonPopulation:
        return AnalyticProduct(Poisson(), n_reps=self.n_reps)

    @property
    @override
    def snd_man(self) -> PopulationShape:
        return Product(CoMShape(), n_reps=self.n_reps)

    @override
    def join_coords(
        self,
        fst_coords: Array,
        snd_coords: Array,
    ) -> Array:
        """Join location and shape parameters.

        Parameters
        ----------
        first : Array
            Natural parameters for Poisson location (rates).
        second : Array
            Natural parameters for shape (dispersion).

        Returns
        -------
        Array
            Natural parameters for COM-Poisson population.
        """
        params_matrix = jnp.stack([fst_coords, snd_coords])
        return params_matrix.T.reshape(-1)

    @override
    def split_coords(
        self,
        coords: Array,
    ) -> tuple[Array, Array]:
        """Split into location and shape parameters.

        Parameters
        ----------
        params : Array
            Natural parameters for COM-Poisson population.

        Returns
        -------
        tuple[Array, Array]
            Location parameters (Poisson rates) and shape parameters (dispersion).
        """
        matrix = coords.reshape(self.n_reps, 2).T
        loc_params = matrix[0]
        shp_params = matrix[1]
        return loc_params, shp_params


@dataclass(frozen=True)
class PopulationLocationEmbedding(
    TupleEmbedding[
        PoissonPopulation,
        CoMPoissonPopulation,
    ]
):
    """Embedding that projects COM-Poisson to Poisson via location parameters.

    For a replicated location-shape manifold with parameters :math:`((l_1,s_1),\\ldots,(l_n,s_n))`,
    this embedding projects to just the location parameters: :math:`(l_1,\\ldots,l_n)`.

    **Purpose**: In mixture models, components may need different rate parameters (locations) but
    shared dispersion parameters (shapes). This embedding enables such structure by:
    - Projecting COM-Poisson parameters down to their rate components (Poisson)
    - Embedding Poisson rates up to full COM-Poisson by combining with shared dispersions

    **Mathematical structure**: The embedding maps between:
    - Sub-manifold: :math:`\\text{Poisson}^n` (location parameters only)
    - Ambient manifold: :math:`\\text{CoMPoisson}^n` (location-shape pairs)

    The first component of each tuple is extracted or preserved during transformations:
    - Projection: :math:`((l_1, s_1), \\ldots, (l_n, s_n)) \\mapsto (l_1, \\ldots, l_n)`
    - Embedding: :math:`(l_1, \\ldots, l_n) + \\text{shapes} \\mapsto ((l_1, s_1), \\ldots, (l_n, s_n))`

    This enables efficient parameterization in mixture models where component-specific behavior
    comes from varying rates while uniform dispersion is maintained across components.
    """

    n_neurons: int

    @property
    @override
    def tup_idx(
        self,
    ) -> int:
        return 0

    @property
    @override
    def amb_man(self) -> CoMPoissonPopulation:
        return CoMPoissonPopulation(n_reps=self.n_neurons)

    @property
    @override
    def sub_man(self) -> PoissonPopulation:
        return AnalyticProduct(Poisson(), n_reps=self.n_neurons)


# Constructor-like factory functions for common instances
def poisson_mixture(n_neurons: int, n_components: int) -> PoissonMixture:
    """Create a mixture of independent Poisson populations.

    Parameters
    ----------
    n_neurons : int
        Number of independent Poisson units in each population
    n_components : int
        Number of mixture components

    Returns
    -------
    PoissonMixture
        An analytic mixture of Poisson populations
    """
    pop_man = AnalyticProduct(Poisson(), n_reps=n_neurons)
    return AnalyticMixture(pop_man, n_components)


def com_poisson_mixture(n_neurons: int, n_components: int) -> CoMPoissonMixture:
    """Create a COM-Poisson mixture with shared dispersion parameters.

    In this model, the location parameters vary across mixture components,
    but the dispersion parameters are shared across all components.

    Parameters
    ----------
    n_neurons : int
        Number of independent COM-Poisson units in each population
    n_components : int
        Number of mixture components

    Returns
    -------
    CoMPoissonMixture
        A differentiable mixture with COM-Poisson components
    """
    # Observable embedding: projects COM-Poisson to Poisson (location only)
    obs_emb = PopulationLocationEmbedding(n_neurons)
    return Mixture(n_components, obs_emb)
