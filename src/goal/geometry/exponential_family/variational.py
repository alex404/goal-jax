"""Variational conjugation for harmoniums.

This module provides abstract base classes for variational inference on harmoniums,
where the approximate posterior has the same structure as the exact conjugate
posterior, with learnable conjugation parameters.

Mathematical framework:
- Generative model: p_Z(z; theta_Z), p(x|z) with natural params theta_X + Theta_{XZ} s_Z(z)
- Approximate posterior: q(z|x) with params theta_Z + Theta_{XZ}^T s_X(x) - rho
- ELBO: E_q[log p(x|z)] - D_KL(q(z|x) || p(z))

Key insight: Same form as true conjugate posterior, but rho is learned rather than
computed from conjugation. The conjugation parameters and generative (harmonium)
parameters together form the complete model parameterization.

Classes:
    VariationalConjugated: Base class combining conjugation params with harmonium
    DifferentiableVariationalConjugated: Adds ELBO and gradient computation
"""

from abc import ABC
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.combinators import Pair
from .base import Differentiable, Generative
from .harmonium import GenerativeHarmonium


@dataclass(frozen=True)
class VariationalConjugated[Observable: Generative, Posterior: Generative](
    Pair[Posterior, GenerativeHarmonium[Observable, Posterior]],
    ABC,
):
    """Variational conjugated harmonium combining conjugation parameters with a generative model.

    This class represents a variational approximation to a conjugated harmonium where
    the conjugation parameters (rho) are learned rather than computed analytically.
    The full parameter space is the product of:
    - Conjugation parameters rho on the posterior manifold
    - Harmonium parameters [theta_X, Theta_XZ, theta_Z]

    Unlike true `Conjugated` harmoniums:
    - The conjugation parameters are learned, not derived from likelihood structure
    - Uses ELBO optimization instead of EM
    - The model is not strictly an exponential family

    The approximate posterior at observation x is:
        q(z|x) has natural params: theta_Z + Theta_{XZ}^T s_X(x) - rho

    Attributes:
        hrm: The underlying GenerativeHarmonium model
    """

    # Field

    hrm: GenerativeHarmonium[Observable, Posterior]
    """The underlying harmonium model."""

    # Pair implementation

    @property
    @override
    def fst_man(self) -> Posterior:
        """First component: conjugation parameter manifold (same as posterior)."""
        return self.hrm.pst_man

    @property
    @override
    def snd_man(self) -> GenerativeHarmonium[Observable, Posterior]:
        """Second component: the harmonium manifold."""
        return self.hrm

    # Convenience properties

    @property
    def obs_man(self) -> Observable:
        """Observable manifold of the underlying harmonium."""
        return self.hrm.obs_man

    @property
    def pst_man(self) -> Posterior:
        """Posterior manifold of the underlying harmonium."""
        return self.hrm.pst_man

    # Parameter access

    def conjugation_params(self, params: Array) -> Array:
        """Extract conjugation parameters (rho) from full parameters.

        Args:
            params: Full model parameters [rho, harmonium_params]

        Returns:
            Conjugation parameters rho
        """
        rho, _ = self.split_coords(params)
        return rho

    def generative_params(self, params: Array) -> Array:
        """Extract harmonium (generative model) parameters from full parameters.

        Args:
            params: Full model parameters [rho, harmonium_params]

        Returns:
            Harmonium parameters [theta_X, Theta_XZ, theta_Z]
        """
        _, hrm_params = self.split_coords(params)
        return hrm_params

    # Core methods

    def approximate_posterior_at(self, params: Array, x: Array) -> Array:
        """Compute approximate posterior natural parameters at observation x.

        q(z|x) has natural params: theta_Z + Theta_{XZ}^T s_X(x) - rho

        Args:
            params: Full model parameters [rho, harmonium_params]
            x: Observation

        Returns:
            Natural parameters of approximate posterior q(z|x)
        """
        rho, hrm_params = self.split_coords(params)
        exact_posterior = self.hrm.posterior_at(hrm_params, x)
        return exact_posterior - rho

    def likelihood_at(self, params: Array, z: Array) -> Array:
        """Compute likelihood natural parameters at latent z.

        Delegates to the underlying harmonium.

        Args:
            params: Full model parameters
            z: Latent state

        Returns:
            Natural parameters of likelihood p(x|z)
        """
        _, hrm_params = self.split_coords(params)
        return self.hrm.likelihood_at(hrm_params, z)

    def prior_params(self, params: Array) -> Array:
        """Extract prior parameters theta_Z from the harmonium.

        Args:
            params: Full model parameters

        Returns:
            Prior natural parameters theta_Z
        """
        _, hrm_params = self.split_coords(params)
        _, _, lat_params = self.hrm.split_coords(hrm_params)
        return lat_params

    # Initialization

    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize full model parameters (conjugation + harmonium).

        Conjugation parameters are initialized to zeros (so q = exact posterior initially).
        Harmonium parameters use the standard harmonium initialization.

        Args:
            key: JAX random key
            location: Location parameter for initialization
            shape: Shape parameter for initialization

        Returns:
            Full model parameters [rho, harmonium_params]
        """
        rho = jnp.zeros(self.fst_man.dim)
        hrm_params = self.hrm.initialize(key, location, shape)
        return self.join_coords(rho, hrm_params)

    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize parameters using sample data for observable biases.

        Conjugation parameters are initialized to zeros.
        Harmonium uses sample-based initialization.

        Args:
            key: JAX random key
            sample: Sample data for initialization
            location: Location parameter
            shape: Shape parameter

        Returns:
            Full model parameters
        """
        rho = jnp.zeros(self.fst_man.dim)
        hrm_params = self.hrm.initialize_from_sample(key, sample, location, shape)
        return self.join_coords(rho, hrm_params)


@dataclass(frozen=True)
class DifferentiableVariationalConjugated[
    Observable: Differentiable,
    Posterior: Differentiable,
](
    VariationalConjugated[Observable, Posterior],
    ABC,
):
    """Variational conjugated harmonium with ELBO computation.

    Extends VariationalConjugated with methods for computing the ELBO and its
    components. Requires both Observable and Posterior to be Differentiable
    (have to_mean and log_partition_function methods).

    The ELBO decomposes as:
        L(x) = E_q[log p(x|z)] - D_KL(q(z|x) || p(z))
             = reconstruction_term - kl_divergence
    """

    # ELBO component: KL divergence (analytic)

    def elbo_divergence(self, params: Array, x: Array) -> Array:
        """Compute KL divergence D_KL(q(z|x) || p(z)).

        For exponential families:
            D_KL(q || p) = (theta_q - theta_p) . mu_q - psi(theta_q) + psi(theta_p)

        Args:
            params: Full model parameters
            x: Observation

        Returns:
            KL divergence (scalar)
        """
        q_params = self.approximate_posterior_at(params, x)
        p_params = self.prior_params(params)

        q_means = self.pst_man.to_mean(q_params)
        psi_q = self.pst_man.log_partition_function(q_params)
        psi_p = self.pst_man.log_partition_function(p_params)

        param_diff = q_params - p_params
        return jnp.dot(param_diff, q_means) - psi_q + psi_p

    # ELBO component: Expected log-likelihood (requires sampling)

    def elbo_reconstruction_term(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> Array:
        """Compute E_q[log p(x|z)] via Monte Carlo sampling.

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

        # Sample z from q(z|x)
        z_samples = self.pst_man.sample(key, q_params, n_samples)

        def log_likelihood_at_z(z: Array) -> Array:
            lkl_params = self.likelihood_at(params, z)
            return (
                jnp.dot(s_x, lkl_params)
                - self.obs_man.log_partition_function(lkl_params)
                + self.obs_man.log_base_measure(x)
            )

        return jnp.mean(jax.vmap(log_likelihood_at_z)(z_samples))

    # ELBO computation

    def elbo_at(
        self,
        key: Array,
        params: Array,
        x: Array,
        n_samples: int,
        kl_weight: float = 1.0,
    ) -> Array:
        """Compute the ELBO at observation x.

        ELBO = E_q[log p(x|z)] - kl_weight * D_KL(q(z|x) || p(z))

        Args:
            key: JAX random key
            params: Full model parameters
            x: Observation
            n_samples: Number of MC samples
            kl_weight: Weight on KL term (1.0 = standard VAE, <1.0 = beta-VAE)

        Returns:
            ELBO value (scalar)
        """
        recon = self.elbo_reconstruction_term(key, params, x, n_samples)
        kl = self.elbo_divergence(params, x)
        return recon - kl_weight * kl

    def elbo_components(
        self,
        key: Array,
        params: Array,
        x: Array,
        n_samples: int,
        kl_weight: float = 1.0,
    ) -> tuple[Array, Array, Array]:
        """Compute ELBO and its components for monitoring.

        Args:
            key: JAX random key
            params: Full model parameters
            x: Observation
            n_samples: Number of MC samples
            kl_weight: Weight on KL term (1.0 = standard VAE, <1.0 = beta-VAE)

        Returns:
            Tuple of (elbo, reconstruction_term, kl_term)
        """
        recon = self.elbo_reconstruction_term(key, params, x, n_samples)
        kl = self.elbo_divergence(params, x)
        elbo = recon - kl_weight * kl
        return elbo, recon, kl

    def mean_elbo(
        self,
        key: Array,
        params: Array,
        xs: Array,
        n_samples: int,
        kl_weight: float = 1.0,
    ) -> Array:
        """Compute mean ELBO over a batch of observations.

        Args:
            key: JAX random key
            params: Full model parameters
            xs: Batch of observations (shape: n_obs, obs_dim)
            n_samples: Number of MC samples per observation
            kl_weight: Weight on KL term (1.0 = standard VAE, <1.0 = beta-VAE)

        Returns:
            Mean ELBO (scalar)
        """
        batch_size = xs.shape[0]
        keys = jax.random.split(key, batch_size)

        elbos = jax.vmap(lambda k, x: self.elbo_at(k, params, x, n_samples, kl_weight))(
            keys, xs
        )

        return jnp.mean(elbos)

    # Conjugation error (measures departure from exact conjugation)

    def reduced_learning_signal(self, params: Array, z: Array) -> Array:
        """Compute the reduced learning signal f_tilde(z).

        f_tilde(z) = rho . s_Z(z) - psi_X(theta_X + Theta_{XZ} . s_Z(z))

        For a perfectly conjugate model, f_tilde(z) would be constant.

        Args:
            params: Full model parameters
            z: Latent sample

        Returns:
            Scalar value of reduced learning signal
        """
        rho, hrm_params = self.split_coords(params)
        s_z = self.pst_man.sufficient_statistic(z)

        # Linear term: rho . s_Z(z)
        term1 = jnp.dot(rho, s_z)

        # Log partition of likelihood: psi_X(theta_X + Theta_XZ . s_Z(z))
        lkl_params = self.hrm.likelihood_at(hrm_params, z)
        term2 = self.obs_man.log_partition_function(lkl_params)

        return term1 - term2

    def conjugation_error(
        self, key: Array, params: Array, n_samples: int = 100
    ) -> Array:
        """Compute the conjugation error: Var[f_tilde(z)] under the prior.

        If perfectly conjugate, f_tilde(z) = const and variance = 0.

        Args:
            key: JAX random key
            params: Full model parameters
            n_samples: Number of samples for variance estimation

        Returns:
            Conjugation error (variance of f_tilde)
        """
        p_params = self.prior_params(params)
        z_samples = self.pst_man.sample(key, p_params, n_samples)

        f_vals = jax.vmap(lambda z: self.reduced_learning_signal(params, z))(z_samples)

        return jnp.var(f_vals)

    # Sampling from generative model

    def sample_from_prior(self, key: Array, params: Array, n: int) -> Array:
        """Sample from the generative model p(x, z).

        Uses ancestral sampling: z ~ p(z), then x ~ p(x|z).
        Note: conjugation parameters don't affect the generative model.

        Args:
            key: JAX random key
            params: Full model parameters
            n: Number of samples

        Returns:
            Samples [x, z] of shape (n, obs_dim + lat_data_dim)
        """
        key1, key2 = jax.random.split(key)

        # Get prior parameters
        p_params = self.prior_params(params)

        # Sample z from prior
        z_samples = self.pst_man.sample(key1, p_params, n)

        # Sample x from likelihood for each z
        def sample_x_given_z(subkey: Array, z: Array) -> Array:
            lkl_params = self.likelihood_at(params, z)
            return self.obs_man.sample(subkey, lkl_params, 1)[0]

        x_keys = jax.random.split(key2, n)
        x_samples = jax.vmap(sample_x_given_z)(x_keys, z_samples)

        return jnp.concatenate([x_samples, z_samples], axis=-1)
