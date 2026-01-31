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
    VariationalHierarchicalMixture: Variational inference for hierarchical mixture models
    DifferentiableVariationalHierarchicalMixture: Adds ELBO for mixture models
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.combinators import Pair
from ..manifold.embedding import IdentityEmbedding, LinearEmbedding
from .base import Differentiable, ExponentialFamily, Generative
from .harmonium import GenerativeHarmonium


@dataclass(frozen=True)
class VariationalConjugated[
    Observable: Generative,
    Posterior: Generative,
    Conjugation: ExponentialFamily,
](
    Pair[Conjugation, GenerativeHarmonium[Observable, Posterior]],
    ABC,
):
    """Variational harmonium with learnable conjugation correction.

    The full parameter space is Pair[rho_params, harmonium_params].

    The rho correction is applied via an embedding that maps Conjugation
    to Posterior space:
        q(z|x) = rho_emb.translate(posterior_at(x), -rho)

    This generalizes:
    - Simple case: Conjugation = Posterior, embedding = Identity, so q = posterior - rho
    - Mixture case: Conjugation = BaseLatent, embedding maps to obs_bias of mixture

    Unlike true `Conjugated` harmoniums:
    - The conjugation parameters are learned, not derived from likelihood structure
    - Uses ELBO optimization instead of EM
    - The model is not strictly an exponential family

    Attributes:
        hrm: The underlying GenerativeHarmonium model
    """

    # Field

    hrm: GenerativeHarmonium[Observable, Posterior]
    """The underlying harmonium model."""

    # Contract: subclasses provide the embedding

    @property
    @abstractmethod
    def rho_emb(self) -> LinearEmbedding[Conjugation, Posterior]:
        """Embedding from conjugation manifold to posterior manifold.

        Defines how rho correction is applied to the posterior parameters.
        """

    # Pair implementation

    @property
    def rho_man(self) -> Conjugation:
        """Manifold for rho correction parameters."""
        return self.rho_emb.sub_man

    @property
    @override
    def fst_man(self) -> Conjugation:
        """First component: conjugation parameter manifold."""
        return self.rho_man

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

        q(z|x) = rho_emb.translate(posterior_at(x), -rho)

        Args:
            params: Full model parameters [rho, harmonium_params]
            x: Observation

        Returns:
            Natural parameters of approximate posterior q(z|x)
        """
        rho, hrm_params = self.split_coords(params)
        exact_posterior = self.hrm.posterior_at(hrm_params, x)
        return self.rho_emb.translate(exact_posterior, -rho)

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
        rho = jnp.zeros(self.rho_man.dim)
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
        rho = jnp.zeros(self.rho_man.dim)
        hrm_params = self.hrm.initialize_from_sample(key, sample, location, shape)
        return self.join_coords(rho, hrm_params)


@dataclass(frozen=True)
class SimpleVariationalConjugated[Observable: Generative, Posterior: Generative](
    VariationalConjugated[Observable, Posterior, Posterior],
    ABC,
):
    """Variational conjugated harmonium with identity rho embedding.

    This is the simple case where Conjugation = Posterior and the embedding is identity,
    so the approximate posterior is: q(z|x) = posterior_at(x) - rho
    """

    @property
    @override
    def rho_emb(self) -> IdentityEmbedding[Posterior]:
        """Identity embedding: rho directly subtracts from posterior params."""
        return IdentityEmbedding(self.hrm.pst_man)


@dataclass(frozen=True)
class DifferentiableVariationalConjugated[
    Observable: Differentiable,
    Posterior: Differentiable,
](
    SimpleVariationalConjugated[Observable, Posterior],
    ABC,
):
    """Variational conjugated harmonium with ELBO computation.

    Extends SimpleVariationalConjugated with methods for computing the ELBO and its
    components. Requires both Observable and Posterior to be Differentiable
    (have to_mean and log_partition_function methods).

    The ELBO decomposes as:
        L(x) = E_q[log p(x|z)] - D_KL(q(z|x) || p(z))
             = reconstruction_term - kl_divergence
    """

    # ELBO component: KL divergence (analytic)

    def elbo_divergence(self, params: Array, x: Array) -> Array:
        """Compute KL divergence D_KL(q(z|x) || p(z)).

        Args:
            params: Full model parameters
            x: Observation

        Returns:
            KL divergence (scalar)
        """
        q_params = self.approximate_posterior_at(params, x)
        p_params = self.prior_params(params)
        return self.pst_man.relative_entropy(q_params, p_params)

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

        f_tilde(z) = rho . s_rho(z) - psi_X(theta_X + Theta_{XZ} . s_Z(z))

        where s_rho(z) is the projection of s_Z(z) onto the conjugation manifold.
        For simple models, s_rho = s_Z (identity embedding).
        For hierarchical mixtures, s_rho extracts the base latent component.

        For a perfectly conjugate model, f_tilde(z) would be constant.

        Args:
            params: Full model parameters
            z: Latent sample

        Returns:
            Scalar value of reduced learning signal
        """
        rho, hrm_params = self.split_coords(params)
        s_z = self.pst_man.sufficient_statistic(z)

        # Project sufficient statistics to conjugation manifold
        # For simple case: identity projection (s_rho = s_z)
        # For hierarchical: extract base latent component from mixture
        s_rho = self.rho_emb.project(s_z)

        # Linear term: rho . s_rho(z)
        term1 = jnp.dot(rho, s_rho)

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

    def conjugation_metrics(
        self, key: Array, params: Array, n_samples: int = 100
    ) -> tuple[Array, Array, Array]:
        """Compute conjugation quality metrics under the prior.

        Returns variance, standard deviation, and R² measuring how well
        the linear correction ρ·s(z) approximates ψ_X(η(z)).

        R² = 1 - Var[f̃] / Var[ψ_X(η(z))]

        where f̃(z) = ρ·s(z) - ψ_X(η(z)) is the residual.

        - R² = 1: Perfect conjugation (ψ_X is exactly linear in s(z))
        - R² = 0: Linear correction explains none of the variance
        - R² < 0: Linear correction makes things worse

        Args:
            key: JAX random key
            params: Full model parameters
            n_samples: Number of samples for estimation

        Returns:
            Tuple of (variance, std, r_squared)
        """
        _, hrm_params = self.split_coords(params)
        p_params = self.prior_params(params)
        z_samples = self.pst_man.sample(key, p_params, n_samples)

        # Compute f_tilde values
        f_vals = jax.vmap(lambda z: self.reduced_learning_signal(params, z))(z_samples)

        # Compute psi_X values (the target we're trying to approximate)
        def psi_x_at_z(z: Array) -> Array:
            lkl_params = self.hrm.likelihood_at(hrm_params, z)
            return self.obs_man.log_partition_function(lkl_params)

        psi_vals = jax.vmap(psi_x_at_z)(z_samples)

        var_f = jnp.var(f_vals)
        std_f = jnp.sqrt(var_f)
        var_psi = jnp.var(psi_vals)

        # R² = 1 - Var[residual] / Var[target]
        # Use threshold to avoid numerical instability
        variance_threshold = 1e-6
        raw_r2 = 1.0 - var_f / jnp.maximum(var_psi, variance_threshold)
        r_squared = jnp.where(
            var_psi > variance_threshold,
            jnp.clip(raw_r2, -10.0, 1.0),
            jnp.nan,
        )

        return var_f, std_f, r_squared

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


@dataclass(frozen=True)
class VariationalHierarchicalMixture[
    Observable: Generative,
    BaseLatent: Differentiable,
](
    VariationalConjugated[Observable, Any, BaseLatent],
    ABC,
):
    """Variational harmonium with mixture latent structure.

    For a three-level hierarchy (Observable <-> BaseLatent <-> Categorical),
    the rho correction only affects the BaseLatent (observable) component
    of the mixture, leaving interaction and categorical unchanged.

    Model structure:
        Observable (X) <-> Base Latent (Y) <-> Categorical (K)
           |                    |                    |
        e.g., Binomials     Bernoullis        Mixture component

    Joint distribution: p(x, y, k) = p(k) * p(y|k) * p(x|y)

    The posterior is a CompleteMixture[BaseLatent], and the rho correction
    is applied only to the observable (BaseLatent) bias, not to the
    mixture interaction or categorical parameters.
    """

    @property
    @abstractmethod
    def n_categories(self) -> int:
        """Number of mixture components."""

    @property
    @abstractmethod
    def bas_lat_man(self) -> BaseLatent:
        """Base latent manifold (before mixture)."""

    @property
    def mix_man(self) -> Any:
        """Complete mixture over base latent.

        Returns CompleteMixture[BaseLatent] - typed as Any to avoid circular imports.
        """
        return self.hrm.pst_man

    @property
    @override
    def rho_emb(self) -> LinearEmbedding[BaseLatent, Any]:
        """Embedding from BaseLatent to mixture's observable component.

        The rho correction only affects the observable (BaseLatent) bias,
        not the mixture interaction or categorical parameters.

        Note: Return type uses Any due to generic type limitations with
        CompleteMixture being a subtype of Harmonium.
        """
        # Lazy import to avoid circular dependency
        from .graphical import ObservableEmbedding

        return ObservableEmbedding(self.mix_man)

    # Mixture-specific methods

    def sample_from_posterior(
        self, key: Array, posterior_params: Array, n_samples: int
    ) -> tuple[Array, Array]:
        """Sample (y, k) pairs from the mixture posterior.

        First samples k from the marginal q(k), then samples y from q(y|k).

        Args:
            key: JAX random key
            posterior_params: Mixture natural parameters
            n_samples: Number of samples

        Returns:
            Tuple of (y_samples, k_samples) where:
            - y_samples: shape (n_samples, base_lat_dim)
            - k_samples: shape (n_samples,) cluster indices
        """
        key_k, key_y = jax.random.split(key)

        # Get cluster probabilities using the correct marginal computation
        probs = self.get_cluster_probs(posterior_params)

        # Sample cluster assignments
        k_samples = jax.random.categorical(key_k, jnp.log(probs), shape=(n_samples,))

        # Get component parameters
        components, _ = self.mix_man.split_natural_mixture(posterior_params)
        comp_params_2d = self.mix_man.cmp_man.to_2d(components)

        # Sample y for each k
        def sample_y_given_k(key: Array, k: Array) -> Array:
            y_params = comp_params_2d[k]
            return self.bas_lat_man.sample(key, y_params, 1)[0]

        y_keys = jax.random.split(key_y, n_samples)
        y_samples = jax.vmap(sample_y_given_k)(y_keys, k_samples)

        # Stop gradients: REINFORCE estimator treats samples as fixed.
        # Essential for non-reparameterizable distributions (e.g., Von Mises
        # rejection sampling), and correct for discrete distributions too.
        return jax.lax.stop_gradient(y_samples), jax.lax.stop_gradient(k_samples)

    def get_cluster_probs(self, mixture_params: Array) -> Array:
        """Extract cluster probabilities from mixture parameters.

        For a mixture q(y,k), the marginal over k is:
            q(k) ∝ exp(θ_K[k] + ψ_Y(θ_Y[k]))

        where ψ_Y is the log-partition function of the base latent.

        Args:
            mixture_params: Natural parameters of the mixture

        Returns:
            Array of shape (n_categories,) with cluster probabilities
        """
        # Get component parameters for each cluster
        components, _ = self.mix_man.split_natural_mixture(mixture_params)
        comp_params_2d = self.mix_man.cmp_man.to_2d(components)

        # Get categorical natural parameters
        _, _, cat_params = self.mix_man.split_coords(mixture_params)

        # Compute log-partition for each component
        log_partitions = jax.vmap(self.bas_lat_man.log_partition_function)(
            comp_params_2d
        )

        # Compute unnormalized log probabilities for each cluster k
        # log q(k) = cat_params[k-1] + ψ_Y(θ_Y[k]) for k > 0
        # log q(0) = 0 + ψ_Y(θ_Y[0])
        n_clusters = self.n_categories
        log_probs = jnp.zeros(n_clusters)

        # First cluster (k=0): implicit cat param is 0
        log_probs = log_probs.at[0].set(log_partitions[0])

        # Other clusters (k=1,...,K-1): use cat_params
        for k in range(1, n_clusters):
            log_probs = log_probs.at[k].set(cat_params[k - 1] + log_partitions[k])

        # Normalize (log-sum-exp trick for numerical stability)
        log_probs = log_probs - jax.scipy.special.logsumexp(log_probs)
        return jnp.exp(log_probs)

    def get_component_params(self, mixture_params: Array, k: int) -> Array:
        """Get base latent parameters for a specific mixture component.

        Args:
            mixture_params: Natural parameters of the mixture
            k: Component index (0 to n_categories-1)

        Returns:
            Natural parameters for component k
        """
        components, _ = self.mix_man.split_natural_mixture(mixture_params)
        return self.mix_man.cmp_man.get_replicate(components, k)


@dataclass(frozen=True)
class DifferentiableVariationalHierarchicalMixture[
    Observable: Differentiable,
    BaseLatent: Differentiable,
](
    VariationalHierarchicalMixture[Observable, BaseLatent],
    ABC,
):
    """Differentiable variational hierarchical mixture with ELBO support.

    Provides ELBO computation for mixture models, using REINFORCE gradients
    for non-reparameterizable sampling from the mixture posterior.
    """

    def compute_log_joint(self, params: Array, x: Array, y: Array, k: Array) -> Array:
        """Compute log p(x, y, k) = log p(k) + log p(y|k) + log p(x|y).

        Args:
            params: Full model parameters
            x: Observation
            y: Base latent (shape: base_lat_dim)
            k: Cluster index (scalar)

        Returns:
            Log joint probability (scalar)
        """
        s_x = self.obs_man.sufficient_statistic(x)

        # Get prior mixture parameters
        _, _, prior_mixture_params = self.hrm.split_coords(
            self.generative_params(params)
        )

        # log p(k)
        _, _, cat_params = self.mix_man.split_coords(prior_mixture_params)
        cat_means = self.mix_man.lat_man.to_mean(cat_params)
        probs = self.mix_man.lat_man.to_probs(cat_means)
        log_pk = jnp.log(probs[k])

        # log p(y|k) - component distribution for cluster k
        components, _ = self.mix_man.split_natural_mixture(prior_mixture_params)
        comp_params_2d = self.mix_man.cmp_man.to_2d(components)
        y_params_k = comp_params_2d[k]
        s_y = self.bas_lat_man.sufficient_statistic(y)
        log_py_given_k = (
            jnp.dot(s_y, y_params_k)
            - self.bas_lat_man.log_partition_function(y_params_k)
            + self.bas_lat_man.log_base_measure(y)
        )

        # log p(x|y)
        lkl_params = self.likelihood_at(params, y)
        log_px_given_y = (
            jnp.dot(s_x, lkl_params)
            - self.obs_man.log_partition_function(lkl_params)
            + self.obs_man.log_base_measure(x)
        )

        return log_pk + log_py_given_k + log_px_given_y

    def compute_log_posterior(
        self, posterior_params: Array, y: Array, k: Array
    ) -> Array:
        """Compute log q(y, k | x) from approximate posterior.

        Args:
            posterior_params: Approximate posterior mixture parameters
            y: Base latent
            k: Cluster index

        Returns:
            Log posterior probability (scalar)
        """
        # log q(k)
        _, _, cat_params = self.mix_man.split_coords(posterior_params)
        cat_means = self.mix_man.lat_man.to_mean(cat_params)
        probs = self.mix_man.lat_man.to_probs(cat_means)
        log_qk = jnp.log(probs[k])

        # log q(y|k)
        components, _ = self.mix_man.split_natural_mixture(posterior_params)
        comp_params_2d = self.mix_man.cmp_man.to_2d(components)
        y_params_k = comp_params_2d[k]
        s_y = self.bas_lat_man.sufficient_statistic(y)
        log_qy_given_k = (
            jnp.dot(s_y, y_params_k)
            - self.bas_lat_man.log_partition_function(y_params_k)
            + self.bas_lat_man.log_base_measure(y)
        )

        return log_qk + log_qy_given_k

    def elbo_at(
        self,
        key: Array,
        params: Array,
        x: Array,
        n_samples: int,
        kl_weight: float = 1.0,
    ) -> Array:
        """Compute ELBO using Monte Carlo samples from the mixture posterior.

        ELBO = E_q[log p(x,y,k) - log q(y,k|x)]

        Args:
            key: JAX random key
            params: Full model parameters
            x: Observation
            n_samples: Number of MC samples
            kl_weight: Weight on KL term (for beta-VAE style training)

        Returns:
            ELBO value (scalar)
        """
        # Get approximate posterior
        q_params = self.approximate_posterior_at(params, x)

        # Sample (y, k) from posterior
        y_samples, k_samples = self.sample_from_posterior(key, q_params, n_samples)

        # Compute log joint and log posterior for each sample
        log_joints = jax.vmap(lambda y, k: self.compute_log_joint(params, x, y, k))(
            y_samples, k_samples
        )

        log_posteriors = jax.vmap(
            lambda y, k: self.compute_log_posterior(q_params, y, k)
        )(y_samples, k_samples)

        # ELBO = E_q[log p - log q]
        # With kl_weight: ELBO = E_q[log p(x|y)] - kl_weight * KL(q||p)
        # But since log p - log q = log p(x|y) + log p(y,k) - log q - (log p(y,k) - log q)
        # Simpler to just weight the log_q term
        if kl_weight != 1.0:
            return jnp.mean(log_joints - kl_weight * log_posteriors)
        return jnp.mean(log_joints - log_posteriors)

    def elbo_components(
        self,
        key: Array,
        params: Array,
        x: Array,
        n_samples: int,
        kl_weight: float = 1.0,
    ) -> tuple[Array, Array, Array]:
        """Compute ELBO and its components for monitoring.

        Returns:
            Tuple of (elbo, reconstruction_term, kl_term)
        """
        # Get approximate posterior
        q_params = self.approximate_posterior_at(params, x)

        # Sample (y, k)
        y_samples, k_samples = self.sample_from_posterior(key, q_params, n_samples)

        # Reconstruction: E_q[log p(x|y)]
        s_x = self.obs_man.sufficient_statistic(x)

        def log_likelihood_at_y(y: Array) -> Array:
            lkl_params = self.likelihood_at(params, y)
            return (
                jnp.dot(s_x, lkl_params)
                - self.obs_man.log_partition_function(lkl_params)
                + self.obs_man.log_base_measure(x)
            )

        recon = jnp.mean(jax.vmap(log_likelihood_at_y)(y_samples))

        # KL = E_q[log q(y,k|x)] - E_q[log p(y,k)]
        log_posteriors = jax.vmap(
            lambda y, k: self.compute_log_posterior(q_params, y, k)
        )(y_samples, k_samples)

        # log p(y,k) from joint minus log p(x|y)
        log_joints = jax.vmap(lambda y, k: self.compute_log_joint(params, x, y, k))(
            y_samples, k_samples
        )

        log_priors = log_joints - jax.vmap(log_likelihood_at_y)(y_samples)
        kl = jnp.mean(log_posteriors - log_priors)

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
        """Compute mean ELBO over a batch of observations."""
        batch_size = xs.shape[0]
        keys = jax.random.split(key, batch_size)

        elbos = jax.vmap(lambda k, x: self.elbo_at(k, params, x, n_samples, kl_weight))(
            keys, xs
        )

        return jnp.mean(elbos)

    def sample_from_model(
        self, key: Array, params: Array, n: int
    ) -> tuple[Array, Array, Array]:
        """Sample from the generative model.

        Uses ancestral sampling:
        1. Sample k ~ p(k)
        2. Sample y ~ p(y|k)
        3. Sample x ~ p(x|y)

        Returns:
            Tuple of (x_samples, y_samples, k_samples)
        """
        key_yk, key_x = jax.random.split(key)

        # Get prior mixture parameters
        _, _, prior_mixture_params = self.hrm.split_coords(
            self.generative_params(params)
        )

        # Sample (y, k) from prior mixture
        y_samples, k_samples = self.sample_from_posterior(
            key_yk, prior_mixture_params, n
        )

        # Sample x given y
        def sample_x_given_y(subkey: Array, y: Array) -> Array:
            lkl_params = self.likelihood_at(params, y)
            return self.obs_man.sample(subkey, lkl_params, 1)[0]

        x_keys = jax.random.split(key_x, n)
        x_samples = jax.vmap(sample_x_given_y)(x_keys, y_samples)

        return x_samples, y_samples, k_samples

    # Conjugation error (measures departure from exact conjugation)

    def reduced_learning_signal(self, params: Array, z: Array) -> Array:
        """Compute the reduced learning signal f_tilde(z).

        f_tilde(z) = rho . s_rho(z) - psi_X(theta_X + Theta_{XZ} . s_Z(z))

        where s_rho(z) is the projection of s_Z(z) onto the conjugation manifold.
        For hierarchical mixtures, s_rho extracts the base latent component.

        For a perfectly conjugate model, f_tilde(z) would be constant.

        Args:
            params: Full model parameters
            z: Latent sample (full mixture sample [y, k])

        Returns:
            Scalar value of reduced learning signal
        """
        rho, hrm_params = self.split_coords(params)
        s_z = self.pst_man.sufficient_statistic(z)

        # Project sufficient statistics to conjugation manifold
        # For hierarchical: extract base latent component from mixture
        s_rho = self.rho_emb.project(s_z)

        # Linear term: rho . s_rho(z)
        term1 = jnp.dot(rho, s_rho)

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

    def conjugation_metrics(
        self, key: Array, params: Array, n_samples: int = 100
    ) -> tuple[Array, Array, Array]:
        """Compute conjugation quality metrics under the prior.

        Returns variance, standard deviation, and R² measuring how well
        the linear correction ρ·s(z) approximates ψ_X(η(z)).

        R² = 1 - Var[f̃] / Var[ψ_X(η(z))]

        where f̃(z) = ρ·s(z) - ψ_X(η(z)) is the residual.

        - R² = 1: Perfect conjugation (ψ_X is exactly linear in s(z))
        - R² = 0: Linear correction explains none of the variance
        - R² < 0: Linear correction makes things worse

        Args:
            key: JAX random key
            params: Full model parameters
            n_samples: Number of samples for estimation

        Returns:
            Tuple of (variance, std, r_squared)
        """
        _, hrm_params = self.split_coords(params)
        p_params = self.prior_params(params)
        z_samples = self.pst_man.sample(key, p_params, n_samples)

        # Compute f_tilde values
        f_vals = jax.vmap(lambda z: self.reduced_learning_signal(params, z))(z_samples)

        # Compute psi_X values (the target we're trying to approximate)
        def psi_x_at_z(z: Array) -> Array:
            lkl_params = self.hrm.likelihood_at(hrm_params, z)
            return self.obs_man.log_partition_function(lkl_params)

        psi_vals = jax.vmap(psi_x_at_z)(z_samples)

        var_f = jnp.var(f_vals)
        std_f = jnp.sqrt(var_f)
        var_psi = jnp.var(psi_vals)

        # R² = 1 - Var[residual] / Var[target]
        # Use threshold to avoid numerical instability
        variance_threshold = 1e-6
        raw_r2 = 1.0 - var_f / jnp.maximum(var_psi, variance_threshold)
        r_squared = jnp.where(
            var_psi > variance_threshold,
            jnp.clip(raw_r2, -10.0, 1.0),
            jnp.nan,
        )

        return var_f, std_f, r_squared
