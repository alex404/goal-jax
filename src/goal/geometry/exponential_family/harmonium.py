"""Core definitions for harmonium models. A harmonium is a type of product exponential family over observable and latent variables, with sufficient statistics given by those of the component exponential families, as well as all their pairwise interactions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.combinators import Triple
from ..manifold.embedding import IdentityEmbedding, LinearEmbedding
from ..manifold.linear import AffineMap, LinearMap
from ..manifold.util import batched_mean
from .base import (
    Analytic,
    Differentiable,
    ExponentialFamily,
    Generative,
)


@dataclass(frozen=True)
class Harmonium[
    Observable: ExponentialFamily,
    Posterior: ExponentialFamily,
](
    ExponentialFamily,
    Triple[Observable, LinearMap[Posterior, Observable], Posterior],
    ABC,
):
    """An exponential family harmonium is a product of two exponential families. The first family is over observable variables, and the second is over latent variables. The two families are coupled through an interaction matrix that captures the dependencies between the observable and latent variables.

    The interaction matrix (int_man) is a LinearMap (typically an EmbeddedMap) that can
    internally restrict interactions to submanifolds via embeddings.

    In theory, the joint distribution of a harmonium takes the form

    $$
    \\log p(x,z) = \\theta_X \\cdot \\mathbf s_X(x) + \\theta_Z \\cdot \\mathbf s_Z(z) + \\mathbf s_X(x)\\cdot \\Theta_{XZ} \\cdot \\mathbf s_Z(z) - \\psi_{XZ}(\\theta_X, \\theta_Z, \\Theta_{XZ}),
    $$

    where:

    - $\\theta_X$ are the observable biases,
    - $\\theta_Z$ are the latent biases,
    - $\\Theta_{XZ}$ is the interaction matrix, and
    - $\\mathbf s_X(x)$ and $\\mathbf s_Z(z)$ are sufficient statistics of the observable and latent exponential families, respectively.
    """

    # Contract

    @property
    @abstractmethod
    def int_man(self) -> LinearMap[Posterior, Observable]:
        """Matrix representation of the interaction matrix.

        The LinearMap (typically EmbeddedMap) internally manages any restriction
        of interactions to submanifolds via embeddings."""

    # Templates

    @property
    def obs_man(self) -> Observable:
        """Manifold of observable biases."""
        return self.int_man.cod_man

    @property
    def pst_man(self) -> Posterior:
        """Manifold of posterior specific latent biases."""
        return self.int_man.dom_man

    @property
    def lkl_fun_man(self) -> AffineMap[Posterior, Observable]:
        """Manifold of likelihood distributions $p(x \\mid z)$."""
        return AffineMap(self.int_man, self.int_man.dom_man)

    @property
    def pst_fun_man(self) -> AffineMap[Observable, Posterior]:
        """Manifold of conditional posterior distributions $p(z \\mid x)$."""
        return AffineMap(self.int_man.trn_man, self.int_man.cod_man)

    def likelihood_function(self, params: Array) -> Array:
        """Natural parameters of the likelihood distribution as an affine function.

        The affine map is $\\eta \\to \\theta_X + \\Theta_{XZ} \\cdot \\eta$.
        """
        obs_params, int_params, _ = self.split_coords(params)
        return self.lkl_fun_man.join_coords(obs_params, int_params)

    def posterior_function(self, params: Array) -> Array:
        """Natural parameters of the posterior distribution as an affine function.

        The affine map is $\\eta \\to \\theta_Z + \\eta \\cdot \\Theta_{XZ}$.
        """
        _, int_params, lat_params = self.split_coords(params)
        int_mat_t = self.int_man.transpose(int_params)
        return self.pst_fun_man.join_coords(lat_params, int_mat_t)

    def likelihood_at(self, params: Array, z: Array) -> Array:
        """Compute natural parameters of likelihood distribution $p(x \\mid z)$ at a given $z$.

        Given a latent state $z$ with sufficient statistics $s(z)$, computes natural parameters $\\theta_X$ of the conditional distribution:

        $$p(x \\mid z) \\propto \\exp(\\theta_X \\cdot s_X(x) + s_X(x) \\cdot \\Theta_{XZ} \\cdot s_Z(z))$$
        """
        # LinearMap handles projection internally via embeddings
        mz = self.pst_man.sufficient_statistic(z)
        return self.lkl_fun_man(self.likelihood_function(params), mz)

    def posterior_at(self, params: Array, x: Array) -> Array:
        """Compute natural parameters of posterior distribution $p(z \\mid x)$.

        Given an observation $x$ with sufficient statistics $s(x)$, computes natural
        parameters $\\theta_Z$ of the conditional distribution:

        $$p(z \\mid x) \\propto \\exp(\\theta_Z \\cdot s_Z(z) + s_X(x) \\cdot \\Theta_{XZ} \\cdot s_Z(z))$$
        """
        # LinearMap handles projection internally via embeddings
        mx = self.obs_man.sufficient_statistic(x)
        return self.pst_fun_man(self.posterior_function(params), mx)

    # Overrides

    @property
    @override
    def fst_man(self) -> Observable:
        return self.obs_man

    @property
    @override
    def snd_man(self) -> LinearMap[Posterior, Observable]:
        return self.int_man

    @property
    @override
    def trd_man(self) -> Posterior:
        return self.pst_man

    @property
    @override
    def data_dim(self) -> int:
        """Total dimension of data points."""
        return self.obs_man.data_dim + self.pst_man.data_dim

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Compute sufficient statistics for joint observation."""
        obs_x = x[: self.obs_man.data_dim]
        lat_x = x[self.obs_man.data_dim :]

        obs_stats = self.obs_man.sufficient_statistic(obs_x)
        lat_stats = self.pst_man.sufficient_statistic(lat_x)

        # LinearMap.outer_product handles embedding internally
        int_stats = self.int_man.outer_product(obs_stats, lat_stats)

        return self.join_coords(obs_stats, int_stats, lat_stats)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Compute log base measure for joint observation."""
        obs_x = x[..., self.obs_man.data_dim :]
        lat_x = x[self.obs_man.data_dim :]

        return self.obs_man.log_base_measure(obs_x) + self.pst_man.log_base_measure(
            lat_x
        )

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize harmonium parameters. Observable and latent biases are initialized using their respective initialization strategies. The interaction matrix is initialized with random entries scaled by 1/sqrt(dim) to maintain reasonable magnitudes for the interactions."""
        from ..manifold.linear import EmbeddedMap

        keys = jax.random.split(key, 3)

        # Initialize biases using component initialization
        obs_params = self.obs_man.initialize(keys[0], location, shape)
        lat_params = self.pst_man.initialize(keys[1], location, shape)

        # Initialize interaction matrix with appropriate scaling
        # Get internal dimensions from the linear map
        if isinstance(self.int_man, EmbeddedMap):
            obs_dim = self.int_man.cod_emb.sub_man.dim
            lat_dim = self.int_man.dom_emb.sub_man.dim
        else:
            # Fallback to full manifold dimensions
            obs_dim = self.obs_man.dim
            lat_dim = self.pst_man.dim

        scaling = shape / jnp.sqrt(obs_dim * lat_dim)

        noise = scaling * jax.random.normal(keys[2], shape=[self.int_man.dim])
        int_params = noise

        return self.join_coords(obs_params, int_params, lat_params)

    @override
    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize harmonium using sample data for observable biases. Uses sample data to initialize observable biases, while latent biases and interaction matrix use standard initialization."""
        from ..manifold.linear import EmbeddedMap

        keys = jax.random.split(key, 3)

        # Initialize observable biases from data
        obs_params = self.obs_man.initialize_from_sample(
            keys[0], sample, location, shape
        )

        # Use standard initialization for everything else
        lat_params = self.pst_man.initialize(keys[1], location, shape)

        # Initialize interaction matrix with appropriate scaling
        if isinstance(self.int_man, EmbeddedMap):
            obs_dim = self.int_man.cod_emb.sub_man.dim  # type: ignore
            lat_dim = self.int_man.dom_emb.sub_man.dim  # type: ignore
        else:
            # Fallback to full manifold dimensions
            obs_dim = self.obs_man.dim
            lat_dim = self.pst_man.dim

        scaling = shape / jnp.sqrt(obs_dim * lat_dim)

        noise = scaling * jax.random.normal(keys[2], shape=[self.int_man.dim])
        int_params = noise
        return self.join_coords(obs_params, int_params, lat_params)


class GibbsHarmonium[
    Observable: Generative,
    Posterior: Generative,
](
    Harmonium[Observable, Posterior],
    Generative,
    ABC,
):
    """A harmonium supporting Gibbs sampling without conjugation structure.

    This class provides sampling and contrastive divergence methods for
    harmoniums where both observable and posterior distributions are Generative.
    No conjugation structure is required, making this suitable for models like
    Restricted Boltzmann Machines.

    The key insight is that Gibbs sampling and contrastive divergence only require:
    - `likelihood_at(params, z)` to get observable params given latent
    - `posterior_at(params, x)` to get latent params given observable
    - Both `obs_man` and `pst_man` to be Generative (have sample/gibbs_step)
    """

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Sample from joint distribution via Gibbs chain with burn-in.

        Uses Gibbs sampling starting from zero initialization with a burn-in
        period to reach the stationary distribution.

        Args:
            key: JAX random key
            params: Natural parameters
            n: Number of samples

        Returns:
            Samples from joint, shape (n, data_dim)
        """
        key_burn, key_collect = jax.random.split(key)

        # Zero initialization
        obs_init = jnp.zeros(self.obs_man.data_dim)
        pst_init = jnp.zeros(self.pst_man.data_dim)
        init_state = jnp.concatenate([obs_init, pst_init])

        # Burn-in (100 steps)
        burned = self.gibbs_chain(key_burn, params, init_state, 100)

        # Collect n samples
        keys = jax.random.split(key_collect, n)

        def collect(carry: Array, step_key: Array) -> tuple[Array, Array]:
            state = self.gibbs_step(step_key, params, carry)
            return state, state

        _, samples = jax.lax.scan(collect, burned, keys)
        return samples

    @override
    def gibbs_step(self, key: Array, params: Array, state: Array) -> Array:
        """Gibbs step alternating between likelihood and posterior.

        1. Sample x ~ p(x|z) using likelihood conditional
        2. Sample z ~ p(z|x) using posterior conditional

        Args:
            key: JAX random key
            params: Natural parameters
            state: Current state [x, z] concatenated

        Returns:
            New state [x_new, z_new] after one Gibbs sweep
        """
        key1, key2 = jax.random.split(key)

        # Split state into observable and latent
        obs_dim = self.obs_man.data_dim
        x, z = state[:obs_dim], state[obs_dim:]

        # Step 1: Sample x|z from likelihood
        lkl_params = self.likelihood_at(params, z)
        x_new = self.obs_man.gibbs_step(key1, lkl_params, x)

        # Step 2: Sample z|x from posterior
        post_params = self.posterior_at(params, x_new)
        z_new = self.pst_man.gibbs_step(key2, post_params, z)

        # Ensure output dtype matches input for JAX scan compatibility
        result = jnp.concatenate([x_new, z_new])
        return result.astype(state.dtype)

    def contrastive_divergence_step(
        self,
        key: Array,
        params: Array,
        x: Array,
        k: int = 1,
    ) -> Array:
        """Compute CD-k gradient contribution for a single observation.

        Algorithm:
        1. Sample z_0 from posterior p(z|x) (positive phase)
        2. Run k Gibbs steps on joint (x,z) alternating x|z and z|x (negative phase)
        3. Return positive_stats - negative_stats

        Args:
            key: JAX random key
            params: Natural parameters
            x: Single observation
            k: Number of Gibbs steps

        Returns:
            CD gradient statistics (same shape as params)
        """
        key1, key2 = jax.random.split(key)

        # Positive phase: sample z from posterior p(z|x), keep x clamped
        post_params = self.posterior_at(params, x)
        z_0 = self.pst_man.gibbs_step(
            key1, post_params, jnp.zeros(self.pst_man.data_dim)
        )

        # Compute positive phase sufficient statistics
        obs_stats_pos = self.obs_man.sufficient_statistic(x)
        z_0_stats = self.pst_man.sufficient_statistic(z_0)
        int_stats_pos = self.int_man.outer_product(obs_stats_pos, z_0_stats)
        positive_stats = self.join_coords(obs_stats_pos, int_stats_pos, z_0_stats)

        # Negative phase: run k Gibbs steps on JOINT (x,z)
        # This alternates between sampling x|z and z|x
        initial_state = jnp.concatenate([x, z_0])
        final_state = self.gibbs_chain(key2, params, initial_state, k)

        # Extract final x and z
        obs_dim = self.obs_man.data_dim
        x_k, z_k = final_state[:obs_dim], final_state[obs_dim:]

        # Compute negative phase sufficient statistics
        obs_stats_neg = self.obs_man.sufficient_statistic(x_k)
        z_k_stats = self.pst_man.sufficient_statistic(z_k)
        int_stats_neg = self.int_man.outer_product(obs_stats_neg, z_k_stats)
        negative_stats = self.join_coords(obs_stats_neg, int_stats_neg, z_k_stats)

        # Return gradient of -log p(x) for direct use with optimizers
        # (negative - positive) approximates E_model[s] - E_data[s]
        return negative_stats - positive_stats

    def mean_contrastive_divergence_gradient(
        self,
        key: Array,
        params: Array,
        xs: Array,
        k: int = 1,
    ) -> Array:
        """Compute average CD-k gradient over a batch of observations.

        Args:
            key: JAX random key
            params: Natural parameters
            xs: Batch of observations, shape (n_obs, obs_dim)
            k: Number of Gibbs steps

        Returns:
            Average CD gradient (same shape as params)
        """
        n = xs.shape[0]
        keys = jax.random.split(key, n)

        cd_grads = jax.vmap(
            self.contrastive_divergence_step, in_axes=(0, None, 0, None)
        )(keys, params, xs, k)

        return jnp.mean(cd_grads, axis=0)


class Conjugated[
    Observable: Differentiable,
    Posterior: ExponentialFamily,
    Prior: Generative,
](
    Harmonium[Observable, Posterior],
    Generative,
    ABC,
):
    """In general, the prior $p(x)$ of a harmonium model is not exponential family distribution. A conjugated harmonium, on the other hand, is one for which the prior $p(z)$ is in the same exponential family as the posterior $p(x \\mid z)$. It follows that a conjugated harmonium has so called "conjugation parameters" that complement the latent parameters, and facilitate a variety of computations and algorithms for harmoniums."""

    # Contract

    @property
    @abstractmethod
    def pst_prr_emb(self) -> LinearEmbedding[Posterior, Prior]:
        """Embedding of the posterior latent submanifold into the prior latent manifold."""

    @abstractmethod
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute conjugation parameters for the harmonium.

        Conjugated harmoniums have so called "conjugation parameters" $\\rho$ that simplify many of the computations in the model.

        """

    # Templates

    @property
    def prr_man(self) -> Prior:
        """Manifold of the prior and conjugation parameters."""
        return self.pst_prr_emb.amb_man

    def prior(self, params: Array) -> Array:
        """Natural parameters of the prior distribution $p(z)$."""
        obs_params, int_params, lat_params = self.split_coords(params)
        lkl_params = self.lkl_fun_man.join_coords(obs_params, int_params)
        rho = self.conjugation_parameters(lkl_params)
        return self.pst_prr_emb.translate(rho, lat_params)

    def split_conjugated(self, params: Array) -> tuple[Array, Array]:
        """Split conjugated harmonium into likelihood and prior."""
        lkl_params = self.likelihood_function(params)
        return lkl_params, self.prior(params)

    # Overrides

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Generate samples from the harmonium distribution."""
        # Split up the sampling key
        key1, key2 = jax.random.split(key)

        # Sample from adjusted latent distribution p(z)
        nat_prior = self.prior(params)
        z_sample = self.prr_man.sample(key1, nat_prior, n)

        # Extract the variables needed to evaluate the likelihood
        z0_sample = self.extract_likelihood_input(z_sample)

        # Vectorize sampling from conditional distributions
        x_params = jax.vmap(self.likelihood_at, in_axes=(None, 0))(params, z0_sample)

        # Sample from conditionals p(x|z) in parallel
        x_sample = jax.vmap(self.obs_man.sample, in_axes=(0, 0, None))(
            jax.random.split(key2, n), x_params, 1
        ).reshape((n, -1))

        # Concatenate samples along data dimension
        return jnp.concatenate([x_sample, z_sample], axis=-1)

    def extract_likelihood_input(self, prr_sample: Array) -> Array:
        """Extract data required to evaluate the likelihood from a prior sample.

        This method takes a sample from the prior distribution p(z) and extracts the variables needed for conditioning the likelihood p(x|·). In standard harmoniums, this returns the full latent sample. In hierarchical models, this extracts only the immediate child latent (e.g., y from a yz sample).

        Parameters
        ----------
        z_sample : Array
            A sample from the prior distribution, shape (n, prior_latent_dim)

        Returns
        -------
        Array
            Variables for likelihood conditioning, shape (n, ·)
        """
        return prr_sample

    # Templates

    def observable_sample(self, key: Array, params: Array, n: int = 1) -> Array:
        xzs = self.sample(key, params, n)
        return xzs[:, : self.obs_man.data_dim]


class SymmetricConjugated[
    Observable: Differentiable,
    Latent: Generative,
](
    Conjugated[Observable, Latent, Latent],
    ABC,
):
    """A differentiable conjugated harmonium where the space of posteriors is the same as the space of priors (as opposed to a strict submanifold)."""

    @property
    @abstractmethod
    def lat_man(self) -> Latent:
        """Manifold of latent biases."""

    @property
    @override
    def pst_prr_emb(self) -> LinearEmbedding[Latent, Latent]:
        """Embedding of the posterior latent submanifold into the prior latent manifold."""
        return IdentityEmbedding(self.lat_man)

    @property
    @override
    def pst_man(self) -> Latent:
        """Manifold of posterior specific latent biases."""
        return self.lat_man

    @property
    @override
    def prr_man(self) -> Latent:
        """Manifold of posterior specific latent biases."""
        return self.lat_man

    def join_conjugated(self, lkl_params: Array, prior_params: Array) -> Array:
        """Join likelihood and prior parameters into a conjugated harmonium."""
        rho = self.conjugation_parameters(lkl_params)
        lat_params = prior_params - rho
        obs_params, int_params = self.lkl_fun_man.split_coords(lkl_params)
        return self.join_coords(obs_params, int_params, lat_params)


class DifferentiableConjugated[
    Observable: Differentiable,
    Posterior: Differentiable,
    Prior: Differentiable,
](
    Conjugated[Observable, Posterior, Prior],
    GibbsHarmonium[Observable, Posterior],
    Differentiable,
    ABC,
):
    """A conjugated harmonium with an analytical log-partition function.

    Inherits Gibbs sampling and contrastive divergence from GibbsHarmonium.
    Provides additional methods using conjugation structure for efficient
    density computation.
    """

    # Overrides

    @override
    def log_partition_function(self, params: Array) -> Array:
        """Compute the log partition function using conjugation parameters.

        The log partition function of a conjugated harmonium can be written as:

        $$\\psi(\\theta) = \\psi_Z(\\theta_Z + \\rho) + \\psi_X(\\theta_X)$$
        """
        # Split parameters
        obs_params, int_params, lat_params = self.split_coords(params)
        lkl_params = self.lkl_fun_man.join_coords(obs_params, int_params)

        # Get conjugation parameters
        chi = self.obs_man.log_partition_function(obs_params)
        rho = self.conjugation_parameters(lkl_params)

        # Compute adjusted latent parameters
        adjusted_lat = self.pst_prr_emb.translate(rho, lat_params)

        # Use latent family's partition function
        return self.prr_man.log_partition_function(adjusted_lat) + chi

    # Templates
    def log_observable_density(self, params: Array, x: Array) -> Array:
        """Compute log density of the observable distribution $p(x)$.

        For a conjugated harmonium, the observable density can be computed as:

        $$
        \\log p(x) = \\theta_X \\cdot s_X(x) + \\psi_Z(\\theta_Z + \\mathbf s_X(x) \\cdot \\Theta_{XZ}) - \\psi_Z(\\theta_Z + \\rho) - \\psi_X(\\theta_X) + \\log \\mu_X(x)
        $$
        """
        obs_params, _, _ = self.split_coords(params)

        chi = self.obs_man.log_partition_function(obs_params)
        obs_stats = self.obs_man.sufficient_statistic(x)
        prr = self.prior(params)

        log_density = jnp.dot(obs_params, obs_stats)
        log_density += self.pst_man.log_partition_function(self.posterior_at(params, x))
        log_density -= self.prr_man.log_partition_function(prr) + chi

        return log_density + self.obs_man.log_base_measure(x)

    def observable_density(self, params: Array, x: Array) -> Array:
        """Compute density of the observable distribution $p(x)$."""
        return jnp.exp(self.log_observable_density(params, x))

    def average_log_observable_density(
        self, params: Array, xs: Array, batch_size: int = 2048
    ) -> Array:
        """Compute average log density over a batch of observations."""

        def _log_density(x: Array) -> Array:
            return self.log_observable_density(params, x)

        return batched_mean(_log_density, xs, batch_size)

    def posterior_statistics(self, params: Array, x: Array) -> Array:
        """
        Compute joint expectations for a single observation. In particular we
            1. Compute sufficient statistics $\\mathbf s_X(x)$,
            2. Get natural parameters of $p(z \\mid x)$ given by $\\theta_Z + \\mathbf s_X(x) \\cdot \\Theta_{XZ}$,
            3. Convert to mean parameters $\\mathbb E[\\mathbf s_Z(z) \\mid x]$, and
            4. Form joint expectations $(\\mathbf s_X(x), \\mathbb E[\\mathbf s_Z(z) \\mid x], \\mathbf s_X(x) \\otimes \\mathbb E[\\mathbf s_Z(z) \\mid x])$.
        """
        # Get sufficient statistics of observation
        obs_stats = self.obs_man.sufficient_statistic(x)

        # Get posterior parameters for this observation
        # AffineMap and LinearMap handle projection internally
        post_map = self.posterior_function(params)
        lat_params = self.pst_fun_man(post_map, obs_stats)

        # Convert to mean parameters (expected sufficient statistics)
        lat_means = self.pst_man.to_mean(lat_params)

        # Form interaction term via outer product
        # LinearMap.outer_product handles projection internally
        int_means = self.int_man.outer_product(obs_stats, lat_means)

        # Join parameters into harmonium point
        return self.join_coords(obs_stats, int_means, lat_means)

    def mean_posterior_statistics(
        self: Self,
        params: Array,
        xs: Array,
        batch_size: int = 256,
    ) -> Array:
        """Compute average joint expectations over a batch of observations."""

        def infer_missing_expectations(x: Array) -> Array:
            return self.posterior_statistics(params, x)

        return batched_mean(infer_missing_expectations, xs, batch_size)


class AnalyticConjugated[
    Observable: Differentiable,
    Latent: Analytic,
](
    SymmetricConjugated[Observable, Latent],
    DifferentiableConjugated[Observable, Latent, Latent],
    Analytic,
    ABC,
):
    """A conjugated harmonium with an analytically tractable negative entropy. Among other things, this allows for closed-form computation of the KL divergence between two conjugated harmoniums, and the expectation-maximization algorithm."""

    # Contract

    @abstractmethod
    def to_natural_likelihood(self, means: Array) -> Array:
        """Given a harmonium density in mean coords, returns the natural parameters of the likelihood."""

    # Overrides

    @override
    def to_natural(self, means: Array) -> Array:
        """Convert from mean to natural parameters.

        Uses `to_natural_likelihood` for observable and interaction components, then computes latent natural parameters using conjugation structure:

        $\\theta_Z = \\nabla \\phi_Z(\\eta_Z) - \\rho$,

        where $\\rho$ are the conjugation parameters of the likelihood.
        """
        lkl_params = self.to_natural_likelihood(means)
        mean_lat = self.split_coords(means)[2]
        nat_lat = self.pst_man.to_natural(mean_lat)
        return self.join_conjugated(lkl_params, nat_lat)

    @override
    def negative_entropy(self, means: Array) -> Array:
        """Compute negative entropy using $\\phi(\\eta) = \\eta \\cdot \\theta - \\psi(\\theta)$, $\\theta = \\nabla \\phi(\\eta)$."""
        params = self.to_natural(means)

        log_partition = self.log_partition_function(params)
        return jnp.dot(params, means) - log_partition

    # Templates

    def expectation_maximization(self, params: Array, xs: Array) -> Array:
        """Perform a single iteration of the EM algorithm."""
        q = self.mean_posterior_statistics(params, xs)
        return self.to_natural(q)
