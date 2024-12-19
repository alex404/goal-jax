"""Model implementations for MNIST benchmarks."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from goal.geometry import Mean, Natural, Point, PositiveDefinite
from goal.models import Covariance, Euclidean, HierarchicalMixtureOfGaussians

from .common import ProbabilisticModel, TwoStageModel


@dataclass
class PCAGMMModel(TwoStageModel[tuple[PCA, GaussianMixture]]):
    _latent_dim: int
    _n_clusters: int

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    def initialize(self, key: Array, data: Array) -> tuple[PCA, GaussianMixture]:
        data_np = np.array(data, dtype=np.float64)
        pca = PCA(n_components=self.latent_dim)
        pca.fit(data_np)
        gmm = GaussianMixture(n_components=self._n_clusters, random_state=int(key[0]))
        return pca, gmm

    def fit(
        self, params: tuple[PCA, GaussianMixture], data: Array, n_steps: int
    ) -> tuple[tuple[PCA, GaussianMixture], Array]:
        pca, gmm = params
        data_np = np.array(data, dtype=np.float64)
        latents = pca.transform(data_np)  # type: ignore
        gmm.fit(latents)

        reconstruction_errors = self.reconstruction_error(params, data)
        return (pca, gmm), jnp.full(n_steps, jnp.mean(reconstruction_errors))

    def encode(self, params: tuple[PCA, GaussianMixture], data: Array) -> Array:
        pca, _ = params
        data_np = np.array(data, dtype=np.float64)
        return jnp.array(pca.transform(data_np))

    def decode(self, params: tuple[PCA, GaussianMixture], latents: Array) -> Array:
        pca, _ = params
        latents_np = np.array(latents, dtype=np.float64)
        return jnp.array(pca.inverse_transform(latents_np))

    def reconstruction_error(
        self, params: tuple[PCA, GaussianMixture], data: Array
    ) -> Array:
        """Compute per-sample reconstruction error using encode/decode."""
        reconstructed = self.decode(params, self.encode(params, data))
        return jnp.mean((data - reconstructed) ** 2, axis=1)

    def generate(
        self, params: tuple[PCA, GaussianMixture], key: Array, n_samples: int
    ) -> Array:
        _, gmm = params
        latent_samples = gmm.sample(n_samples)[0]  # type: ignore
        return self.decode(params, jnp.array(latent_samples))

    def cluster_assignments(
        self, params: tuple[PCA, GaussianMixture], data: Array
    ) -> Array:
        pca, gmm = params
        data_np = np.array(data, dtype=np.float64)
        latents = pca.transform(data_np)  # type: ignore
        return jnp.array(gmm.predict(latents))


type HMoG[Rep: PositiveDefinite] = HierarchicalMixtureOfGaussians[Rep]


@dataclass
class HMoGModel[Rep: PositiveDefinite](ProbabilisticModel[Point[Natural, HMoG[Rep]]]):
    model: HierarchicalMixtureOfGaussians[Rep]

    @property
    def latent_dim(self) -> int:
        return self.model.lat_man.obs_man.data_dim

    @property
    def n_clusters(self) -> int:
        return self.model.lat_man.lat_man.dim + 1

    def initialize(self, key: Array, data: Array) -> Point[Natural, HMoG[Rep]]:
        # Initialize observable normal with data statistics
        obs_means = self.model.obs_man.average_sufficient_statistic(data)
        obs_params = self.model.obs_man.to_natural(obs_means)

        # Create latent categorical prior with slight symmetry breaking
        cat_params = self.model.lat_man.lat_man.shape_initialize(key)
        cat_means = self.model.lat_man.lat_man.to_mean(cat_params)

        # Initialize separated components
        with self.model.lat_man as um:
            components = []
            sg = 1.0
            stp = 4 * sg
            strt = -stp * (self.n_clusters - 1) / 2
            rng = jnp.arange(strt, strt + self.n_clusters * stp, stp)

            for i in range(self.n_clusters):
                comp_means = um.obs_man.join_mean_covariance(
                    Point[Mean, Euclidean](jnp.atleast_1d(rng[i])),
                    Point[Mean, Covariance[PositiveDefinite]](jnp.array([sg**2])),
                )
                components.append(comp_means)

            mix_means = um.join_mean_mixture(components, cat_means)
            mix_params = um.to_natural(mix_means)

        # Initialize interaction matrix scaled by precision
        obs_prs = self.model.obs_man.split_location_precision(obs_params)[1]
        int_mat0 = 0.1 * jax.random.normal(key, shape=(2,))
        int_mat = self.model.int_man.from_dense(
            self.model.obs_man.cov_man.to_dense(obs_prs) @ int_mat0
        )

        lkl_params = self.model.lkl_man.join_params(obs_params, int_mat)
        return self.model.join_conjugated(lkl_params, mix_params)

    def fit(
        self, params: Point[Natural, HMoG[Rep]], data: Array, n_steps: int
    ) -> tuple[Point[Natural, HMoG[Rep]], Array]:
        def em_step(
            carry: Point[Natural, HMoG[Rep]], _: Any
        ) -> tuple[Point[Natural, HMoG[Rep]], Array]:
            ll = self.model.average_log_observable_density(carry, data)
            next_params = self.model.expectation_maximization(carry, data)
            return next_params, ll

        final_params, lls = jax.lax.scan(em_step, params, None, length=n_steps)
        return final_params, lls.ravel()

    def log_likelihood(self, params: Point[Natural, HMoG[Rep]], data: Array) -> Array:
        return self.model.log_observable_density(params, data)

    def generate(
        self, params: Point[Natural, HMoG[Rep]], key: Array, n_samples: int
    ) -> Array:
        return self.model.observable_sample(key, params, n_samples)

    def cluster_assignments(
        self, params: Point[Natural, HMoG[Rep]], data: Array
    ) -> Array:
        with self.model as m:
            # Compute posterior over categorical variables
            cat_pst = m.lat_man.prior(m.posterior_at(params, data))
            with m.lat_man.lat_man as lm:
                probs = lm.to_probs(lm.to_mean(cat_pst))
        return jnp.argmax(probs, axis=-1)
