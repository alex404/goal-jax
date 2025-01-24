"""MNIST benchmark models using scipy/sklearn implementations."""

from dataclasses import dataclass
from functools import partial
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from .common import evaluate_clustering
from .types import MNISTData, TwoStageResults


@dataclass(frozen=True)
class PCAGMM:
    """Two-stage model combining PCA dimensionality reduction with Gaussian Mixture Model clustering."""

    _latent_dim: int
    _n_clusters: int

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    def initialize(self, key: Array, data: Array) -> tuple[PCA, GaussianMixture]:
        """Initialize PCA and GMM components."""
        data_np = np.array(data, dtype=np.float64)
        pca = PCA(n_components=self.latent_dim)
        pca.fit(data_np)
        gmm = GaussianMixture(n_components=self._n_clusters, random_state=int(key[0]))
        return pca, gmm

    @partial(jax.jit, static_argnums=(0,))
    def fit(
        self, params0: tuple[PCA, GaussianMixture], data: Array
    ) -> tuple[tuple[PCA, GaussianMixture], Array]:
        """Fit model to data in two stages: PCA followed by GMM."""
        pca, gmm = params0
        data_np = np.array(data, dtype=np.float64)
        latents = pca.transform(data_np)
        gmm.fit(latents)

        reconstruction_errors = self.reconstruction_error(params0, data)
        return (pca, gmm), jnp.mean(reconstruction_errors)

    def encode(self, params: tuple[PCA, GaussianMixture], data: Array) -> Array:
        """Project data into latent space using PCA."""
        pca, _ = params
        data_np = np.array(data, dtype=np.float64)
        return jnp.array(pca.transform(data_np))

    def decode(self, params: tuple[PCA, GaussianMixture], latents: Array) -> Array:
        """Project latent representations back to data space using PCA."""
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
        """Generate new samples by sampling from GMM and projecting to data space."""
        _, gmm = params
        latent_samples = gmm.sample(n_samples)[0]
        return self.decode(params, jnp.array(latent_samples))

    def cluster_assignments(
        self, params: tuple[PCA, GaussianMixture], data: Array
    ) -> Array:
        """Determine cluster assignments for data points."""
        pca, gmm = params
        data_np = np.array(data, dtype=np.float64)
        latents = pca.transform(data_np)
        return jnp.array(gmm.predict(latents))

    def evaluate(
        self,
        key: Array,
        data: MNISTData,
    ) -> TwoStageResults:
        """Evaluate model performance on MNIST dataset."""
        start_time = time()
        params = self.initialize(key, data.train_images)
        final_params, _ = self.fit(params, data.train_images)

        train_clusters = self.cluster_assignments(final_params, data.train_images)
        test_clusters = self.cluster_assignments(final_params, data.test_images)

        name = self.__class__.__name__

        recon_error = float(
            jnp.mean(self.reconstruction_error(final_params, data.test_images))
        )
        train_accuracy = float(evaluate_clustering(train_clusters, data.train_labels))
        test_accuracy = float(evaluate_clustering(test_clusters, data.test_labels))
        training_time = time() - start_time

        return TwoStageResults(
            model_name=name,
            reconstruction_error=recon_error,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            latent_dim=self.latent_dim,
            n_clusters=self.n_clusters,
            training_time=training_time,
        )
