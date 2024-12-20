"""Model implementations for MNIST benchmarks."""

from dataclasses import dataclass
from time import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.mixture import GaussianMixture

from goal.geometry import Natural, Point, PositiveDefinite
from goal.models import Covariance, Euclidean, HierarchicalMixtureOfGaussians

from .common import (
    MNISTData,
    ProbabilisticModel,
    ProbabilisticResults,
    TwoStageModel,
    TwoStageResults,
)


def evaluate_clustering(cluster_assignments: Array, true_labels: Array) -> float:
    """Evaluate clustering by finding optimal label assignment."""
    n_clusters = int(jnp.max(cluster_assignments)) + 1
    n_classes = int(jnp.max(true_labels)) + 1

    # Compute cluster-class frequency matrix
    freq_matrix = np.zeros((n_clusters, n_classes))
    for i in range(len(true_labels)):
        freq_matrix[int(cluster_assignments[i]), int(true_labels[i])] += 1

    # Assign each cluster to its most frequent class
    cluster_to_class = np.argmax(freq_matrix, axis=1)
    predicted_labels = jnp.array([cluster_to_class[i] for i in cluster_assignments])

    return float(accuracy_score(true_labels, predicted_labels))


@dataclass
class PCAGMM(TwoStageModel[tuple[PCA, GaussianMixture]]):
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
        self, params: tuple[PCA, GaussianMixture], data: Array
    ) -> tuple[tuple[PCA, GaussianMixture], Array]:
        pca, gmm = params
        data_np = np.array(data, dtype=np.float64)
        latents = pca.transform(data_np)  # type: ignore
        gmm.fit(latents)

        reconstruction_errors = self.reconstruction_error(params, data)
        return (pca, gmm), jnp.mean(reconstruction_errors)

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

    def evaluate(
        self,
        key: Array,
        data: MNISTData,
    ) -> TwoStageResults:
        """Evaluate two-stage model performance."""
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


@dataclass
class HMoG[Rep: PositiveDefinite](
    ProbabilisticModel[Point[Natural, HierarchicalMixtureOfGaussians[Rep]]]
):
    model: HierarchicalMixtureOfGaussians[Rep]
    n_epochs: int

    @property
    def latent_dim(self) -> int:
        return self.model.lat_man.obs_man.data_dim

    @property
    def n_clusters(self) -> int:
        return self.model.lat_man.lat_man.dim + 1

    def initialize(
        self, key: Array, data: Array
    ) -> Point[Natural, HierarchicalMixtureOfGaussians[Rep]]:
        keys = jax.random.split(key, 4)
        key_cat, key_vertices, key_cov, key_int = keys

        # Initialize observable parameters from data statistics
        obs_means = self.model.obs_man.average_sufficient_statistic(data)
        obs_params = self.model.obs_man.to_natural(obs_means)

        # Create latent categorical prior with slight symmetry breaking
        cat_params = self.model.lat_man.lat_man.shape_initialize(key_cat)
        cat_means = self.model.lat_man.lat_man.to_mean(cat_params)

        # Initialize separated components in latent space
        with self.model.lat_man as um:
            latent_dim = um.obs_man.data_dim
            scale = 2.0  # Scale factor for separation - 2 standard deviations

            # Generate vertices of a hypercube centered at origin
            vertices = scale * jax.random.choice(
                key_vertices,
                jnp.array([-1.0, 1.0]),
                shape=(self.n_clusters, latent_dim),
            )

            components = []
            key_covs = jax.random.split(key_cov, self.n_clusters)
            for i in range(self.n_clusters):
                cov: Point[Natural, Covariance[PositiveDefinite]] = (
                    um.obs_man.cov_man.shape_initialize(key_covs[i])
                )
                comp_means = um.obs_man.join_location_precision(
                    Point[Natural, Euclidean](vertices[i]), cov
                )
                components.append(comp_means)

            mix_means = um.join_mean_mixture(components, cat_means)
            mix_params = um.to_natural(mix_means)

        int_mat = self.model.int_man.shape_initialize(key_int)

        lkl_params = self.model.lkl_man.join_params(obs_params, int_mat)
        return self.model.join_conjugated(lkl_params, mix_params)

    def fit(
        self, params: Point[Natural, HierarchicalMixtureOfGaussians[Rep]], data: Array
    ) -> tuple[Point[Natural, HierarchicalMixtureOfGaussians[Rep]], Array]:
        def em_step(
            carry: Point[Natural, HierarchicalMixtureOfGaussians[Rep]], _: Any
        ) -> tuple[Point[Natural, HierarchicalMixtureOfGaussians[Rep]], Array]:
            ll = self.model.average_log_observable_density(carry, data)
            next_params = self.model.expectation_maximization(carry, data)
            return next_params, ll

        final_params, lls = jax.lax.scan(em_step, params, None, length=self.n_epochs)
        return final_params, lls.ravel()

    def log_likelihood(
        self, params: Point[Natural, HierarchicalMixtureOfGaussians[Rep]], data: Array
    ) -> Array:
        return self.model.average_log_observable_density(params, data)

    def generate(
        self,
        params: Point[Natural, HierarchicalMixtureOfGaussians[Rep]],
        key: Array,
        n_samples: int,
    ) -> Array:
        return self.model.observable_sample(key, params, n_samples)

    def cluster_assignments(
        self, params: Point[Natural, HierarchicalMixtureOfGaussians[Rep]], data: Array
    ) -> Array:
        def data_point_cluster(x: Array) -> Array:
            with self.model as m:
                # Compute posterior over categorical variables
                cat_pst = m.lat_man.prior(m.posterior_at(params, x))
                with m.lat_man.lat_man as lm:
                    probs = lm.to_probs(lm.to_mean(cat_pst))
            return jnp.argmax(probs, axis=-1)

        return jax.vmap(data_point_cluster)(data)

    def evaluate(
        self,
        key: Array,
        data: MNISTData,
    ) -> ProbabilisticResults:
        """Evaluate probabilistic model performance."""
        start_time = time()
        params = self.initialize(key, data.train_images)
        final_params, train_lls = self.fit(params, data.train_images)

        train_clusters = self.cluster_assignments(final_params, data.train_images)
        test_clusters = self.cluster_assignments(final_params, data.test_images)

        model_name = self.__class__.__name__
        final_test_log_likelihood = float(
            self.log_likelihood(final_params, data.test_images)
        )
        train_accuracy = float(evaluate_clustering(train_clusters, data.train_labels))
        test_accuracy = float(evaluate_clustering(test_clusters, data.test_labels))
        training_time = time() - start_time

        return ProbabilisticResults(
            model_name=model_name,
            train_log_likelihood=train_lls.tolist(),
            final_train_log_likelihood=float(train_lls[-1]),
            final_test_log_likelihood=final_test_log_likelihood,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            latent_dim=self.latent_dim,
            n_clusters=self.n_clusters,
            n_parameters=self.model.dim,
            training_time=training_time,
        )
