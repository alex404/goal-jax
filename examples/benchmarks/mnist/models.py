"""Model implementations for MNIST benchmarks."""

from dataclasses import dataclass
from functools import partial
from time import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, debug
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.mixture import GaussianMixture

from goal.geometry import Mean, Natural, Point, PositiveDefinite
from goal.models import (
    FullNormal,
    HierarchicalMixtureOfGaussians,
    LinearGaussianModel,
    Mixture,
    Normal,
)

from .common import (
    MNISTData,
    ProbabilisticModel,
    ProbabilisticResults,
    TwoStageModel,
    TwoStageResults,
)

JITTER = 1e-3
MIN_VAR = 1


def set_min_component_variance[Rep: PositiveDefinite](
    man: Mixture[Normal[Rep]],
    params: Point[Mean, Mixture[Normal[Rep]]],
) -> Point[Mean, Mixture[Normal[Rep]]]:
    """Set minimum variance for each component in the mixture."""
    nrm_meanss, cat_means = man.split_mean_mixture(params)

    adjusted_means = [
        man.obs_man.regularize_covariance(nrm_means, jitter=JITTER, min_var=MIN_VAR)
        for nrm_means in nrm_meanss
    ]

    return man.join_mean_mixture(adjusted_means, cat_means)


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


@dataclass(frozen=True)
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

    @partial(jax.jit, static_argnums=(0,))
    def fit(
        self, params0: tuple[PCA, GaussianMixture], data: Array
    ) -> tuple[tuple[PCA, GaussianMixture], Array]:
        pca, gmm = params0
        data_np = np.array(data, dtype=np.float64)
        latents = pca.transform(data_np)  # type: ignore
        gmm.fit(latents)

        reconstruction_errors = self.reconstruction_error(params0, data)
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
        final_params, _ = self.fit(params, data.train_images)  # type: ignore

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


@dataclass(frozen=True)
class HMoG[Rep: PositiveDefinite](
    ProbabilisticModel[Point[Natural, HierarchicalMixtureOfGaussians[Rep]]]
):
    model: HierarchicalMixtureOfGaussians[Rep]
    n_epochs: int
    stage1_epochs: int
    stage2_epochs: int

    def initialize(
        self, key: Array, data: Array
    ) -> Point[Natural, HierarchicalMixtureOfGaussians[Rep]]:
        keys = jax.random.split(key, 3)
        key_cat, key_comp, key_int = keys

        # Initialize observable parameters from data statistics
        obs_means = self.model.obs_man.average_sufficient_statistic(data)
        obs_means = self.model.obs_man.regularize_covariance(obs_means, JITTER, MIN_VAR)
        obs_params = self.model.obs_man.to_natural(obs_means)

        # Create latent categorical prior with slight symmetry breaking
        cat_params = self.model.lat_man.lat_man.shape_initialize(key_cat)
        cat_means = self.model.lat_man.lat_man.to_mean(cat_params)

        # Initialize separated components in latent space
        with self.model.lat_man as um:
            key_comps = jax.random.split(key_comp, self.n_clusters)
            components = []
            for key_compi in key_comps:
                comp_params = um.obs_man.shape_initialize(key_compi)
                components.append(um.obs_man.to_mean(comp_params))

            mix_means = um.join_mean_mixture(components, cat_means)
            mix_params = um.to_natural(mix_means)

        int_mat = self.model.int_man.shape_initialize(key_int)

        lkl_params = self.model.lkl_man.join_params(obs_params, int_mat)
        return self.model.join_conjugated(lkl_params, mix_params)

    @partial(jax.jit, static_argnums=(0,))
    def fit(
        self,
        params0: Point[Natural, HierarchicalMixtureOfGaussians[Rep]],
        data: Array,
    ) -> tuple[Point[Natural, HierarchicalMixtureOfGaussians[Rep]], Array]:
        lkl_params0, mix_params0 = self.model.split_conjugated(params0)

        with self.model.lwr_man as lm:
            z = lm.lat_man.standard_normal()
            lgm_params0 = lm.join_conjugated(lkl_params0, lm.lat_man.to_natural(z))

            def em_step1(
                params: Point[Natural, LinearGaussianModel[Rep]], _: Any
            ) -> tuple[Point[Natural, LinearGaussianModel[Rep]], Array]:
                ll = lm.average_log_observable_density(params, data)
                means = lm.expectation_step(params, data)
                obs_means, int_means, lat_means = lm.split_params(means)
                obs_means = lm.obs_man.regularize_covariance(obs_means, JITTER, MIN_VAR)
                means = lm.join_params(obs_means, int_means, lat_means)
                params1 = lm.to_natural(means)
                lkl_params = lm.likelihood_function(params1)
                z = lm.lat_man.to_natural(lm.lat_man.standard_normal())
                next_params = lm.join_conjugated(lkl_params, z)
                debug.print("Stage 1 LL: {}", ll)
                return next_params, ll

            lgm_params1, lls1 = jax.lax.scan(
                em_step1, lgm_params0, None, length=self.stage1_epochs
            )
            lkl_params1 = lm.likelihood_function(lgm_params1)

        def em_step2(
            params: Point[Natural, Mixture[FullNormal]], _: Any
        ) -> tuple[Point[Natural, Mixture[FullNormal]], Array]:
            hmog_params = self.model.join_conjugated(lkl_params1, params)
            ll = self.model.average_log_observable_density(hmog_params, data)
            hmog_means = self.model.expectation_step(hmog_params, data)
            mix_means = self.model.split_params(hmog_means)[2]
            mix_means = set_min_component_variance(self.model.lat_man, mix_means)
            next_params = self.model.lat_man.to_natural(mix_means)
            debug.print("Stage 2 LL: {}", ll)
            return next_params, ll

        mix_params1, lls2 = jax.lax.scan(
            em_step2, mix_params0, None, length=self.stage2_epochs
        )

        def em_step3(
            params: Point[Natural, HierarchicalMixtureOfGaussians[Rep]], _: Any
        ) -> tuple[Point[Natural, HierarchicalMixtureOfGaussians[Rep]], Array]:
            ll = self.model.average_log_observable_density(params, data)
            means = self.model.expectation_step(params, data)
            obs_means, int_means, mix_means = self.model.split_params(means)
            obs_means = lm.obs_man.regularize_covariance(obs_means, JITTER, MIN_VAR)
            mix_means = set_min_component_variance(self.model.lat_man, mix_means)
            means = self.model.join_params(obs_means, int_means, mix_means)
            next_params = self.model.to_natural(means)
            debug.print("Stage 3 LL: {}", ll)
            return next_params, ll

        params1 = self.model.join_conjugated(lkl_params1, mix_params1)
        stage3_epochs = self.n_epochs - self.stage1_epochs - self.stage2_epochs
        final_params, lls3 = jax.lax.scan(em_step3, params1, None, length=stage3_epochs)
        # concatenate log-likelihoods from all stages
        lls = jnp.concatenate([lls1, lls2, lls3])
        return final_params, lls.ravel()

    @property
    def latent_dim(self) -> int:
        return self.model.lat_man.obs_man.data_dim

    @property
    def n_clusters(self) -> int:
        return self.model.lat_man.lat_man.dim + 1

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
        final_params, train_lls = self.fit(params, data.train_images)  # type: ignore

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
