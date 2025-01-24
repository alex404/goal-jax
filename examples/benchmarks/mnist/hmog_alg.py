"""Base class for HMoG implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from time import time
from typing import Any, override

import jax
import jax.numpy as jnp
from jax import Array, debug

from goal.geometry import Natural, Optimizer, OptState, Point, PositiveDefinite
from goal.models import (
    DifferentiableHMoG,
    DifferentiableMixture,
    FullNormal,
    LinearGaussianModel,
    Normal,
)

from .common import evaluate_clustering
from .types import MNISTData, ProbabilisticResults

OBS_JITTER = 1e-6
OBS_MIN_VAR = 1e-5
LAT_JITTER = 1e-5
LAT_MIN_VAR = 1e-3

### ABC ###


@dataclass(frozen=True)
class HMoGBase[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](ABC):
    """Base class for HMoG implementations."""

    # Fields

    model: DifferentiableHMoG[ObsRep, LatRep]
    n_epochs: int

    # Properties

    @property
    def latent_dim(self) -> int:
        return self.model.lat_man.obs_man.data_dim

    @property
    def n_clusters(self) -> int:
        return self.model.lat_man.lat_man.dim + 1

    # Methods

    def initialize(
        self, key: Array, data: Array
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
        """Initialize model parameters."""
        keys = jax.random.split(key, 3)
        key_cat, key_comp, key_int = keys

        obs_means = self.model.obs_man.average_sufficient_statistic(data)
        obs_means = self.model.obs_man.regularize_covariance(
            obs_means, OBS_JITTER, OBS_MIN_VAR
        )
        obs_params = self.model.obs_man.to_natural(obs_means)

        with self.model.upr_hrm as uh:
            cat_params = uh.lat_man.initialize(key_cat)
            key_comps = jax.random.split(key_comp, self.n_clusters)
            components = [uh.obs_man.initialize(key_compi) for key_compi in key_comps]
            mix_params = uh.join_natural_mixture(components, cat_params)

        int_noise = 0.1 * jax.random.normal(key_int, self.model.int_man.shape)
        lkl_params = self.model.lkl_man.join_params(
            obs_params, Point(self.model.int_man.rep.from_dense(int_noise))
        )
        return self.model.join_conjugated(lkl_params, mix_params)

    @partial(jax.jit, static_argnums=(0,))
    @abstractmethod
    def fit(
        self,
        params0: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
        data: Array,
    ) -> tuple[Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], Array]:
        """Train model through all three stages."""
        ...

    def log_likelihood(
        self, params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], data: Array
    ) -> Array:
        return self.model.average_log_observable_density(params, data)

    def generate(
        self,
        params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
        key: Array,
        n_samples: int,
    ) -> Array:
        return self.model.observable_sample(key, params, n_samples)

    def cluster_assignments(
        self, params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], data: Array
    ) -> Array:
        def data_point_cluster(x: Array) -> Array:
            with self.model as m:
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
        start_time = time()
        params = self.initialize(key, data.train_images)
        final_params, train_lls = self.fit(params, data.train_images)

        train_clusters = self.cluster_assignments(final_params, data.train_images)
        test_clusters = self.cluster_assignments(final_params, data.test_images)

        return ProbabilisticResults(
            model_name=self.__class__.__name__,
            train_log_likelihood=train_lls.tolist(),
            final_train_log_likelihood=float(train_lls[-1]),
            final_test_log_likelihood=float(
                self.log_likelihood(final_params, data.test_images)
            ),
            train_accuracy=float(
                evaluate_clustering(train_clusters, data.train_labels)
            ),
            test_accuracy=float(evaluate_clustering(test_clusters, data.test_labels)),
            latent_dim=self.latent_dim,
            n_clusters=self.n_clusters,
            n_parameters=self.model.dim,
            training_time=time() - start_time,
        )


@dataclass(frozen=True)
class GradientDescentHMoG[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    HMoGBase[ObsRep, LatRep]
):
    """Original gradient-based HMoG implementation."""

    stage1_epochs: int
    stage2_epochs: int
    stage2_learning_rate: float
    stage3_learning_rate: float

    @property
    def stage3_epochs(self) -> int:
        return self.n_epochs - self.stage1_epochs - self.stage2_epochs

    @partial(jax.jit, static_argnums=(0,))
    @override
    def fit(
        self,
        params0: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
        sample: Array,
    ) -> tuple[Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], Array]:
        """Three-stage training process combining EM and gradient descent."""
        lkl_params0, mix_params0 = self.model.split_conjugated(params0)

        # Stage 1: EM for LinearGaussianModel
        with self.model.lwr_hrm as lh:
            z = lh.lat_man.standard_normal()
            lgm_params0 = lh.join_conjugated(lkl_params0, lh.lat_man.to_natural(z))

            def stage1_step(
                params: Point[Natural, LinearGaussianModel[ObsRep]], _: Any
            ) -> tuple[Point[Natural, LinearGaussianModel[ObsRep]], Array]:
                ll = lh.average_log_observable_density(params, sample)
                means = lh.expectation_step(params, sample)
                obs_means, int_means, lat_means = lh.split_params(means)
                obs_means = lh.obs_man.regularize_covariance(
                    obs_means, OBS_JITTER, OBS_MIN_VAR
                )
                means = lh.join_params(obs_means, int_means, lat_means)
                params1 = lh.to_natural(means)
                lkl_params = lh.likelihood_function(params1)
                z = lh.lat_man.to_natural(lh.lat_man.standard_normal())
                next_params = lh.join_conjugated(lkl_params, z)
                debug.print("Stage 1 LL: {}", ll)
                return next_params, ll

            lgm_params1, lls1 = jax.lax.scan(
                stage1_step, lgm_params0, None, length=self.stage1_epochs
            )
            lkl_params1 = lh.likelihood_function(lgm_params1)

        # Stage 2: Gradient descent for mixture components
        stage2_optimizer: Optimizer[
            Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]
        ] = Optimizer.adam(learning_rate=self.stage2_learning_rate)
        stage2_opt_state = stage2_optimizer.init(mix_params0)

        def stage2_loss(
            params: Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
        ) -> Array:
            hmog_params = self.model.join_conjugated(lkl_params1, params)
            return -self.model.average_log_observable_density(hmog_params, sample)

        def stage2_step(
            opt_state_and_params: tuple[
                OptState,
                Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
            ],
            _: Any,
        ) -> tuple[
            tuple[
                OptState,
                Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
            ],
            Array,
        ]:
            opt_state, params = opt_state_and_params
            ll, grads = self.model.upr_hrm.value_and_grad(stage2_loss, params)
            opt_state, params = stage2_optimizer.update(opt_state, grads, params)
            debug.print("Stage 2 LL: {}", -ll)
            return (opt_state, params), -ll

        (_, mix_params1), lls2 = jax.lax.scan(
            stage2_step,
            (stage2_opt_state, mix_params0),
            None,
            length=self.stage2_epochs,
        )

        # Stage 3: Gradient descent for full model
        params1 = self.model.join_conjugated(lkl_params1, mix_params1)
        stage3_optimizer: Optimizer[Natural, DifferentiableHMoG[ObsRep, LatRep]] = (
            Optimizer.adam(learning_rate=self.stage3_learning_rate)
        )
        stage3_opt_state = stage3_optimizer.init(params1)

        def stage3_loss(
            params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
        ) -> Array:
            return -self.model.average_log_observable_density(params, sample)

        def stage3_step(
            opt_state_and_params: tuple[
                OptState,
                Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
            ],
            _: Any,
        ) -> tuple[
            tuple[
                OptState,
                Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
            ],
            Array,
        ]:
            opt_state, params = opt_state_and_params
            ll, grads = self.model.value_and_grad(stage3_loss, params)
            opt_state, params = stage3_optimizer.update(opt_state, grads, params)
            debug.print("Stage 3 LL: {}", -ll)
            return (opt_state, params), -ll

        (_, final_params), lls3 = jax.lax.scan(
            stage3_step, (stage3_opt_state, params1), None, length=self.stage3_epochs
        )

        lls = jnp.concatenate([lls1, lls2, lls3])
        return final_params, lls.ravel()
