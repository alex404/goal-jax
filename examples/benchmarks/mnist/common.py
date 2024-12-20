"""Common definitions for MNIST benchmark models."""

from dataclasses import dataclass
from functools import partial
from typing import Literal, Protocol, TypedDict, runtime_checkable

import jax
from jax import Array


@dataclass
class MNISTData:
    """Container for MNIST dataset."""

    train_images: Array  # Shape: (n_train, 784)
    train_labels: Array  # Shape: (n_train,)
    test_images: Array  # Shape: (n_test, 784)
    test_labels: Array  # Shape: (n_test,)


@runtime_checkable
@dataclass(frozen=True)
class BaseModel[P, B](Protocol):
    """Base protocol for all unsupervised models."""

    @property
    def latent_dim(self) -> int:
        """Return the dimensionality of the latent space."""
        ...

    @property
    def n_clusters(self) -> int:
        """Return the number of clusters in the model."""
        ...

    def initialize(self, key: Array, data: Array) -> P:
        """Initialize model parameters based on data dimensions and statistics."""
        ...

    @partial(jax.jit, static_argnums=(0,))
    def fit(self, params0: P, data: Array) -> tuple[P, Array]:
        """Train model parameters, returning final parameters and training metrics."""
        ...

    def generate(self, params: P, key: Array, n_samples: int) -> Array:
        """Generate new samples using the trained model."""
        ...

    def cluster_assignments(self, params: P, data: Array) -> Array:
        """Determine cluster assignments for each sample."""
        ...

    def evaluate(self, key: Array, data: MNISTData) -> B: ...


ModelType = Literal["probabilistic", "two_stage"]


class ProbabilisticResults(TypedDict):
    model_name: str
    train_log_likelihood: list[float]
    final_train_log_likelihood: float
    final_test_log_likelihood: float
    train_accuracy: float
    test_accuracy: float
    latent_dim: int
    n_clusters: int
    n_parameters: int
    training_time: float


class TwoStageResults(TypedDict):
    model_name: str
    reconstruction_error: float  # Single value from one-shot optimization
    train_accuracy: float
    test_accuracy: float
    latent_dim: int
    n_clusters: int
    training_time: float


@runtime_checkable
@dataclass(frozen=True)
class ProbabilisticModel[P](BaseModel[P, ProbabilisticResults], Protocol):
    """Protocol for models that support likelihood computation."""

    def log_likelihood(self, params: P, data: Array) -> Array:
        """Compute per-sample log likelihood under the model."""
        ...


@runtime_checkable
@dataclass(frozen=True)
class TwoStageModel[P](BaseModel[P, TwoStageResults], Protocol):
    """Protocol for models that support two-stage training."""

    def reconstruction_error(self, params: P, data: Array) -> Array:
        """Compute per-sample reconstruction error using encode/decode."""
        ...
