"""Run MNIST benchmarking experiments."""

from time import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score  # type: ignore
from torchvision import datasets, transforms  # type: ignore

from goal.geometry import Diagonal
from goal.models import HierarchicalMixtureOfGaussians

from ...shared import ExamplePaths, initialize_jax, initialize_paths, save_results
from .common import (
    MNISTData,
    ProbabilisticModel,
    ProbabilisticResults,
    TwoStageModel,
    TwoStageResults,
)
from .models import HMoGModel, PCAGMMModel


def load_mnist(paths: ExamplePaths) -> MNISTData:
    """Load and preprocess MNIST dataset.

    Uses torchvision's built-in MNIST dataset with our project's cache directory.
    The data is normalized to [0,1] and flattened to vectors of length 784.
    """

    def transform_tensor(x: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Transform image tensor to flattened array."""
        return x.reshape(-1).astype(np.float32)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(transform_tensor)]
    )

    train_dataset = datasets.MNIST(
        root=str(paths.cache_dir), train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root=str(paths.cache_dir), train=False, download=True, transform=transform
    )

    # Convert to JAX arrays directly from tensors
    train_images = jnp.array(train_dataset.data.numpy()).reshape(-1, 784) / 255.0
    train_labels = jnp.array(train_dataset.targets.numpy())
    test_images = jnp.array(test_dataset.data.numpy()).reshape(-1, 784) / 255.0
    test_labels = jnp.array(test_dataset.targets.numpy())

    return MNISTData(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
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


def evaluate_probabilistic_model(
    model: ProbabilisticModel[Any],
    key: Array,
    data: MNISTData,
    n_steps: int = 100,
) -> ProbabilisticResults:
    """Evaluate probabilistic model performance."""
    start_time = time()
    params = model.initialize(key, data.train_images)
    final_params, train_lls = model.fit(params, data.train_images, n_steps)

    train_clusters = model.cluster_assignments(final_params, data.train_images)
    test_clusters = model.cluster_assignments(final_params, data.test_images)

    return ProbabilisticResults(
        model_name=model.__class__.__name__,
        train_log_likelihood=train_lls.tolist(),
        test_log_likelihood=model.log_likelihood(
            final_params, data.test_images
        ).tolist(),
        final_train_log_likelihood=float(train_lls[-1]),
        final_test_log_likelihood=float(
            jnp.mean(model.log_likelihood(final_params, data.test_images))
        ),
        train_accuracy=float(evaluate_clustering(train_clusters, data.train_labels)),
        test_accuracy=float(evaluate_clustering(test_clusters, data.test_labels)),
        latent_dim=model.latent_dim,
        n_parameters=-1,
        training_time=time() - start_time,
    )


def evaluate_two_stage_model(
    model: TwoStageModel[Any],
    key: Array,
    data: MNISTData,
    n_steps: int = 100,
) -> TwoStageResults:
    """Evaluate two-stage model performance."""
    start_time = time()
    params = model.initialize(key, data.train_images)
    final_params, _ = model.fit(params, data.train_images, n_steps)

    train_clusters = model.cluster_assignments(final_params, data.train_images)
    test_clusters = model.cluster_assignments(final_params, data.test_images)

    return TwoStageResults(
        model_name=model.__class__.__name__,
        reconstruction_error=float(
            jnp.mean(model.reconstruction_error(final_params, data.test_images))
        ),
        train_accuracy=float(evaluate_clustering(train_clusters, data.train_labels)),
        test_accuracy=float(evaluate_clustering(test_clusters, data.test_labels)),
        latent_dim=model.latent_dim,
        n_parameters=-1,
        training_time=time() - start_time,
    )


def create_models(data_dim: int):
    """Initialize model instances for benchmarking."""
    return [
        PCAGMMModel(32, 10),
        HMoGModel(
            model=HierarchicalMixtureOfGaussians(
                obs_dim=data_dim,
                obs_rep=Diagonal,
                lat_dim=32,
                n_components=10,
            )
        ),
    ]


def main():
    initialize_jax()
    paths = initialize_paths(__file__)
    key = jax.random.PRNGKey(0)

    data = load_mnist(paths)
    models = create_models(data.train_images.shape[1])

    results = {}
    for model in models:
        key, subkey = jax.random.split(key)
        if isinstance(model, ProbabilisticModel):
            results[model.__class__.__name__] = evaluate_probabilistic_model(
                model, subkey, data
            )
        else:
            results[model.__class__.__name__] = evaluate_two_stage_model(
                model, subkey, data
            )

    save_results(results, paths)


if __name__ == "__main__":
    main()
