"""Run MNIST benchmarking experiments."""

from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from torchvision import datasets, transforms  # type: ignore

from goal.geometry import Diagonal
from goal.models import HierarchicalMixtureOfGaussians

from ...shared import ExamplePaths, initialize_jax, initialize_paths, save_results
from .common import MNISTData
from .models import PCAGMM, HMoG


def create_parser() -> ArgumentParser:
    """Create argument parser for MNIST benchmarks."""
    parser = ArgumentParser(description="Run MNIST benchmarking experiments")

    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        choices=["pcagmm", "hmog"],
        required=True,
        help="Model architecture to evaluate",
    )
    model_group.add_argument(
        "--latent-dim",
        type=int,
        required=True,
        help="Dimensionality of latent space",
    )
    model_group.add_argument(
        "--n-clusters",
        type=int,
        required=True,
        help="Number of clusters/mixture components",
    )

    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    training_group.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to run computations on",
    )

    return parser


def load_mnist(paths: ExamplePaths) -> MNISTData:
    """Load and preprocess MNIST dataset."""

    def transform_tensor(x: NDArray[np.uint8]) -> NDArray[np.float32]:
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


def create_model(
    model_name: str, latent_dim: int, n_clusters: int, data_dim: int, n_epochs: int
):
    """Create model instance based on configuration."""
    if model_name == "pcagmm":
        return PCAGMM(latent_dim, n_clusters)
    if model_name == "hmog":
        return HMoG(
            n_epochs=n_epochs,
            model=HierarchicalMixtureOfGaussians(
                obs_dim=data_dim,
                obs_rep=Diagonal,
                lat_dim=latent_dim,
                n_components=n_clusters,
            ),
        )
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    # Save results with configuration in filename
    experiment = (
        f"model_{args.model}_"
        f"ld{args.latent_dim}_"
        f"nc{args.n_clusters}_"
        f"e{args.n_epochs}"
    )

    initialize_jax(device=args.device)
    paths = initialize_paths(__file__, experiment)
    key = jax.random.PRNGKey(0)

    data = load_mnist(paths)
    model = create_model(
        args.model,
        args.latent_dim,
        args.n_clusters,
        data.train_images.shape[1],
        args.n_epochs,
    )

    results = model.evaluate(key, data)

    save_results(results, paths)


if __name__ == "__main__":
    main()
