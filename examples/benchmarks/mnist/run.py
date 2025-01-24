"""Run MNIST benchmarking experiments."""

from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from torchvision import datasets, transforms

from goal.geometry import Diagonal
from goal.models import differentiable_hmog

from ...shared import ExamplePaths, initialize_jax, initialize_paths, save_results
from .hmog_alg import GradientDescentHMoG, MinibatchHMoG
from .scipy_alg import PCAGMM
from .types import MNISTData


def create_parser() -> ArgumentParser:
    """Create argument parser for MNIST benchmarks."""
    parser = ArgumentParser(description="Run MNIST benchmarking experiments")

    model_group = parser.add_argument_group("Model Configuration")
    _ = model_group.add_argument(
        "--model",
        choices=["pcagmm", "hmog"],
        default="hmog",
        help="Model architecture to evaluate",
    )
    _ = model_group.add_argument(
        "--latent-dim",
        type=int,
        default=10,
        help="Dimensionality of latent space",
    )
    _ = model_group.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of clusters/mixture components",
    )

    training_group = parser.add_argument_group("Training Configuration")
    # stage 1 epochs
    _ = training_group.add_argument(
        "--stage1-epochs",
        type=int,
        default=100,
        help="Number of training epochs for stage 1",
    )
    # stage 2 epochs
    _ = training_group.add_argument(
        "--stage2-epochs",
        type=int,
        default=100,
        help="Number of training epochs for stage 2",
    )
    _ = training_group.add_argument(
        "--stage3-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    _ = training_group.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to run computations on",
    )
    # batch size, default 0 means full batch
    _ = training_group.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size for training",
    )

    # Jit disabled by default, flag for true
    _ = training_group.add_argument(
        "--jit",
        action="store_true",
        help="Enable JIT compilation for JAX functions",
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
    model_name: str,
    latent_dim: int,
    n_clusters: int,
    data_dim: int,
    stage1_epochs: int,
    stage2_epochs: int,
    n_epochs: int,
    batch_size: int,
):
    """Create model instance based on configuration."""
    if model_name == "pcagmm":
        return PCAGMM(latent_dim, n_clusters)
    if model_name == "hmog":
        if batch_size == 0:
            return GradientDescentHMoG(
                stage1_epochs=stage1_epochs,
                stage2_epochs=stage2_epochs,
                n_epochs=n_epochs,
                stage2_learning_rate=1e-3,
                stage3_learning_rate=3e-4,
                model=differentiable_hmog(
                    obs_dim=data_dim,
                    obs_rep=Diagonal,
                    lat_dim=latent_dim,
                    n_components=n_clusters,
                    lat_rep=Diagonal,
                ),
            )
        return MinibatchHMoG(
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
            n_epochs=n_epochs,
            stage2_learning_rate=1e-3,
            stage3_learning_rate=3e-4,
            model=differentiable_hmog(
                obs_dim=data_dim,
                obs_rep=Diagonal,
                lat_dim=latent_dim,
                n_components=n_clusters,
                lat_rep=Diagonal,
            ),
            stage2_batch_size=batch_size,
            stage3_batch_size=batch_size,
        )

    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    # Save results with configuration in filename
    experiment = f"model_{args.model}_ld{args.latent_dim}_nc{args.n_clusters}"

    print(f"Running experiment: {experiment}")
    print(f"with JIT: {args.jit}")
    initialize_jax(device=args.device, disable_jit=not (args.jit))
    paths = initialize_paths(__file__, experiment)
    key = jax.random.PRNGKey(0)

    n_epochs = args.stage1_epochs + args.stage2_epochs + args.stage3_epochs

    data = load_mnist(paths)
    model = create_model(
        args.model,
        args.latent_dim,
        args.n_clusters,
        data.train_images.shape[1],
        args.stage1_epochs,
        args.stage2_epochs,
        n_epochs,
        args.batch_size,
    )

    results = model.evaluate(key, data)

    save_results(results, paths)


if __name__ == "__main__":
    main()
