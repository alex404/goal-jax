import jax.numpy as jnp

from goal.exponential_family import Mean
from goal.linear import (
    Diagonal,
    LinearMap,
    Manifold,
    Point,
    PositiveDefiniteMap,
    Scale,
    TypeMap,
)
from goal.manifold import Dual


# First let's define a simple test manifold
class TestManifold(Manifold):
    @property
    def dimension(self) -> int:
        return 2


def test_map_relationships() -> None:
    manifold = TestManifold()

    # Test 1: Basic outer product with explicit coordinate types
    scale_map = PositiveDefiniteMap(manifold, manifold, Scale())
    x: Point[Mean, TestManifold] = Point(jnp.array([1.0, 2.0]))

    # This is where we hit our typing question
    outer_prod: Point[
        TypeMap[Mean, Dual[Mean]], LinearMap[Scale, TestManifold, TestManifold]
    ] = scale_map.outer_product(x, x)
    print(f"Type of outer product: {type(outer_prod)}")

    # Test 2: Try with different matrix types
    diag_map = PositiveDefiniteMap(manifold, manifold, Diagonal())
    outer_diag: Point[
        TypeMap[Mean, Dual[Mean]], LinearMap[Diagonal, TestManifold, TestManifold]
    ] = diag_map.outer_product(x, x)
    print(f"Parameters of diagonal outer product: {outer_diag.params}")


if __name__ == "__main__":
    test_map_relationships()
