"""Linear and affine transformations between manifolds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from jax import Array

from ..manifold import Coordinates, Dual, Manifold, Point, Product
from .matrix import MatrixRep, Square

### Linear Maps ###


@dataclass(frozen=True)
class LinearMap[R: MatrixRep, M: Manifold, N: Manifold](Manifold):
    """Linear map between manifolds using a specific matrix representation.

    A linear map $L: V \\to W$ between vector spaces satisfies:

    $$L(\\alpha x + \\beta y) = \\alpha L(x) + \\beta L(y)$$

    The map is stored in a specific representation (full, symmetric, etc) defined by R.

    Args:
        domain: The source/domain manifold $V$
        codomain: The target/codomain manifold $W$
        rep: The matrix representation strategy
    """

    rep: R
    domain: M
    codomain: N

    @property
    def dim(self) -> int:
        return self.rep.num_params(self.shape)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the linear maps."""
        return (self.codomain.dim, self.domain.dim)

    def __call__[C: Coordinates](
        self,
        f: Point[C, Self],
        p: Point[Dual[C], M],
    ) -> Point[C, N]:
        """Apply the linear map to transform a point."""
        return Point(self.rep.matvec(self.shape, f.params, p.params))

    def from_dense(self, matrix: Array) -> Point[Any, Self]:
        """Create point from dense matrix."""
        return Point(self.rep.from_dense(matrix))

    def outer_product[C: Coordinates](
        self, w: Point[C, N], v: Point[C, M]
    ) -> Point[C, Self]:
        """Outer product of points."""
        return Point(self.rep.outer_product(w.params, v.params))

    def transpose[C: Coordinates](
        self: LinearMap[R, M, N], f: Point[C, LinearMap[R, M, N]]
    ) -> Point[C, LinearMap[R, N, M]]:
        """Transpose of the linear map."""
        return Point(self.rep.transpose(self.shape, f.params))

    def to_dense[C: Coordinates](self, f: Point[C, Self]) -> Array:
        """Convert to dense matrix representation."""
        return self.rep.to_dense(self.shape, f.params)

    def to_columns[C: Coordinates](self, f: Point[C, Self]) -> list[Point[C, N]]:
        """Split linear map into list of column vectors as points in the codomain."""
        cols = self.rep.to_cols(self.shape, f.params)
        return [Point(col) for col in cols]

    def from_columns[C: Coordinates](
        self: LinearMap[R, M, N], cols: list[Point[C, N]]
    ) -> Point[C, LinearMap[R, M, N]]:
        """Construct linear map from list of column vectors as points."""
        col_arrays = [col.params for col in cols]
        params = self.rep.from_cols(col_arrays)
        return Point(params)


@dataclass(frozen=True)
class SquareMap[R: Square, M: Manifold](LinearMap[R, M, M]):
    """Square linear map with domain and codomain the same manifold.

    Args:
        rep: Matrix representation strategy
        domain: Source and target manifold

    """

    def __init__(self, rep: R, domain: M):
        super().__init__(rep, domain, domain)

    def inverse[C: Coordinates](self, f: Point[C, Self]) -> Point[Dual[C], Self]:
        """Matrix inverse (requires square matrix)."""
        return Point(self.rep.inverse(self.shape, f.params))

    def logdet[C: Coordinates](self, f: Point[C, Self]) -> Array:
        """Log determinant (requires square matrix)."""
        return self.rep.logdet(self.shape, f.params)


### Affine Maps ###


@dataclass(frozen=True)
class AffineMap[R: MatrixRep, M: Manifold, N: Manifold](Product[N, LinearMap[R, M, N]]):
    """Affine transformation $f(x) = A \\cdot x + b$.

    Args:
        rep: Matrix representation strategy
        domain: Source manifold
        codomain: Target manifold
    """

    def __init__(self, rep: R, domain: M, codomain: N):
        super().__init__(codomain, LinearMap(rep, domain, codomain))

    def __call__[C: Coordinates](
        self,
        f: Point[C, Self],
        p: Point[Dual[C], M],
    ) -> Point[C, N]:
        """Apply the affine transformation."""
        bias, linear = self.split_params(f)
        transformed = self.second(linear, p)
        return transformed + bias


@dataclass(frozen=True)
class AffineSubMap[R: MatrixRep, M: Manifold, N: Manifold, O: Manifold](
    Product[Product[N, O], LinearMap[R, M, N]]
):
    """Affine transformation targeting the first component of a product manifold."""

    def __init__(self, rep: R, domain: M, codomain: Product[N, O]):
        super().__init__(codomain, LinearMap(rep, domain, codomain.first))

    def __call__[C: Coordinates](
        self,
        f: Point[C, Self],
        p: Point[Dual[C], M],
    ) -> Point[C, Product[N, O]]:
        """Apply the affine transformation."""
        bias, linear = self.split_params(f)
        transformed = self.second(linear, p)
        nbias, obias = self.first.split_params(bias)
        return self.first.join_params(nbias + transformed, obias)
