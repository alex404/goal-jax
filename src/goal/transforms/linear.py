"""Linear and affine transformations between manifolds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold import Coordinates, Dual, Manifold, Point
from .matrix import MatrixRep, Square

### Maps ###


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AffineMap[R: MatrixRep, M: Manifold, N: Manifold](Manifold):
    """Affine transformation $f(x) = Ax + b$.

    Args:
        rep: Matrix representation strategy
        domain: Source manifold
        codomain: Target manifold
        bias: Translation vector
    """

    rep: R
    domain: M
    codomain: N
    bias: Array

    @property
    def dimension(self) -> int:
        """Total dimension includes linear map + bias parameters."""
        return self.rep.num_params(self.shape) + self.codomain.dimension

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the linear component."""
        return (self.codomain.dimension, self.domain.dimension)

    def split_params[C: Coordinates](
        self, p: Point[C, Self]
    ) -> tuple[Point[C, N], Point[C, LinearMap[R, M, N]]]:
        """Split parameters into bias and linear components."""
        bias_dim = self.codomain.dimension
        return Point(p.params[:bias_dim]), Point(p.params[bias_dim:])

    def join_params[C: Coordinates](
        self,
        bias: Point[C, N],
        linear: Point[C, LinearMap[R, M, N]],
    ) -> Point[C, Self]:
        """Join bias and linear parameters."""
        return Point(jnp.concatenate([bias.params, linear.params]))

    def __call__[C: Coordinates](
        self,
        f: Point[C, Self],
        p: Point[Dual[C], M],
    ) -> Point[C, N]:
        """Apply the affine transformation."""
        bias, linear = self.split_params(f)
        transformed = self.rep.matvec(linear.params, p.params, self.shape)
        return Point(transformed + bias.params + self.bias)


@jax.tree_util.register_dataclass
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
    def dimension(self) -> int:
        return self.rep.num_params(self.shape)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the linear maps."""
        return (self.codomain.dimension, self.domain.dimension)

    def __call__[C: Coordinates](
        self,
        f: Point[C, Self],
        p: Point[Dual[C], M],
    ) -> Point[C, N]:
        """Apply the linear map to transform a point."""
        return Point(self.rep.matvec(f.params, p.params, self.shape))

    def from_dense(self, matrix: Array) -> Point[Any, Self]:
        """Create point from dense matrix."""
        return Point(self.rep.from_dense(matrix))

    def outer_product[C: Coordinates](
        self, v: Point[C, M], w: Point[C, N]
    ) -> Point[C, Self]:
        """Outer product of points."""
        return Point(self.rep.outer_product(v.params, w.params))

    def transpose[C: Coordinates](
        self: LinearMap[R, M, N], f: Point[C, LinearMap[R, M, N]]
    ) -> Point[C, LinearMap[R, N, M]]:
        """Transpose of the linear map."""
        return Point(self.rep.transpose(f.params, self.shape))

    def to_dense[C: Coordinates](self, f: Point[C, Self]) -> Array:
        """Convert to dense matrix representation."""
        return self.rep.to_dense(f.params, self.shape)

    def to_columns[C: Coordinates](self, f: Point[C, Self]) -> list[Point[C, N]]:
        """Split linear map into list of column vectors as points in the codomain."""
        cols = self.rep.to_cols(f.params, self.shape)
        return [Point(col) for col in cols]

    def from_columns[C: Coordinates](
        self: LinearMap[R, M, N], cols: list[Point[C, N]]
    ) -> Point[C, LinearMap[R, M, N]]:
        """Construct linear map from list of column vectors as points."""
        col_arrays = [col.params for col in cols]
        params = self.rep.from_cols(col_arrays, self.shape)
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
        return Point(self.rep.inverse(f.params, self.shape))

    def logdet[C: Coordinates](self, f: Point[C, Self]) -> Array:
        """Log determinant (requires square matrix)."""
        return self.rep.logdet(f.params, self.shape)
