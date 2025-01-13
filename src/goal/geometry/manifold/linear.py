"""Linear and affine transformations between manifolds. The implementation focuses on:

- Type safety through generics,
- Efficient matrix representations, and
- Mathematical correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Self, TypeVar

from jax import Array

from .manifold import Coordinates, Dual, Manifold, Pair, Point
from .matrix import MatrixRep, Square
from .subspace import Subspace

### Linear Maps ###

Rep = TypeVar("Rep", bound="MatrixRep", covariant=True)
Domain = TypeVar("Domain", bound="Manifold", covariant=True)
Codomain = TypeVar("Codomain", bound="Manifold", contravariant=True)


@dataclass(frozen=True)
class LinearMap(Generic[Rep, Domain, Codomain], Manifold):
    """Linear map between manifolds using a specific matrix representation.

    Type Parameters:
        Rep: Matrix representation type (covariant). A more specific representation can be used where a more general one is expected. For example, a SymmetricMatrix representation can be used where a RectangularMatrix is expected because SymmetricMatrix is-a RectangularMatrix.

        Domain: Source manifold type (covariant). A map defined on a more general domain can accept points from a more specific domain. For example, a LinearMap[R, Shape, C] can accept points from Circle because Circle is-a Shape.

        Codomain: Target manifold type (contravariant). A map producing elements of a more specific type can be used where a map producing a more general type is expected. For example, a LinearMap[R, D, Circle] can be used where a LinearMap[R, D, Shape] is expected because Circle is-a Shape.

    A linear map $L: V \\to W$ between vector spaces satisfies:

    $$L(\\alpha x + \\beta y) = \\alpha L(x) + \\beta L(y)$$

    The map is stored in a specific representation (full, symmetric, etc) defined by R.

    Args:
        rep: The matrix representation strategy
        dom_man: The source/domain manifold $V$
        cod_man: The target/codomain manifold $W$
    """

    rep: Rep
    dom_man: Domain
    cod_man: Codomain

    @property
    def dim(self) -> int:
        return self.rep.num_params(self.shape)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the linear maps."""
        return (self.cod_man.dim, self.dom_man.dim)

    def __call__[C: Coordinates](
        self,
        f: Point[C, Self],
        p: Point[Dual[C], Domain],
    ) -> Point[C, Codomain]:
        """Apply the linear map to transform a point."""
        return Point(self.rep.matvec(self.shape, f.params, p.params))

    def transpose_manifold(self) -> LinearMap[Rep, Codomain, Domain]:
        """Transpose the linear map."""
        return LinearMap(self.rep, self.cod_man, self.dom_man)

    def from_dense(self, matrix: Array) -> Point[Any, Self]:
        """Create point from dense matrix."""
        return Point(self.rep.from_dense(matrix))

    def outer_product[C: Coordinates](
        self, w: Point[C, Codomain], v: Point[C, Domain]
    ) -> Point[C, Self]:
        """Outer product of points."""
        return Point(self.rep.outer_product(w.params, v.params))

    def transpose[C: Coordinates](
        self: LinearMap[Rep, Domain, Codomain],
        f: Point[C, LinearMap[Rep, Domain, Codomain]],
    ) -> Point[C, LinearMap[Rep, Codomain, Domain]]:
        """Transpose of the linear map."""
        return Point(self.rep.transpose(self.shape, f.params))

    def transpose_apply[C: Coordinates](
        self: LinearMap[Rep, Domain, Codomain],
        f: Point[C, LinearMap[Rep, Domain, Codomain]],
        p: Point[Dual[C], Codomain],
    ) -> Point[C, Domain]:
        """Apply the transpose of the linear map."""
        trn_man = self.transpose_manifold()
        f_trn = self.transpose(f)
        return trn_man(f_trn, p)

    def to_dense[C: Coordinates](self, f: Point[C, Self]) -> Array:
        """Convert to dense matrix representation."""
        return self.rep.to_dense(self.shape, f.params)

    def to_columns[C: Coordinates](self, f: Point[C, Self]) -> list[Point[C, Codomain]]:
        """Split linear map into list of column vectors as points in the codomain."""
        cols = self.rep.to_cols(self.shape, f.params)
        return [Point(col) for col in cols]

    def from_columns[C: Coordinates](
        self: LinearMap[Rep, Domain, Codomain], cols: list[Point[C, Codomain]]
    ) -> Point[C, LinearMap[Rep, Domain, Codomain]]:
        """Construct linear map from list of column vectors as points."""
        col_arrays = [col.params for col in cols]
        params = self.rep.from_cols(col_arrays)
        return Point(params)

    def map_diagonal[C: Coordinates](
        self, f: Point[C, Self], diagonal_fn: Callable[[Array], Array]
    ) -> Point[C, Self]:
        """Apply a function to the diagonal elements while preserving matrix structure.

        Args:
            f: Point in the linear map manifold
            diagonal_fn: Function to apply to diagonal elements

        Returns:
            New point with modified diagonal elements
        """
        return Point(self.rep.map_diagonal(self.shape, f.params, diagonal_fn))


@dataclass(frozen=True)
class SquareMap[R: Square, M: Manifold](LinearMap[R, M, M]):
    """Square linear map with domain and codomain the same manifold.

    Args:
        rep: Matrix representation strategy
        dom_man: Source and target manifold

    """

    def __init__(self, rep: R, dom_man: M):
        super().__init__(rep, dom_man, dom_man)

    def inverse[C: Coordinates](self, f: Point[C, Self]) -> Point[Dual[C], Self]:
        """Matrix inverse (requires square matrix)."""
        return Point(self.rep.inverse(self.shape, f.params))

    def logdet[C: Coordinates](self, f: Point[C, Self]) -> Array:
        """Log determinant (requires square matrix)."""
        return self.rep.logdet(self.shape, f.params)

    def is_positive_definite[C: Coordinates](self, p: Point[C, Self]) -> Array:
        """Check if matrix is positive definite."""
        return self.rep.is_positive_definite(self.shape, p.params)


### Affine Maps ###


SubCodomain = TypeVar("SubCodomain", bound="Manifold", contravariant=True)


@dataclass(frozen=True)
class AffineMap(
    Pair[Codomain, LinearMap[Rep, Domain, SubCodomain]],
    Generic[Rep, Domain, SubCodomain, Codomain],
):
    """Affine transformation targeting a subspace of the codomain.

    Args:
        rep: Matrix representation strategy
        dom_man: Source manifold
        cod_sub: Target manifold and subspace

    """

    cod_sub: Subspace[Codomain, SubCodomain]

    def __init__(
        self, rep: Rep, dom_man: Domain, cod_sub: Subspace[Codomain, SubCodomain]
    ):
        super().__init__(cod_sub.sup_man, LinearMap(rep, dom_man, cod_sub.sub_man))
        object.__setattr__(self, "cod_sub", cod_sub)

    def __call__[C: Coordinates](
        self,
        f: Point[C, Self],
        p: Point[Dual[C], Domain],
    ) -> Point[C, Codomain]:
        """Apply the affine transformation."""
        bias: Point[C, Codomain]
        bias, linear = self.split_params(f)
        subshift: Point[C, SubCodomain] = self.snd_man(linear, p)
        return self.cod_sub.translate(bias, subshift)
