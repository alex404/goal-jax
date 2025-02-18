"""Linear and affine transformations between manifolds. The implementation focuses on:

- Type safety through generics,
- Efficient matrix representations, and
- Mathematical correctness.


#### Class Hierarchy

![Class Hierarchy](linear.svg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Self, override

from jax import Array

from .base import Coordinates, Dual, Manifold, Point
from .combinators import Pair, Replicated
from .matrix import MatrixRep, Square
from .subspace import Subspace

### Linear Maps ###


@dataclass(frozen=True)
class LinearMap[Rep: MatrixRep, Domain: Manifold, Codomain: Manifold](Manifold):
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

    # Fields

    rep: Rep
    dom_man: Domain
    cod_man: Codomain

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return self.rep.num_params(self.shape)

    # Methods

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the linear maps."""
        return (self.cod_man.dim, self.dom_man.dim)

    @property
    def trn_man(self) -> LinearMap[Rep, Codomain, Domain]:
        """Manifold of transposed linear maps."""
        return LinearMap(self.rep, self.cod_man, self.dom_man)

    @property
    def row_man(self) -> Replicated[Domain]:
        """Returns the manifold of row vectors as points in the domain."""
        return Replicated(self.dom_man, self.cod_man.dim)

    @property
    def col_man(self) -> Replicated[Codomain]:
        """Returns the manifold of column vectors as points in the codomain."""
        return Replicated(self.cod_man, self.dom_man.dim)

    def __call__[C: Coordinates](
        self,
        f: Point[C, Self],
        p: Point[Dual[C], Domain],
    ) -> Point[C, Codomain]:
        """Apply the linear map to transform a point."""
        return self.cod_man.point(self.rep.matvec(self.shape, f.array, p.array))

    def from_dense[C: Coordinates](self, matrix: Array) -> Point[C, Self]:
        """Create point from dense matrix."""
        return self.point(self.rep.from_dense(matrix))

    def outer_product[C: Coordinates](
        self, w: Point[C, Codomain], v: Point[C, Domain]
    ) -> Point[C, Self]:
        """Outer product of points."""
        return self.point(self.rep.outer_product(w.array, v.array))

    def transpose[C: Coordinates](
        self: LinearMap[Rep, Domain, Codomain],
        f: Point[C, LinearMap[Rep, Domain, Codomain]],
    ) -> Point[C, LinearMap[Rep, Codomain, Domain]]:
        """Transpose of the linear map."""
        return self.trn_man.point(self.rep.transpose(self.shape, f.array))

    def transpose_apply[C: Coordinates](
        self: LinearMap[Rep, Domain, Codomain],
        f: Point[C, LinearMap[Rep, Domain, Codomain]],
        p: Point[Dual[C], Codomain],
    ) -> Point[C, Domain]:
        """Apply the transpose of the linear map."""
        f_trn = self.transpose(f)
        return self.trn_man(f_trn, p)

    def to_dense[C: Coordinates](self, f: Point[C, Self]) -> Array:
        """Convert to dense matrix representation."""
        return self.rep.to_dense(self.shape, f.array)

    def to_rows[C: Coordinates](
        self, f: Point[C, Self]
    ) -> Point[C, Replicated[Domain]]:
        """Split linear map into dense row vectors."""
        matrix = self.rep.to_dense(self.shape, f.array)
        return self.row_man.point(matrix)

    def to_columns[C: Coordinates](
        self, f: Point[C, Self]
    ) -> Point[C, Replicated[Codomain]]:
        """Split linear map into dense column vectors."""
        matrix = self.rep.to_dense(self.shape, f.array)
        return self.col_man.point(matrix.T)

    def from_rows[C: Coordinates](
        self, rows: Point[C, Replicated[Domain]]
    ) -> Point[C, Self]:
        """Construct linear map from dense row vectors."""
        return self.point(self.rep.from_dense(rows.array))

    def from_columns[C: Coordinates](
        self, columns: Point[C, Replicated[Codomain]]
    ) -> Point[C, Self]:
        """Construct linear map from dense column vectors."""
        return self.point(self.rep.from_dense(columns.array.T))

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
        return self.point(self.rep.map_diagonal(self.shape, f.array, diagonal_fn))

    def embed_rep[C: Coordinates, NewRep: MatrixRep](
        self,
        f: Point[C, Self],
        target_rep: type[NewRep],
    ) -> tuple[
        LinearMap[NewRep, Domain, Codomain],
        Point[C, LinearMap[NewRep, Domain, Codomain]],
    ]:
        """Embed linear map into more complex representation."""
        target_man = LinearMap(target_rep(), self.dom_man, self.cod_man)
        params = self.rep.embed_params(self.shape, f.array, target_rep)
        return target_man, target_man.point(params)

    def project_rep[C: Coordinates, NewRep: MatrixRep](
        self,
        f: Point[C, Self],
        target_rep: type[NewRep],
    ) -> tuple[
        LinearMap[NewRep, Domain, Codomain],
        Point[C, LinearMap[NewRep, Domain, Codomain]],
    ]:
        """Project linear map to simpler representation."""
        target_man = LinearMap(target_rep(), self.dom_man, self.cod_man)
        params = self.rep.project_params(self.shape, f.array, target_rep)
        return target_man, target_man.point(params)


@dataclass(frozen=True)
class SquareMap[R: Square, M: Manifold](LinearMap[R, M, M]):
    """Square linear map with domain and codomain the same manifold.

    Args:
        rep: Matrix representation strategy
        dom_man: Source and target manifold

    """

    # Constructor

    def __init__(self, rep: R, dom_man: M):
        super().__init__(rep, dom_man, dom_man)

    # Methods

    def inverse[C: Coordinates](self, f: Point[C, Self]) -> Point[Dual[C], Self]:
        """Matrix inverse (requires square matrix)."""
        return self.point(self.rep.inverse(self.shape, f.array))

    def logdet[C: Coordinates](self, f: Point[C, Self]) -> Array:
        """Log determinant (requires square matrix)."""
        return self.rep.logdet(self.shape, f.array)

    def is_positive_definite[C: Coordinates](self, p: Point[C, Self]) -> Array:
        """Check if matrix is positive definite."""
        return self.rep.is_positive_definite(self.shape, p.array)


### Affine Maps ###


@dataclass(frozen=True)
class AffineMap[
    Rep: MatrixRep,
    Domain: Manifold,
    SubCodomain: Manifold,
    Codomain: Manifold,
](
    Pair[Codomain, LinearMap[Rep, Domain, SubCodomain]],
):
    """Affine transformation targeting a subspace of the codomain.

    Args:
        rep: Matrix representation strategy
        dom_man: Source manifold
        cod_sub: Target manifold and subspace

    """

    # Fields

    rep: Rep
    dom_man: Domain
    cod_sub: Subspace[Codomain, SubCodomain]

    # Overrides

    @property
    @override
    def fst_man(self) -> Codomain:
        return self.cod_sub.sup_man

    @property
    @override
    def snd_man(self) -> LinearMap[Rep, Domain, SubCodomain]:
        return LinearMap(self.rep, self.dom_man, self.cod_sub.sub_man)

    # Methods

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
