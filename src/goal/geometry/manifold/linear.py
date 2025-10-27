"""This module provides a hierarchy of linear operators that transform points between manifolds, with specific matrix representations for efficient implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Self, override

from jax import Array

from .base import Coordinates, Dual, Manifold, Point
from .combinators import Pair, Replicated
from .embedding import LinearEmbedding
from .matrix import MatrixRep, Square

### Linear Maps ###


@dataclass(frozen=True)
class LinearMap[Rep: MatrixRep, Domain: Manifold, Codomain: Manifold](Manifold):
    """LinearMap provides a type-safe way to represent and compute linear transformations between manifolds. The representation strategy (Rep) determines how matrix data is stored and manipulated, enabling efficient operations for special matrix structures like diagonal or symmetric matrices.

    In theory, a linear map $L: V \\to W$ between vector spaces satisfies:

    $$L(\\alpha x + \\beta y) = \\alpha L(x) + \\beta L(y)$$

    for all $x, y \\in V$ and scalars $\\alpha, \\beta$. The map preserves vector space operations like addition and scalar multiplication.

    """

    # Fields

    rep: Rep
    """The matrix representation for this linear map."""

    dom_man: Domain
    """The domain of the linear map."""

    cod_man: Codomain
    """The codomain of the linear map."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return self.rep.num_params(self.matrix_shape)

    # Methods

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """Mathematical dimensions of the linear map $L: V \\to W$.

        Returns the shape $(\\dim(W), \\dim(V))$ of the underlying matrix, which is
        invariant across all matrix representations. This is distinct from the storage
        shape of the Point array, which is always flat.

        In practice, the matrix representation determines how the parameter array is
        stored and interpreted:

        - A scale matrix $L: \\mathbb{R}^3 \\to \\mathbb{R}^3$ has `matrix_shape = (3, 3)`
          but stores only 1 parameter (the scaling factor)
        - A diagonal matrix $L: \\mathbb{R}^3 \\to \\mathbb{R}^3$ with the same shape stores 3 parameters
        - A general rectangular matrix $L: \\mathbb{R}^2 \\to \\mathbb{R}^3$ with `matrix_shape = (3, 2)`
          stores $3 \\times 2 = 6$ parameters

        The matrix representation methods use `matrix_shape` as metadata to interpret
        the flat parameter array correctly, while `coordinates_shape` describes how
        the array is physically stored.
        """
        return (self.cod_man.dim, self.dom_man.dim)

    @property
    def trn_man(self) -> LinearMap[Rep, Codomain, Domain]:
        """Manifold of transposed linear maps."""
        return LinearMap(self.rep, self.cod_man, self.dom_man)

    @property
    def row_man(self) -> Replicated[Domain]:
        """The manifold of row vectors."""
        return Replicated(self.dom_man, self.cod_man.dim)

    @property
    def col_man(self) -> Replicated[Codomain]:
        """The manifold of column vectors."""
        return Replicated(self.cod_man, self.dom_man.dim)

    def __call__[C: Coordinates](
        self,
        f: Point[C, Self],
        p: Point[Dual[C], Domain],
    ) -> Point[C, Codomain]:
        """Apply the linear map to transform a point."""
        return self.cod_man.point(self.rep.matvec(self.matrix_shape, f.array, p.array))

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
        return self.trn_man.point(self.rep.transpose(self.matrix_shape, f.array))

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
        return self.rep.to_dense(self.matrix_shape, f.array)

    def to_rows[C: Coordinates](
        self, f: Point[C, Self]
    ) -> Point[C, Replicated[Domain]]:
        """Split linear map into dense row vectors."""
        matrix = self.rep.to_dense(self.matrix_shape, f.array)
        return self.row_man.point(matrix)

    def to_columns[C: Coordinates](
        self, f: Point[C, Self]
    ) -> Point[C, Replicated[Codomain]]:
        """Split linear map into dense column vectors."""
        matrix = self.rep.to_dense(self.matrix_shape, f.array)
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
        """Apply a function to the diagonal elements while preserving matrix structure."""
        return self.point(
            self.rep.map_diagonal(self.matrix_shape, f.array, diagonal_fn)
        )

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
        params = self.rep.embed_params(self.matrix_shape, f.array, target_rep)
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
        params = self.rep.project_params(self.matrix_shape, f.array, target_rep)
        return target_man, target_man.point(params)


@dataclass(frozen=True)
class SquareMap[R: Square, M: Manifold](LinearMap[R, M, M]):
    """SquareMap provides specialized operations for square matrices, like determinants, inverses, and tests for positive definiteness."""

    # Constructor

    def __init__(self, rep: R, dom_man: M):
        super().__init__(rep, dom_man, dom_man)

    # Methods

    def inverse[C: Coordinates](self, f: Point[C, Self]) -> Point[Dual[C], Self]:
        """Matrix inverse (requires square matrix)."""
        return self.point(self.rep.inverse(self.matrix_shape, f.array))

    def logdet[C: Coordinates](self, f: Point[C, Self]) -> Array:
        """Log determinant (requires square matrix)."""
        return self.rep.logdet(self.matrix_shape, f.array)

    def is_positive_definite[C: Coordinates](self, p: Point[C, Self]) -> Array:
        """Check if matrix is positive definite."""
        return self.rep.is_positive_definite(self.matrix_shape, p.array)


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
    """AffineMap combines a linear transformation with a translation component. The codomain is represented as an :class:`~geometry.manifold.embedding.Embedding` to allow transformations that translate only a subset of the codomain.

    In theory, an affine map $A: V \\to W$ has the form:

    $$A(x) = L(x) + b$$

    where $L: V \\to W$ is a linear map and $b \\in W$ is a translation vector.
    The map preserves affine combinations: for points $x_i$ and weights $\\alpha_i$ where
    $\\sum_i \\alpha_i = 1$, we have:

    $$A(\\sum_i \\alpha_i x_i) = \\sum_i \\alpha_i A(x_i)$$


    """

    # Fields

    rep: Rep
    """The matrix representation for this affine map."""

    dom_man: Domain
    """The domain of the affine map."""

    cod_emb: LinearEmbedding[SubCodomain, Codomain]
    """The submanifold of the codomain targetted by the affine map."""

    # Overrides

    @property
    @override
    def fst_man(self) -> Codomain:
        return self.cod_emb.amb_man

    @property
    @override
    def snd_man(self) -> LinearMap[Rep, Domain, SubCodomain]:
        return LinearMap(self.rep, self.dom_man, self.cod_emb.sub_man)

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
        return self.cod_emb.translate(bias, subshift)
