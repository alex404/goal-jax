"""This module provides a hierarchy of linear operators that transform points between manifolds, with specific matrix representations for efficient implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self, override

import jax.numpy as jnp
from jax import Array

from .base import Coordinates, Dual, Manifold, Point, expand_dual
from .combinators import Pair, Replicated
from .embedding import LinearEmbedding
from .matrix import MatrixRep, Square

### Linear Maps ###


@dataclass(frozen=True)
class LinearMap[Domain: Manifold, Codomain: Manifold](Manifold, ABC):
    """Abstract base class for linear transformations between manifolds.

    A linear map $L: V \\to W$ between vector spaces satisfies:

    $$L(\\alpha x + \\beta y) = \\alpha L(x) + \\beta L(y)$$

    for all $x, y \\in V$ and scalars $\\alpha, \\beta$. The map preserves vector space
    operations like addition and scalar multiplication.

    Concrete implementations include RectangularMap (explicit matrix representation) and
    BlockMap (composition of submaps).
    """

    @property
    @abstractmethod
    def dom_man(self) -> Domain:
        """The domain manifold."""
        pass

    @property
    @abstractmethod
    def cod_man(self) -> Codomain:
        """The codomain manifold."""
        pass

    @property
    @abstractmethod
    def trn_man(self) -> LinearMap[Codomain, Domain]:
        """Manifold of transposed linear maps."""
        pass

    @abstractmethod
    def __call__[C: Coordinates](
        self,
        f: Point[C, Self],
        p: Point[Dual[C], Domain],
    ) -> Point[C, Codomain]:
        """Apply the linear map to transform a point."""
        pass

    @abstractmethod
    def transpose[C: Coordinates](
        self,
        f: Point[C, Self],
    ) -> Point[C, LinearMap[Codomain, Domain]]:
        """Transpose of the linear map."""
        pass

    @abstractmethod
    def outer_product[C: Coordinates](
        self, w: Point[C, Codomain], v: Point[C, Domain]
    ) -> Point[C, Self]:
        """Outer product of points."""
        pass


@dataclass(frozen=True)
class BlockMap[Domain: Manifold, Codomain: Manifold](LinearMap[Domain, Codomain]):
    """BlockMap represents a linear transformation as a block diagonal composition of submaps.

    Each block is a linear map from a subdomain to a subcodomain, and the overall
    transformation is the sum of these block contributions. This enables efficient
    composition of heterogeneous linear maps.
    """

    # Fields

    blocks: list[
        tuple[
            LinearEmbedding[Manifold, Codomain],
            MatrixRep,
            LinearEmbedding[Manifold, Domain],
        ]
    ]
    """The matrix representations of the submaps."""

    @property
    @override
    def dom_man(self) -> Domain:
        """The domain manifold."""
        return self.blocks[0][2].amb_man

    @property
    @override
    def cod_man(self) -> Codomain:
        """The codomain manifold."""
        return self.blocks[0][0].amb_man

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return sum(
            rep.num_params((cod_emb.sub_man.dim, dom_emb.sub_man.dim))
            for cod_emb, rep, dom_emb in self.blocks
        )

    @override
    def __call__[C: Coordinates](
        self,
        f: Point[C, Self],
        p: Point[Dual[C], Domain],
    ) -> Point[C, Codomain]:
        result: Point[C, Codomain] = self.cod_man.zeros()
        param_offset = 0

        for cod_emb, rep, dom_emb in self.blocks:
            block_param_size = rep.num_params(
                (cod_emb.sub_man.dim, dom_emb.sub_man.dim)
            )
            sub_map = RectangularMap(rep, dom_emb.sub_man, cod_emb.sub_man)
            sub_params: Point[C, RectangularMap[Manifold, Manifold]] = sub_map.point(
                f.array[param_offset : param_offset + block_param_size]
            )

            block_dom_point = dom_emb.project(p)
            block_output = sub_map(sub_params, block_dom_point)

            result += cod_emb.embed(block_output)
            param_offset += block_param_size

        return result

    @property
    @override
    def trn_man(self) -> BlockMap[Codomain, Domain]:
        """Manifold of transposed linear maps."""
        transposed_blocks = [
            (dom_emb, rep, cod_emb) for cod_emb, rep, dom_emb in self.blocks
        ]
        return BlockMap(transposed_blocks)

    @override
    def transpose[C: Coordinates](
        self,
        f: Point[C, Self],
    ) -> Point[C, BlockMap[Codomain, Domain]]:
        """Transpose of the block linear map."""
        transposed_params = []
        param_offset = 0
        for cod_emb, rep, dom_emb in self.blocks:
            block_param_size = rep.num_params(
                (cod_emb.sub_man.dim, dom_emb.sub_man.dim)
            )
            sub_map = RectangularMap(rep, dom_emb.sub_man, cod_emb.sub_man)
            sub_params: Point[C, RectangularMap[Manifold, Manifold]] = sub_map.point(
                f.array[param_offset : param_offset + block_param_size]
            )
            sub_trn = sub_map.transpose(sub_params)
            transposed_params.append(sub_trn.array)
            param_offset += block_param_size

        transposed_array = jnp.concatenate(transposed_params)
        return self.trn_man.point(transposed_array)

    @override
    def outer_product[C: Coordinates](
        self, w: Point[C, Codomain], v: Point[C, Domain]
    ) -> Point[C, Self]:
        """Outer product of points."""
        outer_params = []
        for cod_emb, rep, dom_emb in self.blocks:
            block_w = cod_emb.project(expand_dual(w))
            block_v = dom_emb.project(expand_dual(v))
            sub_map = RectangularMap(rep, dom_emb.sub_man, cod_emb.sub_man)
            block_outer = sub_map.outer_product(block_w, block_v)
            outer_params.append(block_outer.array)
        outer_array = jnp.concatenate(outer_params)
        return self.point(outer_array)


@dataclass(frozen=True)
class RectangularMap[Domain: Manifold, Codomain: Manifold](LinearMap[Domain, Codomain]):
    """RectangularMap represents a linear transformation with an explicit matrix representation.

    The representation strategy (Rep) determines how matrix data is stored and manipulated,
    enabling efficient operations for special matrix structures like diagonal or symmetric matrices.
    """

    # Fields

    rep: MatrixRep
    """The matrix representation strategy for this linear map."""

    _dom_man: Domain
    """The domain of the linear map."""

    _cod_man: Codomain
    """The codomain of the linear map."""

    # Overrides

    @property
    @override
    def dom_man(self) -> Domain:
        return self._dom_man

    @property
    @override
    def cod_man(self) -> Codomain:
        return self._cod_man

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
    @override
    def trn_man(self) -> RectangularMap[Codomain, Domain]:
        """Manifold of transposed linear maps."""
        return RectangularMap(self.rep, self.cod_man, self.dom_man)

    @property
    def row_man(self) -> Replicated[Domain]:
        """The manifold of row vectors."""
        return Replicated(self.dom_man, self.cod_man.dim)

    @property
    def col_man(self) -> Replicated[Codomain]:
        """The manifold of column vectors."""
        return Replicated(self.cod_man, self.dom_man.dim)

    @override
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

    @override
    def outer_product[C: Coordinates](
        self, w: Point[C, Codomain], v: Point[C, Domain]
    ) -> Point[C, Self]:
        """Outer product of points."""
        return self.point(self.rep.outer_product(w.array, v.array))

    @override
    def transpose[C: Coordinates](
        self: RectangularMap[Domain, Codomain],
        f: Point[C, RectangularMap[Domain, Codomain]],
    ) -> Point[C, RectangularMap[Codomain, Domain]]:
        """Transpose of the linear map."""
        return self.trn_man.point(self.rep.transpose(self.matrix_shape, f.array))

    def transpose_apply[C: Coordinates](
        self: RectangularMap[Domain, Codomain],
        f: Point[C, RectangularMap[Domain, Codomain]],
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

    def embed_rep[C: Coordinates](
        self,
        f: Point[C, Self],
        target_rep: MatrixRep,
    ) -> tuple[
        RectangularMap[Domain, Codomain],
        Point[C, RectangularMap[Domain, Codomain]],
    ]:
        """Embed linear map into more complex representation."""
        target_man = RectangularMap(target_rep, self.dom_man, self.cod_man)
        params = self.rep.embed_params(self.matrix_shape, f.array, target_rep)
        return target_man, target_man.point(params)

    def project_rep[C: Coordinates](
        self,
        f: Point[C, Self],
        target_rep: MatrixRep,
    ) -> tuple[
        RectangularMap[Domain, Codomain],
        Point[C, RectangularMap[Domain, Codomain]],
    ]:
        """Project linear map to simpler representation."""
        target_man = RectangularMap(target_rep, self.dom_man, self.cod_man)
        params = self.rep.project_params(self.matrix_shape, f.array, target_rep)
        return target_man, target_man.point(params)


@dataclass(frozen=True)
class SquareMap[M: Manifold](RectangularMap[M, M]):
    """SquareMap provides specialized operations for square matrices, like determinants, inverses, and tests for positive definiteness."""

    # Constructor

    rep: Square

    def __init__(self, rep: MatrixRep, dom_man: M):
        # Check that the representation is square
        if not issubclass(type(rep), Square):
            raise TypeError("SquareMap requires a square matrix representation.")

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
    Domain: Manifold,
    SubCodomain: Manifold,
    Codomain: Manifold,
](
    Pair[Codomain, LinearMap[Domain, SubCodomain]],
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

    map_man: LinearMap[Domain, SubCodomain]
    """The linear transformation for this affine map."""

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
    def snd_man(self) -> LinearMap[Domain, SubCodomain]:
        return self.map_man

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
