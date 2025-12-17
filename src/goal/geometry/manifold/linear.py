"""This module provides a hierarchy of linear operators that transform points between manifolds, with specific matrix representations for efficient implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import jax.numpy as jnp
from jax import Array

from .base import Manifold
from .combinators import Pair
from .embedding import IdentityEmbedding, LinearComposedEmbedding, LinearEmbedding
from .matrix import MatrixRep, Square

### Linear Maps ###


@dataclass(frozen=True)
class LinearMap[Domain: Manifold, Codomain: Manifold](Manifold, ABC):
    """Abstract base class for linear transformations between manifolds.

    A linear map $L: V \\to W$ between vector spaces satisfies:

    $$L(\\alpha x + \\beta y) = \\alpha L(x) + \\beta L(y)$$

    for all $x, y \\in V$ and scalars $\\alpha, \\beta$. The map preserves vector space
    operations like addition and scalar multiplication.

    Concrete implementations include AmbientMap (explicit matrix representation) and
    BlockMap (composition of submaps).
    """

    @property
    @abstractmethod
    def dom_man(self) -> Domain:
        """The domain manifold."""

    @property
    @abstractmethod
    def cod_man(self) -> Codomain:
        """The codomain manifold."""

    @property
    @abstractmethod
    def trn_man(self) -> LinearMap[Codomain, Domain]:
        """Manifold of transposed linear maps."""

    @abstractmethod
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        """Apply the linear map to transform a point.

        Args:
            f_coords: Parameters of the linear map
            v_coords: Point in the domain to transform

        Returns:
            Transformed point in the codomain
        """

    @abstractmethod
    def transpose(self, f_coords: Array) -> Array:
        """Transpose of the linear map.

        Args:
            f_coords: Parameters of the linear map

        Returns:
            Parameters of the transposed linear map
        """

    def transpose_apply(self, f_coords: Array, w_coords: Array) -> Array:
        """Apply the transpose of the linear map.

        Args:
            f_coords: Parameters of the linear map
            w_coords: Point in the codomain

        Returns:
            Transformed point in the domain
        """
        f_trn_coords = self.transpose(f_coords)
        return self.trn_man(f_trn_coords, w_coords)

    @abstractmethod
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        """Outer product of points.

        Args:
            w_coords: Point in the codomain
            v_coords: Point in the domain

        Returns:
            Parameters of the outer product linear map
        """

    @abstractmethod
    def map_domain_embedding[NewDomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Manifold, Domain]], LinearEmbedding[Manifold, NewDomain]
        ],
    ) -> LinearMap[NewDomain, Codomain]:
        """Transform the domain embedding(s) using the given function.

        This enables operations like tensoring with a new factor by wrapping
        the domain embedding in a more complex structure.
        """

    @abstractmethod
    def map_codomain_embedding[NewCodomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Manifold, Codomain]],
            LinearEmbedding[Manifold, NewCodomain],
        ],
    ) -> LinearMap[Domain, NewCodomain]:
        """Transform the codomain embedding(s) using the given function.

        This enables operations like tensoring with a new factor by wrapping
        the codomain embedding in a more complex structure.
        """

    def prepend_embedding[NewDomain: Manifold](
        self,
        emb: LinearEmbedding[Domain, NewDomain],
    ) -> LinearMap[NewDomain, Codomain]:
        """Prepend an embedding to the domain."""
        return self.map_domain_embedding(
            lambda dom_emb: LinearComposedEmbedding(dom_emb, emb)
        )

    def append_embedding[NewCodomain: Manifold](
        self,
        emb: LinearEmbedding[Codomain, NewCodomain],
    ) -> LinearMap[Domain, NewCodomain]:
        """Append an embedding to the codomain."""
        return self.map_codomain_embedding(
            lambda cod_emb: LinearComposedEmbedding(cod_emb, emb)
        )


@dataclass(frozen=True)
class EmbeddedMap[Domain: Manifold, Codomain: Manifold](LinearMap[Domain, Codomain]):
    """EmbeddedMap represents a linear transformation with explicit embeddings.

    The embeddings manage the relationship between internal matrix dimensions
    and external manifold dimensions, allowing linear maps to operate on
    restricted subspaces of the domain and codomain.
    """

    # Fields

    rep: MatrixRep
    """The matrix representation strategy for this linear map."""

    dom_emb: LinearEmbedding[Manifold, Domain]
    """Embedding from internal domain to external domain manifold."""

    cod_emb: LinearEmbedding[Manifold, Codomain]
    """Embedding from internal codomain to external codomain manifold."""

    # Properties

    @property
    @override
    def dom_man(self) -> Domain:
        return self.dom_emb.amb_man

    @property
    @override
    def cod_man(self) -> Codomain:
        return self.cod_emb.amb_man

    @property
    @override
    def dim(self) -> int:
        return self.rep.num_params(self.matrix_shape)

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """Shape of the matrix operating on internal dimensions.

        Returns the shape $(\\dim(internal\\_codomain), \\dim(internal\\_domain))$
        of the underlying matrix in the internal coordinate system.
        """
        return (self.cod_emb.sub_man.dim, self.dom_emb.sub_man.dim)

    @property
    @override
    def trn_man(self) -> EmbeddedMap[Codomain, Domain]:
        """Manifold of transposed linear maps."""
        return EmbeddedMap(self.rep, self.cod_emb, self.dom_emb)

    # Methods

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        """Apply the linear map to transform a point.

        Args:
            f_coords: Parameters of the linear map
            v_coords: Point in the domain to transform

        Returns:
            Transformed point in the codomain
        """
        # 1. Project to internal domain
        internal_v = self.dom_emb.project(v_coords)

        # 2. Apply matrix operation on internal dimensions
        internal_result = self.rep.matvec(self.matrix_shape, f_coords, internal_v)

        # 3. Embed back to external codomain
        return self.cod_emb.embed(internal_result)

    @override
    def transpose(self, f_coords: Array) -> Array:
        """Transpose of the linear map.

        Args:
            f_coords: Parameters of the linear map

        Returns:
            Parameters of the transposed linear map
        """
        return self.rep.transpose(self.matrix_shape, f_coords)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        """Outer product of points.

        Args:
            w_coords: Point in the codomain
            v_coords: Point in the domain

        Returns:
            Parameters of the outer product linear map
        """
        # Project both to internal spaces
        internal_w = self.cod_emb.project(w_coords)
        internal_v = self.dom_emb.project(v_coords)

        # Compute outer product in internal space
        return self.rep.outer_product(internal_w, internal_v)

    @override
    def map_domain_embedding[NewDomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Manifold, Domain]], LinearEmbedding[Manifold, NewDomain]
        ],
    ) -> EmbeddedMap[NewDomain, Codomain]:
        new_dom_emb = f(self.dom_emb)
        return EmbeddedMap(self.rep, new_dom_emb, self.cod_emb)

    @override
    def map_codomain_embedding[NewCodomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Manifold, Codomain]],
            LinearEmbedding[Manifold, NewCodomain],
        ],
    ) -> EmbeddedMap[Domain, NewCodomain]:
        new_cod_emb = f(self.cod_emb)
        return EmbeddedMap(self.rep, self.dom_emb, new_cod_emb)

    @override
    def prepend_embedding[NewDomain: Manifold](
        self,
        emb: LinearEmbedding[Domain, NewDomain],
    ) -> EmbeddedMap[NewDomain, Codomain]:
        """Prepend an embedding to the domain."""
        return self.map_domain_embedding(
            lambda dom_emb: LinearComposedEmbedding(dom_emb, emb)
        )

    @override
    def append_embedding[NewCodomain: Manifold](
        self,
        emb: LinearEmbedding[Codomain, NewCodomain],
    ) -> EmbeddedMap[Domain, NewCodomain]:
        """Append an embedding to the codomain."""
        return self.map_codomain_embedding(
            lambda cod_emb: LinearComposedEmbedding(cod_emb, emb)
        )

    def from_matrix(self, matrix: Array) -> Array:
        """Create coordinates from dense 2D matrix (in internal dimensions).

        Args:
            matrix: Dense 2D matrix array with shape (internal_cod_dim, internal_dom_dim)

        Returns:
            Coordinates in this manifold's representation (1D array in storage format)
        """
        return self.rep.from_matrix(matrix)

    def to_matrix(self, f_coords: Array) -> Array:
        """Convert to dense 2D matrix representation (in internal dimensions).

        Args:
            f_coords: Coordinates of the linear map (1D array in representation storage format)

        Returns:
            Dense 2D matrix array with shape (internal_cod_dim, internal_dom_dim),
            where internal dimensions are determined by the embeddings (cod_emb.sub_man.dim, dom_emb.sub_man.dim)
        """
        return self.rep.to_matrix(self.matrix_shape, f_coords)

    def map_diagonal(
        self, f_coords: Array, diagonal_f: Callable[[Array], Array]
    ) -> Array:
        """Apply a function to the diagonal elements while preserving matrix structure.

        Args:
            f_coords: Coordinates of the linear map
            diagonal_f: Function to apply to diagonal elements

        Returns:
            Modified coordinates
        """
        return self.rep.map_diagonal(self.matrix_shape, f_coords, diagonal_f)

    def embed_rep(
        self, f_coords: Array, target_rep: MatrixRep
    ) -> tuple[EmbeddedMap[Domain, Codomain], Array]:
        """Embed linear map into more complex representation.

        Args:
            f_coords: Coordinates of the linear map
            target_rep: Target matrix representation

        Returns:
            Tuple of (new manifold, new coordinates)
        """
        target_man = EmbeddedMap(target_rep, self.dom_emb, self.cod_emb)
        coords = self.rep.embed_params(self.matrix_shape, f_coords, target_rep)
        return target_man, coords

    def project_rep(
        self, f_coords: Array, target_rep: MatrixRep
    ) -> tuple[EmbeddedMap[Domain, Codomain], Array]:
        """Project linear map to simpler representation.

        Args:
            f_coords: Coordinates of the linear map
            target_rep: Target matrix representation

        Returns:
            Tuple of (new manifold, new coordinates)
        """
        target_man = EmbeddedMap(target_rep, self.dom_emb, self.cod_emb)
        coords = self.rep.project_params(self.matrix_shape, f_coords, target_rep)
        return target_man, coords


@dataclass(frozen=True)
class BlockMap[Domain: Manifold, Codomain: Manifold](LinearMap[Domain, Codomain]):
    """BlockMap represents a linear transformation as a block diagonal composition of submaps.

    Each block is an embedded linear map, and the overall transformation is the sum of
    these block contributions. This enables efficient composition of heterogeneous linear
    maps with different matrix representations and embeddings.
    """

    # Fields

    blocks: list[LinearMap[Domain, Codomain]]
    """The embedded linear maps that compose this block map."""

    @property
    @override
    def dom_man(self) -> Domain:
        """The domain manifold."""
        return self.blocks[0].dom_man

    @property
    @override
    def cod_man(self) -> Codomain:
        """The codomain manifold."""
        return self.blocks[0].cod_man

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return sum(block.dim for block in self.blocks)

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        """Apply the block linear map to transform a point.

        Args:
            f_coords: Coordinates of the block linear map
            v_coords: Point in the domain to transform

        Returns:
            Transformed point in the codomain
        """
        result = self.cod_man.zeros()
        for block, block_coords in zip(self.blocks, self.coord_blocks(f_coords)):
            result = result + block(block_coords, v_coords)
        return result

    @property
    @override
    def trn_man(self) -> BlockMap[Codomain, Domain]:
        """Manifold of transposed linear maps."""
        transposed_blocks = [block.trn_man for block in self.blocks]
        return BlockMap(transposed_blocks)

    def coord_blocks(self, coords: Array) -> list[Array]:
        """Section coordinate array into sub-coordinates for each block.

        Args:
            coords: Coordinate array for the block map

        Returns:
            List of coordinate arrays, one for each block
        """
        sections = []
        offset = 0
        for block in self.blocks:
            dim = block.dim
            sections.append(coords[offset : offset + dim])
            offset += dim
        return sections

    @override
    def transpose(self, f_coords: Array) -> Array:
        """Transpose of the block linear map.

        Args:
            f_coords: Coordinates of the block linear map

        Returns:
            Coordinates of the transposed linear map
        """
        transposed_coords = [
            block.transpose(block_coords)
            for block, block_coords in zip(self.blocks, self.coord_blocks(f_coords))
        ]
        return jnp.concatenate(transposed_coords)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        """Outer product of points.

        Args:
            w_coords: Point in the codomain
            v_coords: Point in the domain

        Returns:
            Parameters of the outer product linear map
        """
        outer_params = [
            block.outer_product(w_coords, v_coords) for block in self.blocks
        ]
        return jnp.concatenate(outer_params)

    @override
    def map_domain_embedding[NewDomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Manifold, Domain]], LinearEmbedding[Manifold, NewDomain]
        ],
    ) -> BlockMap[NewDomain, Codomain]:
        return BlockMap([block.map_domain_embedding(f) for block in self.blocks])

    @override
    def map_codomain_embedding[NewCodomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Manifold, Codomain]],
            LinearEmbedding[Manifold, NewCodomain],
        ],
    ) -> BlockMap[Domain, NewCodomain]:
        return BlockMap([block.map_codomain_embedding(f) for block in self.blocks])


class AmbientMap[Domain: Manifold, Codomain: Manifold](EmbeddedMap[Domain, Codomain]):
    """AmbientMap represents a linear transformation with identity embeddings.

    This is a convenience class that wraps EmbeddedMap with identity embeddings,
    providing a simpler constructor for the common case where the linear map
    operates on the full domain and codomain manifolds without restriction.
    """

    def __init__(self, rep: MatrixRep, dom_man: Domain, cod_man: Codomain):
        """Create a AmbientMap with identity embeddings.

        Args:
            rep: The matrix representation strategy
            dom_man: The domain manifold
            cod_man: The codomain manifold
        """
        super().__init__(rep, IdentityEmbedding(dom_man), IdentityEmbedding(cod_man))


@dataclass(frozen=True)
class SquareMap[M: Manifold](AmbientMap[M, M]):
    """SquareMap provides specialized operations for square matrices, like determinants, inverses, and tests for positive definiteness."""

    # Constructor

    rep: Square

    def __init__(self, rep: MatrixRep, dom_man: M):
        # Check that the representation is square
        if not issubclass(type(rep), Square):
            raise TypeError("SquareMap requires a square matrix representation.")

        super().__init__(rep, dom_man, dom_man)

    # Methods

    def inverse(self, f_coords: Array) -> Array:
        """Matrix inverse (requires square matrix).

        Args:
            f_coords: Parameters of the square matrix

        Returns:
            Parameters of the inverse matrix
        """
        return self.rep.inverse(self.matrix_shape, f_coords)

    def logdet(self, f_coords: Array) -> Array:
        """Log determinant (requires square matrix).

        Args:
            f_coords: Parameters of the square matrix

        Returns:
            Log determinant scalar
        """
        return self.rep.logdet(self.matrix_shape, f_coords)

    def is_positive_definite(self, f_coords: Array) -> Array:
        """Check if matrix is positive definite.

        Args:
            v_coords: Parameters of the square matrix

        Returns:
            Boolean indicating positive definiteness
        """
        return self.rep.is_positive_definite(self.matrix_shape, f_coords)


### Affine Maps ###


@dataclass(frozen=True)
class AffineMap[
    Domain: Manifold,
    Codomain: Manifold,
](
    Pair[Codomain, LinearMap[Domain, Codomain]],
):
    """AffineMap combines a linear transformation with a translation component.

    In theory, an affine map $A: V \\to W$ has the form:

    $$A(x) = L(x) + b$$

    where $L: V \\to W$ is a linear map and $b \\in W$ is a translation vector.
    The map preserves affine combinations: for points $x_i$ and weights $\\alpha_i$ where
    $\\sum_i \\alpha_i = 1$, we have:

    $$A(\\sum_i \\alpha_i x_i) = \\sum_i \\alpha_i A(x_i)$$

    The linear map (which may be an EmbeddedMap) internally handles any embedding
    structure between internal and external spaces.
    """

    # Fields

    map_man: LinearMap[Domain, Codomain]
    """The linear transformation for this affine map."""

    dom_man: Domain
    """The domain of the affine map."""

    # Overrides

    @property
    @override
    def fst_man(self) -> Codomain:
        return self.map_man.cod_man

    @property
    @override
    def snd_man(self) -> LinearMap[Domain, Codomain]:
        return self.map_man

    # Methods

    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        """Apply the affine transformation.

        Args:
            f_coords: Parameters of the affine map (concatenation of bias and linear map)
            v_coords: Point in the domain

        Returns:
            Transformed point in the codomain
        """
        bias, linear = self.split_coords(f_coords)
        shift = self.snd_man(linear, v_coords)
        # The linear map (if EmbeddedMap) already handles embedding internally,
        # so shift is in codomain space. Just add the bias.
        return bias + shift
