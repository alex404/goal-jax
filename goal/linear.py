"""Linear operators with structural specializations.

This module provides a hierarchy of linear operators with increasingly specialized
structure. Each specialization maintains key matrix operations (multiplication, determinant,
inverse) while exploiting structure for efficiency:

Operator Type    | Storage   | Matmul    | Inverse/Det
----------------|-----------|-----------|-------------
Rectangular     | O(n²)     | O(n²)     | O(n³)
Symmetric       | O(n²/2)   | O(n²)     | O(n³)
Pos. Definite   | O(n²/2)   | O(n²)     | O(n³) Cholesky
Diagonal        | O(n)      | O(n)      | O(n)
Scale           | O(1)      | O(n)      | O(1)
Identity        | O(1)      | O(1)      | O(1)

Notes:
   - All operators are immutable to align with JAX's functional style
   - Specialized operators maintain their structure under inversion
   - Operations use numerically stable algorithms where possible

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jax import Array

from goal.manifold import Coordinates, Dual, Manifold, Point

### TypeVar Definitions ###

LM = TypeVar("LM", bound="LinearMap[Any, Any]")
RE = TypeVar("RE", bound="Rectangular[Any, Any]")
# SQ = TypeVar("SQ", bound="Square")
# SY = TypeVar("SY", bound="Symmetric")
# PD = TypeVar("PD", bound="PositiveDefinite")
# DI = TypeVar("DI", bound="Diagonal")
# SC = TypeVar("SC", bound="Scale")
# ID = TypeVar("ID", bound="Identity")

C = TypeVar("C", bound=Coordinates)
D = TypeVar("D", bound=Coordinates)
# M = TypeVar("M", bound="Manifold")
# N = TypeVar("N", bound="Manifold")


class TypeMap[C, D](Coordinates): ...


### Classes ###


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LinearMap[M: Manifold, N: Manifold](Manifold, ABC):
    """Manifold of linear maps between vector spaces.

    A linear map $L: V \\mapsto W$ between vector spaces satisfies $L(\\alpha x + \\beta y) = \\alpha L(x) + \\beta L(y)$.

    The manifold itself just stores the source and target manifolds and serves as a factory for creating specific linear transformations as points.
    """

    domain: N
    codomain: M

    @property
    def dimension(self) -> int:
        """Dimension of the linear map manifold itself."""
        return self.codomain.dimension * self.domain.dimension

    def __call__(self: LM, f: Point[TypeMap[C, D], LM], p: Point[D, N]) -> Point[C, M]:
        """Apply the linear map represented by `f` to transform the point `p` in the domain of `f`."""
        ...

    @abstractmethod
    def transpose(
        self: LM, f: Point[TypeMap[C, D], LM]
    ) -> Point[TypeMap[Dual[D], Dual[C]], LM]:
        """Transpose of the linear map.

        For a map $L: V \\mapsto W$, returns $L^T: W^* \\mapsto V^*$. When $V$ and $W$ are dual spaces, the coordinate systems remain unchanged, and since this is the standard scenario we consider in this library, we type `transpose` under this assumption.
        """
        ...

    # @abstractmethod
    # def outerproduct(
    #     self: LM,
    #     v: Point[C, M],  # Vector in source space
    #     w: Point[D, N],  # Vector in target space
    # ) -> Point[TypeMap[C, Dual[D]], LM]:
    #     """Outer product $v \\otimes w$ of vectors in the `domain` and `codomain`."""

    @abstractmethod
    def to_matrix(self: LM, point: Point[TypeMap[C, D], LM]) -> Array:
        """Convert a point in this manifold to its matrix representation."""
        ...

    @abstractmethod
    def from_matrix(
        self: LM, matrix: Array
    ) -> Point[TypeMap[Coordinates, Coordinates], LM]:
        """Create a point in this manifold from a matrix representation."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the linear maps in this manifold: (target_dim, source_dim)."""
        return (self.codomain.dimension, self.domain.dimension)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TransposedLinearMap[N: Manifold, M: Manifold](LinearMap[M, N], ABC): ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Rectangular[M: Manifold, N: Manifold](LinearMap[M, N]):
    """Unconstrained map between two vector spaces, represented by its full matrix form $A \\in \\mathbb R^{m \\times n}$."""

    def __call__(
        self: "Rectangular[M, N]",
        f: Point[TypeMap[C, D], "Rectangular[M, N]"],
        p: Point[D, N],
    ) -> Point[C, M]:
        """Matrix-vector product $Av$."""
        matrix = self.to_matrix(f)
        result = jnp.dot(matrix, p.params)
        return Point(result)

    def opposite_space(self: "Rectangular[M, N]") -> "Rectangular[N, M]":
        """Map in the opposite direction."""
        return Rectangular(self.codomain, self.domain)

    def transpose(
        self: RE, f: Point[TypeMap[C, D], RE]
    ) -> Point[TypeMap[Dual[D], Dual[C]], "Rectangular[N, M]"]:
        """Transpose of the linear map."""
        op = self.opposite_space()
        trans: Point[TypeMap[Dual[D], Dual[C]], RE] = op.from_matrix(
            self.to_matrix(f).T
        )
        return trans

    def to_matrix(self: "Rectangular[M, N]", point: Point[TypeMap[C, D], RE]) -> Array:
        return point.params

    def from_matrix(
        self: "Rectangular[M, N]", matrix: Array
    ) -> Point[TypeMap[Coordinates, Coordinates], "Rectangular[M, N]"]:
        return Point(matrix)

    # def transpose(self: RE) -> TransposedLinearMap[RE]:
    #     """Transpose of the linear map."""
    #     return replace(self, params=self.matrix.T.flatten(), shape=self.shape[::-1])

    #
    # @property
    # def matrix(self) -> Array:
    #     """Returns the full matrix representation $A$."""
    #     return self.params.reshape(self.shape)
    #
    # @classmethod
    # def from_matrix(cls, matrix: Array) -> "Rectangular":
    #     """Construct from a matrix."""
    #     return cls(matrix.flatten(), matrix.shape)
    #
    # @classmethod
    # def outer_product(cls, v1: Array, v2: Array) -> "Rectangular":
    #     """Construct from outer product $\\mathbf{v} \\otimes \\mathbf{w}$."""
    #     outer = jnp.outer(v1, v2)
    #     return cls(outer.flatten(), outer.shape)


# @jax.tree_util.register_dataclass
# @dataclass(frozen=True)
# class Square(Rectangular):
#     """Arbitrary square matrices $A \\in \\mathbb R^{n \\times n}$.
#
#     Properties:
#         - LU decomposition used for determinant/inverse
#         - Parameterized by full matrix
#     """
#
#     side_length: int
#
#     def inverse(self: SQ) -> SQ:
#         prms: Array = self.params
#         inv: Array = cast(Array, jnp.linalg.inv(prms))
#         return replace(self, params=inv)
#
#     def transpose(self: SQ) -> SQ:
#         return replace(self, params=self.matrix.T)
#
#     def logdet(self) -> Array:
#         return jnp.linalg.slogdet(self.matrix)[1]  # type: ignore
#
#     @property
#     def matrix(self) -> Array:
#         return self.params
#
#     @classmethod
#     def from_matrix(cls: Type[SQ], matrix: Array) -> SQ:
#         return cls(matrix, matrix.shape[0])
#
#     @classmethod
#     def outer_product(cls: Type[SQ], v1: Array, v2: Array) -> SQ:
#         return cls(jnp.outer(v1, v2), v1.shape[0])
#
#
# @jax.tree_util.register_dataclass
# @dataclass(frozen=True)
# class Symmetric(Square):
#     """Represents matrices $A$ where $A = A^{T}$.
#
#     Properties:
#         - Real eigenvalues
#         - Orthogonal eigenvectors
#         - Parameterized by upper triangular elements
#     """
#
#     def matvec(self, v: Array) -> Array:
#         return jnp.dot(self.matrix, v)
#
#     def transpose(self: SY) -> SY:
#         return self
#
#     def logdet(self) -> Array:
#         return jnp.linalg.slogdet(self.matrix)[1]  # type: ignore
#
#     def inverse(self: SY) -> SY:
#         matrix = self.matrix
#         inv = jnp.linalg.inv(matrix)  # type: ignore
#         i_upper = jnp.triu_indices(matrix.shape[0])
#         return replace(self, params=inv[i_upper])
#
#     @property
#     def matrix(self) -> Array:
#         vec = self.params
#         dim = self.side_length
#         matrix = jnp.zeros((dim, dim))
#         i_upper = jnp.triu_indices(dim)
#         matrix = matrix.at[i_upper].set(vec)
#         return matrix + jnp.triu(matrix, k=1).T
#
#     @classmethod
#     def from_matrix(cls: Type[SY], matrix: Array) -> SY:
#         i_upper = jnp.triu_indices(matrix.shape[0])
#         return cls(matrix[i_upper], matrix.shape[0])
#
#     @classmethod
#     def outer_product(cls: Type[SY], v1: Array, v2: Array) -> SY:
#         matrix = jnp.outer(v1, v2)
#         i_upper = jnp.triu_indices(matrix.shape[0])
#         return cls(matrix[i_upper], matrix.shape[0])
#
#
# @jax.tree_util.register_dataclass
# @dataclass(frozen=True)
# class PositiveDefinite(Symmetric):
#     """Symmetric positive definite matrix operator. Mathematical Definition where $A$ satisfies
#         $x^T A x > 0, \\forall x \\neq 0$
#
#     Properties:
#         - All eigenvalues strictly positive
#         - Unique Cholesky decomposition $A = LL^T$
#         - Inverse exists and is positive definite
#         - Parameterized by upper triangular elements
#     """
#
#     def matvec(self, v: Array) -> Array:
#         chol = self.cholesky
#         return chol @ (chol.T @ v)
#
#     def inverse(self: PD) -> PD:
#         chol = self.cholesky
#         inv_chol = jax.scipy.linalg.solve_triangular(
#             chol, jnp.eye(chol.shape[0]), lower=True
#         )
#         return self.from_cholesky(inv_chol.T)
#
#     def logdet(self) -> Array:
#         return 2.0 * jnp.sum(jnp.log(jnp.diag(self.cholesky)))
#
#     @cached_property
#     def cholesky(self) -> Array:
#         """Compute the Cholesky decomposition."""
#         return jnp.linalg.cholesky(self.matrix)  # type: ignore
#
#     @classmethod
#     def from_cholesky(cls: Type[PD], chol: Array) -> PD:
#         matrix = chol @ chol.T
#         i_upper = jnp.triu_indices(matrix.shape[0])
#         return cls(matrix[i_upper], matrix.shape[0])
#
#     @classmethod
#     def from_matrix(cls: Type[PD], matrix: Array) -> PD:
#         i_upper = jnp.triu_indices(matrix.shape[0])
#         return cls(matrix[i_upper], matrix.shape[0])
#
#     @classmethod
#     def outer_product(cls: Type[PD], v1: Array, v2: Array) -> PD:
#         matrix = jnp.outer(v1, v2)
#         i_upper = jnp.triu_indices(matrix.shape[0])
#         return cls(matrix[i_upper], matrix.shape[0])
#
#
# @jax.tree_util.register_dataclass
# @dataclass(frozen=True)
# class Diagonal(PositiveDefinite):
#     """Diagonal matrix operator.
#
#     Properties:
#
#         - Most operations are element-wise
#         - Computational complexity scales linearly with dimension
#     """
#
#     def matvec(self, v: Array) -> Array:
#         return self.params * v
#
#     def inverse(self: DI) -> DI:
#         return replace(self, params=1.0 / self.params)
#
#     @property
#     def matrix(self) -> Array:
#         return jnp.diag(self.params)
#
#     @cached_property
#     def cholesky(self) -> Array:
#         return jnp.diag(jnp.sqrt(self.params))
#
#     @classmethod
#     def from_matrix(cls: Type[DI], matrix: Array) -> DI:
#         return cls(jnp.diag(matrix), matrix.shape[0])
#
#     @classmethod
#     def outer_product(cls: Type[DI], v1: Array, v2: Array) -> DI:
#         return cls(v1 * v2, v1.shape[0])
#
#
# @jax.tree_util.register_dataclass
# @dataclass(frozen=True)
# class Scale(Diagonal):
#     """Scalar multiple of identity matrix defined by a single parameter.
#
#     Properties:
#
#         - Efficiently represents isotropic scaling
#         - Inverse is the reciprocal
#     """
#
#     def logdet(self) -> Array:
#         return jnp.squeeze(self.side_length * jnp.log(self.params))
#
#     def inverse(self: SC) -> SC:
#         return replace(self, params=1.0 / self.params)
#
#     @property
#     def matrix(self) -> Array:
#         return self.params * jnp.eye(self.side_length)
#
#     @classmethod
#     def from_matrix(cls: Type[SC], matrix: Array) -> SC:
#         return cls(jnp.mean(jnp.diag(matrix)), matrix.shape[0])
#
#     @classmethod
#     def outer_product(cls: Type[SC], v1: Array, v2: Array) -> SC:
#         return cls(jnp.atleast_1d(jnp.mean(v1 * v2)), v1.shape[0])
#
#
# @jax.tree_util.register_dataclass
# @dataclass(frozen=True)
# class Identity(Scale):
#     """Identity operator. Most operations are $O(1)$."""
#
#     def matvec(self, v: Array) -> Array:
#         return v
#
#     def logdet(self) -> Array:
#         return jnp.array(0.0)
#
#     @classmethod
#     def from_matrix(cls: Type[ID], matrix: Array) -> ID:
#         return cls(jnp.array([]), matrix.shape[0])
#
#     @classmethod
#     def outer_product(cls: Type[ID], v1: Array, v2: Array) -> ID:
#         return cls(jnp.array([]), v1.shape[0])
