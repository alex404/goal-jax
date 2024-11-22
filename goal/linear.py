"""Linear operators with structural specializations.

This module provides a hierarchy of linear operators with increasingly specialized
structure. Each specialization maintains key matrix operations (multiplication, determinant,
inverse) while exploiting structure for efficiency:

Operator Type    | Storage   | Matmul    | Inverse/Det
----------------|-----------|-----------|-------------
Dense           | O(n²)     | O(n²)     | O(n³)
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
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Type, TypeVar, cast

import jax
import jax.numpy as jnp
from jax import Array

### TypeVar Definitions ###

LO = TypeVar("LO", bound="LinearOperator")
SQ = TypeVar("SQ", bound="Square")
SY = TypeVar("SY", bound="Symmetric")
PD = TypeVar("PD", bound="PositiveDefinite")
DI = TypeVar("DI", bound="Diagonal")
SC = TypeVar("SC", bound="Scale")
ID = TypeVar("ID", bound="Identity")

### Classes ###


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LinearOperator(ABC):
    """Abstract base class for linear maps between vector spaces. A linear map $L: V \\mapsto W$ between vector spaces satisfies
        $L(\\alpha x + \\beta y) = \\alpha L(x) + \\beta L(y)$.

    In general we store the matrix representation $A \\in \\mathbb R^{m \\times n}$ of the linear map, but subclasses may make parametric assumptions that allow more efficient representations.

    JAX Implementation:
        - All operations return new objects (immutable)
        - Uses JAX's array operations for efficiency
        - Supports vectorization via jit/vmap
    """

    params: Array
    """Parametric representation of the linear map."""

    def __add__(self: LO, other: LO) -> LO:
        return replace(self, params=self.params + other.params)

    def __sub__(self: LO, other: LO) -> LO:
        return replace(self, params=self.params - other.params)

    def __mul__(self: LO, scalar: float) -> LO:
        return replace(self, params=scalar * self.params)

    def __rmul__(self: LO, scalar: float) -> LO:
        return self.__mul__(scalar)

    @abstractmethod
    def matvec(self, v: Array) -> Array:
        """Matrix-vector product $Av$."""
        ...

    @abstractmethod
    def transpose(self) -> "LinearOperator":
        """Transpose of the linear map."""
        ...

    @abstractmethod
    def logdet(self) -> Array:
        """Compute the log determinant of the matrix $\\log|A|$."""
        ...

    @abstractmethod
    def inverse(self) -> "LinearOperator":
        """Compute the inverse $A^{-1}$ of the matrix."""
        ...

    def __matmul__(self, v: Array) -> Array:
        """Allow @ syntax for matrix-vector product."""
        return self.matvec(v)

    @property
    @abstractmethod
    def matrix(self) -> Array:
        """Returns the full matrix representation $A$."""
        ...

    @classmethod
    @abstractmethod
    def from_matrix(cls, matrix: Array) -> "LinearOperator":
        """Construct the linear map by extracting parameters from the given matrix."""
        ...

    @classmethod
    @abstractmethod
    def outer_product(cls, v1: Array, v2: Array) -> "LinearOperator":
        """Construct a linear map from the outer product $\\mathbf{v} \\otimes \\mathbf{w}$."""
        ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Square(LinearOperator):
    """Arbitrary square matrices $A \\in \\mathbb R^{n \\times n}$.

    Properties:
        - LU decomposition used for determinant/inverse
        - Parameterized by full matrix
    """

    side_length: int

    def inverse(self: SQ) -> SQ:
        prms: Array = self.params
        inv: Array = cast(Array, jnp.linalg.inv(prms))
        return replace(self, params=inv)

    def transpose(self: SQ) -> SQ:
        return replace(self, params=self.matrix.T)

    def logdet(self) -> Array:
        return jnp.linalg.slogdet(self.matrix)[1]  # type: ignore

    @property
    def matrix(self) -> Array:
        return self.params

    @classmethod
    def from_matrix(cls: Type[SQ], matrix: Array) -> SQ:
        return cls(matrix, matrix.shape[0])

    @classmethod
    def outer_product(cls: Type[SQ], v1: Array, v2: Array) -> SQ:
        return cls(jnp.outer(v1, v2), v1.shape[0])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Symmetric(Square):
    """Represents matrices $A$ where $A = A^{T}$.

    Properties:
        - Real eigenvalues
        - Orthogonal eigenvectors
        - Parameterized by upper triangular elements
    """

    def matvec(self, v: Array) -> Array:
        return jnp.dot(self.matrix, v)

    def transpose(self: SY) -> SY:
        return self

    def logdet(self) -> Array:
        return jnp.linalg.slogdet(self.matrix)[1]  # type: ignore

    def inverse(self: SY) -> SY:
        matrix = self.matrix
        inv = jnp.linalg.inv(matrix)  # type: ignore
        i_upper = jnp.triu_indices(matrix.shape[0])
        return replace(self, params=inv[i_upper])

    @property
    def matrix(self) -> Array:
        vec = self.params
        dim = self.side_length
        matrix = jnp.zeros((dim, dim))
        i_upper = jnp.triu_indices(dim)
        matrix = matrix.at[i_upper].set(vec)
        return matrix + jnp.triu(matrix, k=1).T

    @classmethod
    def from_matrix(cls: Type[SY], matrix: Array) -> SY:
        i_upper = jnp.triu_indices(matrix.shape[0])
        return cls(matrix[i_upper], matrix.shape[0])

    @classmethod
    def outer_product(cls: Type[SY], v1: Array, v2: Array) -> SY:
        matrix = jnp.outer(v1, v2)
        i_upper = jnp.triu_indices(matrix.shape[0])
        return cls(matrix[i_upper], matrix.shape[0])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PositiveDefinite(Symmetric):
    """Symmetric positive definite matrix operator. Mathematical Definition where $A$ satisfies
        $x^T A x > 0, \\forall x \\neq 0$

    Properties:
        - All eigenvalues strictly positive
        - Unique Cholesky decomposition $A = LL^T$
        - Inverse exists and is positive definite
        - Parameterized by upper triangular elements
    """

    def matvec(self, v: Array) -> Array:
        chol = self.cholesky
        return chol @ (chol.T @ v)

    def inverse(self: PD) -> PD:
        chol = self.cholesky
        inv_chol = jax.scipy.linalg.solve_triangular(
            chol, jnp.eye(chol.shape[0]), lower=True
        )
        return self.from_cholesky(inv_chol.T)

    def logdet(self) -> Array:
        return 2.0 * jnp.sum(jnp.log(jnp.diag(self.cholesky)))

    @cached_property
    def cholesky(self) -> Array:
        """Compute the Cholesky decomposition."""
        return jnp.linalg.cholesky(self.matrix)  # type: ignore

    @classmethod
    def from_cholesky(cls: Type[PD], chol: Array) -> PD:
        matrix = chol @ chol.T
        i_upper = jnp.triu_indices(matrix.shape[0])
        return cls(matrix[i_upper], matrix.shape[0])

    @classmethod
    def from_matrix(cls: Type[PD], matrix: Array) -> PD:
        i_upper = jnp.triu_indices(matrix.shape[0])
        return cls(matrix[i_upper], matrix.shape[0])

    @classmethod
    def outer_product(cls: Type[PD], v1: Array, v2: Array) -> PD:
        matrix = jnp.outer(v1, v2)
        i_upper = jnp.triu_indices(matrix.shape[0])
        return cls(matrix[i_upper], matrix.shape[0])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Diagonal(PositiveDefinite):
    """Diagonal matrix operator.

    Properties:

        - Most operations are element-wise
        - Computational complexity scales linearly with dimension
    """

    def matvec(self, v: Array) -> Array:
        return self.params * v

    def inverse(self: DI) -> DI:
        return replace(self, params=1.0 / self.params)

    @property
    def matrix(self) -> Array:
        return jnp.diag(self.params)

    @cached_property
    def cholesky(self) -> Array:
        return jnp.diag(jnp.sqrt(self.params))

    @classmethod
    def from_matrix(cls: Type[DI], matrix: Array) -> DI:
        return cls(jnp.diag(matrix), matrix.shape[0])

    @classmethod
    def outer_product(cls: Type[DI], v1: Array, v2: Array) -> DI:
        return cls(v1 * v2, v1.shape[0])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Scale(Diagonal):
    """Scalar multiple of identity matrix defined by a single parameter.

    Properties:

        - Efficiently represents isotropic scaling
        - Inverse is the reciprocal
    """

    def logdet(self) -> Array:
        return self.side_length * jnp.log(self.params)

    def inverse(self: SC) -> SC:
        return replace(self, params=1.0 / self.params)

    @property
    def matrix(self) -> Array:
        return self.params * jnp.eye(self.side_length)

    @classmethod
    def from_matrix(cls: Type[SC], matrix: Array) -> SC:
        return cls(jnp.mean(jnp.diag(matrix)), matrix.shape[0])

    @classmethod
    def outer_product(cls: Type[SC], v1: Array, v2: Array) -> SC:
        return cls(jnp.mean(v1 * v2), v1.shape[0])

    @classmethod
    def from_params(cls: Type[SC], params: Array, side_length: int) -> SC:
        return cls(params, side_length)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Identity(Scale):
    """Identity operator. Most operations are $O(1)$."""

    def matvec(self, v: Array) -> Array:
        return v

    def logdet(self) -> Array:
        return jnp.array(0.0)

    @classmethod
    def from_matrix(cls: Type[ID], matrix: Array) -> ID:
        return cls(jnp.array([]), matrix.shape[0])

    @classmethod
    def outer_product(cls: Type[ID], v1: Array, v2: Array) -> ID:
        return cls(jnp.array([]), v1.shape[0])

    @classmethod
    def from_params(cls: Type[ID], params: Array, side_length: int) -> ID:
        return cls(jnp.array([]), side_length)
