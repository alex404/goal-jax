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
from dataclasses import dataclass
from typing import Type, TypeVar, cast

import jax.numpy as jnp
from jax import Array

### TypeVar Definitions ###

LO = TypeVar("LO", bound="LinearOperator")
S = TypeVar("S", bound="Square")
SYM = TypeVar("SYM", bound="Symmetric")
PD = TypeVar("PD", bound="PositiveDefinite")
D = TypeVar("D", bound="Diagonal")
SC = TypeVar("SC", bound="Scale")
ID = TypeVar("ID", bound="Identity")

### Classes ###


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
        return type(self)(self.params + other.params)

    def __sub__(self: LO, other: LO) -> LO:
        return type(self)(self.params - other.params)

    def __mul__(self: LO, scalar: float) -> LO:
        return type(self)(scalar * self.params)

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


@dataclass(frozen=True)
class Square(LinearOperator):
    """Arbitrary square matrices $A \\in \\mathbb R^{n \\times n}$.

    Properties:
        - LU decomposition used for determinant/inverse
        - Parameterized by full matrix
    """

    def matvec(self, v: Array) -> Array:
        return jnp.dot(self.matrix, v)

    def inverse(self) -> "Square":
        prms: Array = self.params
        inv: Array = cast(Array, jnp.linalg.inv(prms))
        return Square(inv)

    def transpose(self) -> "Square":
        return Square(self.matrix.T)

    def logdet(self) -> Array:
        return jnp.linalg.slogdet(self.matrix)[1]  # type: ignore

    @property
    def side_length(self) -> int:
        """Side length of square matrix."""
        return self.params.shape[0]

    @property
    def matrix(self) -> Array:
        return self.params

    @classmethod
    def from_matrix(cls: Type[S], matrix: Array) -> S:
        return cls(matrix)

    @classmethod
    def outer_product(cls: Type[S], v1: Array, v2: Array) -> S:
        return cls(jnp.outer(v1, v2))

    @classmethod
    def from_params(cls: Type[S], params: Array, side_length: int) -> S:
        """Construct from parameters and side length."""
        side_length = side_length
        return cls(params)


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

    def transpose(self: SYM) -> SYM:
        return self

    @property
    def side_length(self) -> int:
        return int((jnp.sqrt(1 + 8 * self.params.size) - 1) / 2)

    def logdet(self) -> Array:
        return jnp.linalg.slogdet(self.matrix)[1]  # type: ignore

    def inverse(self: SYM) -> SYM:
        matrix = self.matrix
        inv = jnp.linalg.inv(matrix)  # type: ignore
        i_upper = jnp.triu_indices(matrix.shape[0])
        return type(self)(inv[i_upper])

    @property
    def matrix(self) -> Array:
        vec = self.params
        n = vec.size
        dim = int((jnp.sqrt(1 + 8 * n) - 1) / 2)
        matrix = jnp.zeros((dim, dim))
        i_upper = jnp.triu_indices(dim)
        matrix = matrix.at[i_upper].set(vec)
        return matrix + jnp.triu(matrix, k=1).T

    @classmethod
    def from_matrix(cls: Type[SYM], matrix: Array) -> SYM:
        i_upper = jnp.triu_indices(matrix.shape[0])
        return cls(matrix[i_upper])

    @classmethod
    def outer_product(cls: Type[SYM], v1: Array, v2: Array) -> SYM:
        matrix = jnp.outer(v1, v2)
        i_upper = jnp.triu_indices(matrix.shape[0])
        return cls(matrix[i_upper])

    @classmethod
    def from_params(cls: Type[SYM], params: Array, side_length: int) -> SYM:
        return cls(params)


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
        inv_chol = jnp.linalg.inv(self.cholesky)  # type: ignore
        return self.from_cholesky(inv_chol.T)

    def logdet(self) -> Array:
        return 2.0 * jnp.sum(jnp.log(jnp.diag(self.cholesky)))

    @property
    def cholesky(self) -> Array:
        """Compute the Cholesky decomposition."""
        return jnp.linalg.cholesky(self.matrix)  # type: ignore

    @classmethod
    def from_cholesky(cls: Type[PD], chol: Array) -> PD:
        matrix = chol @ chol.T
        i_upper = jnp.triu_indices(matrix.shape[0])
        return cls(matrix[i_upper])

    @classmethod
    def from_matrix(cls: Type[PD], matrix: Array) -> PD:
        i_upper = jnp.triu_indices(matrix.shape[0])
        return cls(matrix[i_upper])

    @classmethod
    def outer_product(cls: Type[PD], v1: Array, v2: Array) -> PD:
        matrix = jnp.outer(v1, v2)
        i_upper = jnp.triu_indices(matrix.shape[0])
        return cls(matrix[i_upper])

    @classmethod
    def from_params(cls: Type[PD], params: Array, side_length: int) -> PD:
        return cls(params)


@dataclass(frozen=True)
class Diagonal(PositiveDefinite):
    """Diagonal matrix operator.

    Properties:

        - Most operations are element-wise
        - Computational complexity scales linearly with dimension
    """

    def matvec(self, v: Array) -> Array:
        return self.params * v

    def inverse(self: D) -> D:
        return type(self)(1.0 / self.params)

    @property
    def matrix(self) -> Array:
        return jnp.diag(self.params)

    @property
    def cholesky(self) -> Array:
        return jnp.diag(jnp.sqrt(self.params))

    @classmethod
    def from_matrix(cls: Type[D], matrix: Array) -> D:
        return cls(jnp.diag(matrix))

    @classmethod
    def outer_product(cls: Type[D], v1: Array, v2: Array) -> D:
        return cls(v1 * v2)

    @classmethod
    def from_params(cls: Type[D], params: Array, side_length: int) -> D:
        return cls(params)


@dataclass(frozen=True)
class Scale(Diagonal):
    """Scalar multiple of identity matrix defined by a single parameter.

    Properties:

        - Efficiently represents isotropic scaling
        - Inverse is the reciprocal
    """

    _side_length: int

    def logdet(self) -> Array:
        return self._side_length * jnp.log(self.params)

    def inverse(self: SC) -> SC:
        return type(self)(1.0 / self.params, self._side_length)

    @property
    def matrix(self) -> Array:
        return self.params * jnp.eye(self._side_length)

    @classmethod
    def from_matrix(cls: Type[SC], matrix: Array) -> SC:
        return cls(jnp.mean(jnp.diag(matrix)), matrix.shape[0])

    @classmethod
    def outer_product(cls: Type[SC], v1: Array, v2: Array) -> SC:
        return cls(jnp.mean(v1 * v2), v1.shape[0])

    @classmethod
    def from_params(cls: Type[SC], params: Array, side_length: int) -> SC:
        return cls(params, side_length)


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
