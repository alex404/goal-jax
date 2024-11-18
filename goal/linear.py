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

import jax.numpy as jnp

from goal.manifold import Manifold


class LinearOperator(Manifold, ABC):
    """Abstract base class for linear maps between vector spaces. A linear map $L: V \\mapsto W$ between vector spaces satisfies
        $L(\\alpha x + \\beta y) = \\alpha L(x) + \\beta L(y)$.

    In general we store the matrix representation $A \\in \\mathbb R^{m \\times n}$ of the linear map, but subclasses may make parametric assumptions that allow more efficient representations.

    JAX Implementation:
        - All operations return new objects (immutable)
        - Uses JAX's array operations for efficiency
        - Supports vectorization via jit/vmap
    """

    @abstractmethod
    def matvec(self, v: jnp.ndarray) -> jnp.ndarray:
        """Matrix-vector product $Av$."""
        pass

    @abstractmethod
    def transpose(self) -> "LinearOperator":
        """Transpose of the linear map."""
        pass

    @abstractmethod
    def parameter_count(self) -> int:
        """Number of parameters needed to represent the map."""
        pass

    @abstractmethod
    def logdet(self) -> jnp.ndarray:
        """Compute log determinant."""
        pass

    @abstractmethod
    def inverse(self) -> "LinearOperator":
        """Compute inverse."""
        pass

    def __matmul__(self, v: jnp.ndarray) -> jnp.ndarray:
        """Allow @ syntax for matrix-vector product."""
        return self.matvec(v)

    @property
    @abstractmethod
    def matrix(self) -> jnp.ndarray:
        """Full matrix representation."""
        pass

    @classmethod
    @abstractmethod
    def from_matrix(cls, matrix: jnp.ndarray) -> "LinearOperator":
        """Construct from matrix."""
        pass

    @classmethod
    @abstractmethod
    def outer_product(cls, v1: jnp.ndarray, v2: jnp.ndarray) -> "LinearOperator":
        """Construct from outer product v1 v2^T."""
        pass


@dataclass(frozen=True)
class Square(LinearOperator):
    """Arbitrary square matrices $A \\in \\mathbb R^{n \\times n}$.

    Properties:

        - LU decomposition used for determinant/inverse
        - Parameterized by full matrix

    """

    def matvec(self, v: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(self.matrix, v)

    def inverse(self) -> "Square":
        return Square(jnp.linalg.inv(self.params))

    def transpose(self) -> "Square":
        return Square(self.matrix.T)

    def parameter_count(self) -> int:
        return self.matrix.size

    def logdet(self) -> jnp.ndarray:
        return jnp.linalg.slogdet(self.matrix)[1]

    @property
    def matrix(self) -> jnp.ndarray:
        return self.params

    @classmethod
    def from_matrix(cls, matrix: jnp.ndarray) -> "Square":
        return cls(matrix)

    @classmethod
    def outer_product(cls, v1: jnp.ndarray, v2: jnp.ndarray) -> "Square":
        return cls(jnp.outer(v1, v2))


@dataclass(frozen=True)
class Symmetric(LinearOperator):
    """Represents matrices $A$ where $A = A^{T}$.

    Properties:
        - Real eigenvalues
        - Orthogonal eigenvectors
        - Parameterized by upper triangular elements
    """

    def matvec(self, v: jnp.ndarray) -> jnp.ndarray:
        """Matrix-vector product using vectorized operations."""
        # Reconstruct matrix using vectorized operations
        matrix = _expand_triangular(self.params)
        return jnp.dot(matrix, v)

    def transpose(self) -> "Symmetric":
        return self

    def parameter_count(self) -> int:
        return self.params.size

    def logdet(self) -> jnp.ndarray:
        matrix = _expand_triangular(self.params)
        return jnp.linalg.slogdet(matrix)[1]

    def inverse(self) -> "Symmetric":
        """Compute inverse."""
        matrix = _expand_triangular(self.params)
        inv = jnp.linalg.inv(matrix)
        i_upper = jnp.triu_indices(matrix.shape[0])
        return Symmetric(inv[i_upper])

    @property
    def matrix(self) -> jnp.ndarray:
        n = self.params.size
        dim = int((jnp.sqrt(1 + 8 * n) - 1) / 2)
        matrix = jnp.zeros((dim, dim))
        i_upper = jnp.triu_indices(dim)
        matrix = matrix.at[i_upper].set(self.params)
        return matrix + jnp.triu(matrix, k=1).T

    @classmethod
    def from_matrix(cls, matrix: jnp.ndarray) -> "Symmetric":
        i_upper = jnp.triu_indices(matrix.shape[0])
        return cls(matrix[i_upper])

    @classmethod
    def outer_product(cls, v1: jnp.ndarray, v2: jnp.ndarray) -> "Symmetric":
        """Create symmetric matrix from outer product."""
        matrix = jnp.outer(v1, v2)
        i_upper = jnp.triu_indices(matrix.shape[0])
        return cls(matrix[i_upper])


@dataclass(frozen=True)
class PositiveDefinite(LinearOperator):
    """Symmetric positive definite matrix operator. Mathematical Definition where $A$ satisfies
        $x^T A x > 0, \\forall x \\neq 0$

    Properties:
        - All eigenvalues strictly positive
        - Unique Cholesky decomposition $A = LL^T$
        - Inverse exists and is positive definite
        - Parameterized by upper triangular elements

    Implementation:
        Stores both upper triangular elements and Cholesky factor
    """

    _chol: jnp.ndarray  # Lower triangular Cholesky factor for computation

    def matvec(self, v: jnp.ndarray) -> jnp.ndarray:
        """Matrix-vector product using Cholesky factor for efficiency."""
        return self._chol @ (self._chol.T @ v)

    def transpose(self) -> "PositiveDefinite":
        return self

    def parameter_count(self) -> int:
        return self.params.size

    def logdet(self) -> jnp.ndarray:
        """Compute logdet using Cholesky factor."""
        return 2.0 * jnp.sum(jnp.log(jnp.diag(self._chol)))

    def inverse(self) -> "PositiveDefinite":
        """Compute inverse via Cholesky decomposition."""
        inv_chol = jnp.linalg.inv(self._chol)
        # The transpose of inv_chol is the Cholesky factor of the inverse
        return PositiveDefinite.from_cholesky(inv_chol.T)

    @classmethod
    def from_cholesky(cls, chol: jnp.ndarray) -> "PositiveDefinite":
        """Create from Cholesky factor."""
        matrix = chol @ chol.T
        i_upper = jnp.triu_indices(matrix.shape[0])
        return cls(matrix[i_upper], chol)

    @property
    def cholesky(self) -> jnp.ndarray:
        return self._chol

    @property
    def matrix(self) -> jnp.ndarray:
        return self._chol @ self._chol.T

    @classmethod
    def from_matrix(cls, matrix: jnp.ndarray) -> "PositiveDefinite":
        chol = jnp.linalg.cholesky(matrix)
        i_upper = jnp.triu_indices(matrix.shape[0])
        return cls(matrix[i_upper], chol)

    @classmethod
    def outer_product(cls, v1: jnp.ndarray, v2: jnp.ndarray) -> "PositiveDefinite":
        """Create positive definite matrix from outer product."""
        matrix = jnp.outer(v1, v2)
        i_upper = jnp.triu_indices(matrix.shape[0])
        chol = jnp.linalg.cholesky(matrix)  # This might fail if not PD
        return cls(matrix[i_upper], chol)


@dataclass(frozen=True)
class Diagonal(LinearOperator):
    """Diagonal matrix operator."""

    def matvec(self, v: jnp.ndarray) -> jnp.ndarray:
        """Matrix-vector product (elementwise multiply)."""
        return self.params * v

    def transpose(self) -> "Diagonal":
        return self

    def parameter_count(self) -> int:
        return self.params.size

    def logdet(self) -> jnp.ndarray:
        return jnp.sum(jnp.log(self.params))

    def inverse(self) -> "Diagonal":
        return Diagonal(1.0 / self.params)

    @property
    def cholesky(self) -> jnp.ndarray:
        return jnp.diag(jnp.sqrt(self.params))

    @property
    def matrix(self) -> jnp.ndarray:
        return jnp.diag(self.params)

    @classmethod
    def from_matrix(cls, matrix: jnp.ndarray) -> "Diagonal":
        return cls(jnp.diag(matrix))

    @classmethod
    def outer_product(cls, v1: jnp.ndarray, v2: jnp.ndarray) -> "Diagonal":
        """Create diagonal matrix from element-wise product."""
        return cls(v1 * v2)


@dataclass(frozen=True)
class Scale(LinearOperator):
    """Scalar multiple of identity matrix."""

    _dim: int

    def matvec(self, v: jnp.ndarray) -> jnp.ndarray:
        return self.params * v

    def transpose(self) -> "Scale":
        return self

    def parameter_count(self) -> int:
        return 1

    def logdet(self) -> jnp.ndarray:
        return self._dim * jnp.log(self.params)

    def inverse(self) -> "Scale":
        return Scale(1.0 / self.params, self._dim)

    @property
    def cholesky(self) -> jnp.ndarray:
        return jnp.sqrt(self.params) * jnp.eye(self._dim)

    @property
    def matrix(self) -> jnp.ndarray:
        return self.params * jnp.eye(self._dim)

    @classmethod
    def from_matrix(cls, matrix: jnp.ndarray) -> "Scale":
        """Create scale matrix using average of diagonal."""
        return cls(jnp.mean(jnp.diag(matrix)), matrix.shape[0])

    @classmethod
    def outer_product(cls, v1: jnp.ndarray, v2: jnp.ndarray) -> "Scale":
        """Create scale matrix using average of element-wise product."""
        return cls(jnp.mean(v1 * v2), v1.shape[0])


@dataclass
class Identity(LinearOperator):
    """Identity operator."""

    _dim: int

    def matvec(self, v: jnp.ndarray) -> jnp.ndarray:
        return v

    def transpose(self) -> "Identity":
        return self

    def parameter_count(self) -> int:
        return 0

    def to_vector(self) -> jnp.ndarray:
        return jnp.array([])

    def logdet(self) -> jnp.ndarray:
        return jnp.array(0.0)

    def inverse(self) -> "Identity":
        return self

    @property
    def matrix(self) -> jnp.ndarray:
        return jnp.eye(self._dim)

    @classmethod
    def from_matrix(cls, matrix: jnp.ndarray) -> "Identity":
        return cls(jnp.array([]), matrix.shape[0])

    @classmethod
    def outer_product(cls, v1: jnp.ndarray, v2: jnp.ndarray) -> "Identity":
        """Create identity matrix (ignores inputs)."""
        return cls(jnp.array([]), v1.shape[0])


def _expand_triangular(vec: jnp.ndarray) -> jnp.ndarray:
    """Expand vector of upper triangular elements to symmetric matrix."""
    n = vec.size
    dim = int((jnp.sqrt(1 + 8 * n) - 1) / 2)
    matrix = jnp.zeros((dim, dim))
    i_upper = jnp.triu_indices(dim)
    matrix = matrix.at[i_upper].set(vec)
    return matrix + jnp.triu(matrix, k=1).T
