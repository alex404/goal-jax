"""Linear operators with structural specializations.

This module provides a hierarchy of linear operators with increasingly specialized
structure. Each specialization maintains key matrix operations (multiplication, determinant,
inverse) while exploiting structure for efficiency:

Operator Type    | Storage   | Matmul    | Inverse/Det
----------------|------------|-----------|-------------
Rectangular     | $O(n^2)$   | $O(n^2)$  | $O(n^3)$
Symmetric       | $O(n^2/2)$ | $O(n^2)$  | $O(n^3)$
Pos. Definite   | $O(n^2/2)$ | $O(n^2)$  | $O(n^3)$ (Cholesky)
Diagonal        | $O(n)$     | $O(n)$    | $O(n^3)$
Scale           | $O(1)$     | $O(n)$    | $O(1)$
Identity        | $O(1)$     | $O(1)$    | $O(1)$

Notes:
   - All operators are immutable to align with JAX's functional style
   - Specialized operators maintain their structure under inversion
   - Operations use numerically stable algorithms where possible

"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax import Array

### Matrix Representations ###


class MatrixRep(ABC):
    """Base class defining how to interpret and manipulate matrix parameters.

    All matrix parameters are stored as 1D arrays for compatibility with Point. Each subclass defines how to reshape and manipulate these parameters while maintaining their specific structure (full, symmetric, diagonal, etc.)
    """

    @abstractmethod
    def matvec(self, params: Array, vector: Array, shape: tuple[int, int]) -> Array:
        """Matrix-vector multiplication."""
        ...

    @abstractmethod
    def transpose(self, params: Array, shape: tuple[int, int]) -> Array:
        """Transform parameters to represent the transposed matrix."""
        ...

    @abstractmethod
    def to_dense(self, params: Array, shape: tuple[int, int]) -> Array:
        """Convert 1D parameters to dense matrix form."""
        ...

    @abstractmethod
    def to_cols(self, params: Array, shape: tuple[int, int]) -> list[Array]:
        """Convert parameter vector to list of column vectors."""
        ...

    @abstractmethod
    def from_cols(self, cols: list[Array], shape: tuple[int, int]) -> Array:
        """Construct parameter vector from list of column vectors."""
        ...

    @abstractmethod
    def from_dense(self, matrix: Array) -> Array:
        """Convert dense matrix to 1D parameters."""
        ...

    @abstractmethod
    def num_params(self, matrix_shape: tuple[int, int]) -> int:
        """Shape of 1D parameter array needed for matrix dimensions."""
        ...

    @abstractmethod
    def outer_product(self, v1: Array, v2: Array) -> Array:
        """Construct parameters from outer product $v_1 \\otimes v_2$."""
        ...


class Rectangular(MatrixRep):
    """Full matrix representation with no special structure."""

    def matvec(self, params: Array, vector: Array, shape: tuple[int, int]) -> Array:
        matrix = self.to_dense(params, shape)
        return jnp.dot(matrix, vector)

    def transpose(self, params: Array, shape: tuple[int, int]) -> Array:
        matrix = self.to_dense(params, shape).T
        return matrix.reshape(-1)

    def to_dense(self, params: Array, shape: tuple[int, int]) -> Array:
        return params.reshape(shape)

    def to_cols(self, params: Array, shape: tuple[int, int]) -> list[Array]:
        matrix = self.to_dense(params, shape)
        return [matrix[:, j] for j in range(shape[1])]

    def from_cols(self, cols: list[Array], shape: tuple[int, int]) -> Array:
        """Construct parameter vector from list of column vectors."""
        matrix = jnp.column_stack(cols)
        return self.from_dense(matrix)

    def from_dense(self, matrix: Array) -> Array:
        return matrix.reshape(-1)

    def num_params(self, matrix_shape: tuple[int, int]) -> int:
        n, m = matrix_shape
        return n * m

    def outer_product(self, v1: Array, v2: Array) -> Array:
        """Create parameters from outer product."""
        matrix = jnp.outer(v1, v2)
        return self.from_dense(matrix)


class Square(Rectangular):
    """Square matrix representation.

    Properties:
        - $n \\times n$ matrix shape
        - Parameters stored as full matrix in row-major order
    """

    def inverse(self, params: Array, shape: tuple[int, int]) -> Array:
        matrix = self.to_dense(params, shape)
        inv = jnp.linalg.inv(matrix)  # type: ignore
        return self.from_dense(inv)

    def logdet(self, params: Array, shape: tuple[int, int]) -> Array:
        matrix = self.to_dense(params, shape)
        return jnp.linalg.slogdet(matrix)[1]  # type: ignore


class Symmetric(Square):
    """Symmetric matrix representation where $A = A^T$.

    Properties:
        - Stores only upper/lower triangular elements
        - Parameter vector contains n*(n+1)/2 elements for $n \\times n$ matrix
        - Self-transpose, real eigenvalues
    """

    def transpose(self, params: Array, shape: tuple[int, int]) -> Array:
        """Symmetric matrices are self-transpose."""
        return params

    def to_dense(self, params: Array, shape: tuple[int, int]) -> Array:
        n = shape[0]
        matrix = jnp.zeros((n, n))
        i_upper = jnp.triu_indices(n)
        matrix = matrix.at[i_upper].set(params)
        return matrix + jnp.triu(matrix, k=1).T

    def from_dense(self, matrix: Array) -> Array:
        n = matrix.shape[0]
        i_upper = jnp.triu_indices(n)
        return matrix[i_upper]

    def num_params(self, matrix_shape: tuple[int, int]) -> int:
        n = matrix_shape[0]
        return (n * (n + 1)) // 2


class PositiveDefinite(Symmetric):
    """Symmetric positive definite matrix representation.

    Properties:
        - Symmetric $n \\times n$ matrix with $x^T A x > 0$ for all $x \\neq 0$
        - All eigenvalues strictly positive
        - Unique Cholesky decomposition $A = LL^T$
        - Parameterized by upper triangular elements
    """

    def inverse(self, params: Array, shape: tuple[int, int]) -> Array:
        """Inverse via Cholesky decomposition."""
        chol = self.cholesky(params, shape)
        n = shape[0]
        eye = jnp.eye(n)
        # Solve L L^T x = I
        inv_chol = jax.scipy.linalg.solve_triangular(chol, eye, lower=True)
        inv = inv_chol.T @ inv_chol
        return self.from_dense(inv)

    def logdet(self, params: Array, shape: tuple[int, int]) -> Array:
        """Log determinant via Cholesky."""
        chol = self.cholesky(params, shape)
        return 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))

    def cholesky(self, params: Array, shape: tuple[int, int]) -> Array:
        """Compute lower triangular Cholesky factor L where A = LL^T."""
        matrix = self.to_dense(params, shape)
        return jnp.linalg.cholesky(matrix)  # type: ignore

    def apply_cholesky(
        self, params: Array, vector: Array, shape: tuple[int, int]
    ) -> Array:
        chol = self.cholesky(params, shape)
        return (chol @ vector.T).T


class Diagonal(PositiveDefinite):
    """Diagonal matrix representation $A = \\text{diag}(a_1, ..., a_n)$.

    Properties:
        - Only diagonal elements stored, zero elsewhere
        - Most operations reduce to element-wise operations on diagonal
        - $O(n)$ storage and operations
    """

    def matvec(self, params: Array, vector: Array, shape: tuple[int, int]) -> Array:
        return params * vector

    def transpose(self, params: Array, shape: tuple[int, int]) -> Array:
        return params

    def to_dense(self, params: Array, shape: tuple[int, int]) -> Array:
        n = shape[0]
        matrix = jnp.zeros((n, n))
        return matrix.at[jnp.diag_indices(n)].set(params)

    def from_dense(self, matrix: Array) -> Array:
        return jnp.diag(matrix)

    def num_params(self, matrix_shape: tuple[int, int]) -> int:
        return matrix_shape[0]

    def inverse(self, params: Array, shape: tuple[int, int]) -> Array:
        return 1.0 / params

    def logdet(self, params: Array, shape: tuple[int, int]) -> Array:
        return jnp.sum(jnp.log(params))

    def cholesky(self, params: Array, shape: tuple[int, int]) -> Array:
        return jnp.sqrt(params)

    def outer_product(self, v1: Array, v2: Array) -> Array:
        """Create parameters from outer product, keeping only diagonal."""
        return v1 * v2

    def apply_cholesky(
        self, params: Array, vector: Array, shape: tuple[int, int]
    ) -> Array:
        return jnp.sqrt(params) * vector


class Scale(Diagonal):
    """Scale transformation $A = \\alpha I$.

    Properties:
        - Single parameter $\\alpha$ represents uniform scaling
        - All operations are $O(1)$ in storage
        - Matrix operations reduce to scalar operations on $\\alpha$
    """

    def matvec(self, params: Array, vector: Array, shape: tuple[int, int]) -> Array:
        return params[0] * vector

    def to_dense(self, params: Array, shape: tuple[int, int]) -> Array:
        n = shape[0]
        return params[0] * jnp.eye(n)

    def from_dense(self, matrix: Array) -> Array:
        return jnp.array([jnp.mean(jnp.diag(matrix))])

    def num_params(self, matrix_shape: tuple[int, int]) -> int:
        return 1

    def logdet(self, params: Array, shape: tuple[int, int]) -> Array:
        n = shape[0]
        return n * jnp.log(params[0])

    def outer_product(self, v1: Array, v2: Array) -> Array:
        """Average outer product to single scale parameter."""
        return jnp.array([jnp.mean(v1 * v2)])

    def apply_cholesky(
        self, params: Array, vector: Array, shape: tuple[int, int]
    ) -> Array:
        return jnp.sqrt(params[0]) * vector


class Identity(Scale):
    """Identity transformation $A = I$.

    Properties:
        - Zero parameters - fully determined by shape
        - All operations are $O(1)$ and parameter-free
        - Acts as multiplicative identity in composition
    """

    def matvec(self, params: Array, vector: Array, shape: tuple[int, int]) -> Array:
        return vector

    def to_dense(self, params: Array, shape: tuple[int, int]) -> Array:
        n = shape[0]
        return jnp.eye(n)

    def from_dense(self, matrix: Array) -> Array:
        return jnp.array([])

    def num_params(self, matrix_shape: tuple[int, int]) -> int:
        return 0

    def inverse(self, params: Array, shape: tuple[int, int]) -> Array:
        return params

    def logdet(self, params: Array, shape: tuple[int, int]) -> Array:
        return jnp.array(0.0)

    def outer_product(self, v1: Array, v2: Array) -> Array:
        """Identity ignores input vectors."""
        return jnp.array([])

    def apply_cholesky(
        self, params: Array, vector: Array, shape: tuple[int, int]
    ) -> Array:
        return vector
