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
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

### Helper Functions ###


def _matmat(
    rep: MatrixRep,
    shape: tuple[int, int],
    params: Array,
    right_rep: MatrixRep,
    right_shape: tuple[int, int],
    right_params: Array,
) -> tuple[MatrixRep, tuple[int, int], Array]:
    out_shape = (shape[0], right_shape[1])
    out_rep: MatrixRep
    out_params: Array
    match rep:
        case Identity():
            out_rep = right_rep
            out_params = right_params
        case Scale():
            out_rep = right_rep
            out_params = params[0] * right_params
        case Diagonal():
            match right_rep:
                case Diagonal():
                    out_rep = Diagonal()
                    out_params = params * right_params
                case _:
                    right_dense = right_rep.to_dense(right_shape, right_params)
                    out_dense = params[:, None] * right_dense
                    out_rep = Rectangular() if right_rep is Rectangular() else Square()
                    out_params = out_rep.from_dense(out_dense)
        case _:
            match right_rep:
                case Identity() | Scale() | Diagonal():
                    right_params_t = right_rep.transpose(right_shape, right_params)
                    right_shape_t = (right_shape[1], right_shape[0])
                    return _matmat(
                        right_rep, right_shape_t, right_params_t, rep, shape, params
                    )
                case _:
                    left_dense = rep.to_dense(shape, params)
                    right_dense = right_rep.to_dense(right_shape, right_params)
                    out_rep = (
                        Square() if out_shape[0] == out_shape[1] else Rectangular()
                    )
                    out_dense = left_dense @ right_dense
                    out_params = out_rep.from_dense(out_dense)
    return out_rep, out_shape, out_params


### Matrix Representations ###


class MatrixRep(ABC):
    """Base class defining how to interpret and manipulate matrix parameters.

    All matrix parameters are stored as 1D arrays for compatibility with Point. Each subclass defines how to reshape and manipulate these parameters while maintaining their specific structure (full, symmetric, diagonal, etc.)
    """

    @classmethod
    @abstractmethod
    def matvec(cls, shape: tuple[int, int], params: Array, vector: Array) -> Array:
        """Matrix-vector multiplication."""
        ...

    @classmethod
    def matmat(
        cls,
        shape: tuple[int, int],
        params: Array,
        right_rep: MatrixRep,
        right_shape: tuple[int, int],
        right_params: Array,
    ) -> tuple[MatrixRep, tuple[int, int], Array]:
        """Multiply matrices, returning optimal representation type and parameters.

        Args:
            shape: (m,n) shape of left matrix
            params: Parameters of left matrix
            right_rep: Matrix representation class of right matrix
            right_shape: (n,p) shape of right matrix
            right_params: Parameters of right matrix

        Returns:
            Tuple of:
                - Optimal matrix representation class
                - Resulting (m,p) shape
                - Parameters for resulting matrix
        """
        return _matmat(cls(), shape, params, right_rep, right_shape, right_params)

    @classmethod
    @abstractmethod
    def transpose(cls, shape: tuple[int, int], params: Array) -> Array:
        """Transform parameters to represent the transposed matrix."""
        ...

    @classmethod
    @abstractmethod
    def to_dense(cls, shape: tuple[int, int], params: Array) -> Array:
        """Convert 1D parameters to dense matrix form."""
        ...

    @classmethod
    @abstractmethod
    def to_cols(cls, shape: tuple[int, int], params: Array) -> list[Array]:
        """Convert parameter vector to list of column vectors."""
        ...

    @classmethod
    @abstractmethod
    def num_params(cls, shape: tuple[int, int]) -> int:
        """Shape of 1D parameter array needed for matrix dimensions."""
        ...

    @classmethod
    @abstractmethod
    def from_cols(cls, cols: list[Array]) -> Array:
        """Construct parameter vector from list of column vectors."""
        ...

    @classmethod
    @abstractmethod
    def from_dense(cls, matrix: Array) -> Array:
        """Convert dense matrix to 1D parameters."""
        ...

    @classmethod
    @abstractmethod
    def outer_product(cls, v1: Array, v2: Array) -> Array:
        """Construct parameters from outer product $v_1 \\otimes v_2$."""
        ...

    @classmethod
    @abstractmethod
    def map_diagonal(
        cls, shape: tuple[int, int], params: Array, f: Callable[[Array], Array]
    ) -> Array:
        """Apply function f to diagonal elements while preserving matrix structure."""
        ...


class Rectangular(MatrixRep):
    """Full matrix representation with no special structure."""

    @classmethod
    def matvec(cls, shape: tuple[int, int], params: Array, vector: Array) -> Array:
        matrix = cls.to_dense(shape, params)
        return jnp.dot(matrix, vector)

    @classmethod
    def transpose(cls, shape: tuple[int, int], params: Array) -> Array:
        matrix = cls.to_dense(shape, params).T
        return matrix.reshape(-1)

    @classmethod
    def to_dense(cls, shape: tuple[int, int], params: Array) -> Array:
        return params.reshape(shape)

    @classmethod
    def to_cols(cls, shape: tuple[int, int], params: Array) -> list[Array]:
        matrix = cls.to_dense(shape, params)
        return [matrix[:, j] for j in range(shape[1])]

    @classmethod
    def from_cols(cls, cols: list[Array]) -> Array:
        """Construct parameter vector from list of column vectors."""
        matrix = jnp.column_stack(cols)
        return cls.from_dense(matrix)

    @classmethod
    def from_dense(cls, matrix: Array) -> Array:
        return matrix.reshape(-1)

    @classmethod
    def num_params(cls, shape: tuple[int, int]) -> int:
        n, m = shape
        return n * m

    @classmethod
    def outer_product(cls, v1: Array, v2: Array) -> Array:
        """Create parameters from outer product."""
        matrix = jnp.outer(v1, v2)
        return cls.from_dense(matrix)

    @classmethod
    def map_diagonal(
        cls, shape: tuple[int, int], params: Array, f: Callable[[Array], Array]
    ) -> Array:
        """Map function over diagonal elements of matrix."""
        matrix = cls.to_dense(shape, params)
        diag = jnp.diag(matrix)
        new_diag = f(diag)
        new_matrix = matrix.at[jnp.diag_indices(shape[0])].set(new_diag)
        return cls.from_dense(new_matrix)


class Square(Rectangular):
    """Square matrix representation.

    Properties:
        - $n \\times n$ matrix shape
        - Parameters stored as full matrix in row-major order
    """

    @classmethod
    def is_positive_definite(cls, shape: tuple[int, int], params: Array) -> Array:
        """Check if symmetric matrix is positive definite using eigenvalues."""
        matrix = cls.to_dense(shape, params)
        eigenvals = jnp.linalg.eigvalsh(matrix)  # type: ignore
        return jnp.all(eigenvals > 0)

    @classmethod
    def inverse(cls, shape: tuple[int, int], params: Array) -> Array:
        matrix = cls.to_dense(shape, params)
        inv = jnp.linalg.inv(matrix)  # type: ignore
        return cls.from_dense(inv)

    @classmethod
    def logdet(cls, shape: tuple[int, int], params: Array) -> Array:
        matrix = cls.to_dense(shape, params)
        return jnp.linalg.slogdet(matrix)[1]  # type: ignore


class Symmetric(Square):
    """Symmetric matrix representation where $A = A^T$.

    Properties:
        - Stores only upper/lower triangular elements
        - Parameter vector contains n*(n+1)/2 elements for $n \\times n$ matrix
        - Self-transpose, real eigenvalues
    """

    @classmethod
    def transpose(cls, shape: tuple[int, int], params: Array) -> Array:
        """Symmetric matrices are self-transpose."""
        return params

    @classmethod
    def to_dense(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        matrix = jnp.zeros((n, n))
        i_upper = jnp.triu_indices(n)
        matrix = matrix.at[i_upper].set(params)
        return matrix + jnp.triu(matrix, k=1).T

    @classmethod
    def from_dense(cls, matrix: Array) -> Array:
        n = matrix.shape[0]
        i_upper = jnp.triu_indices(n)
        return matrix[i_upper]

    @classmethod
    def num_params(cls, shape: tuple[int, int]) -> int:
        n = shape[0]
        return (n * (n + 1)) // 2


class PositiveDefinite(Symmetric):
    """Symmetric positive definite matrix representation.

    A symmetric matrix $A$ is positive definite if and only if:

    1. $x^T A x > 0$ for all $x \\neq 0$
    2. All eigenvalues $\\lambda_i > 0$
    3. Unique Cholesky decomposition exists: $A = LL^T$
    4. All leading principal minors are positive

    Properties:
        - Inverse is also positive definite
        - Determinant is positive
        - Stable and efficient algorithms via Cholesky factorization
    """

    @classmethod
    def is_positive_definite(cls, shape: tuple[int, int], params: Array) -> Array:
        """Check positive definiteness via Cholesky decomposition.

        Uses Cholesky attempt as a numerically stable way to check positive
        definiteness. More efficient than computing eigenvalues.
        """
        matrix = cls.to_dense(shape, params)
        # Use linalg_ops.cholesky which returns NaN on failure
        chol = jax.lax.linalg.cholesky(matrix)
        return jnp.all(jnp.isfinite(chol))

    @classmethod
    def inverse(cls, shape: tuple[int, int], params: Array) -> Array:
        """Inverse via Cholesky decomposition."""
        chol = cls.cholesky(shape, params)
        n = shape[0]
        eye = jnp.eye(n)
        # Solve L L^T x = I
        inv_chol = jax.scipy.linalg.solve_triangular(chol, eye, lower=True)
        inv = inv_chol.T @ inv_chol
        return cls.from_dense(inv)

    @classmethod
    def logdet(cls, shape: tuple[int, int], params: Array) -> Array:
        """Log determinant via Cholesky."""
        chol = cls.cholesky(shape, params)
        return 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))

    @classmethod
    def cholesky(cls, shape: tuple[int, int], params: Array) -> Array:
        """Compute lower triangular Cholesky factor L where A = LL^T."""
        matrix = cls.to_dense(shape, params)
        return jnp.linalg.cholesky(matrix)  # type: ignore

    @classmethod
    def apply_cholesky(
        cls, shape: tuple[int, int], params: Array, vector: Array
    ) -> Array:
        chol = cls.cholesky(shape, params)
        return (chol @ vector.T).T


class Diagonal(PositiveDefinite):
    """Diagonal matrix representation $A = \\text{diag}(a_1, ..., a_n)$.

    Properties:
        - Only diagonal elements stored, zero elsewhere
        - Most operations reduce to element-wise operations on diagonal
        - $O(n)$ storage and operations
    """

    @classmethod
    def is_positive_definite(cls, shape: tuple[int, int], params: Array) -> Array:
        """Check if all diagonal elements are positive."""
        return jnp.all(params > 0)

    @classmethod
    def matvec(cls, shape: tuple[int, int], params: Array, vector: Array) -> Array:
        return params * vector

    @classmethod
    def transpose(cls, shape: tuple[int, int], params: Array) -> Array:
        return params

    @classmethod
    def to_dense(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        matrix = jnp.zeros((n, n))
        return matrix.at[jnp.diag_indices(n)].set(params)

    @classmethod
    def from_dense(cls, matrix: Array) -> Array:
        return jnp.diag(matrix)

    @classmethod
    def num_params(cls, shape: tuple[int, int]) -> int:
        return shape[0]

    @classmethod
    def inverse(cls, shape: tuple[int, int], params: Array) -> Array:
        return 1.0 / params

    @classmethod
    def logdet(cls, shape: tuple[int, int], params: Array) -> Array:
        return jnp.sum(jnp.log(params))

    @classmethod
    def cholesky(cls, shape: tuple[int, int], params: Array) -> Array:
        return jnp.sqrt(params)

    @classmethod
    def outer_product(cls, v1: Array, v2: Array) -> Array:
        """Create parameters from outer product, keeping only diagonal."""
        return v1 * v2

    @classmethod
    def apply_cholesky(
        cls, shape: tuple[int, int], params: Array, vector: Array
    ) -> Array:
        return jnp.sqrt(params) * vector

    @classmethod
    def map_diagonal(
        cls, shape: tuple[int, int], params: Array, f: Callable[[Array], Array]
    ) -> Array:
        return f(params)


class Scale(Diagonal):
    """Scale transformation $A = \\alpha I$.

    Properties:
        - Single parameter $\\alpha$ represents uniform scaling
        - All operations are $O(1)$ in storage
        - Matrix operations reduce to scalar operations on $\\alpha$
    """

    @classmethod
    def is_positive_definite(cls, shape: tuple[int, int], params: Array) -> Array:
        """Check if scale factor is positive."""
        return params[0] > 0

    @classmethod
    def matvec(cls, shape: tuple[int, int], params: Array, vector: Array) -> Array:
        return params[0] * vector

    @classmethod
    def to_dense(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        return params[0] * jnp.eye(n)

    @classmethod
    def from_dense(cls, matrix: Array) -> Array:
        return jnp.array([jnp.mean(jnp.diag(matrix))])

    @classmethod
    def num_params(cls, shape: tuple[int, int]) -> int:
        return 1

    @classmethod
    def logdet(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        return n * jnp.log(params[0])

    @classmethod
    def outer_product(cls, v1: Array, v2: Array) -> Array:
        """Average outer product to single scale parameter."""
        return jnp.array([jnp.mean(v1 * v2)])

    @classmethod
    def map_diagonal(
        cls, shape: tuple[int, int], params: Array, f: Callable[[Array], Array]
    ) -> Array:
        return jnp.array([f(params[0])])


class Identity(Scale):
    """Identity transformation $A = I$.

    Properties:
        - Zero parameters - fully determined by shape
        - All operations are $O(1)$ and parameter-free
        - Acts as multiplicative identity in composition
    """

    @classmethod
    def matvec(cls, shape: tuple[int, int], params: Array, vector: Array) -> Array:
        return vector

    @classmethod
    def is_positive_definite(cls, shape: tuple[int, int], params: Array) -> Array:
        """Identity is always positive definite."""
        return jnp.array(True)

    @classmethod
    def to_dense(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        return jnp.eye(n)

    @classmethod
    def from_dense(cls, matrix: Array) -> Array:
        _ = matrix
        return jnp.array([])

    @classmethod
    def num_params(cls, shape: tuple[int, int]) -> int:
        return 0

    @classmethod
    def inverse(cls, shape: tuple[int, int], params: Array) -> Array:
        return params

    @classmethod
    def logdet(cls, shape: tuple[int, int], params: Array) -> Array:
        return jnp.array(0.0)

    @classmethod
    def outer_product(cls, v1: Array, v2: Array) -> Array:
        """Identity ignores input vectors."""
        return jnp.array([])

    @classmethod
    def apply_cholesky(
        cls, shape: tuple[int, int], params: Array, vector: Array
    ) -> Array:
        return vector
