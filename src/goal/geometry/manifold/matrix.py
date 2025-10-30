"""This module provides a hierarchy of matrix representations with increasingly specialized structure, providing
    - Progressive specialization from general to highly constrained structures,
    - storage-efficient representations for special matrix types,
    - optimized implementations of key operations like multiplication and inversion, and
    - Type-safe handling of different matrix constraints (symmetry, positive-definiteness).

The following table provides an overview of the storage and computational complexity of each matrix representation:

+----------------+---------------+---------------+---------------+
| Operator Type  | Storage       | Matmul        | Inverse/Det   |
+================+===============+===============+===============+
| Rectangular    | $O(n^2)$      | $O(n^2)$      | $O(n^3)$      |
+----------------+---------------+---------------+---------------+
| Symmetric      | $O(n^2/2)$    | $O(n^2)$      | $O(n^3)$      |
+----------------+---------------+---------------+---------------+
| Pos. Definite  | $O(n^2/2)$    | $O(n^2)$      | $O(n^3)$      |
|                |               |               | (Cholesky)    |
+----------------+---------------+---------------+---------------+
| Diagonal       | $O(n)$        | $O(n)$        | $O(n)$        |
+----------------+---------------+---------------+---------------+
| Scale          | $O(1)$        | $O(n)$        | $O(1)$        |
+----------------+---------------+---------------+---------------+
| Identity       | $O(1)$        | $O(1)$        | $O(1)$        |
+----------------+---------------+---------------+---------------+
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import override

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


def _diag_indices_in_triangular(n: int) -> Array:
    """Return indices of diagonal elements in upper triangular storage.

    For an $n \times n$ matrix stored in upper triangular format $(n(n+1)/2)$,
    returns the indices where diagonal elements are stored.
    """
    i_diag = jnp.arange(n)
    return i_diag * (2 * n - i_diag + 1) // 2


class MatrixRep(ABC):
    """MatrixRep acts as a strategy pattern for matrix operations, allowing different representations (full, symmetric, diagonal) to share the same interface while using optimized implementations. It defines operations that maintain the specific structure of each matrix type.

    In theory, each subclass corresponds to a specific matrix structure with constraints on the entries, and implements the corresponding linear algebraic operations while preserving those constraints.

    All matrix parameters are stored as 1D arrays for compatibility with Point. The flattened representation enables unified parameter handling across different matrix structures, while each operation respects the specific matrix type's constraints.
    """

    @classmethod
    @abstractmethod
    def matvec(cls, shape: tuple[int, int], params: Array, vector: Array) -> Array:
        """Matrix-vector multiplication."""

    @classmethod
    def matmat(
        cls,
        shape: tuple[int, int],
        params: Array,
        right_rep: MatrixRep,
        right_shape: tuple[int, int],
        right_params: Array,
    ) -> tuple[MatrixRep, tuple[int, int], Array]:
        """Multiply matrices, returning optimal representation type and parameters."""
        return _matmat(cls(), shape, params, right_rep, right_shape, right_params)

    @classmethod
    @abstractmethod
    def transpose(cls, shape: tuple[int, int], params: Array) -> Array:
        """Transform parameters to represent the transposed matrix."""

    @classmethod
    @abstractmethod
    def to_dense(cls, shape: tuple[int, int], params: Array) -> Array:
        """Convert 1D parameters to dense matrix form."""

    @classmethod
    @abstractmethod
    def num_params(cls, shape: tuple[int, int]) -> int:
        """Shape of 1D parameter array needed for matrix dimensions."""

    @classmethod
    @abstractmethod
    def from_dense(cls, matrix: Array) -> Array:
        """Convert dense matrix to 1D parameters."""

    @classmethod
    @abstractmethod
    def outer_product(cls, v1: Array, v2: Array) -> Array:
        """Construct parameters from outer product $v_1 \\otimes v_2$."""

    @classmethod
    @abstractmethod
    def map_diagonal(
        cls, shape: tuple[int, int], params: Array, f: Callable[[Array], Array]
    ) -> Array:
        """Apply function f to diagonal elements while preserving matrix structure."""

    @classmethod
    def embed_params(
        cls, shape: tuple[int, int], params: Array, target_rep: type[MatrixRep]
    ) -> Array:
        """Recursively embed params into more complex representation."""
        if not issubclass(cls, target_rep):
            raise TypeError(f"Cannot embed {cls} into {target_rep}")

        cur_rep = cls
        cur_params = params
        while cur_rep is not target_rep:
            cur_params = cur_rep.embed_in_super(shape, cur_params)
            cur_rep: type[MatrixRep] = cur_rep.__base__  # pyright: ignore[reportAssignmentType]
        return cur_params

    @classmethod
    def project_params(
        cls, shape: tuple[int, int], params: Array, target_rep: type[MatrixRep]
    ) -> Array:
        """Recursively project params to simpler representation."""
        if not issubclass(target_rep, cls):
            raise TypeError(f"Cannot project {cls} to {target_rep}")

        # Build path of representations from target up to cls
        path: list[type[MatrixRep]] = []
        cur_rep: type[MatrixRep] = target_rep
        while cur_rep != cls:
            path.append(cur_rep)
            cur_rep = cur_rep.__base__  # pyright: ignore[reportAssignmentType]
        path = list(reversed(path))

        # Project through path
        cur_params = params
        for sub_rep in path:
            cur_params = sub_rep.project_from_super(shape, cur_params)
        return cur_params

    @classmethod
    @abstractmethod
    def embed_in_super(cls, shape: tuple[int, int], params: Array) -> Array:
        """Embed parameters into immediate parent representation."""

    @classmethod
    @abstractmethod
    def project_from_super(cls, shape: tuple[int, int], params: Array) -> Array:
        """Project parameters from immediate parent representation."""


class Rectangular(MatrixRep):
    """Rectangular stores all $m \\times n$ entries explicitly. This representation supports arbitrary linear transformations without constraints.

    In theory, this represents the full space of linear maps from $\\mathbb{R}^n$ to $\\mathbb{R}^m$. It requires $mn$ parameters for storage.

    Properties:
        - Full $m \\times n$ matrix representation
        - $O(mn)$ parameters stored in row-major order
    """

    @classmethod
    @override
    def matvec(cls, shape: tuple[int, int], params: Array, vector: Array) -> Array:
        matrix = cls.to_dense(shape, params)
        return jnp.dot(matrix, vector)

    @classmethod
    @override
    def transpose(cls, shape: tuple[int, int], params: Array) -> Array:
        matrix = cls.to_dense(shape, params).T
        return matrix.reshape(-1)

    @classmethod
    @override
    def to_dense(cls, shape: tuple[int, int], params: Array) -> Array:
        return params.reshape(shape)

    @classmethod
    @override
    def from_dense(cls, matrix: Array) -> Array:
        return matrix.reshape(-1)

    @classmethod
    @override
    def num_params(cls, shape: tuple[int, int]) -> int:
        n, m = shape
        return n * m

    @classmethod
    @override
    def outer_product(cls, v1: Array, v2: Array) -> Array:
        """Create parameters from outer product."""
        matrix = jnp.outer(v1, v2)
        return cls.from_dense(matrix)

    @classmethod
    @override
    def map_diagonal(
        cls, shape: tuple[int, int], params: Array, f: Callable[[Array], Array]
    ) -> Array:
        """Map function over diagonal elements of matrix."""
        matrix = cls.to_dense(shape, params)
        diag = jnp.diag(matrix)
        new_diag = f(diag)
        new_matrix = matrix.at[jnp.diag_indices(shape[0])].set(new_diag)
        return cls.from_dense(new_matrix)

    @classmethod
    @override
    def embed_in_super(cls, shape: tuple[int, int], params: Array) -> Array:
        raise TypeError("Rectangular is most complex representation")

    @classmethod
    @override
    def project_from_super(cls, shape: tuple[int, int], params: Array) -> Array:
        raise TypeError("No more complex rep to project from")


class Square(Rectangular):
    """Square matrices enable additional operations like determinants and inverses.

    Properties:
        - $n \\times n$ matrix shape (same dimension for rows and columns)
        - Supports determinant and inverse operations
        - Forms an algebra under matrix multiplication
        - Storage: $O(n^2)$ parameters
    """

    @classmethod
    def is_positive_definite(cls, shape: tuple[int, int], params: Array) -> Array:
        """Check if symmetric matrix is positive definite using eigenvalues."""
        matrix = cls.to_dense(shape, params)
        eigenvals = jnp.linalg.eigvalsh(matrix)  # pyright: ignore[reportUnknownVariableType]
        return jnp.all(eigenvals > 0)

    @classmethod
    def inverse(cls, shape: tuple[int, int], params: Array) -> Array:
        matrix = cls.to_dense(shape, params)
        inv = jnp.linalg.inv(matrix)  # pyright: ignore[reportUnknownVariableType]
        return cls.from_dense(inv)

    @classmethod
    def logdet(cls, shape: tuple[int, int], params: Array) -> Array:
        matrix = cls.to_dense(shape, params)
        return jnp.linalg.slogdet(matrix)[1]  # pyright: ignore[reportUnknownVariableType]

    @classmethod
    @override
    def embed_in_super(cls, shape: tuple[int, int], params: Array) -> Array:
        return params  # Square to Rectangular is identity

    @classmethod
    @override
    def project_from_super(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        return params.reshape(n, n).reshape(-1)


class Symmetric(Square):
    """Symmetric matrices require roughly half the storage of general square matrices and arise naturally in many statistical applications like covariance matrices. In theory, symmetric matrices satisfy $A = A^T$ and have a complete basis of orthogonal eigenvectors with real eigenvalues.

    Properties:
        - Self-transpose with guaranteed real eigenvalues
        - Orthogonally diagonalizable
        - Stores only upper/lower triangular elements
        - Storage: $O(n^2/2)$ parameters
    """

    @classmethod
    @override
    def transpose(cls, shape: tuple[int, int], params: Array) -> Array:
        """Symmetric matrices are self-transpose."""
        return params

    @classmethod
    @override
    def to_dense(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        matrix = jnp.zeros((n, n))
        i_upper = jnp.triu_indices(n)
        matrix = matrix.at[i_upper].set(params)
        return matrix + jnp.triu(matrix, k=1).T

    @classmethod
    @override
    def from_dense(cls, matrix: Array) -> Array:
        n = matrix.shape[0]
        i_upper = jnp.triu_indices(n)
        return matrix[i_upper]

    @classmethod
    @override
    def num_params(cls, shape: tuple[int, int]) -> int:
        n = shape[0]
        return (n * (n + 1)) // 2

    @classmethod
    @override
    def embed_in_super(cls, shape: tuple[int, int], params: Array) -> Array:
        """Convert upper triangular parameters to full square matrix parameters."""
        # To Square means including each off-diagonal element twice
        matrix = cls.to_dense(shape, params)
        return matrix.reshape(-1)

    @classmethod
    @override
    def project_from_super(cls, shape: tuple[int, int], params: Array) -> Array:
        """Extract upper triangular parameters from full square matrix parameters."""
        n = shape[0]
        matrix = params.reshape(n, n)
        return cls.from_dense(matrix)


class PositiveDefinite(Symmetric):
    """PositiveDefinite matrices enable numerically stable operations through Cholesky decomposition and arise naturally in statistical models as covariance matrices. They ensure that quadratic forms like $x^TAx$ represent valid variance-like quantities.

    A symmetric matrix $A$ is positive definite if and only if any of these equivalent conditions hold:
        1. $x^T A x > 0$ for all $x \\neq 0$
        2. All eigenvalues $\\lambda_i > 0$
        3. Unique Cholesky decomposition exists: $A = LL^T$

    Properties:
        - Inverse is also positive definite
        - Determinant is positive
        - Stable and efficient algorithms via Cholesky factorization
        - Forms a convex cone (not a vector space)
        - Storage: $O(n^2/2)$ parameters
    """

    @classmethod
    def _cholesky(cls, shape: tuple[int, int], params: Array) -> Array:
        matrix = cls.to_dense(shape, params)
        return jnp.linalg.cholesky(matrix)  # pyright: ignore[reportUnknownVariableType]

    @classmethod
    def cholesky_matvec(
        cls, shape: tuple[int, int], params: Array, vector: Array
    ) -> Array:
        """Compute cholesky factorization and apply to vector or batch thereof."""
        chol = cls._cholesky(shape, params)
        # Handle both single vectors and batches
        if vector.ndim == 1:
            return chol @ vector
        return (chol @ vector.T).T

    @classmethod
    def cholesky_whiten(
        cls,
        shape: tuple[int, int],
        mean1: Array,
        params1: Array,
        mean2: Array,
        params2: Array,
    ) -> tuple[Array, Array]:
        """Whiten a distribution (mean1, params1) with respect to another (mean2, params2).

        Perform the transformation:
        - new_mean = L^(-1) @ (mean1 - mean2)
        - new_params = L^(-1) @ params1 @ L^(-T)

        Where L is the Cholesky factor of params2 such that L @ L.T = params2.
        """

        # Get dense matrices
        matrix2 = cls.to_dense(shape, params2)
        matrix1 = cls.to_dense(shape, params1)

        # Compute Cholesky and transform mean
        chol = jnp.linalg.cholesky(matrix2)  # pyright: ignore[reportUnknownVariableType]
        centered_mean = mean1 - mean2
        whitened_mean = jax.scipy.linalg.solve_triangular(
            chol, centered_mean, lower=True
        )

        # Transform covariance
        temp = jax.scipy.linalg.solve_triangular(chol, matrix1, lower=True)
        whitened_matrix = jax.scipy.linalg.solve_triangular(chol, temp.T, lower=True).T

        return whitened_mean, cls.from_dense(whitened_matrix)

    @classmethod
    @override
    def is_positive_definite(cls, shape: tuple[int, int], params: Array) -> Array:
        """Check positive definiteness via Cholesky decomposition."""
        matrix = cls.to_dense(shape, params)
        # Use linalg_ops.cholesky which returns NaN on failure
        chol = jax.lax.linalg.cholesky(matrix)
        return jnp.all(jnp.isfinite(chol))

    @classmethod
    @override
    def inverse(cls, shape: tuple[int, int], params: Array) -> Array:
        """Inverse via Cholesky decomposition."""
        chol = cls._cholesky(shape, params)
        n = shape[0]
        eye = jnp.eye(n)
        # Solve L L^T x = I
        inv_chol = jax.scipy.linalg.solve_triangular(chol, eye, lower=True)
        inv = inv_chol.T @ inv_chol
        return cls.from_dense(inv)

    @classmethod
    @override
    def logdet(cls, shape: tuple[int, int], params: Array) -> Array:
        """Log determinant via Cholesky."""
        chol = cls._cholesky(shape, params)
        return 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))

    @classmethod
    @override
    def embed_in_super(cls, shape: tuple[int, int], params: Array) -> Array:
        return params

    @classmethod
    @override
    def project_from_super(cls, shape: tuple[int, int], params: Array) -> Array:
        return params


class Diagonal(PositiveDefinite):
    """Diagonal matrix representation $A = \\text{diag}(a_1, ..., a_n)$. Most operations with Diagonal matrices become $O(n)$ instead of $O(n^2)$ or $O(n^3)$.

    Properties:
        - Only diagonal elements stored, zero elsewhere
        - Most operations reduce to element-wise operations on diagonal
        - Eigenvalues are the diagonal elements
        - Forms a commutative subalgebra (AB = BA for diagonal matrices)
        - Storage and operations: $O(n)$ complexity
    """

    @classmethod
    @override
    def is_positive_definite(cls, shape: tuple[int, int], params: Array) -> Array:
        """Check if all diagonal elements are positive."""
        return jnp.all(params > 0)

    @classmethod
    @override
    def matvec(cls, shape: tuple[int, int], params: Array, vector: Array) -> Array:
        return params * vector

    @classmethod
    @override
    def transpose(cls, shape: tuple[int, int], params: Array) -> Array:
        return params

    @classmethod
    @override
    def to_dense(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        matrix = jnp.zeros((n, n))
        return matrix.at[jnp.diag_indices(n)].set(params)

    @classmethod
    @override
    def from_dense(cls, matrix: Array) -> Array:
        return jnp.diag(matrix)

    @classmethod
    @override
    def num_params(cls, shape: tuple[int, int]) -> int:
        return shape[0]

    @classmethod
    @override
    def inverse(cls, shape: tuple[int, int], params: Array) -> Array:
        return 1.0 / params

    @classmethod
    @override
    def logdet(cls, shape: tuple[int, int], params: Array) -> Array:
        return jnp.sum(jnp.log(params))

    @classmethod
    @override
    def _cholesky(cls, shape: tuple[int, int], params: Array) -> Array:
        return jnp.sqrt(params)

    @classmethod
    @override
    def outer_product(cls, v1: Array, v2: Array) -> Array:
        """Create parameters from outer product, keeping only diagonal."""
        return v1 * v2

    @classmethod
    @override
    def cholesky_matvec(
        cls, shape: tuple[int, int], params: Array, vector: Array
    ) -> Array:
        return vector * jnp.sqrt(params)

    # In class Diagonal
    @classmethod
    @override
    def cholesky_whiten(
        cls,
        shape: tuple[int, int],
        mean1: Array,
        params1: Array,
        mean2: Array,
        params2: Array,
    ) -> tuple[Array, Array]:
        # For diagonal matrices, Cholesky is just sqrt of diagonal elements
        # So whitening is just division by sqrt(params2)
        sqrt_params2 = jnp.sqrt(params2)
        whitened_mean = (mean1 - mean2) / sqrt_params2
        whitened_params = params1 / params2

        return whitened_mean, whitened_params

    @classmethod
    @override
    def map_diagonal(
        cls, shape: tuple[int, int], params: Array, f: Callable[[Array], Array]
    ) -> Array:
        return f(params)

    @classmethod
    @override
    def embed_in_super(cls, shape: tuple[int, int], params: Array) -> Array:
        """Put diagonal elements into upper triangular format."""
        n = shape[0]
        out_params = jnp.zeros(PositiveDefinite.num_params(shape))
        diag_indices = _diag_indices_in_triangular(n)
        return out_params.at[diag_indices].set(params)

    @classmethod
    @override
    def project_from_super(cls, shape: tuple[int, int], params: Array) -> Array:
        """Extract diagonal elements from upper triangular format."""
        n = shape[0]
        # Use our num_params to verify we're getting correct number of diagonal elements
        diag_indices = _diag_indices_in_triangular(n)
        return params[diag_indices]


class Scale(Diagonal):
    """Scale transformation $A = \\alpha I$.

    Properties:
        - Single parameter $\\alpha$ represents uniform scaling
        - All operations are $O(1)$ in storage
        - Matrix operations reduce to scalar operations on $\\alpha$
    """

    @classmethod
    @override
    def is_positive_definite(cls, shape: tuple[int, int], params: Array) -> Array:
        """Check if scale factor is positive."""
        return params[0] > 0

    @classmethod
    @override
    def matvec(cls, shape: tuple[int, int], params: Array, vector: Array) -> Array:
        return params[0] * vector

    @classmethod
    @override
    def to_dense(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        return params[0] * jnp.eye(n)

    @classmethod
    @override
    def from_dense(cls, matrix: Array) -> Array:
        return jnp.array([jnp.mean(jnp.diag(matrix))])

    @classmethod
    @override
    def num_params(cls, shape: tuple[int, int]) -> int:
        return 1

    @classmethod
    @override
    def logdet(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        return n * jnp.log(params[0])

    @classmethod
    @override
    def outer_product(cls, v1: Array, v2: Array) -> Array:
        """Average outer product to single scale parameter."""
        return jnp.array([jnp.mean(v1 * v2)])

    @classmethod
    @override
    def map_diagonal(
        cls, shape: tuple[int, int], params: Array, f: Callable[[Array], Array]
    ) -> Array:
        return jnp.array([f(params[0])])

    @classmethod
    @override
    def embed_in_super(cls, shape: tuple[int, int], params: Array) -> Array:
        """Expand scalar to diagonal vector."""
        n = shape[0]
        return jnp.full(n, params[0])

    @classmethod
    @override
    def project_from_super(cls, shape: tuple[int, int], params: Array) -> Array:
        """Average diagonal elements to scalar."""
        return jnp.array([jnp.mean(params)])


class Identity(Scale):
    """Identity transformation $A = I$.

    Properties:
        - Zero parameters - fully determined by shape
        - All operations are $O(1)$ and parameter-free
        - Acts as multiplicative identity in composition
    """

    @classmethod
    @override
    def matvec(cls, shape: tuple[int, int], params: Array, vector: Array) -> Array:
        return vector

    @classmethod
    @override
    def is_positive_definite(cls, shape: tuple[int, int], params: Array) -> Array:
        """Identity is always positive definite."""
        return jnp.array(True)

    @classmethod
    @override
    def to_dense(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        return jnp.eye(n)

    @classmethod
    @override
    def from_dense(cls, matrix: Array) -> Array:
        _ = matrix
        return jnp.array([])

    @classmethod
    @override
    def num_params(cls, shape: tuple[int, int]) -> int:
        return 0

    @classmethod
    @override
    def inverse(cls, shape: tuple[int, int], params: Array) -> Array:
        return params

    @classmethod
    @override
    def logdet(cls, shape: tuple[int, int], params: Array) -> Array:
        return jnp.array(0.0)

    @classmethod
    @override
    def outer_product(cls, v1: Array, v2: Array) -> Array:
        """Identity ignores input vectors."""
        return jnp.array([])

    @classmethod
    @override
    def cholesky_matvec(
        cls, shape: tuple[int, int], params: Array, vector: Array
    ) -> Array:
        return vector

    # In class Scale
    @classmethod
    @override
    def cholesky_whiten(
        cls,
        shape: tuple[int, int],
        mean1: Array,
        params1: Array,
        mean2: Array,
        params2: Array,
    ) -> tuple[Array, Array]:
        # For scale matrices, even simpler - just scalar division
        sqrt_scale = jnp.sqrt(params2[0])
        whitened_mean = (mean1 - mean2) / sqrt_scale
        whitened_params = jnp.array([params1[0] / params2[0]])

        return whitened_mean, whitened_params

    @classmethod
    @override
    def embed_in_super(cls, shape: tuple[int, int], params: Array) -> Array:
        """Empty params to unit scalar."""
        return jnp.array([1.0])

    @classmethod
    @override
    def project_from_super(cls, shape: tuple[int, int], params: Array) -> Array:
        """Scalar to empty params."""
        return jnp.array([])
