"""Storage-efficient matrix representations for use as strategies in linear maps.

A ``MatrixRep`` defines how to store a matrix as a flat parameter array and how to perform linear algebra (matvec, transpose, inverse, etc.) while respecting structural constraints. ``EmbeddedMap`` in ``linear.py`` plugs a rep into the manifold system; this module is purely about the matrix operations themselves.

The hierarchy from most to least general is::

    Rectangular > Square > Symmetric > PositiveDefinite > Diagonal > Scale > Identity

Each level exploits additional structure for cheaper storage and operations:

+----------------+---------------+---------------+---------------+
| Representation | Storage       | Matmul        | Inverse/Det   |
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

TODO: A ``Convolutional`` rep could fit naturally here. It would store a kernel and
implement ``matvec`` via convolution on a flat array (i.e. a compactly-stored Toeplitz
matrix), with ``shape = (output_len, input_len)`` preserving the existing contract.
Multi-channel and 2D structure would be handled in ``linear.py`` via ``BlockMap``
(one block per channel pair) or embeddings that reshape between flat and spatial layouts.
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
    """Compute matrix-matrix product, choosing the tightest representation for the result.

    Dispatches on both operands' representations: Identity and Scale short-circuit,
    Diagonal * Diagonal stays Diagonal, and mixed cases fall back to dense multiplication
    with the result stored as Square or Rectangular based on output shape.
    """
    out_shape = (shape[0], right_shape[1])
    out_rep: MatrixRep
    out_params: Array
    match rep:
        case Identity():
            # Identity * B = B, so result has same rep as right matrix
            out_rep = right_rep
            out_params = right_params
        case Scale():
            # Scale (aI) * B = aB, so just scale the right matrix parameters
            out_rep = right_rep
            out_params = params[0] * right_params
        case Diagonal():
            # Diagonal multiplication: D * X
            match right_rep:
                case Diagonal():
                    # Diagonal * Diagonal = Diagonal (element-wise product of diagonals)
                    out_rep = Diagonal()
                    out_params = params * right_params
                case _:
                    # Diagonal * (non-Diagonal): result is dense
                    # Convert right matrix to dense and multiply: diag(params) @ right_dense
                    # Broadcasting: params[:, None] * right_dense treats params as column vector
                    right_dense = right_rep.to_matrix(right_shape, right_params)
                    out_dense = params[:, None] * right_dense
                    # Determine output representation based on output shape
                    # If output is square, use Square; otherwise use Rectangular
                    out_rep = (
                        Square() if out_shape[0] == out_shape[1] else Rectangular()
                    )
                    out_params = out_rep.from_matrix(out_dense)
        case _:
            # General matrix (Rectangular/Square/Symmetric/PositiveDefinite) multiplication
            match right_rep:
                case Identity() | Scale() | Diagonal():
                    # A * S = (S * A^T)^T — S is symmetric so S^T = S
                    left_params_t = rep.transpose(shape, params)
                    left_shape_t = (shape[1], shape[0])
                    result_rep, result_shape, result_params = _matmat(
                        right_rep, right_shape, right_params, rep, left_shape_t, left_params_t
                    )
                    result_params_t = result_rep.transpose(result_shape, result_params)
                    result_shape_t = (result_shape[1], result_shape[0])
                    return result_rep, result_shape_t, result_params_t
                case _:
                    # Both general matrices: fall back to dense multiplication
                    # Determine output representation based on output shape:
                    # - If m == p, result is Square (can support additional operations)
                    # - Otherwise, result is Rectangular (general m x p matrix)
                    left_dense = rep.to_matrix(shape, params)
                    right_dense = right_rep.to_matrix(right_shape, right_params)
                    out_rep = (
                        Square() if out_shape[0] == out_shape[1] else Rectangular()
                    )
                    out_dense = left_dense @ right_dense
                    out_params = out_rep.from_matrix(out_dense)
    return out_rep, out_shape, out_params


def _diag_indices_in_triangular(n: int) -> Array:
    """Return indices of diagonal elements in upper triangular storage.

    For an $n \times n$ matrix stored in upper triangular format $(n(n+1)/2)$,
    returns the indices where diagonal elements are stored.
    """
    i_diag = jnp.arange(n)
    return i_diag * (2 * n - i_diag + 1) // 2


class MatrixRep(ABC):
    """Strategy interface for matrix storage and operations.

    Each subclass defines how to pack a matrix into a flat 1D parameter array and how to perform linear algebra (matvec, transpose, outer product, etc.) while preserving the structural constraint (symmetry, diagonal, etc.). Representations are stateless --- two instances of the same class are equal.

    The ``embed_params`` / ``project_params`` methods convert between representations by walking the linear inheritance chain (e.g. Diagonal -> PositiveDefinite -> Symmetric -> Square -> Rectangular).
    """

    @override
    def __eq__(self, other: object) -> bool:
        """Compare matrix representations by type (not instance identity).

        Since MatrixRep instances are stateless, two instances are equal if they
        have the same class type.
        """
        return type(self) is type(other)

    @override
    def __hash__(self) -> int:
        """Hash by class type to maintain hash/equality contract."""
        return hash(type(self))

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
    def to_matrix(cls, shape: tuple[int, int], params: Array) -> Array:
        """Convert 1D parameters to dense matrix form."""

    @classmethod
    @abstractmethod
    def num_params(cls, shape: tuple[int, int]) -> int:
        """Shape of 1D parameter array needed for matrix dimensions."""

    @classmethod
    @abstractmethod
    def from_matrix(cls, matrix: Array) -> Array:
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
    def get_diagonal(cls, shape: tuple[int, int], params: Array) -> Array:
        """Extract diagonal elements from the matrix."""
        return jnp.diag(cls.to_matrix(shape, params))

    @classmethod
    def embed_params(
        cls, shape: tuple[int, int], params: Array, target_rep: MatrixRep
    ) -> Array:
        """Recursively embed params into more complex representation."""
        if not issubclass(cls, type(target_rep)):
            raise TypeError(f"Cannot embed {cls} into {target_rep}")

        cur_rep = cls
        cur_params = params
        while cur_rep is not type(target_rep):
            cur_params = cur_rep.embed_in_super(shape, cur_params)
            cur_rep: type[MatrixRep] = cur_rep.__base__  # pyright: ignore[reportAssignmentType]
        return cur_params

    @classmethod
    def project_params(
        cls, shape: tuple[int, int], params: Array, target_rep: MatrixRep
    ) -> Array:
        """Recursively project params to simpler representation."""
        if not issubclass(type(target_rep), cls):
            raise TypeError(f"Cannot project {cls} to {target_rep}")

        # Build path of representations from target up to cls
        path: list[type[MatrixRep]] = []
        cur_rep: type[MatrixRep] = type(target_rep)
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
    """Full $m \\times n$ matrix, stored in row-major order. No structural constraints."""

    @classmethod
    @override
    def matvec(cls, shape: tuple[int, int], params: Array, vector: Array) -> Array:
        matrix = cls.to_matrix(shape, params)
        return jnp.dot(matrix, vector)

    @classmethod
    @override
    def transpose(cls, shape: tuple[int, int], params: Array) -> Array:
        matrix = cls.to_matrix(shape, params).T
        return matrix.reshape(-1)

    @classmethod
    @override
    def to_matrix(cls, shape: tuple[int, int], params: Array) -> Array:
        return params.reshape(shape)

    @classmethod
    @override
    def from_matrix(cls, matrix: Array) -> Array:
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
        return cls.from_matrix(matrix)

    @classmethod
    @override
    def map_diagonal(
        cls, shape: tuple[int, int], params: Array, f: Callable[[Array], Array]
    ) -> Array:
        """Map function over diagonal elements of matrix."""
        matrix = cls.to_matrix(shape, params)
        diag = jnp.diag(matrix)
        new_diag = f(diag)
        new_matrix = matrix.at[jnp.diag_indices(shape[0])].set(new_diag)
        return cls.from_matrix(new_matrix)

    @classmethod
    @override
    def embed_in_super(cls, shape: tuple[int, int], params: Array) -> Array:
        raise TypeError("Rectangular is most complex representation")

    @classmethod
    @override
    def project_from_super(cls, shape: tuple[int, int], params: Array) -> Array:
        raise TypeError("No more complex rep to project from")


class Square(Rectangular):
    """Square $n \\times n$ matrix, adding inverse, determinant, and positive-definiteness checks."""

    @classmethod
    def is_positive_definite(cls, shape: tuple[int, int], params: Array) -> Array:
        """Check if symmetric matrix is positive definite using eigenvalues."""
        matrix = cls.to_matrix(shape, params)
        eigenvals = jnp.linalg.eigvalsh(matrix)
        return jnp.all(eigenvals > 0)

    @classmethod
    def inverse(cls, shape: tuple[int, int], params: Array) -> Array:
        matrix = cls.to_matrix(shape, params)
        inv = jnp.linalg.inv(matrix)
        return cls.from_matrix(inv)

    @classmethod
    def logdet(cls, shape: tuple[int, int], params: Array) -> Array:
        matrix = cls.to_matrix(shape, params)
        return jnp.linalg.slogdet(matrix)[1]

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
    """Symmetric matrix ($A = A^T$), stored as upper-triangular elements. Roughly half the storage of a full square matrix."""

    @classmethod
    @override
    def transpose(cls, shape: tuple[int, int], params: Array) -> Array:
        """Symmetric matrices are self-transpose."""
        return params

    @classmethod
    @override
    def to_matrix(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        matrix = jnp.zeros((n, n))
        i_upper = jnp.triu_indices(n)
        matrix = matrix.at[i_upper].set(params)
        return matrix + jnp.triu(matrix, k=1).T

    @classmethod
    @override
    def from_matrix(cls, matrix: Array) -> Array:
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
    def get_diagonal(cls, shape: tuple[int, int], params: Array) -> Array:
        """Extract diagonal from packed upper-triangular storage in O(n)."""
        return params[_diag_indices_in_triangular(shape[0])]

    @classmethod
    @override
    def map_diagonal(
        cls, shape: tuple[int, int], params: Array, f: Callable[[Array], Array]
    ) -> Array:
        """Apply f to diagonal elements in packed upper-triangular storage in O(n)."""
        diag_idx = _diag_indices_in_triangular(shape[0])
        return params.at[diag_idx].set(f(params[diag_idx]))

    @classmethod
    @override
    def embed_in_super(cls, shape: tuple[int, int], params: Array) -> Array:
        """Convert upper triangular parameters to full square matrix parameters."""
        # To Square means including each off-diagonal element twice
        matrix = cls.to_matrix(shape, params)
        return matrix.reshape(-1)

    @classmethod
    @override
    def project_from_super(cls, shape: tuple[int, int], params: Array) -> Array:
        """Extract upper triangular parameters from full square matrix parameters."""
        n = shape[0]
        matrix = params.reshape(n, n)
        return cls.from_matrix(matrix)


class PositiveDefinite(Symmetric):
    """Symmetric positive-definite matrix, using Cholesky decomposition for stable inverse and log-determinant.

    Mathematically, $A$ is positive definite iff $x^T A x > 0$ for all $x \\neq 0$, equivalently iff a unique Cholesky factorization $A = LL^T$ exists.
    """

    @classmethod
    def _cholesky(cls, shape: tuple[int, int], params: Array) -> Array:
        matrix = cls.to_matrix(shape, params)
        return jnp.linalg.cholesky(matrix)

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

        Transforms the first Gaussian into a coordinate system in which the second Gaussian
        standard normal N(0, I). This is useful for computing KL divergences
        and other relative measures between Gaussians.

        Perform the transformation:
        - new_mean = L^(-1) @ (mean1 - mean2)
        - new_params = L^(-1) @ params1 @ L^(-T)

        Where L is the Cholesky factor of params2 such that L @ L.T = params2.
        """

        # Get dense matrices
        matrix2 = cls.to_matrix(shape, params2)
        matrix1 = cls.to_matrix(shape, params1)

        # Compute Cholesky and transform mean
        chol = jnp.linalg.cholesky(matrix2)
        centered_mean = mean1 - mean2
        whitened_mean = jax.scipy.linalg.solve_triangular(
            chol, centered_mean, lower=True
        )

        # Transform covariance
        temp = jax.scipy.linalg.solve_triangular(chol, matrix1, lower=True)
        whitened_matrix = jax.scipy.linalg.solve_triangular(chol, temp.T, lower=True).T

        return whitened_mean, cls.from_matrix(whitened_matrix)

    @classmethod
    @override
    def is_positive_definite(cls, shape: tuple[int, int], params: Array) -> Array:
        """Check positive definiteness via Cholesky decomposition."""
        matrix = cls.to_matrix(shape, params)
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
        return cls.from_matrix(inv)

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
    """Diagonal matrix $A = \\text{diag}(a_1, \\ldots, a_n)$, storing only the $n$ diagonal entries.

    All operations reduce to element-wise arithmetic, giving $O(n)$ storage and compute.
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
    def to_matrix(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        matrix = jnp.zeros((n, n))
        return matrix.at[jnp.diag_indices(n)].set(params)

    @classmethod
    @override
    def from_matrix(cls, matrix: Array) -> Array:
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
    def get_diagonal(cls, shape: tuple[int, int], params: Array) -> Array:
        return params

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
    """Scalar multiple of the identity, $A = \\alpha I$. Single parameter, $O(1)$ storage."""

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
    def to_matrix(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        return params[0] * jnp.eye(n)

    @classmethod
    @override
    def from_matrix(cls, matrix: Array) -> Array:
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
    def get_diagonal(cls, shape: tuple[int, int], params: Array) -> Array:
        return jnp.full(shape[0], params[0])

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
    """The identity matrix $A = I$. Zero parameters --- fully determined by shape."""

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
    def to_matrix(cls, shape: tuple[int, int], params: Array) -> Array:
        n = shape[0]
        return jnp.eye(n)

    @classmethod
    @override
    def from_matrix(cls, matrix: Array) -> Array:
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
        # Identity has no parameters; whitening is just centering the mean
        return mean1 - mean2, params1

    @classmethod
    @override
    def get_diagonal(cls, shape: tuple[int, int], params: Array) -> Array:
        return jnp.ones(shape[0])

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
