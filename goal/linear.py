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
from typing import Callable, NoReturn, Optional, Union

import jax.numpy as jnp
import numpy as np


class LinearMap(ABC):
    """Abstract base class for linear maps between vector spaces. A linear map $L: V \\mapsto W$ between vector spaces satisfies
        $L(\\alpha x + \\beta y) = \\alpha L(x) + \\beta L(y)$.

    JAX Implementation:
        - All operations return new objects (immutable)
        - Uses JAX's array operations for efficiency
        - Supports vectorization via jit/vmap
    """

    @abstractmethod
    def matvec(self, v: jnp.ndarray) -> jnp.ndarray:
        """Matrix-vector product Av."""
        pass

    @abstractmethod
    def transpose(self) -> "LinearMap":
        """Transpose of the linear map."""
        pass

    @abstractmethod
    def parameter_count(self) -> int:
        """Number of parameters needed to represent the map."""
        pass

    @abstractmethod
    def to_vector(self) -> jnp.ndarray:
        """Convert to parameter vector."""
        pass

    @abstractmethod
    def logdet(self) -> jnp.ndarray:
        """Compute log determinant."""
        pass

    def __matmul__(self, v: jnp.ndarray) -> jnp.ndarray:
        """Allow @ syntax for matrix-vector product."""
        return self.matvec(v)


@dataclass
class Square(LinearMap):
    """Arbitrary square matrices $A \\in \\mathbb R^{n \\times n}$.

    Properties:

        - LU decomposition used for determinant/inverse

    """

    matrix: jnp.ndarray

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            self.matrix = _ensure_ndarray(self.matrix)
            if self.matrix.ndim != 2:
                _raise_value_error("Matrix must be 2-dimensional")

    def matvec(self, v: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(self.matrix, v)

    def transpose(self) -> "Square":
        return Square(self.matrix.T)

    def parameter_count(self) -> int:
        return self.matrix.size

    def to_vector(self) -> jnp.ndarray:
        return self.matrix.ravel()

    def logdet(self) -> jnp.ndarray:
        return jnp.linalg.slogdet(self.matrix)[1]


@dataclass
class Symmetric(LinearMap):
    """Represents matrices $A$ where $A = A^{T}$.

    Properties:
        - Real eigenvalues
        - Orthogonal eigenvectors
    """

    matrix: jnp.ndarray

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            self.matrix = _ensure_ndarray(self.matrix)
            if self.matrix.ndim != 2:
                _raise_value_error("Matrix must be 2-dimensional")
        if not jnp.allclose(self.matrix, self.matrix.T):
            _raise_value_error("Matrix must be symmetric")

    def matvec(self, v: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(self.matrix, v)

    def transpose(self) -> "Symmetric":
        return self

    def parameter_count(self) -> int:
        dim = self.matrix.shape[0]
        return (dim * (dim + 1)) // 2

    def to_vector(self) -> jnp.ndarray:
        return _symmetric_to_vector(self.matrix)

    def logdet(self) -> jnp.ndarray:
        return jnp.linalg.slogdet(self.matrix)[1]

    def inverse(self) -> "Symmetric":
        """Compute inverse ensuring symmetry."""
        inv = jnp.linalg.inv(self.matrix)
        inv = 0.5 * (inv + inv.T)
        return Symmetric(inv)

    @staticmethod
    def from_vector(vec: jnp.ndarray, dim: Optional[int] = None) -> "Symmetric":
        """Create from vector of upper triangular elements."""
        dim = 1 if len(vec) == 1 else int((jnp.sqrt(1 + 8 * len(vec)) - 1) / 2)
        return Symmetric(_vector_to_symmetric(vec, dim))


@dataclass
class PositiveDefinite(Symmetric):
    """Symmetric positive definite matrix operator. Mathematical Definition where $A$ satisfies
        $x^T A x > 0, \\forall x \\neq 0$

    Properties:
        - All eigenvalues strictly positive
        - Unique Cholesky decomposition $A = LL^T$
        - Inverse exists and is positive definite

    Implementation:
        Uses Cholesky decomposition for numerical stability
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            jnp.linalg.cholesky(self.matrix)
        except ValueError:
            _raise_value_error("Matrix must be positive definite")

    def inverse(self) -> "PositiveDefinite":
        """Compute inverse via Cholesky decomposition."""
        chol = jnp.linalg.cholesky(self.matrix)
        inv = jnp.linalg.inv(chol)
        inv = inv.T @ inv
        return PositiveDefinite(inv)

    def logdet(self) -> jnp.ndarray:
        """Compute logdet using Cholesky decomposition."""
        chol = jnp.linalg.cholesky(self.matrix)
        return 2 * jnp.sum(jnp.log(jnp.diag(chol)))

    @staticmethod
    def from_vector(vec: jnp.ndarray, dim: Optional[int] = None) -> "PositiveDefinite":
        """Create from vector of upper triangular elements."""
        dim = 1 if len(vec) == 1 else int((jnp.sqrt(1 + 8 * len(vec)) - 1) / 2)
        return PositiveDefinite(_vector_to_symmetric(vec, dim))


@dataclass
class Diagonal(PositiveDefinite):
    """Diagonal matrix operator."""

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            self.matrix = _ensure_ndarray(self.matrix)
            if self.matrix.ndim != 2:
                _raise_value_error("Matrix must be 2-dimensional")
        if not jnp.allclose(self.matrix, jnp.diag(jnp.diag(self.matrix))):
            _raise_value_error("Matrix must be diagonal")
        super().__post_init__()

    def parameter_count(self) -> int:
        return self.matrix.shape[0]

    def to_vector(self) -> jnp.ndarray:
        return jnp.diag(self.matrix)

    def inverse(self) -> "Diagonal":
        """Compute inverse of diagonal matrix."""
        return Diagonal(jnp.diag(1.0 / jnp.diag(self.matrix)))

    def logdet(self) -> jnp.ndarray:
        return jnp.sum(jnp.log(jnp.diag(self.matrix)))


@dataclass
class Scale(Diagonal):
    """Scalar multiple of identity matrix."""

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            self.matrix = _ensure_ndarray(self.matrix)
            if self.matrix.ndim != 2:
                _raise_value_error("Matrix must be 2-dimensional")
        diag = jnp.diag(self.matrix)
        if not jnp.allclose(diag, diag[0]):
            _raise_value_error("Diagonal elements must be equal")
        super().__post_init__()

    def parameter_count(self) -> int:
        return 1

    def to_vector(self) -> jnp.ndarray:
        return jnp.array([jnp.diag(self.matrix)[0]])

    def inverse(self) -> "Scale":
        """Compute inverse of scale operator."""
        scale = jnp.diag(self.matrix)[0]
        return Scale(jnp.eye(self.matrix.shape[0]) / scale)

    @staticmethod
    def from_scale(scale: float, dim: int) -> "Scale":
        """Create from scale factor."""
        return Scale(scale * jnp.eye(dim))


@dataclass
class Identity(Scale):
    """Identity operator.

    Implementation:
        - Stores only dimension
        - Most operations are no-ops or return self
    """

    dim: int

    def __init__(self, dim: int):
        self.dim = dim

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


def _raise_type_error(name: str, expected: str) -> NoReturn:
    """Raise type error with consistent message."""
    raise TypeError(f"{name} must be {expected}")


def _raise_value_error(msg: str) -> NoReturn:
    """Raise value error with consistent message."""
    raise ValueError(msg)


def _ensure_ndarray(x: Union[float, int, jnp.ndarray]) -> jnp.ndarray:
    """Convert input to ndarray."""
    if isinstance(x, (float, int)):
        return jnp.array([[x]])
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        x_arr = jnp.asarray(x)
        if x_arr.ndim == 0 or x_arr.ndim == 1:
            return x_arr.reshape(1, 1) if x_arr.size == 1 else x_arr
        return x_arr
    _raise_type_error("Input", "numeric or ndarray")


def _vector_to_symmetric(vec: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Convert vector of length dim*(dim+1)/2 to symmetric matrix."""
    n = len(vec)
    if n == 1:
        return jnp.array([[vec[0]]])

    if n != dim * (dim + 1) // 2:
        _raise_value_error(f"Vector length {n} incompatible with dimension {dim}")

    i_upper = jnp.triu_indices(dim)
    matrix = jnp.zeros((dim, dim))
    matrix = matrix.at[i_upper].set(vec)
    return matrix + jnp.triu(matrix, k=1).T


def _symmetric_to_vector(matrix: jnp.ndarray) -> jnp.ndarray:
    """Convert symmetric matrix to vector containing upper triangular elements."""
    if not jnp.allclose(matrix, matrix.T):
        _raise_value_error("Matrix must be symmetric")
    if matrix.ndim != 2:
        _raise_value_error("Matrix must be 2-dimensional")
    return matrix[jnp.triu_indices(matrix.shape[0])]


def _get_scale_and_dim(value: Union[jnp.ndarray, float, int]) -> tuple[float, int]:
    """Extract scale and dimension from input."""
    value_arr = _ensure_ndarray(value)
    if value_arr.ndim == 2:
        dim = value_arr.shape[0]
        scale = float(value_arr[0, 0])
    else:
        dim = value_arr.shape[0]
        scale = float(value_arr[0])
    return scale, dim


def create_general(matrix: Union[jnp.ndarray, float, int]) -> Square:
    """Create general linear map."""
    arr = _ensure_ndarray(matrix)
    return Square(arr)


def create_symmetric(matrix: Union[jnp.ndarray, float, int]) -> Symmetric:
    """Create symmetric linear map."""
    arr = _ensure_ndarray(matrix)
    return Symmetric(arr)


def create_posdef(matrix: Union[jnp.ndarray, float, int]) -> PositiveDefinite:
    """Create positive definite linear map."""
    arr = _ensure_ndarray(matrix)
    return PositiveDefinite(arr)


def create_diagonal(matrix: Union[jnp.ndarray, float, int]) -> Diagonal:
    """Create diagonal linear map."""
    arr = _ensure_ndarray(matrix)
    return Diagonal(arr)


def create_scale(matrix: Union[jnp.ndarray, float, int]) -> Scale:
    """Create scale linear map."""
    scale, dim = _get_scale_and_dim(matrix)
    return Scale.from_scale(scale, dim)


def create_identity(matrix: Union[jnp.ndarray, float, int]) -> Identity:
    """Create identity linear map."""
    _, dim = _get_scale_and_dim(matrix)
    return Identity(dim)


_OPERATOR_CONSTRUCTORS: dict[
    str, Callable[[Union[jnp.ndarray, float, int]], LinearMap]
] = {
    "general": create_general,
    "symmetric": create_symmetric,
    "posdef": create_posdef,
    "diagonal": create_diagonal,
    "scale": create_scale,
    "identity": create_identity,
}


def create_linear_map(
    matrix: Union[jnp.ndarray, float, int], operator_type: str = "general"
) -> LinearMap:
    """Create appropriate linear map from matrix specification."""
    try:
        constructor = _OPERATOR_CONSTRUCTORS.get(operator_type)
        if constructor is None:
            _raise_value_error(f"Unknown operator type: {operator_type}")
        return constructor(matrix)
    except ValueError as e:
        _raise_value_error(f"Failed to create {operator_type} operator: {e!s}")
