import equinox as eqx
import jax.numpy as jnp


class LinearOperator(eqx.Module):
    """Base class for linear operators"""

    matrix: jnp.ndarray

    def __init__(self, matrix: jnp.ndarray):
        self.matrix = matrix

    def multiply(self, vector: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(self.matrix, vector)

    def transpose_multiply(self, vector: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(self.matrix.T, vector)

    def inverse(self) -> "LinearOperator":
        return LinearOperator(jnp.linalg.inv(self.matrix))

    def log_determinant(self) -> float:
        return float(jnp.linalg.slogdet(self.matrix)[1])


class DiagonalOperator(LinearOperator):
    """Diagonal matrix representation"""

    diagonal: jnp.ndarray

    def __init__(self, diagonal: jnp.ndarray):
        self.diagonal = diagonal
        super().__init__(jnp.diag(diagonal))

    def multiply(self, vector: jnp.ndarray) -> jnp.ndarray:
        return self.diagonal * vector

    def transpose_multiply(self, vector: jnp.ndarray) -> jnp.ndarray:
        return self.multiply(vector)  # Symmetric

    def inverse(self) -> "DiagonalOperator":
        return DiagonalOperator(1.0 / self.diagonal)

    def log_determinant(self) -> float:
        return float(jnp.sum(jnp.log(jnp.abs(self.diagonal))))


def schur_complement(
    a_inv: LinearOperator, b: jnp.ndarray, c: jnp.ndarray, d: LinearOperator
) -> LinearOperator:
    """Compute Schur complement D - C A^{-1} B"""
    if isinstance(d, DiagonalOperator):
        intermediate = jnp.dot(c, a_inv.multiply(b))
        return DiagonalOperator(d.diagonal - jnp.diag(intermediate))
    return LinearOperator(d.matrix - jnp.dot(c, a_inv.multiply(b)))
