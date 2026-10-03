"""Subparameterization of the likelihood interaction whose conjugation parameter
lies analytically on a chosen submanifold.

For a linear-Gaussian likelihood ``x | y ~ N(mu + W s_Y(y), Sigma)`` (``Sigma``
diagonal, the location-only case), the LGM conjugation identity
(:mod:`goal.models.harmonium.lgm`, class ``LGM`` docstring) gives the latent-side
second-order conjugation parameter

    P^sigma = 1/2 W^T Sigma W,

which is *added onto the latent precision* to form the prior. A generic dense
loading ``W`` makes ``P^sigma`` dense -- for an MVN prior that densifies the
covariance, for a Boltzmann prior it densifies the pairwise couplings (blowing up
the junction-tree treewidth). We want a **subparameterization** of ``W`` for which
``P^sigma`` is *supported on a chosen graph* ``G`` for every value of the free
parameters, so the induced prior never leaves the ``G``-sparse submanifold.

Key identity
------------
``(W^T Sigma W)_{ij} = w_i^T Sigma w_j`` -- the ``Sigma``-inner-product of loading
columns ``i`` and ``j``. Hence the support of ``P^sigma`` is exactly the
``Sigma``-overlap graph of ``W``'s columns. Restricting each column's *support*
(an observable receptive field) fixes that overlap graph analytically.

Construction (node + edge blocks)
---------------------------------
Partition the observable coordinates into a private block per latent node and a
shared block per graph edge. Latent column ``i`` is free on its own node block and
on the blocks of edges incident to ``i``, and zero elsewhere. Then columns ``i``,
``j`` share support iff ``(i, j)`` is an edge of ``G``, so ``P^sigma`` is supported
exactly on ``G`` -- for *any* free loadings. This is a coordinate subspace (a 0/1
mask) of ``Theta_XY``: the requested subparameterization.

Run::

    uv run python -m examples.variational_mnist.conjugation_subparameterization
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def node_edge_support_mask(
    n_nodes: int, edges: list[tuple[int, int]], block: int = 2
) -> tuple[Array, list[int]]:
    """0/1 column-support mask over ``Theta_XY`` realizing a receptive-field cover of ``G``.

    Observable layout: ``n_nodes`` private blocks of width ``block`` (one per latent
    node), then ``len(edges)`` shared blocks (one per edge). Column ``i`` is
    supported on its node block and on the block of every edge incident to ``i``.

    Returns ``(mask, obs_dim)`` with ``mask`` shape ``(obs_dim, n_nodes)``.
    """
    n_edges = len(edges)
    obs_dim = (n_nodes + n_edges) * block
    mask = np.zeros((obs_dim, n_nodes))
    for i in range(n_nodes):  # private node blocks
        mask[i * block : (i + 1) * block, i] = 1.0
    for k, (a, b) in enumerate(edges):  # shared edge blocks
        rows = slice((n_nodes + k) * block, (n_nodes + k + 1) * block)
        mask[rows, a] = 1.0
        mask[rows, b] = 1.0
    return jnp.array(mask), obs_dim


def target_support(n_nodes: int, edges: list[tuple[int, int]]) -> Array:
    """Symmetric 0/1 matrix: diagonal plus the (undirected) edges of ``G``."""
    s = np.eye(n_nodes)
    for a, b in edges:
        s[a, b] = s[b, a] = 1.0
    return jnp.array(s)


def induced_conjugation_precision(w: Array, sigma_diag: Array) -> Array:
    """P^sigma up to the 1/2 factor: ``W^T Sigma W`` with diagonal ``Sigma``."""
    return w.T @ (sigma_diag[:, None] * w)


def check_masked_loading(
    n_nodes: int, edges: list[tuple[int, int]], key: Array, block: int = 2
) -> None:
    mask, obs_dim = node_edge_support_mask(n_nodes, edges, block)
    supp = target_support(n_nodes, edges)

    k_w, k_s = jax.random.split(key)
    w = mask * jax.random.normal(k_w, mask.shape)  # free loadings within the mask
    sigma = jnp.exp(
        jax.random.normal(k_s, (obs_dim,))
    )  # arbitrary positive diagonal Sigma
    p = induced_conjugation_precision(w, sigma)

    off = jnp.abs(p) * (1.0 - supp)  # entries that MUST be zero
    on_offdiag = jnp.abs(p) * (
        supp - jnp.eye(n_nodes)
    )  # edges: should be generically nonzero
    off_max = float(jnp.max(off))
    on_min = float(jnp.min(on_offdiag[on_offdiag > 0])) if edges else float("nan")
    recovered = (jnp.abs(p) > 1e-9).astype(float)
    matches = bool(jnp.all(recovered == supp))

    print(
        f"  nodes={n_nodes} edges={len(edges)} obs_dim={obs_dim}  Theta_XY free params={int(mask.sum())}/{mask.size}"
    )
    print(f"    max |P^sigma| OFF support   : {off_max:.2e}   (must be ~0)")
    print(
        f"    min |P^sigma| ON  edges      : {on_min:.2e}   (nonzero => not a trivial fit)"
    )
    print(f"    recovered support == G       : {matches}")
    assert off_max < 1e-10, "P^sigma leaked off the target graph"
    assert matches, "recovered support does not equal G"


def check_sigma_orthonormal_diagonal(obs_dim: int, n_nodes: int, key: Array) -> None:
    """The ``S = diagonal`` instance: W = Sigma^{-1/2} Q D (Q Stiefel) => W^T Sigma W = D^2."""
    k_q, k_d, k_s = jax.random.split(key, 3)
    sigma = jnp.exp(jax.random.normal(k_s, (obs_dim,)))
    q, _ = jnp.linalg.qr(
        jax.random.normal(k_q, (obs_dim, n_nodes))
    )  # orthonormal columns
    d = jax.random.normal(k_d, (n_nodes,))
    w = (sigma**-0.5)[:, None] * q * d[None, :]
    p = induced_conjugation_precision(w, sigma)
    off_diag_max = float(jnp.max(jnp.abs(p - jnp.diag(jnp.diagonal(p)))))
    diag_err = float(jnp.max(jnp.abs(jnp.diagonal(p) - d**2)))
    print(f"  diagonal case: obs_dim={obs_dim} n_nodes={n_nodes}")
    print(f"    max |off-diagonal P^sigma|   : {off_diag_max:.2e}   (must be ~0)")
    print(f"    ||diag(P^sigma) - D^2||_inf   : {diag_err:.2e}")
    assert off_diag_max < 1e-10 and diag_err < 1e-10


def conv2d_transpose_matrix(h: int, w: int, kernel: Array) -> Array:
    """Stride-1 'same' transposed-conv loading ``W`` (obs grid = latent grid = h x w).

    Column ``q`` (latent at grid cell) writes the ``kernel`` into the output patch
    centered at ``q`` -- i.e. ``W`` is block-Toeplitz and each latent's footprint is
    the ``kh x kw`` neighborhood. Single channel; shape ``(h*w, h*w)``.
    """
    kh, kw = kernel.shape
    rh, rw = kh // 2, kw // 2
    n = h * w
    mat = np.zeros((n, n))
    for rq in range(h):
        for cq in range(w):
            q = rq * w + cq
            for dr in range(-rh, rh + 1):
                for dc in range(-rw, rw + 1):
                    rp, cp = rq + dr, cq + dc
                    if 0 <= rp < h and 0 <= cp < w:
                        mat[rp * w + cp, q] = float(kernel[dr + rh, dc + rw])
    return jnp.array(mat)


def chebyshev_graph(h: int, w: int, radius: int) -> Array:
    """Symmetric 0/1 support: cells within Chebyshev distance ``radius`` on an h x w grid."""
    s = np.zeros((h * w, h * w))
    for r1 in range(h):
        for c1 in range(w):
            i = r1 * w + c1
            for r2 in range(h):
                for c2 in range(w):
                    if max(abs(r1 - r2), abs(c1 - c2)) <= radius:
                        s[i, r2 * w + c2] = 1.0
    return jnp.array(s)


def check_conv_loading(h: int, w: int, ksize: int, key: Array) -> None:
    """A stride-1 ksize x ksize conv induces P^sigma on the distance-(ksize-1) grid graph."""
    k_k, k_s = jax.random.split(key)
    kernel = jax.random.normal(k_k, (ksize, ksize))
    conv = conv2d_transpose_matrix(h, w, kernel)
    sigma = jnp.exp(jax.random.normal(k_s, (h * w,)))
    p = induced_conjugation_precision(conv, sigma)

    radius = (
        ksize - 1
    )  # two radius-(ksize//2) footprints overlap within Chebyshev 2*(ksize//2)
    supp = chebyshev_graph(h, w, radius)
    off_max = float(jnp.max(jnp.abs(p) * (1.0 - supp)))
    realized = float(jnp.mean((jnp.abs(p) > 1e-9)[supp > 0]))
    max_neighbors = int(jnp.max(jnp.sum(jnp.abs(p) > 1e-9, axis=1)))
    print(
        f"  conv {ksize}x{ksize} on {h}x{w} latent grid  (kernel params={ksize * ksize})"
    )
    print(f"    P^sigma support radius       : Chebyshev <= {radius}")
    print(f"    max |P^sigma| beyond radius  : {off_max:.2e}   (must be ~0)")
    print(f"    fraction of dist<={radius} edges realized: {realized:.2f}")
    print(
        f"    max couplings per latent      : {max_neighbors}  (=> thin-grid treewidth ~ 2*width)"
    )
    assert off_max < 1e-10, (
        "conv-induced P^sigma leaked beyond its footprint-overlap radius"
    )


def grid_edges(h: int, w: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            i = r * w + c
            if c + 1 < w:
                edges.append((i, r * w + c + 1))
            if r + 1 < h:
                edges.append((i, (r + 1) * w + c))
    return edges


def main() -> None:
    key = jax.random.PRNGKey(0)
    print("Masked (node+edge receptive-field) loading -> P^sigma supported on G:")
    # chain
    check_masked_loading(6, [(i, i + 1) for i in range(5)], jax.random.fold_in(key, 1))
    # 3x3 grid (chordal after the grid is itself not chordal, but support test is graph-agnostic;
    # tractability needs a chordal G -- here we just verify the support identity)
    check_masked_loading(9, grid_edges(3, 3), jax.random.fold_in(key, 2))
    # a denser random-ish graph
    check_masked_loading(
        5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (2, 4)], jax.random.fold_in(key, 3)
    )

    print("\nSigma-orthonormal factorization -> P^sigma diagonal:")
    check_sigma_orthonormal_diagonal(40, 8, jax.random.fold_in(key, 4))

    print("\nStride-1 conv loading -> P^sigma on the distance-(k-1) grid graph:")
    check_conv_loading(16, 4, 3, jax.random.fold_in(key, 5))  # thin grid (efficient)
    check_conv_loading(
        8, 8, 3, jax.random.fold_in(key, 6)
    )  # square grid (sparse but high treewidth)
    check_conv_loading(16, 4, 5, jax.random.fold_in(key, 7))  # 5x5 kernel -> wider band

    print(
        "\nAll subparameterization checks passed: P^sigma lies on the target submanifold analytically."
    )


if __name__ == "__main__":
    main()
