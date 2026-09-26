"""Checks for :class:`~examples.variational_mnist.lattice_convolution.LatticeConvolution`.

Covers same-lattice single-channel, strided upsampling, and multi-channel cases:

1. **Application** matches an independent transposed-conv reference (scatter form).
2. **Transpose**: ``transpose_apply`` equals ``W^T w``.
3. **outer_product** equals ``d/dkernel <w, W(kernel) v>`` (autodiff).
4. **Conjugation submanifold**: ``P^sigma = W^T Sigma W`` (over the *input* units) is
   supported exactly on ``induced_coupling_graph()`` for any kernel and any diagonal
   ``Sigma`` -- and that graph equals an independent footprint-overlap computation.
5. **MNIST-shaped deployment**: a coarse 7x7 latent upsampled to 28x28 induces a
   bounded-degree grid graph on the latent (junction-tree-tractable).

Run::

    uv run python -m examples.variational_mnist.validate_lattice_convolution
"""

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

from goal.models.base.gaussian.generalized import Euclidean  # noqa: E402

from .lattice_convolution import LatticeConvolution  # noqa: E402


def make(
    in_lattice: tuple[int, ...],
    stride: tuple[int, ...],
    kernel: tuple[int, ...],
    cin: int = 1,
    cout: int = 1,
) -> LatticeConvolution[Euclidean, Euclidean]:
    in_n = prod(in_lattice) * cin
    out_n = prod(tuple(np.array(in_lattice) * np.array(stride))) * cout
    return LatticeConvolution.create(
        Euclidean(in_n), Euclidean(out_n), in_lattice, stride, kernel, cin, cout
    )


def reference_apply(conv: LatticeConvolution[Euclidean, Euclidean], kernel: Array, v: Array) -> Array:
    """Independent transposed-conv (scatter): each input cell writes its kernel into
    the output patch at ``q*stride`` centered by ``kernel//2``, summed over channels."""
    in_lat, stride, kshape = conv.in_lattice, conv.stride, conv.kernel_shape
    out_lat = conv.out_lattice
    cin, cout = conv.in_channels, conv.out_channels
    center = np.array(kshape) // 2
    out = np.zeros((prod(out_lat), cout))
    v_arr = np.array(v).reshape(prod(in_lat), cin)
    ker = np.array(kernel).reshape(prod(kshape), cin, cout)
    in_cells = list(np.ndindex(*in_lat))
    for qi, q in enumerate(in_cells):
        for ki, k in enumerate(np.ndindex(*kshape)):
            p = np.array(q) * np.array(stride) + np.array(k) - center
            if np.all((p >= 0) & (p < np.array(out_lat))):
                pi = int(np.ravel_multi_index(tuple(p), out_lat))
                out[pi] += ker[ki].T @ v_arr[qi]  # (cout,) += K[ci,co]^T v[ci]
    return jnp.array(out.reshape(-1))


def footprint_overlap_graph(conv: LatticeConvolution[Euclidean, Euclidean]) -> Array:
    """Independent (per input cell) footprint-overlap graph tensored over input channels."""
    in_lat, stride, kshape, out_lat = conv.in_lattice, conv.stride, conv.kernel_shape, conv.out_lattice
    center = np.array(kshape) // 2
    cells = list(np.ndindex(*in_lat))
    foot: list[set[int]] = []
    for q in cells:
        fp = set()
        for k in np.ndindex(*kshape):
            p = np.array(q) * np.array(stride) + np.array(k) - center
            if np.all((p >= 0) & (p < np.array(out_lat))):
                fp.add(int(np.ravel_multi_index(tuple(p), out_lat)))
        foot.append(fp)
    n = len(cells)
    spatial = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if foot[i] & foot[j]:
                spatial[i, j] = 1.0
    return jnp.array(np.kron(spatial, np.ones((conv.in_channels, conv.in_channels))))


def main() -> None:
    key = jax.random.PRNGKey(0)
    # (in_lattice, stride, kernel, cin, cout)
    cases = [
        ((10,), (1,), (3,), 1, 1),        # same-lattice single-channel (regression)
        ((16, 4), (1, 1), (3, 3), 1, 1),  # thin-strip stride-1
        ((7, 7), (4, 4), (6, 6), 1, 1),   # coarse -> fine upsampling (MNIST-shaped, 1ch)
        ((7, 7), (4, 4), (6, 6), 3, 1),   # multi-channel latent -> grayscale
        ((5, 5), (2, 2), (4, 4), 2, 2),   # multi in + multi out, strided
    ]

    for in_lat, stride, kshape, cin, cout in cases:
        conv = make(in_lat, stride, kshape, cin, cout)
        in_n, out_n = conv.dom_man.dim, conv.cod_man.dim
        k_k, k_v, k_w, k_s = jax.random.split(
            jax.random.fold_in(key, hash((in_lat, stride, kshape, cin, cout)) % 997), 4
        )
        kernel = jax.random.normal(k_k, (conv.dim,))
        v = jax.random.normal(k_v, (in_n,))
        w = jax.random.normal(k_w, (out_n,))
        sigma = jnp.exp(jax.random.normal(k_s, (out_n,)))
        wmat = conv.to_dense(kernel)

        app_err = float(jnp.max(jnp.abs(conv(kernel, v) - reference_apply(conv, kernel, v))))
        trn_err = float(jnp.max(jnp.abs(conv.transpose_apply(kernel, w) - wmat.T @ w)))
        op = conv.outer_product(w, v)
        op_ad = jax.grad(lambda kk: jnp.dot(w, conv(kk, v)))(kernel)
        op_err = float(jnp.max(jnp.abs(op - op_ad)))

        p_sigma = wmat.T @ (sigma[:, None] * wmat)  # over input units (in_n x in_n)
        supp = conv.induced_coupling_graph()
        off_max = float(jnp.max(jnp.abs(p_sigma) * (1.0 - supp)))
        realized = (jnp.abs(p_sigma) > 1e-9).astype(float)
        graph_tight = bool(jnp.all(realized == supp))
        graph_matches_ref = bool(jnp.all(supp == footprint_overlap_graph(conv)))
        max_deg = int(jnp.max(jnp.sum(supp, axis=1)))

        print(f"in_lat={in_lat} stride={stride} kernel={kshape} Cin={cin} Cout={cout}")
        print(f"  dims: {in_n} -> {out_n} (out_lattice={conv.out_lattice}), kernel params={conv.dim}")
        print(f"  apply vs reference       : {app_err:.2e}")
        print(f"  transpose vs W^T          : {trn_err:.2e}")
        print(f"  outer_product vs autodiff : {op_err:.2e}")
        print(f"  P^sigma off induced graph : {off_max:.2e}   (must be ~0)")
        print(f"  induced == realized supp  : {graph_tight}   induced == ref overlap: {graph_matches_ref}")
        print(f"  max couplings / input unit: {max_deg}")
        assert app_err < 1e-10 and trn_err < 1e-10 and op_err < 1e-10
        assert off_max < 1e-10 and graph_tight and graph_matches_ref

    # 5. MNIST-shaped tractability summary
    conv = make((7, 7), (4, 4), (6, 6))
    deg = int(jnp.max(jnp.sum(conv.induced_coupling_graph(), axis=1)))
    msg = f" (nn on 7x7 => treewidth ~7, junction-tree tractable)."
    print(f"\nMNIST decoder 7x7 -> 28x28 (k=6,s=4): induced latent graph max degree {deg}" + msg)

    print("\nAll LatticeConvolution checks passed.")


if __name__ == "__main__":
    main()
