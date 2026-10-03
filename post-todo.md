# Post-branch TODO

Work for later that the `clique-container` branch does not need.

## Version bump: Python and all dependencies

- Raise `requires-python` from `>=3.12` to `>=3.14`, then update the libraries and regenerate
  `uv.lock` with `uv sync --all-extras`.
- Python 3.14 evaluates annotations lazily (PEP 649/749), which makes
  `from __future__ import annotations` unnecessary. 38 files in `src/`, `tests/` and `examples/`
  carry it.
- Bump every dependency, not only Python. As of 2026-10-03 the lockfile is behind on most
  packages: jax/jaxlib 0.9.0 to 0.11.2, optax 0.2.6 to 0.2.8, numpy 2.4 to 2.5, scipy 1.17 to
  1.18, matplotlib 3.10 to 3.11, and the CUDA wheels.
- JAX 0.9 to 0.11 breaking changes that might touch us:
  - `jnp.empty`/`empty_like` now return uninitialized memory;
  - `jnp.atleast_*d` return tuples instead of lists when given several arguments (we only
    call them with one);
  - removed `jax.core` internals and `jax.custom_remat`;
  - scalar constructors (`jnp.float32` etc.) are typed now, so basedpyright may report new
    errors.
  None of the removed APIs appear in `src/`, `tests/` or `examples/`.
- Check first that JAX, optax and basedpyright support 3.14.
- After the bump, run the full suite, basedpyright, sphinx and the examples against the current
  baseline.

## Move the clique code from `coupling` into `algebra`

`models/base/gaussian/coupling/junction_tree.py` (`JunctionTree`, `ChainTree`, and
`_triangulate`, `_maximal_cliques`, `_spanning_tree`) is static graph structure in pure Python
and numpy, with no reference to `Manifold`. That is the definition of `geometry/algebra/`.
`dense.py` and `sum_product.py` are Boltzmann inference kernels and stay where they are.

Possible code sharing:

- **`JunctionTree` as a `Cliques`.** Its maximal cliques would be `cliques`. Its
  `chordal_edges` would then be `Cliques.edges`, and its adjacency `Cliques.graph`.
- **Triangulation.** `_triangulate` computes the fill-in of a graph. The conjugation calculus in
  `CLIQUE-TODO.md` needs the same thing, because the prior's family must contain the fill-in
  that marginalizing the observables creates. A shared fill-in routine in `algebra` could serve
  both.
- **`ChainTree`.** It may correspond to the chain case of the composed graphs (HMoG). Worth
  checking once the conjugation work has settled.
