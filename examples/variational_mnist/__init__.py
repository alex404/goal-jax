"""Variational MNIST clustering example.

Demonstrates variational conjugated model for MNIST digit clustering:
- Observable: Binomials (pixel count data in [0, n_trials])
- Base latent: Bernoullis (binary hidden units)
- Mixture: Categorical over K clusters

The key insight is using analytical rho computation via least-squares
regression at each training step, which prevents the optimizer from
"cheating" by flattening the likelihood.

Run with:
    python -m examples.variational_mnist.train
    python -m examples.variational_mnist.plot
"""
