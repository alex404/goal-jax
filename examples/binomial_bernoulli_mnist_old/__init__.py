"""Binomial-Bernoulli Mixture MNIST clustering example.

Demonstrates clustering MNIST digits with a hierarchical mixture model:
- Observable: Binomials (pixel count data)
- Latent: Bernoullis (binary hidden units)
- Mixture: Categorical over 10 clusters

Run with:
    python -m examples.binomial_bernoulli_mnist.run
    python -m examples.binomial_bernoulli_mnist.plot
"""
