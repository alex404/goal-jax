"""Evaluation utilities for Variational MNIST.

This module provides:
- Cluster purity computation
- Normalized Mutual Information (NMI)
- Cluster accuracy via Hungarian algorithm
- Reconstruction error
"""

import numpy as np
from jax import Array


def compute_cluster_purity(
    assignments: Array,
    labels: Array,
    n_clusters: int,
) -> float:
    """Compute cluster purity given true labels.

    Purity measures the fraction of samples in each cluster that belong to
    the majority class for that cluster.

    Args:
        assignments: Cluster assignments (shape: n_samples,)
        labels: True labels (shape: n_samples,)
        n_clusters: Number of clusters

    Returns:
        Purity score in [0, 1]
    """
    assignments_np = np.array(assignments)
    labels_np = np.array(labels)
    contingency = np.zeros((n_clusters, 10), dtype=np.int32)
    np.add.at(contingency, (assignments_np, labels_np), 1)
    return float(contingency.max(axis=1).sum() / len(labels))


def compute_nmi(assignments: Array, labels: Array) -> float:
    """Compute Normalized Mutual Information between assignments and labels.

    NMI measures the mutual dependence between cluster assignments and true
    labels, normalized to [0, 1].

    Args:
        assignments: Cluster assignments (shape: n_samples,)
        labels: True labels (shape: n_samples,)

    Returns:
        NMI score in [0, 1]
    """
    from sklearn.metrics import normalized_mutual_info_score

    return float(
        normalized_mutual_info_score(
            np.array(labels), np.array(assignments), average_method="arithmetic"
        )
    )


def compute_cluster_accuracy(
    assignments: Array,
    labels: Array,
    n_clusters: int,
    n_classes: int = 10,
) -> float:
    """Compute accuracy via Hungarian algorithm for optimal cluster-to-class mapping.

    This finds the optimal one-to-one assignment of clusters to classes that
    maximizes classification accuracy.

    Args:
        assignments: Cluster assignments (shape: n_samples,)
        labels: True labels (shape: n_samples,)
        n_clusters: Number of clusters
        n_classes: Number of classes (default: 10 for MNIST)

    Returns:
        Accuracy score in [0, 1]
    """
    from scipy.optimize import linear_sum_assignment

    # Build contingency matrix
    assignments_np = np.array(assignments)
    labels_np = np.array(labels)
    contingency = np.zeros((n_clusters, n_classes), dtype=np.int32)
    np.add.at(contingency, (assignments_np, labels_np), 1)

    # Find optimal assignment using Hungarian algorithm
    # We negate the contingency matrix since linear_sum_assignment minimizes
    row_ind, col_ind = linear_sum_assignment(-contingency)

    return float(contingency[row_ind, col_ind].sum() / len(labels))


def compute_adjusted_rand_index(assignments: Array, labels: Array) -> float:
    """Compute Adjusted Rand Index between assignments and labels.

    ARI measures similarity between two clusterings, adjusted for chance.
    Range: [-1, 1], where 1 indicates perfect agreement.

    Args:
        assignments: Cluster assignments (shape: n_samples,)
        labels: True labels (shape: n_samples,)

    Returns:
        ARI score in [-1, 1]
    """
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(np.array(labels), np.array(assignments)))


def compute_homogeneity_completeness(
    assignments: Array, labels: Array
) -> tuple[float, float, float]:
    """Compute homogeneity, completeness, and V-measure.

    - Homogeneity: each cluster contains only members of a single class
    - Completeness: all members of a class are assigned to the same cluster
    - V-measure: harmonic mean of homogeneity and completeness

    Args:
        assignments: Cluster assignments (shape: n_samples,)
        labels: True labels (shape: n_samples,)

    Returns:
        Tuple of (homogeneity, completeness, v_measure)
    """
    from sklearn.metrics import homogeneity_completeness_v_measure

    h, c, v = homogeneity_completeness_v_measure(
        np.array(labels), np.array(assignments)
    )
    return float(h), float(c), float(v)


def evaluate_clustering(
    assignments: Array,
    labels: Array,
    n_clusters: int,
) -> dict[str, float]:
    """Compute all clustering evaluation metrics.

    Args:
        assignments: Cluster assignments (shape: n_samples,)
        labels: True labels (shape: n_samples,)
        n_clusters: Number of clusters

    Returns:
        Dictionary with all metrics
    """
    purity = compute_cluster_purity(assignments, labels, n_clusters)
    nmi = compute_nmi(assignments, labels)
    accuracy = compute_cluster_accuracy(assignments, labels, n_clusters)
    ari = compute_adjusted_rand_index(assignments, labels)
    homogeneity, completeness, v_measure = compute_homogeneity_completeness(
        assignments, labels
    )

    return {
        "purity": purity,
        "nmi": nmi,
        "accuracy": accuracy,
        "ari": ari,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
    }
