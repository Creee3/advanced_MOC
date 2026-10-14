"""Equation-level ports of the official LADIES and AS-GCN sampling proposals.

Upstream snapshots, commits, and license status are in SOURCES.md. These are
NumPy implementations of the mathematical operations, not copied TF trainers.
No inverse-probability weighting is applied to natural-language messages.
"""

import numpy as np


def _support(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("support must be a finite matrix")
    return result


def normalize_mass(mass):
    mass = np.asarray(mass, dtype=np.float64)
    if mass.ndim != 1 or not np.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("sampling masses must be finite and nonnegative")
    total = mass.sum()
    return mass / total if total > 0 else np.zeros_like(mass)


def ladies_probabilities(support):
    """LADIES ladies_sampler: pi = sum(U * U, axis=0), p = pi / sum(pi)."""
    support = _support(support)
    return normalize_mass(np.square(support).sum(axis=0))


def asgcn_probabilities(support, parent_features, candidate_features, weights, sample_count):
    """AS-GCN SamplerAdapt.one_layer_sampling proposal, with explicit weights.

    Preserve the *sampler.py* column convention (w[:,1] for parents,
    w[:,0] for candidates), including its sum across parent features. The
    original propagator uses a different column convention; we do not silently
    reconcile that difference. This function supplies no pretrained weights.
    """
    support = _support(support)
    parents = np.asarray(parent_features, dtype=np.float64)
    candidates = np.asarray(candidate_features, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if parents.ndim != 2 or candidates.ndim != 2:
        raise ValueError("features must be matrices")
    if weights.shape != (candidates.shape[1], 2) or parents.shape[1] != candidates.shape[1]:
        raise ValueError("weights must have shape [feature_dimension, 2]")
    if support.shape != (len(parents), len(candidates)) or sample_count < 1:
        raise ValueError("support shape or sample_count is invalid")
    if not all(np.isfinite(x).all() for x in (parents, candidates, weights)):
        raise ValueError("nonfinite features or weights")
    h_v = np.sum(parents @ weights[:, 1])
    h_u = candidates @ weights[:, 0]
    attention = (np.maximum(h_v + h_u, 0) + 1) / sample_count
    feature_proxy = np.maximum(h_u, 0) + 1
    return normalize_mass(np.sqrt(np.square(support).sum(axis=0)) * attention * feature_proxy)


def weighted_without_replacement(probabilities, count, rng):
    """LADIES' bounded weighted draw, using a local RNG rather than global seed."""
    probabilities = normalize_mass(probabilities)
    count = min(max(0, int(count)), int(np.count_nonzero(probabilities)))
    if not count:
        return np.empty(0, dtype=np.int64)
    return rng.choice(len(probabilities), size=count, p=probabilities, replace=False)
