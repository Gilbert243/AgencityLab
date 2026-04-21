"""
agencity_clustering.py

Clustering of systems based on agencity signatures.

Steps:
1. Generate signals
2. Normalize signals
3. Compute signatures
4. Build feature vectors
5. Compute distances
6. Cluster (k-means)
"""

import numpy as np

from agencitylab.analysis.signature import agencity_signature


# ===============================
# Utilities
# ===============================

def normalize(u):
    return u / (np.std(u) + 1e-12)


def generate_signals(xi):
    return {
        "Constant": normalize(np.ones_like(xi)),
        "Sinus": normalize(np.sin(xi)),
        "White noise": normalize(np.random.randn(len(xi))),
        "Sinus + noise": normalize(np.sin(xi) + 0.3 * np.random.randn(len(xi))),
        "Multi-scale": normalize(np.sin(xi) + 0.3 * np.cos(3 * xi)),
    }


# ===============================
# Feature extraction
# ===============================

def signature_to_vector(sig):
    """
    Convert signature to numerical vector for ML.
    """
    return np.concatenate([
        sig["b_std"],           # spectrum shape
        [sig["growth_trend"]],  # slope
        [sig["peak_tau"]],      # scale info
    ])


# ===============================
# Distance
# ===============================

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def compute_distance_matrix(vectors):
    n = len(vectors)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            D[i, j] = euclidean_distance(vectors[i], vectors[j])

    return D


# ===============================
# Simple K-Means (from scratch)
# ===============================

def kmeans(X, k=2, n_iter=20):
    n, d = X.shape

    # random init
    indices = np.random.choice(n, k, replace=False)
    centroids = X[indices]

    for _ in range(n_iter):
        # assign
        labels = np.array([
            np.argmin([euclidean_distance(x, c) for c in centroids])
            for x in X
        ])

        # update
        for i in range(k):
            points = X[labels == i]
            if len(points) > 0:
                centroids[i] = np.mean(points, axis=0)

    return labels, centroids


# ===============================
# Main
# ===============================

def main():
    xi = np.linspace(0, 20, 500)
    taus = [0.5, 1, 2, 5, 10, 20]

    signals = generate_signals(xi)

    print("\n=== COMPUTING SIGNATURES ===")

    signatures = {}
    vectors = []
    names = []

    for name, u in signals.items():
        sig = agencity_signature(xi, u, taus)
        vec = signature_to_vector(sig)

        signatures[name] = sig
        vectors.append(vec)
        names.append(name)

        print(f"{name:15} → type: {sig['signature_type']}")

    X = np.array(vectors)

    # ===============================
    # Distance matrix
    # ===============================
    print("\n=== DISTANCE MATRIX ===")
    D = compute_distance_matrix(X)

    for i, name_i in enumerate(names):
        row = " ".join(f"{D[i,j]:6.2f}" for j in range(len(names)))
        print(f"{name_i:15} | {row}")

    # ===============================
    # Clustering
    # ===============================
    print("\n=== CLUSTERING ===")
    labels, centroids = kmeans(X, k=3)

    for name, label in zip(names, labels):
        print(f"{name:15} → Cluster {label}")


if __name__ == "__main__":
    main()