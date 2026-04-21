"""
agencity_clustering_visual.py

Clustering + 2D visualization of agencity signatures.
"""

import numpy as np
import matplotlib.pyplot as plt

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
# Signature → vector
# ===============================

def signature_to_vector(sig):
    return np.concatenate([
        sig["b_std"],
        [sig["growth_trend"]],
        [sig["peak_tau"]],
    ])


# ===============================
# PCA (simple implementation)
# ===============================

def pca_2d(X):
    # center
    X_centered = X - np.mean(X, axis=0)

    # covariance
    C = np.cov(X_centered, rowvar=False)

    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(C)

    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    # project to 2D
    return X_centered @ eigvecs[:, :2]


# ===============================
# K-means (reuse simple)
# ===============================

def kmeans(X, k=3, n_iter=20):
    n = len(X)
    indices = np.random.choice(n, k, replace=False)
    centroids = X[indices]

    for _ in range(n_iter):
        labels = np.array([
            np.argmin([np.linalg.norm(x - c) for c in centroids])
            for x in X
        ])

        for i in range(k):
            pts = X[labels == i]
            if len(pts) > 0:
                centroids[i] = np.mean(pts, axis=0)

    return labels


# ===============================
# Main
# ===============================

def main():
    xi = np.linspace(0, 20, 500)
    taus = [0.5, 1, 2, 5, 10, 20]

    signals = generate_signals(xi)

    print("\n=== COMPUTING SIGNATURES ===")

    vectors = []
    names = []

    for name, u in signals.items():
        sig = agencity_signature(xi, u, taus)
        vec = signature_to_vector(sig)

        vectors.append(vec)
        names.append(name)

        print(f"{name:15} → {sig['signature_type']}")

    X = np.array(vectors)

    # ===============================
    # PCA projection
    # ===============================
    X_2D = pca_2d(X)

    # ===============================
    # Clustering
    # ===============================
    labels = kmeans(X, k=3)

    # ===============================
    # Visualization
    # ===============================
    plt.figure()

    for i, (x, y) in enumerate(X_2D):
        plt.scatter(x, y)
        plt.text(x, y, names[i])

    plt.title("Agencity Signature Space (PCA projection)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()