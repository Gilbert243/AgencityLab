"""
agencity_study.py

Publication-style experiment:
- Multi-system comparison
- Signature extraction
- PCA projection
- Clustering
- Clean figures
"""

import numpy as np
import matplotlib.pyplot as plt

from agencitylab.analysis.signature import agencity_signature


# ===============================
# 1. Signal generation
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
# 2. Signature → vector
# ===============================

def signature_to_vector(sig):
    vec = np.concatenate([
        sig["b_std"],
        [sig["growth_trend"]],
        [sig["peak_tau"]],
    ])
    return vec / (np.linalg.norm(vec) + 1e-12)


# ===============================
# 3. PCA
# ===============================

def pca_2d(X):
    Xc = X - np.mean(X, axis=0)
    C = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    return Xc @ eigvecs[:, idx[:2]]


# ===============================
# 4. Clustering (k-means)
# ===============================

def kmeans(X, k=3, n_iter=30):
    n = len(X)
    centroids = X[np.random.choice(n, k, replace=False)]

    for _ in range(n_iter):
        labels = np.array([
            np.argmin([np.linalg.norm(x - c) for c in centroids])
            for x in X
        ])

        for i in range(k):
            pts = X[labels == i]
            if len(pts):
                centroids[i] = pts.mean(axis=0)

    return labels


# ===============================
# 5. Plot helpers
# ===============================

def plot_signals(xi, signals):
    plt.figure()
    for name, u in signals.items():
        plt.plot(xi, u, label=name)
    plt.title("Input signals")
    plt.legend()
    plt.grid()


def plot_spectrum_example(xi, u, taus):
    from agencitylab.analysis.multi_scale import agencity_spectrum

    spec = agencity_spectrum(xi, u, taus)

    tau = [s["tau"] for s in spec]
    bstd = [s["b_std"] for s in spec]

    plt.figure()
    plt.plot(tau, bstd, marker="o")
    plt.title("Agencity spectrum (example)")
    plt.xlabel("tau")
    plt.ylabel("b_std")
    plt.grid()


def plot_embedding(X2D, names, labels):
    plt.figure()

    for i, (x, y) in enumerate(X2D):
        plt.scatter(x, y)
        plt.text(x, y, names[i])

    plt.title("Agencity signature space (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()


# ===============================
# 6. Main experiment
# ===============================

def main():
    xi = np.linspace(0, 20, 500)
    taus = [0.5, 1, 2, 5, 10, 20]

    # Signals
    signals = generate_signals(xi)

    # Plot signals
    plot_signals(xi, signals)

    # Compute signatures
    vectors = []
    names = []

    print("\n=== SIGNATURES ===")

    for name, u in signals.items():
        sig = agencity_signature(xi, u, taus)
        vec = signature_to_vector(sig)

        vectors.append(vec)
        names.append(name)

        print(f"{name:15} → {sig['signature_type']}")

    X = np.array(vectors)

    # PCA projection
    X2D = pca_2d(X)

    # Clustering
    labels = kmeans(X, k=3)

    # Plot embedding
    plot_embedding(X2D, names, labels)

    # Spectrum example
    plot_spectrum_example(xi, signals["Multi-scale"], taus)

    # Final display
    plt.show()


if __name__ == "__main__":
    main()
    