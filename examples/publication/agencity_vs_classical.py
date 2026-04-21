"""
agencity_vs_classical.py

Robust comparison between Agencity and classical metrics.

Metrics:
- Agencity (b_std)
- Shannon entropy (discrete)
- Variance
- Energy
- Autocorrelation

Includes:
- Table output
- Comparison plots
- Agencity spectra
"""

import numpy as np
import matplotlib.pyplot as plt

from agencitylab.api import compute_agencity
from agencitylab.analysis.multi_scale import agencity_spectrum


# ===============================
# 1. SAFE NORMALIZATION
# ===============================

def normalize(u):
    std = np.std(u)
    if std < 1e-12:
        return np.zeros_like(u)
    return u / std


# ===============================
# 2. SIGNAL GENERATION
# ===============================

def generate_signals(xi):
    return {
        "Constant": normalize(np.ones_like(xi)),
        "Sinus": normalize(np.sin(xi)),
        "White noise": normalize(np.random.randn(len(xi))),
        "Sinus + noise": normalize(np.sin(xi) + 0.3 * np.random.randn(len(xi))),
        "Multi-scale": normalize(np.sin(xi) + 0.3 * np.cos(3 * xi)),
    }


# ===============================
# 3. CLASSICAL METRICS (FIXED)
# ===============================

def shannon_entropy(x, bins=50):
    hist, _ = np.histogram(x, bins=bins, density=False)
    p = hist / np.sum(hist)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def signal_energy(x):
    return float(np.mean(x**2))


def autocorr_strength(x):
    x = x - np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


# ===============================
# 4. ANALYSIS
# ===============================

def analyze_signal(xi, u):
    result = compute_agencity(xi, u)

    return {
        "b_std": float(np.std(result.b)),
        "entropy": shannon_entropy(u),
        "variance": float(np.var(u)),
        "energy": signal_energy(u),
        "autocorr": autocorr_strength(u),
        "b": result.b
    }


# ===============================
# 5. VISUALIZATION
# ===============================

def plot_comparison(results):
    names = list(results.keys())

    metrics = ["b_std", "entropy", "variance", "energy", "autocorr"]

    plt.figure()

    for m in metrics:
        values = [results[n][m] for n in names]
        plt.plot(names, values, marker="o", label=m)

    plt.title("Agencity vs Classical Metrics")
    plt.xticks(rotation=20)
    plt.legend()
    plt.grid()


def plot_spectra(xi, signals, taus):
    plt.figure()

    for name, u in signals.items():
        spec = agencity_spectrum(xi, u, taus)
        tau = [s["tau"] for s in spec]
        bstd = [s["b_std"] for s in spec]

        plt.plot(tau, bstd, marker="o", label=name)

    plt.title("Agencity Spectra Comparison")
    plt.xlabel("tau")
    plt.ylabel("b_std")
    plt.legend()
    plt.grid()


def plot_b_timeseries(results):
    plt.figure()

    for name, data in results.items():
        plt.plot(data["b"], label=name)

    plt.title("Agencity time series b(t)")
    plt.xlabel("index")
    plt.ylabel("b")
    plt.legend()
    plt.grid()


# ===============================
# 6. CORRELATION ANALYSIS
# ===============================

def compute_correlation(results):
    metrics = ["b_std", "entropy", "variance", "energy", "autocorr"]

    matrix = []

    for m in metrics:
        matrix.append([results[n][m] for n in results])

    matrix = np.array(matrix)

    corr = np.corrcoef(matrix)

    print("\n=== CORRELATION MATRIX ===")
    print("Metrics:", metrics)
    print(np.round(corr, 3))


# ===============================
# 7. MAIN
# ===============================

def main():
    xi = np.linspace(0, 20, 500)
    taus = [0.5, 1, 2, 5, 10, 20]

    signals = generate_signals(xi)

    results = {}

    print("\n=== COMPARISON TABLE ===\n")
    print(f"{'Signal':15} | b_std | Entropy | Var | Energy | Autocorr")
    print("-" * 70)

    for name, u in signals.items():
        metrics = analyze_signal(xi, u)
        results[name] = metrics

        print(f"{name:15} | "
              f"{metrics['b_std']:.3f} | "
              f"{metrics['entropy']:.3f} | "
              f"{metrics['variance']:.3f} | "
              f"{metrics['energy']:.3f} | "
              f"{metrics['autocorr']:.3f}")

    # Figures
    plot_comparison(results)
    plot_spectra(xi, signals, taus)
    plot_b_timeseries(results)

    # Correlation
    compute_correlation(results)

    plt.show()


if __name__ == "__main__":
    main()