import numpy as np

from agencitylab.analysis.signature import (
    agencity_signature,
    print_signature,
)


def main():
    xi = np.linspace(0, 20, 500)

    # Test signal
    u = np.sin(xi) + 0.3 * np.cos(3 * xi)

    taus = [0.5, 1, 2, 5, 10, 20, 30]

    sig = agencity_signature(xi, u, taus)

    print_signature(sig)


if __name__ == "__main__":
    main()