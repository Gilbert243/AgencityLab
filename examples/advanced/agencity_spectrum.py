"""Executable multiscale AgencityLab example using the public API."""

from __future__ import annotations

import numpy as np

from agencitylab import compute_agencity
from agencitylab.api import compute_agencity_spectrum


def main() -> None:
    xi = np.linspace(0.0, 20.0, 801)
    u = np.sin(xi) + 0.3 * np.cos(3.0 * xi)

    result = compute_agencity(
        u=u,
        xi=xi,
        A_ref=1.0,
        tau=2.0,
        w=1.5,
        P_c=2.0,
    )
    print("single-scale mean |b|:", result.b_mean)

    taus = [0.5, 1.0, 2.0, 4.0]
    spectrum = compute_agencity_spectrum(
        u,
        xi,
        taus,
        A_ref=1.0,
        P_c=2.0,
    )

    print("tau:", spectrum["tau"])
    print("w:", spectrum["w"])
    print("mean |b| by scale:", spectrum["b_mean"])
    print("window mode:", spectrum["window_mode"])


if __name__ == "__main__":
    main()
