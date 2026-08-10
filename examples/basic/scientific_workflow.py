"""Reproducible signal -> result -> diagnostic -> figure -> export workflow."""

from __future__ import annotations

import numpy as np

from agencitylab import scientific_workflow


def main() -> None:
    xi = np.linspace(0.0, 40.0, 1601)
    u = np.sin(xi) + 0.15 * np.sin(2.0 * xi)

    study = scientific_workflow(
        u,
        xi,
        A_ref=1.0,
        tau=2.0,
        w=1.5,
        P_c=5.0,
        unit="rad",
        coordinate_unit="s",
        power_unit="W",
        export_dir="agencity_output",
        show=False,
    )

    print(study.report)
    print("\nCreated files:")
    for name, path in study.exports.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
