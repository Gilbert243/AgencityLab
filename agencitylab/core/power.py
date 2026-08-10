"""
power.py

Characteristic power utilities for AgencityLab.

IMPORTANT
---------
In the stabilized Agencity theory:

    Pc is a structural property
    of the containing physical system.

Pc should therefore come from:
    - physical models,
    - canonical registries,
    - system metadata,
    - energetic scales.

It should NOT fundamentally depend on u(t).
"""

from __future__ import annotations

import numpy as np

from .safeguards import (
    ensure_positive,
)

from agencitylab.constants.characteristic_powers import (
    resolve_characteristic_power,
)


# ============================================================
# CANONICAL API
# ============================================================

def characteristic_power(
    value=None,
    *,
    system=None,
    domain=None,
    reference_energy=None,
    tau=None,
    A_ref=None,
    default: float = 1.0,
    verbose: bool = False,
):
    """
    Resolve characteristic power Pc.

    Priority
    --------
    1) explicit value
    2) reference_energy / tau
    3) canonical registry
    4) A_ref^2 / tau
    5) default
    """

    if verbose:
        print("[power] resolving Pc")

    # ========================================================
    # explicit
    # ========================================================

    if value is not None:

        if verbose:
            print("[power] explicit value")

        return ensure_positive(value)

    # ========================================================
    # energy / tau
    # ========================================================

    if (
        reference_energy is not None
        and tau is not None
    ):

        if verbose:
            print("[power] using E_ref / tau")

        return ensure_positive(
            np.asarray(
                reference_energy,
                dtype=float,
            )
            / np.asarray(
                tau,
                dtype=float,
            )
        )

    # ========================================================
    # canonical registry
    # ========================================================

    Pc = resolve_characteristic_power(
        system=system,
        domain=domain,
        Pc="auto",
        default=default,
    )

    if Pc is not None:

        Pc = ensure_positive(Pc)

        if verbose:
            print(f"[power] canonical Pc={Pc}")

        return Pc

    # ========================================================
    # A_ref² / tau
    # ========================================================

    if (
        A_ref is not None
        and tau is not None
    ):

        if verbose:
            print("[power] using A_ref² / tau")

        A_ref = ensure_positive(A_ref)

        tau = ensure_positive(tau)

        return ensure_positive(
            (A_ref ** 2) / tau
        )

    # ========================================================
    # fallback
    # ========================================================

    if verbose:
        print("[power] using default")

    return ensure_positive(default)


# ============================================================
# OPTIONAL HEURISTIC ESTIMATION
# ============================================================

def estimate_characteristic_power(
    signal,
    *,
    tau,
    method: str = "rms",
    scale=None,
    A_ref=None,
    verbose: bool = False,
):
    """
    Experimental heuristic estimation of Pc.

    WARNING
    -------
    This function is NOT part of the canonical theory.

    It derives energetic scales from the observable signal.
    """

    x = np.asarray(
        signal,
        dtype=float,
    )

    tau = ensure_positive(tau)

    if verbose:
        print(
            f"[power] heuristic estimation "
            f"method={method}"
        )

    # ========================================================
    # A_ref override
    # ========================================================

    if A_ref is not None:

        scale = ensure_positive(A_ref)

    # ========================================================
    # derive scale
    # ========================================================

    elif scale is None:

        if method == "rms":

            scale = np.sqrt(
                np.nanmean(
                    np.square(x)
                )
            )

        elif method == "variance":

            scale = np.sqrt(
                np.nanvar(x)
            )

        elif method == "amplitude":

            scale = (
                np.nanmax(x)
                - np.nanmin(x)
            )

        else:

            raise ValueError(
                "Unknown power estimation method"
            )

    scale = ensure_positive(scale)

    P_c = ensure_positive(
        (scale ** 2) / tau
    )

    if verbose:
        print(f"[power] estimated Pc={P_c}")

    return P_c