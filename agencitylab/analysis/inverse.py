"""Recoverable macroscopic information from the Agencity flux.

Volume 2 Chapter 11 proves that ``u -> b`` is non-injective.  These helpers
therefore recover only the information that the source states is available from
``b`` when characteristic power is known: ``beta``, ``|J|`` and the structural
direction modulo a sign.  They never attempt to reconstruct the observable
``u`` or the individual magnitudes ``D`` and ``S``.
"""

from __future__ import annotations

import numpy as np


def recoverable_agencity_signature(b, P_c) -> dict[str, object]:
    """Recover the Chapter-11 macroscopic signature from ``b`` and ``P_c``.

    ``P_c`` must be strictly positive at every requested sample.  When
    ``P_c = 0``, the forward relation maps every intrinsic state to ``b = 0``
    and ``beta`` is therefore not recoverable from the flux alone.

    The returned ``orientation_mod_pi`` represents the direction of the
    structural vector ``(M, O)`` up to sign.  It is undefined when ``beta = 0``
    because a zero flux cannot reveal whether the cause was ``S = 0`` or
    ``J = 0``.  ``absolute_contrast`` is exactly ``|J| = |beta|`` wherever the
    canonical structural direction exists.
    """
    b_arr, pc_arr = np.broadcast_arrays(
        np.asarray(b, dtype=complex),
        np.asarray(P_c, dtype=float),
    )
    if not np.all(np.isfinite(b_arr)):
        raise ValueError("b must contain only finite values")
    if not np.all(np.isfinite(pc_arr)):
        raise ValueError("P_c must contain only finite values")
    if np.any(pc_arr <= 0.0):
        raise ValueError(
            "P_c must be strictly positive to recover beta from b; P_c=0 erases intrinsic-state information"
        )

    beta = b_arr / pc_arr
    absolute_contrast = np.abs(beta)
    direction_defined = absolute_contrast > 0.0
    orientation_mod_pi = np.full(beta.shape, np.nan, dtype=float)
    orientation_mod_pi[direction_defined] = np.mod(
        np.angle(beta[direction_defined]),
        np.pi,
    )

    if beta.ndim == 0:
        return {
            "beta": beta.item(),
            "absolute_contrast": float(absolute_contrast),
            "orientation_mod_pi": float(orientation_mod_pi),
            "direction_defined": bool(direction_defined),
        }
    return {
        "beta": beta,
        "absolute_contrast": absolute_contrast,
        "orientation_mod_pi": orientation_mod_pi,
        "direction_defined": direction_defined,
    }


__all__ = ["recoverable_agencity_signature"]
