"""
memory.py

Memory operator M.

Canonical theory:
    M(t*) = tanh(CRM(A*)(t*))

where:
    CRM uses:
        w = tau / A_fact

Definitions
-----------
tau :
    Structural characteristic time of the containing system.

A_fact :
    Structural activity factor associated with the dominant
    physical organization mechanism.

Examples:
    convection  -> Nusselt number
    oscillator  -> quality factor Q
    flow        -> Reynolds or Peclet number

Important
---------
tau and A_fact are structural parameters and must remain
independent from the observable signal u(t).
"""

from __future__ import annotations

import numpy as np

from .crm import (
    causal_moving_correlation,
)

from .safeguards import (
    replace_non_finite,
    safe_tanh,
)

from .validation import (
    validate_signal,
    validate_axis,
    validate_positive_scalar,
)


# ============================================================
# MEMORY
# ============================================================

def memory(
    activity_signal,
    tau,
    *,
    axis,
    activity_factor="auto",
    mechanism=None,
    domain=None,
    system_type=None,
    return_correlation: bool = False,
    verbose: bool = False,
):
    """
    Compute memory operator M.

    Parameters
    ----------
    activity_signal :
        Reduced activity signal A*.

    tau :
        Structural characteristic time.

    axis :
        Observation coordinate.

    activity_factor :
        Structural activity factor A_fact.

        - "auto" -> resolved automatically
        - float  -> explicit value

    mechanism :
        Dominant physical mechanism.

    domain :
        Scientific domain.

    system_type :
        Structural system category.

    return_correlation :
        If True, also return raw CRM.

    verbose :
        Enable diagnostics.

    Returns
    -------
    mem :
        Memory operator M.

    or

    (mem, crm)
    """

    x = validate_signal(
        activity_signal,
        name="activity_signal",
    ).ravel()

    axis = validate_axis(
        axis,
        expected_length=len(x),
        name="axis",
    )

    tau = validate_positive_scalar(
        tau,
        name="tau",
    )

    if verbose:

        print(
            "[memory] "
            "Computing CRM on activity"
        )

    crm = causal_moving_correlation(
        x,
        tau=tau,
        axis=axis,
        activity_factor=activity_factor,
        mechanism=mechanism,
        domain=domain,
        system_type=system_type,
        verbose=verbose,
    )

    crm = replace_non_finite(
        crm,
        default=0.0,
    )

    if verbose:

        print(
            "[memory] "
            "Applying tanh compression"
        )

    mem = safe_tanh(crm)

    mem = replace_non_finite(
        mem,
        default=0.0,
    )

    if verbose:

        print(
            "[memory] "
            f"mean={np.mean(mem):.6f}"
        )

        print(
            "[memory] "
            f"std={np.std(mem):.6f}"
        )

    if return_correlation:
        return mem, crm

    return mem


# ============================================================
# PIPELINE ALIAS
# ============================================================

def memory_from_signal(
    *args,
    **kwargs,
):
    """
    Alias for pipeline readability.
    """
    return memory(
        *args,
        **kwargs,
    )