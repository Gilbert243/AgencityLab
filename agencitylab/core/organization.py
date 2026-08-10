"""
organization.py

Organization operator O.

Canonical theory:
    O(t*) = tanh(CRM(X*)(t*))

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
# ORGANIZATION
# ============================================================

def organization(
    activation_signal,
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
    Compute organization operator O.

    Parameters
    ----------
    activation_signal :
        Reduced activation signal X*.

    tau :
        Structural characteristic time.

    axis :
        Observation coordinate.

    activity_factor :
        Structural activity factor A_fact.

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
    org :
        Organization operator O.

    or

    (org, crm)
    """

    x = validate_signal(
        activation_signal,
        name="activation_signal",
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
            "[organization] "
            "Computing CRM on activation"
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
            "[organization] "
            "Applying tanh compression"
        )

    org = safe_tanh(crm)

    org = replace_non_finite(
        org,
        default=0.0,
    )

    if verbose:

        print(
            "[organization] "
            f"mean={np.mean(org):.6f}"
        )

        print(
            "[organization] "
            f"std={np.std(org):.6f}"
        )

    if return_correlation:
        return org, crm

    return org


# ============================================================
# PIPELINE ALIAS
# ============================================================

def organization_from_signal(
    *args,
    **kwargs,
):
    """
    Alias for pipeline readability.
    """
    return organization(
        *args,
        **kwargs,
    )