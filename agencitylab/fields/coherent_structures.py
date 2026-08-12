"""Deprecated historical coherent-structure compatibility boundary.

AgencityLab 1.1.2 provides research reference structures under
``agencitylab.fields.coherent``. The old ``detect_domain_walls`` placeholder
never implemented a scientifically defined detector, so it remains retired
rather than being silently mapped to a profile generator or heuristic.
"""


def detect_domain_walls(*args, **kwargs):
    """Reject the retired empty domain-wall detector placeholder.

    Use ``domain_wall_profile()`` / ``domain_wall_residual()`` for the explicit
    real-sector reference solution. Detection heuristics, if added later, must
    be separately specified and must not be confused with those references.
    """

    del args, kwargs
    raise NotImplementedError(
        "detect_domain_walls() was an empty historical detector placeholder; "
        "use agencitylab.fields.coherent reference functions explicitly."
    )
