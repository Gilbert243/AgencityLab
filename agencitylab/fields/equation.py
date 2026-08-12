"""Deprecated historical field-equation compatibility boundary.

AgencityLab 1.1.2 provides explicit research equations under
``agencitylab.fields.dynamics``. The old generic ``field_rhs`` placeholder had
no unambiguous theoretical meaning and is intentionally not mapped to one of
those equations silently.
"""


def field_rhs(*args, **kwargs):
    """Reject the retired ambiguous field-equation placeholder.

    Use ``klein_gordon_acceleration()``,
    ``dissipative_klein_gordon_acceleration()``, or ``tdgl_rhs()`` explicitly.
    """

    del args, kwargs
    raise NotImplementedError(
        "field_rhs() is a retired ambiguous compatibility placeholder; choose "
        "klein_gordon_acceleration(), dissipative_klein_gordon_acceleration(), "
        "or tdgl_rhs() explicitly."
    )
