"""Reserved dynamical-field equations for the v1.2 research milestone."""


def field_rhs(*args, **kwargs):
    """Reject the historical ungrounded field-equation placeholder in v1.1."""
    del args, kwargs
    raise NotImplementedError(
        "Dynamical Agencity field equations are not implemented in v1.1; "
        "they are reserved for the v1.2 research milestone."
    )
