"""Reserved dynamical-field metrics for the v1.2 research milestone."""


def field_energy(*args, **kwargs):
    """Reject the historical ungrounded field-energy placeholder in v1.1."""
    del args, kwargs
    raise NotImplementedError(
        "Dynamical Agencity field energy is not defined in v1.1; "
        "it is reserved for the v1.2 research milestone."
    )
