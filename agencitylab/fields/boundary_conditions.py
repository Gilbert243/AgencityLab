"""Reserved boundary conditions for the v1.2 dynamical-field research milestone."""


def zero_boundary(*args, **kwargs):
    """Reject the historical identity boundary-condition placeholder in v1.1."""
    del args, kwargs
    raise NotImplementedError(
        "Dynamical Agencity field boundary conditions are not implemented in v1.1; "
        "they are reserved for the v1.2 research milestone."
    )
