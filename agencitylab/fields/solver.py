"""Reserved dynamical-field solver for the v1.2 research milestone."""


def solve_field(*args, **kwargs):
    """Reject the historical no-op solver placeholder in v1.1."""
    del args, kwargs
    raise NotImplementedError(
        "Dynamical Agencity field solvers are not implemented in v1.1; "
        "they are reserved for the v1.2 research milestone."
    )
