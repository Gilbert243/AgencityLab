"""Reserved coherent structures for the v1.2 dynamical-field research milestone."""


def detect_domain_walls(*args, **kwargs):
    """Reject the historical empty domain-wall detector placeholder in v1.1."""
    del args, kwargs
    raise NotImplementedError(
        "Domain walls and coherent structures are not implemented in v1.1; "
        "they are reserved for the v1.2 research milestone."
    )
