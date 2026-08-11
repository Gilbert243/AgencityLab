"""Reserved variational field theory for the v1.2 research milestone."""


def action(*args, **kwargs):
    """Reject the historical ungrounded variational-action placeholder in v1.1."""
    del args, kwargs
    raise NotImplementedError(
        "A variational dynamical Agencity field is not implemented in v1.1; "
        "it is reserved for the v1.2 research milestone."
    )
