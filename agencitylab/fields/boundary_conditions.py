"""Deprecated historical boundary-condition compatibility boundary.

The supported field boundary contracts are now ``PeriodicBoundary``,
``DirichletBoundary``, and ``NeumannBoundary`` from
``agencitylab.fields.numerics``. The old ``zero_boundary`` function was an
identity placeholder and is not kept as a misleading numerical operation.
"""


def zero_boundary(*args, **kwargs):
    """Reject the retired identity boundary placeholder.

    Use ``DirichletBoundary(value=0.0)`` explicitly when a zero fixed boundary
    is intended.
    """

    del args, kwargs
    raise NotImplementedError(
        "zero_boundary() was a historical identity placeholder; use "
        "DirichletBoundary(value=0.0) explicitly."
    )
