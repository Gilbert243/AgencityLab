"""Deprecated historical field-solver compatibility boundary.

AgencityLab 1.1.2 exposes explicit deterministic research simulators under
``agencitylab.fields.dynamics``. The old generic ``solve_field`` placeholder is
not assigned a hidden default equation.
"""


def solve_field(*args, **kwargs):
    """Reject the retired generic solver placeholder.

    Use ``simulate_klein_gordon()``, ``simulate_dissipative_klein_gordon()``,
    or ``simulate_tdgl()`` explicitly.
    """

    del args, kwargs
    raise NotImplementedError(
        "solve_field() is a retired ambiguous compatibility placeholder; choose "
        "simulate_klein_gordon(), simulate_dissipative_klein_gordon(), or "
        "simulate_tdgl() explicitly."
    )
