"""Deprecated historical field-metrics compatibility boundary.

Field-energy primitives are now defined explicitly in
``agencitylab.fields.physics``. The former generic ``field_energy`` placeholder
had no stable input contract and is not silently mapped to one particular
integration convention.
"""


def field_energy(*args, **kwargs):
    """Reject the retired ungrounded field-energy placeholder.

    Use ``field_energy_density()`` and ``total_field_energy()`` with explicit
    gradient norms and volume elements.
    """

    del args, kwargs
    raise NotImplementedError(
        "field_energy() is a retired compatibility placeholder; use "
        "field_energy_density() and total_field_energy() with explicit inputs."
    )
