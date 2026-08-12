"""Reserved explicit variational-action evaluator for future research work.

AgencityLab 1.1.2 implements the quartic potential and classical equations of
motion, but it does not expose a generic discretized action functional. The
historical ``action`` placeholder therefore remains an explicit unsupported
boundary rather than fabricating an action from numerical conventions.
"""


def action(*args, **kwargs):
    """Reject the historical ungrounded variational-action placeholder."""

    del args, kwargs
    raise NotImplementedError(
        "A generic discretized field action is not part of AgencityLab 1.1.2; "
        "use the explicit quartic potential and classical dynamics APIs."
    )
