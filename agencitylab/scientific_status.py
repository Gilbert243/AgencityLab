"""Scientific classification used by AgencityLab extensions.

The status describes the epistemic position of a model relative to the accepted
Theory of Agencity.  It is deliberately independent from software maturity,
test status, release stability, or implementation quality.
"""

from __future__ import annotations

from enum import Enum


class ScientificStatus(str, Enum):
    """Stable scientific-status taxonomy for AgencityLab models and extensions.

    ``canonical``
        Accepted theory definition and reference implementation.
    ``experimental``
        Direct computational extension intended for evaluation without becoming
        a new canonical law.
    ``research``
        Mathematical model proposed by the theory volumes and implemented for
        simulation or study without established empirical validation.
    ``speculative``
        More strongly hypothetical theoretical extension, including quantum
        agencity, agentons, and fundamental cosmological applications.

    These values never encode whether code is stable, tested, or production-ready.
    """

    CANONICAL = "canonical"
    EXPERIMENTAL = "experimental"
    RESEARCH = "research"
    SPECULATIVE = "speculative"

    def __str__(self) -> str:
        """Serialize naturally as the stable lower-case scientific label."""
        return self.value
