"""Retired historical Agencity dynamical-system compatibility boundary.

This module used to describe an ad-hoc reduced dynamical system as
``canonical``. That description was scientifically incorrect.

The theory states ``beta = J * U`` and ``b = P_c * beta``, while the legacy
implementation computed ``beta`` from products of ``tanh`` factors and formed a
reduced ``b`` from a discrete variation of that quantity. Those equations are
not retained as an executable compatibility path.

The :class:`AgencityState` container remains importable only so old serialized
or user code can identify the historical state shape. For canonical Agencity
computation use :func:`agencitylab.compute_agencity`. Domain-specific dynamical
models must be stated explicitly and must not masquerade as the canonical
observable pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings


@dataclass(slots=True)
class AgencityState:
    """Historical reduced-state container; not a canonical Theory state vector."""

    X_star: float
    A_star: float
    M: float
    O: float
    P_c: float


def _retired(api_name: str):
    warnings.warn(
        f"agencitylab.dynamics.system.{api_name} is retired because the historical "
        "equations were not the canonical Theory of Agencity. Use "
        "agencitylab.compute_agencity for beta = J * U and b = P_c * beta.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise RuntimeError(
        "The theory states beta = J * U and b = P_c * beta, while the legacy "
        "implementation currently represented beta with tanh-based factors and b "
        "with a discrete beta variation. This misleading legacy dynamical model "
        "has been retired rather than preserved as canonical physics."
    )


def beta_from_state(state: AgencityState) -> float:
    """Retired legacy beta construction; use ``compute_agencity`` instead."""

    _retired("beta_from_state")


def agencity_rhs(xi_star, y, constitutive_laws=None):
    """Retired legacy Agencity-specific RHS; no canonical replacement exists."""

    _retired("agencity_rhs")


def default_system_rhs(xi_star, y):
    """Retired legacy default RHS; no canonical dynamical closure is asserted."""

    _retired("default_system_rhs")


def beta_and_b_from_trajectory(xi_star, trajectory, delta_star: float = 1.0):
    """Retired legacy trajectory reduction; use canonical result fields instead."""

    _retired("beta_and_b_from_trajectory")
