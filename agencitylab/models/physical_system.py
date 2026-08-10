from dataclasses import dataclass


@dataclass
class PhysicalSystem:

    tau: float

    Pc: float

    activity_factor: float = 1.0

    domain: str = "generic"

    mechanism: str = "passive"