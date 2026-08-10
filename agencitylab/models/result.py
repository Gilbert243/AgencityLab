"""
result.py

Result model for AgencityLab.

AgencityResult is the canonical container returned by the
AgencityLab computation pipeline.

It stores:
    - raw observable signal,
    - normalized signal,
    - activation/activity,
    - CRM-derived operators,
    - structured agencity quantities,
    - physical canonical parameters,
    - metadata and analysis artifacts.

This version is:
    - complex-aware,
    - serialization-safe,
    - multiscale-ready,
    - physically contextualized.
"""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
    is_dataclass,
)

from pathlib import Path

from typing import (
    Any,
    Dict,
    Optional,
)

import numpy as np

from .metadata import ExperimentMetadata


# ============================================================
# SERIALIZATION HELPERS
# ============================================================

def _serialize(value: Any) -> Any:
    """
    Recursively serialize:
        - numpy arrays,
        - complex numbers,
        - dataclasses,
        - nested structures.
    """

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _serialize(value.to_dict())

    if is_dataclass(value):
        return _serialize(asdict(value))

    if isinstance(value, np.ndarray):

        if np.iscomplexobj(value):

            return {
                "__complex_array__": True,
                "real": value.real.tolist(),
                "imag": value.imag.tolist(),
            }

        return value.tolist()

    if isinstance(value, complex):

        return {
            "__complex__": True,
            "real": float(value.real),
            "imag": float(value.imag),
        }

    if isinstance(value, np.complexfloating):

        c = complex(value)

        return {
            "__complex__": True,
            "real": float(c.real),
            "imag": float(c.imag),
        }

    if isinstance(value, (np.floating, np.integer)):
        return value.item()

    if isinstance(value, dict):
        return {
            str(k): _serialize(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_serialize(v) for v in value]

    return value


def _deserialize(value: Any) -> Any:
    """
    Deserialize serialized structures.
    """

    if isinstance(value, dict):

        if value.get("__complex__"):

            return complex(
                value["real"],
                value["imag"],
            )

        if value.get("__complex_array__"):

            real = np.asarray(
                value["real"],
                dtype=float,
            )

            imag = np.asarray(
                value["imag"],
                dtype=float,
            )

            return real + 1j * imag

        return {
            k: _deserialize(v)
            for k, v in value.items()
        }

    if isinstance(value, list):

        return np.asarray([
            _deserialize(v)
            for v in value
        ])

    return value


# ============================================================
# RESULT MODEL
# ============================================================

@dataclass(slots=True)
class AgencityResult:
    """
    Canonical Agencity result container.
    """

    # ========================================================
    # RAW SIGNALS
    # ========================================================

    xi: np.ndarray

    u: np.ndarray

    u_star: np.ndarray

    X_star: np.ndarray

    A_star: np.ndarray

    t_star: np.ndarray

    # ========================================================
    # STRUCTURAL PARAMETERS
    # ========================================================

    tau: float

    P_c: Any

    A_ref: float = 1.0

    A_fact: float = 1.0

    resolution_scale: Optional[float] = None

    # ========================================================
    # STRUCTURAL OPERATORS
    # ========================================================

    M: np.ndarray = field(default_factory=lambda: np.array([]))

    O: np.ndarray = field(default_factory=lambda: np.array([]))

    D: np.ndarray = field(default_factory=lambda: np.array([]))

    S: np.ndarray = field(default_factory=lambda: np.array([]))

    J: np.ndarray = field(default_factory=lambda: np.array([]))

    U: np.ndarray = field(default_factory=lambda: np.array([]))

    beta: np.ndarray = field(default_factory=lambda: np.array([]))

    b_reduced: np.ndarray = field(default_factory=lambda: np.array([]))

    b: np.ndarray = field(default_factory=lambda: np.array([]))

    theta: Optional[np.ndarray] = None

    # ========================================================
    # PHYSICAL CONTEXT
    # ========================================================

    unit: str = ""

    observable_kind: str = ""

    domain: str = ""

    system_type: str = ""

    mechanism: str = ""

    # ========================================================
    # SCIENTIFIC CONTEXT
    # ========================================================

    metadata: ExperimentMetadata = field(
        default_factory=ExperimentMetadata
    )

    config: Dict[str, Any] = field(
        default_factory=dict
    )

    analysis: Dict[str, Any] = field(
        default_factory=dict
    )

    signature: Optional[dict] = None

    multiscale: Optional[Any] = None

    report_text: Optional[str] = None

    # ========================================================
    # VALIDATION
    # ========================================================

    def __post_init__(self):

        # ----------------------------------------------------
        # arrays
        # ----------------------------------------------------

        self.xi = np.asarray(self.xi, dtype=float)

        self.u = np.asarray(self.u, dtype=float)

        self.u_star = np.asarray(self.u_star)

        self.X_star = np.asarray(self.X_star)

        self.A_star = np.asarray(self.A_star)

        self.t_star = np.asarray(
            self.t_star,
            dtype=float,
        )

        self.M = np.asarray(self.M)

        self.O = np.asarray(self.O)

        self.D = np.asarray(self.D)

        self.S = np.asarray(self.S)

        self.J = np.asarray(self.J)

        self.U = np.asarray(self.U)

        self.beta = np.asarray(self.beta)

        self.b_reduced = np.asarray(self.b_reduced)

        self.b = np.asarray(self.b)

        self.P_c = np.asarray(self.P_c)

        # ----------------------------------------------------
        # theta
        # ----------------------------------------------------

        if self.theta is None:

            if self.U.size:

                self.theta = np.unwrap(
                    np.angle(self.U)
                )

            else:

                self.theta = np.unwrap(
                    np.angle(self.b)
                )

        else:

            self.theta = np.asarray(
                self.theta,
                dtype=float,
            )

        # ----------------------------------------------------
        # lengths
        # ----------------------------------------------------

        n = self.xi.shape[0]

        arrays = (
            "u",
            "u_star",
            "X_star",
            "A_star",
            "t_star",
            "M",
            "O",
            "D",
            "S",
            "J",
            "U",
            "beta",
            "b_reduced",
            "b",
            "theta",
        )

        for name in arrays:

            arr = getattr(self, name)

            if arr.ndim != 1:

                raise ValueError(
                    f"{name} must be 1D"
                )

            if arr.shape[0] != n:

                raise ValueError(
                    f"{name} length mismatch"
                )

        # ----------------------------------------------------
        # scalar validation
        # ----------------------------------------------------

        if float(self.tau) <= 0:

            raise ValueError(
                "tau must be positive"
            )

        if float(self.A_ref) <= 0:

            raise ValueError(
                "A_ref must be positive"
            )

        if float(self.A_fact) <= 0:

            raise ValueError(
                "A_fact must be positive"
            )

    # ========================================================
    # DERIVED PROPERTIES
    # ========================================================

    @property
    def b_abs(self):

        return np.abs(self.b)

    @property
    def beta_abs(self):

        return np.abs(self.beta)

    @property
    def U_abs(self):

        return np.abs(self.U)

    @property
    def eta(self):

        eps = 1e-12

        return self.b_abs / np.maximum(
            np.abs(self.P_c),
            eps,
        )

    @property
    def b_mean(self):

        if not self.b.size:
            return 0.0

        return float(
            np.mean(self.b_abs)
        )

    @property
    def beta_mean(self):

        if not self.beta.size:
            return 0.0

        return float(
            np.mean(self.beta_abs)
        )

    @property
    def theta_mean(self):

        if not self.theta.size:
            return 0.0

        return float(
            np.mean(self.theta)
        )

    @property
    def theta_std(self):

        if not self.theta.size:
            return 0.0

        return float(
            np.std(self.theta)
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self):

        return {

            "n_samples":
                int(self.xi.size),

            "tau":
                float(self.tau),

            "Pc_mean":
                float(
                    np.mean(np.abs(self.P_c))
                ),

            "A_ref":
                float(self.A_ref),

            "A_fact":
                float(self.A_fact),

            "resolution_scale":
                self.resolution_scale,

            "b_mean":
                self.b_mean,

            "b_std":
                float(np.std(self.b_abs)),

            "b_peak":
                float(np.max(self.b_abs)),

            "beta_mean":
                self.beta_mean,

            "beta_max":
                float(np.max(self.beta_abs)),

            "J_mean":
                float(np.mean(self.J)),

            "D_mean":
                float(np.mean(self.D)),

            "S_mean":
                float(np.mean(self.S)),

            "M_mean":
                float(np.mean(self.M)),

            "O_mean":
                float(np.mean(self.O)),

            "theta_mean":
                self.theta_mean,

            "theta_std":
                self.theta_std,

            "unit":
                self.unit,

            "observable_kind":
                self.observable_kind,

            "domain":
                self.domain,

            "system_type":
                self.system_type,

            "mechanism":
                self.mechanism,
        }

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(self):

        return {

            "xi":
                _serialize(self.xi),

            "u":
                _serialize(self.u),

            "u_star":
                _serialize(self.u_star),

            "X_star":
                _serialize(self.X_star),

            "A_star":
                _serialize(self.A_star),

            "t_star":
                _serialize(self.t_star),

            "tau":
                float(self.tau),

            "P_c":
                _serialize(self.P_c),

            "A_ref":
                float(self.A_ref),

            "A_fact":
                float(self.A_fact),

            "resolution_scale":
                self.resolution_scale,

            "M":
                _serialize(self.M),

            "O":
                _serialize(self.O),

            "D":
                _serialize(self.D),

            "S":
                _serialize(self.S),

            "J":
                _serialize(self.J),

            "U":
                _serialize(self.U),

            "beta":
                _serialize(self.beta),

            "b_reduced":
                _serialize(self.b_reduced),

            "b":
                _serialize(self.b),

            "theta":
                _serialize(self.theta),

            "unit":
                self.unit,

            "observable_kind":
                self.observable_kind,

            "domain":
                self.domain,

            "system_type":
                self.system_type,

            "mechanism":
                self.mechanism,

            "metadata":
                self.metadata.to_dict(),

            "config":
                dict(self.config),

            "analysis":
                _serialize(self.analysis),

            "signature":
                _serialize(self.signature),

            "multiscale":
                _serialize(self.multiscale),

            "report_text":
                self.report_text,
        }

    # ========================================================
    # DESERIALIZATION
    # ========================================================

    @classmethod
    def from_dict(cls, data):

        data = dict(data or {})

        metadata = data.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            ExperimentMetadata,
        ):
            metadata = ExperimentMetadata.from_dict(
                metadata
            )

        def arr(key, default=None):

            if key not in data:
                return default

            return np.asarray(
                _deserialize(data[key])
            )

        xi = arr("xi")

        if xi is None:

            raise ValueError(
                "xi is required"
            )

        n = len(xi)

        def default_array(
            key,
            value=0.0,
            dtype=float,
        ):

            if key in data:

                return np.asarray(
                    _deserialize(data[key]),
                    dtype=dtype,
                )

            return np.full(
                n,
                value,
                dtype=dtype,
            )

        return cls(

            xi=np.asarray(
                xi,
                dtype=float,
            ),

            u=default_array("u"),

            u_star=default_array("u_star"),

            X_star=default_array("X_star"),

            A_star=default_array("A_star"),

            t_star=default_array("t_star"),

            tau=float(
                data.get("tau", 1.0)
            ),

            P_c=np.asarray(
                _deserialize(
                    data.get("P_c", 1.0)
                )
            ),

            A_ref=float(
                data.get("A_ref", 1.0)
            ),

            A_fact=float(
                data.get("A_fact", 1.0)
            ),

            resolution_scale=data.get(
                "resolution_scale"
            ),

            M=default_array("M"),

            O=default_array("O"),

            D=default_array("D"),

            S=default_array("S"),

            J=default_array("J"),

            U=default_array(
                "U",
                dtype=complex,
            ),

            beta=default_array(
                "beta",
                dtype=complex,
            ),

            b_reduced=default_array(
                "b_reduced",
                dtype=complex,
            ),

            b=default_array(
                "b",
                dtype=complex,
            ),

            theta=default_array("theta"),

            unit=data.get("unit", ""),

            observable_kind=data.get(
                "observable_kind",
                "",
            ),

            domain=data.get(
                "domain",
                "",
            ),

            system_type=data.get(
                "system_type",
                "",
            ),

            mechanism=data.get(
                "mechanism",
                "",
            ),

            metadata=metadata,

            config=dict(
                data.get("config", {})
            ),

            analysis=dict(
                _deserialize(
                    data.get(
                        "analysis",
                        {},
                    )
                )
            ),

            signature=_deserialize(
                data.get(
                    "signature",
                    None,
                )
            ),

            multiscale=_deserialize(
                data.get(
                    "multiscale",
                    None,
                )
            ),

            report_text=data.get(
                "report_text"
            ),
        )

    # ========================================================
    # EXPORTS
    # ========================================================

    def to_dataframe(self):

        try:
            import pandas as pd

        except ImportError as exc:

            raise ImportError(
                "pandas required"
            ) from exc

        return pd.DataFrame({

            "xi":
                self.xi,

            "u":
                self.u,

            "u_star":
                self.u_star,

            "X_star":
                self.X_star,

            "A_star":
                self.A_star,

            "M":
                self.M,

            "O":
                self.O,

            "D":
                self.D,

            "S":
                self.S,

            "J":
                self.J,

            "theta":
                self.theta,

            "beta_real":
                np.real(self.beta),

            "beta_imag":
                np.imag(self.beta),

            "beta_abs":
                np.abs(self.beta),

            "b_real":
                np.real(self.b),

            "b_imag":
                np.imag(self.b),

            "b_abs":
                np.abs(self.b),
        })

    # ========================================================
    # XARRAY
    # ========================================================

    def to_xarray(self):

        try:
            import xarray as xr

        except ImportError as exc:

            raise ImportError(
                "xarray required"
            ) from exc

        return xr.Dataset(

            data_vars={

                "u":
                    ("xi", self.u),

                "u_star":
                    ("xi", self.u_star),

                "X_star":
                    ("xi", self.X_star),

                "A_star":
                    ("xi", self.A_star),

                "M":
                    ("xi", self.M),

                "O":
                    ("xi", self.O),

                "D":
                    ("xi", self.D),

                "S":
                    ("xi", self.S),

                "J":
                    ("xi", self.J),

                "theta":
                    ("xi", self.theta),

                "beta_abs":
                    ("xi", np.abs(self.beta)),

                "b_abs":
                    ("xi", np.abs(self.b)),
            },

            coords={
                "xi": self.xi
            },

            attrs={

                "tau":
                    float(self.tau),

                "A_ref":
                    float(self.A_ref),

                "A_fact":
                    float(self.A_fact),

                "metadata":
                    self.metadata.to_dict(),

                "summary":
                    self.summary(),
            },
        )

    # ========================================================
    # SAVE / LOAD
    # ========================================================

    def save_json(
        self,
        path,
    ):

        from agencitylab.io.save import save

        return save(
            self.to_dict(),
            path,
        )

    @classmethod
    def load_json(
        cls,
        path,
    ):

        from agencitylab.io.load import load

        payload = load(path)

        return cls.from_dict(
            payload
        )

    # ========================================================
    # ATTACHMENTS
    # ========================================================

    def attach_analysis(
        self,
        analysis,
    ):

        self.analysis = dict(
            analysis or {}
        )

        return self

    def attach_report(
        self,
        report_text,
    ):

        self.report_text = report_text

        return self