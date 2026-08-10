"""
Core mathematical engine for AgencityLab.

Canonical pipeline
------------------
    u → u* → X* → A* → M → O → (D, S)
      → J → U → β → b

The core layer implements:

    - canonical operators,
    - temporal structure,
    - organization dynamics,
    - agencity computation,
    - coherence analysis,
    - physical scaling,
    - multiscale structure.

IMPORTANT
---------
The stabilized theory distinguishes:

    Structural quantities
    ---------------------
        tau
        Pc
        A_fact

    Observable dynamics
    -------------------
        u(t)
        X(t)
        A(t)
        M(t)
        O(t)
        β(t)
        b(t)

Some functions remain available as
experimental heuristics for exploratory workflows.
"""

# ============================================================
# SIGNAL PREPROCESSING
# ============================================================

from .normalization import (
    center_signal,
    compute_reference_scale,
    normalize_signal,
)

# ============================================================
# DYNAMICS
# ============================================================

from .activation import (
    activation,
    activation_from_signal,
    reduced_coordinate,
)

from .activity import (
    activity,
    activity_from_signal,
)

# ============================================================
# TEMPORAL STRUCTURE
# ============================================================

from .autocorr import (
    autocorrelation,
    normalized_autocorrelation,
)

from .tau import (
    characteristic_time,
    estimate_tau,
)

from .crm import (
    causal_moving_correlation,
    crm_tau,
)

# ============================================================
# MEMORY & ORGANIZATION
# ============================================================

from .memory import (
    memory,
    memory_from_signal,
)

from .organization import (
    organization,
    organization_from_signal,
)

# ============================================================
# COHERENCE
# ============================================================

from .coherence import (

    # structural orientation
    compute_theta,

    # angular utilities
    wrap_angle,
    angular_difference,

    # circular statistics
    circular_mean,
    circular_variance,
    circular_std,
    resultant_length,

    # coherence metrics
    angular_variance,
    phase_coherence,
    directional_stability,

    # matrix analysis
    coherence_matrix,

    # diagnostics
    CoherenceDiagnostic,
    coherence_diagnostic,
)

# ============================================================
# INTENSITIES & STRUCTURE
# ============================================================

from .intensity import (
    compute_dynamic_intensity,
    compute_structural_intensity,
    compute_intensities,
)

from .contrast import (
    compute_contrast,
)

from .orientation import (
    compute_orientation,
    compute_angle,
)

# ============================================================
# AGENCITY CORE
# ============================================================

from .beta import (
    beta,
    structured_agencity,
    compute_beta,
)

from .agencity import (
    agencity,
    agencity_rate,
    decompose_agencity,
    agencity_criteria,
    compute_full_agencity,
)

# ============================================================
# MULTISCALE
# ============================================================

from .multiscale import (
    multiscale_agencity,
)

# ============================================================
# POWER
# ============================================================

from .power import (
    characteristic_power,
    estimate_characteristic_power,
)

# ============================================================
# SAFEGUARDS
# ============================================================

from .safeguards import (
    EPS,
    safe_divide,
    safe_tanh,
    saturate,
    replace_non_finite,
)

# ============================================================
# VALIDATION
# ============================================================

from .validation import (
    as_float_array,
    validate_axis,
    validate_signal,
    validate_window_size,
)

# ============================================================
# PUBLIC API
# ============================================================

__all__ = [

    # ========================================================
    # preprocessing
    # ========================================================

    "center_signal",
    "compute_reference_scale",
    "normalize_signal",

    # ========================================================
    # dynamics
    # ========================================================

    "activation",
    "activation_from_signal",
    "reduced_coordinate",

    "activity",
    "activity_from_signal",

    # ========================================================
    # temporal structure
    # ========================================================

    "autocorrelation",
    "normalized_autocorrelation",

    "characteristic_time",
    "estimate_tau",

    "causal_moving_correlation",
    "crm_tau",

    # ========================================================
    # memory & organization
    # ========================================================

    "memory",
    "memory_from_signal",

    "organization",
    "organization_from_signal",

    # ========================================================
    # coherence
    # ========================================================

    "compute_theta",

    "wrap_angle",
    "angular_difference",

    "circular_mean",
    "circular_variance",
    "circular_std",
    "resultant_length",

    "angular_variance",
    "phase_coherence",
    "directional_stability",

    "coherence_matrix",

    "CoherenceDiagnostic",
    "coherence_diagnostic",

    # ========================================================
    # intensities & structure
    # ========================================================

    "compute_dynamic_intensity",
    "compute_structural_intensity",
    "compute_intensities",

    "compute_contrast",

    "compute_orientation",
    "compute_angle",

    # ========================================================
    # agencity
    # ========================================================

    "beta",
    "structured_agencity",
    "compute_beta",

    "agencity",
    "agencity_rate",
    "decompose_agencity",
    "agencity_criteria",
    "compute_full_agencity",

    # ========================================================
    # multiscale
    # ========================================================

    "multiscale_agencity",

    # ========================================================
    # power
    # ========================================================

    "characteristic_power",
    "estimate_characteristic_power",

    # ========================================================
    # safeguards
    # ========================================================

    "EPS",
    "safe_divide",
    "safe_tanh",
    "saturate",
    "replace_non_finite",

    # ========================================================
    # validation
    # ========================================================

    "as_float_array",
    "validate_axis",
    "validate_signal",
    "validate_window_size",
]