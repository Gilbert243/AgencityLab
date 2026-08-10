"""
Core data models for AgencityLab.

This module exposes the central data structures used across the framework:

- configuration
- experimental context
- metadata (including physical normalization context)
- signals and datasets
- results

These models are designed for:
- scientific reproducibility
- composability
- compatibility with the Agencity theory (A_ref, τ, P_c)
"""

# ============================================================
# CONFIGURATION
# ============================================================

from .config_model import AnalysisConfig

# ============================================================
# CONTEXT (EXPERIMENTAL / PHYSICAL)
# ============================================================

from .context import Context

# ============================================================
# METADATA (INCLUDES A_ref CONTEXT)
# ============================================================

from .metadata import ExperimentMetadata

# ============================================================
# DATA STRUCTURES
# ============================================================

from .signal import AgencitySignal
from .dataset import AgencityDataset
from .experiment import AgencityExperiment

# ============================================================
# RESULTS
# ============================================================

from .result import AgencityResult

# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    # config
    "AnalysisConfig",

    # context
    "Context",

    # metadata
    "ExperimentMetadata",

    # data
    "AgencitySignal",
    "AgencityDataset",
    "AgencityExperiment",

    # result
    "AgencityResult",
]