"""
General-purpose utilities for AgencityLab.
"""

from .exceptions import AgencityError, DataValidationError, OptionalDependencyError
from .interpolation import interpolate_1d, resample_uniform
from .logging import get_logger, set_log_level
from .math_utils import (
    clip_safely,
    ensure_1d,
    finite_difference,
    moving_average,
    nan_safe_mean,
)
from .profiling import profile_function
from .sliding_window import sliding_window_view_1d

__all__ = [
    "AgencityError",
    "DataValidationError",
    "OptionalDependencyError",
    "clip_safely",
    "ensure_1d",
    "finite_difference",
    "get_logger",
    "interpolate_1d",
    "moving_average",
    "nan_safe_mean",
    "profile_function",
    "resample_uniform",
    "set_log_level",
    "sliding_window_view_1d",
]
