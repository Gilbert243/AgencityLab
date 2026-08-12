"""Source-defined mathematical extensions beyond the canonical scalar core.

These helpers implement only the portions explicitly defined in Volume 2.
They do not promote experimental extension machinery into the canonical
``u -> beta -> b`` pipeline.
"""

from .intensity_alternatives import (
    INTENSITY_ALTERNATIVES_STATUS,
    printed_offset_ratio_candidate,
    raw_ratio_intensity,
    sum_intensity,
    sum_log_intensity,
)
from .riemannian import (
    RIEMANNIAN_EXTENSION_STATUS,
    riemannian_dynamic_intensity,
    riemannian_inner_product,
    riemannian_speed,
)
from .window_criteria import mean_contrast_criterion, orientational_entropy_criterion

__all__ = [
    "INTENSITY_ALTERNATIVES_STATUS",
    "RIEMANNIAN_EXTENSION_STATUS",
    "mean_contrast_criterion",
    "orientational_entropy_criterion",
    "printed_offset_ratio_candidate",
    "raw_ratio_intensity",
    "riemannian_dynamic_intensity",
    "riemannian_inner_product",
    "riemannian_speed",
    "sum_intensity",
    "sum_log_intensity",
]
