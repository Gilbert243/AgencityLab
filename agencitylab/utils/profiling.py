"""
Profiling helpers for AgencityLab.
"""

from __future__ import annotations

from functools import wraps
from time import perf_counter
from typing import Callable, Any, Dict


def profile_function(func: Callable) -> Callable:
    """
    Decorator that records execution time in a return dictionary when possible.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        duration = perf_counter() - start
        if isinstance(result, dict):
            result = dict(result)
            result["profiling_seconds"] = duration
            return result
        return result
    return wrapper
