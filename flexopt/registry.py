"""Registry for user-provided system models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


SystemCallable = Callable[[np.ndarray], np.ndarray]

_SYSTEMS: dict[str, SystemCallable] = {}


def register_system(name: str, system_fn: SystemCallable) -> None:
    """Register a callable that maps input arrays to output arrays."""
    _SYSTEMS[name] = system_fn


def get_system(name: str) -> SystemCallable:
    """Return a registered system callable."""
    try:
        return _SYSTEMS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_SYSTEMS)) or "<none>"
        raise KeyError(f"Unknown system '{name}'. Registered systems: {known}") from exc
