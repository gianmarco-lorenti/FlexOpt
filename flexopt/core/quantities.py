"""Quantity-related domain models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Bounds:
    """Store lower and upper numeric bounds.

    Parameters
    ----------
    lower:
        Optional lower bound. ``None`` means no lower bound is enforced.
    upper:
        Optional upper bound. ``None`` means no upper bound is enforced.

    Returns
    -------
    None
        This dataclass stores the validated bounds.
    """

    lower: Optional[float] = None
    upper: Optional[float] = None

    @classmethod
    def from_tuple(cls, tupl: tuple) -> Bounds:
        """Build bounds from a pair of lower and upper values."""
        if not isinstance(tupl, tuple):
            raise TypeError("Bounds must be specified as a tuple of (lower, upper).")
        if len(tupl) != 2:
            raise ValueError("Bound specifications must contain two values.")
        return cls(lower=tupl[0], upper=tupl[1])

    def __post_init__(self) -> None:
        # Allow open bounds while validating any finite numeric limits
        if self.lower is not None and (
            not np.isscalar(self.lower) or not isinstance(self.lower, Number)
        ):
            raise TypeError("`lower` must be a numeric scalar or None.")
        if self.upper is not None and (
            not np.isscalar(self.upper) or not isinstance(self.upper, Number)
        ):
            raise TypeError("`upper` must be a numeric scalar or None.")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            raise ValueError("`lower` cannot be greater than `upper`.")


@dataclass(frozen=True)
class Quantity:
    """Represent a named system input or output.

    Parameters
    ----------
    name:
        Quantity identifier.
    bounds:
        Optional bounds for variable inputs.
    value:
        Optional fixed value for fixed inputs.
    description:
        Optional human-readable description.

    Returns
    -------
    None
        This dataclass stores the validated quantity definition.
    """

    name: str
    bounds: Bounds | None = None
    value: float | None = None
    description: str | None = None

    @classmethod
    def build(
        cls,
        name: str,
        spec=None,
        *,
        description: str | None = None,
    ) -> Quantity:
        """Build a quantity from a compact specification.

        Parameters
        ----------
        name:
            Quantity identifier.
        spec:
            Optional compact specification. ``None`` creates a plain quantity,
            a scalar creates a fixed-value quantity, and a pair creates bounds.
        description:
            Optional human-readable description.

        Returns
        -------
        Quantity
            Validated quantity definition.
        """
        if spec is None:
            return cls(name=name, description=description)
        if isinstance(spec, tuple):
            return cls(
                name=name,
                bounds=Bounds.from_tuple(spec),
                description=description,
            )
        return cls(name=name, value=spec, description=description)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("`name` must be a string.")
        if not self.name:
            raise ValueError("Quantity name cannot be empty.")
        if self.bounds is not None and not isinstance(self.bounds, Bounds):
            raise TypeError("`bounds` must be a Bounds instance or None.")
        if self.value is not None and (
            not np.isscalar(self.value) or not isinstance(self.value, Number)
        ):
            raise TypeError("`value` must be a numeric scalar or None.í")
        if self.value is not None and self.bounds is not None:
            raise ValueError("A quantity cannot have both `value` and `bounds`.")
