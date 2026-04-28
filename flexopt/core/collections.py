"""Collection helpers for core domain models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


class ItemCollection(Sequence):
    """Store typed items with optional collection-level validation.

    Parameters
    ----------
    items:
        Items to store in the collection.
    item_type:
        Expected type for every stored item.
    unique_by:
        Attribute name that must be unique across all items. If ``None``,
        uniqueness is not enforced.
    **attribute_filters:
        Attribute-value pairs that every item must satisfy.

    Returns
    -------
    None
    """

    def __init__(
        self,
        items,
        *,
        item_type,
        unique_by: str | None = None,
        **attribute_filters,
    ) -> None:
        if not isinstance(item_type, type):
            raise TypeError("`item_type` must be a type.")

        # Freeze input items so the collection stays immutable
        self.items = tuple(items)
        self.item_type = item_type
        self.unique_by = unique_by
        self.attribute_filters = dict(attribute_filters)
        self._ensure(
            item_type=self.item_type,
            unique_by=self.unique_by,
            **self.attribute_filters,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __eq__(self, other) -> bool:
        if not isinstance(other, ItemCollection):
            return NotImplemented
        return (
            self.items == other.items
            and self.item_type is other.item_type
            and self.unique_by == other.unique_by
            and self.attribute_filters == other.attribute_filters
        )

    def _class_has_attribute(self, attribute: str) -> bool:
        return (
            hasattr(self.item_type, attribute)
            or attribute in getattr(self.item_type, "__annotations__", {})
        )

    def _ensure_attribute_exists(self, attribute: str) -> None:
        if self.items:
            # Check the attribute on actual items when the collection is not empty
            if all(hasattr(item, attribute) for item in self.items):
                return
        elif self._class_has_attribute(attribute):
            # Fall back to the declared item type when there are no items yet
            return
        raise TypeError(
            f"`{self.item_type.__name__}` items must define attribute '{attribute}'."
        )

    def _ensure_item_type(self, item_type) -> None:
        if not isinstance(item_type, type):
            raise TypeError("`item_type` must be a type.")
        if not issubclass(self.item_type, item_type):
            raise TypeError(
                f"Expected collection of {item_type.__name__}, "
                f"got {self.item_type.__name__}."
            )
        for item in self.items:
            if not isinstance(item, item_type):
                raise TypeError(
                    f"`items` must contain {item_type.__name__} objects."
                )

    def _ensure_unique_by(self, attribute: str) -> None:
        values = self.values_by(attribute)
        if len(values) != len(set(values)):
            raise ValueError(
                f"Items must have unique values for attribute '{attribute}'."
            )

    def _ensure_attribute_filters(self, attribute_filters) -> None:
        for attribute, expected in attribute_filters.items():
            self._ensure_attribute_exists(attribute)
            for item in self.items:
                if getattr(item, attribute) != expected:
                    raise ValueError(
                        f"All items must have {attribute}={expected!r}; "
                        f"got {getattr(item, attribute)!r}."
                    )

    def _ensure(
        self,
        *,
        item_type=None,
        unique_by: str | None = None,
        **attribute_filters,
    ) -> None:
        if item_type is not None:
            # Accept collections whose declared type is the same or more specific
            self._ensure_item_type(item_type)
        if unique_by is not None:
            self._ensure_unique_by(unique_by)
        self._ensure_attribute_filters(attribute_filters)

    def ensure(
        self,
        *,
        item_type=None,
        unique_by: str | None = None,
        **attribute_filters,
    ) -> None:
        """Validate the collection against additional constraints.

        Parameters
        ----------
        item_type:
            Optional supertype that the collection item type must satisfy.
        unique_by:
            Optional attribute name that must be unique across all items.
        **attribute_filters:
            Attribute-value pairs that every item must satisfy.

        Returns
        -------
        None
            This method raises if any requested constraint is violated.
        """
        self._ensure(
            item_type=item_type,
            unique_by=unique_by,
            **attribute_filters,
        )

    def index(self, key, attribute: str | None = None) -> int:
        """Return the position of one item in the collection.

        Parameters
        ----------
        key:
            Item to search for, or attribute value when ``attribute`` is given.
        attribute:
            Optional attribute name used for matching. When omitted, the search
            is performed directly on the stored items.

        Returns
        -------
        int
            Integer position of the matched item.
        """
        if attribute is None:
            try:
                return self.items.index(key)
            except ValueError as exc:
                raise KeyError(f"Unknown item {key!r}.") from exc

        self._ensure_attribute_exists(attribute)
        values = self.values_by(attribute)
        try:
            return values.index(key)
        except ValueError as exc:
            raise KeyError(
                f"Unknown value {key!r} for attribute '{attribute}'."
            ) from exc

    def values_by(self, attribute: str) -> tuple[object, ...]:
        """Retrieve the values of a given attribute for all items.

        Parameters
        ----------
        attribute:
            Attribute name to inspect.

        Returns
        -------
        tuple[object, ...]
            Tuple of attribute values in collection order.
        """
        self._ensure_attribute_exists(attribute)
        # Return the requested attribute for every item
        return tuple(getattr(item, attribute) for item in self.items)

    def lookup_by(self, attribute: str) -> dict[object, object]:
        """Build a lookup table from an attribute value to the item.

        Parameters
        ----------
        attribute:
            Attribute name used as dictionary key.

        Returns
        -------
        dict[object, object]
            Mapping from attribute values to the corresponding items.
        """
        # Build dictionary keys from the requested attribute
        self._ensure_attribute_exists(attribute)
        values = tuple(getattr(item, attribute) for item in self.items)
        if len(values) != len(set(values)):
            raise ValueError(
                f"Cannot index by non-unique attribute '{attribute}'."
            )
        return {value: item for value, item in zip(values, self.items, strict=True)}


@dataclass(frozen=True)
class NamedArray:
    """Bind an ordered collection of items to a 2D numeric array."""

    columns: ItemCollection
    values: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.columns, ItemCollection):
            raise TypeError("`columns` must be an ItemCollection instance.")

        if not isinstance(self.values, np.ndarray):
            raise TypeError("`values` must be a numpy.ndarray.")

        # TODO: Consider allowing 1D arrays for single-row data and reshaping them internally
        if self.values.ndim == 1:
            array = self.values.reshape(1, -1)
        else:
            array = self.values
        if array.ndim != 2:
            raise ValueError("`values` must be a 1D or 2D array.")
        if array.shape[1] != len(self.columns):
            raise ValueError(
                "The number of array columns must match the number of column items."
            )
        object.__setattr__(self, "values", array)

    @property
    def n_points(self) -> int:
        return self.values.shape[0]

    @property
    def n_columns(self) -> int:
        return self.values.shape[1]

    def sel_rows(self, key) -> NamedArray:
        """Return a new named array with a selected subset of rows."""
        selected_values = self.values[key]
        if selected_values.ndim == 1:
            selected_values = selected_values.reshape(1, -1)
        return NamedArray(columns=self.columns, values=selected_values)

    def sel_cols(self, key, attribute: str | None = None) -> NamedArray:
        """Return a new named array with a selected subset of columns.

        Parameters
        ----------
        key:
            Direct column item, attribute value, or list/tuple of either.
        attribute:
            Optional attribute name used to match columns semantically.

        Returns
        -------
        NamedArray
            Named array restricted to the selected columns.
        """
        keys = key if isinstance(key, (list, tuple)) else (key,)
        column_indices = [
            self.columns.index(column_key, attribute=attribute) for column_key in keys
        ]

        selected_values = self.values[:, column_indices]
        selected_items = tuple(self.columns[index] for index in column_indices)
        return NamedArray(
            columns=ItemCollection(
                selected_items,
                item_type=self.columns.item_type,
                unique_by=self.columns.unique_by,
                **self.columns.attribute_filters,
            ),
            values=selected_values,
        )
