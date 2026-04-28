"""System domain model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from flexopt.core.collections import ItemCollection, NamedArray
from flexopt.core.quantities import Quantity


@dataclass(frozen=True)
class System:
    """Describe a black-box system and its interface.

    Parameters
    ----------
    name:
        System identifier.
    inputs:
        Declared system inputs.
    outputs:
        Declared system outputs.
    system_fn:
        Callable implementing the system evaluation. It receives a 2D input
        array ordered like ``inputs`` and returns a 2D output array ordered like
        ``outputs``.

    Returns
    -------
    None
        This dataclass stores the validated system definition.
    """

    name: str
    inputs: ItemCollection
    outputs: ItemCollection
    system_fn: Callable

    @classmethod
    def build(cls, *, name: str, inputs, outputs, system_fn: Callable) -> System:
        """Build a system from ordered input and output names.

        Parameters
        ----------
        name:
            System identifier.
        inputs:
            Input names in system order.
        outputs:
            Output names in system order.
        system_fn:
            Callable implementing the system evaluation.

        Returns
        -------
        System
            Validated system definition.
        """
        return cls(
            name=name,
            inputs=ItemCollection(
                tuple(Quantity.build(input_name) for input_name in inputs),
                item_type=Quantity,
                unique_by="name",
            ),
            outputs=ItemCollection(
                tuple(Quantity.build(output_name) for output_name in outputs),
                item_type=Quantity,
                unique_by="name",
            ),
            # TODO: Checks on function ?
            system_fn=system_fn,
        )

    @property
    def n_inputs(self) -> int:
        """Number of declared system inputs."""
        return len(self.inputs)

    @property
    def n_outputs(self) -> int:
        """Number of declared system outputs."""
        return len(self.outputs)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("System name cannot be empty.")
        if not isinstance(self.inputs, ItemCollection):
            raise TypeError("`inputs` must be an ItemCollection instance.")
        if not isinstance(self.outputs, ItemCollection):
            raise TypeError("`outputs` must be an ItemCollection instance.")
        if not callable(self.system_fn):
            raise TypeError("`system_fn` must be callable.")
        if not self.inputs:
            raise ValueError("System must define at least one input.")
        if not self.outputs:
            raise ValueError("System must define at least one output.")

        # Keep collections aligned with the expected domain type
        self.inputs.ensure(item_type=Quantity, unique_by="name")
        self.outputs.ensure(item_type=Quantity, unique_by="name")

        # TODO: Assess whether to allow bounds and values on system quantities
        for quantity in self.inputs:
            if quantity.bounds is not None or quantity.value is not None:
                raise ValueError(
                    f"System input '{quantity.name}' cannot define bounds or value."
                )
        for quantity in self.outputs:
            if quantity.bounds is not None or quantity.value is not None:
                raise ValueError(
                    f"System output '{quantity.name}' cannot define bounds or value."
                )

        # Keep input and output names globally unique across the system
        input_names = self.inputs.values_by("name")
        output_names = self.outputs.values_by("name")
        names = input_names + output_names
        if len(names) != len(set(names)):
            raise ValueError("System inputs and outputs must have unique names.")

    def evaluate(self, inputs: NamedArray) -> NamedArray:
        """Evaluate system outputs for a batch of inputs.

        Parameters
        ----------
        inputs:
            Full system inputs ordered like ``self.inputs``.

        Returns
        -------
        NamedArray
            System outputs ordered like ``self.outputs``.
        """
        if not isinstance(inputs, NamedArray):
            raise TypeError("`inputs` must be a NamedArray instance.")
        if inputs.columns != self.inputs:
            raise ValueError("Input columns must match system inputs.")

        outputs = NamedArray(
            columns=self.outputs,
            values=self.system_fn(inputs.values),
        )
        if outputs.n_points != inputs.n_points:
            raise ValueError("Outputs must have one row per input point.")
        return outputs
