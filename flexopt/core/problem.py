"""Problem domain model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from flexopt.core.collections import ItemCollection
from flexopt.core.expressions import Expression, validate_expression
from flexopt.core.quantities import Quantity
from flexopt.core.system import System


@dataclass(frozen=True)
class Problem:
    """Describe an optimization problem built on top of a system.

    Parameters
    ----------
    system:
        Optional system being optimized.
    variables:
        Inputs optimized within their bounds.
    objectives:
        Objective expressions.
    ineq_constraints:
        Inequality constraint expressions.
    eq_constraints:
        Equality constraint expressions.
    fixed_inputs:
        Inputs held constant during optimization.

    Returns
    -------
    None
        This dataclass stores the validated optimization problem definition.
    """

    system: System | None
    variables: ItemCollection
    objectives: ItemCollection
    ineq_constraints: ItemCollection
    eq_constraints: ItemCollection
    fixed_inputs: ItemCollection

    @classmethod
    def build(
        cls,
        *,
        system: System | None = None,
        variables: Mapping[str, object],
        objectives: Mapping[str, str],
        fixed_inputs: Mapping[str, object] | None = None,
        ineq_constraints: Mapping[str, str] | None = None,
        eq_constraints: Mapping[str, str] | None = None,
    ) -> Problem:
        """Build a problem from compact quantity and expression specs.

        Parameters
        ----------
        system:
            System being optimized.
        variables:
            Mapping from variable names to bound specs.
        objectives:
            Mapping from objective names to formulas.
        fixed_inputs:
            Optional mapping from fixed input names to values.
        ineq_constraints:
            Optional mapping from inequality constraint names to formulas.
        eq_constraints:
            Optional mapping from equality constraint names to formulas.

        Returns
        -------
        Problem
            Validated optimization problem definition.
        """
        fixed_inputs = {} if fixed_inputs is None else fixed_inputs
        ineq_constraints = {} if ineq_constraints is None else ineq_constraints
        eq_constraints = {} if eq_constraints is None else eq_constraints

        return cls(
            system=system,
            variables=cls._quantity_collection(variables),
            objectives=cls._expression_collection(objectives, "objective"),
            fixed_inputs=cls._quantity_collection(fixed_inputs),
            ineq_constraints=cls._expression_collection(
                ineq_constraints,
                "ineq_constraint",
            ),
            eq_constraints=cls._expression_collection(
                eq_constraints,
                "eq_constraint",
            ),
        )

    @staticmethod
    def _quantity_collection(specs: Mapping[str, object]) -> ItemCollection:
        if not isinstance(specs, Mapping):
            raise TypeError("Quantity specs must be a mapping from names to specs.")
        return ItemCollection(
            tuple(Quantity.build(name, spec) for name, spec in specs.items()),
            item_type=Quantity,
            unique_by="name",
        )

    @staticmethod
    def _expression_collection(
        specs: Mapping[str, str],
        kind: str,
    ) -> ItemCollection:
        if not isinstance(specs, Mapping):
            raise TypeError("Expression specs must be a mapping from names to formulas.")
        return ItemCollection(
            tuple(
                Expression(name=name, formula=formula, kind=kind)
                for name, formula in specs.items()
            ),
            item_type=Expression,
            unique_by="name",
            kind=kind,
        )

    @property
    def inputs(self) -> ItemCollection:
        """Declared input quantities for this problem."""
        if self.system is not None:
            return self.system.inputs
        return ItemCollection(
            tuple(self.variables) + tuple(self.fixed_inputs),
            item_type=Quantity,
            unique_by="name",
        )

    @property
    def outputs(self) -> ItemCollection:
        """Declared system outputs, if a system is defined."""
        if self.system is None:
            return ItemCollection((), item_type=Quantity, unique_by="name")
        return self.system.outputs

    @property
    def has_system(self) -> bool:
        """Whether the problem evaluates an external system."""
        return self.system is not None

    @property
    def n_variables(self) -> int:
        """Number of optimized variables."""
        return len(self.variables)

    @property
    def n_objectives(self) -> int:
        """Number of objective expressions."""
        return len(self.objectives)

    @property
    def n_ineq_constraints(self) -> int:
        """Number of inequality constraint expressions."""
        return len(self.ineq_constraints)

    @property
    def n_eq_constraints(self) -> int:
        """Number of equality constraint expressions."""
        return len(self.eq_constraints)

    @property
    def has_ineq_constraints(self) -> bool:
        """Whether the problem defines inequality constraints."""
        return bool(self.ineq_constraints)

    @property
    def has_eq_constraints(self) -> bool:
        """Whether the problem defines equality constraints."""
        return bool(self.eq_constraints)

    @property
    def has_constraints(self) -> bool:
        """Whether the problem defines any constraints."""
        return self.has_ineq_constraints or self.has_eq_constraints

    def __post_init__(self) -> None:
        if self.system is not None and not isinstance(self.system, System):
            raise TypeError("`system` must be a System instance or None.")
        if not isinstance(self.variables, ItemCollection):
            raise TypeError("`variables` must be an ItemCollection instance.")
        if not isinstance(self.objectives, ItemCollection):
            raise TypeError("`objectives` must be an ItemCollection instance.")
        if not isinstance(self.ineq_constraints, ItemCollection):
            raise TypeError("`ineq_constraints` must be an ItemCollection instance.")
        if not isinstance(self.eq_constraints, ItemCollection):
            raise TypeError("`eq_constraints` must be an ItemCollection instance.")
        if not isinstance(self.fixed_inputs, ItemCollection):
            raise TypeError("`fixed_inputs` must be an ItemCollection instance.")
        if not self.variables:
            raise ValueError("Problem must define at least one variable.")
        if not self.objectives:
            raise ValueError("Problem must define at least one objective.")

        # Lock each collection to the expected domain type and expression role
        self.fixed_inputs.ensure(item_type=Quantity, unique_by="name")
        self.variables.ensure(item_type=Quantity, unique_by="name")
        self.objectives.ensure(
            item_type=Expression,
            unique_by="name",
            kind="objective",
        )
        self.ineq_constraints.ensure(
            item_type=Expression,
            unique_by="name",
            kind="ineq_constraint",
        )
        self.eq_constraints.ensure(
            item_type=Expression,
            unique_by="name",
            kind="eq_constraint",
        )

        # Build a canonical lookup for declared inputs
        input_names = set(self.inputs.values_by("name"))
        fixed_input_names = set(self.fixed_inputs.values_by("name"))
        variable_names = set(self.variables.values_by("name"))

        # Reject variables assigned to both fixed and optimized sets
        if fixed_input_names & variable_names:
            overlap = ", ".join(sorted(fixed_input_names & variable_names))
            raise ValueError(
                f"Fixed inputs and variables must be disjoint. Overlap: {overlap}."
            )

        for quantity in self.fixed_inputs:
            if self.has_system and quantity.name not in input_names:
                raise ValueError(
                    f"Fixed input '{quantity.name}' is not declared by the system."
                )
            if quantity.value is None:
                raise ValueError(
                    f"Fixed input '{quantity.name}' must define a value."
                )
            if quantity.bounds is not None:
                raise ValueError(
                    f"Fixed input '{quantity.name}' cannot define bounds."
                )

        for quantity in self.variables:
            if self.has_system and quantity.name not in input_names:
                raise ValueError(
                    f"Variable '{quantity.name}' is not declared by the system."
                )
            if quantity.bounds is None:
                raise ValueError(
                    f"Variable '{quantity.name}' must define bounds."
                )
            if quantity.value is not None:
                raise ValueError(
                    f"Variable '{quantity.name}' cannot define a fixed value."
                )

        # Make fixed and optimized variables cover every declared system input
        if self.has_system:
            missing = input_names - (fixed_input_names | variable_names)
        else:
            missing = set()
        if missing:
            raise ValueError(
                "Fixed inputs and variables must cover every system input. "
                f"Missing: {', '.join(sorted(missing))}."
            )

        # Expressions may only reference declared inputs and system outputs
        allowed_symbols = input_names | set(self.outputs.values_by("name"))
        for expression in (
            tuple(self.objectives)
            + tuple(self.ineq_constraints)
            + tuple(self.eq_constraints)
        ):
            validate_expression(expression, allowed_symbols=allowed_symbols)
