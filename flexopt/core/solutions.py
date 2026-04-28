"""Runtime containers for evaluated solution batches."""

from __future__ import annotations

from dataclasses import dataclass

from flexopt.core.collections import NamedArray


@dataclass(frozen=True)
class Solutions:
    """Store solution candidates and optional evaluation results.

    Parameters
    ----------
    variables:
        Candidate variable values.
    objectives:
        Optional evaluated objective values.
    ineq_constraints:
        Optional evaluated inequality constraint values.
    eq_constraints:
        Optional evaluated equality constraint values.
    inputs:
        Optional full system input values.
    outputs:
        Optional evaluated system output values.

    Returns
    -------
    None
        This dataclass stores a validated solution batch.
    """

    variables: NamedArray
    objectives: NamedArray | None = None
    ineq_constraints: NamedArray | None = None
    eq_constraints: NamedArray | None = None
    inputs: NamedArray | None = None
    outputs: NamedArray | None = None

    @property
    def is_evaluated(self) -> bool:
        """Whether this batch includes required evaluation results."""
        return self.objectives is not None and self.outputs is not None

    @property
    def n_points(self) -> int:
        """Number of solution points stored in the batch."""
        return self.variables.n_points

    def __post_init__(self) -> None:
        if not isinstance(self.variables, NamedArray):
            raise TypeError("`variables` must be a NamedArray instance.")

        # Keep every named array aligned on the same solution batch
        for attribute in (
            "objectives",
            "ineq_constraints",
            "eq_constraints",
            "inputs",
            "outputs",
        ):
            value = getattr(self, attribute)
            if value is None:
                continue
            if not isinstance(value, NamedArray):
                raise TypeError(f"`{attribute}` must be a NamedArray instance or None.")
            if value.n_points != self.n_points:
                raise ValueError(
                    f"`{attribute}` must have the same number of points as "
                    "`variables`."
                )

        # Avoid partially evaluated states such as objectives without outputs
        has_any_results = any(
            getattr(self, attribute) is not None
            for attribute in (
                "objectives",
                "ineq_constraints",
                "eq_constraints",
                "inputs",
                "outputs",
            )
        )
        has_required_results = self.objectives is not None and self.outputs is not None
        if has_any_results and not has_required_results:
            raise ValueError(
                "Evaluated solutions must define both `objectives` and `outputs`."
            )

    def with_results(
        self,
        *,
        objectives: NamedArray,
        outputs: NamedArray,
        ineq_constraints: NamedArray | None = None,
        eq_constraints: NamedArray | None = None,
        inputs: NamedArray | None = None,
    ) -> Solutions:
        """Return a copy enriched with evaluation results.

        Parameters
        ----------
        objectives:
            Evaluated objective values.
        outputs:
            Evaluated system output values.
        ineq_constraints:
            Optional evaluated inequality constraint values.
        eq_constraints:
            Optional evaluated equality constraint values.
        inputs:
            Optional full system input values.

        Returns
        -------
        Solutions
            New solution batch with the same variables and the provided results.
        """
        return Solutions(
            variables=self.variables,
            objectives=objectives,
            ineq_constraints=ineq_constraints,
            eq_constraints=eq_constraints,
            inputs=inputs,
            outputs=outputs,
        )
