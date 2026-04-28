"""Runtime evaluation for optimization problems."""

from __future__ import annotations

import numpy as np

from flexopt.core.collections import NamedArray
from flexopt.core.expressions import evaluate_expression
from flexopt.core.problem import Problem
from flexopt.core.solutions import Solutions


class Evaluator:
    """Evaluate solution candidates for one problem."""

    def __init__(self, problem: Problem) -> None:
        if not isinstance(problem, Problem):
            raise TypeError("`problem` must be a Problem instance.")
        self.problem = problem

    def _ensure_variables(self, solutions: Solutions) -> NamedArray:
        if not isinstance(solutions, Solutions):
            raise TypeError("`solutions` must be a Solutions instance.")

        if solutions.variables.columns != self.problem.variables:
            raise ValueError(
                "Solution variable columns must match problem variables."
            )
        return solutions.variables

    def _context_from_named_arrays(self, *arrays: NamedArray) -> dict[str, np.ndarray]:
        context: dict[str, np.ndarray] = {}
        for array in arrays:
            for item in array.columns:
                context[item.name] = array.sel_cols(item).values.ravel()
        return context

    def _evaluate_expressions(
        self,
        expressions,
        context,
        n_points: int,
    ) -> NamedArray | None:
        if not expressions:
            return None
        columns = []
        for expression in expressions:
            values = np.asarray(evaluate_expression(expression, context), dtype=float)
            if values.ndim == 0:
                values = np.full(n_points, float(values))
            if values.ndim != 1:
                raise ValueError(
                    f"Expression '{expression.name}' must evaluate to one dimension."
                )
            if values.size != n_points:
                raise ValueError(
                    f"Expression '{expression.name}' must have one value per "
                    "solution point."
                )
            columns.append(values)
        return NamedArray(
            columns=expressions,
            values=np.column_stack(columns),
        )

    def build_full_inputs(self, variables: NamedArray) -> NamedArray:
        """Build complete batched system inputs from variable values.

        Parameters
        ----------
        variables:
            Candidate variable values ordered like ``self.problem.variables``.

        Returns
        -------
        NamedArray
            Full system input values ordered like ``self.problem.inputs``.
        """
        # Preserve the canonical system input order when assembling the full vector
        fixed_inputs_by_name = self.problem.fixed_inputs.lookup_by("name")
        variable_names = set(self.problem.variables.values_by("name"))
        input_columns = []
        for input_name in self.problem.inputs.values_by("name"):
            if input_name in variable_names:
                input_columns.append(
                    variables.sel_cols(input_name, attribute="name").values.ravel()
                )
            else:
                fixed_input = fixed_inputs_by_name[input_name]
                input_columns.append(
                    np.full(variables.n_points, fixed_input.value, dtype=float)
                )
        return NamedArray(
            columns=self.problem.inputs,
            values=np.column_stack(input_columns),
        )

    def evaluate(self, solutions: Solutions) -> Solutions:
        """Return a new solution batch enriched with evaluation results.

        Parameters
        ----------
        solutions:
            Candidate solutions to evaluate.

        Returns
        -------
        Solutions
            Evaluated solutions with inputs, outputs, objectives, and optional
            constraint values.
        """
        variables = self._ensure_variables(solutions)
        inputs = self.build_full_inputs(variables)
        n_points = variables.n_points
        if self.problem.has_system:
            outputs = self.problem.system.evaluate(inputs)
        else:
            outputs = NamedArray(
                columns=self.problem.outputs,
                values=np.empty((n_points, 0), dtype=float),
            )

        context = self._context_from_named_arrays(inputs, outputs)
        objectives = self._evaluate_expressions(
            self.problem.objectives,
            context,
            n_points,
        )
        ineq_constraints = self._evaluate_expressions(
            self.problem.ineq_constraints,
            context,
            n_points,
        )
        eq_constraints = self._evaluate_expressions(
            self.problem.eq_constraints,
            context,
            n_points,
        )

        return solutions.with_results(
            inputs=inputs,
            outputs=outputs,
            objectives=objectives,
            ineq_constraints=ineq_constraints,
            eq_constraints=eq_constraints,
        )


def evaluate(problem: Problem, solutions: Solutions) -> Solutions:
    """Evaluate solution candidates for a problem.

    Parameters
    ----------
    problem:
        Problem to evaluate.
    solutions:
        Candidate solutions to evaluate.

    Returns
    -------
    Solutions
        Evaluated solution batch.
    """
    return Evaluator(problem).evaluate(solutions)
