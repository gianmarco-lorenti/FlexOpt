"""Adapter from local dataclasses to a pymoo elementwise problem."""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from flexopt.core import Evaluator, NamedArray, Problem, Solutions


def _named_array_row(named_array: NamedArray) -> dict[str, float]:
    return {
        item.name: float(value)
        for item, value in zip(
            named_array.columns,
            named_array.values[0],
            strict=True,
        )
    }


class PymooProblem(ElementwiseProblem):
    """Elementwise pymoo problem built from a Problem description."""

    def __init__(self, problem: Problem):
        self.problem = problem
        self.evaluator = Evaluator(problem)

        self.variables = list(problem.variables)

        xl = np.array([quantity.bounds.lower for quantity in self.variables], dtype=float)
        xu = np.array([quantity.bounds.upper for quantity in self.variables], dtype=float)

        super().__init__(
            n_var=problem.n_variables,
            n_obj=problem.n_objectives,
            n_ieq_constr=problem.n_ineq_constraints,
            n_eq_constr=problem.n_eq_constraints,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        variables = NamedArray(
            columns=self.problem.variables,
            values=np.asarray(x, dtype=float),
        )
        evaluated = self.evaluator.evaluate(Solutions(variables=variables))

        out["F"] = evaluated.objectives.values[0]
        if self.problem.has_ineq_constraints:
            out["G"] = evaluated.ineq_constraints.values[0]
        if self.problem.has_eq_constraints:
            out["H"] = evaluated.eq_constraints.values[0]

        out["details"] = {
            "inputs": _named_array_row(evaluated.inputs),
            "outputs": _named_array_row(evaluated.outputs),
        }
