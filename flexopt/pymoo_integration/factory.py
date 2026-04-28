"""Convenience functions to build pymoo problems from local configs."""

from __future__ import annotations

from flexopt.config import load_problem
from flexopt.core import Problem, System
from flexopt.pymoo_integration.problem import PymooProblem


def build_problem(problem: Problem, system: System | None = None) -> PymooProblem:
    """Transform a local Problem into the corresponding pymoo object."""
    if system is not None and problem.system != system:
        raise ValueError("`problem.system` must match the passed `system`.")
    return PymooProblem(problem=problem)


def build_problem_from_file(path: str) -> PymooProblem:
    return build_problem(load_problem(path))
