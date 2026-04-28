"""Small config-driven helpers for building pymoo problems."""

from .config import (
    load_config,
    load_problem,
    load_system_and_problem,
    parse_problem,
    parse_system,
    parse_system_and_problem,
)
from .core import (
    Bounds,
    Evaluator,
    Expression,
    ItemCollection,
    NamedArray,
    NamedItems,
    Problem,
    Quantity,
    Solutions,
    System,
    evaluate,
)
from .registry import register_system, get_system


def build_problem(*args, **kwargs):
    """Build a pymoo problem from a core problem."""
    from flexopt.pymoo_integration.factory import build_problem as _build

    return _build(*args, **kwargs)


def build_problem_from_file(*args, **kwargs):
    """Build a pymoo problem from a config file."""
    from flexopt.pymoo_integration.factory import (
        build_problem_from_file as _build_from_file,
    )

    return _build_from_file(*args, **kwargs)


def __getattr__(name: str):
    if name == "PymooProblem":
        from flexopt.pymoo_integration.problem import PymooProblem

        return PymooProblem
    raise AttributeError(f"module 'flexible_problem' has no attribute {name!r}")


__all__ = [
    "Bounds",
    "evaluate",
    "Evaluator",
    "Expression",
    "ItemCollection",
    "NamedArray",
    "NamedItems",
    "Problem",
    "PymooProblem",
    "Quantity",
    "Solutions",
    "System",
    "build_problem",
    "build_problem_from_file",
    "load_config",
    "load_problem",
    "load_system_and_problem",
    "parse_problem",
    "parse_system",
    "parse_system_and_problem",
    "register_system",
    "get_system",
]
