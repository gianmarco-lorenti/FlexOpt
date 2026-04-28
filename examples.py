"""Tiny runnable example showing the intended structure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

from flexopt import build_problem_from_file, register_system


def analytic_system(inputs: np.ndarray) -> np.ndarray:
    """Small analytic system used to demonstrate config-driven optimization."""
    x1 = inputs[:, 0]
    x2 = inputs[:, 1]

    y1 = np.sin(x1 + x2)
    y2 = np.cos(x1 * x2)

    return np.column_stack((y1, y2))


def run_example():
    register_system("analytic_system", analytic_system)

    config_path = Path(__file__).with_name("example_problem.json")
    pymoo_problem = build_problem_from_file(str(config_path))

    result = minimize(
        pymoo_problem,
        GA(pop_size=30),
        ("n_gen", 40),
        seed=7,
        verbose=False,
    )

    print("Best variable vector found:")
    print(result.X)
    print("Objective value:")
    print(result.F)


if __name__ == "__main__":
    run_example()
