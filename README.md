# flexible_problem

This folder contains a small pattern for building `pymoo` problems from a
lightweight config file.

## Idea

1. Define a black-box system through ordered `inputs` and `outputs`.
2. Mark each input as either:
   - variable, with `bounds`
   - fixed input, with `value`
3. Register a Python function that computes outputs from the full input dict.
4. Define objective and constraint expressions over inputs and outputs.

The config stays user-facing and compact. Internally, the builder still creates
separate `System` and `Problem` dataclasses.

## Main dataclasses

- `Bounds`
- `Quantity`
- `Expression`
- `System`
- `Problem`

`System` describes the black box. `Problem` describes the optimization built on
top of it.

## Minimal config shape

```json
{
  "name": "analytic_example",
  "function": "analytic_system",
  "inputs": [
    {"name": "x1", "bounds": [0.0, 3.14]},
    {"name": "x2", "value": 1.5}
  ],
  "outputs": ["y1", "y2"],
  "objectives": [{"name": "obj1", "formula": "y1"}],
  "ineq_constraints": [{"name": "con1", "formula": "y2 - 0.5"}]
}
```

Rules for each input:

- `value` means fixed input
- `bounds` means variable input
- both together are an error
- neither is an error

Constraint expressions follow the `pymoo` convention `g(x) <= 0`.
Equality constraints can optionally be provided under `eq_constraints`.

## Example usage

Install dependencies first:

```bash
pip install pymoo numexpr PyYAML numpy
```

Then run:

```bash
python -m flexible_problem.examples
```
