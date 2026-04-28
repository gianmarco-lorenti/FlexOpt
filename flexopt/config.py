"""Load JSON/YAML configs and parse them into core objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from flexopt.core import Problem, System
from flexopt.registry import get_system


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML configuration file."""
    config_path = Path(path)
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        return json.loads(config_path.read_text())

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML support requires PyYAML. Install it or use JSON."
            ) from exc
        return yaml.safe_load(config_path.read_text())

    raise ValueError(f"Unsupported config format '{suffix}'. Use .json or .yaml/.yml")


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise TypeError(f"`{name}` must be a mapping.")
    return section


def _name_from_item(item, label: str) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping) and isinstance(item.get("name"), str):
        return item["name"]
    raise TypeError(f"{label} entries must be names or mappings with a name.")


def _names(raw, label: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        return tuple(str(name) for name in raw)
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return tuple(_name_from_item(item, label) for item in raw)
    raise TypeError(f"`{label}` must be a sequence or mapping.")


def _quantity_specs(raw, *, spec_key: str, label: str) -> dict[str, object]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {
            name: _normalize_quantity_spec(spec, spec_key=spec_key, name=str(name))
            for name, spec in raw.items()
        }
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        specs: dict[str, object] = {}
        for item in raw:
            if not isinstance(item, Mapping):
                raise TypeError(f"{label} entries must be mappings.")
            name = item.get("name")
            if not isinstance(name, str):
                raise TypeError(f"{label} entries must define a string name.")
            if spec_key not in item:
                raise ValueError(f"{label} '{name}' must define `{spec_key}`.")
            specs[name] = _normalize_quantity_spec(
                item[spec_key],
                spec_key=spec_key,
                name=name,
            )
        return specs
    raise TypeError(f"`{label}` must be a mapping or sequence.")


def _normalize_quantity_spec(spec, *, spec_key: str, name: str):
    if spec_key == "bounds" and isinstance(spec, list):
        return tuple(spec)
    return spec


def _expression_specs(raw, *, label: str) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        specs: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, Mapping):
                raise TypeError(f"{label} entries must be mappings.")
            name = item.get("name")
            formula = item.get("formula")
            if not isinstance(name, str):
                raise TypeError(f"{label} entries must define a string name.")
            if not isinstance(formula, str):
                raise TypeError(f"{label} '{name}' must define a string formula.")
            specs[name] = formula
        return specs
    raise TypeError(f"`{label}` must be a mapping or sequence.")


def _split_input_specs(raw_inputs) -> tuple[dict[str, object], dict[str, object]]:
    variables: dict[str, object] = {}
    fixed_inputs: dict[str, object] = {}
    if raw_inputs is None:
        return variables, fixed_inputs
    if not isinstance(raw_inputs, Sequence) or isinstance(raw_inputs, str):
        raise TypeError("`inputs` must be a sequence when used as problem inputs.")

    for item in raw_inputs:
        if not isinstance(item, Mapping):
            raise TypeError("Problem input entries must be mappings.")
        name = item.get("name")
        if not isinstance(name, str):
            raise TypeError("Problem input entries must define a string name.")
        has_bounds = "bounds" in item
        has_value = "value" in item
        if has_bounds == has_value:
            raise ValueError(
                f"Input '{name}' must define exactly one of `bounds` or `value`."
            )
        if has_bounds:
            variables[name] = _normalize_quantity_spec(
                item["bounds"],
                spec_key="bounds",
                name=name,
            )
        else:
            fixed_inputs[name] = item["value"]
    return variables, fixed_inputs


def _system_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "system" in config:
        return _section(config, "system")
    if "function" in config or "outputs" in config:
        return config
    return {}


def _problem_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _section(config, "problem") or config


def parse_system(config: Mapping[str, Any]) -> System | None:
    """Parse the optional system section of a config."""
    raw_system = _system_config(config)
    if not raw_system:
        return None

    system_name = raw_system.get("name") or config.get("name")
    if not system_name:
        raise ValueError("System configs must define `name`.")

    function_name = raw_system.get("function")
    if not function_name:
        raise ValueError("System configs must define `function`.")

    inputs = _names(raw_system.get("inputs"), "system inputs")
    outputs = _names(raw_system.get("outputs"), "system outputs")
    if not inputs:
        raise ValueError("System configs must define at least one input.")
    if not outputs:
        raise ValueError("System configs must define at least one output.")

    return System.build(
        name=system_name,
        inputs=inputs,
        outputs=outputs,
        system_fn=get_system(function_name),
    )


def parse_problem(config: Mapping[str, Any], system: System | None = None) -> Problem:
    """Parse the problem section of a config."""
    raw_problem = _problem_config(config)

    variables = _quantity_specs(
        raw_problem.get("variables"),
        spec_key="bounds",
        label="variables",
    )
    fixed_inputs = _quantity_specs(
        raw_problem.get("fixed_inputs"),
        spec_key="value",
        label="fixed_inputs",
    )

    if not variables and "inputs" in raw_problem:
        variables, fixed_inputs = _split_input_specs(raw_problem.get("inputs"))

    return Problem.build(
        system=system,
        variables=variables,
        fixed_inputs=fixed_inputs,
        objectives=_expression_specs(
            raw_problem.get("objectives"),
            label="objectives",
        ),
        ineq_constraints=_expression_specs(
            raw_problem.get("ineq_constraints", raw_problem.get("constraints")),
            label="ineq_constraints",
        ),
        eq_constraints=_expression_specs(
            raw_problem.get("eq_constraints", raw_problem.get("equalities")),
            label="eq_constraints",
        ),
    )


def parse_system_and_problem(config: Mapping[str, Any]) -> tuple[System | None, Problem]:
    """Parse a config mapping into an optional system and a problem."""
    system = parse_system(config)
    problem = parse_problem(config, system=system)
    return system, problem


def load_system_and_problem(path: str | Path) -> tuple[System | None, Problem]:
    """Load and parse a config file into an optional system and a problem."""
    return parse_system_and_problem(load_config(path))


def load_problem(path: str | Path) -> Problem:
    """Load and parse a config file into a problem."""
    return load_system_and_problem(path)[1]
