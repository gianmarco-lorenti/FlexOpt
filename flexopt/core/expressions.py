"""Expression domain models."""

from __future__ import annotations

from dataclasses import dataclass
import numexpr as ne


EXPRESSION_KINDS = frozenset({"objective", "ineq_constraint", "eq_constraint"})


@dataclass(frozen=True)
class Expression:
    """Represent an objective or constraint formula.

    Parameters
    ----------
    name:
        Expression identifier.
    formula:
        Formula to evaluate.
    kind:
        Expression role, such as objective or constraint.

    Returns
    -------
    None
        This dataclass stores the validated expression definition.
    """

    name: str
    formula: str
    kind: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Expression name cannot be empty.")
        if not self.formula:
            raise ValueError(f"Expression '{self.name}' must define a formula.")
        if self.kind not in EXPRESSION_KINDS:
            raise ValueError(
                f"Unsupported expression kind '{self.kind}' for '{self.name}'."
            )
        validate_expression(self)


def referenced_symbols(expression: Expression) -> set[str]:
    """Extract variable-like names used by an expression formula.

    Parameters
    ----------
    expression:
        Expression whose formula should be inspected.

    Returns
    -------
    set[str]
        Symbol names referenced by the expression formula.
    """
    # Reuse numexpr's parser so supported functions are not treated as symbols
    return set(ne.NumExpr(expression.formula).input_names)


def validate_expression(
    expression: Expression, allowed_symbols: set[str] | None = None
) -> None:
    """Check that an expression compiles and only references known symbols.

    Parameters
    ----------
    expression:
        Expression to validate.
    allowed_symbols:
        Optional set of symbols that the expression is allowed to reference.

    Returns
    -------
    None
        This function raises if validation fails.
    """
    try:
        ne.validate(expression.formula)
    except Exception as exc:
        raise ValueError(
            f"Expression '{expression.name}' is not a valid numexpr formula: "
            f"{expression.formula}"
        ) from exc

    if allowed_symbols is None:
        return

    # Report symbols used in the formula but missing from the allowed context
    unknown = sorted(referenced_symbols(expression) - allowed_symbols)
    if unknown:
        raise ValueError(
            f"Expression '{expression.name}' references unknown symbols: "
            f"{', '.join(unknown)}"
        )


def evaluate_expression(expression: Expression, context: dict[str, object]):
    """Evaluate a validated expression with numexpr.

    Parameters
    ----------
    expression:
        Expression to evaluate.
    context:
        Mapping from symbol names to numeric values.

    Returns
    -------
    object
        Evaluated scalar or array value.
    """
    try:
        value = ne.evaluate(expression.formula, local_dict=context)
    except Exception as exc:
        raise ValueError(
            f"Could not evaluate expression '{expression.name}': {expression.formula}"
        ) from exc
    return value
