"""Links between the parameters of a ``FitModel``, and inequality constraints (Physics and Methods, section 9.15).

* A **link** (``Parameter.tie``) makes a parameter a function of others: ``sigma_R_1`` (share σR/R of population 1),
  ``2*R_1``, ``R_1 + t_shell_1``. A linked parameter is not fitted; its value follows the others at every step of a
  fit, and its error comes from theirs through the derivatives of the expression (``propagated``).
* An **inequality** (``Inequality``) keeps a fitted parameter on one side of an expression: ``D_2 >= 2*R_2 + 3``.
  The fit searches the parameter between the bound and the far end of its range (a change of variables, as the
  contact D ≥ 2R that every paracrystal has; no penalty).

Names: a component's parameter is ``<key>_<n>`` (``R_1``, ``sigma_nn_2``, n from 1), a global one its key
(``background``, ``res_width``). Expressions are numbers, names, + − * / **, parentheses and the functions of
``FUNCTIONS`` (``pi`` and ``e`` as constants); they are parsed with ``ast`` and evaluated by a fixed set of
operations, never by ``eval``.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping, Optional

MAX_LENGTH = 200
MAX_NODES = 80
FUNCTIONS = {"sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10, "sin": math.sin,
             "cos": math.cos, "tan": math.tan, "abs": abs, "min": min, "max": max}
CONSTANTS = {"pi": math.pi, "e": math.e}
RELATIONS = (">=", "<=")
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
LINEAR_KEYS = frozenset({"Int", "background", "res_amplitude", "diffuse_amplitude"})
"""The scales (``fit_model.LINEAR``; repeated here: this module does not import the model)."""


def _power(base: float, exponent: float) -> float:
    if abs(exponent) > 1e3:
        raise ValueError("An exponent in a link must stay below 1000")
    return math.pow(base, exponent)


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: _power}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class Expression:
    """A parsed expression of parameter names; ``names`` are those it uses, ``expression(values)`` its value."""

    def __init__(self, text: str):
        text = str(text or "").strip()
        if not text:
            raise ValueError("An empty expression")
        if len(text) > MAX_LENGTH:
            raise ValueError(f"An expression of at most {MAX_LENGTH} characters, please")
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Not an expression: {text!r}") from exc
        if sum(1 for _node in ast.walk(tree)) > MAX_NODES:
            raise ValueError("This expression is too long")
        self.text = text
        self.names: set[str] = set()
        self._call = self._compile(tree.body)
        self.names = frozenset(self.names)

    def _compile(self, node) -> Callable[[Mapping[str, float]], float]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            value = float(node.value)
            return lambda _values: value
        if isinstance(node, ast.Name):
            if node.id in CONSTANTS:
                value = CONSTANTS[node.id]
                return lambda _values: value
            if node.id in FUNCTIONS:
                raise ValueError(f"{node.id} is a function: write {node.id}(…)")
            name = node.id
            self.names.add(name)
            return lambda values: float(values[name])
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right, apply = self._compile(node.left), self._compile(node.right), _BINARY[type(node.op)]
            return lambda values: apply(left(values), right(values))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            inner, apply = self._compile(node.operand), _UNARY[type(node.op)]
            return lambda values: apply(inner(values))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS
                and not node.keywords and node.args):
            function, arguments = FUNCTIONS[node.func.id], [self._compile(argument) for argument in node.args]
            if node.func.id not in ("min", "max") and len(arguments) != 1:
                raise ValueError(f"{node.func.id} takes one value")
            return lambda values: float(function(*(argument(values) for argument in arguments)))
        raise ValueError(f"Not allowed in a link: {ast.unparse(node) if hasattr(ast, 'unparse') else type(node).__name__}")

    def __call__(self, values: Mapping[str, float]) -> float:
        try:
            value = self._call(values)
        except KeyError as exc:
            raise ValueError(f"Unknown parameter {exc.args[0]} in {self.text!r}") from None
        except (ArithmeticError, ValueError, TypeError) as exc:
            raise ValueError(f"{self.text!r} cannot be evaluated here: {exc}") from None
        if not math.isfinite(value):
            raise ValueError(f"{self.text!r} is not a finite number here")
        return value


@dataclass(frozen=True)
class Inequality:
    """``name`` ``relation`` ``expression``: e.g. ``D_2 >= 2*R_2``."""
    name: str
    relation: str
    expression: str

    def __post_init__(self):
        if self.relation not in RELATIONS:
            raise ValueError(f"A constraint is >= or <=, not {self.relation!r}")

    @property
    def text(self) -> str:
        return f"{self.name} {self.relation} {self.expression}"

    def to_dict(self) -> dict:
        return {"parameter": self.name, "relation": self.relation, "expression": self.expression}

    @classmethod
    def parse(cls, text: str) -> "Inequality":
        """``"D_2 >= 2*R_2"`` (also ``≥``, ``≤``)."""
        text = str(text).replace("≥", ">=").replace("≤", "<=")
        for relation in RELATIONS:
            if relation in text:
                name, _sep, expression = text.partition(relation)
                name = name.strip()
                if not NAME.fullmatch(name):
                    raise ValueError(f"Write a constraint as NAME >= expression, e.g. D_1 >= 2*R_1 (not {text!r})")
                Expression(expression)
                return cls(name, relation, expression.strip())
        raise ValueError(f"A constraint needs >= or <=: {text!r}")


# -- names ------------------------------------------------------------------------------------------------------


def parameter_name(path: tuple) -> str:
    owner, key = path
    return key if owner == "globals" else f"{key}_{int(owner) + 1}"


def names(model) -> dict[str, tuple]:
    """Every parameter of ``model`` in use, by its name in links."""
    return {parameter_name(path): path for path, _parameter in model.parameters()}


def values(model) -> dict[str, float]:
    return {parameter_name(path): float(parameter.value) for path, parameter in model.parameters()}


def rename(text: str, mapping: Mapping[str, Optional[str]]) -> Optional[str]:
    """``text`` with names replaced by ``mapping`` (``None``: the name is gone, and so is the expression)."""
    gone = False

    def swap(found):
        nonlocal gone
        name = found.group(0)
        if name not in mapping:
            return name
        if mapping[name] is None:
            gone = True
            return name
        return mapping[name]

    out = NAME.sub(swap, text)
    return None if gone else out


def removal_mapping(model, removed: int) -> dict[str, Optional[str]]:
    """The names of a model's components after component ``removed`` (0-based) is taken out."""
    mapping: dict[str, Optional[str]] = {}
    for index, component in enumerate(model.components):
        for key in component.params:
            old = f"{key}_{index + 1}"
            mapping[old] = None if index == removed else (f"{key}_{index}" if index > removed else old)
    return mapping


# -- the links of a model ---------------------------------------------------------------------------------------


class Links:
    """The parsed links and inequalities of a model; false when it has none (a fit then runs exactly as without)."""

    def __init__(self, model):
        self.paths = names(model)
        self.ties: list[tuple[str, tuple, Expression]] = []
        for path, parameter in model.parameters():
            if parameter.tie:
                expression = Expression(parameter.tie)
                unknown = sorted(expression.names - set(self.paths))
                if unknown:
                    raise ValueError(f"{parameter_name(path)} is linked to {', '.join(unknown)}, which the model does "
                                     "not have")
                self.ties.append((parameter_name(path), path, expression))
        self.ties = self._ordered(self.ties)
        self.inequalities: list[tuple[Inequality, tuple, Expression]] = []
        for inequality in getattr(model, "inequalities", ()) or ():
            if inequality.name not in self.paths:
                raise ValueError(f"The constraint {inequality.text} names {inequality.name}, which the model does not "
                                 "have")
            expression = Expression(inequality.expression)
            unknown = sorted(expression.names - set(self.paths))
            if unknown:
                raise ValueError(f"The constraint {inequality.text} uses {', '.join(unknown)}, which the model does "
                                 "not have")
            self.inequalities.append((inequality, self.paths[inequality.name], expression))
        self.tied = {path for _name, path, _expression in self.ties}

    def __bool__(self) -> bool:
        return bool(self.ties or self.inequalities)

    @staticmethod
    def _ordered(ties):
        """Ties in an order where every tie comes after the ties it reads; a circle of links is refused."""
        by_name = {name: (name, path, expression) for name, path, expression in ties}
        done, busy, out = set(), set(), []

        def visit(name, chain):
            if name in done or name not in by_name:
                return
            if name in busy:
                raise ValueError("These links refer to each other in a circle: " + " → ".join(chain + [name]))
            busy.add(name)
            for other in sorted(by_name[name][2].names):
                visit(other, chain + [name])
            busy.discard(name)
            done.add(name)
            out.append(by_name[name])

        for name in by_name:
            visit(name, [])
        return out

    def depends_on(self, name: str) -> set[str]:
        """Every name the value of the tie ``name`` reads, through other ties too."""
        expressions = {tie_name: expression for tie_name, _path, expression in self.ties}
        seen, stack = set(), [name]
        while stack:
            current = stack.pop()
            for other in expressions.get(current, Expression("0")).names:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        return seen

    def apply(self, model, given: Optional[Mapping[str, float]] = None):
        """``model`` with every linked value computed from the others (``model`` itself when there is no tie).
        ``ValueError`` when an expression cannot be evaluated."""
        if not self.ties:
            return model
        current = dict(given) if given is not None else values(model)
        changes = {}
        for name, path, expression in self.ties:
            value = expression(current)
            current[name] = value
            changes[path] = value
        return model.with_values(changes)

    def bound(self, inequality_index: int, current: Mapping[str, float]) -> float:
        return self.inequalities[inequality_index][2](current)


def links_of(model) -> Links:
    return Links(model)


def apply_links(model):
    """``model`` with its linked values computed from the others (unchanged without links)."""
    links = Links(model)
    return links.apply(model) if links.ties else model


def linked(model, path: tuple, expression: Optional[str]):
    """``model`` with ``path`` linked to ``expression`` (fixed while linked) or unlinked (``None``: it keeps its value
    and stays fixed). ``ValueError`` when the expression is not allowed, names a parameter the model does not have,
    or closes a circle of links."""
    if expression is None:
        return model.with_parameter(path, tie=None)
    text = Expression(expression).text
    candidate = model.with_parameter(path, tie=text, free=False)
    links = Links(candidate)
    if parameter_name(path) in links.depends_on(parameter_name(path)):
        raise ValueError(f"{parameter_name(path)} cannot depend on itself")
    _scale_rules(candidate, links)
    return links.apply(candidate)


def _scale_rules(model, links: Links) -> None:
    """A size, distance or shape value cannot follow a fitted scale (the scales are solved for the others)."""
    for name, path, _expression in links.ties:
        if path[1] in LINEAR_KEYS:
            continue
        for other in links.depends_on(name):
            other_path = links.paths.get(other)
            if other_path is not None and other_path[1] in LINEAR_KEYS:
                raise ValueError(f"{name} cannot follow the scale {other}: link sizes to sizes and scales to scales")


def with_inequalities(model, inequalities: Iterable[Inequality]):
    """``model`` with these inequality constraints (checked against its names)."""
    candidate = replace(model, inequalities=tuple(inequalities))
    Links(candidate)
    return candidate


def prune(model) -> tuple[object, list[str]]:
    """``model`` without the links and constraints that no longer hold (a name the model lost, a circle): what was
    dropped is returned as text."""
    dropped = []
    present = names(model)
    for path, parameter in model.parameters():
        if parameter.tie:
            try:
                expression = Expression(parameter.tie)
                ok = expression.names <= set(present)
            except ValueError:
                ok = False
            if not ok:
                dropped.append(f"{parameter_name(path)} = {parameter.tie}")
                model = model.with_parameter(path, tie=None)
    kept = []
    for inequality in getattr(model, "inequalities", ()) or ():
        try:
            ok = inequality.name in present and Expression(inequality.expression).names <= set(present)
        except ValueError:
            ok = False
        if ok:
            kept.append(inequality)
        else:
            dropped.append(inequality.text)
    if len(kept) != len(getattr(model, "inequalities", ()) or ()):
        model = replace(model, inequalities=tuple(kept))
    try:
        model = apply_links(model)
    except ValueError as exc:
        for path, parameter in model.parameters():
            if parameter.tie:
                model = model.with_parameter(path, tie=None)
        dropped.append(str(exc))
    return model, dropped


def renumbered(model, removed: int):
    """Links and constraints of ``model`` after component ``removed`` is taken out (call before removing it):
    names of later components move down by one, links reading the removed one are dropped."""
    mapping = removal_mapping(model, removed)
    for path, parameter in model.parameters():
        if parameter.tie:
            model = model.with_parameter(path, tie=rename(parameter.tie, mapping))
    kept = []
    for inequality in getattr(model, "inequalities", ()) or ():
        name = mapping.get(inequality.name, inequality.name)
        expression = rename(inequality.expression, mapping)
        if name is not None and expression is not None:
            kept.append(Inequality(name, inequality.relation, expression))
    return replace(model, inequalities=tuple(kept))


def describe(model) -> list[str]:
    """Every link and constraint as text (``R_2 = 2*R_1``, ``D_1 >= 2*R_1``)."""
    out = [f"{parameter_name(path)} = {parameter.tie}" for path, parameter in model.parameters() if parameter.tie]
    return out + [inequality.text for inequality in getattr(model, "inequalities", ()) or ()]


__all__ = ["CONSTANTS", "FUNCTIONS", "Expression", "Inequality", "Links", "apply_links", "describe", "linked",
           "links_of", "names", "parameter_name", "prune", "rename", "renumbered", "values", "with_inequalities"]
