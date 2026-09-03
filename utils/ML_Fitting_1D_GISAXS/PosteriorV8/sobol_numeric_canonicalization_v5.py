"""Versioned numeric backends for V5 query/codec replay.

The ordinary GUI path uses the platform ``libm`` backend for speed.  Frozen
direct-Sobol artifacts use a fixed Decimal context instead: binary64 inputs
are converted exactly with :meth:`Decimal.from_float`, transcendental results
are evaluated at the declared precision, and the final Decimal is converted
back to the nearest binary64 value.  No ambient Decimal context, environment
variable, or per-call override is consulted.

The deterministic policy is a reproducibility contract for a finite frozen
design, not a claim that arbitrary real arithmetic is platform independent.
Every promoted design must still pass the byte-exact cross-platform manifest
gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    ROUND_HALF_EVEN,
    localcontext,
)
from functools import lru_cache
from hashlib import sha256
import json
import math


V5_FAST_NUMERIC_POLICY_VERSION = "posterior_v8_platform_libm_binary64_fast_v1"
V5_DETERMINISTIC_DECIMAL_PRECISION = 80
V5_DETERMINISTIC_DECIMAL_EMIN = -999999
V5_DETERMINISTIC_DECIMAL_EMAX = 999999
V5_DETERMINISTIC_DECIMAL_TRAPS = (
    "InvalidOperation",
    "DivisionByZero",
    "Overflow",
)
V5_DETERMINISTIC_NUMERIC_POLICY_VERSION = (
    "posterior_v8_decimal80_half_even_exact_float_input_binary64_output_v2"
)
V5_NUMERIC_POLICY_VERSIONS = (
    V5_FAST_NUMERIC_POLICY_VERSION,
    V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
)
V5_NUMERIC_POLICY_CONTRACT_SCHEMA = "gisaxs.posterior_v8.numeric_policy/v1"
V5_NUMERIC_POLICY_CONTRACT_VERSION = (
    "posterior_v8_explicit_fast_or_decimal80_transcendental_contract_v1"
)

_DECIMAL_PI = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944592307816406286"
    "2089986280348253421170679"
)
_FIXED_DECIMAL_CONTEXT = Context(
    prec=V5_DETERMINISTIC_DECIMAL_PRECISION,
    rounding=ROUND_HALF_EVEN,
    Emin=V5_DETERMINISTIC_DECIMAL_EMIN,
    Emax=V5_DETERMINISTIC_DECIMAL_EMAX,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[InvalidOperation, DivisionByZero, Overflow],
)


def validate_v5_numeric_policy(value: object) -> str:
    if not isinstance(value, str) or value not in V5_NUMERIC_POLICY_VERSIONS:
        raise ValueError(f"numeric policy must be one of {V5_NUMERIC_POLICY_VERSIONS}")
    return value


def v5_numeric_policy_payload(policy_version: str) -> dict[str, object]:
    """Return the complete semantic contract behind one numeric-policy ID."""

    policy = validate_v5_numeric_policy(policy_version)
    common = {
        "schema": V5_NUMERIC_POLICY_CONTRACT_SCHEMA,
        "version": V5_NUMERIC_POLICY_CONTRACT_VERSION,
        "policy_version": policy,
        "input_domain": "finite_binary64",
        "output_domain": "finite_binary64",
        "signed_zero_output": "canonical_positive_zero",
    }
    if policy == V5_FAST_NUMERIC_POLICY_VERSION:
        return {
            **common,
            "deterministic_across_platform_libm": False,
            "backend": "python_math_platform_libm_binary64",
            "operations": ["log", "exp", "sqrt", "hypot", "atan"],
        }
    return {
        **common,
        "deterministic_across_platform_libm": True,
        "backend": "python_decimal_context_from_exact_binary64_input",
        "decimal_context": {
            "precision": V5_DETERMINISTIC_DECIMAL_PRECISION,
            "rounding": "ROUND_HALF_EVEN",
            "Emin": V5_DETERMINISTIC_DECIMAL_EMIN,
            "Emax": V5_DETERMINISTIC_DECIMAL_EMAX,
            "capitals": 1,
            "clamp": 0,
            "traps": list(V5_DETERMINISTIC_DECIMAL_TRAPS),
        },
        "operations": {
            "log": "decimal.Context.ln",
            "exp": "decimal.Context.exp",
            "sqrt": "decimal.Context.sqrt",
            "hypot": "sqrt(x*x+y*y)_in_fixed_context",
            "atan": (
                "odd_Taylor_abs_le_0.5;pi_over_4_reduction_for_0.5_abs_x_le_2;"
                "reciprocal_pi_over_2_reduction_for_abs_x_gt_2"
            ),
        },
        "decimal_input": "Decimal.from_float_exact_binary64",
        "binary64_output": "nearest_binary64_after_fixed_context_evaluation",
    }


def v5_numeric_policy_sha256(policy_version: str) -> str:
    payload = v5_numeric_policy_payload(policy_version)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _finite_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} input must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} input must be finite")
    return result


def _result_float(value: Decimal, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise OverflowError(f"{name} result is not finite binary64")
    return 0.0 if result == 0.0 else result


def _decimal_context():
    return localcontext(_FIXED_DECIMAL_CONTEXT)


@lru_cache(maxsize=65536)
def _decimal_log(value: float) -> float:
    numeric = _finite_float(value, "log")
    if numeric <= 0.0:
        raise ValueError("log input must be strictly positive")
    with _decimal_context() as context:
        result = context.ln(Decimal.from_float(numeric))
    return _result_float(result, "log")


@lru_cache(maxsize=65536)
def _decimal_exp(value: float) -> float:
    numeric = _finite_float(value, "exp")
    with _decimal_context() as context:
        result = context.exp(Decimal.from_float(numeric))
    return _result_float(result, "exp")


@lru_cache(maxsize=65536)
def _decimal_sqrt(value: float) -> float:
    numeric = _finite_float(value, "sqrt")
    if numeric < 0.0:
        raise ValueError("sqrt input must be non-negative")
    with _decimal_context() as context:
        result = context.sqrt(Decimal.from_float(numeric))
    return _result_float(result, "sqrt")


def _atan_series(value: Decimal, context) -> Decimal:
    """Evaluate atan after range reduction to ``abs(value) <= 0.5``."""

    if not value:
        return Decimal(0)
    square = context.multiply(value, value)
    power = value
    total = value
    sign = -1
    denominator = 3
    # With |value| <= .5 the terms contract by at least 1/4.  The explicit
    # ceiling is a fail-closed guard against an accidental range-reduction bug.
    threshold = Decimal(1).scaleb(-(context.prec + 8))
    for _ in range(512):
        power = context.multiply(power, square)
        term = context.divide(power, Decimal(denominator))
        total = context.subtract(total, term) if sign < 0 else context.add(total, term)
        if abs(term) < threshold:
            return +total
        sign *= -1
        denominator += 2
    raise ArithmeticError("deterministic atan series did not converge")


@lru_cache(maxsize=65536)
def _decimal_atan(value: float) -> float:
    numeric = _finite_float(value, "atan")
    with _decimal_context() as context:
        x = Decimal.from_float(numeric)
        negative = x < 0
        x = abs(x)
        # The Taylor series is intentionally restricted to |argument| <= .5.
        # Reciprocal reduction is therefore safe only above 2; values in
        # (.5, 2] use the pi/4 identity whose reduced argument is at most 1/3.
        if x > 2:
            result = context.subtract(
                context.divide(_DECIMAL_PI, Decimal(2)),
                _atan_series(context.divide(Decimal(1), x), context),
            )
        elif x > Decimal("0.5"):
            reduced = context.divide(
                context.subtract(x, Decimal(1)),
                context.add(x, Decimal(1)),
            )
            result = context.add(
                context.divide(_DECIMAL_PI, Decimal(4)),
                _atan_series(reduced, context),
            )
        else:
            result = _atan_series(x, context)
        if negative:
            result = context.minus(result)
    return _result_float(result, "atan")


@lru_cache(maxsize=65536)
def _decimal_hypot(left: float, right: float) -> float:
    x = _finite_float(left, "hypot")
    y = _finite_float(right, "hypot")
    with _decimal_context() as context:
        dx = Decimal.from_float(x)
        dy = Decimal.from_float(y)
        square_sum = context.add(context.multiply(dx, dx), context.multiply(dy, dy))
        result = context.sqrt(square_sum)
    return _result_float(result, "hypot")


@dataclass(frozen=True)
class V5NumericOps:
    """Closed numeric operation set derived only from a frozen policy string."""

    version: str

    def __post_init__(self) -> None:
        validate_v5_numeric_policy(self.version)

    @property
    def deterministic(self) -> bool:
        return self.version == V5_DETERMINISTIC_NUMERIC_POLICY_VERSION

    def log(self, value: float) -> float:
        return _decimal_log(value) if self.deterministic else math.log(value)

    def exp(self, value: float) -> float:
        return _decimal_exp(value) if self.deterministic else math.exp(value)

    def sqrt(self, value: float) -> float:
        return _decimal_sqrt(value) if self.deterministic else math.sqrt(value)

    def hypot(self, left: float, right: float) -> float:
        return (
            _decimal_hypot(left, right)
            if self.deterministic
            else math.hypot(left, right)
        )

    def atan(self, value: float) -> float:
        return _decimal_atan(value) if self.deterministic else math.atan(value)


@lru_cache(maxsize=len(V5_NUMERIC_POLICY_VERSIONS))
def v5_numeric_ops(policy_version: str) -> V5NumericOps:
    return V5NumericOps(validate_v5_numeric_policy(policy_version))


__all__ = [
    "V5_DETERMINISTIC_DECIMAL_EMAX",
    "V5_DETERMINISTIC_DECIMAL_EMIN",
    "V5_DETERMINISTIC_DECIMAL_PRECISION",
    "V5_DETERMINISTIC_DECIMAL_TRAPS",
    "V5_DETERMINISTIC_NUMERIC_POLICY_VERSION",
    "V5_FAST_NUMERIC_POLICY_VERSION",
    "V5_NUMERIC_POLICY_VERSIONS",
    "V5_NUMERIC_POLICY_CONTRACT_SCHEMA",
    "V5_NUMERIC_POLICY_CONTRACT_VERSION",
    "V5NumericOps",
    "v5_numeric_policy_payload",
    "v5_numeric_policy_sha256",
    "v5_numeric_ops",
    "validate_v5_numeric_policy",
]
