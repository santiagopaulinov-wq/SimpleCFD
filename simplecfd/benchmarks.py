from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NumericExpectation:
    """Expected scalar/vector value, acceptable range, or both."""

    expected: Any | None = None
    lower: Any | None = None
    upper: Any | None = None
    rtol: float = 1e-8
    atol: float = 1e-10

    def __post_init__(self) -> None:
        if self.expected is None and self.lower is None and self.upper is None:
            raise ValueError("numeric expectation must define expected, lower, or upper")
        _validate_nonnegative_finite("rtol", self.rtol)
        _validate_nonnegative_finite("atol", self.atol)
        for name, value in (
            ("expected", self.expected),
            ("lower", self.lower),
            ("upper", self.upper),
        ):
            if value is not None:
                _validate_numeric_array(name, value)

    def failure_message(self, name: str, actual: Any) -> str | None:
        actual_array = _as_array("actual", actual)
        if self.expected is not None:
            expected_array = _as_array("expected", self.expected)
            try:
                close = np.allclose(
                    actual_array,
                    expected_array,
                    rtol=self.rtol,
                    atol=self.atol,
                )
            except ValueError:
                close = False
            if not close:
                return (
                    f"{name} expected {expected_array.tolist()} within "
                    f"rtol={self.rtol}, atol={self.atol}; got {actual_array.tolist()}"
                )

        if self.lower is not None:
            lower_array = _as_array("lower", self.lower)
            try:
                above_lower = np.all(actual_array >= lower_array - self.atol)
            except ValueError:
                above_lower = False
            if not above_lower:
                return (
                    f"{name} expected >= {lower_array.tolist()} with atol={self.atol}; "
                    f"got {actual_array.tolist()}"
                )

        if self.upper is not None:
            upper_array = _as_array("upper", self.upper)
            try:
                below_upper = np.all(actual_array <= upper_array + self.atol)
            except ValueError:
                below_upper = False
            if not below_upper:
                return (
                    f"{name} expected <= {upper_array.tolist()} with atol={self.atol}; "
                    f"got {actual_array.tolist()}"
                )

        return None


@dataclass(frozen=True)
class NumericalBenchmark:
    case_name: str
    variant: str
    configuration: dict[str, Any]
    pressure: NumericExpectation | None = None
    velocity: NumericExpectation | None = None
    mass_flow: NumericExpectation | None = None
    residual: NumericExpectation | None = None
    continuity_residual: NumericExpectation | None = None
    momentum_residual: NumericExpectation | None = None
    iterations: NumericExpectation | None = None

    def __post_init__(self) -> None:
        _validate_name("case_name", self.case_name)
        _validate_name("variant", self.variant)
        if not isinstance(self.configuration, dict):
            raise ValueError("configuration must be a dictionary")
        if not self.expectations:
            raise ValueError("benchmark must define at least one expectation")

    @property
    def expectations(self) -> dict[str, NumericExpectation]:
        candidates = {
            "pressure": self.pressure,
            "velocity": self.velocity,
            "mass_flow": self.mass_flow,
            "residual": self.residual,
            "continuity_residual": self.continuity_residual,
            "momentum_residual": self.momentum_residual,
            "iterations": self.iterations,
        }
        return {
            name: expectation
            for name, expectation in candidates.items()
            if expectation is not None
        }

    def check_result(
        self,
        solver_result: dict[str, Any],
        *,
        pressure: Any,
        velocity: Any,
        mass_flow: Any,
    ) -> list[str]:
        actuals = {
            "pressure": pressure,
            "velocity": velocity,
            "mass_flow": mass_flow,
            "residual": solver_result["residual"],
            "continuity_residual": solver_result["continuity_residual"],
            "momentum_residual": solver_result["momentum_residual"],
            "iterations": solver_result["iterations"],
        }
        failures = []
        for name, expectation in self.expectations.items():
            message = expectation.failure_message(name, actuals[name])
            if message is not None:
                failures.append(message)
        return failures

    def assert_result(
        self,
        solver_result: dict[str, Any],
        *,
        pressure: Any,
        velocity: Any,
        mass_flow: Any,
    ) -> None:
        failures = self.check_result(
            solver_result,
            pressure=pressure,
            velocity=velocity,
            mass_flow=mass_flow,
        )
        if failures:
            joined = "\n".join(f"- {failure}" for failure in failures)
            raise AssertionError(f"benchmark {self.case_name}[{self.variant}] failed:\n{joined}")


def _validate_name(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_nonnegative_finite(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative finite number")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")


def _validate_numeric_array(name: str, value: Any) -> None:
    array = _as_array(name, value)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values")


def _as_array(name: str, value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.dtype == object:
        raise ValueError(f"{name} must be numeric")
    return array
