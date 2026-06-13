from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import product
from typing import Any

import numpy as np

from simplecfd.benchmarks import NumericalBenchmark
from simplecfd.cases import (
    build_case_by_name,
    get_case_benchmarks,
    list_available_cases,
)
from simplecfd.schemes.base import ConvectionScheme
from simplecfd.schemes import CentralDifference, Upwind


SCHEME_FACTORIES = {
    "upwind": Upwind,
    "central_difference": CentralDifference,
}


def compare_case_variants(
    case_name: str,
    solver_configurations: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    configurations = list(solver_configurations)
    if not configurations:
        raise ValueError("solver_configurations must contain at least one variant")

    seen_names: set[str] = set()
    results = []
    for index, configuration in enumerate(configurations, start=1):
        if not isinstance(configuration, Mapping):
            raise ValueError("each solver configuration must be a mapping")

        variant_name = _variant_name(configuration, index)
        if variant_name in seen_names:
            raise ValueError(f"duplicate solver configuration name '{variant_name}'")
        seen_names.add(variant_name)

        case_kwargs = {
            key: value
            for key, value in configuration.items()
            if key not in {"name", "label", "benchmark", "benchmark_variant"}
        }
        case = build_case_by_name(case_name, **case_kwargs)
        initial_state = case.solver.convergence_state()
        solver_result = case.solver.solve()
        final_pressure = case.field.p.copy()
        final_velocity = case.field.u.copy()
        final_pressure_correction = case.field.p_prime.copy()
        final_mass_flow = (
            case.definition.properties.density
            * final_velocity
            * case.geometry.velocity_areas
        )
        pressure_positions = case.geometry.pressure_positions
        velocity_positions = case.geometry.velocity_positions
        continuity_residual_vector = solver_result["continuity_residual_vector"].copy()
        momentum_residual_vector = solver_result["momentum_residual_vector"].copy()
        stability = numerical_stability_metrics(
            residual=solver_result["residual"],
            residual_history=solver_result["residual_history"],
            final_pressure=final_pressure,
            final_velocity=final_velocity,
            final_pressure_correction=final_pressure_correction,
            continuity_residual_vector=continuity_residual_vector,
            momentum_residual_vector=momentum_residual_vector,
        )
        convergence = normalized_convergence_metrics(
            initial_residual=initial_state["residual"],
            residual_history=solver_result["residual_history"],
            final_residual=solver_result["residual"],
            tolerance=case.solver.tolerance,
            iterations=solver_result["iterations"],
        )
        benchmark = _matching_benchmark(case_name, case, configuration)
        benchmark_error = benchmark_error_metrics(
            benchmark=benchmark,
            solver_result=solver_result,
            pressure=final_pressure,
            velocity=final_velocity,
            mass_flow=final_mass_flow,
        )
        linear_solve_history = _linear_solve_history(solver_result)
        linear_solve_counts = _linear_solve_counts(solver_result)
        computational_cost = _computational_cost(linear_solve_counts)
        failure_reason = _failure_reason(
            solver_result=solver_result,
            tolerance=case.solver.tolerance,
            max_iterations=case.solver.max_iterations,
        )
        results.append(
            {
                "case_name": case_name,
                "name": variant_name,
                "configuration": dict(case_kwargs),
                "converged": solver_result["converged"],
                "iterations": solver_result["iterations"],
                "initial_residual": convergence["initial_residual"],
                "residual": solver_result["residual"],
                "final_residual": solver_result["residual"],
                "residual_history": list(solver_result["residual_history"]),
                "residual_reduction_factor": convergence["residual_reduction_factor"],
                "residual_reduction_per_iteration": convergence[
                    "residual_reduction_per_iteration"
                ],
                "iterations_to_tolerance": convergence["iterations_to_tolerance"],
                "continuity_residual": solver_result["continuity_residual"],
                "momentum_residual": solver_result["momentum_residual"],
                "continuity_residual_vector": continuity_residual_vector,
                "momentum_residual_vector": momentum_residual_vector,
                "pressure_positions": pressure_positions,
                "velocity_positions": velocity_positions,
                "final_mass_flow": final_mass_flow.copy(),
                "final_mass_flow_mean": float(np.mean(final_mass_flow)),
                "final_pressure": final_pressure,
                "final_velocity": final_velocity,
                "final_pressure_correction": final_pressure_correction,
                "numerically_stable": stability["stable"],
                "numerical_stability": stability,
                "benchmark_variant": benchmark.variant if benchmark is not None else None,
                "benchmark_passed": benchmark_error["passed"],
                "benchmark_error": benchmark_error["max_relative_error"],
                "benchmark_error_metrics": benchmark_error,
                "failure_reason": failure_reason,
                "computational_cost": computational_cost,
                "linear_solve_history": linear_solve_history,
                "linear_solve_counts": linear_solve_counts,
                "linear_solves_total": linear_solve_counts["total"],
                "linear_solves_per_iteration": _linear_solves_per_iteration(
                    linear_solve_counts,
                    solver_result["iterations"],
                ),
                "cost_relative": float("nan"),
                "normalized_metrics": {
                    **convergence,
                    "stable": stability["stable"],
                    "final_mass_flow_mean": float(np.mean(final_mass_flow)),
                    "benchmark_error": benchmark_error["max_relative_error"],
                    "benchmark_passed": benchmark_error["passed"],
                    "failure_reason": failure_reason,
                    "computational_cost": computational_cost,
                    "linear_solve_counts": linear_solve_counts,
                    "linear_solves_total": linear_solve_counts["total"],
                    "linear_solves_per_iteration": _linear_solves_per_iteration(
                        linear_solve_counts,
                        solver_result["iterations"],
                    ),
                    "cost_relative": float("nan"),
                },
            }
        )

    _assign_relative_costs(results)
    return results


def compare_registered_cases(
    case_names: Iterable[str] | None = None,
    schemes: Iterable[str | ConvectionScheme | type[ConvectionScheme]] = (
        "upwind",
        "central_difference",
    ),
    couplings: Iterable[str] = ("simple", "simplec"),
    *,
    record_errors: bool = True,
    **common_configuration: Any,
) -> list[dict[str, Any]]:
    """Run a comparison table across registered cases, schemes, and couplings."""
    selected_cases = list(list_available_cases() if case_names is None else case_names)
    selected_schemes = list(schemes)
    selected_couplings = list(couplings)
    if not selected_cases:
        raise ValueError("case_names must contain at least one registered case")
    if not selected_schemes:
        raise ValueError("schemes must contain at least one scheme")
    if not selected_couplings:
        raise ValueError("couplings must contain at least one coupling strategy")

    rows = []
    for case_name, scheme_spec, coupling in product(
        selected_cases,
        selected_schemes,
        selected_couplings,
    ):
        scheme_name, scheme = _resolve_scheme(scheme_spec)
        row_name = f"{case_name}:{scheme_name}:{coupling}"
        configuration = {
            **common_configuration,
            "name": row_name,
            "scheme": scheme,
            "coupling": coupling,
        }
        try:
            row = compare_case_variants(case_name, [configuration])[0]
        except Exception as exc:
            if not record_errors:
                raise
            row = _failed_comparison_row(
                case_name=case_name,
                name=row_name,
                scheme_name=scheme_name,
                coupling=coupling,
                configuration=configuration,
                error=exc,
            )
        else:
            row["scheme_name"] = scheme_name
            row["coupling"] = coupling
        rows.append(row)

    _assign_relative_costs(rows)
    return rows


def compare_upwind_vs_central_difference(
    case_name: str = "versteeg_6_2",
    **common_configuration: Any,
) -> list[dict[str, Any]]:
    return compare_case_variants(
        case_name,
        [
            {
                **common_configuration,
                "name": "upwind",
                "scheme": Upwind(),
            },
            {
                **common_configuration,
                "name": "central_difference",
                "scheme": CentralDifference(),
            },
        ],
    )


def compare_simple_vs_simplec(
    case_names: Iterable[str] = ("versteeg_6_2", "linear_nozzle_1d"),
    **common_configuration: Any,
) -> list[dict[str, Any]]:
    results = []
    for case_name in case_names:
        results.extend(
            compare_case_variants(
                case_name,
                [
                    {
                        **common_configuration,
                        "name": "simple",
                        "coupling": "simple",
                    },
                    {
                        **common_configuration,
                        "name": "simplec",
                        "coupling": "simplec",
                    },
                ],
            )
        )
    return results


def compare_simple_family(
    case_names: Iterable[str] = (
        "versteeg_6_2",
        "linear_nozzle_1d",
        "smooth_linear_nozzle_1d",
        "strong_contraction_1d",
    ),
    **common_configuration: Any,
) -> list[dict[str, Any]]:
    """Compare SIMPLE, SIMPLEC, and SIMPLER on representative registered cases."""
    results = []
    for case_name in case_names:
        results.extend(
            compare_case_variants(
                case_name,
                [
                    {
                        **common_configuration,
                        "name": "simple",
                        "coupling": "simple",
                    },
                    {
                        **common_configuration,
                        "name": "simplec",
                        "coupling": "simplec",
                    },
                    {
                        **common_configuration,
                        "name": "simpler",
                        "coupling": "simpler",
                    },
                ],
            )
        )
    return results


def numerical_stability_metrics(
    *,
    residual: float,
    residual_history: Iterable[float],
    final_pressure,
    final_velocity,
    final_pressure_correction,
    continuity_residual_vector,
    momentum_residual_vector,
    stability_limit: float = 1e12,
) -> dict[str, float | bool]:
    residual_history_array = np.asarray(list(residual_history), dtype=float)
    arrays = (
        np.asarray(final_pressure, dtype=float),
        np.asarray(final_velocity, dtype=float),
        np.asarray(final_pressure_correction, dtype=float),
        np.asarray(continuity_residual_vector, dtype=float),
        np.asarray(momentum_residual_vector, dtype=float),
        residual_history_array,
    )
    all_finite = bool(np.isfinite(residual) and all(np.all(np.isfinite(a)) for a in arrays))
    max_abs_pressure = _max_abs(final_pressure)
    max_abs_velocity = _max_abs(final_velocity)
    max_abs_pressure_correction = _max_abs(final_pressure_correction)
    max_abs_continuity_residual = _max_abs(continuity_residual_vector)
    max_abs_momentum_residual = _max_abs(momentum_residual_vector)
    max_abs_residual_history = _max_abs(residual_history_array)
    bounded = bool(
        abs(float(residual)) <= stability_limit
        and max_abs_pressure <= stability_limit
        and max_abs_velocity <= stability_limit
        and max_abs_pressure_correction <= stability_limit
        and max_abs_continuity_residual <= stability_limit
        and max_abs_momentum_residual <= stability_limit
        and max_abs_residual_history <= stability_limit
    )
    return {
        "stable": all_finite and bounded,
        "all_finite": all_finite,
        "bounded": bounded,
        "stability_limit": float(stability_limit),
        "final_residual_finite": bool(np.isfinite(residual)),
        "max_abs_pressure": max_abs_pressure,
        "max_abs_velocity": max_abs_velocity,
        "max_abs_pressure_correction": max_abs_pressure_correction,
        "max_abs_continuity_residual": max_abs_continuity_residual,
        "max_abs_momentum_residual": max_abs_momentum_residual,
        "max_abs_residual_history": max_abs_residual_history,
    }


def normalized_convergence_metrics(
    *,
    initial_residual: float,
    residual_history: Iterable[float],
    final_residual: float,
    tolerance: float,
    iterations: int,
) -> dict[str, float | int | None]:
    history = np.asarray(list(residual_history), dtype=float)
    initial = float(initial_residual)
    final = float(final_residual)
    first_iteration_to_tolerance = _first_iteration_to_tolerance(history, tolerance)
    reduction_factor = _safe_ratio(initial, final)
    reduction_per_iteration = _reduction_per_iteration(
        initial=initial,
        final=final,
        iterations=iterations,
    )
    return {
        "initial_residual": initial,
        "final_residual": final,
        "residual_reduction_factor": reduction_factor,
        "residual_reduction_per_iteration": reduction_per_iteration,
        "iterations_to_tolerance": first_iteration_to_tolerance,
    }


def benchmark_error_metrics(
    *,
    benchmark: NumericalBenchmark | None,
    solver_result: Mapping[str, Any],
    pressure,
    velocity,
    mass_flow,
) -> dict[str, Any]:
    if benchmark is None:
        return {
            "benchmark_variant": None,
            "passed": None,
            "failures": [],
            "max_relative_error": float("nan"),
            "max_normalized_error": float("nan"),
            "metric_errors": {},
        }

    actuals = {
        "pressure": pressure,
        "velocity": velocity,
        "mass_flow": mass_flow,
        "residual": solver_result["residual"],
        "continuity_residual": solver_result["continuity_residual"],
        "momentum_residual": solver_result["momentum_residual"],
        "iterations": solver_result["iterations"],
    }
    metric_errors = {
        name: _expectation_error(actuals[name], expectation)
        for name, expectation in benchmark.expectations.items()
    }
    max_relative_error = _max_metric_error(metric_errors, "relative_error")
    max_normalized_error = _max_metric_error(metric_errors, "normalized_error")
    failures = benchmark.check_result(
        solver_result,
        pressure=pressure,
        velocity=velocity,
        mass_flow=mass_flow,
    )
    return {
        "benchmark_variant": benchmark.variant,
        "passed": not failures,
        "failures": failures,
        "max_relative_error": max_relative_error,
        "max_normalized_error": max_normalized_error,
        "metric_errors": metric_errors,
    }


def _variant_name(configuration: Mapping[str, Any], index: int) -> str:
    name = configuration.get("name", configuration.get("label", f"variant_{index}"))
    if not isinstance(name, str) or not name.strip():
        raise ValueError("solver configuration name must be a non-empty string")
    return name


def _max_abs(values) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def _first_iteration_to_tolerance(
    residual_history: np.ndarray,
    tolerance: float,
) -> int | None:
    finite_hits = np.flatnonzero(np.isfinite(residual_history) & (residual_history < tolerance))
    if finite_hits.size == 0:
        return None
    return int(finite_hits[0] + 1)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if denominator == 0.0:
        return float("inf") if numerator > 0.0 else float("nan")
    return float(numerator / denominator)


def _reduction_per_iteration(
    *,
    initial: float,
    final: float,
    iterations: int,
) -> float:
    if iterations <= 0 or initial <= 0.0 or final < 0.0:
        return float("nan")
    if not np.isfinite(initial) or not np.isfinite(final):
        return float("nan")
    if final == 0.0:
        return 0.0
    return float((final / initial) ** (1.0 / iterations))


def _computational_cost(linear_solve_counts: Mapping[str, Any]) -> float:
    return float(linear_solve_counts["total"])


def _failure_reason(
    *,
    solver_result: Mapping[str, Any],
    tolerance: float,
    max_iterations: int,
) -> str:
    if solver_result["converged"]:
        return ""
    return (
        "did not converge within "
        f"{max_iterations} iterations; final residual "
        f"{solver_result['residual']} >= tolerance {tolerance}"
    )


def _linear_solve_history(solver_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"total": int(entry["total"]), "by_kind": dict(entry["by_kind"])}
        for entry in solver_result.get("linear_solve_history", [])
    ]


def _linear_solve_counts(solver_result: Mapping[str, Any]) -> dict[str, Any]:
    counts = solver_result.get("linear_solve_counts")
    if counts is not None:
        return {"total": int(counts["total"]), "by_kind": dict(counts["by_kind"])}

    by_kind: dict[str, int] = {}
    for entry in _linear_solve_history(solver_result):
        for solve_kind, count in entry["by_kind"].items():
            by_kind[solve_kind] = by_kind.get(solve_kind, 0) + int(count)
    return {"total": sum(by_kind.values()), "by_kind": by_kind}


def _linear_solves_per_iteration(
    linear_solve_counts: Mapping[str, Any],
    iterations: int,
) -> float:
    if iterations <= 0:
        return float("nan")
    return float(linear_solve_counts["total"] / iterations)


def _assign_relative_costs(rows: list[dict[str, Any]]) -> None:
    finite_costs = [
        row["computational_cost"]
        for row in rows
        if np.isfinite(row.get("computational_cost", float("nan")))
        and row.get("computational_cost", 0.0) > 0.0
    ]
    if not finite_costs:
        return
    baseline = min(finite_costs)
    for row in rows:
        cost = row.get("computational_cost", float("nan"))
        row["cost_relative"] = float(cost / baseline) if np.isfinite(cost) else float("inf")
        if "normalized_metrics" in row:
            row["normalized_metrics"]["cost_relative"] = row["cost_relative"]


def _matching_benchmark(
    case_name: str,
    case,
    configuration: Mapping[str, Any],
) -> NumericalBenchmark | None:
    requested_variant = configuration.get("benchmark_variant", configuration.get("benchmark"))
    benchmarks = get_case_benchmarks(case_name)
    if requested_variant is not None:
        for benchmark in benchmarks:
            if benchmark.variant == requested_variant:
                return benchmark
        return None

    actual_configuration = _canonical_configuration(case, configuration)
    for benchmark in benchmarks:
        if _configuration_matches(actual_configuration, benchmark.configuration):
            return benchmark
    return None


def _canonical_configuration(case, configuration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "coupling": configuration.get("coupling", "simple"),
        "scheme": _scheme_name(case.definition.scheme),
        "tolerance": case.solver.tolerance,
        "max_iterations": case.solver.max_iterations,
        "pressure_relaxation": case.step_solver.pressure_relaxation,
        "velocity_relaxation": case.step_solver.velocity_relaxation,
    }


def _configuration_matches(
    actual_configuration: Mapping[str, Any],
    benchmark_configuration: Mapping[str, Any],
) -> bool:
    for key, expected in benchmark_configuration.items():
        if key not in actual_configuration:
            return False
        actual = actual_configuration[key]
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if not np.isclose(actual, expected, rtol=1e-12, atol=1e-15):
                return False
        elif actual != expected:
            return False
    return True


def _expectation_error(actual: Any, expectation) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    relative_errors = []
    normalized_errors = []

    if expectation.expected is not None:
        expected_array = np.asarray(expectation.expected, dtype=float)
        difference = np.abs(actual_array - expected_array)
        relative_errors.append(_relative_error(difference, expected_array))
        tolerance_scale = expectation.atol + expectation.rtol * np.abs(expected_array)
        normalized_errors.append(_max_with_broadcast(difference, tolerance_scale))

    if expectation.lower is not None:
        lower_array = np.asarray(expectation.lower, dtype=float)
        lower_excess = np.maximum(lower_array - actual_array, 0.0)
        relative_errors.append(_relative_error(lower_excess, lower_array))
        normalized_errors.append(_max_with_broadcast(lower_excess, np.maximum(expectation.atol, 1e-300)))

    if expectation.upper is not None:
        upper_array = np.asarray(expectation.upper, dtype=float)
        upper_excess = np.maximum(actual_array - upper_array, 0.0)
        relative_errors.append(_relative_error(upper_excess, upper_array))
        normalized_errors.append(_max_with_broadcast(upper_excess, np.maximum(expectation.atol, 1e-300)))

    return {
        "relative_error": max(relative_errors) if relative_errors else float("nan"),
        "normalized_error": max(normalized_errors) if normalized_errors else float("nan"),
    }


def _relative_error(difference: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.maximum(np.abs(reference), 1.0)
    return _max_with_broadcast(difference, denominator)


def _max_with_broadcast(numerator: np.ndarray, denominator) -> float:
    try:
        ratio = numerator / denominator
    except ValueError:
        return float("inf")
    if np.asarray(ratio).size == 0:
        return 0.0
    return float(np.max(np.asarray(ratio, dtype=float)))


def _max_metric_error(metric_errors: Mapping[str, Mapping[str, float]], key: str) -> float:
    values = [
        error[key]
        for error in metric_errors.values()
        if np.isfinite(error.get(key, float("nan")))
    ]
    if not values:
        return float("nan")
    return float(max(values))


def _resolve_scheme(
    scheme: str | ConvectionScheme | type[ConvectionScheme],
) -> tuple[str, ConvectionScheme]:
    if isinstance(scheme, str):
        normalized = scheme.strip().lower()
        try:
            return normalized, SCHEME_FACTORIES[normalized]()
        except KeyError as exc:
            available = ", ".join(sorted(SCHEME_FACTORIES))
            raise ValueError(
                f"unknown convection scheme '{scheme}'. Available schemes: {available}"
            ) from exc
    if isinstance(scheme, type):
        instance = scheme()
        return _scheme_name(instance), instance
    return _scheme_name(scheme), scheme


def _scheme_name(scheme: ConvectionScheme) -> str:
    if isinstance(scheme, Upwind):
        return "upwind"
    if isinstance(scheme, CentralDifference):
        return "central_difference"
    return scheme.__class__.__name__.lower()


def _failed_comparison_row(
    *,
    case_name: str,
    name: str,
    scheme_name: str,
    coupling: str,
    configuration: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "name": name,
        "scheme_name": scheme_name,
        "coupling": coupling,
        "configuration": dict(configuration),
        "converged": False,
        "iterations": 0,
        "initial_residual": float("inf"),
        "residual": float("inf"),
        "final_residual": float("inf"),
        "residual_history": [],
        "residual_reduction_factor": float("nan"),
        "residual_reduction_per_iteration": float("nan"),
        "iterations_to_tolerance": None,
        "continuity_residual": float("inf"),
        "momentum_residual": float("inf"),
        "continuity_residual_vector": np.array([], dtype=float),
        "momentum_residual_vector": np.array([], dtype=float),
        "pressure_positions": np.array([], dtype=float),
        "velocity_positions": np.array([], dtype=float),
        "final_mass_flow": np.array([], dtype=float),
        "final_mass_flow_mean": float("nan"),
        "final_pressure": np.array([], dtype=float),
        "final_velocity": np.array([], dtype=float),
        "final_pressure_correction": np.array([], dtype=float),
        "numerically_stable": False,
        "numerical_stability": {
            "stable": False,
            "all_finite": False,
            "bounded": False,
            "stability_limit": 1e12,
            "final_residual_finite": False,
            "max_abs_pressure": float("nan"),
            "max_abs_velocity": float("nan"),
            "max_abs_pressure_correction": float("nan"),
            "max_abs_continuity_residual": float("nan"),
            "max_abs_momentum_residual": float("nan"),
            "max_abs_residual_history": float("nan"),
        },
        "benchmark_variant": None,
        "benchmark_passed": False,
        "benchmark_error": float("nan"),
        "benchmark_error_metrics": {
            "benchmark_variant": None,
            "passed": False,
            "failures": [f"{error.__class__.__name__}: {error}"],
            "max_relative_error": float("nan"),
            "max_normalized_error": float("nan"),
            "metric_errors": {},
        },
        "failure_reason": f"{error.__class__.__name__}: {error}",
        "computational_cost": float("inf"),
        "linear_solve_history": [],
        "linear_solve_counts": {"total": 0, "by_kind": {}},
        "linear_solves_total": 0,
        "linear_solves_per_iteration": float("nan"),
        "cost_relative": float("inf"),
        "normalized_metrics": {
            "initial_residual": float("inf"),
            "final_residual": float("inf"),
            "residual_reduction_factor": float("nan"),
            "residual_reduction_per_iteration": float("nan"),
            "iterations_to_tolerance": None,
            "stable": False,
            "final_mass_flow_mean": float("nan"),
            "benchmark_error": float("nan"),
            "benchmark_passed": False,
            "failure_reason": f"{error.__class__.__name__}: {error}",
            "computational_cost": float("inf"),
            "linear_solve_counts": {"total": 0, "by_kind": {}},
            "linear_solves_total": 0,
            "linear_solves_per_iteration": float("nan"),
            "cost_relative": float("inf"),
        },
        "error": f"{error.__class__.__name__}: {error}",
    }
