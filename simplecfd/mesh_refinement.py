from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from simplecfd.comparison import compare_case_variants


PROFILE_FIELDS = ("pressure", "velocity", "mass_flow")


def run_mesh_refinement_study(
    case_name: str = "linear_nozzle_1d",
    node_counts: Iterable[int] = (5, 7, 9, 11),
    *,
    length: float = 1.0,
    sample_count: int = 101,
    case_configuration: Mapping[str, Any] | None = None,
    **case_kwargs: Any,
) -> dict[str, Any]:
    """Run a 1D mesh-refinement family and compare profiles on common grids.

    The target case must accept `n_pressure` and `dx` keyword arguments. Each
    mesh uses a uniform spacing `dx = length / (n_pressure - 1)`. Profiles are
    interpolated to common pressure and velocity grids, then compared against
    the finest mesh and against successive refinements.
    """

    counts = _normalize_node_counts(node_counts)
    _validate_length(length)
    _validate_sample_count(sample_count)
    configuration = _normalize_case_configuration(case_configuration, case_kwargs)

    solver_configurations = [
        {
            **configuration,
            "name": f"n{count}",
            "n_pressure": count,
            "dx": length / (count - 1),
        }
        for count in counts
    ]
    rows = compare_case_variants(case_name, solver_configurations)
    h_values = np.array([length / (count - 1) for count in counts], dtype=float)
    sample_positions = _sample_positions(rows, length=length, sample_count=sample_count)
    enriched_rows = _with_mesh_metadata(
        rows=rows,
        counts=counts,
        h_values=h_values,
        sample_positions=sample_positions,
    )
    reference_row = enriched_rows[-1]
    _attach_reference_errors(enriched_rows, reference_row)

    return {
        "case_name": case_name,
        "node_counts": counts,
        "length": float(length),
        "sample_positions": {
            key: value.copy() for key, value in sample_positions.items()
        },
        "runs": enriched_rows,
        "reference": reference_row,
        "spatial_convergence": _spatial_convergence(enriched_rows, h_values),
        "mass_flow": _mass_flow_summary(enriched_rows),
        "residuals": _residual_summary(enriched_rows),
    }


def _normalize_node_counts(node_counts: Iterable[int]) -> tuple[int, ...]:
    counts = tuple(sorted(node_counts))
    if len(counts) < 2:
        raise ValueError("node_counts must contain at least two meshes")
    if len(set(counts)) != len(counts):
        raise ValueError("node_counts must not contain duplicates")
    for count in counts:
        if not isinstance(count, int) or isinstance(count, bool) or count < 3:
            raise ValueError("node_counts must contain integers >= 3")
    return counts


def _validate_length(length: float) -> None:
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be a positive finite number")


def _validate_sample_count(sample_count: int) -> None:
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 2
    ):
        raise ValueError("sample_count must be an integer >= 2")


def _normalize_case_configuration(
    case_configuration: Mapping[str, Any] | None,
    case_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    configuration = dict({} if case_configuration is None else case_configuration)
    configuration.update(case_kwargs)
    reserved = {"name", "n_pressure", "dx"}
    conflicts = sorted(reserved.intersection(configuration))
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(f"mesh refinement controls these case keys: {joined}")
    return configuration


def _sample_positions(
    rows: list[dict[str, Any]],
    *,
    length: float,
    sample_count: int,
) -> dict[str, np.ndarray]:
    velocity_mins = [float(np.min(row["velocity_positions"])) for row in rows]
    velocity_maxs = [float(np.max(row["velocity_positions"])) for row in rows]
    velocity_start = max(velocity_mins)
    velocity_end = min(velocity_maxs)
    if velocity_start >= velocity_end:
        raise ValueError("velocity profiles do not share an overlapping spatial interval")

    return {
        "pressure": np.linspace(0.0, float(length), sample_count),
        "velocity": np.linspace(velocity_start, velocity_end, sample_count),
        "mass_flow": np.linspace(velocity_start, velocity_end, sample_count),
    }


def _with_mesh_metadata(
    *,
    rows: list[dict[str, Any]],
    counts: tuple[int, ...],
    h_values: np.ndarray,
    sample_positions: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    enriched_rows = []
    for row, count, h_value in zip(rows, counts, h_values):
        profiles = {
            "pressure": _interpolate_profile(
                row["pressure_positions"],
                row["final_pressure"],
                sample_positions["pressure"],
            ),
            "velocity": _interpolate_profile(
                row["velocity_positions"],
                row["final_velocity"],
                sample_positions["velocity"],
            ),
            "mass_flow": _interpolate_profile(
                row["velocity_positions"],
                row["final_mass_flow"],
                sample_positions["mass_flow"],
            ),
        }
        enriched = {
            **row,
            "n_pressure": count,
            "n_velocity": count - 1,
            "h": float(h_value),
            "interpolated_profiles": profiles,
        }
        enriched_rows.append(enriched)
    return enriched_rows


def _interpolate_profile(
    positions,
    values,
    sample_positions: np.ndarray,
) -> np.ndarray:
    return np.interp(
        sample_positions,
        np.asarray(positions, dtype=float),
        np.asarray(values, dtype=float),
    )


def _attach_reference_errors(
    rows: list[dict[str, Any]],
    reference_row: dict[str, Any],
) -> None:
    reference_profiles = reference_row["interpolated_profiles"]
    for row in rows:
        row["reference_errors"] = {
            field: _profile_error_metrics(
                row["interpolated_profiles"][field],
                reference_profiles[field],
            )
            for field in PROFILE_FIELDS
        }
        row["mass_flow_mean_error"] = float(
            abs(row["final_mass_flow_mean"] - reference_row["final_mass_flow_mean"])
        )
        row["residual_to_reference_ratio"] = _safe_ratio(
            row["final_residual"],
            reference_row["final_residual"],
        )


def _profile_error_metrics(values: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.asarray(values, dtype=float) - np.asarray(reference, dtype=float)
    return {
        "linf": float(np.linalg.norm(difference, ord=np.inf)),
        "l2": float(np.sqrt(np.mean(difference**2))),
    }


def _spatial_convergence(
    rows: list[dict[str, Any]],
    h_values: np.ndarray,
) -> dict[str, Any]:
    convergence = {}
    for field in PROFILE_FIELDS:
        reference_linf = np.array(
            [row["reference_errors"][field]["linf"] for row in rows],
            dtype=float,
        )
        successive_differences = np.array(
            [
                _profile_error_metrics(
                    rows[i]["interpolated_profiles"][field],
                    rows[i + 1]["interpolated_profiles"][field],
                )["linf"]
                for i in range(len(rows) - 1)
            ],
            dtype=float,
        )
        convergence[field] = {
            "reference_linf": reference_linf,
            "reference_l2": np.array(
                [row["reference_errors"][field]["l2"] for row in rows],
                dtype=float,
            ),
            "successive_linf": successive_differences,
            "error_reduction": _successive_ratios(reference_linf),
            "observed_order": _observed_orders(successive_differences, h_values),
            "monotone_to_reference": _monotone_nonincreasing(reference_linf),
        }
    return convergence


def _successive_ratios(values: np.ndarray) -> np.ndarray:
    ratios = np.full(max(values.size - 1, 0), np.nan, dtype=float)
    for i in range(ratios.size):
        ratios[i] = _safe_ratio(values[i], values[i + 1])
    return ratios


def _observed_orders(successive_differences: np.ndarray, h_values: np.ndarray) -> np.ndarray:
    if successive_differences.size < 2:
        return np.array([], dtype=float)
    orders = np.full(successive_differences.size - 1, np.nan, dtype=float)
    for i in range(orders.size):
        numerator = _safe_ratio(successive_differences[i], successive_differences[i + 1])
        denominator = _safe_ratio(h_values[i], h_values[i + 1])
        if numerator > 0.0 and denominator > 0.0 and denominator != 1.0:
            orders[i] = float(np.log(numerator) / np.log(denominator))
    return orders


def _monotone_nonincreasing(values: np.ndarray) -> bool:
    if values.size <= 1:
        return True
    return bool(np.all(values[:-1] + 1e-14 >= values[1:]))


def _mass_flow_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    final_means = np.array([row["final_mass_flow_mean"] for row in rows], dtype=float)
    reference = float(final_means[-1])
    return {
        "final_means": final_means,
        "reference": reference,
        "difference_to_reference": np.abs(final_means - reference),
        "max_difference_to_reference": float(np.max(np.abs(final_means - reference))),
        "successive_difference": np.abs(np.diff(final_means)),
    }


def _residual_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    final_residuals = np.array([row["final_residual"] for row in rows], dtype=float)
    return {
        "final": final_residuals,
        "continuity": np.array([row["continuity_residual"] for row in rows], dtype=float),
        "momentum": np.array([row["momentum_residual"] for row in rows], dtype=float),
        "converged": tuple(bool(row["converged"]) for row in rows),
        "iterations": np.array([row["iterations"] for row in rows], dtype=int),
        "all_converged": all(bool(row["converged"]) for row in rows),
        "max_final_residual": float(np.max(final_residuals)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if denominator == 0.0:
        if numerator == 0.0:
            return float("nan")
        return float("inf")
    return float(numerator / denominator)
