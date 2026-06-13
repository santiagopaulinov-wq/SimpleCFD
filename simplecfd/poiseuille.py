from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from simplecfd.analytic_benchmarks import (
    error_norms,
    finish_plot,
    markdown_table,
    observed_order,
    plot_error_convergence,
    plot_observed_orders,
    trapezoid,
    validate_positive,
    write_csv,
)
from simplecfd.coefficients import LinearSystem
from simplecfd.linalg import tdma


@dataclass(frozen=True)
class PoiseuilleProblem:
    """Plane Poiseuille flow between fixed parallel plates.

    The channel spans `0 <= y <= channel_height`, has no-slip walls, and is
    driven by a positive pressure drop over `length`.
    """

    channel_height: float = 1.0
    length: float = 1.0
    dynamic_viscosity: float = 1.0
    pressure_drop: float = 8.0
    n_nodes: int = 33

    def __post_init__(self) -> None:
        validate_positive("channel_height", self.channel_height)
        validate_positive("length", self.length)
        validate_positive("dynamic_viscosity", self.dynamic_viscosity)
        validate_positive("pressure_drop", self.pressure_drop)
        if not isinstance(self.n_nodes, int) or isinstance(self.n_nodes, bool):
            raise ValueError("n_nodes must be an integer")
        if self.n_nodes < 3:
            raise ValueError("n_nodes must be at least 3")

    @property
    def pressure_gradient_magnitude(self) -> float:
        return self.pressure_drop / self.length

    @property
    def spacing(self) -> float:
        return self.channel_height / (self.n_nodes - 1)

    @property
    def coordinates(self) -> np.ndarray:
        return np.linspace(0.0, self.channel_height, self.n_nodes)

    @property
    def centerline_velocity(self) -> float:
        return (
            self.pressure_gradient_magnitude
            * self.channel_height**2
            / (8.0 * self.dynamic_viscosity)
        )

    @property
    def mean_velocity(self) -> float:
        return (
            self.pressure_gradient_magnitude
            * self.channel_height**2
            / (12.0 * self.dynamic_viscosity)
        )

    @property
    def flow_rate_per_unit_width(self) -> float:
        return self.mean_velocity * self.channel_height

    def analytic_velocity(self, y: np.ndarray | None = None) -> np.ndarray:
        coordinates = self.coordinates if y is None else np.asarray(y, dtype=float)
        return (
            self.pressure_gradient_magnitude
            * coordinates
            * (self.channel_height - coordinates)
            / (2.0 * self.dynamic_viscosity)
        )


@dataclass(frozen=True)
class PoiseuilleResult:
    problem: PoiseuilleProblem
    coordinates: np.ndarray
    numerical_velocity: np.ndarray
    analytic_velocity: np.ndarray
    pointwise_error: np.ndarray
    l1_error: float
    l2_error: float
    linf_error: float
    numerical_flow_rate: float
    analytic_flow_rate: float
    flow_rate_relative_error: float

    @property
    def centerline_velocity(self) -> float:
        index = self.problem.n_nodes // 2
        return float(self.numerical_velocity[index])


def solve_poiseuille(problem: PoiseuilleProblem | None = None) -> PoiseuilleResult:
    selected_problem = PoiseuilleProblem() if problem is None else problem
    y = selected_problem.coordinates
    system = assemble_poiseuille_system(selected_problem)
    numerical = tdma(system)
    analytic = selected_problem.analytic_velocity(y)
    error = numerical - analytic
    norms = error_norms(error)
    numerical_flow_rate = trapezoid(numerical, y)
    analytic_flow_rate = selected_problem.flow_rate_per_unit_width
    return PoiseuilleResult(
        problem=selected_problem,
        coordinates=y,
        numerical_velocity=numerical,
        analytic_velocity=analytic,
        pointwise_error=error,
        l1_error=norms.l1,
        l2_error=norms.l2,
        linf_error=norms.linf,
        numerical_flow_rate=numerical_flow_rate,
        analytic_flow_rate=analytic_flow_rate,
        flow_rate_relative_error=abs(numerical_flow_rate - analytic_flow_rate)
        / analytic_flow_rate,
    )


def assemble_poiseuille_system(problem: PoiseuilleProblem) -> LinearSystem:
    """Assemble a second-order finite-difference Poiseuille velocity system."""
    n = problem.n_nodes
    lower = np.zeros(n)
    diagonal = np.ones(n)
    upper = np.zeros(n)
    rhs = np.zeros(n)

    forcing = problem.pressure_gradient_magnitude * problem.spacing**2
    forcing /= problem.dynamic_viscosity

    for i in range(1, n - 1):
        lower[i] = -1.0
        diagonal[i] = 2.0
        upper[i] = -1.0
        rhs[i] = forcing

    return LinearSystem(lower=lower, diagonal=diagonal, upper=upper, rhs=rhs)


def run_poiseuille_refinement(
    *,
    channel_height: float = 1.0,
    length: float = 1.0,
    dynamic_viscosity: float = 1.0,
    pressure_drop: float = 8.0,
    node_counts: tuple[int, ...] = (9, 17, 33, 65),
) -> list[PoiseuilleResult]:
    return [
        solve_poiseuille(
            PoiseuilleProblem(
                channel_height=channel_height,
                length=length,
                dynamic_viscosity=dynamic_viscosity,
                pressure_drop=pressure_drop,
                n_nodes=n_nodes,
            )
        )
        for n_nodes in node_counts
    ]


def generate_poiseuille_benchmark(
    output_dir: str | Path,
    *,
    problem: PoiseuilleProblem | None = None,
    refinement_nodes: tuple[int, ...] = (9, 17, 33, 65),
) -> dict[str, Any]:
    """Solve Poiseuille flow and write reproducible CSV, Markdown, and PNG artifacts."""
    selected_problem = PoiseuilleProblem() if problem is None else problem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = solve_poiseuille(selected_problem)
    refinement = run_poiseuille_refinement(
        channel_height=selected_problem.channel_height,
        length=selected_problem.length,
        dynamic_viscosity=selected_problem.dynamic_viscosity,
        pressure_drop=selected_problem.pressure_drop,
        node_counts=refinement_nodes,
    )
    convergence_rows = _convergence_rows(refinement)

    paths = {
        "summary_markdown": output_path / "poiseuille_summary.md",
        "profile_csv": output_path / "poiseuille_profile.csv",
        "convergence_csv": output_path / "poiseuille_convergence.csv",
        "profile_png": output_path / "poiseuille_profile.png",
        "convergence_png": output_path / "poiseuille_flow_convergence.png",
        "profile_error_png": output_path / "poiseuille_profile_error_convergence.png",
        "profile_order_png": output_path / "poiseuille_profile_observed_orders.png",
    }

    write_csv(
        paths["profile_csv"],
        ("y", "numerical_velocity", "analytic_velocity", "pointwise_error"),
        _profile_rows(result),
    )
    write_csv(
        paths["convergence_csv"],
        (
            "n_nodes",
            "dy",
            "l1_error",
            "l2_error",
            "linf_error",
            "observed_l1_error_order",
            "observed_l2_error_order",
            "observed_linf_error_order",
            "numerical_flow_rate",
            "analytic_flow_rate",
            "flow_rate_relative_error",
            "observed_flow_order",
        ),
        convergence_rows,
    )
    paths["summary_markdown"].write_text(
        _summary_markdown(result, convergence_rows, paths),
        encoding="utf-8",
    )
    _plot_profile(result, paths["profile_png"])
    _plot_convergence(convergence_rows, paths["convergence_png"])
    _plot_profile_error_convergence(convergence_rows, paths["profile_error_png"])
    _plot_profile_observed_orders(convergence_rows, paths["profile_order_png"])

    return {
        "result": result,
        "refinement": refinement,
        "convergence_rows": convergence_rows,
        "paths": paths,
    }


def _profile_rows(result: PoiseuilleResult) -> list[dict[str, float]]:
    return [
        {
            "y": y,
            "numerical_velocity": numerical,
            "analytic_velocity": analytic,
            "pointwise_error": error,
        }
        for y, numerical, analytic, error in zip(
            result.coordinates,
            result.numerical_velocity,
            result.analytic_velocity,
            result.pointwise_error,
        )
    ]


def _convergence_rows(results: list[PoiseuilleResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: PoiseuilleResult | None = None
    for result in results:
        order = ""
        if previous is not None:
            previous_error = previous.flow_rate_relative_error
            current_error = result.flow_rate_relative_error
            if previous_error > 0.0 and current_error > 0.0:
                order = observed_order(
                    previous_error,
                    current_error,
                    previous.problem.spacing,
                    result.problem.spacing,
                )
        rows.append(
            {
                "n_nodes": result.problem.n_nodes,
                "dy": result.problem.spacing,
                "l1_error": result.l1_error,
                "l2_error": result.l2_error,
                "linf_error": result.linf_error,
                "numerical_flow_rate": result.numerical_flow_rate,
                "analytic_flow_rate": result.analytic_flow_rate,
                "flow_rate_relative_error": result.flow_rate_relative_error,
                "observed_flow_order": order,
            }
        )
        previous = result
    return _with_profile_orders(rows)


def _plot_profile(result: PoiseuilleResult, path: Path) -> None:
    fine_y = np.linspace(0.0, result.problem.channel_height, 300)
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(result.problem.analytic_velocity(fine_y), fine_y, label="Analytic", linewidth=2.0)
    plt.plot(
        result.numerical_velocity,
        result.coordinates,
        "o",
        label=f"TDMA, n={result.problem.n_nodes}",
    )
    plt.xlabel("Velocity")
    plt.ylabel("Wall-normal coordinate")
    plt.title("Plane Poiseuille velocity profile")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    finish_plot(path)


def _plot_convergence(rows: list[dict[str, Any]], path: Path) -> None:
    spacing = np.asarray([row["dy"] for row in rows], dtype=float)
    errors = np.asarray([row["flow_rate_relative_error"] for row in rows], dtype=float)
    plt.figure(figsize=(6.4, 4.2))
    plt.loglog(spacing, errors, "o-", linewidth=1.8)
    plt.gca().invert_xaxis()
    plt.xlabel("Grid spacing")
    plt.ylabel("Relative flow-rate error")
    plt.title("Poiseuille flow-rate convergence")
    plt.grid(True, which="both", linestyle="--", alpha=0.45)
    finish_plot(path)


def _plot_profile_error_convergence(rows: list[dict[str, Any]], path: Path) -> None:
    plot_error_convergence(
        rows,
        path,
        error_columns=("l1_error", "l2_error", "linf_error"),
        labels=("L1", "L2", "Linf"),
        title="Poiseuille profile error convergence",
    )


def _plot_profile_observed_orders(rows: list[dict[str, Any]], path: Path) -> None:
    plot_observed_orders(
        rows,
        path,
        order_columns=(
            "observed_l1_error_order",
            "observed_l2_error_order",
            "observed_linf_error_order",
        ),
        labels=("L1", "L2", "Linf"),
        title="Poiseuille profile observed orders",
    )


def _summary_markdown(
    result: PoiseuilleResult,
    convergence_rows: list[dict[str, Any]],
    paths: dict[str, Path],
) -> str:
    problem = result.problem
    return "\n".join(
        [
            "# Plane Poiseuille Benchmark",
            "",
            "## Problem",
            "",
            f"- Channel height: {problem.channel_height}",
            f"- Length: {problem.length}",
            f"- Dynamic viscosity: {problem.dynamic_viscosity}",
            f"- Pressure drop: {problem.pressure_drop}",
            f"- Nodes: {problem.n_nodes}",
            "",
            "## Validation",
            "",
            f"- Analytic centerline velocity: {problem.centerline_velocity:.12g}",
            f"- Numerical centerline velocity: {result.centerline_velocity:.12g}",
            f"- L1 profile error: {result.l1_error:.12g}",
            f"- L2 profile error: {result.l2_error:.12g}",
            f"- Linf profile error: {result.linf_error:.12g}",
            f"- Analytic flow rate per unit width: {result.analytic_flow_rate:.12g}",
            f"- Numerical flow rate per unit width: {result.numerical_flow_rate:.12g}",
            f"- Flow-rate relative error: {result.flow_rate_relative_error:.12g}",
            "",
            "## Refinement",
            "",
            markdown_table(
                convergence_rows,
                (
                    "n_nodes",
                    "dy",
                    "l1_error",
                    "l2_error",
                    "linf_error",
                    "observed_l2_error_order",
                    "flow_rate_relative_error",
                    "observed_flow_order",
                ),
            ),
            "",
            "## Artifacts",
            "",
            f"- Profile CSV: `{paths['profile_csv'].name}`",
            f"- Convergence CSV: `{paths['convergence_csv'].name}`",
            f"- Profile figure: `{paths['profile_png'].name}`",
            f"- Convergence figure: `{paths['convergence_png'].name}`",
            f"- Profile error figure: `{paths['profile_error_png'].name}`",
            f"- Profile observed-order figure: `{paths['profile_order_png'].name}`",
            "",
        ]
    )


def _with_profile_orders(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    previous: dict[str, Any] | None = None
    for row in rows:
        enriched_row = dict(row)
        if previous is None:
            enriched_row["observed_l1_error_order"] = ""
            enriched_row["observed_l2_error_order"] = ""
            enriched_row["observed_linf_error_order"] = ""
        else:
            for column in ("l1_error", "l2_error", "linf_error"):
                enriched_row[f"observed_{column}_order"] = observed_order(
                    float(previous[column]),
                    float(row[column]),
                    float(previous["dy"]),
                    float(row["dy"]),
                )
        enriched.append(enriched_row)
        previous = row
    return enriched
