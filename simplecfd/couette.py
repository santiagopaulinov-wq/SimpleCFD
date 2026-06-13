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
    validate_finite,
    validate_positive,
    write_csv,
)
from simplecfd.coefficients import LinearSystem
from simplecfd.linalg import tdma


@dataclass(frozen=True)
class CouetteProblem:
    """Steady plane Couette flow between no-slip parallel plates."""

    channel_height: float = 1.0
    lower_wall_velocity: float = 0.0
    upper_wall_velocity: float = 1.0
    dynamic_viscosity: float = 1.0
    n_nodes: int = 33

    def __post_init__(self) -> None:
        validate_positive("channel_height", self.channel_height)
        validate_positive("dynamic_viscosity", self.dynamic_viscosity)
        validate_finite("lower_wall_velocity", self.lower_wall_velocity)
        validate_finite("upper_wall_velocity", self.upper_wall_velocity)
        if not isinstance(self.n_nodes, int) or isinstance(self.n_nodes, bool):
            raise ValueError("n_nodes must be an integer")
        if self.n_nodes < 3:
            raise ValueError("n_nodes must be at least 3")

    @property
    def spacing(self) -> float:
        return self.channel_height / (self.n_nodes - 1)

    @property
    def coordinates(self) -> np.ndarray:
        return np.linspace(0.0, self.channel_height, self.n_nodes)

    @property
    def velocity_difference(self) -> float:
        return self.upper_wall_velocity - self.lower_wall_velocity

    @property
    def wall_shear_stress(self) -> float:
        return self.dynamic_viscosity * self.velocity_difference / self.channel_height

    @property
    def mean_velocity(self) -> float:
        return 0.5 * (self.lower_wall_velocity + self.upper_wall_velocity)

    @property
    def flow_rate_per_unit_width(self) -> float:
        return self.mean_velocity * self.channel_height

    @property
    def kinetic_energy_per_unit_width(self) -> float:
        u0 = self.lower_wall_velocity
        u1 = self.upper_wall_velocity
        return self.channel_height * (u0**2 + u0 * u1 + u1**2) / 3.0

    def analytic_velocity(self, y: np.ndarray | None = None) -> np.ndarray:
        coordinates = self.coordinates if y is None else np.asarray(y, dtype=float)
        return self.lower_wall_velocity + self.velocity_difference * (
            coordinates / self.channel_height
        )


@dataclass(frozen=True)
class CouetteResult:
    problem: CouetteProblem
    coordinates: np.ndarray
    numerical_velocity: np.ndarray
    analytic_velocity: np.ndarray
    pointwise_error: np.ndarray
    l1_error: float
    l2_error: float
    linf_error: float
    numerical_flow_rate: float
    analytic_flow_rate: float
    flow_rate_absolute_error: float
    numerical_kinetic_energy: float
    analytic_kinetic_energy: float
    kinetic_energy_relative_error: float


def solve_couette(problem: CouetteProblem | None = None) -> CouetteResult:
    selected_problem = CouetteProblem() if problem is None else problem
    y = selected_problem.coordinates
    numerical = tdma(assemble_couette_system(selected_problem))
    analytic = selected_problem.analytic_velocity(y)
    error = numerical - analytic
    norms = error_norms(error)
    numerical_flow_rate = trapezoid(numerical, y)
    analytic_flow_rate = selected_problem.flow_rate_per_unit_width
    numerical_energy = trapezoid(numerical**2, y)
    analytic_energy = selected_problem.kinetic_energy_per_unit_width
    return CouetteResult(
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
        flow_rate_absolute_error=abs(numerical_flow_rate - analytic_flow_rate),
        numerical_kinetic_energy=numerical_energy,
        analytic_kinetic_energy=analytic_energy,
        kinetic_energy_relative_error=abs(numerical_energy - analytic_energy)
        / analytic_energy,
    )


def assemble_couette_system(problem: CouetteProblem) -> LinearSystem:
    """Assemble the second-order Laplace equation for steady Couette flow."""
    n = problem.n_nodes
    lower = np.zeros(n)
    diagonal = np.ones(n)
    upper = np.zeros(n)
    rhs = np.zeros(n)

    rhs[0] = problem.lower_wall_velocity
    rhs[-1] = problem.upper_wall_velocity
    for i in range(1, n - 1):
        lower[i] = -1.0
        diagonal[i] = 2.0
        upper[i] = -1.0

    return LinearSystem(lower=lower, diagonal=diagonal, upper=upper, rhs=rhs)


def run_couette_refinement(
    *,
    channel_height: float = 1.0,
    lower_wall_velocity: float = 0.0,
    upper_wall_velocity: float = 1.0,
    dynamic_viscosity: float = 1.0,
    node_counts: tuple[int, ...] = (9, 17, 33, 65),
) -> list[CouetteResult]:
    return [
        solve_couette(
            CouetteProblem(
                channel_height=channel_height,
                lower_wall_velocity=lower_wall_velocity,
                upper_wall_velocity=upper_wall_velocity,
                dynamic_viscosity=dynamic_viscosity,
                n_nodes=n_nodes,
            )
        )
        for n_nodes in node_counts
    ]


def generate_couette_benchmark(
    output_dir: str | Path,
    *,
    problem: CouetteProblem | None = None,
    refinement_nodes: tuple[int, ...] = (9, 17, 33, 65),
) -> dict[str, Any]:
    selected_problem = CouetteProblem() if problem is None else problem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = solve_couette(selected_problem)
    refinement = run_couette_refinement(
        channel_height=selected_problem.channel_height,
        lower_wall_velocity=selected_problem.lower_wall_velocity,
        upper_wall_velocity=selected_problem.upper_wall_velocity,
        dynamic_viscosity=selected_problem.dynamic_viscosity,
        node_counts=refinement_nodes,
    )
    convergence_rows = _convergence_rows(refinement)

    paths = {
        "summary_markdown": output_path / "couette_summary.md",
        "profile_csv": output_path / "couette_profile.csv",
        "convergence_csv": output_path / "couette_convergence.csv",
        "profile_png": output_path / "couette_profile.png",
        "convergence_png": output_path / "couette_energy_convergence.png",
        "profile_error_png": output_path / "couette_profile_error_convergence.png",
        "profile_order_png": output_path / "couette_profile_observed_orders.png",
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
            "flow_rate_absolute_error",
            "numerical_kinetic_energy",
            "analytic_kinetic_energy",
            "kinetic_energy_relative_error",
            "observed_energy_order",
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


def _profile_rows(result: CouetteResult) -> list[dict[str, float]]:
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


def _convergence_rows(results: list[CouetteResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: CouetteResult | None = None
    for result in results:
        order = ""
        if previous is not None:
            order = observed_order(
                previous.kinetic_energy_relative_error,
                result.kinetic_energy_relative_error,
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
                "flow_rate_absolute_error": result.flow_rate_absolute_error,
                "numerical_kinetic_energy": result.numerical_kinetic_energy,
                "analytic_kinetic_energy": result.analytic_kinetic_energy,
                "kinetic_energy_relative_error": result.kinetic_energy_relative_error,
                "observed_energy_order": order,
            }
        )
        previous = result
    return _with_profile_orders(rows)


def _plot_profile(result: CouetteResult, path: Path) -> None:
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
    plt.title("Plane Couette velocity profile")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    finish_plot(path)


def _plot_convergence(rows: list[dict[str, Any]], path: Path) -> None:
    spacing = np.asarray([row["dy"] for row in rows], dtype=float)
    errors = np.asarray([row["kinetic_energy_relative_error"] for row in rows], dtype=float)
    plt.figure(figsize=(6.4, 4.2))
    plt.loglog(spacing, errors, "o-", linewidth=1.8)
    plt.gca().invert_xaxis()
    plt.xlabel("Grid spacing")
    plt.ylabel("Relative kinetic-energy error")
    plt.title("Couette kinetic-energy convergence")
    plt.grid(True, which="both", linestyle="--", alpha=0.45)
    finish_plot(path)


def _plot_profile_error_convergence(rows: list[dict[str, Any]], path: Path) -> None:
    plot_error_convergence(
        rows,
        path,
        error_columns=("l1_error", "l2_error", "linf_error"),
        labels=("L1", "L2", "Linf"),
        title="Couette profile error convergence",
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
        title="Couette profile observed orders",
    )


def _summary_markdown(
    result: CouetteResult,
    convergence_rows: list[dict[str, Any]],
    paths: dict[str, Path],
) -> str:
    problem = result.problem
    return "\n".join(
        [
            "# Plane Couette Benchmark",
            "",
            "## Problem",
            "",
            f"- Channel height: {problem.channel_height}",
            f"- Lower wall velocity: {problem.lower_wall_velocity}",
            f"- Upper wall velocity: {problem.upper_wall_velocity}",
            f"- Dynamic viscosity: {problem.dynamic_viscosity}",
            f"- Nodes: {problem.n_nodes}",
            "",
            "## Validation",
            "",
            f"- Wall shear stress: {problem.wall_shear_stress:.12g}",
            f"- L1 profile error: {result.l1_error:.12g}",
            f"- L2 profile error: {result.l2_error:.12g}",
            f"- Linf profile error: {result.linf_error:.12g}",
            f"- Analytic flow rate per unit width: {result.analytic_flow_rate:.12g}",
            f"- Numerical flow rate per unit width: {result.numerical_flow_rate:.12g}",
            f"- Analytic kinetic energy per unit width: {result.analytic_kinetic_energy:.12g}",
            f"- Numerical kinetic energy per unit width: {result.numerical_kinetic_energy:.12g}",
            f"- Kinetic-energy relative error: {result.kinetic_energy_relative_error:.12g}",
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
                    "kinetic_energy_relative_error",
                    "observed_energy_order",
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
