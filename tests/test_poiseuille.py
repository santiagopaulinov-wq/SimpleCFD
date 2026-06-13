import csv
from pathlib import Path

import numpy as np
import pytest

from simplecfd import PoiseuilleProblem, generate_poiseuille_benchmark, solve_poiseuille
from simplecfd.poiseuille import assemble_poiseuille_system, run_poiseuille_refinement


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_poiseuille_analytic_solution_matches_known_profile_values():
    problem = PoiseuilleProblem(
        channel_height=2.0,
        length=4.0,
        dynamic_viscosity=0.5,
        pressure_drop=8.0,
        n_nodes=5,
    )

    velocity = problem.analytic_velocity()

    np.testing.assert_allclose(problem.pressure_gradient_magnitude, 2.0)
    np.testing.assert_allclose(problem.centerline_velocity, 2.0)
    np.testing.assert_allclose(problem.mean_velocity, 4.0 / 3.0)
    np.testing.assert_allclose(problem.flow_rate_per_unit_width, 8.0 / 3.0)
    np.testing.assert_allclose(velocity, [0.0, 1.5, 2.0, 1.5, 0.0])


def test_poiseuille_finite_difference_solution_recovers_quadratic_profile():
    problem = PoiseuilleProblem(n_nodes=17)

    result = solve_poiseuille(problem)

    np.testing.assert_allclose(result.numerical_velocity, result.analytic_velocity, atol=1e-13)
    np.testing.assert_allclose(result.centerline_velocity, problem.centerline_velocity)
    assert result.l1_error < 1e-12
    assert result.linf_error < 1e-12
    assert result.l2_error < 1e-12


def test_poiseuille_system_keeps_no_slip_boundaries_and_symmetric_matrix():
    problem = PoiseuilleProblem(n_nodes=7)

    system = assemble_poiseuille_system(problem)

    np.testing.assert_allclose(system.diagonal[[0, -1]], [1.0, 1.0])
    np.testing.assert_allclose(system.rhs[[0, -1]], [0.0, 0.0])
    np.testing.assert_allclose(system.lower[1:-1], -1.0)
    np.testing.assert_allclose(system.diagonal[1:-1], 2.0)
    np.testing.assert_allclose(system.upper[1:-1], -1.0)
    np.testing.assert_allclose(system.lower[2:-1], system.upper[1:-2])


def test_poiseuille_flow_rate_converges_at_second_order_with_trapezoid_integration():
    results = run_poiseuille_refinement(node_counts=(9, 17, 33, 65))
    errors = np.array([result.flow_rate_relative_error for result in results])

    assert np.all(np.diff(errors) < 0.0)
    observed_orders = np.log(errors[:-1] / errors[1:]) / np.log(
        np.array([result.problem.spacing for result in results[:-1]])
        / np.array([result.problem.spacing for result in results[1:]])
    )
    np.testing.assert_allclose(observed_orders, 2.0, rtol=0.02)


def test_poiseuille_benchmark_writes_reproducible_artifacts():
    output_dir = Path("outputs") / "test_poiseuille_benchmark"

    report = generate_poiseuille_benchmark(
        output_dir,
        problem=PoiseuilleProblem(n_nodes=17),
        refinement_nodes=(9, 17, 33),
    )

    paths = report["paths"]
    assert set(paths) == {
        "summary_markdown",
        "profile_csv",
        "convergence_csv",
        "profile_png",
        "convergence_png",
        "profile_error_png",
        "profile_order_png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())

    profile_rows = _read_csv(paths["profile_csv"])
    convergence_rows = _read_csv(paths["convergence_csv"])
    assert len(profile_rows) == 17
    assert len(convergence_rows) == 3
    assert profile_rows[0]["y"] == "0.0"
    assert float(profile_rows[8]["pointwise_error"]) == pytest.approx(0.0, abs=1e-13)
    assert {"l1_error", "l2_error", "linf_error"}.issubset(convergence_rows[0])
    assert convergence_rows[1]["observed_l2_error_order"] == ""
    assert convergence_rows[0]["observed_flow_order"] == ""
    assert float(convergence_rows[-1]["observed_flow_order"]) == pytest.approx(2.0, rel=0.02)

    markdown = paths["summary_markdown"].read_text(encoding="utf-8")
    assert "# Plane Poiseuille Benchmark" in markdown
    assert "L1 profile error" in markdown
    assert "Analytic centerline velocity" in markdown
    assert "poiseuille_profile.png" in markdown


def test_poiseuille_problem_validates_physical_inputs():
    with pytest.raises(ValueError, match="channel_height"):
        PoiseuilleProblem(channel_height=0.0)
    with pytest.raises(ValueError, match="dynamic_viscosity"):
        PoiseuilleProblem(dynamic_viscosity=-1.0)
    with pytest.raises(ValueError, match="n_nodes"):
        PoiseuilleProblem(n_nodes=2)
