import csv
from pathlib import Path

import numpy as np
import pytest

from simplecfd import CouetteProblem, generate_couette_benchmark, solve_couette
from simplecfd.couette import assemble_couette_system, run_couette_refinement


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_couette_analytic_solution_matches_known_linear_profile():
    problem = CouetteProblem(
        channel_height=2.0,
        lower_wall_velocity=-1.0,
        upper_wall_velocity=3.0,
        dynamic_viscosity=0.5,
        n_nodes=5,
    )

    velocity = problem.analytic_velocity()

    np.testing.assert_allclose(velocity, [-1.0, 0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(problem.mean_velocity, 1.0)
    np.testing.assert_allclose(problem.flow_rate_per_unit_width, 2.0)
    np.testing.assert_allclose(problem.wall_shear_stress, 1.0)
    np.testing.assert_allclose(problem.kinetic_energy_per_unit_width, 14.0 / 3.0)


def test_couette_discrete_system_applies_no_slip_dirichlet_boundaries():
    problem = CouetteProblem(lower_wall_velocity=2.0, upper_wall_velocity=5.0, n_nodes=7)

    system = assemble_couette_system(problem)

    np.testing.assert_allclose(system.diagonal[[0, -1]], [1.0, 1.0])
    np.testing.assert_allclose(system.rhs[[0, -1]], [2.0, 5.0])
    np.testing.assert_allclose(system.lower[1:-1], -1.0)
    np.testing.assert_allclose(system.diagonal[1:-1], 2.0)
    np.testing.assert_allclose(system.upper[1:-1], -1.0)
    np.testing.assert_allclose(system.rhs[1:-1], 0.0)


def test_couette_numerical_solution_recovers_linear_profile_and_boundaries():
    problem = CouetteProblem(
        channel_height=1.5,
        lower_wall_velocity=0.25,
        upper_wall_velocity=1.75,
        n_nodes=19,
    )

    result = solve_couette(problem)

    np.testing.assert_allclose(result.numerical_velocity, result.analytic_velocity, atol=1e-13)
    np.testing.assert_allclose(result.numerical_velocity[[0, -1]], [0.25, 1.75])
    assert result.l1_error < 1e-12
    assert result.linf_error < 1e-12
    assert result.l2_error < 1e-12
    assert result.flow_rate_absolute_error < 1e-12


def test_couette_kinetic_energy_converges_at_second_order():
    results = run_couette_refinement(node_counts=(9, 17, 33, 65))
    errors = np.array([result.kinetic_energy_relative_error for result in results])

    assert np.all(np.diff(errors) < 0.0)
    observed_orders = np.log(errors[:-1] / errors[1:]) / np.log(
        np.array([result.problem.spacing for result in results[:-1]])
        / np.array([result.problem.spacing for result in results[1:]])
    )
    np.testing.assert_allclose(observed_orders, 2.0, rtol=0.02)


def test_couette_benchmark_writes_reproducible_artifacts():
    output_dir = Path("outputs") / "test_couette_benchmark"

    report = generate_couette_benchmark(
        output_dir,
        problem=CouetteProblem(n_nodes=17),
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
    assert profile_rows[0]["numerical_velocity"] == "0.0"
    assert profile_rows[-1]["numerical_velocity"] == "1.0"
    assert float(profile_rows[8]["pointwise_error"]) == pytest.approx(0.0, abs=1e-13)
    assert {"l1_error", "l2_error", "linf_error"}.issubset(convergence_rows[0])
    assert convergence_rows[1]["observed_l2_error_order"] == ""
    assert convergence_rows[0]["observed_energy_order"] == ""
    assert float(convergence_rows[-1]["observed_energy_order"]) == pytest.approx(2.0, rel=0.02)

    markdown = paths["summary_markdown"].read_text(encoding="utf-8")
    assert "# Plane Couette Benchmark" in markdown
    assert "L1 profile error" in markdown
    assert "Wall shear stress" in markdown
    assert "couette_profile.png" in markdown


def test_couette_problem_validates_inputs():
    with pytest.raises(ValueError, match="channel_height"):
        CouetteProblem(channel_height=0.0)
    with pytest.raises(ValueError, match="dynamic_viscosity"):
        CouetteProblem(dynamic_viscosity=-1.0)
    with pytest.raises(ValueError, match="upper_wall_velocity"):
        CouetteProblem(upper_wall_velocity=np.inf)
    with pytest.raises(ValueError, match="n_nodes"):
        CouetteProblem(n_nodes=2)
