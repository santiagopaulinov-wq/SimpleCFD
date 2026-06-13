import numpy as np
import pytest

from simplecfd.cases import list_available_cases
from simplecfd.comparison import (
    compare_case_variants,
    compare_registered_cases,
    compare_simple_family,
    compare_simple_vs_simplec,
    compare_upwind_vs_central_difference,
)
from simplecfd.schemes import CentralDifference, Upwind


def test_compare_upwind_vs_central_difference_returns_comparable_metrics():
    results = compare_upwind_vs_central_difference(max_iterations=100)

    assert [result["name"] for result in results] == ["upwind", "central_difference"]
    assert {result["case_name"] for result in results} == {"versteeg_6_2"}

    upwind, central = results
    assert upwind["converged"] is True
    assert central["converged"] is False
    assert len(upwind["residual_history"]) == upwind["iterations"]
    assert len(central["residual_history"]) == central["iterations"]

    for result in results:
        assert result["iterations"] <= result["configuration"]["max_iterations"]
        assert np.isfinite(result["residual"])
        assert np.isfinite(result["continuity_residual"])
        assert np.isfinite(result["momentum_residual"])
        assert result["final_pressure"].shape == (5,)
        assert result["final_velocity"].shape == (4,)
        assert result["final_pressure_correction"].shape == (5,)
        assert result["final_mass_flow"].shape == (4,)
        assert np.isfinite(result["final_mass_flow_mean"])
        assert result["continuity_residual_vector"].shape == (5,)
        assert result["momentum_residual_vector"].shape == (4,)
        assert np.all(np.isfinite(result["final_pressure"]))
        assert np.all(np.isfinite(result["final_velocity"]))
        assert np.all(np.isfinite(result["final_pressure_correction"]))


def test_compare_case_variants_accepts_registered_case_and_solver_configurations():
    results = compare_case_variants(
        "versteeg_6_2",
        [
            {"name": "upwind_short", "scheme": Upwind(), "max_iterations": 3},
            {
                "name": "central_short",
                "scheme": CentralDifference(),
                "max_iterations": 3,
            },
        ],
    )

    assert [result["name"] for result in results] == ["upwind_short", "central_short"]
    assert all(result["iterations"] == 3 for result in results)
    assert all(result["residual_history"] for result in results)


def test_compare_simple_vs_simplec_reports_metrics_for_registered_1d_cases():
    results = compare_simple_vs_simplec(max_iterations=5)

    assert len(results) == 4
    assert {(result["case_name"], result["name"]) for result in results} == {
        ("versteeg_6_2", "simple"),
        ("versteeg_6_2", "simplec"),
        ("linear_nozzle_1d", "simple"),
        ("linear_nozzle_1d", "simplec"),
    }

    for result in results:
        assert result["iterations"] <= 5
        assert result["final_residual"] == result["residual"]
        assert isinstance(result["converged"], bool)
        assert isinstance(result["numerically_stable"], bool)
        assert result["numerical_stability"]["stable"] is result["numerically_stable"]
        assert result["numerical_stability"]["all_finite"] is True
        assert result["numerical_stability"]["final_residual_finite"] is True
        assert isinstance(result["numerical_stability"]["bounded"], bool)
        assert result["numerical_stability"]["stability_limit"] == 1e12
        assert result["numerical_stability"]["max_abs_pressure"] >= 0.0
        assert result["numerical_stability"]["max_abs_velocity"] >= 0.0
        assert result["numerical_stability"]["max_abs_residual_history"] >= 0.0
        assert np.isfinite(result["final_residual"])


def test_compare_case_variants_returns_field_copies():
    result = compare_case_variants(
        "versteeg_6_2",
        [{"name": "upwind", "scheme": Upwind(), "max_iterations": 3}],
    )[0]
    pressure_before = result["final_pressure"].copy()
    velocity_before = result["final_velocity"].copy()

    result["final_pressure"] += 1.0
    result["final_velocity"] += 1.0
    second_result = compare_case_variants(
        "versteeg_6_2",
        [{"name": "upwind", "scheme": Upwind(), "max_iterations": 3}],
    )[0]

    np.testing.assert_allclose(second_result["final_pressure"], pressure_before)
    np.testing.assert_allclose(second_result["final_velocity"], velocity_before)


def test_compare_case_variants_reports_final_mass_flow_by_velocity_node():
    result = compare_case_variants(
        "versteeg_6_2",
        [{"name": "upwind", "scheme": Upwind(), "max_iterations": 100}],
    )[0]
    expected_mass_flow = result["final_velocity"] * np.array([0.45, 0.35, 0.25, 0.15])

    np.testing.assert_allclose(result["final_mass_flow"], expected_mass_flow)
    np.testing.assert_allclose(result["final_mass_flow_mean"], np.mean(expected_mass_flow))
    np.testing.assert_allclose(result["pressure_positions"], [0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(result["velocity_positions"], [0.25, 0.75, 1.25, 1.75])


def test_compare_case_variants_reports_nonuniform_geometry_positions():
    result = compare_case_variants(
        "linear_nozzle_1d",
        [
            {
                "name": "nonuniform",
                "n_pressure": 5,
                "dx": np.array([0.1, 0.2, 0.3, 0.4]),
                "max_iterations": 3,
            }
        ],
    )[0]

    np.testing.assert_allclose(result["pressure_positions"], [0.0, 0.1, 0.3, 0.6, 1.0])
    np.testing.assert_allclose(result["velocity_positions"], [0.05, 0.2, 0.45, 0.8])


def test_compare_case_variants_reports_normalized_convergence_metrics():
    result = compare_case_variants(
        "versteeg_6_2",
        [{"name": "upwind", "scheme": Upwind(), "max_iterations": 100}],
    )[0]

    assert result["initial_residual"] > result["final_residual"]
    assert result["final_residual"] == result["residual"]
    assert result["residual_reduction_factor"] > 1.0
    assert 0.0 < result["residual_reduction_per_iteration"] < 1.0
    assert result["iterations_to_tolerance"] == result["iterations"]
    assert result["normalized_metrics"]["initial_residual"] == result["initial_residual"]
    assert result["normalized_metrics"]["final_residual"] == result["final_residual"]
    assert result["normalized_metrics"]["iterations_to_tolerance"] == result["iterations"]
    assert result["normalized_metrics"]["stable"] is result["numerically_stable"]
    assert result["normalized_metrics"]["final_mass_flow_mean"] == result["final_mass_flow_mean"]


def test_compare_case_variants_reports_error_against_matching_benchmark():
    result = compare_case_variants(
        "versteeg_6_2",
        [
            {
                "name": "golden",
                "scheme": Upwind(),
                "coupling": "simple",
                "tolerance": 1e-5,
                "max_iterations": 100,
                "pressure_relaxation": 0.7,
                "velocity_relaxation": 0.7,
            }
        ],
    )[0]

    assert result["benchmark_variant"] == "golden_simple_upwind"
    assert result["benchmark_passed"] is True
    assert result["benchmark_error"] < 1e-12
    assert result["benchmark_error_metrics"]["max_normalized_error"] <= 1.0
    assert result["benchmark_error_metrics"]["failures"] == []
    assert result["normalized_metrics"]["benchmark_passed"] is True


def test_compare_case_variants_can_select_benchmark_variant_without_passing_it_to_case_builder():
    result = compare_case_variants(
        "versteeg_6_2",
        [
            {
                "name": "explicit_benchmark",
                "scheme": Upwind(),
                "benchmark_variant": "golden_simple_upwind",
                "max_iterations": 2,
            }
        ],
    )[0]

    assert result["benchmark_variant"] == "golden_simple_upwind"
    assert result["benchmark_passed"] is False
    assert result["benchmark_error"] > 0.0
    assert result["benchmark_error_metrics"]["failures"]


def test_compare_case_variants_reports_relative_cost_inside_comparison_table():
    fast, slow = compare_case_variants(
        "versteeg_6_2",
        [
            {"name": "fast", "scheme": Upwind(), "max_iterations": 2},
            {"name": "slow", "scheme": Upwind(), "max_iterations": 4},
        ],
    )

    assert fast["computational_cost"] == fast["linear_solves_total"] == 4
    assert slow["computational_cost"] == slow["linear_solves_total"] == 8
    assert fast["cost_relative"] == 1.0
    assert slow["cost_relative"] == 2.0
    assert slow["normalized_metrics"]["cost_relative"] == 2.0


def test_compare_registered_cases_builds_case_scheme_coupling_table():
    rows = compare_registered_cases(
        case_names=["constant_area_1d", "smooth_linear_nozzle_1d"],
        schemes=["upwind", CentralDifference],
        couplings=["simple", "simplec"],
        max_iterations=5,
    )

    assert len(rows) == 8
    assert {
        (row["case_name"], row["scheme_name"], row["coupling"])
        for row in rows
    } == {
        ("constant_area_1d", "upwind", "simple"),
        ("constant_area_1d", "upwind", "simplec"),
        ("constant_area_1d", "central_difference", "simple"),
        ("constant_area_1d", "central_difference", "simplec"),
        ("smooth_linear_nozzle_1d", "upwind", "simple"),
        ("smooth_linear_nozzle_1d", "upwind", "simplec"),
        ("smooth_linear_nozzle_1d", "central_difference", "simple"),
        ("smooth_linear_nozzle_1d", "central_difference", "simplec"),
    }

    upwind_rows = [
        row
        for row in rows
        if row["scheme_name"] == "upwind" and "error" not in row
    ]
    assert all(row["iterations"] <= 5 for row in upwind_rows)
    assert all(row["residual_history"] for row in upwind_rows)
    assert all(row["final_mass_flow"].shape == row["final_velocity"].shape for row in upwind_rows)
    assert all(row["final_pressure"].size > 0 for row in upwind_rows)


def test_compare_registered_cases_records_numerical_failures_as_rows():
    rows = compare_registered_cases(
        case_names=["constant_area_1d"],
        schemes=["central_difference"],
        couplings=["simple"],
        max_iterations=5,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["converged"] is False
    assert row["numerically_stable"] is False
    assert row["residual_history"] == []
    assert row["final_pressure"].shape == (0,)
    assert "ZeroDivisionError" in row["error"]


def test_compare_registered_cases_compares_simple_against_simplec_for_registered_cases():
    rows = compare_registered_cases(
        case_names=list_available_cases(),
        schemes=["upwind"],
        couplings=["simple", "simplec"],
        max_iterations=10,
    )

    assert len(rows) == 2 * len(list_available_cases())
    assert {
        (row["case_name"], row["coupling"])
        for row in rows
    } == {
        (case_name, coupling)
        for case_name in list_available_cases()
        for coupling in ("simple", "simplec")
    }

    simple_rows = [row for row in rows if row["coupling"] == "simple"]
    assert all(row["final_mass_flow"].shape == row["final_velocity"].shape for row in simple_rows)
    assert all(row["residual_history"] for row in simple_rows)

    constant_simplec = next(
        row
        for row in rows
        if row["case_name"] == "constant_area_1d" and row["coupling"] == "simplec"
    )
    assert constant_simplec["converged"] is False
    assert "error" not in constant_simplec
    assert constant_simplec["iterations"] == 10
    assert constant_simplec["residual_history"]


def test_compare_registered_cases_compares_simple_simplec_and_simpler():
    rows = compare_registered_cases(
        case_names=["versteeg_6_2", "linear_nozzle_1d"],
        schemes=["upwind"],
        couplings=["simple", "simplec", "simpler"],
        max_iterations=3,
    )

    assert len(rows) == 6
    assert {
        (row["case_name"], row["coupling"])
        for row in rows
    } == {
        (case_name, coupling)
        for case_name in ("versteeg_6_2", "linear_nozzle_1d")
        for coupling in ("simple", "simplec", "simpler")
    }

    simpler_rows = [row for row in rows if row["coupling"] == "simpler"]
    assert all("error" not in row for row in simpler_rows)
    assert all(row["iterations"] <= 3 for row in simpler_rows)
    assert all(row["final_pressure"].size > 0 for row in simpler_rows)
    assert all(row["final_mass_flow"].shape == row["final_velocity"].shape for row in simpler_rows)

    expected_by_coupling = {
        "simple": {"momentum": 3, "pressure_correction": 3},
        "simplec": {"momentum": 3, "pressure_correction": 3},
        "simpler": {"absolute_pressure": 3, "momentum": 3, "pressure_correction": 3},
    }
    for row in rows:
        assert row["linear_solve_counts"] == {
            "total": sum(expected_by_coupling[row["coupling"]].values()),
            "by_kind": expected_by_coupling[row["coupling"]],
        }
        assert row["linear_solves_per_iteration"] == len(
            expected_by_coupling[row["coupling"]]
        )
        assert row["normalized_metrics"]["linear_solve_counts"] == row[
            "linear_solve_counts"
        ]


def test_compare_simple_family_validates_simpler_against_simple_and_simplec_cases():
    case_names = (
        "versteeg_6_2",
        "linear_nozzle_1d",
        "smooth_linear_nozzle_1d",
        "strong_contraction_1d",
    )
    rows = compare_simple_family(case_names=case_names, max_iterations=20)

    assert len(rows) == len(case_names) * 3
    assert {
        (row["case_name"], row["name"])
        for row in rows
    } == {
        (case_name, method)
        for case_name in case_names
        for method in ("simple", "simplec", "simpler")
    }

    for row in rows:
        assert row["final_residual"] == row["residual"]
        assert np.isfinite(row["continuity_residual"])
        assert np.isfinite(row["momentum_residual"])
        assert row["final_pressure"].shape == row["pressure_positions"].shape
        assert row["final_velocity"].shape == row["velocity_positions"].shape
        assert row["final_mass_flow"].shape == row["final_velocity"].shape
        assert row["computational_cost"] == row["linear_solves_total"]
        assert row["normalized_metrics"]["computational_cost"] == row["computational_cost"]
        assert row["normalized_metrics"]["failure_reason"] == row["failure_reason"]
        assert sum(row["linear_solve_counts"]["by_kind"].values()) == row[
            "linear_solves_total"
        ]
        if row["converged"]:
            assert row["failure_reason"] == ""
            assert row["iterations_to_tolerance"] == row["iterations"]
        else:
            assert "did not converge" in row["failure_reason"]
            assert row["iterations_to_tolerance"] is None

    simpler_rows = [row for row in rows if row["name"] == "simpler"]
    assert all(row["converged"] is True for row in simpler_rows)
    assert all(
        row["linear_solve_counts"]["by_kind"] == {
            "absolute_pressure": row["iterations"],
            "momentum": row["iterations"],
            "pressure_correction": row["iterations"],
        }
        for row in simpler_rows
    )


def test_compare_registered_cases_can_raise_numerical_failures():
    with pytest.raises(ZeroDivisionError):
        compare_registered_cases(
            case_names=["constant_area_1d"],
            schemes=["central_difference"],
            couplings=["simple"],
            max_iterations=5,
            record_errors=False,
        )


def test_compare_case_variants_rejects_empty_configurations():
    with pytest.raises(ValueError, match="solver_configurations"):
        compare_case_variants("versteeg_6_2", [])


def test_compare_case_variants_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        compare_case_variants(
            "versteeg_6_2",
            [
                {"name": "same", "scheme": Upwind()},
                {"name": "same", "scheme": CentralDifference()},
            ],
        )


def test_compare_case_variants_rejects_unknown_case_with_clear_error():
    with pytest.raises(ValueError, match="unknown case 'missing'"):
        compare_case_variants("missing", [{"name": "upwind", "scheme": Upwind()}])


def test_compare_registered_cases_rejects_unknown_scheme_with_clear_error():
    with pytest.raises(ValueError, match="unknown convection scheme"):
        compare_registered_cases(
            case_names=["versteeg_6_2"],
            schemes=["missing"],
            couplings=["simple"],
        )
