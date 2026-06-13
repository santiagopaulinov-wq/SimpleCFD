import csv
from pathlib import Path

from simplecfd.cases import list_available_cases
from simplecfd.reports import generate_method_comparison_report


REPORT_TEST_DIR = Path("outputs") / "test_reports"


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_method_comparison_report_writes_markdown_and_csv_artifacts():
    report = generate_method_comparison_report(
        REPORT_TEST_DIR / "artifacts",
        case_names=["versteeg_6_2"],
        methods=("simple", "simplec", "simpler", "piso"),
        max_iterations=3,
    )

    paths = report["paths"]
    assert set(paths) == {
        "markdown",
        "summary_csv",
        "residual_history_csv",
        "pressure_profile_csv",
        "velocity_profile_csv",
        "plots_dir",
    }
    assert all(path.exists() for path in paths.values())
    assert paths["plots_dir"].is_dir()

    summary_rows = _read_csv(paths["summary_csv"])
    assert [row["method"] for row in summary_rows] == ["SIMPLE", "SIMPLEC", "SIMPLER", "PISO"]
    assert all(row["case_name"] == "versteeg_6_2" for row in summary_rows)
    assert {
        "initial_residual",
        "final_residual",
        "cost_relative",
        "benchmark_error",
        "failure_reason",
    }.issubset(
        summary_rows[0]
    )
    assert summary_rows[2]["error"] == ""
    assert "unknown pressure-velocity coupling 'piso'" in summary_rows[3]["error"]
    assert summary_rows[0]["linear_solves_total"] == "6"
    assert summary_rows[0]["computational_cost"] == "6.0"
    assert summary_rows[0]["linear_solves_per_iteration"] == "2.0"
    assert summary_rows[0]["momentum_linear_solves"] == "3"
    assert summary_rows[0]["pressure_correction_linear_solves"] == "3"
    assert summary_rows[0]["absolute_pressure_linear_solves"] == "0"
    assert summary_rows[1]["linear_solves_total"] == "6"
    assert summary_rows[2]["linear_solves_total"] == "9"
    assert summary_rows[2]["computational_cost"] == "9.0"
    assert summary_rows[2]["absolute_pressure_linear_solves"] == "3"
    assert summary_rows[3]["linear_solves_total"] == "0"
    assert summary_rows[3]["failure_reason"].startswith("ValueError:")

    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "# SimpleCFD Method Comparison" in markdown
    assert "SIMPLEC" in markdown
    assert "method_comparison_residual_histories.csv" in markdown
    assert "failure_reason" in markdown
    assert "Per-run plots" in markdown


def test_method_comparison_report_writes_residual_histories_and_final_profiles():
    report = generate_method_comparison_report(
        REPORT_TEST_DIR / "profiles",
        case_names=["versteeg_6_2"],
        methods=("simple",),
        max_iterations=3,
    )

    residual_rows = _read_csv(report["paths"]["residual_history_csv"])
    pressure_rows = _read_csv(report["paths"]["pressure_profile_csv"])
    velocity_rows = _read_csv(report["paths"]["velocity_profile_csv"])

    assert [int(row["iteration"]) for row in residual_rows] == [1, 2, 3]
    assert all(row["method"] == "SIMPLE" for row in residual_rows)
    assert len(pressure_rows) == 5
    assert len(velocity_rows) == 4
    assert [int(row["node_index"]) for row in pressure_rows] == [0, 1, 2, 3, 4]
    assert [int(row["node_index"]) for row in velocity_rows] == [0, 1, 2, 3]
    assert [float(row["position"]) for row in pressure_rows] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert [float(row["position"]) for row in velocity_rows] == [0.25, 0.75, 1.25, 1.75]
    assert all(row["field"] == "pressure" for row in pressure_rows)
    assert all(row["field"] == "velocity" for row in velocity_rows)
    assert len(report["plot_paths"]) == 4
    assert {path.name for path in report["plot_paths"]} == {
        "versteeg_6_2_upwind_simple_residual.png",
        "versteeg_6_2_upwind_simple_pressure.png",
        "versteeg_6_2_upwind_simple_velocity.png",
        "versteeg_6_2_upwind_simple_mass_flow.png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in report["plot_paths"])


def test_method_comparison_report_defaults_to_all_registered_cases_and_four_methods():
    report = generate_method_comparison_report(REPORT_TEST_DIR / "defaults", max_iterations=1)

    summary_rows = report["summary_rows"]
    cases = {row["case_name"] for row in summary_rows}
    methods = {row["method"] for row in summary_rows}

    assert cases == set(list_available_cases())
    assert methods == {"SIMPLE", "SIMPLEC", "SIMPLER", "PISO"}
    assert len(summary_rows) == len(list_available_cases()) * 4
