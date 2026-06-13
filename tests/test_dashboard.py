from pathlib import Path

from simplecfd.dashboard import (
    DASHBOARD_CASES,
    DASHBOARD_METHODS,
    NOZZLE_CASES,
    create_dashboard_figures,
    export_dashboard_artifacts,
    run_dashboard_case,
    summarize_history,
)
from simplecfd.telemetry import collect_simple_history


class FakeFigure:
    def __init__(self, name):
        self.name = name

    def write_html(self, path, include_plotlyjs="cdn"):
        Path(path).write_text(
            f"<html>{self.name}:{include_plotlyjs}</html>",
            encoding="utf-8",
        )

    def write_image(self, path):
        Path(path).write_bytes(f"png:{self.name}".encode("ascii"))


def test_dashboard_exposes_requested_cases_and_methods_only():
    assert DASHBOARD_CASES == (
        "versteeg_6_2",
        "linear_nozzle_1d",
        "smooth_linear_nozzle_1d",
        "strong_contraction_1d",
    )
    assert DASHBOARD_METHODS == ("simple", "simplec", "simpler")
    assert "versteeg_6_2" not in NOZZLE_CASES


def test_dashboard_summary_matches_collected_telemetry(versteeg_example_6_2_case):
    history = collect_simple_history(versteeg_example_6_2_case, max_iterations=3)

    summary = summarize_history(history, versteeg_example_6_2_case)

    assert summary["iterations"] == history.final_iteration
    assert summary["final_residual"] == history.residual_norm[-1]
    assert summary["continuity_residual"] == history.continuity_norm[-1]
    assert summary["momentum_residual"] == history.momentum_norm[-1]
    assert summary["converged"] is False


def test_dashboard_run_accepts_nozzle_parameters_and_simplec_method():
    case, history, summary = run_dashboard_case(
        "linear_nozzle_1d",
        "simplec",
        {
            "tolerance": 1e-4,
            "max_iterations": 4,
            "pressure_relaxation": 0.45,
            "velocity_relaxation": 0.45,
            "density": 1.0,
            "inlet_area": 1.0,
            "outlet_area": 0.55,
            "n_pressure": 6,
            "dx": 0.2,
            "mass_flow_guess": 0.6,
            "inlet_pressure": 5.0,
            "outlet_pressure": 1.0,
        },
    )

    assert case.geometry.n_pressure == 6
    assert history.final_iteration <= 4
    assert summary["iterations"] == history.final_iteration


def test_dashboard_export_writes_csv_html_and_png():
    output_dir = Path("outputs") / "test_dashboard_export"
    case, history, summary = run_dashboard_case(
        "versteeg_6_2",
        "simple",
        {"max_iterations": 2},
    )
    figures = {
        "velocity": FakeFigure("velocity"),
        "pressure": FakeFigure("pressure"),
        "momentum_residual": FakeFigure("momentum"),
        "continuity_residual": FakeFigure("continuity"),
        "global_residual": FakeFigure("global"),
    }

    paths = export_dashboard_artifacts(
        output_dir,
        "versteeg_6_2",
        "simple",
        case,
        history,
        summary,
        figures,
    )

    assert paths["summary_csv"].exists()
    assert paths["history_csv"].exists()
    assert "final_residual" in paths["summary_csv"].read_text(encoding="utf-8")
    assert "global_residual" in paths["history_csv"].read_text(encoding="utf-8")
    for path in paths["html"].values():
        assert Path(path).read_text(encoding="utf-8").startswith("<html>")
    for path in paths["png"].values():
        assert Path(path).read_bytes().startswith(b"\x89PNG")


def test_dashboard_plotly_figures_require_plotly_when_called():
    try:
        import plotly  # noqa: F401
    except ModuleNotFoundError:
        return

    case, history, _ = run_dashboard_case(
        "versteeg_6_2",
        "simple",
        {"max_iterations": 1},
    )
    figures = create_dashboard_figures(history, case.solver.tolerance)

    assert set(figures) == {
        "velocity",
        "pressure",
        "momentum_residual",
        "continuity_residual",
        "global_residual",
    }
