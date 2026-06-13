import csv
import json
from pathlib import Path

from simplecfd.cli import main


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_cli_lists_registered_cases(capsys):
    exit_code = main(["list-cases"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "versteeg_6_2" in captured.out
    assert "linear_nozzle_1d" in captured.out


def test_cli_lists_available_methods(capsys):
    exit_code = main(["list-methods"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.splitlines() == ["simple", "simplec", "simpler"]


def test_cli_runs_combination_and_exports_artifacts(capsys):
    output_dir = Path("outputs") / "test_cli_run"

    exit_code = main(
        [
            "run",
            "--case",
            "versteeg_6_2",
            "--method",
            "simple",
            "--scheme",
            "upwind",
            "--max-iterations",
            "3",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(output_dir / "summary.csv") in captured.out
    assert str(output_dir / "result.json") in captured.out

    expected_files = {
        "summary.csv",
        "residual_history.csv",
        "pressure_profile.csv",
        "velocity_profile.csv",
        "mass_flow_profile.csv",
        "result.json",
    }
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})
    assert (output_dir / "plots").is_dir()

    summary = _read_csv(output_dir / "summary.csv")
    assert len(summary) == 1
    assert summary[0]["case_name"] == "versteeg_6_2"
    assert summary[0]["method"] == "simple"
    assert summary[0]["scheme_name"] == "upwind"
    assert summary[0]["iterations"] == "3"

    residual_rows = _read_csv(output_dir / "residual_history.csv")
    pressure_rows = _read_csv(output_dir / "pressure_profile.csv")
    velocity_rows = _read_csv(output_dir / "velocity_profile.csv")
    mass_flow_rows = _read_csv(output_dir / "mass_flow_profile.csv")
    assert len(residual_rows) == 3
    assert len(pressure_rows) == 5
    assert len(velocity_rows) == 4
    assert len(mass_flow_rows) == 4

    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["case_name"] == "versteeg_6_2"
    assert result["coupling"] == "simple"
    assert result["scheme_name"] == "upwind"
    assert len(result["final_pressure"]) == 5
    assert len(result["final_velocity"]) == 4


def test_cli_generates_poiseuille_benchmark_artifacts(capsys):
    output_dir = Path("outputs") / "test_cli_poiseuille"

    exit_code = main(
        [
            "poiseuille",
            "--output-dir",
            str(output_dir),
            "--n-nodes",
            "17",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(output_dir / "poiseuille_summary.md") in captured.out
    assert str(output_dir / "poiseuille_profile.png") in captured.out
    assert str(output_dir / "poiseuille_flow_convergence.png") in captured.out
    assert (output_dir / "poiseuille_profile.csv").exists()
    assert (output_dir / "poiseuille_convergence.csv").exists()


def test_cli_generates_couette_benchmark_artifacts(capsys):
    output_dir = Path("outputs") / "test_cli_couette"

    exit_code = main(
        [
            "couette",
            "--output-dir",
            str(output_dir),
            "--n-nodes",
            "17",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(output_dir / "couette_summary.md") in captured.out
    assert str(output_dir / "couette_profile.png") in captured.out
    assert str(output_dir / "couette_energy_convergence.png") in captured.out
    assert (output_dir / "couette_profile.csv").exists()
    assert (output_dir / "couette_convergence.csv").exists()


def test_cli_runs_analytic_verification_report(capsys):
    output_dir = Path("outputs") / "test_cli_verify_analytic"

    exit_code = main(["verify-analytic", "--output-dir", str(output_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(output_dir / "analytic_verification_summary.md") in captured.out
    assert str(output_dir / "analytic_verification_summary.csv") in captured.out
    assert (output_dir / "poiseuille" / "poiseuille_convergence.csv").exists()
    assert (output_dir / "couette" / "couette_convergence.csv").exists()
