import csv
from pathlib import Path

from simplecfd.verification import generate_analytic_verification_report


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_analytic_verification_report_runs_all_builtin_analytic_benchmarks():
    output_dir = Path("outputs") / "test_analytic_verification"

    report = generate_analytic_verification_report(output_dir)

    paths = report["paths"]
    assert paths["summary_markdown"].exists()
    assert paths["summary_csv"].exists()
    assert paths["poiseuille_dir"].is_dir()
    assert paths["couette_dir"].is_dir()
    assert (paths["poiseuille_dir"] / "poiseuille_convergence.csv").exists()
    assert (paths["couette_dir"] / "couette_convergence.csv").exists()

    rows = _read_csv(paths["summary_csv"])
    assert [row["benchmark"] for row in rows] == ["poiseuille", "couette"]
    assert all(float(row["l1_error"]) < 1e-12 for row in rows)
    assert all(float(row["l2_error"]) < 1e-12 for row in rows)
    assert all(float(row["linf_error"]) < 1e-12 for row in rows)
    assert rows[0]["integral_metric"] == "flow_rate"
    assert rows[1]["integral_metric"] == "kinetic_energy"
    assert all(row["profile_order_status"] for row in rows)

    markdown = paths["summary_markdown"].read_text(encoding="utf-8")
    assert "# SimpleCFD Analytic Verification" in markdown
    assert "not applicable" in markdown
