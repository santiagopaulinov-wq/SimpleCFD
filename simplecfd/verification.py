from __future__ import annotations

from pathlib import Path
from typing import Any

from simplecfd.analytic_benchmarks import markdown_table, write_csv
from simplecfd.couette import generate_couette_benchmark
from simplecfd.poiseuille import generate_poiseuille_benchmark


def generate_analytic_verification_report(output_dir: str | Path) -> dict[str, Any]:
    """Run all built-in analytic benchmarks and summarize their verification data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    poiseuille = generate_poiseuille_benchmark(output_path / "poiseuille")
    couette = generate_couette_benchmark(output_path / "couette")

    summary_rows = [
        _poiseuille_summary_row(poiseuille),
        _couette_summary_row(couette),
    ]
    paths = {
        "summary_markdown": output_path / "analytic_verification_summary.md",
        "summary_csv": output_path / "analytic_verification_summary.csv",
        "poiseuille_dir": output_path / "poiseuille",
        "couette_dir": output_path / "couette",
    }

    write_csv(
        paths["summary_csv"],
        (
            "benchmark",
            "n_nodes",
            "l1_error",
            "l2_error",
            "linf_error",
            "integral_metric",
            "integral_relative_error",
            "finest_observed_integral_order",
            "profile_order_status",
        ),
        summary_rows,
    )
    paths["summary_markdown"].write_text(
        _summary_markdown(summary_rows, paths),
        encoding="utf-8",
    )

    return {
        "poiseuille": poiseuille,
        "couette": couette,
        "summary_rows": summary_rows,
        "paths": paths,
    }


def _poiseuille_summary_row(report: dict[str, Any]) -> dict[str, Any]:
    result = report["result"]
    rows = report["convergence_rows"]
    return {
        "benchmark": "poiseuille",
        "n_nodes": result.problem.n_nodes,
        "l1_error": result.l1_error,
        "l2_error": result.l2_error,
        "linf_error": result.linf_error,
        "integral_metric": "flow_rate",
        "integral_relative_error": result.flow_rate_relative_error,
        "finest_observed_integral_order": rows[-1]["observed_flow_order"],
        "profile_order_status": _profile_order_status(rows),
    }


def _couette_summary_row(report: dict[str, Any]) -> dict[str, Any]:
    result = report["result"]
    rows = report["convergence_rows"]
    return {
        "benchmark": "couette",
        "n_nodes": result.problem.n_nodes,
        "l1_error": result.l1_error,
        "l2_error": result.l2_error,
        "linf_error": result.linf_error,
        "integral_metric": "kinetic_energy",
        "integral_relative_error": result.kinetic_energy_relative_error,
        "finest_observed_integral_order": rows[-1]["observed_energy_order"],
        "profile_order_status": _profile_order_status(rows),
    }


def _profile_order_status(rows: list[dict[str, Any]]) -> str:
    order_columns = (
        "observed_l1_error_order",
        "observed_l2_error_order",
        "observed_linf_error_order",
    )
    if any(row[column] != "" for row in rows for column in order_columns):
        return "measured"
    return "not_applicable_roundoff_profile_error"


def _summary_markdown(summary_rows: list[dict[str, Any]], paths: dict[str, Path]) -> str:
    return "\n".join(
        [
            "# SimpleCFD Analytic Verification",
            "",
            "## Summary",
            "",
            markdown_table(
                summary_rows,
                (
                    "benchmark",
                    "n_nodes",
                    "l1_error",
                    "l2_error",
                    "linf_error",
                    "integral_metric",
                    "integral_relative_error",
                    "finest_observed_integral_order",
                    "profile_order_status",
                ),
            ),
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{paths['summary_csv'].name}`",
            f"- Poiseuille artifacts: `{paths['poiseuille_dir'].name}/`",
            f"- Couette artifacts: `{paths['couette_dir'].name}/`",
            "",
            "Profile order is reported as not applicable when nodal profile errors are at the roundoff floor.",
            "",
        ]
    )
