from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from simplecfd.cases import list_available_cases
from simplecfd.comparison import compare_registered_cases


DEFAULT_REPORT_METHODS = ("simple", "simplec", "simpler", "piso")
DEFAULT_REPORT_SCHEMES = ("upwind",)

SUMMARY_COLUMNS = (
    "case_name",
    "method",
    "scheme_name",
    "converged",
    "iterations",
    "iterations_to_tolerance",
    "initial_residual",
    "final_residual",
    "residual_reduction_factor",
    "residual_reduction_per_iteration",
    "continuity_residual",
    "momentum_residual",
    "numerically_stable",
    "final_mass_flow_mean",
    "benchmark_variant",
    "benchmark_passed",
    "benchmark_error",
    "failure_reason",
    "computational_cost",
    "linear_solves_total",
    "linear_solves_per_iteration",
    "momentum_linear_solves",
    "absolute_pressure_linear_solves",
    "pressure_correction_linear_solves",
    "other_linear_solves",
    "cost_relative",
    "error",
)


def generate_method_comparison_report(
    output_dir: str | Path,
    *,
    case_names=None,
    methods=DEFAULT_REPORT_METHODS,
    schemes=DEFAULT_REPORT_SCHEMES,
    record_errors: bool = True,
    **common_configuration: Any,
) -> dict[str, Any]:
    """Generate Markdown and CSV artifacts for method comparisons."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_cases = tuple(list_available_cases() if case_names is None else case_names)
    selected_methods = tuple(methods)
    selected_schemes = tuple(schemes)

    rows = compare_registered_cases(
        case_names=selected_cases,
        schemes=selected_schemes,
        couplings=selected_methods,
        record_errors=record_errors,
        **common_configuration,
    )
    summary_rows = [_summary_row(row) for row in rows]
    residual_rows = _residual_history_rows(rows)
    pressure_rows = _profile_rows(rows, field_name="pressure", values_key="final_pressure")
    velocity_rows = _profile_rows(rows, field_name="velocity", values_key="final_velocity")
    plot_paths = generate_comparison_plots(rows, output_path / "plots")

    paths = {
        "markdown": output_path / "method_comparison_report.md",
        "summary_csv": output_path / "method_comparison_summary.csv",
        "residual_history_csv": output_path / "method_comparison_residual_histories.csv",
        "pressure_profile_csv": output_path / "method_comparison_pressure_profiles.csv",
        "velocity_profile_csv": output_path / "method_comparison_velocity_profiles.csv",
        "plots_dir": output_path / "plots",
    }

    _write_csv(paths["summary_csv"], SUMMARY_COLUMNS, summary_rows)
    _write_csv(
        paths["residual_history_csv"],
        (
            "case_name",
            "method",
            "scheme_name",
            "iteration",
            "residual",
        ),
        residual_rows,
    )
    _write_csv(
        paths["pressure_profile_csv"],
        (
            "case_name",
            "method",
            "scheme_name",
            "node_index",
            "position",
            "field",
            "value",
        ),
        pressure_rows,
    )
    _write_csv(
        paths["velocity_profile_csv"],
        (
            "case_name",
            "method",
            "scheme_name",
            "node_index",
            "position",
            "field",
            "value",
        ),
        velocity_rows,
    )
    paths["markdown"].write_text(
        _markdown_report(
            summary_rows=summary_rows,
            selected_cases=selected_cases,
            selected_methods=selected_methods,
            selected_schemes=selected_schemes,
            paths=paths,
            plot_paths=plot_paths,
        ),
        encoding="utf-8",
    )

    return {
        "rows": rows,
        "summary_rows": summary_rows,
        "residual_history_rows": residual_rows,
        "pressure_profile_rows": pressure_rows,
        "velocity_profile_rows": velocity_rows,
        "plot_paths": plot_paths,
        "paths": paths,
    }


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    linear_solve_counts = row.get("linear_solve_counts", {"by_kind": {}})
    linear_solve_counts_by_kind = linear_solve_counts.get("by_kind", {})
    return {
        "case_name": row["case_name"],
        "method": _method_label(row.get("coupling", row.get("method", ""))),
        "scheme_name": row.get("scheme_name", ""),
        "converged": row["converged"],
        "iterations": row["iterations"],
        "iterations_to_tolerance": row["iterations_to_tolerance"],
        "initial_residual": row["initial_residual"],
        "final_residual": row["final_residual"],
        "residual_reduction_factor": row["residual_reduction_factor"],
        "residual_reduction_per_iteration": row["residual_reduction_per_iteration"],
        "continuity_residual": row["continuity_residual"],
        "momentum_residual": row["momentum_residual"],
        "numerically_stable": row["numerically_stable"],
        "final_mass_flow_mean": row["final_mass_flow_mean"],
        "benchmark_variant": row["benchmark_variant"],
        "benchmark_passed": row["benchmark_passed"],
        "benchmark_error": row["benchmark_error"],
        "failure_reason": row.get("failure_reason", ""),
        "computational_cost": row["computational_cost"],
        "linear_solves_total": row.get("linear_solves_total", ""),
        "linear_solves_per_iteration": row.get("linear_solves_per_iteration", ""),
        "momentum_linear_solves": linear_solve_counts_by_kind.get("momentum", 0),
        "absolute_pressure_linear_solves": linear_solve_counts_by_kind.get(
            "absolute_pressure",
            0,
        ),
        "pressure_correction_linear_solves": linear_solve_counts_by_kind.get(
            "pressure_correction",
            0,
        ),
        "other_linear_solves": linear_solve_counts_by_kind.get("other", 0),
        "cost_relative": row["cost_relative"],
        "error": row.get("error", ""),
    }


def _residual_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history_rows = []
    for row in rows:
        for iteration, residual in enumerate(row["residual_history"], start=1):
            history_rows.append(
                {
                    "case_name": row["case_name"],
                    "method": _method_label(row.get("coupling", "")),
                    "scheme_name": row.get("scheme_name", ""),
                    "iteration": iteration,
                    "residual": residual,
                }
            )
    return history_rows


def _profile_rows(
    rows: list[dict[str, Any]],
    *,
    field_name: str,
    values_key: str,
) -> list[dict[str, Any]]:
    profile_rows = []
    for row in rows:
        values = np.asarray(row[values_key], dtype=float)
        positions_key = f"{field_name}_positions"
        positions = np.asarray(row.get(positions_key, np.arange(values.size)), dtype=float)
        for node_index, value in enumerate(values):
            profile_rows.append(
                {
                    "case_name": row["case_name"],
                    "method": _method_label(row.get("coupling", "")),
                    "scheme_name": row.get("scheme_name", ""),
                    "node_index": node_index,
                    "position": positions[node_index] if node_index < positions.size else "",
                    "field": field_name,
                    "value": value,
                }
            )
    return profile_rows


def generate_comparison_plots(rows: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    """Create residual/profile plots for each successful comparison row."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    for row in rows:
        if row.get("error"):
            continue
        if not row.get("residual_history"):
            continue
        prefix = _plot_prefix(row)
        plot_paths.extend(
            [
                _plot_residual_history(row, output_path / f"{prefix}_residual.png"),
                _plot_profile(
                    row,
                    output_path / f"{prefix}_pressure.png",
                    positions_key="pressure_positions",
                    values_key="final_pressure",
                    title="Final pressure profile",
                    ylabel="Pressure [Pa]",
                ),
                _plot_profile(
                    row,
                    output_path / f"{prefix}_velocity.png",
                    positions_key="velocity_positions",
                    values_key="final_velocity",
                    title="Final velocity profile",
                    ylabel="Velocity [m/s]",
                ),
                _plot_profile(
                    row,
                    output_path / f"{prefix}_mass_flow.png",
                    positions_key="velocity_positions",
                    values_key="final_mass_flow",
                    title="Final mass-flow profile",
                    ylabel="Mass flow",
                ),
            ]
        )
    return plot_paths


def _plot_residual_history(row: dict[str, Any], path: Path) -> Path:
    residual_history = np.asarray(row["residual_history"], dtype=float)
    iterations = np.arange(1, residual_history.size + 1)
    plt.figure(figsize=(6.4, 4.2))
    plt.semilogy(iterations, np.maximum(np.abs(residual_history), 1e-300), marker="o")
    plt.title(_plot_title(row, "Residual history"))
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.grid(True, which="both", linestyle="--", alpha=0.45)
    _finish_plot(path)
    return path


def _plot_profile(
    row: dict[str, Any],
    path: Path,
    *,
    positions_key: str,
    values_key: str,
    title: str,
    ylabel: str,
) -> Path:
    positions = np.asarray(row[positions_key], dtype=float)
    values = np.asarray(row[values_key], dtype=float)
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(positions, values, marker="o", linewidth=1.8)
    plt.title(_plot_title(row, title))
    plt.xlabel("Position [m]")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.45)
    _finish_plot(path)
    return path


def _finish_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _write_csv(path: Path, columns, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in columns})


def _markdown_report(
    *,
    summary_rows: list[dict[str, Any]],
    selected_cases: tuple[str, ...],
    selected_methods: tuple[str, ...],
    selected_schemes: tuple[Any, ...],
    paths: dict[str, Path],
    plot_paths: list[Path],
) -> str:
    lines = [
        "# SimpleCFD Method Comparison",
        "",
        f"Cases: {', '.join(selected_cases)}",
        f"Methods: {', '.join(_method_label(method) for method in selected_methods)}",
        f"Schemes: {', '.join(str(scheme) for scheme in selected_schemes)}",
        "",
        "## Summary",
        "",
        _markdown_table(
            summary_rows,
            (
                "case_name",
                "method",
                "scheme_name",
                "converged",
                "iterations",
                "final_residual",
                "residual_reduction_per_iteration",
                "numerically_stable",
                "final_mass_flow_mean",
                "linear_solves_per_iteration",
                "benchmark_error",
                "cost_relative",
                "failure_reason",
                "error",
            ),
        ),
        "",
        "## Artifacts",
        "",
        f"- Summary table: `{paths['summary_csv'].name}`",
        f"- Residual histories: `{paths['residual_history_csv'].name}`",
        f"- Final pressure profiles: `{paths['pressure_profile_csv'].name}`",
        f"- Final velocity profiles: `{paths['velocity_profile_csv'].name}`",
        f"- Per-run plots: `{paths['plots_dir'].name}/` ({len(plot_paths)} PNG files)",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]], columns) -> str:
    if not rows:
        return "_No rows generated._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_markdown_value(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _method_label(method: Any) -> str:
    labels = {
        "simple": "SIMPLE",
        "simplec": "SIMPLEC",
        "simpler": "SIMPLER",
        "piso": "PISO",
    }
    return labels.get(str(method).strip().lower(), str(method).upper())


def _plot_prefix(row: dict[str, Any]) -> str:
    pieces = (
        row.get("case_name", "case"),
        row.get("scheme_name", "scheme"),
        _method_label(row.get("coupling", "method")).lower(),
    )
    return "_".join(_safe_filename_piece(piece) for piece in pieces)


def _plot_title(row: dict[str, Any], title: str) -> str:
    return (
        f"{title} - {row.get('case_name')} "
        f"[{_method_label(row.get('coupling', ''))}, {row.get('scheme_name', '')}]"
    )


def _safe_filename_piece(value: Any) -> str:
    text = str(value).strip().lower()
    return "".join(character if character.isalnum() else "_" for character in text).strip("_")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
    return value


def _markdown_value(value: Any) -> str:
    value = _csv_value(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate SimpleCFD method comparison reports.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=Path("outputs") / "method_comparison",
        help="Directory where Markdown and CSV report artifacts will be written.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override max_iterations for every generated run.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Override tolerance for every generated run.",
    )
    args = parser.parse_args(argv)

    configuration = {}
    if args.max_iterations is not None:
        configuration["max_iterations"] = args.max_iterations
    if args.tolerance is not None:
        configuration["tolerance"] = args.tolerance

    report = generate_method_comparison_report(args.output_dir, **configuration)
    print(report["paths"]["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
