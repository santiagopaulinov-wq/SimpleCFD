from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from simplecfd.cases import COUPLING_STRATEGIES, list_available_cases
from simplecfd.comparison import SCHEME_FACTORIES, compare_registered_cases
from simplecfd.couette import CouetteProblem, generate_couette_benchmark
from simplecfd.poiseuille import PoiseuilleProblem, generate_poiseuille_benchmark
from simplecfd.reports import generate_comparison_plots
from simplecfd.verification import generate_analytic_verification_report


SUMMARY_COLUMNS = (
    "case_name",
    "method",
    "scheme_name",
    "converged",
    "iterations",
    "iterations_to_tolerance",
    "initial_residual",
    "final_residual",
    "continuity_residual",
    "momentum_residual",
    "numerically_stable",
    "final_mass_flow_mean",
    "computational_cost",
    "cost_relative",
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simplecfd",
        description="Minimal SimpleCFD command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_cases = subparsers.add_parser("list-cases", help="List registered cases.")
    list_cases.set_defaults(handler=_handle_list_cases)

    list_methods = subparsers.add_parser("list-methods", help="List available methods.")
    list_methods.set_defaults(handler=_handle_list_methods)

    run = subparsers.add_parser(
        "run",
        help="Run one case/method/scheme combination and export artifacts.",
    )
    run.add_argument("--case", required=True, help="Registered case name.")
    run.add_argument(
        "--method",
        default="simple",
        choices=sorted(COUPLING_STRATEGIES),
        help="Pressure-velocity coupling method.",
    )
    run.add_argument(
        "--scheme",
        default="upwind",
        choices=sorted(SCHEME_FACTORIES),
        help="Convection scheme.",
    )
    run.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where result artifacts will be written.",
    )
    run.add_argument("--max-iterations", type=int, default=None)
    run.add_argument("--tolerance", type=float, default=None)
    run.add_argument("--pressure-relaxation", type=float, default=None)
    run.add_argument("--velocity-relaxation", type=float, default=None)
    run.set_defaults(handler=_handle_run)

    poiseuille = subparsers.add_parser(
        "poiseuille",
        help="Generate the plane Poiseuille analytic benchmark artifacts.",
    )
    poiseuille.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where benchmark artifacts will be written.",
    )
    poiseuille.add_argument("--channel-height", type=float, default=1.0)
    poiseuille.add_argument("--length", type=float, default=1.0)
    poiseuille.add_argument("--dynamic-viscosity", type=float, default=1.0)
    poiseuille.add_argument("--pressure-drop", type=float, default=8.0)
    poiseuille.add_argument("--n-nodes", type=int, default=33)
    poiseuille.set_defaults(handler=_handle_poiseuille)

    couette = subparsers.add_parser(
        "couette",
        help="Generate the plane Couette analytic benchmark artifacts.",
    )
    couette.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where benchmark artifacts will be written.",
    )
    couette.add_argument("--channel-height", type=float, default=1.0)
    couette.add_argument("--lower-wall-velocity", type=float, default=0.0)
    couette.add_argument("--upper-wall-velocity", type=float, default=1.0)
    couette.add_argument("--dynamic-viscosity", type=float, default=1.0)
    couette.add_argument("--n-nodes", type=int, default=33)
    couette.set_defaults(handler=_handle_couette)

    verify_analytic = subparsers.add_parser(
        "verify-analytic",
        help="Run all built-in analytic benchmarks and summarize verification artifacts.",
    )
    verify_analytic.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where verification artifacts will be written.",
    )
    verify_analytic.set_defaults(handler=_handle_verify_analytic)

    return parser


def _handle_list_cases(args: argparse.Namespace) -> int:
    for case_name in list_available_cases():
        print(case_name)
    return 0


def _handle_list_methods(args: argparse.Namespace) -> int:
    for method_name in sorted(COUPLING_STRATEGIES):
        print(method_name)
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    configuration = _run_configuration(args)
    rows = compare_registered_cases(
        case_names=[args.case],
        schemes=[args.scheme],
        couplings=[args.method],
        record_errors=False,
        **configuration,
    )
    row = rows[0]
    paths = export_run_artifacts(row, args.output_dir)

    print(paths["summary_csv"])
    print(paths["result_json"])
    return 0


def _handle_poiseuille(args: argparse.Namespace) -> int:
    report = generate_poiseuille_benchmark(
        args.output_dir,
        problem=PoiseuilleProblem(
            channel_height=args.channel_height,
            length=args.length,
            dynamic_viscosity=args.dynamic_viscosity,
            pressure_drop=args.pressure_drop,
            n_nodes=args.n_nodes,
        ),
    )

    print(report["paths"]["summary_markdown"])
    print(report["paths"]["profile_png"])
    print(report["paths"]["convergence_png"])
    return 0


def _handle_couette(args: argparse.Namespace) -> int:
    report = generate_couette_benchmark(
        args.output_dir,
        problem=CouetteProblem(
            channel_height=args.channel_height,
            lower_wall_velocity=args.lower_wall_velocity,
            upper_wall_velocity=args.upper_wall_velocity,
            dynamic_viscosity=args.dynamic_viscosity,
            n_nodes=args.n_nodes,
        ),
    )

    print(report["paths"]["summary_markdown"])
    print(report["paths"]["profile_png"])
    print(report["paths"]["convergence_png"])
    return 0


def _handle_verify_analytic(args: argparse.Namespace) -> int:
    report = generate_analytic_verification_report(args.output_dir)

    print(report["paths"]["summary_markdown"])
    print(report["paths"]["summary_csv"])
    return 0


def _run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    configuration = {}
    optional_fields = (
        "max_iterations",
        "tolerance",
        "pressure_relaxation",
        "velocity_relaxation",
    )
    for field_name in optional_fields:
        value = getattr(args, field_name)
        if value is not None:
            configuration[field_name] = value
    return configuration


def export_run_artifacts(row: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary_csv": output_path / "summary.csv",
        "residual_history_csv": output_path / "residual_history.csv",
        "pressure_profile_csv": output_path / "pressure_profile.csv",
        "velocity_profile_csv": output_path / "velocity_profile.csv",
        "mass_flow_profile_csv": output_path / "mass_flow_profile.csv",
        "result_json": output_path / "result.json",
        "plots_dir": output_path / "plots",
    }

    _write_csv(paths["summary_csv"], SUMMARY_COLUMNS, [_summary_row(row)])
    _write_csv(
        paths["residual_history_csv"],
        ("iteration", "residual"),
        _residual_rows(row),
    )
    _write_csv(
        paths["pressure_profile_csv"],
        ("node_index", "position", "pressure"),
        _profile_rows(row, positions_key="pressure_positions", values_key="final_pressure"),
    )
    _write_csv(
        paths["velocity_profile_csv"],
        ("node_index", "position", "velocity"),
        _profile_rows(row, positions_key="velocity_positions", values_key="final_velocity"),
    )
    _write_csv(
        paths["mass_flow_profile_csv"],
        ("node_index", "position", "mass_flow"),
        _profile_rows(row, positions_key="velocity_positions", values_key="final_mass_flow"),
    )
    paths["result_json"].write_text(
        json.dumps(_json_safe(row), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    generate_comparison_plots([row], paths["plots_dir"])
    return paths


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_name": row["case_name"],
        "method": row.get("coupling", ""),
        "scheme_name": row.get("scheme_name", ""),
        "converged": row["converged"],
        "iterations": row["iterations"],
        "iterations_to_tolerance": row["iterations_to_tolerance"],
        "initial_residual": row["initial_residual"],
        "final_residual": row["final_residual"],
        "continuity_residual": row["continuity_residual"],
        "momentum_residual": row["momentum_residual"],
        "numerically_stable": row["numerically_stable"],
        "final_mass_flow_mean": row["final_mass_flow_mean"],
        "computational_cost": row["computational_cost"],
        "cost_relative": row["cost_relative"],
    }


def _residual_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"iteration": iteration, "residual": residual}
        for iteration, residual in enumerate(row["residual_history"], start=1)
    ]


def _profile_rows(
    row: dict[str, Any],
    *,
    positions_key: str,
    values_key: str,
) -> list[dict[str, Any]]:
    positions = np.asarray(row[positions_key], dtype=float)
    values = np.asarray(row[values_key], dtype=float)
    value_column = {
        "final_pressure": "pressure",
        "final_velocity": "velocity",
        "final_mass_flow": "mass_flow",
    }[values_key]
    return [
        {
            "node_index": index,
            "position": positions[index],
            value_column: value,
        }
        for index, value in enumerate(values)
    ]


def _write_csv(path: Path, columns, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in columns})


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(_csv_value(value))
    if not isinstance(value, (str, int, float, bool, type(None))):
        return value.__class__.__name__
    return value


if __name__ == "__main__":
    raise SystemExit(main())
