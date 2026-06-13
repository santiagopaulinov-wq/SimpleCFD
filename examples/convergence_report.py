from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simplecfd.cases import build_case_by_name
from simplecfd.telemetry import SimpleTelemetry, collect_simple_history


CASES = (
    "versteeg_6_2",
    "linear_nozzle_1d",
    "smooth_linear_nozzle_1d",
    "strong_contraction_1d",
)
METHODS = ("simple", "simplec", "simpler")
OUTPUT_DIR = Path("outputs") / "convergence_report"


def method_label(method: str) -> str:
    return {
        "simple": "SIMPLE",
        "simplec": "SIMPLEC",
        "simpler": "SIMPLER",
    }[method]


def positive_for_log(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(values), 1e-300)


def finish_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_node_history(
    *,
    history: SimpleTelemetry,
    values: np.ndarray,
    path: Path,
    title: str,
    ylabel: str,
    label_prefix: str,
    log_y: bool = False,
    norm_values: np.ndarray | None = None,
    tolerance: float | None = None,
) -> Path:
    plt.figure(figsize=(7.0, 4.4))
    for node_index in range(values.shape[1]):
        series = values[:, node_index]
        if log_y:
            series = positive_for_log(series)
            plt.semilogy(
                history.iterations,
                series,
                marker="o",
                linewidth=1.3,
                label=f"{label_prefix}{node_index}",
            )
        else:
            plt.plot(
                history.iterations,
                series,
                marker="o",
                linewidth=1.5,
                label=f"{label_prefix}{node_index}",
            )

    if norm_values is not None:
        if log_y:
            plt.semilogy(
                history.iterations,
                positive_for_log(norm_values),
                color="black",
                linewidth=2.0,
                label="inf-norm",
            )
        else:
            plt.plot(
                history.iterations,
                norm_values,
                color="black",
                linewidth=2.0,
                label="inf-norm",
            )
    if tolerance is not None:
        plt.axhline(tolerance, color="gray", linestyle=":", label="tolerance")

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.grid(True, which="both" if log_y else "major", linestyle="--", alpha=0.45)
    plt.legend()
    finish_plot(path)
    return path


def write_history_rows(
    rows: list[dict[str, object]],
    *,
    case_name: str,
    method: str,
    history: SimpleTelemetry,
) -> None:
    for row_index, iteration in enumerate(history.iterations):
        for node_index, value in enumerate(history.velocity[row_index]):
            rows.append(
                {
                    "case_name": case_name,
                    "method": method_label(method),
                    "iteration": int(iteration),
                    "variable": "velocity",
                    "node_index": node_index,
                    "value": float(value),
                }
            )
        for node_index, value in enumerate(history.pressure[row_index]):
            rows.append(
                {
                    "case_name": case_name,
                    "method": method_label(method),
                    "iteration": int(iteration),
                    "variable": "pressure",
                    "node_index": node_index,
                    "value": float(value),
                }
            )
        for node_index, value in enumerate(history.momentum_residual[row_index]):
            rows.append(
                {
                    "case_name": case_name,
                    "method": method_label(method),
                    "iteration": int(iteration),
                    "variable": "momentum_residual",
                    "node_index": node_index,
                    "value": float(value),
                }
            )
        rows.append(
            {
                "case_name": case_name,
                "method": method_label(method),
                "iteration": int(iteration),
                "variable": "momentum_residual_norm",
                "node_index": -1,
                "value": float(history.momentum_norm[row_index]),
            }
        )
        for node_index, value in enumerate(history.continuity_residual[row_index]):
            rows.append(
                {
                    "case_name": case_name,
                    "method": method_label(method),
                    "iteration": int(iteration),
                    "variable": "continuity_residual",
                    "node_index": node_index,
                    "value": float(value),
                }
            )
        rows.append(
            {
                "case_name": case_name,
                "method": method_label(method),
                "iteration": int(iteration),
                "variable": "continuity_residual_norm",
                "node_index": -1,
                "value": float(history.continuity_norm[row_index]),
            }
        )


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: list[dict[str, object]]) -> str:
    columns = (
        "case_name",
        "method",
        "converged",
        "iterations",
        "final_residual",
        "continuity_residual",
        "momentum_residual",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_report(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    plot_paths: list[Path] = []

    for case_name in CASES:
        for method in METHODS:
            case = build_case_by_name(case_name, coupling=method)
            history = collect_simple_history(case)
            method_name = method_label(method)
            prefix = f"{case_name}_{method_name}"
            tolerance = case.solver.tolerance
            final_residual = float(history.residual_norm[-1])
            continuity_residual = float(history.continuity_norm[-1])
            momentum_residual = float(history.momentum_norm[-1])

            plot_paths.extend(
                [
                    plot_node_history(
                        history=history,
                        values=history.velocity,
                        path=output_dir / f"{prefix}_velocity_convergence.png",
                        title=f"Velocity convergence - {case_name} [{method_name}]",
                        ylabel="Velocity",
                        label_prefix="u",
                    ),
                    plot_node_history(
                        history=history,
                        values=history.pressure,
                        path=output_dir / f"{prefix}_pressure_convergence.png",
                        title=f"Pressure convergence - {case_name} [{method_name}]",
                        ylabel="Pressure",
                        label_prefix="p",
                    ),
                    plot_node_history(
                        history=history,
                        values=history.momentum_residual,
                        path=output_dir / f"{prefix}_momentum_residual.png",
                        title=f"Momentum residual - {case_name} [{method_name}]",
                        ylabel="Absolute residual",
                        label_prefix="Rmom",
                        log_y=True,
                        norm_values=history.momentum_norm,
                        tolerance=tolerance,
                    ),
                    plot_node_history(
                        history=history,
                        values=history.continuity_residual,
                        path=output_dir / f"{prefix}_continuity_residual.png",
                        title=f"Continuity residual - {case_name} [{method_name}]",
                        ylabel="Absolute residual",
                        label_prefix="Rcont",
                        log_y=True,
                        norm_values=history.continuity_norm,
                        tolerance=tolerance,
                    ),
                ]
            )
            write_history_rows(
                history_rows,
                case_name=case_name,
                method=method,
                history=history,
            )
            summary_rows.append(
                {
                    "case_name": case_name,
                    "method": method_name,
                    "converged": final_residual < tolerance,
                    "iterations": int(history.final_iteration),
                    "final_residual": final_residual,
                    "continuity_residual": continuity_residual,
                    "momentum_residual": momentum_residual,
                }
            )

    history_csv = output_dir / "convergence_data.csv"
    summary_csv = output_dir / "convergence_summary.csv"
    summary_md = output_dir / "summary.md"
    write_csv(
        history_csv,
        ("case_name", "method", "iteration", "variable", "node_index", "value"),
        history_rows,
    )
    write_csv(
        summary_csv,
        (
            "case_name",
            "method",
            "converged",
            "iterations",
            "final_residual",
            "continuity_residual",
            "momentum_residual",
        ),
        summary_rows,
    )
    summary_md.write_text(markdown_table(summary_rows) + "\n", encoding="utf-8")
    return {
        "output_dir": output_dir,
        "history_csv": history_csv,
        "summary_csv": summary_csv,
        "summary_md": summary_md,
        "plot_paths": plot_paths,
        "summary_rows": summary_rows,
    }


def main() -> int:
    report = generate_report()
    print(report["output_dir"])
    print(report["history_csv"])
    print(report["summary_csv"])
    print(report["summary_md"])
    print(markdown_table(report["summary_rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
