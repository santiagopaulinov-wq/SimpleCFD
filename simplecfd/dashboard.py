from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from simplecfd.cases import SolverCase, build_case_by_name
from simplecfd.telemetry import SimpleTelemetry, collect_simple_history


DASHBOARD_CASES = (
    "versteeg_6_2",
    "linear_nozzle_1d",
    "smooth_linear_nozzle_1d",
    "strong_contraction_1d",
)
DASHBOARD_METHODS = ("simple", "simplec", "simpler")
NOZZLE_CASES = tuple(case for case in DASHBOARD_CASES if case != "versteeg_6_2")
OUTPUT_DIR = Path("outputs") / "convergence_dashboard"


def run_dashboard_case(
    case_name: str,
    method: str,
    configuration: dict[str, Any],
) -> tuple[SolverCase, SimpleTelemetry, dict[str, Any]]:
    """Run one dashboard case and return the case, full history, and summary."""
    case = build_case_by_name(case_name, coupling=method, **configuration)
    history = collect_simple_history(case)
    summary = summarize_history(history, case)
    return case, history, summary


def summarize_history(history: SimpleTelemetry, case: SolverCase) -> dict[str, Any]:
    final = history.final_snapshot()
    return {
        "converged": bool(final["residual_norm"] < case.solver.tolerance),
        "iterations": int(history.final_iteration),
        "final_residual": float(final["residual_norm"]),
        "continuity_residual": float(final["continuity_norm"]),
        "momentum_residual": float(final["momentum_norm"]),
    }


def create_dashboard_figures(history: SimpleTelemetry, tolerance: float) -> dict[str, Any]:
    import plotly.graph_objects as go

    return {
        "velocity": _node_figure(
            go,
            history.iterations,
            history.velocity,
            "Velocidad nodal vs iteracion",
            "Velocidad",
            "u",
        ),
        "pressure": _node_figure(
            go,
            history.iterations,
            history.pressure,
            "Presion nodal vs iteracion",
            "Presion",
            "p",
        ),
        "momentum_residual": _residual_figure(
            go,
            history.iterations,
            history.momentum_residual,
            history.momentum_norm,
            tolerance,
            "Residual de momentum vs iteracion",
            "Rmom",
        ),
        "continuity_residual": _residual_figure(
            go,
            history.iterations,
            history.continuity_residual,
            history.continuity_norm,
            tolerance,
            "Residual de continuidad vs iteracion",
            "Rcont",
        ),
        "global_residual": _scalar_figure(
            go,
            history.iterations,
            history.residual_norm,
            tolerance,
            "Residual global vs iteracion",
            "Residual global",
        ),
    }


def export_dashboard_artifacts(
    output_dir: str | Path,
    case_name: str,
    method: str,
    case: SolverCase,
    history: SimpleTelemetry,
    summary: dict[str, Any],
    figures: dict[str, Any],
) -> dict[str, Path | dict[str, str]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = f"{case_name}_{method}".lower()

    paths: dict[str, Path | dict[str, str]] = {
        "summary_csv": output_path / f"{prefix}_summary.csv",
        "history_csv": output_path / f"{prefix}_history.csv",
        "html": {},
        "png": {},
    }
    _write_summary_csv(paths["summary_csv"], case_name, method, summary)
    _write_history_csv(paths["history_csv"], case, history)

    html_paths: dict[str, str] = {}
    png_paths: dict[str, str] = {}
    for name, figure in figures.items():
        html_path = output_path / f"{prefix}_{name}.html"
        png_path = output_path / f"{prefix}_{name}.png"
        figure.write_html(str(html_path), include_plotlyjs="cdn")
        html_paths[name] = str(html_path)
        _write_fallback_png(png_path, name, history, case.solver.tolerance)
        png_paths[name] = str(png_path)

    paths["html"] = html_paths
    paths["png"] = png_paths
    return paths


def _write_png(figure: Any, png_path: Path, output_path: Path) -> None:
    temp_root = output_path / ".tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    previous_tempdir = tempfile.tempdir
    previous_env = {name: os.environ.get(name) for name in ("TMP", "TEMP", "TMPDIR")}
    try:
        tempfile.tempdir = str(temp_root)
        for name in previous_env:
            os.environ[name] = str(temp_root)
        figure.write_image(str(png_path))
    finally:
        tempfile.tempdir = previous_tempdir
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _write_fallback_png(
    path: Path,
    figure_name: str,
    history: SimpleTelemetry,
    tolerance: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.8), dpi=160)
    if figure_name == "velocity":
        _plot_node_series(axis, history.iterations, history.velocity, "u")
        axis.set_ylabel("Velocidad")
        axis.set_title("Velocidad nodal vs iteracion")
    elif figure_name == "pressure":
        _plot_node_series(axis, history.iterations, history.pressure, "p")
        axis.set_ylabel("Presion")
        axis.set_title("Presion nodal vs iteracion")
    elif figure_name == "momentum_residual":
        _plot_residual_series(
            axis,
            history.iterations,
            history.momentum_residual,
            history.momentum_norm,
            tolerance,
            "Rmom",
        )
        axis.set_title("Residual de momentum vs iteracion")
    elif figure_name == "continuity_residual":
        _plot_residual_series(
            axis,
            history.iterations,
            history.continuity_residual,
            history.continuity_norm,
            tolerance,
            "Rcont",
        )
        axis.set_title("Residual de continuidad vs iteracion")
    elif figure_name == "global_residual":
        axis.semilogy(
            history.iterations,
            _positive_for_log(history.residual_norm),
            marker="o",
            linewidth=1.8,
            label="Residual global",
        )
        axis.axhline(tolerance, color="gray", linestyle=":", label="tolerance")
        axis.set_ylabel("Residual absoluto")
        axis.set_title("Residual global vs iteracion")
    else:
        raise ValueError(f"unknown dashboard figure '{figure_name}'")

    axis.set_xlabel("Iteracion")
    axis.grid(True, which="both", linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _plot_node_series(
    axis: Any,
    iterations: np.ndarray,
    values: np.ndarray,
    prefix: str,
) -> None:
    for node in range(values.shape[1]):
        axis.plot(
            iterations,
            values[:, node],
            marker="o",
            linewidth=1.4,
            label=f"{prefix}{node}",
        )


def _plot_residual_series(
    axis: Any,
    iterations: np.ndarray,
    values: np.ndarray,
    norm: np.ndarray,
    tolerance: float,
    prefix: str,
) -> None:
    positive_values = _positive_for_log(values)
    for node in range(values.shape[1]):
        axis.semilogy(
            iterations,
            positive_values[:, node],
            marker="o",
            linewidth=1.2,
            label=f"{prefix}{node}",
        )
    axis.semilogy(
        iterations,
        _positive_for_log(norm),
        color="black",
        linewidth=1.8,
        label=f"||{prefix}||inf",
    )
    axis.axhline(tolerance, color="gray", linestyle=":", label="tolerance")
    axis.set_ylabel("Residual absoluto")


def _node_figure(
    go: Any,
    iterations: np.ndarray,
    values: np.ndarray,
    title: str,
    yaxis_title: str,
    prefix: str,
) -> Any:
    figure = go.Figure()
    for node in range(values.shape[1]):
        figure.add_trace(
            go.Scatter(
                x=iterations,
                y=values[:, node],
                mode="lines+markers",
                name=f"{prefix}{node}",
            )
        )
    return _style_figure(figure, title, yaxis_title)


def _residual_figure(
    go: Any,
    iterations: np.ndarray,
    values: np.ndarray,
    norm: np.ndarray,
    tolerance: float,
    title: str,
    prefix: str,
) -> Any:
    figure = go.Figure()
    positive_values = _positive_for_log(values)
    for node in range(values.shape[1]):
        figure.add_trace(
            go.Scatter(
                x=iterations,
                y=positive_values[:, node],
                mode="lines+markers",
                name=f"{prefix}{node}",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=iterations,
            y=_positive_for_log(norm),
            mode="lines",
            name=f"||{prefix}||inf",
            line={"color": "black", "width": 2.4},
        )
    )
    _add_tolerance_line(go, figure, iterations, tolerance)
    return _style_figure(figure, title, "Residual absoluto", log_y=True)


def _scalar_figure(
    go: Any,
    iterations: np.ndarray,
    values: np.ndarray,
    tolerance: float,
    title: str,
    name: str,
) -> Any:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=iterations,
            y=_positive_for_log(values),
            mode="lines+markers",
            name=name,
            line={"color": "#1f2937", "width": 2.4},
        )
    )
    _add_tolerance_line(go, figure, iterations, tolerance)
    return _style_figure(figure, title, "Residual absoluto", log_y=True)


def _style_figure(figure: Any, title: str, yaxis_title: str, log_y: bool = False) -> Any:
    figure.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Iteracion",
        yaxis_title=yaxis_title,
        legend_title_text="Serie",
        margin={"l": 48, "r": 20, "t": 58, "b": 44},
        height=420,
    )
    if log_y:
        figure.update_yaxes(type="log")
    return figure


def _add_tolerance_line(
    go: Any,
    figure: Any,
    iterations: np.ndarray,
    tolerance: float,
) -> None:
    figure.add_trace(
        go.Scatter(
            x=[int(iterations[0]), int(iterations[-1])],
            y=[tolerance, tolerance],
            mode="lines",
            name="tolerance",
            line={"color": "#6b7280", "dash": "dot"},
        )
    )


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(values), 1e-15)


def _write_summary_csv(
    path: Path,
    case_name: str,
    method: str,
    summary: dict[str, Any],
) -> None:
    columns = (
        "case_name",
        "method",
        "converged",
        "iterations",
        "final_residual",
        "continuity_residual",
        "momentum_residual",
    )
    row = {"case_name": case_name, "method": method, **summary}
    _write_csv(path, columns, [row])


def _write_history_csv(path: Path, case: SolverCase, history: SimpleTelemetry) -> None:
    rows = []
    for row_index, iteration in enumerate(history.iterations):
        for node, value in enumerate(history.velocity[row_index]):
            rows.append(
                {
                    "iteration": int(iteration),
                    "variable": "velocity",
                    "node_index": node,
                    "position": float(case.geometry.velocity_positions[node]),
                    "value": float(value),
                }
            )
        for node, value in enumerate(history.pressure[row_index]):
            rows.append(
                {
                    "iteration": int(iteration),
                    "variable": "pressure",
                    "node_index": node,
                    "position": float(case.geometry.pressure_positions[node]),
                    "value": float(value),
                }
            )
        for node, value in enumerate(history.momentum_residual[row_index]):
            rows.append(
                {
                    "iteration": int(iteration),
                    "variable": "momentum_residual",
                    "node_index": node,
                    "position": float(case.geometry.velocity_positions[node]),
                    "value": float(value),
                }
            )
        for node, value in enumerate(history.continuity_residual[row_index]):
            rows.append(
                {
                    "iteration": int(iteration),
                    "variable": "continuity_residual",
                    "node_index": node,
                    "position": float(case.geometry.pressure_positions[node]),
                    "value": float(value),
                }
            )
        rows.extend(
            [
                {
                    "iteration": int(iteration),
                    "variable": "momentum_residual_norm",
                    "node_index": "",
                    "position": "",
                    "value": float(history.momentum_norm[row_index]),
                },
                {
                    "iteration": int(iteration),
                    "variable": "continuity_residual_norm",
                    "node_index": "",
                    "position": "",
                    "value": float(history.continuity_norm[row_index]),
                },
                {
                    "iteration": int(iteration),
                    "variable": "global_residual",
                    "node_index": "",
                    "position": "",
                    "value": float(history.residual_norm[row_index]),
                },
            ]
        )
    _write_csv(
        path,
        ("iteration", "variable", "node_index", "position", "value"),
        rows,
    )


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
