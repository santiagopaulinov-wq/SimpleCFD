from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROUND_OFF_ERROR_FLOOR = 1e-12


@dataclass(frozen=True)
class ErrorNorms:
    l1: float
    l2: float
    linf: float


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finish_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    widths = np.diff(coordinates)
    averages = 0.5 * (values[:-1] + values[1:])
    return float(np.sum(widths * averages))


def l2_error(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error**2)))


def linf_error(error: np.ndarray) -> float:
    return float(np.max(np.abs(error)))


def error_norms(error: np.ndarray) -> ErrorNorms:
    absolute_error = np.abs(np.asarray(error, dtype=float))
    return ErrorNorms(
        l1=float(np.mean(absolute_error)),
        l2=float(np.sqrt(np.mean(absolute_error**2))),
        linf=float(np.max(absolute_error)),
    )


def observed_order(
    previous_error: float,
    current_error: float,
    previous_spacing: float,
    current_spacing: float,
    *,
    error_floor: float = ROUND_OFF_ERROR_FLOOR,
) -> float | str:
    if previous_error <= error_floor or current_error <= error_floor:
        return ""
    return float(
        np.log(previous_error / current_error)
        / np.log(previous_spacing / current_spacing)
    )


def observed_order_rows(
    rows: list[dict[str, Any]],
    *,
    error_columns: tuple[str, ...],
    spacing_column: str = "dy",
) -> list[dict[str, Any]]:
    enriched = []
    previous: dict[str, Any] | None = None
    for row in rows:
        enriched_row = dict(row)
        if previous is None:
            for column in error_columns:
                enriched_row[f"observed_{column}_order"] = ""
        else:
            for column in error_columns:
                enriched_row[f"observed_{column}_order"] = observed_order(
                    float(previous[column]),
                    float(row[column]),
                    float(previous[spacing_column]),
                    float(row[spacing_column]),
                )
        enriched.append(enriched_row)
        previous = row
    return enriched


def plot_error_convergence(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    error_columns: tuple[str, ...],
    labels: tuple[str, ...],
    title: str,
    spacing_column: str = "dy",
) -> None:
    spacing = np.asarray([row[spacing_column] for row in rows], dtype=float)
    plt.figure(figsize=(6.4, 4.2))
    for column, label in zip(error_columns, labels):
        errors = np.asarray([row[column] for row in rows], dtype=float)
        plt.loglog(spacing, np.maximum(errors, ROUND_OFF_ERROR_FLOOR), "o-", label=label)
    plt.gca().invert_xaxis()
    plt.xlabel("Grid spacing")
    plt.ylabel("Error norm")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.45)
    plt.legend()
    finish_plot(path)


def plot_observed_orders(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    order_columns: tuple[str, ...],
    labels: tuple[str, ...],
    title: str,
    spacing_column: str = "dy",
) -> None:
    spacing = np.asarray([row[spacing_column] for row in rows], dtype=float)
    plt.figure(figsize=(6.4, 4.2))
    plotted = False
    for column, label in zip(order_columns, labels):
        points = [
            (h, row[column])
            for h, row in zip(spacing, rows)
            if row.get(column, "") != ""
        ]
        if not points:
            continue
        plotted = True
        x = np.asarray([point[0] for point in points], dtype=float)
        y = np.asarray([point[1] for point in points], dtype=float)
        plt.plot(x, y, "o-", label=label)
    if plotted:
        plt.gca().invert_xaxis()
        plt.legend()
    else:
        plt.text(
            0.5,
            0.5,
            "Observed order is undefined\n(errors at roundoff floor)",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    plt.xlabel("Grid spacing")
    plt.ylabel("Observed order")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.45)
    finish_plot(path)


def markdown_table(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(format_markdown(row[column]) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def format_markdown(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def validate_positive(name: str, value: float) -> None:
    if not np.isscalar(value) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")


def validate_finite(name: str, value: float) -> None:
    if not np.isscalar(value) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
