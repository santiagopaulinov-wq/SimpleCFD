from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from simplecfd.telemetry import SimpleTelemetry, collect_versteeg_convergence


PRESSURE_NODE_NAMES = ["A", "B", "C", "D", "E"]
VELOCITY_NODE_NAMES = ["1", "2", "3", "4"]
OUTPUT_DIR = Path("outputs") / "convergence"


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(values), 1e-15)


def _finish_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_velocity(history: SimpleTelemetry, output_dir: Path) -> Path:
    path = output_dir / "velocity_convergence.png"
    for node, label in enumerate(VELOCITY_NODE_NAMES):
        plt.plot(
            history.iterations,
            history.velocity[:, node],
            marker="o",
            linewidth=1.6,
            label=f"u{label}",
        )
    plt.title("Velocity convergence - Versteeg example 6.2")
    plt.xlabel("Iteration")
    plt.ylabel("Velocity [m/s]")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    _finish_plot(path)
    return path


def plot_pressure(history: SimpleTelemetry, output_dir: Path) -> Path:
    path = output_dir / "pressure_convergence.png"
    for node, label in enumerate(PRESSURE_NODE_NAMES):
        plt.plot(
            history.iterations,
            history.pressure[:, node],
            marker="o",
            linewidth=1.6,
            label=f"p{label}",
        )
    plt.title("Pressure convergence - Versteeg example 6.2")
    plt.xlabel("Iteration")
    plt.ylabel("Relative pressure [Pa]")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    _finish_plot(path)
    return path


def plot_momentum_balance(
    history: SimpleTelemetry,
    output_dir: Path,
    tolerance: float,
) -> Path:
    path = output_dir / "momentum_balance_convergence.png"
    for node, label in enumerate(VELOCITY_NODE_NAMES):
        plt.semilogy(
            history.iterations,
            _positive_for_log(history.momentum_residual[:, node]),
            marker="o",
            linewidth=1.4,
            label=f"Rmom {label}",
        )
    plt.semilogy(
        history.iterations,
        _positive_for_log(history.momentum_norm),
        color="black",
        linewidth=2.0,
        label="||Rmom||inf",
    )
    plt.axhline(tolerance, color="gray", linestyle=":", label="tolerance")
    plt.title("Momentum balance convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Absolute residual")
    plt.grid(True, which="both", linestyle="--", alpha=0.45)
    plt.legend()
    _finish_plot(path)
    return path


def plot_continuity_balance(
    history: SimpleTelemetry,
    output_dir: Path,
    tolerance: float,
) -> Path:
    path = output_dir / "continuity_balance_convergence.png"
    for node, label in enumerate(PRESSURE_NODE_NAMES):
        plt.semilogy(
            history.iterations,
            _positive_for_log(history.continuity_residual[:, node]),
            marker="o",
            linewidth=1.4,
            label=f"Rcont {label}",
        )
    plt.semilogy(
        history.iterations,
        _positive_for_log(history.continuity_norm),
        color="black",
        linewidth=2.0,
        label="||Rcont||inf",
    )
    plt.axhline(tolerance, color="gray", linestyle=":", label="tolerance")
    plt.title("Continuity balance convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Absolute residual")
    plt.grid(True, which="both", linestyle="--", alpha=0.45)
    plt.legend()
    _finish_plot(path)
    return path


def create_convergence_plots(
    output_dir: Path = OUTPUT_DIR,
    history: SimpleTelemetry | None = None,
    tolerance: float = 1e-5,
    max_iterations: int = 100,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if history is None:
        history = collect_versteeg_convergence(
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    return [
        plot_velocity(history, output_dir),
        plot_pressure(history, output_dir),
        plot_momentum_balance(history, output_dir, tolerance),
        plot_continuity_balance(history, output_dir, tolerance),
    ]


if __name__ == "__main__":
    paths = create_convergence_plots()
    for path in paths:
        print(path)
