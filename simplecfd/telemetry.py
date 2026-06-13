from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simplecfd.assembly.momentum import MomentumAssembler
from simplecfd.assembly.pressure_correction import PressureCorrectionAssembler
from typing import Any

from simplecfd.cases import SolverCase, build_case_by_name, build_versteeg_example_6_2_case
from simplecfd.fields import Field
from simplecfd.geometry import Geometry


@dataclass
class SimpleTelemetry:
    iterations: np.ndarray
    pressure: np.ndarray
    velocity: np.ndarray
    momentum_residual: np.ndarray
    continuity_residual: np.ndarray

    def __post_init__(self) -> None:
        self.iterations = np.asarray(self.iterations, dtype=int).copy()
        self.pressure = np.asarray(self.pressure, dtype=float).copy()
        self.velocity = np.asarray(self.velocity, dtype=float).copy()
        self.momentum_residual = np.asarray(self.momentum_residual, dtype=float).copy()
        self.continuity_residual = np.asarray(self.continuity_residual, dtype=float).copy()

        row_count = self.iterations.size
        arrays = (
            self.pressure,
            self.velocity,
            self.momentum_residual,
            self.continuity_residual,
        )
        if any(array.ndim != 2 for array in arrays):
            raise ValueError("telemetry arrays must be 2D")
        if any(array.shape[0] != row_count for array in arrays):
            raise ValueError("telemetry arrays must have one row per iteration")

    @property
    def momentum_norm(self) -> np.ndarray:
        return np.linalg.norm(self.momentum_residual, ord=np.inf, axis=1)

    @property
    def continuity_norm(self) -> np.ndarray:
        return np.linalg.norm(self.continuity_residual, ord=np.inf, axis=1)

    @property
    def residual_norm(self) -> np.ndarray:
        return np.maximum(self.momentum_norm, self.continuity_norm)

    @property
    def final_iteration(self) -> int:
        return int(self.iterations[-1])

    def snapshot(self, index: int) -> dict:
        return {
            "iteration": int(self.iterations[index]),
            "pressure": self.pressure[index].copy(),
            "velocity": self.velocity[index].copy(),
            "momentum_residual": self.momentum_residual[index].copy(),
            "continuity_residual": self.continuity_residual[index].copy(),
            "momentum_norm": float(self.momentum_norm[index]),
            "continuity_norm": float(self.continuity_norm[index]),
            "residual_norm": float(self.residual_norm[index]),
        }

    def final_snapshot(self) -> dict:
        return self.snapshot(-1)


def momentum_residual_vector(
    geometry: Geometry,
    field: Field,
    momentum_asm: MomentumAssembler,
) -> np.ndarray:
    coeffs = momentum_asm.assemble(field)
    residual = np.zeros(geometry.n_velocity)

    for i in range(geometry.n_velocity):
        residual[i] = coeffs.a_p[i] * field.u[i] - coeffs.source[i]
        if i > 0:
            residual[i] -= coeffs.a_w[i] * field.u[i - 1]
        if i < geometry.n_velocity - 1:
            residual[i] -= coeffs.a_e[i] * field.u[i + 1]

    return residual


def continuity_residual_vector(
    field: Field,
    momentum_asm: MomentumAssembler,
    p_corr_asm: PressureCorrectionAssembler,
) -> np.ndarray:
    momentum = momentum_asm.assemble(field)
    return p_corr_asm.continuity_residual(field, momentum)


def collect_simple_history(
    case: SolverCase,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> SimpleTelemetry:
    tolerance = case.solver.tolerance if tolerance is None else tolerance
    max_iterations = case.solver.max_iterations if max_iterations is None else max_iterations

    geometry = case.geometry
    field = case.field
    momentum_asm = case.momentum_asm
    p_corr_asm = case.pressure_correction_asm
    step_solver = case.step_solver

    iterations: list[int] = []
    pressure: list[np.ndarray] = []
    velocity: list[np.ndarray] = []
    momentum_residual: list[np.ndarray] = []
    continuity_residual: list[np.ndarray] = []

    def record(iteration: int) -> None:
        iterations.append(iteration)
        pressure.append(field.p.copy())
        velocity.append(field.u.copy())
        momentum_residual.append(momentum_residual_vector(geometry, field, momentum_asm))
        continuity_residual.append(
            continuity_residual_vector(field, momentum_asm, p_corr_asm),
        )

    record(0)
    for iteration in range(1, max_iterations + 1):
        step_solver.run_single_iteration()
        record(iteration)

        momentum_norm = np.linalg.norm(momentum_residual[-1], ord=np.inf)
        continuity_norm = np.linalg.norm(continuity_residual[-1], ord=np.inf)
        if max(momentum_norm, continuity_norm) < tolerance:
            break

    return SimpleTelemetry(
        iterations=np.asarray(iterations),
        pressure=np.vstack(pressure),
        velocity=np.vstack(velocity),
        momentum_residual=np.vstack(momentum_residual),
        continuity_residual=np.vstack(continuity_residual),
    )


def collect_versteeg_convergence(
    tolerance: float = 1e-5,
    max_iterations: int = 100,
    density: float = 1.0,
    pressure_relaxation: float = 0.7,
    velocity_relaxation: float = 0.7,
) -> SimpleTelemetry:
    case: VersteegExample62Case = build_versteeg_example_6_2_case(
        density=density,
        tolerance=tolerance,
        max_iterations=max_iterations,
        pressure_relaxation=pressure_relaxation,
        velocity_relaxation=velocity_relaxation,
    )
    return collect_simple_history(case, tolerance=tolerance, max_iterations=max_iterations)


def collect_case_convergence(
    case_name: str,
    method: str = "simple",
    **configuration: Any,
) -> SimpleTelemetry:
    """Build a registered case and collect per-iteration convergence telemetry."""
    case = build_case_by_name(case_name, coupling=method, **configuration)
    return collect_simple_history(
        case,
        tolerance=case.solver.tolerance,
        max_iterations=case.solver.max_iterations,
    )
