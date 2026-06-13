from __future__ import annotations

import numpy as np

from simplecfd.fields import Field
from simplecfd.geometry import Geometry
from simplecfd.simple_loop import PressureVelocityStepSolver


class PressureVelocitySolver:
    """Global pressure-velocity coupling iteration controller.

    After each `PressureVelocityStepSolver.run_single_iteration()` call,
    convergence is checked with the same physical residuals used by the
    discretized equations: continuity from the pressure-correction RHS, then
    momentum from the assembled momentum balance. Both are reduced with the
    infinity norm, and the global residual is the maximum of those two norms.
    """

    def __init__(
        self,
        geometry: Geometry,
        field: Field,
        step_solver: PressureVelocityStepSolver,
        density: float = 1.0,
        tolerance: float = 1e-5,
        max_iterations: int = 100,
    ):
        self.geometry = geometry
        self.field = field
        self.step_solver = step_solver
        self.density = density
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def calculate_continuity_residual_vector(self) -> np.ndarray:
        """Return the pressure-correction source vector b for the current field.

        In the pressure-correction equation, b = Fw* - Fe* is the local mass
        imbalance. Boundary entries are fixed to zero by the assembler.
        """
        momentum_coeffs = self.step_solver.momentum_asm.assemble(self.field)
        return self.step_solver.p_corr_asm.continuity_residual(self.field, momentum_coeffs)

    def calculate_mass_residual_vector(self) -> np.ndarray:
        """Compatibility alias for the continuity residual vector."""
        return self.calculate_continuity_residual_vector()

    def calculate_mass_residual_norm(self, order: float = np.inf) -> float:
        """Calculate the norm of the continuity residual vector b."""
        residual_vector = self.calculate_continuity_residual_vector()
        return float(np.linalg.norm(residual_vector, ord=order))

    def calculate_momentum_residual_vector(self) -> np.ndarray:
        """Return residuals of the discretized momentum equations.

        Each entry evaluates `aP*uP - aW*uW - aE*uE - source` for one velocity
        node, skipping west/east neighbor terms at boundaries.
        """
        coeffs = self.step_solver.momentum_asm.assemble(self.field)
        residual = np.zeros(self.geometry.n_velocity)

        for i in range(self.geometry.n_velocity):
            residual[i] = coeffs.a_p[i] * self.field.u[i] - coeffs.source[i]
            if i > 0:
                residual[i] -= coeffs.a_w[i] * self.field.u[i - 1]
            if i < self.geometry.n_velocity - 1:
                residual[i] -= coeffs.a_e[i] * self.field.u[i + 1]

        return residual

    def calculate_momentum_residual_norm(self, order: float = np.inf) -> float:
        """Calculate the norm of the momentum residual vector."""
        residual_vector = self.calculate_momentum_residual_vector()
        return float(np.linalg.norm(residual_vector, ord=order))

    def calculate_max_mass_residual(self) -> float:
        """Return the infinity norm of the mass-residual vector b."""
        return self.calculate_mass_residual_norm(order=np.inf)

    def solve(self) -> dict:
        """Run pressure-velocity iterations until convergence or iteration limit."""
        return self._iterate_until_converged(iteration=1, residual_history=[])

    def convergence_state(self) -> dict:
        """Evaluate the stopping criterion for the current field.

        Evaluation order:
        1. Build the continuity residual vector from the pressure-correction RHS.
        2. Build the momentum residual vector from the momentum balance.
        3. Compute infinity norms for both vectors.
        4. Stop when `max(continuity_norm, momentum_norm) < tolerance`.
        """
        continuity_residual_vector = self.calculate_continuity_residual_vector()
        momentum_residual_vector = self.calculate_momentum_residual_vector()
        continuity_residual = float(np.linalg.norm(continuity_residual_vector, ord=np.inf))
        momentum_residual = float(np.linalg.norm(momentum_residual_vector, ord=np.inf))
        residual = max(continuity_residual, momentum_residual)
        return {
            "residual": residual,
            "continuity_residual": continuity_residual,
            "continuity_residual_vector": continuity_residual_vector,
            "momentum_residual": momentum_residual,
            "momentum_residual_vector": momentum_residual_vector,
            "converged": residual < self.tolerance,
        }

    def build_result(
        self,
        converged: bool,
        iterations: int,
        residual_history: list[float],
        state: dict,
    ) -> dict:
        linear_solve_counts = self.step_solver.linear_solve_totals()
        return {
            "converged": converged,
            "iterations": iterations,
            "residual": state["residual"],
            "residual_history": residual_history,
            "continuity_residual": state["continuity_residual"],
            "continuity_residual_vector": state["continuity_residual_vector"].copy(),
            "momentum_residual": state["momentum_residual"],
            "momentum_residual_vector": state["momentum_residual_vector"].copy(),
            "linear_solve_history": [
                {"total": entry["total"], "by_kind": dict(entry["by_kind"])}
                for entry in self.step_solver.linear_solve_history
            ],
            "linear_solve_counts": {
                "total": linear_solve_counts["total"],
                "by_kind": dict(linear_solve_counts["by_kind"]),
            },
        }

    def _iterate_until_converged(
        self,
        iteration: int,
        residual_history: list[float],
    ) -> dict:
        if iteration > self.max_iterations:
            return self.build_result(
                converged=False,
                iterations=self.max_iterations,
                residual_history=residual_history,
                state=self.convergence_state(),
            )

        self.step_solver.run_single_iteration()

        state = self.convergence_state()
        residual_history.append(state["residual"])

        if state["converged"]:
            return self.build_result(
                converged=True,
                iterations=iteration,
                residual_history=residual_history,
                state=state,
            )

        return self._iterate_until_converged(
            iteration=iteration + 1,
            residual_history=residual_history,
        )


SIMPLESolver = PressureVelocitySolver
