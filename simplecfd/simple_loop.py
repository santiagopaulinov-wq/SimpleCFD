from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Protocol

import numpy as np

from simplecfd.assembly.pressure_absolute import AbsolutePressureAssembler
from simplecfd.assembly.momentum import MomentumAssembler
from simplecfd.assembly.pressure_correction import PressureCorrectionAssembler
from simplecfd.coefficients import LinearSystem, MomentumCoefficients
from simplecfd.fields import Field
from simplecfd.geometry import Geometry
from simplecfd.linalg import tdma


@dataclass
class CouplingIterationContext:
    """Operations available to one pressure-velocity coupling iteration."""

    step_solver: "PressureVelocityStepSolver"

    @property
    def geometry(self) -> Geometry:
        return self.step_solver.geometry

    @property
    def field(self) -> Field:
        return self.step_solver.field

    @property
    def pressure_relaxation(self) -> float:
        return self.step_solver.pressure_relaxation

    @property
    def velocity_relaxation(self) -> float:
        return self.step_solver.velocity_relaxation

    def assemble_momentum_system(self) -> tuple[MomentumCoefficients, LinearSystem]:
        return self.step_solver.assemble_momentum_system()

    def assemble_pressure_correction_system(
        self,
        momentum_coeffs: MomentumCoefficients,
    ) -> LinearSystem:
        return self.step_solver.assemble_pressure_correction_system(momentum_coeffs)

    def solve_linear_system(self, system: LinearSystem) -> np.ndarray:
        return self.step_solver.solve_linear_system(system)

    def run_pressure_velocity_stage(self) -> np.ndarray:
        return self.step_solver.run_pressure_velocity_stage()

    def predict_momentum(self) -> MomentumCoefficients:
        return self.step_solver.predict_momentum()

    def solve_absolute_pressure(
        self,
        momentum_coeffs: MomentumCoefficients | None = None,
    ) -> np.ndarray:
        return self.step_solver.solve_absolute_pressure(momentum_coeffs)

    def solve_pressure_correction(
        self,
        momentum_coeffs: MomentumCoefficients,
    ) -> np.ndarray:
        return self.step_solver.solve_pressure_correction(momentum_coeffs)

    def correct_pressure(self, p_prime: np.ndarray) -> None:
        self.step_solver.correct_pressure(p_prime)

    def correct_velocity(
        self,
        momentum_coeffs: MomentumCoefficients,
        p_prime: np.ndarray,
    ) -> None:
        self.step_solver.correct_velocity(momentum_coeffs, p_prime)


class PressureVelocityCouplingStrategy(Protocol):
    def run_iteration(self, context: CouplingIterationContext) -> np.ndarray:
        ...

    def pressure_correction_momentum_coefficients(
        self,
        geometry: Geometry,
        momentum_coeffs: MomentumCoefficients,
        velocity_relaxation: float = 1.0,
    ) -> MomentumCoefficients:
        ...

    def momentum_system_coefficients(
        self,
        momentum_coeffs: MomentumCoefficients,
        old_velocity: np.ndarray,
        velocity_relaxation: float,
    ) -> MomentumCoefficients:
        ...

    def relax_velocity(
        self,
        old_velocity: np.ndarray,
        starred_velocity: np.ndarray,
        velocity_relaxation: float,
    ) -> np.ndarray:
        ...

    def apply_correction(
        self,
        geometry: Geometry,
        field: Field,
        momentum_coeffs: MomentumCoefficients,
        p_prime: np.ndarray,
        pressure_relaxation: float,
    ) -> None:
        ...

    def correct_pressure(
        self,
        field: Field,
        p_prime: np.ndarray,
        pressure_relaxation: float,
    ) -> None:
        ...

    def correct_velocity(
        self,
        geometry: Geometry,
        field: Field,
        momentum_coeffs: MomentumCoefficients,
        p_prime: np.ndarray,
    ) -> None:
        ...


class SIMPLECouplingStrategy:
    """SIMPLE pressure-velocity coupling strategy."""

    def run_iteration(self, context: CouplingIterationContext) -> np.ndarray:
        return context.run_pressure_velocity_stage()

    def pressure_correction_momentum_coefficients(
        self,
        geometry: Geometry,
        momentum_coeffs: MomentumCoefficients,
        velocity_relaxation: float = 1.0,
    ) -> MomentumCoefficients:
        return momentum_coeffs

    def momentum_system_coefficients(
        self,
        momentum_coeffs: MomentumCoefficients,
        old_velocity: np.ndarray,
        velocity_relaxation: float,
    ) -> MomentumCoefficients:
        return momentum_coeffs

    def relax_velocity(
        self,
        old_velocity: np.ndarray,
        starred_velocity: np.ndarray,
        velocity_relaxation: float,
    ) -> np.ndarray:
        return old_velocity + velocity_relaxation * (starred_velocity - old_velocity)

    def apply_correction(
        self,
        geometry: Geometry,
        field: Field,
        momentum_coeffs: MomentumCoefficients,
        p_prime: np.ndarray,
        pressure_relaxation: float,
    ) -> None:
        field.p_prime = p_prime.copy()
        self.correct_pressure(field, p_prime, pressure_relaxation)
        self.correct_velocity(geometry, field, momentum_coeffs, p_prime)

    def correct_pressure(
        self,
        field: Field,
        p_prime: np.ndarray,
        pressure_relaxation: float,
    ) -> None:
        field.p += pressure_relaxation * p_prime

    def correct_velocity(
        self,
        geometry: Geometry,
        field: Field,
        momentum_coeffs: MomentumCoefficients,
        p_prime: np.ndarray,
    ) -> None:
        for i in range(geometry.n_velocity):
            west_node = i
            east_node = i + 1
            field.u[i] += momentum_coeffs.d[i] * (
                p_prime[west_node] - p_prime[east_node]
            )


class SIMPLECCouplingStrategy(SIMPLECouplingStrategy):
    """First SIMPLEC variant using a corrected velocity-pressure coefficient."""

    def momentum_system_coefficients(
        self,
        momentum_coeffs: MomentumCoefficients,
        old_velocity: np.ndarray,
        velocity_relaxation: float,
    ) -> MomentumCoefficients:
        if velocity_relaxation <= 0.0 or velocity_relaxation > 1.0:
            raise ValueError("velocity_relaxation must be in the interval (0, 1]")
        return replace(
            momentum_coeffs,
            a_p=momentum_coeffs.a_p / velocity_relaxation,
            source=momentum_coeffs.source
            + ((1.0 - velocity_relaxation) / velocity_relaxation)
            * momentum_coeffs.a_p
            * old_velocity,
        )

    def pressure_correction_momentum_coefficients(
        self,
        geometry: Geometry,
        momentum_coeffs: MomentumCoefficients,
        velocity_relaxation: float = 1.0,
    ) -> MomentumCoefficients:
        return replace(
            momentum_coeffs,
            d=self.correction_coefficients(
                geometry,
                momentum_coeffs,
                velocity_relaxation,
            ),
        )

    def correction_coefficients(
        self,
        geometry: Geometry,
        momentum_coeffs: MomentumCoefficients,
        velocity_relaxation: float = 1.0,
    ) -> np.ndarray:
        if velocity_relaxation <= 0.0 or velocity_relaxation > 1.0:
            raise ValueError("velocity_relaxation must be in the interval (0, 1]")
        modified_a_p = momentum_coeffs.a_p / velocity_relaxation
        denominator = modified_a_p - momentum_coeffs.a_w - momentum_coeffs.a_e
        singular = np.isclose(denominator, 0.0, rtol=0.0, atol=1e-14)
        if np.any(singular):
            nodes = ", ".join(str(i) for i in np.flatnonzero(singular))
            raise ZeroDivisionError(
                "SIMPLEC correction coefficient denominator is zero "
                f"at velocity node(s): {nodes}"
            )
        return geometry.velocity_areas / denominator

    def relax_velocity(
        self,
        old_velocity: np.ndarray,
        starred_velocity: np.ndarray,
        velocity_relaxation: float,
    ) -> np.ndarray:
        return starred_velocity.copy()

    def apply_correction_with_velocity_relaxation(
        self,
        geometry: Geometry,
        field: Field,
        momentum_coeffs: MomentumCoefficients,
        p_prime: np.ndarray,
        pressure_relaxation: float,
        velocity_relaxation: float = 1.0,
    ) -> None:
        simplec_momentum = self.pressure_correction_momentum_coefficients(
            geometry,
            momentum_coeffs,
            velocity_relaxation,
        )
        field.p_prime = p_prime.copy()
        self.correct_pressure(field, p_prime, pressure_relaxation)
        super().correct_velocity(geometry, field, simplec_momentum, p_prime)

    def correct_velocity(
        self,
        geometry: Geometry,
        field: Field,
        momentum_coeffs: MomentumCoefficients,
        p_prime: np.ndarray,
    ) -> None:
        super().correct_velocity(geometry, field, momentum_coeffs, p_prime)


class SIMPLERCouplingStrategy(SIMPLECouplingStrategy):
    """SIMPLER pressure-velocity coupling for the 1D staggered formulation."""

    def run_iteration(self, context: CouplingIterationContext) -> np.ndarray:
        context.solve_absolute_pressure()
        momentum_coeffs = context.predict_momentum()
        p_prime = context.solve_pressure_correction(momentum_coeffs)
        context.correct_velocity(momentum_coeffs, p_prime)
        return p_prime


class PressureVelocityStepSolver:
    """Run one pressure-velocity coupling iteration for the selected strategy."""

    def __init__(
        self,
        geometry: Geometry,
        field: Field,
        momentum_assembler: MomentumAssembler,
        pressure_correction_assembler: PressureCorrectionAssembler,
        pressure_relaxation: float = 0.7,
        velocity_relaxation: float = 0.7,
        coupling_strategy: PressureVelocityCouplingStrategy | None = None,
    ):
        self._validate_relaxation("pressure_relaxation", pressure_relaxation)
        self._validate_relaxation("velocity_relaxation", velocity_relaxation)

        self.geometry = geometry
        self.field = field
        self.momentum_asm = momentum_assembler
        self.p_corr_asm = pressure_correction_assembler
        self.pressure_relaxation = float(pressure_relaxation)
        self.velocity_relaxation = float(velocity_relaxation)
        self.coupling_strategy = (
            SIMPLECouplingStrategy() if coupling_strategy is None else coupling_strategy
        )
        self.linear_solve_history: list[dict[str, int | dict[str, int]]] = []
        self._active_linear_solve_counts: dict[str, int] | None = None
        self._suppress_linear_solve_record = False

    def run_single_iteration(self) -> np.ndarray:
        self._begin_linear_solve_iteration()
        try:
            iteration_runner = getattr(self.coupling_strategy, "run_iteration", None)
            if iteration_runner is None:
                return self.run_pressure_velocity_stage()
            return iteration_runner(CouplingIterationContext(self))
        finally:
            self._end_linear_solve_iteration()

    def run_pressure_velocity_stage(self) -> np.ndarray:
        momentum_coeffs = self.predict_momentum()
        p_prime = self.solve_pressure_correction(momentum_coeffs)
        self.apply_pressure_velocity_correction(momentum_coeffs, p_prime)
        return p_prime

    def predict_momentum(self) -> MomentumCoefficients:
        momentum_coeffs, momentum_system = self.assemble_momentum_system()
        old_velocity = self.field.u.copy()
        starred_velocity = self.solve_stage_linear_system(momentum_system, "momentum")
        self.field.u = self.relax_velocity(old_velocity, starred_velocity)
        return momentum_coeffs

    def solve_absolute_pressure(
        self,
        momentum_coeffs: MomentumCoefficients | None = None,
    ) -> np.ndarray:
        if momentum_coeffs is None:
            momentum_coeffs, _ = self.assemble_momentum_system()
        pseudo_velocity = self.momentum_asm.pseudo_velocities(
            self.field,
            momentum_coeffs,
        )
        pressure_system = AbsolutePressureAssembler(
            self.geometry,
            self.p_corr_asm.density,
        ).assemble(self.field, momentum_coeffs, pseudo_velocity)
        self.field.p = self.solve_stage_linear_system(
            pressure_system,
            "absolute_pressure",
        )
        return self.field.p

    def solve_pressure_correction(
        self,
        momentum_coeffs: MomentumCoefficients,
    ) -> np.ndarray:
        pressure_system = self.assemble_pressure_correction_system(momentum_coeffs)
        p_prime = self.solve_stage_linear_system(
            pressure_system,
            "pressure_correction",
        )
        self.field.p_prime = p_prime.copy()
        return p_prime

    def assemble_momentum_system(self) -> tuple[MomentumCoefficients, LinearSystem]:
        coeffs = self.momentum_asm.assemble(self.field)
        system_coeffs = self.momentum_system_coefficients(
            coeffs,
            self.field.u.copy(),
        )
        return coeffs, self._build_momentum_system(system_coeffs)

    def momentum_system_coefficients(
        self,
        momentum_coeffs: MomentumCoefficients,
        old_velocity: np.ndarray,
    ) -> MomentumCoefficients:
        coefficient_builder = getattr(
            self.coupling_strategy,
            "momentum_system_coefficients",
            None,
        )
        if coefficient_builder is None:
            return momentum_coeffs
        return coefficient_builder(
            momentum_coeffs,
            old_velocity,
            self.velocity_relaxation,
        )

    def assemble_pressure_correction_system(
        self,
        momentum_coeffs: MomentumCoefficients,
    ) -> LinearSystem:
        pressure_momentum = self.pressure_correction_momentum_coefficients(
            momentum_coeffs,
        )
        return self.p_corr_asm.assemble(self.field, pressure_momentum)

    def pressure_correction_momentum_coefficients(
        self,
        momentum_coeffs: MomentumCoefficients,
    ) -> MomentumCoefficients:
        coefficient_builder = getattr(
            self.coupling_strategy,
            "pressure_correction_momentum_coefficients",
            None,
        )
        if coefficient_builder is None:
            return momentum_coeffs
        return coefficient_builder(
            self.geometry,
            momentum_coeffs,
            self.velocity_relaxation,
        )

    def solve_linear_system(self, system: LinearSystem) -> np.ndarray:
        if not self._suppress_linear_solve_record:
            self.record_linear_solve("other")
        return tdma(system)

    def solve_stage_linear_system(
        self,
        system: LinearSystem,
        solve_kind: str,
    ) -> np.ndarray:
        self.record_linear_solve(solve_kind)
        previous_suppression = self._suppress_linear_solve_record
        self._suppress_linear_solve_record = True
        try:
            return self.solve_linear_system(system)
        finally:
            self._suppress_linear_solve_record = previous_suppression

    def record_linear_solve(self, solve_kind: str) -> None:
        if self._active_linear_solve_counts is None:
            return
        self._active_linear_solve_counts[solve_kind] = (
            self._active_linear_solve_counts.get(solve_kind, 0) + 1
        )

    def linear_solve_totals(self) -> dict[str, int | dict[str, int]]:
        by_kind: dict[str, int] = {}
        for entry in self.linear_solve_history:
            for solve_kind, count in entry["by_kind"].items():
                by_kind[solve_kind] = by_kind.get(solve_kind, 0) + count
        return {"total": sum(by_kind.values()), "by_kind": by_kind}

    def relax_velocity(
        self,
        old_velocity: np.ndarray,
        starred_velocity: np.ndarray,
    ) -> np.ndarray:
        return self.coupling_strategy.relax_velocity(
            old_velocity,
            starred_velocity,
            self.velocity_relaxation,
        )

    def apply_pressure_velocity_correction(
        self,
        momentum_coeffs: MomentumCoefficients,
        p_prime: np.ndarray,
    ) -> None:
        relaxed_correction = getattr(
            self.coupling_strategy,
            "apply_correction_with_velocity_relaxation",
            None,
        )
        if relaxed_correction is not None:
            relaxed_correction(
                self.geometry,
                self.field,
                momentum_coeffs,
                p_prime,
                self.pressure_relaxation,
                self.velocity_relaxation,
            )
            return
        self.coupling_strategy.apply_correction(
            self.geometry,
            self.field,
            momentum_coeffs,
            p_prime,
            self.pressure_relaxation,
        )

    def apply_simple_correction(
        self,
        momentum_coeffs: MomentumCoefficients,
        p_prime: np.ndarray,
    ) -> None:
        """Compatibility alias for the generic correction hook."""
        self.apply_pressure_velocity_correction(momentum_coeffs, p_prime)

    def correct_pressure(self, p_prime: np.ndarray) -> None:
        self.coupling_strategy.correct_pressure(
            self.field,
            p_prime,
            self.pressure_relaxation,
        )

    def correct_velocity(
        self,
        momentum_coeffs: MomentumCoefficients,
        p_prime: np.ndarray,
    ) -> None:
        velocity_momentum = self.pressure_correction_momentum_coefficients(
            momentum_coeffs,
        )
        self.coupling_strategy.correct_velocity(
            self.geometry,
            self.field,
            velocity_momentum,
            p_prime,
        )

    def _build_momentum_system(self, coeffs: MomentumCoefficients) -> LinearSystem:
        n = self.geometry.n_velocity
        lower = np.zeros(n)
        diagonal = coeffs.a_p.copy()
        upper = np.zeros(n)
        rhs = coeffs.source.copy()

        for i in range(n):
            if i > 0:
                lower[i] = -coeffs.a_w[i]
            if i < n - 1:
                upper[i] = -coeffs.a_e[i]

        return LinearSystem(lower=lower, diagonal=diagonal, upper=upper, rhs=rhs)

    def _begin_linear_solve_iteration(self) -> None:
        self._active_linear_solve_counts = {}

    def _end_linear_solve_iteration(self) -> None:
        counts = {} if self._active_linear_solve_counts is None else dict(
            self._active_linear_solve_counts
        )
        self.linear_solve_history.append(
            {
                "total": sum(counts.values()),
                "by_kind": counts,
            }
        )
        self._active_linear_solve_counts = None

    def _validate_relaxation(self, name: str, value: float) -> None:
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in the interval (0, 1]")


SimpleStepSolver = PressureVelocityStepSolver
