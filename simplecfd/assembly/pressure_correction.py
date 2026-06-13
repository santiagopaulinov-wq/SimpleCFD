from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simplecfd.coefficients import (
    LinearSystem,
    MomentumCoefficients,
    PressureCorrectionCoefficients,
)
from simplecfd.fields import Field
from simplecfd.geometry import Geometry


@dataclass
class PressureCorrectionAssembler:
    """Assemble the pressure-correction equation for pressure-velocity coupling.

    For Versteeg and Malalasekera example 6.2:

        aW = (rho*d*A)_w
        aE = (rho*d*A)_e
        aP = aW + aE
        b  = Fw* - Fe*

    Boundary pressure corrections are fixed:

        p'A = 0
        p'E = 0
    """

    geometry: Geometry
    density: float

    def assemble_coefficients(
        self,
        fields: Field,
        momentum: MomentumCoefficients,
    ) -> PressureCorrectionCoefficients:
        fields.validate_against(self.geometry)

        n = self.geometry.n_pressure
        coeffs = PressureCorrectionCoefficients(
            a_w=np.zeros(n),
            a_e=np.zeros(n),
            a_p=np.ones(n),
            source=np.zeros(n),
        )

        for i in range(1, n - 1):
            self.build_internal_node(i, fields, momentum, coeffs)

        return coeffs

    def assemble(
        self,
        fields: Field,
        momentum: MomentumCoefficients,
    ) -> LinearSystem:
        coeffs = self.assemble_coefficients(fields, momentum)
        n = self.geometry.n_pressure

        lower = np.zeros(n)
        diagonal = np.ones(n)
        upper = np.zeros(n)
        rhs = np.zeros(n)

        for i in range(1, n - 1):
            lower[i] = -coeffs.a_w[i]
            diagonal[i] = coeffs.a_p[i]
            upper[i] = -coeffs.a_e[i]
            rhs[i] = coeffs.source[i]

        return LinearSystem(lower=lower, diagonal=diagonal, upper=upper, rhs=rhs)

    def continuity_residual(
        self,
        fields: Field,
        momentum: MomentumCoefficients,
    ) -> np.ndarray:
        """Return b = Fw* - Fe* from the pressure-correction system."""
        return self.assemble(fields, momentum).rhs.copy()

    def build_internal_node(
        self,
        i: int,
        fields: Field,
        momentum: MomentumCoefficients,
        coeffs: PressureCorrectionCoefficients,
    ) -> None:
        if i <= 0 or i >= self.geometry.n_pressure - 1:
            raise ValueError("build_internal_node only accepts internal pressure nodes")

        west_velocity_node = i - 1
        east_velocity_node = i

        coeffs.a_w[i] = (
            self.density
            * momentum.d[west_velocity_node]
            * self.geometry.velocity_area(west_velocity_node)
        )
        coeffs.a_e[i] = (
            self.density
            * momentum.d[east_velocity_node]
            * self.geometry.velocity_area(east_velocity_node)
        )
        coeffs.a_p[i] = coeffs.a_w[i] + coeffs.a_e[i]
        coeffs.source[i] = self.starred_mass_flux(west_velocity_node, fields) - (
            self.starred_mass_flux(east_velocity_node, fields)
        )

    def starred_mass_flux(self, velocity_node: int, fields: Field) -> float:
        return (
            self.density
            * self.geometry.velocity_area(velocity_node)
            * fields.u[velocity_node]
        )
