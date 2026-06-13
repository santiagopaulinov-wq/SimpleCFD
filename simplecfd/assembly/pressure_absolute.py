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
class AbsolutePressureAssembler:
    """Assemble the SIMPLER absolute-pressure equation on a 1D staggered mesh.

    For an internal pressure node P:

        u_w = uhat_w + d_w * (p_W - p_P)
        u_e = uhat_e + d_e * (p_P - p_E)
        d   = A / aP

    Substituting those relations in continuity gives:

        aP * p_P = aW * p_W + aE * p_E + Fhat_w - Fhat_e
    """

    geometry: Geometry
    density: float

    def assemble_coefficients(
        self,
        fields: Field,
        momentum: MomentumCoefficients,
        pseudo_velocity: np.ndarray,
    ) -> PressureCorrectionCoefficients:
        fields.validate_against(self.geometry)
        pseudo_velocity = self._validate_pseudo_velocity(pseudo_velocity)

        n = self.geometry.n_pressure
        coeffs = PressureCorrectionCoefficients(
            a_w=np.zeros(n),
            a_e=np.zeros(n),
            a_p=np.ones(n),
            source=np.zeros(n),
        )

        for i in range(1, n - 1):
            self.build_internal_node(i, momentum, pseudo_velocity, coeffs)

        return coeffs

    def assemble(
        self,
        fields: Field,
        momentum: MomentumCoefficients,
        pseudo_velocity: np.ndarray,
    ) -> LinearSystem:
        coeffs = self.assemble_coefficients(fields, momentum, pseudo_velocity)
        n = self.geometry.n_pressure

        lower = np.zeros(n)
        diagonal = np.ones(n)
        upper = np.zeros(n)
        rhs = np.zeros(n)

        lower[1:-1] = -coeffs.a_w[1:-1]
        diagonal[1:-1] = coeffs.a_p[1:-1]
        upper[1:-1] = -coeffs.a_e[1:-1]
        rhs[0] = fields.p[0]
        rhs[1:-1] = coeffs.source[1:-1]
        rhs[-1] = fields.p[-1]

        return LinearSystem(lower=lower, diagonal=diagonal, upper=upper, rhs=rhs)

    def build_internal_node(
        self,
        i: int,
        momentum: MomentumCoefficients,
        pseudo_velocity: np.ndarray,
        coeffs: PressureCorrectionCoefficients,
    ) -> None:
        if i <= 0 or i >= self.geometry.n_pressure - 1:
            raise ValueError("build_internal_node only accepts internal pressure nodes")

        west_velocity_node = i - 1
        east_velocity_node = i
        d = self.pressure_velocity_coefficients(momentum)

        coeffs.a_w[i] = (
            self.density
            * self.geometry.velocity_area(west_velocity_node)
            * d[west_velocity_node]
        )
        coeffs.a_e[i] = (
            self.density
            * self.geometry.velocity_area(east_velocity_node)
            * d[east_velocity_node]
        )
        coeffs.a_p[i] = coeffs.a_w[i] + coeffs.a_e[i]
        coeffs.source[i] = self.pseudo_mass_flux(
            west_velocity_node,
            pseudo_velocity,
        ) - self.pseudo_mass_flux(east_velocity_node, pseudo_velocity)

    def pressure_velocity_coefficients(
        self,
        momentum: MomentumCoefficients,
    ) -> np.ndarray:
        d = np.zeros(self.geometry.n_velocity, dtype=float)
        nonzero = momentum.a_p != 0.0
        d[nonzero] = self.geometry.velocity_areas[nonzero] / momentum.a_p[nonzero]
        return d

    def pseudo_mass_flux(
        self,
        velocity_node: int,
        pseudo_velocity: np.ndarray,
    ) -> float:
        return (
            self.density
            * self.geometry.velocity_area(velocity_node)
            * pseudo_velocity[velocity_node]
        )

    def _validate_pseudo_velocity(self, pseudo_velocity: np.ndarray) -> np.ndarray:
        pseudo_velocity = np.asarray(pseudo_velocity, dtype=float)
        if pseudo_velocity.shape != (self.geometry.n_velocity,):
            raise ValueError("pseudo_velocity must have geometry.n_velocity entries")
        return pseudo_velocity
