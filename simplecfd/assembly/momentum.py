from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simplecfd.boundary import InletStagnationPressure, OutletFixedPressure
from simplecfd.coefficients import MomentumCoefficients
from simplecfd.fields import Field
from simplecfd.geometry import Geometry
from simplecfd.momentum_terms import MomentumTerm
from simplecfd.schemes.base import ConvectionScheme


@dataclass
class MomentumAssembler:
    geometry: Geometry
    density: float
    scheme: ConvectionScheme
    inlet: InletStagnationPressure | None = None
    outlet: OutletFixedPressure | None = None
    terms: tuple[MomentumTerm, ...] = ()

    def __post_init__(self) -> None:
        self.terms = tuple(self.terms)

    def assemble(self, fields: Field) -> MomentumCoefficients:
        n = self.geometry.n_velocity
        coeffs = MomentumCoefficients(
            a_w=np.zeros(n),
            a_e=np.zeros(n),
            a_p=np.zeros(n),
            source=np.zeros(n),
            d=np.zeros(n),
            f_w=np.zeros(n),
            f_e=np.zeros(n),
        )

        if self.inlet is not None:
            self.inlet.apply(self.geometry, self.density, fields, coeffs)
        if self.outlet is not None:
            self.outlet.apply(self.geometry, self.density, fields, coeffs)

        for i in range(1, n - 1):
            self.build_internal_node(i, fields, coeffs)

        for term in self.terms:
            term.apply(self.geometry, self.density, fields, coeffs)

        self.update_pressure_correction_coefficients(coeffs)
        return coeffs

    def build_internal_node(
        self,
        i: int,
        fields: Field,
        coeffs: MomentumCoefficients,
    ) -> None:
        """Assemble one generic velocity-control-volume node.

        Internal Versteeg example 6.2 nozzle momentum node:

            Fw = rho * A_w * (u_W + u_P) / 2
            Fe = rho * A_e * (u_P + u_E) / 2
            aW = selected_scheme.a_w(Fw)
            aE = selected_scheme.a_e(Fe)
            aP = aW + aE + (Fe - Fw)
            Su = (p_left - p_right) * A_u
            d = A_u / aP

        The book uses the average of the velocity values straddling each face
        to compute Fw and Fe. The configured convection scheme then supplies
        the momentum coefficients. Boundaries are intentionally excluded: nodes
        1 and 4 in the worked example need inlet/outlet-specific treatment.
        """
        fields.validate_against(self.geometry)
        if i <= 0 or i >= self.geometry.n_velocity - 1:
            raise ValueError("build_internal_node only accepts non-boundary velocity nodes")

        velocity_area = self.geometry.velocity_area(i)
        west_area = self.geometry.pressure_area(i)
        east_area = self.geometry.pressure_area(i + 1)
        pressure_left = fields.p[i]
        pressure_right = fields.p[i + 1]

        u_w = self.average_west_velocity(i, fields)
        u_e = self.average_east_velocity(i, fields)
        f_w = self.density * west_area * u_w
        f_e = self.density * east_area * u_e

        coeffs.f_w[i] = f_w
        coeffs.f_e[i] = f_e
        coeffs.a_w[i] = self.scheme.west_coefficient(f_w)
        coeffs.a_e[i] = self.scheme.east_coefficient(f_e)
        coeffs.source[i] = (pressure_left - pressure_right) * velocity_area
        coeffs.a_p[i] = coeffs.a_w[i] + coeffs.a_e[i] + (f_e - f_w)
        coeffs.d[i] = velocity_area / coeffs.a_p[i] if coeffs.a_p[i] != 0.0 else 0.0

    def pressure_source(self, fields: Field) -> np.ndarray:
        """Return the pressure part of each velocity-equation source term."""
        fields.validate_against(self.geometry)
        source = np.zeros(self.geometry.n_velocity, dtype=float)

        for i in range(self.geometry.n_velocity):
            velocity_area = self.geometry.velocity_area(i)
            if i == 0 and self.inlet is not None:
                source[i] = (
                    self.inlet.stagnation_pressure - fields.p[1]
                ) * velocity_area
            elif i == self.geometry.n_velocity - 1 and self.outlet is not None:
                source[i] = (fields.p[i] - self.outlet.pressure) * velocity_area
            else:
                source[i] = (fields.p[i] - fields.p[i + 1]) * velocity_area

        return source

    def source_without_pressure(
        self,
        fields: Field,
        coeffs: MomentumCoefficients,
    ) -> np.ndarray:
        """Return source terms with the pressure contribution removed."""
        return coeffs.source - self.pressure_source(fields)

    def pseudo_velocities(
        self,
        fields: Field,
        coeffs: MomentumCoefficients,
    ) -> np.ndarray:
        """Return SIMPLER pseudo-velocities on the staggered velocity nodes.

        Versteeg and Malalasekera write the momentum equation as
        u = u_hat + d * (p_w - p_e).  This helper computes u_hat from neighbor
        velocity contributions and the non-pressure source while leaving the
        pressure term available for a later SIMPLER pressure equation.
        """
        fields.validate_against(self.geometry)
        source_without_pressure = self.source_without_pressure(fields, coeffs)
        pseudo = np.zeros(self.geometry.n_velocity, dtype=float)

        for i in range(self.geometry.n_velocity):
            if coeffs.a_p[i] == 0.0:
                continue
            neighbor_contribution = 0.0
            if i > 0:
                neighbor_contribution += coeffs.a_w[i] * fields.u[i - 1]
            if i < self.geometry.n_velocity - 1:
                neighbor_contribution += coeffs.a_e[i] * fields.u[i + 1]
            pseudo[i] = (
                neighbor_contribution + source_without_pressure[i]
            ) / coeffs.a_p[i]

        return pseudo

    def update_pressure_correction_coefficients(
        self,
        coeffs: MomentumCoefficients,
    ) -> None:
        for i in range(self.geometry.n_velocity):
            coeffs.d[i] = (
                self.geometry.velocity_area(i) / coeffs.a_p[i]
                if coeffs.a_p[i] != 0.0
                else 0.0
            )

    def average_west_velocity(self, i: int, fields: Field) -> float:
        return 0.5 * (fields.u[i - 1] + fields.u[i])

    def average_east_velocity(self, i: int, fields: Field) -> float:
        return 0.5 * (fields.u[i] + fields.u[i + 1])

    def mass_flux(self, face_i: int, fields: Field) -> float:
        """Mass flux at pressure face bounding a velocity control volume."""
        fields.validate_against(self.geometry)
        face_i = max(0, min(face_i, self.geometry.n_pressure - 1))
        if face_i == 0:
            velocity = fields.u[0]
        elif face_i == self.geometry.n_pressure - 1:
            velocity = fields.u[-1]
        else:
            velocity = 0.5 * (fields.u[face_i - 1] + fields.u[face_i])
        return self.density * self.geometry.pressure_area(face_i) * velocity
