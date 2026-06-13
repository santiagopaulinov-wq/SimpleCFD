from __future__ import annotations

from dataclasses import dataclass

from simplecfd.coefficients import MomentumCoefficients
from simplecfd.fields import Field
from simplecfd.geometry import Geometry


@dataclass(frozen=True)
class InletStagnationPressure:
    """Inlet boundary from Versteeg and Malalasekera example 6.2.

    The inlet pressure is specified as stagnation pressure in an upstream
    plenum. For the first velocity control volume the book combines Bernoulli
    and continuity:

        uA = u1 * A1 / AA
        pA = p0 - 0.5 * rho * uA**2

    and uses the deferred-correction form of equation (6.69):

        [Fe + Fw * 0.5 * (A1 / AA)**2] u1
            = (p0 - pB) A1 + Fw * (A1 / AA) * u1_old
    """

    stagnation_pressure: float

    def static_pressure(self, density: float, inlet_velocity: float) -> float:
        return self.stagnation_pressure - 0.5 * density * inlet_velocity**2

    def inlet_plane_velocity(self, geometry: Geometry, fields: Field) -> float:
        return fields.u[0] * geometry.velocity_area(0) / geometry.inlet_area

    def apply(
        self,
        geometry: Geometry,
        density: float,
        fields: Field,
        coeffs: MomentumCoefficients,
    ) -> None:
        fields.validate_against(geometry)
        i = 0

        a_1 = geometry.velocity_area(i)
        a_a = geometry.inlet_area
        a_b = geometry.pressure_area(1)
        area_ratio = a_1 / a_a

        u_a = self.inlet_plane_velocity(geometry, fields)
        f_w = density * u_a * a_a
        f_e = density * 0.5 * (fields.u[0] + fields.u[1]) * a_b

        coeffs.f_w[i] = f_w
        coeffs.f_e[i] = f_e
        coeffs.a_w[i] = 0.0
        coeffs.a_e[i] = 0.0
        coeffs.a_p[i] = f_e + f_w * 0.5 * area_ratio**2
        coeffs.source[i] = (
            (self.stagnation_pressure - fields.p[1]) * a_1
            + f_w * area_ratio * fields.u[0]
        )
        coeffs.d[i] = a_1 / coeffs.a_p[i]
