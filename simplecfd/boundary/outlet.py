from __future__ import annotations

from dataclasses import dataclass

from simplecfd.coefficients import MomentumCoefficients
from simplecfd.fields import Field
from simplecfd.geometry import Geometry


@dataclass(frozen=True)
class OutletFixedPressure:
    """Outlet boundary with fixed static pressure.

    For the last velocity control volume of example 6.2:

        Fw = rho * A_w * (u_W + u_P) / 2
        Fe = rho * A_4 * u_4
        aW = Fw
        aE = 0
        aP = aW + aE + (Fe - Fw) = Fe
        Su = (pD - pE) * A4
    """

    pressure: float

    def apply(
        self,
        geometry: Geometry,
        density: float,
        fields: Field,
        coeffs: MomentumCoefficients,
    ) -> None:
        fields.validate_against(geometry)
        i = geometry.n_velocity - 1

        velocity_area = geometry.velocity_area(i)
        west_pressure_area = geometry.pressure_area(i)
        f_w = density * west_pressure_area * 0.5 * (fields.u[i - 1] + fields.u[i])
        f_e = density * velocity_area * fields.u[i]

        coeffs.f_w[i] = f_w
        coeffs.f_e[i] = f_e
        coeffs.a_w[i] = max(f_w, 0.0)
        coeffs.a_e[i] = max(-f_e, 0.0)
        coeffs.a_p[i] = coeffs.a_w[i] + coeffs.a_e[i] + (f_e - f_w)
        coeffs.source[i] = (fields.p[i] - self.pressure) * velocity_area
        coeffs.d[i] = velocity_area / coeffs.a_p[i]
