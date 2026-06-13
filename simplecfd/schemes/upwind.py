from __future__ import annotations

from simplecfd.schemes.base import ConvectionScheme


class Upwind(ConvectionScheme):
    def interpolate(self, west_value: float, east_value: float, mass_flux: float) -> float:
        return west_value if mass_flux >= 0.0 else east_value

    def west_coefficient(self, mass_flux: float) -> float:
        return max(mass_flux, 0.0)

    def east_coefficient(self, mass_flux: float) -> float:
        return max(-mass_flux, 0.0)
