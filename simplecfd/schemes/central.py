from __future__ import annotations

from simplecfd.schemes.base import ConvectionScheme


class CentralDifference(ConvectionScheme):
    def interpolate(self, west_value: float, east_value: float, mass_flux: float) -> float:
        return 0.5 * (west_value + east_value)

    def west_coefficient(self, mass_flux: float) -> float:
        return 0.5 * mass_flux

    def east_coefficient(self, mass_flux: float) -> float:
        return -0.5 * mass_flux
