from __future__ import annotations

from abc import ABC, abstractmethod


class ConvectionScheme(ABC):
    """Velocity interpolation policy used by assemblers."""

    @abstractmethod
    def interpolate(self, west_value: float, east_value: float, mass_flux: float) -> float:
        """Return a face value using the selected convection scheme."""

    @abstractmethod
    def west_coefficient(self, mass_flux: float) -> float:
        """Return the west-neighbor convective coefficient contribution."""

    @abstractmethod
    def east_coefficient(self, mass_flux: float) -> float:
        """Return the east-neighbor convective coefficient contribution."""
