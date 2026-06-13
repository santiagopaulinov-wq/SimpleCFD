from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

# Tu código de clases aquí abajo...

@dataclass(frozen=True)
class Geometry:
    """Geometry for a 1D staggered pressure-velocity coupling mesh.

    Pressure values live at pressure nodes A, B, C, ...
    Velocity values live at faces/control volumes between pressure nodes:

    ```text
    p0    p1    p2    p3
      u0    u1    u2
    ```
    """

    pressure_areas: np.ndarray
    velocity_areas: np.ndarray
    dx: float | np.ndarray

    def __post_init__(self) -> None:
        pressure_areas = np.asarray(self.pressure_areas, dtype=float).copy()
        velocity_areas = np.asarray(self.velocity_areas, dtype=float).copy()
        dx = self._normalize_dx(self.dx, pressure_areas.size)

        object.__setattr__(self, "pressure_areas", pressure_areas)
        object.__setattr__(self, "velocity_areas", velocity_areas)
        object.__setattr__(self, "dx", dx)

        if pressure_areas.ndim != 1:
            raise ValueError("pressure_areas must be a 1D array")
        if velocity_areas.ndim != 1:
            raise ValueError("velocity_areas must be a 1D array")
        if pressure_areas.size < 2:
            raise ValueError("at least two pressure nodes are required")
        if velocity_areas.size != pressure_areas.size - 1:
            raise ValueError("velocity_areas must have n_pressure - 1 entries")
        if np.any(pressure_areas <= 0.0):
            raise ValueError("pressure_areas must be positive")
        if np.any(velocity_areas <= 0.0):
            raise ValueError("velocity_areas must be positive")
        if np.any(self.dx_values <= 0.0):
            raise ValueError("dx must be positive")

    @classmethod
    def from_pressure_areas(
        cls,
        pressure_areas: np.ndarray,
        dx: float | np.ndarray,
    ) -> "Geometry":
        """Create face areas by averaging adjacent pressure-node areas."""
        pressure_areas = np.asarray(pressure_areas, dtype=float)
        velocity_areas = 0.5 * (pressure_areas[:-1] + pressure_areas[1:])
        return cls(pressure_areas=pressure_areas, velocity_areas=velocity_areas, dx=dx)

    @classmethod
    def linear_nozzle(
        cls,
        inlet_area: float,
        outlet_area: float,
        n_pressure: int,
        dx: float | np.ndarray,
    ) -> "Geometry":
        """Build a simple linearly varying nozzle area distribution."""
        pressure_areas = np.linspace(inlet_area, outlet_area, n_pressure)
        return cls.from_pressure_areas(pressure_areas=pressure_areas, dx=dx)

    @classmethod
    def versteeg_example_6_2(cls) -> "Geometry":
        """Geometry data from Versteeg and Malalasekera example 6.2.

        Pressure-node areas:
        AA = 0.5, AB = 0.4, AC = 0.3, AD = 0.2, AE = 0.1 m2

        Velocity-node areas:
        A1 = 0.45, A2 = 0.35, A3 = 0.25, A4 = 0.15 m2

        The nozzle length is 2.0 m with four uniform intervals, so dx = 0.5 m.
        """
        return cls(
            pressure_areas=np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=float),
            velocity_areas=np.array([0.45, 0.35, 0.25, 0.15], dtype=float),
            dx=0.5,
        )

    @property
    def n_pressure(self) -> int:
        return self.pressure_areas.size

    @property
    def n_velocity(self) -> int:
        return self.velocity_areas.size

    def pressure_area(self, i: int) -> float:
        return float(self.pressure_areas[i])

    def velocity_area(self, i: int) -> float:
        return float(self.velocity_areas[i])

    @property
    def inlet_area(self) -> float:
        return self.pressure_area(0)

    @property
    def outlet_area(self) -> float:
        return self.pressure_area(self.n_pressure - 1)

    def face_area_from_pressure_nodes(self, left_pressure_node: int) -> float:
        """Area at the face between pressure nodes i and i + 1."""
        return self.average_pressure_area(left_pressure_node, left_pressure_node + 1)

    def average_pressure_area(
        self,
        left_pressure_node: int,
        right_pressure_node: int,
    ) -> float:
        return 0.5 * (
            self.pressure_area(left_pressure_node) + self.pressure_area(right_pressure_node)
        )

    @property
    def dx_values(self) -> np.ndarray:
        dx = np.asarray(self.dx, dtype=float)
        if dx.ndim == 0:
            return np.full(self.n_velocity, float(dx), dtype=float)
        return dx.copy()

    @property
    def pressure_positions(self) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(self.dx_values)))

    @property
    def velocity_positions(self) -> np.ndarray:
        return self.pressure_positions[:-1] + 0.5 * self.dx_values

    def velocity_control_volume_width(self, i: int) -> float:
        return float(self.dx_values[i])

    def west_velocity_spacing(self, i: int) -> float:
        if i <= 0:
            raise ValueError("west spacing is only defined for non-inlet velocity nodes")
        positions = self.velocity_positions
        return float(positions[i] - positions[i - 1])

    def east_velocity_spacing(self, i: int) -> float:
        if i >= self.n_velocity - 1:
            raise ValueError("east spacing is only defined for non-outlet velocity nodes")
        positions = self.velocity_positions
        return float(positions[i + 1] - positions[i])

    @property
    def length(self) -> float:
        return float(np.sum(self.dx_values))

    def _normalize_dx(self, dx: float | np.ndarray, n_pressure: int) -> float | np.ndarray:
        if isinstance(dx, Real) and not isinstance(dx, bool):
            return float(dx)

        dx_array = np.asarray(dx, dtype=float)
        if dx_array.ndim != 1:
            raise ValueError("dx must be a scalar or a 1D array")
        if dx_array.size != n_pressure - 1:
            raise ValueError("dx array must have n_pressure - 1 entries")
        return dx_array.copy()
