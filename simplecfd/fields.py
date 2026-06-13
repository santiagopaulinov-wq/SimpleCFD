from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simplecfd.geometry import Geometry


@dataclass
class Field:
    """Primary and correction variables for pressure-velocity coupling.

    `p` and `p_prime` are pressure-node arrays. `u` is the staggered velocity
    array between pressure nodes.
    """

    p: np.ndarray
    u: np.ndarray
    p_prime: np.ndarray

    def __post_init__(self) -> None:
        self.p = np.asarray(self.p, dtype=float).copy()
        self.u = np.asarray(self.u, dtype=float).copy()
        self.p_prime = np.asarray(self.p_prime, dtype=float).copy()

        if self.p.ndim != 1:
            raise ValueError("p must be a 1D array")
        if self.u.ndim != 1:
            raise ValueError("u must be a 1D array")
        if self.p_prime.ndim != 1:
            raise ValueError("p_prime must be a 1D array")
        if self.p_prime.size != self.p.size:
            raise ValueError("p_prime must have the same size as p")
        if self.u.size != self.p.size - 1:
            raise ValueError("u must have n_pressure - 1 entries")

    @classmethod
    def zeros(cls, n_pressure: int, n_velocity: int) -> "Field":
        return cls(
            p=np.zeros(n_pressure, dtype=float),
            u=np.zeros(n_velocity, dtype=float),
            p_prime=np.zeros(n_pressure, dtype=float),
        )

    @classmethod
    def from_geometry(
        cls,
        geometry: Geometry,
        pressure: float = 0.0,
        velocity: float = 0.0,
    ) -> "Field":
        return cls(
            p=np.full(geometry.n_pressure, pressure, dtype=float),
            u=np.full(geometry.n_velocity, velocity, dtype=float),
            p_prime=np.zeros(geometry.n_pressure, dtype=float),
        )

    @classmethod
    def initial_nozzle_guess(
        cls,
        geometry: Geometry,
        density: float,
        mass_flow_guess: float,
        inlet_pressure: float,
        outlet_pressure: float,
    ) -> "Field":
        """Initial fields used in Versteeg example 6.2.

        The book guesses a mass flow rate K and initializes velocity with:

            u_i = K / (rho * A_i)

        The pressure field is initialized as a linear variation between inlet
        stagnation pressure and outlet static pressure.
        """
        if density <= 0.0:
            raise ValueError("density must be positive")

        p = np.linspace(inlet_pressure, outlet_pressure, geometry.n_pressure)
        u = mass_flow_guess / (density * geometry.velocity_areas)
        return cls(
            p=p,
            u=u,
            p_prime=np.zeros(geometry.n_pressure, dtype=float),
        )

    @classmethod
    def versteeg_example_6_2_initial_guess(cls) -> "Field":
        """Initial p and u fields from Versteeg and Malalasekera example 6.2."""
        geometry = Geometry.versteeg_example_6_2()
        return cls.initial_nozzle_guess(
            geometry=geometry,
            density=1.0,
            mass_flow_guess=1.0,
            inlet_pressure=10.0,
            outlet_pressure=0.0,
        )

    def copy(self) -> "Field":
        return Field(self.p.copy(), self.u.copy(), self.p_prime.copy())

    def validate_against(self, geometry: Geometry) -> None:
        if self.p.size != geometry.n_pressure:
            raise ValueError("p size does not match geometry.n_pressure")
        if self.p_prime.size != geometry.n_pressure:
            raise ValueError("p_prime size does not match geometry.n_pressure")
        if self.u.size != geometry.n_velocity:
            raise ValueError("u size does not match geometry.n_velocity")

    def reset_pressure_correction(self) -> None:
        self.p_prime.fill(0.0)
