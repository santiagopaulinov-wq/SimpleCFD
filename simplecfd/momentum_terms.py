from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Protocol

import numpy as np

from simplecfd.coefficients import MomentumCoefficients
from simplecfd.fields import Field
from simplecfd.geometry import Geometry


class MomentumTerm(Protocol):
    """Extensible contribution to the discretized momentum equation."""

    def apply(
        self,
        geometry: Geometry,
        density: float,
        fields: Field,
        coeffs: MomentumCoefficients,
    ) -> None:
        ...


@dataclass(frozen=True)
class MomentumDiffusion:
    """1D diffusion contribution for velocity control volumes.

    The coefficient is the transported momentum diffusivity. For a Newtonian
    laminar case this is the dynamic viscosity, mu. Internal velocity nodes get
    the standard finite-volume conductances:

        Dw = gamma * Aw / delta_x_w
        De = gamma * Ae / delta_x_e

    Boundary velocity nodes are left to the boundary-condition objects.
    """

    diffusivity: float

    def __post_init__(self) -> None:
        _validate_non_negative_finite("diffusivity", self.diffusivity)

    def apply(
        self,
        geometry: Geometry,
        density: float,
        fields: Field,
        coeffs: MomentumCoefficients,
    ) -> None:
        fields.validate_against(geometry)
        if geometry.n_velocity <= 2 or self.diffusivity == 0.0:
            return

        for i in range(1, geometry.n_velocity - 1):
            d_w = (
                self.diffusivity
                * geometry.pressure_area(i)
                / geometry.west_velocity_spacing(i)
            )
            d_e = (
                self.diffusivity
                * geometry.pressure_area(i + 1)
                / geometry.east_velocity_spacing(i)
            )
            coeffs.a_w[i] += d_w
            coeffs.a_e[i] += d_e
            coeffs.a_p[i] += d_w + d_e


@dataclass(frozen=True)
class LinearFrictionLoss:
    """Distributed linear momentum sink.

    `coefficient` is a non-negative resistance per control-volume volume. The
    term represents a sink `-coefficient * u`, so finite-volume linearization
    adds `coefficient * A_u * delta_x_u` to `aP`.
    """

    coefficient: float

    def __post_init__(self) -> None:
        _validate_non_negative_finite("coefficient", self.coefficient)

    def apply(
        self,
        geometry: Geometry,
        density: float,
        fields: Field,
        coeffs: MomentumCoefficients,
    ) -> None:
        fields.validate_against(geometry)
        if self.coefficient == 0.0:
            return

        coeffs.a_p += (
            self.coefficient * geometry.velocity_areas * geometry.dx_values
        )


@dataclass(frozen=True)
class LocalizedLoss:
    """Quadratic local loss sink applied at selected velocity nodes.

    The loss model is linearized with the current velocity field:

        force_loss = 0.5 * K * rho * A_u * |u_old| * u

    and therefore contributes only to `aP`.
    """

    nodes: tuple[int, ...]
    loss_coefficient: float

    def __post_init__(self) -> None:
        _validate_non_negative_finite("loss_coefficient", self.loss_coefficient)
        normalized_nodes = tuple(self._validate_node(node) for node in self.nodes)
        object.__setattr__(self, "nodes", normalized_nodes)

    def apply(
        self,
        geometry: Geometry,
        density: float,
        fields: Field,
        coeffs: MomentumCoefficients,
    ) -> None:
        fields.validate_against(geometry)
        if self.loss_coefficient == 0.0:
            return

        for node in self.nodes:
            if node < 0 or node >= geometry.n_velocity:
                raise ValueError("localized loss node is outside the velocity mesh")
            coeffs.a_p[node] += (
                0.5
                * self.loss_coefficient
                * density
                * geometry.velocity_area(node)
                * abs(fields.u[node])
            )

    def _validate_node(self, node: int) -> int:
        if not isinstance(node, Integral) or isinstance(node, bool):
            raise ValueError("localized loss nodes must be integers")
        return int(node)


def _validate_non_negative_finite(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative finite number")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
