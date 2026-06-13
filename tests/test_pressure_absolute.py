import numpy as np
import pytest

from simplecfd.assembly import AbsolutePressureAssembler, MomentumAssembler
from simplecfd.boundary import InletStagnationPressure, OutletFixedPressure
from simplecfd.fields import Field
from simplecfd.geometry import Geometry
from simplecfd.schemes import Upwind


def simpler_pressure_state():
    geometry = Geometry.versteeg_example_6_2()
    fields = Field.versteeg_example_6_2_initial_guess()
    momentum_assembler = MomentumAssembler(
        geometry=geometry,
        density=1.0,
        scheme=Upwind(),
        inlet=InletStagnationPressure(stagnation_pressure=10.0),
        outlet=OutletFixedPressure(pressure=0.0),
    )
    momentum = momentum_assembler.assemble(fields)
    pseudo_velocity = momentum_assembler.pseudo_velocities(fields, momentum)
    return geometry, fields, momentum, pseudo_velocity


@pytest.mark.parametrize(
    "node",
    [1, 2, 3],
    ids=["pressure_node_B", "pressure_node_C", "pressure_node_D"],
)
def test_each_internal_absolute_pressure_node_uses_simpler_coefficients(node):
    geometry, fields, momentum, pseudo_velocity = simpler_pressure_state()
    assembler = AbsolutePressureAssembler(geometry, density=1.0)

    coeffs = assembler.assemble_coefficients(fields, momentum, pseudo_velocity)

    west_velocity_node = node - 1
    east_velocity_node = node
    d_w = geometry.velocity_area(west_velocity_node) / momentum.a_p[west_velocity_node]
    d_e = geometry.velocity_area(east_velocity_node) / momentum.a_p[east_velocity_node]
    expected_a_w = 1.0 * geometry.velocity_area(west_velocity_node) * d_w
    expected_a_e = 1.0 * geometry.velocity_area(east_velocity_node) * d_e

    np.testing.assert_allclose(coeffs.a_w[node], expected_a_w)
    np.testing.assert_allclose(coeffs.a_e[node], expected_a_e)
    np.testing.assert_allclose(coeffs.a_p[node], expected_a_w + expected_a_e)


@pytest.mark.parametrize(
    "node",
    [1, 2, 3],
    ids=["pressure_node_B", "pressure_node_C", "pressure_node_D"],
)
def test_each_internal_absolute_pressure_node_rhs_uses_pseudo_mass_flux(node):
    geometry, fields, momentum, pseudo_velocity = simpler_pressure_state()
    assembler = AbsolutePressureAssembler(geometry, density=1.0)

    coeffs = assembler.assemble_coefficients(fields, momentum, pseudo_velocity)
    system = assembler.assemble(fields, momentum, pseudo_velocity)

    west_velocity_node = node - 1
    east_velocity_node = node
    expected_rhs = (
        1.0
        * geometry.velocity_area(west_velocity_node)
        * pseudo_velocity[west_velocity_node]
    ) - (
        1.0
        * geometry.velocity_area(east_velocity_node)
        * pseudo_velocity[east_velocity_node]
    )

    np.testing.assert_allclose(coeffs.source[node], expected_rhs)
    np.testing.assert_allclose(system.rhs[node], expected_rhs)


def test_absolute_pressure_system_fixes_boundary_pressures():
    geometry, fields, momentum, pseudo_velocity = simpler_pressure_state()

    system = AbsolutePressureAssembler(geometry, density=1.0).assemble(
        fields,
        momentum,
        pseudo_velocity,
    )

    np.testing.assert_allclose(system.diagonal[[0, -1]], [1.0, 1.0])
    np.testing.assert_allclose(system.rhs[[0, -1]], fields.p[[0, -1]])
    np.testing.assert_allclose(system.lower[[0, -1]], [0.0, 0.0])
    np.testing.assert_allclose(system.upper[[0, -1]], [0.0, 0.0])


def test_absolute_pressure_assembler_recomputes_d_from_area_over_ap():
    geometry, fields, momentum, pseudo_velocity = simpler_pressure_state()
    altered_momentum = MomentumAssembler(
        geometry=geometry,
        density=1.0,
        scheme=Upwind(),
        inlet=InletStagnationPressure(stagnation_pressure=10.0),
        outlet=OutletFixedPressure(pressure=0.0),
    ).assemble(fields)
    altered_momentum.d = np.full(geometry.n_velocity, 999.0)
    assembler = AbsolutePressureAssembler(geometry, density=1.0)

    d = assembler.pressure_velocity_coefficients(altered_momentum)
    coeffs = assembler.assemble_coefficients(fields, altered_momentum, pseudo_velocity)

    np.testing.assert_allclose(d, geometry.velocity_areas / momentum.a_p)
    np.testing.assert_allclose(
        coeffs.a_w[1:-1],
        geometry.velocity_areas[:-1] * d[:-1],
    )
    np.testing.assert_allclose(
        coeffs.a_e[1:-1],
        geometry.velocity_areas[1:] * d[1:],
    )


def test_pressure_correction_assembler_remains_existing_separate_equation():
    from simplecfd.assembly import PressureCorrectionAssembler

    assert PressureCorrectionAssembler is not AbsolutePressureAssembler
