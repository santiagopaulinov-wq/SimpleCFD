import numpy as np
import pytest

from simplecfd.assembly import MomentumAssembler, PressureCorrectionAssembler
from simplecfd.boundary import InletStagnationPressure, OutletFixedPressure
from simplecfd.fields import Field
from simplecfd.geometry import Geometry
from simplecfd.schemes import Upwind
from simplecfd.simple_loop import SIMPLECCouplingStrategy


def solved_momentum_state():
    geometry = Geometry.versteeg_example_6_2()
    initial_fields = Field.versteeg_example_6_2_initial_guess()
    momentum = MomentumAssembler(
        geometry=geometry,
        density=1.0,
        scheme=Upwind(),
        inlet=InletStagnationPressure(stagnation_pressure=10.0),
        outlet=OutletFixedPressure(pressure=0.0),
    ).assemble(initial_fields)
    fields = Field(
        p=initial_fields.p,
        u=np.array([2.19935, 3.02289, 3.50087, 4.10926]),
        p_prime=np.zeros(geometry.n_pressure),
    )
    return geometry, fields, momentum


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (
            1,
            {
                "a_w": 0.14251,
                "a_e": 0.11909,
                "a_p": 0.26161,
                "source": -0.06830,
            },
        ),
        (
            2,
            {
                "a_w": 0.11909,
                "a_e": 0.05859,
                "a_p": 0.17769,
                "source": 0.18279,
            },
        ),
        (
            3,
            {
                "a_w": 0.05859,
                "a_e": 0.02250,
                "a_p": 0.08109,
                "source": 0.25882,
            },
        ),
    ],
    ids=["pressure_node_B", "pressure_node_C", "pressure_node_D"],
)
def test_each_internal_pressure_correction_node_matches_versteeg_example_6_2(
    node,
    expected,
):
    geometry, fields, momentum = solved_momentum_state()
    coeffs = PressureCorrectionAssembler(geometry, density=1.0).assemble_coefficients(
        fields,
        momentum,
    )

    np.testing.assert_allclose(coeffs.a_w[node], expected["a_w"], rtol=1e-4)
    np.testing.assert_allclose(coeffs.a_e[node], expected["a_e"], rtol=1e-4)
    np.testing.assert_allclose(coeffs.a_p[node], expected["a_p"], rtol=1e-4)
    np.testing.assert_allclose(coeffs.source[node], expected["source"], rtol=1e-4)


def test_pressure_correction_linear_system_keeps_boundary_corrections_fixed():
    geometry, fields, momentum = solved_momentum_state()
    system = PressureCorrectionAssembler(geometry, density=1.0).assemble(fields, momentum)

    np.testing.assert_allclose(system.diagonal[[0, -1]], [1.0, 1.0])
    np.testing.assert_allclose(system.rhs[[0, -1]], [0.0, 0.0])
    np.testing.assert_allclose(system.lower[1], -0.14251, rtol=1e-4)
    np.testing.assert_allclose(system.upper[1], -0.11909, rtol=1e-4)


@pytest.mark.parametrize(
    "node",
    [0, -1],
    ids=["pressure_boundary_A", "pressure_boundary_E"],
)
def test_each_pressure_correction_boundary_is_fixed(node):
    geometry, fields, momentum = solved_momentum_state()
    assembler = PressureCorrectionAssembler(geometry, density=1.0)
    coeffs = assembler.assemble_coefficients(fields, momentum)
    system = assembler.assemble(fields, momentum)

    np.testing.assert_allclose(coeffs.a_w[node], 0.0)
    np.testing.assert_allclose(coeffs.a_e[node], 0.0)
    np.testing.assert_allclose(coeffs.a_p[node], 1.0)
    np.testing.assert_allclose(coeffs.source[node], 0.0)
    np.testing.assert_allclose(system.lower[node], 0.0)
    np.testing.assert_allclose(system.diagonal[node], 1.0)
    np.testing.assert_allclose(system.upper[node], 0.0)
    np.testing.assert_allclose(system.rhs[node], 0.0)


def test_pressure_correction_continuity_residual_matches_linear_system_rhs():
    geometry, fields, momentum = solved_momentum_state()
    assembler = PressureCorrectionAssembler(geometry, density=1.0)
    system = assembler.assemble(fields, momentum)

    np.testing.assert_allclose(
        assembler.continuity_residual(fields, momentum),
        system.rhs,
    )


def test_simplec_pressure_correction_equation_uses_modified_d_coefficients():
    geometry, fields, momentum = solved_momentum_state()
    velocity_relaxation = 0.7
    simplec_momentum = SIMPLECCouplingStrategy().pressure_correction_momentum_coefficients(
        geometry,
        momentum,
        velocity_relaxation,
    )
    coeffs = PressureCorrectionAssembler(geometry, density=1.0).assemble_coefficients(
        fields,
        simplec_momentum,
    )

    expected_d = geometry.velocity_areas / (
        momentum.a_p / velocity_relaxation - momentum.a_w - momentum.a_e
    )

    np.testing.assert_allclose(simplec_momentum.d, expected_d)
    np.testing.assert_allclose(coeffs.a_w[1:4], geometry.velocity_areas[:3] * expected_d[:3])
    np.testing.assert_allclose(coeffs.a_e[1:4], geometry.velocity_areas[1:4] * expected_d[1:4])
    np.testing.assert_allclose(coeffs.a_p[1:4], coeffs.a_w[1:4] + coeffs.a_e[1:4])
    np.testing.assert_allclose(
        coeffs.source[1:4],
        [-0.06830, 0.18279, 0.25882],
        rtol=1e-4,
    )
