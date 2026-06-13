import numpy as np

from simplecfd.assembly.momentum import MomentumAssembler
from simplecfd.boundary import InletStagnationPressure, OutletFixedPressure
from simplecfd.cases import build_versteeg_example_6_2_case
from simplecfd.fields import Field
from simplecfd.geometry import Geometry
from simplecfd.schemes import Upwind
from simplecfd.simple_loop import SIMPLECCouplingStrategy


def test_pressure_and_non_pressure_sources_reconstruct_momentum_source():
    geometry = Geometry.versteeg_example_6_2()
    fields = Field.versteeg_example_6_2_initial_guess()
    assembler = MomentumAssembler(
        geometry=geometry,
        density=1.0,
        scheme=Upwind(),
        inlet=InletStagnationPressure(stagnation_pressure=10.0),
        outlet=OutletFixedPressure(pressure=0.0),
    )
    coeffs = assembler.assemble(fields)

    pressure_source = assembler.pressure_source(fields)
    source_without_pressure = assembler.source_without_pressure(fields, coeffs)

    np.testing.assert_allclose(pressure_source, [1.125, 0.875, 0.625, 0.375])
    np.testing.assert_allclose(source_without_pressure, [2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(source_without_pressure + pressure_source, coeffs.source)


def test_pseudo_velocities_remove_only_pressure_source():
    geometry = Geometry.versteeg_example_6_2()
    fields = Field.versteeg_example_6_2_initial_guess()
    assembler = MomentumAssembler(
        geometry=geometry,
        density=1.0,
        scheme=Upwind(),
        inlet=InletStagnationPressure(stagnation_pressure=10.0),
        outlet=OutletFixedPressure(pressure=0.0),
    )
    coeffs = assembler.assemble(fields)

    pseudo = assembler.pseudo_velocities(fields, coeffs)

    pressure_source = assembler.pressure_source(fields)
    np.testing.assert_allclose(
        pseudo + pressure_source / coeffs.a_p,
        (
            coeffs.source
            + np.array(
                [
                    coeffs.a_e[0] * fields.u[1],
                    coeffs.a_w[1] * fields.u[0] + coeffs.a_e[1] * fields.u[2],
                    coeffs.a_w[2] * fields.u[1] + coeffs.a_e[2] * fields.u[3],
                    coeffs.a_w[3] * fields.u[2],
                ],
            )
        )
        / coeffs.a_p,
    )


def test_simple_momentum_coefficients_remain_unchanged():
    case = build_versteeg_example_6_2_case(coupling="simple")

    coeffs, system = case.step_solver.assemble_momentum_system()

    np.testing.assert_allclose(
        coeffs.a_w,
        [0.0, 1.015872, 1.028571, 1.066666],
        rtol=1e-5,
    )
    np.testing.assert_allclose(coeffs.a_e, [0.0, 0.0, 0.0, 0.0], rtol=1e-5)
    np.testing.assert_allclose(
        coeffs.a_p,
        [1.42087, 1.028571, 1.066666, 1.0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        coeffs.source,
        [3.125, 0.875, 0.625, 0.375],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        coeffs.d,
        [0.31670, 0.34027, 0.23437, 0.15],
        rtol=1e-4,
    )
    np.testing.assert_allclose(system.rhs, coeffs.source)


def test_simplec_momentum_coefficients_remain_unchanged_before_pressure_correction():
    case = build_versteeg_example_6_2_case(coupling="simplec")

    coeffs, _ = case.step_solver.assemble_momentum_system()
    simplec_coeffs = SIMPLECCouplingStrategy().pressure_correction_momentum_coefficients(
        case.geometry,
        coeffs,
        case.step_solver.velocity_relaxation,
    )

    np.testing.assert_allclose(
        coeffs.a_w,
        [0.0, 1.015872, 1.028571, 1.066666],
        rtol=1e-5,
    )
    np.testing.assert_allclose(coeffs.a_e, [0.0, 0.0, 0.0, 0.0], rtol=1e-5)
    np.testing.assert_allclose(
        coeffs.a_p,
        [1.42087, 1.028571, 1.066666, 1.0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        coeffs.source,
        [3.125, 0.875, 0.625, 0.375],
        rtol=1e-5,
    )
    np.testing.assert_allclose(simplec_coeffs.a_w, coeffs.a_w)
    np.testing.assert_allclose(simplec_coeffs.a_e, coeffs.a_e)
    np.testing.assert_allclose(simplec_coeffs.a_p, coeffs.a_p)
    np.testing.assert_allclose(simplec_coeffs.source, coeffs.source)
    np.testing.assert_allclose(
        simplec_coeffs.d,
        case.geometry.velocity_areas
        / (coeffs.a_p / case.step_solver.velocity_relaxation - coeffs.a_w - coeffs.a_e),
    )
