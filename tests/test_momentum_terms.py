import numpy as np
import pytest

from simplecfd.assembly.momentum import MomentumAssembler
from simplecfd.cases import (
    BoundaryConditions,
    FlowProperties,
    ProblemDefinition,
    SolverControls,
    build_case,
)
from simplecfd.fields import Field
from simplecfd.geometry import Geometry
from simplecfd.momentum_terms import (
    LinearFrictionLoss,
    LocalizedLoss,
    MomentumDiffusion,
)
from simplecfd.schemes import Upwind


def test_momentum_diffusion_adds_internal_neighbor_conductances():
    geometry = Geometry.versteeg_example_6_2()
    fields = Field.versteeg_example_6_2_initial_guess()
    baseline = MomentumAssembler(geometry, density=1.0, scheme=Upwind()).assemble(fields)

    coeffs = MomentumAssembler(
        geometry,
        density=1.0,
        scheme=Upwind(),
        terms=(MomentumDiffusion(diffusivity=0.1),),
    ).assemble(fields)

    np.testing.assert_allclose(coeffs.a_w[1:3], baseline.a_w[1:3] + [0.08, 0.06])
    np.testing.assert_allclose(coeffs.a_e[1:3], baseline.a_e[1:3] + [0.06, 0.04])
    np.testing.assert_allclose(coeffs.a_p[1:3], baseline.a_p[1:3] + [0.14, 0.10])
    np.testing.assert_allclose(coeffs.source, baseline.source)
    np.testing.assert_allclose(
        coeffs.d[1:3],
        geometry.velocity_areas[1:3] / coeffs.a_p[1:3],
    )


def test_momentum_diffusion_uses_nonuniform_velocity_spacing():
    geometry = Geometry.from_pressure_areas(
        pressure_areas=np.array([1.0, 0.8, 0.5, 0.3]),
        dx=np.array([0.1, 0.2, 0.4]),
    )
    fields = Field.initial_nozzle_guess(
        geometry=geometry,
        density=1.0,
        mass_flow_guess=0.5,
        inlet_pressure=4.0,
        outlet_pressure=1.0,
    )
    baseline = MomentumAssembler(geometry, density=1.0, scheme=Upwind()).assemble(fields)

    coeffs = MomentumAssembler(
        geometry,
        density=1.0,
        scheme=Upwind(),
        terms=(MomentumDiffusion(diffusivity=0.2),),
    ).assemble(fields)

    np.testing.assert_allclose(coeffs.a_w[1], baseline.a_w[1] + 0.2 * 0.8 / 0.15)
    np.testing.assert_allclose(coeffs.a_e[1], baseline.a_e[1] + 0.2 * 0.5 / 0.3)
    np.testing.assert_allclose(
        coeffs.a_p[1],
        baseline.a_p[1] + 0.2 * 0.8 / 0.15 + 0.2 * 0.5 / 0.3,
    )


def test_linear_friction_loss_adds_distributed_diagonal_sink():
    geometry = Geometry.versteeg_example_6_2()
    fields = Field.versteeg_example_6_2_initial_guess()
    baseline = MomentumAssembler(geometry, density=1.0, scheme=Upwind()).assemble(fields)

    coeffs = MomentumAssembler(
        geometry,
        density=1.0,
        scheme=Upwind(),
        terms=(LinearFrictionLoss(coefficient=2.0),),
    ).assemble(fields)

    np.testing.assert_allclose(coeffs.a_p, baseline.a_p + geometry.velocity_areas)
    np.testing.assert_allclose(coeffs.a_w, baseline.a_w)
    np.testing.assert_allclose(coeffs.a_e, baseline.a_e)
    np.testing.assert_allclose(coeffs.source, baseline.source)
    np.testing.assert_allclose(coeffs.d, geometry.velocity_areas / coeffs.a_p)


def test_linear_friction_loss_uses_nonuniform_control_volume_widths():
    geometry = Geometry.from_pressure_areas(
        pressure_areas=np.array([1.0, 0.8, 0.5, 0.3]),
        dx=np.array([0.1, 0.2, 0.4]),
    )
    fields = Field.initial_nozzle_guess(
        geometry=geometry,
        density=1.0,
        mass_flow_guess=0.5,
        inlet_pressure=4.0,
        outlet_pressure=1.0,
    )
    baseline = MomentumAssembler(geometry, density=1.0, scheme=Upwind()).assemble(fields)

    coeffs = MomentumAssembler(
        geometry,
        density=1.0,
        scheme=Upwind(),
        terms=(LinearFrictionLoss(coefficient=3.0),),
    ).assemble(fields)

    np.testing.assert_allclose(
        coeffs.a_p,
        baseline.a_p + 3.0 * geometry.velocity_areas * geometry.dx_values,
    )


def test_localized_loss_adds_quadratic_diagonal_sink_at_selected_nodes():
    geometry = Geometry.versteeg_example_6_2()
    fields = Field.versteeg_example_6_2_initial_guess()
    baseline = MomentumAssembler(geometry, density=1.0, scheme=Upwind()).assemble(fields)

    coeffs = MomentumAssembler(
        geometry,
        density=1.0,
        scheme=Upwind(),
        terms=(LocalizedLoss(nodes=(2,), loss_coefficient=3.0),),
    ).assemble(fields)

    expected_a_p = baseline.a_p.copy()
    expected_a_p[2] += 1.5
    np.testing.assert_allclose(coeffs.a_p, expected_a_p)
    np.testing.assert_allclose(
        coeffs.d[1:3],
        geometry.velocity_areas[1:3] / coeffs.a_p[1:3],
    )


def test_problem_definition_wires_momentum_terms_into_case_assembler():
    geometry = Geometry.versteeg_example_6_2()
    terms = [LinearFrictionLoss(coefficient=0.25)]
    problem = ProblemDefinition(
        geometry=geometry,
        initial_field=Field.versteeg_example_6_2_initial_guess(),
        properties=FlowProperties(),
        boundaries=BoundaryConditions(),
        scheme=Upwind(),
        controls=SolverControls(),
        momentum_terms=terms,
    )

    case = build_case(problem)

    assert problem.momentum_terms == tuple(terms)
    assert case.momentum_asm.terms == tuple(terms)


@pytest.mark.parametrize(
    "term",
    [
        MomentumDiffusion(diffusivity=0.0),
        LinearFrictionLoss(coefficient=0.0),
        LocalizedLoss(nodes=(1,), loss_coefficient=0.0),
    ],
)
def test_zero_strength_momentum_terms_preserve_baseline_coefficients(term):
    geometry = Geometry.versteeg_example_6_2()
    fields = Field.versteeg_example_6_2_initial_guess()
    baseline = MomentumAssembler(geometry, density=1.0, scheme=Upwind()).assemble(fields)

    coeffs = MomentumAssembler(
        geometry,
        density=1.0,
        scheme=Upwind(),
        terms=(term,),
    ).assemble(fields)

    np.testing.assert_allclose(coeffs.a_w, baseline.a_w)
    np.testing.assert_allclose(coeffs.a_e, baseline.a_e)
    np.testing.assert_allclose(coeffs.a_p, baseline.a_p)
    np.testing.assert_allclose(coeffs.source, baseline.source)
    np.testing.assert_allclose(coeffs.d, baseline.d)
