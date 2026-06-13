import numpy as np

from simplecfd.assembly import MomentumAssembler, PressureCorrectionAssembler
from simplecfd.boundary import InletStagnationPressure, OutletFixedPressure
from simplecfd.cases import build_case_by_name
from simplecfd.coefficients import LinearSystem
from simplecfd.fields import Field
from simplecfd.geometry import Geometry
from simplecfd.linalg import tdma
from simplecfd.schemes import Upwind


def dense_tridiagonal(system: LinearSystem) -> np.ndarray:
    matrix = np.diag(system.diagonal)
    matrix += np.diag(system.lower[1:], k=-1)
    matrix += np.diag(system.upper[:-1], k=1)
    return matrix


def constant_area_uniform_state() -> tuple[Geometry, Field]:
    geometry = Geometry(
        pressure_areas=np.ones(5),
        velocity_areas=np.ones(4),
        dx=0.25,
    )
    fields = Field(
        p=np.linspace(4.0, 1.0, geometry.n_pressure),
        u=np.full(geometry.n_velocity, 2.0),
        p_prime=np.zeros(geometry.n_pressure),
    )
    return geometry, fields


def versteeg_starred_state():
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


def test_internal_upwind_momentum_coefficients_satisfy_finite_volume_balance():
    geometry = Geometry(
        pressure_areas=np.array([1.4, 1.2, 1.0, 0.8]),
        velocity_areas=np.array([1.3, 1.1, 0.9]),
        dx=0.5,
    )
    fields = Field(
        p=np.array([8.0, 6.0, 3.5, 1.0]),
        u=np.array([1.0, 1.5, 2.0]),
        p_prime=np.zeros(geometry.n_pressure),
    )

    coeffs = MomentumAssembler(geometry, density=1.2, scheme=Upwind()).assemble(fields)

    west_flux = 1.2 * geometry.pressure_area(1) * 0.5 * (fields.u[0] + fields.u[1])
    east_flux = 1.2 * geometry.pressure_area(2) * 0.5 * (fields.u[1] + fields.u[2])
    pressure_source = (fields.p[1] - fields.p[2]) * geometry.velocity_area(1)

    np.testing.assert_allclose(coeffs.f_w[1], west_flux)
    np.testing.assert_allclose(coeffs.f_e[1], east_flux)
    np.testing.assert_allclose(coeffs.a_w[1], max(west_flux, 0.0))
    np.testing.assert_allclose(coeffs.a_e[1], max(-east_flux, 0.0))
    np.testing.assert_allclose(
        coeffs.a_p[1],
        coeffs.a_w[1] + coeffs.a_e[1] + coeffs.f_e[1] - coeffs.f_w[1],
    )
    np.testing.assert_allclose(coeffs.source[1], pressure_source)
    np.testing.assert_allclose(coeffs.d[1], geometry.velocity_area(1) / coeffs.a_p[1])


def test_boundary_sources_match_inlet_stagnation_and_outlet_static_pressure():
    geometry, fields = constant_area_uniform_state()
    inlet = InletStagnationPressure(stagnation_pressure=6.0)
    outlet = OutletFixedPressure(pressure=1.0)

    coeffs = MomentumAssembler(
        geometry=geometry,
        density=1.0,
        scheme=Upwind(),
        inlet=inlet,
        outlet=outlet,
    ).assemble(fields)

    np.testing.assert_allclose(inlet.static_pressure(1.0, fields.u[0]), 4.0)
    np.testing.assert_allclose(coeffs.source[0], (6.0 - fields.p[1]) * 1.0 + 2.0 * 2.0)
    np.testing.assert_allclose(coeffs.source[-1], (fields.p[-2] - 1.0) * 1.0)
    assert coeffs.a_w[0] == 0.0
    assert coeffs.a_e[-1] == 0.0
    assert coeffs.a_p[0] > 0.0
    assert coeffs.a_p[-1] > 0.0


def test_pressure_correction_matrix_has_conservative_internal_rows():
    geometry, fields, momentum = versteeg_starred_state()

    coeffs = PressureCorrectionAssembler(geometry, density=1.0).assemble_coefficients(
        fields,
        momentum,
    )

    np.testing.assert_allclose(coeffs.a_p[1:-1], coeffs.a_w[1:-1] + coeffs.a_e[1:-1])
    assert np.all(coeffs.a_w[1:-1] > 0.0)
    assert np.all(coeffs.a_e[1:-1] > 0.0)
    starred_flux = geometry.velocity_areas * fields.u
    np.testing.assert_allclose(
        np.sum(coeffs.source[1:-1]),
        starred_flux[0] - starred_flux[-1],
        rtol=1e-12,
    )


def test_pressure_correction_solution_enforces_discrete_mass_conservation():
    geometry, fields, momentum = versteeg_starred_state()
    assembler = PressureCorrectionAssembler(geometry, density=1.0)
    p_prime = tdma(assembler.assemble(fields, momentum))

    corrected_velocity = fields.u + momentum.d * (p_prime[:-1] - p_prime[1:])
    corrected_flux = geometry.velocity_areas * corrected_velocity

    np.testing.assert_allclose(np.diff(corrected_flux), 0.0, atol=1e-10)


def test_tdma_matches_dense_solver_for_diagonally_dominant_system():
    solution = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    system = LinearSystem(
        lower=np.array([0.0, -0.4, -0.6, -0.8, -1.0]),
        diagonal=np.array([2.5, 3.0, 3.5, 4.0, 4.5]),
        upper=np.array([-0.3, -0.5, -0.7, -0.9, 0.0]),
        rhs=np.zeros(5),
    )
    system.rhs[:] = dense_tridiagonal(system) @ solution

    np.testing.assert_allclose(tdma(system), solution)
    np.testing.assert_allclose(tdma(system), np.linalg.solve(dense_tridiagonal(system), system.rhs))


def test_momentum_stage_linear_solve_satisfies_assembled_equations():
    case = build_case_by_name(
        "versteeg_6_2",
        velocity_relaxation=1.0,
        pressure_relaxation=1.0,
    )
    momentum_coeffs, momentum_system = case.step_solver.assemble_momentum_system()

    starred_velocity = case.step_solver.solve_linear_system(momentum_system)
    residual = dense_tridiagonal(momentum_system) @ starred_velocity - momentum_system.rhs

    np.testing.assert_allclose(residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        momentum_coeffs.a_p * starred_velocity
        - momentum_coeffs.source
        - np.r_[0.0, momentum_coeffs.a_w[1:] * starred_velocity[:-1]]
        - np.r_[momentum_coeffs.a_e[:-1] * starred_velocity[1:], 0.0],
        0.0,
        atol=1e-12,
    )


def test_constant_area_case_converges_to_bernoulli_velocity_and_uniform_mass_flow():
    case = build_case_by_name(
        "constant_area_1d",
        tolerance=1e-9,
        max_iterations=500,
        pressure_relaxation=0.7,
        velocity_relaxation=0.7,
    )

    result = case.solver.solve()

    expected_velocity = np.sqrt(2.0 * (5.0 - 1.0))
    mass_flow = case.geometry.velocity_areas * case.field.u
    assert result["converged"]
    np.testing.assert_allclose(case.field.u, expected_velocity, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(mass_flow, mass_flow[0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(case.field.p[1:], 1.0, rtol=1e-9, atol=1e-9)
    assert result["continuity_residual"] < 1e-10
    assert result["momentum_residual"] < 1e-8
