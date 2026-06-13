import numpy as np
import pytest

from simplecfd.boundary import InletStagnationPressure, OutletFixedPressure
from simplecfd.cases import (
    aggressive_contraction_1d_problem,
    BoundaryConditions,
    CaseRegistry,
    constant_area_1d_problem,
    expansion_1d_problem,
    fine_mesh_nozzle_1d_problem,
    FlowProperties,
    nearly_constant_area_1d_problem,
    poor_initial_guess_1d_problem,
    ProblemDefinition,
    SolverControls,
    build_case,
    build_case_by_name,
    build_coupling_strategy,
    build_problem_by_name,
    build_versteeg_example_6_2_case,
    linear_nozzle_1d_problem,
    list_available_cases,
    smooth_linear_nozzle_1d_problem,
    strong_contraction_1d_problem,
    versteeg_example_6_2_problem,
)
from simplecfd.fields import Field
from simplecfd.geometry import Geometry
from simplecfd.schemes import CentralDifference, Upwind
from simplecfd.simple_loop import (
    SIMPLECouplingStrategy,
    SIMPLECCouplingStrategy,
    SIMPLERCouplingStrategy,
)


class TrackingCouplingStrategy:
    def relax_velocity(
        self,
        old_velocity,
        starred_velocity,
        velocity_relaxation,
    ):
        return old_velocity + velocity_relaxation * (starred_velocity - old_velocity)

    def apply_correction(
        self,
        geometry,
        field,
        momentum_coeffs,
        p_prime,
        pressure_relaxation,
    ):
        field.p_prime = p_prime.copy()
        self.correct_pressure(field, p_prime, pressure_relaxation)
        self.correct_velocity(geometry, field, momentum_coeffs, p_prime)

    def correct_pressure(self, field, p_prime, pressure_relaxation):
        field.p += pressure_relaxation * p_prime

    def correct_velocity(self, geometry, field, momentum_coeffs, p_prime):
        for i in range(geometry.n_velocity):
            field.u[i] += momentum_coeffs.d[i] * (p_prime[i] - p_prime[i + 1])


REGISTERED_1D_CASES = {
    "constant_area_1d": {
        "factory": constant_area_1d_problem,
        "pressure_areas": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "inlet_pressure": 5.0,
        "outlet_pressure": 1.0,
        "max_iterations": 200,
        "n_pressure": 6,
    },
    "smooth_linear_nozzle_1d": {
        "factory": smooth_linear_nozzle_1d_problem,
        "pressure_areas": np.array([1.0, 0.94, 0.88, 0.82, 0.76, 0.7]),
        "inlet_pressure": 5.0,
        "outlet_pressure": 1.0,
        "max_iterations": 200,
        "n_pressure": 6,
    },
    "aggressive_contraction_1d": {
        "factory": aggressive_contraction_1d_problem,
        "pressure_areas": np.array([1.0, 0.85, 0.7, 0.55, 0.4, 0.25]),
        "inlet_pressure": 8.0,
        "outlet_pressure": 1.0,
        "max_iterations": 300,
        "n_pressure": 6,
    },
    "strong_contraction_1d": {
        "factory": strong_contraction_1d_problem,
        "pressure_areas": np.linspace(1.0, 0.15, 8),
        "inlet_pressure": 10.0,
        "outlet_pressure": 1.0,
        "max_iterations": 500,
        "n_pressure": 8,
    },
    "expansion_1d": {
        "factory": expansion_1d_problem,
        "pressure_areas": np.linspace(0.45, 1.1, 7),
        "inlet_pressure": 6.0,
        "outlet_pressure": 1.0,
        "max_iterations": 400,
        "n_pressure": 7,
    },
    "nearly_constant_area_1d": {
        "factory": nearly_constant_area_1d_problem,
        "pressure_areas": np.linspace(1.0, 0.97, 7),
        "inlet_pressure": 5.0,
        "outlet_pressure": 1.0,
        "max_iterations": 250,
        "n_pressure": 7,
    },
    "fine_mesh_nozzle_1d": {
        "factory": fine_mesh_nozzle_1d_problem,
        "pressure_areas": np.linspace(1.0, 0.75, 12),
        "inlet_pressure": 5.0,
        "outlet_pressure": 1.0,
        "max_iterations": 1000,
        "n_pressure": 12,
    },
    "poor_initial_guess_1d": {
        "factory": poor_initial_guess_1d_problem,
        "pressure_areas": np.linspace(1.0, 0.55, 6),
        "inlet_pressure": 6.0,
        "outlet_pressure": 1.0,
        "max_iterations": 500,
        "n_pressure": 6,
    },
}


def test_problem_definition_builds_solver_case_without_versteeg_factory():
    geometry = Geometry.linear_nozzle(
        inlet_area=1.0,
        outlet_area=0.5,
        n_pressure=4,
        dx=0.25,
    )
    initial_field = Field.initial_nozzle_guess(
        geometry=geometry,
        density=1.2,
        mass_flow_guess=0.6,
        inlet_pressure=5.0,
        outlet_pressure=1.0,
    )
    scheme = CentralDifference()
    problem = ProblemDefinition(
        geometry=geometry,
        initial_field=initial_field,
        properties=FlowProperties(density=1.2),
        boundaries=BoundaryConditions(
            inlet=InletStagnationPressure(stagnation_pressure=5.0),
            outlet=OutletFixedPressure(pressure=1.0),
        ),
        scheme=scheme,
        controls=SolverControls(
            tolerance=1e-6,
            max_iterations=25,
            pressure_relaxation=0.4,
            velocity_relaxation=0.5,
        ),
    )

    case = build_case(problem)

    assert case.definition is problem
    assert case.geometry is geometry
    assert case.momentum_asm.geometry is geometry
    assert case.momentum_asm.density == 1.2
    assert case.momentum_asm.scheme is scheme
    assert case.momentum_asm.inlet is problem.boundaries.inlet
    assert case.momentum_asm.outlet is problem.boundaries.outlet
    assert case.pressure_correction_asm.density == 1.2
    assert case.step_solver.pressure_relaxation == 0.4
    assert case.step_solver.velocity_relaxation == 0.5
    assert case.step_solver.coupling_strategy is problem.coupling_strategy
    assert case.solver.tolerance == 1e-6
    assert case.solver.max_iterations == 25


def test_problem_definition_uses_simple_coupling_strategy_by_default():
    problem = versteeg_example_6_2_problem()

    assert isinstance(problem.coupling_strategy, SIMPLECouplingStrategy)
    assert not isinstance(problem.coupling_strategy, SIMPLECCouplingStrategy)


def test_coupling_strategy_builder_resolves_simple_simplec_and_simpler():
    assert isinstance(build_coupling_strategy("simple"), SIMPLECouplingStrategy)
    assert isinstance(build_coupling_strategy("simplec"), SIMPLECCouplingStrategy)
    assert isinstance(build_coupling_strategy("simpler"), SIMPLERCouplingStrategy)


def test_coupling_strategy_builder_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown pressure-velocity coupling"):
        build_coupling_strategy("piso")


def test_build_case_copies_initial_fields_so_definitions_are_reusable():
    geometry = Geometry.versteeg_example_6_2()
    initial_field = Field.versteeg_example_6_2_initial_guess()
    problem = ProblemDefinition(
        geometry=geometry,
        initial_field=initial_field,
        properties=FlowProperties(density=1.0),
        boundaries=BoundaryConditions(),
        scheme=Upwind(),
        controls=SolverControls(),
    )

    case = build_case(problem)
    case.field.p += 1.0
    case.field.u += 1.0
    second_case = build_case(problem)

    np.testing.assert_allclose(initial_field.p, Field.versteeg_example_6_2_initial_guess().p)
    np.testing.assert_allclose(initial_field.u, Field.versteeg_example_6_2_initial_guess().u)
    np.testing.assert_allclose(second_case.field.p, Field.versteeg_example_6_2_initial_guess().p)
    np.testing.assert_allclose(second_case.field.u, Field.versteeg_example_6_2_initial_guess().u)
    assert case.field is not initial_field
    assert second_case.field is not initial_field
    assert second_case.field is not case.field


def test_versteeg_factory_remains_golden_case_on_problem_definition_layer():
    scheme = Upwind()
    problem = versteeg_example_6_2_problem(
        density=1.1,
        tolerance=2e-5,
        max_iterations=33,
        pressure_relaxation=0.42,
        velocity_relaxation=0.62,
        scheme=scheme,
    )
    case = build_versteeg_example_6_2_case(
        density=1.1,
        tolerance=2e-5,
        max_iterations=33,
        pressure_relaxation=0.42,
        velocity_relaxation=0.62,
        scheme=scheme,
    )

    assert problem.properties.density == 1.1
    assert problem.scheme is scheme
    assert isinstance(problem.boundaries.inlet, InletStagnationPressure)
    assert isinstance(problem.boundaries.outlet, OutletFixedPressure)
    assert case.definition.properties.density == 1.1
    assert case.definition.scheme is scheme
    assert case.solver.tolerance == 2e-5
    assert case.solver.max_iterations == 33
    assert case.step_solver.pressure_relaxation == 0.42
    assert case.step_solver.velocity_relaxation == 0.62


def test_default_case_registry_lists_versteeg_case():
    assert "versteeg_6_2" in list_available_cases()
    assert "linear_nozzle_1d" in list_available_cases()
    assert "constant_area_1d" in list_available_cases()
    assert "smooth_linear_nozzle_1d" in list_available_cases()
    assert "aggressive_contraction_1d" in list_available_cases()
    assert "strong_contraction_1d" in list_available_cases()
    assert "expansion_1d" in list_available_cases()
    assert "nearly_constant_area_1d" in list_available_cases()
    assert "fine_mesh_nozzle_1d" in list_available_cases()
    assert "poor_initial_guess_1d" in list_available_cases()


def test_linear_nozzle_1d_problem_builds_registered_second_case():
    problem = linear_nozzle_1d_problem(
        density=1.2,
        n_pressure=6,
        inlet_area=1.1,
        outlet_area=0.6,
        coupling="simplec",
    )
    case = build_case_by_name(
        "linear_nozzle_1d",
        density=1.2,
        n_pressure=6,
        inlet_area=1.1,
        outlet_area=0.6,
        coupling="simplec",
    )

    assert problem.geometry.n_pressure == 6
    assert problem.geometry.n_velocity == 5
    assert case.geometry.n_pressure == 6
    assert case.field.p.shape == (6,)
    assert case.field.u.shape == (5,)
    assert isinstance(case.step_solver.coupling_strategy, SIMPLECCouplingStrategy)


def test_linear_nozzle_1d_problem_accepts_nonuniform_spacing():
    dx = np.array([0.1, 0.15, 0.25, 0.3, 0.4])
    case = build_case_by_name(
        "linear_nozzle_1d",
        n_pressure=6,
        inlet_area=1.0,
        outlet_area=0.6,
        dx=dx,
        max_iterations=5,
    )

    np.testing.assert_allclose(case.geometry.dx_values, dx)
    np.testing.assert_allclose(case.geometry.pressure_positions, [0.0, 0.1, 0.25, 0.5, 0.8, 1.2])
    np.testing.assert_allclose(case.geometry.velocity_positions, [0.05, 0.175, 0.375, 0.65, 1.0])
    momentum_coeffs, momentum_system = case.step_solver.assemble_momentum_system()

    assert momentum_system.diagonal.shape == (5,)
    assert np.all(np.isfinite(momentum_coeffs.a_p))
    assert np.all(np.isfinite(momentum_system.rhs))


@pytest.mark.parametrize("case_name", REGISTERED_1D_CASES)
def test_registered_1d_cases_build_problem_definitions(case_name):
    expectation = REGISTERED_1D_CASES[case_name]

    problem = expectation["factory"]()
    case = build_case_by_name(case_name)

    assert isinstance(problem, ProblemDefinition)
    assert isinstance(case.definition, ProblemDefinition)
    assert case.geometry.n_pressure == expectation["n_pressure"]
    assert case.geometry.n_velocity == expectation["n_pressure"] - 1
    np.testing.assert_allclose(case.geometry.pressure_areas, expectation["pressure_areas"])
    np.testing.assert_allclose(
        case.geometry.velocity_areas,
        0.5 * (expectation["pressure_areas"][:-1] + expectation["pressure_areas"][1:]),
    )
    np.testing.assert_allclose(
        case.field.p,
        np.linspace(
            expectation["inlet_pressure"],
            expectation["outlet_pressure"],
            case.geometry.n_pressure,
        ),
    )
    assert case.solver.max_iterations == expectation["max_iterations"]
    assert isinstance(case.definition.boundaries.inlet, InletStagnationPressure)
    assert isinstance(case.definition.boundaries.outlet, OutletFixedPressure)


@pytest.mark.parametrize("case_name", REGISTERED_1D_CASES)
def test_registered_1d_cases_assemble_finite_systems(case_name):
    case = build_case_by_name(case_name)

    momentum_coeffs, momentum_system = case.step_solver.assemble_momentum_system()
    pressure_system = case.step_solver.assemble_pressure_correction_system(momentum_coeffs)

    assert momentum_system.diagonal.shape == (case.geometry.n_velocity,)
    assert pressure_system.diagonal.shape == (case.geometry.n_pressure,)
    assert np.all(np.isfinite(momentum_coeffs.a_p))
    assert np.all(np.isfinite(momentum_coeffs.a_w))
    assert np.all(np.isfinite(momentum_coeffs.a_e))
    assert np.all(np.isfinite(momentum_coeffs.source))
    assert np.all(np.isfinite(momentum_coeffs.d))
    assert np.all(np.isfinite(momentum_system.lower))
    assert np.all(np.isfinite(momentum_system.diagonal))
    assert np.all(np.isfinite(momentum_system.upper))
    assert np.all(np.isfinite(momentum_system.rhs))
    assert np.all(np.isfinite(pressure_system.lower))
    assert np.all(np.isfinite(pressure_system.diagonal))
    assert np.all(np.isfinite(pressure_system.upper))
    assert np.all(np.isfinite(pressure_system.rhs))


@pytest.mark.parametrize("case_name", REGISTERED_1D_CASES)
def test_registered_1d_cases_run_stably(case_name):
    case = build_case_by_name(case_name)

    result = case.solver.solve()

    assert result["converged"] is True
    assert result["iterations"] <= case.solver.max_iterations
    assert result["residual"] < case.solver.tolerance
    assert np.all(np.isfinite(result["residual_history"]))
    assert np.all(np.isfinite(result["continuity_residual_vector"]))
    assert np.all(np.isfinite(result["momentum_residual_vector"]))
    assert np.all(np.isfinite(case.field.p))
    assert np.all(np.isfinite(case.field.u))
    assert np.max(np.abs(case.field.p)) < 1e3
    assert np.max(np.abs(case.field.u)) < 1e3


def test_case_registry_builds_versteeg_problem_by_name():
    scheme = Upwind()

    problem = build_problem_by_name(
        "versteeg_6_2",
        density=1.1,
        tolerance=2e-5,
        max_iterations=33,
        pressure_relaxation=0.42,
        velocity_relaxation=0.62,
        scheme=scheme,
    )

    assert problem.properties.density == 1.1
    assert problem.scheme is scheme
    assert problem.controls.tolerance == 2e-5
    assert problem.controls.max_iterations == 33
    assert problem.controls.pressure_relaxation == 0.42
    assert problem.controls.velocity_relaxation == 0.62
    assert isinstance(problem.coupling_strategy, SIMPLECouplingStrategy)


def test_case_registry_builds_versteeg_problem_with_simplec_coupling():
    problem = build_problem_by_name("versteeg_6_2", coupling="simplec")
    case = build_case_by_name("versteeg_6_2", coupling="simplec")

    assert isinstance(problem.coupling_strategy, SIMPLECCouplingStrategy)
    assert isinstance(case.step_solver.coupling_strategy, SIMPLECCouplingStrategy)


def test_case_registry_builds_versteeg_problem_with_simpler_coupling():
    problem = build_problem_by_name("versteeg_6_2", coupling="simpler")
    case = build_case_by_name("versteeg_6_2", coupling="simpler")

    assert isinstance(problem.coupling_strategy, SIMPLERCouplingStrategy)
    assert isinstance(case.step_solver.coupling_strategy, SIMPLERCouplingStrategy)


def test_build_case_by_name_runs_versteeg_with_simpler_coupling():
    case = build_case_by_name("versteeg_6_2", coupling="simpler", max_iterations=3)

    p_prime = case.step_solver.run_single_iteration()

    assert p_prime.shape == (case.geometry.n_pressure,)
    assert np.all(np.isfinite(case.field.p))
    assert np.all(np.isfinite(case.field.u))
    assert np.all(np.isfinite(case.field.p_prime))


def test_case_registry_builds_solver_case_by_name():
    case = build_case_by_name(
        "versteeg_6_2",
        density=1.1,
        max_iterations=33,
    )

    assert case.definition.properties.density == 1.1
    assert case.solver.max_iterations == 33
    assert isinstance(case.step_solver.coupling_strategy, SIMPLECouplingStrategy)


def test_case_registry_overrides_coupling_strategy_without_changing_case_definition_data():
    strategy = TrackingCouplingStrategy()
    default_problem = build_problem_by_name("versteeg_6_2")
    custom_problem = build_problem_by_name("versteeg_6_2", coupling_strategy=strategy)
    custom_case = build_case_by_name("versteeg_6_2", coupling_strategy=strategy)

    assert custom_problem.coupling_strategy is strategy
    assert custom_case.step_solver.coupling_strategy is strategy
    np.testing.assert_allclose(
        custom_problem.geometry.pressure_areas,
        default_problem.geometry.pressure_areas,
    )
    np.testing.assert_allclose(
        custom_problem.geometry.velocity_areas,
        default_problem.geometry.velocity_areas,
    )
    np.testing.assert_allclose(custom_problem.initial_field.p, default_problem.initial_field.p)
    np.testing.assert_allclose(custom_problem.initial_field.u, default_problem.initial_field.u)
    assert type(custom_problem.boundaries.inlet) is type(default_problem.boundaries.inlet)
    assert type(custom_problem.boundaries.outlet) is type(default_problem.boundaries.outlet)


def test_versteeg_facade_uses_registered_case_compatibly():
    case = build_versteeg_example_6_2_case(
        density=1.1,
        max_iterations=33,
    )

    assert case.definition.properties.density == 1.1
    assert case.solver.max_iterations == 33


def test_versteeg_facade_accepts_coupling_strategy_override():
    strategy = TrackingCouplingStrategy()

    case = build_versteeg_example_6_2_case(coupling_strategy=strategy)

    assert case.definition.coupling_strategy is strategy
    assert case.step_solver.coupling_strategy is strategy


def test_case_registry_registers_custom_problem_factory():
    registry = CaseRegistry(problem_factories={})
    registry.register("custom_nozzle", versteeg_example_6_2_problem)

    assert registry.list_cases() == ("custom_nozzle",)
    assert registry.build_case("custom_nozzle").geometry.n_pressure == 5


def test_case_registry_rejects_duplicate_names_with_clear_error():
    registry = CaseRegistry(problem_factories={})
    registry.register("custom_nozzle", versteeg_example_6_2_problem)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("custom_nozzle", versteeg_example_6_2_problem)


def test_case_registry_rejects_unknown_case_with_available_names():
    registry = CaseRegistry(problem_factories={})
    registry.register("custom_nozzle", versteeg_example_6_2_problem)

    with pytest.raises(ValueError, match="unknown case 'missing'.*custom_nozzle"):
        registry.build_case("missing")


def test_case_registry_rejects_invalid_factory_result_with_clear_error():
    registry = CaseRegistry(problem_factories={})
    registry.register("bad_case", lambda: object())

    with pytest.raises(ValueError, match="ProblemDefinition"):
        registry.build_case("bad_case")


@pytest.mark.parametrize("density", [0.0, -1.0, float("inf")])
def test_flow_properties_reject_invalid_density_with_clear_error(density):
    with pytest.raises(ValueError, match="density"):
        FlowProperties(density=density)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan")])
def test_solver_controls_reject_invalid_tolerance_with_clear_error(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        SolverControls(tolerance=tolerance)


@pytest.mark.parametrize("max_iterations", [0, -1, 1.5])
def test_solver_controls_reject_invalid_max_iterations_with_clear_error(max_iterations):
    with pytest.raises(ValueError, match="max_iterations"):
        SolverControls(max_iterations=max_iterations)


@pytest.mark.parametrize("pressure_relaxation", [0.0, -0.1, 1.1])
def test_solver_controls_reject_invalid_pressure_relaxation_with_clear_error(
    pressure_relaxation,
):
    with pytest.raises(ValueError, match="pressure_relaxation"):
        SolverControls(pressure_relaxation=pressure_relaxation)


@pytest.mark.parametrize("velocity_relaxation", [0.0, -0.1, 1.1])
def test_solver_controls_reject_invalid_velocity_relaxation_with_clear_error(
    velocity_relaxation,
):
    with pytest.raises(ValueError, match="velocity_relaxation"):
        SolverControls(velocity_relaxation=velocity_relaxation)


def test_problem_definition_rejects_missing_required_piece_with_clear_error():
    with pytest.raises(ValueError, match="scheme"):
        ProblemDefinition(
            geometry=Geometry.versteeg_example_6_2(),
            initial_field=Field.versteeg_example_6_2_initial_guess(),
            properties=FlowProperties(),
            boundaries=BoundaryConditions(),
            scheme=None,
            controls=SolverControls(),
        )


def test_problem_definition_rejects_missing_coupling_strategy_with_clear_error():
    with pytest.raises(ValueError, match="coupling_strategy"):
        ProblemDefinition(
            geometry=Geometry.versteeg_example_6_2(),
            initial_field=Field.versteeg_example_6_2_initial_guess(),
            properties=FlowProperties(),
            boundaries=BoundaryConditions(),
            scheme=Upwind(),
            controls=SolverControls(),
            coupling_strategy=None,
        )


def test_problem_definition_rejects_initial_field_that_does_not_match_geometry():
    with pytest.raises(ValueError, match="geometry.n_pressure"):
        ProblemDefinition(
            geometry=Geometry.linear_nozzle(
                inlet_area=1.0,
                outlet_area=0.5,
                n_pressure=4,
                dx=0.25,
            ),
            initial_field=Field.versteeg_example_6_2_initial_guess(),
            properties=FlowProperties(),
            boundaries=BoundaryConditions(),
            scheme=Upwind(),
            controls=SolverControls(),
        )


def test_build_case_rejects_missing_problem_with_clear_error():
    with pytest.raises(ValueError, match="problem"):
        build_case(None)


def test_versteeg_case_builder_rejects_invalid_controls_with_clear_error():
    with pytest.raises(ValueError, match="max_iterations"):
        build_versteeg_example_6_2_case(max_iterations=0)
