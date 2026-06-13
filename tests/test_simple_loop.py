import numpy as np
import pytest

from simplecfd.cases import build_versteeg_example_6_2_case
from simplecfd.coefficients import LinearSystem
from simplecfd.linalg import tdma
from simplecfd.simple_loop import (
    CouplingIterationContext,
    PressureVelocityStepSolver,
    SIMPLECouplingStrategy,
    SIMPLECCouplingStrategy,
    SIMPLERCouplingStrategy,
    SimpleStepSolver,
)


def test_simple_step_solver_resolves_and_corrects_fields(versteeg_example_6_2_case):
    solver = versteeg_example_6_2_case.step_solver
    field = versteeg_example_6_2_case.field

    p_prime = solver.run_single_iteration()

    assert len(p_prime) == 5
    assert abs(p_prime[0]) < 1e-5
    assert abs(p_prime[4]) < 1e-5
    assert len(field.p) == 5
    assert len(field.u) == 4


def test_simple_coupling_strategy_runs_one_pressure_velocity_stage(
    versteeg_example_6_2_case,
    monkeypatch,
):
    solver = versteeg_example_6_2_case.step_solver
    calls = []

    def fake_stage():
        calls.append("stage")
        return np.arange(solver.geometry.n_pressure, dtype=float)

    monkeypatch.setattr(solver, "run_pressure_velocity_stage", fake_stage)

    p_prime = solver.run_single_iteration()

    assert calls == ["stage"]
    np.testing.assert_allclose(p_prime, np.arange(solver.geometry.n_pressure))


def test_coupling_iteration_context_exposes_step_solver_operations(
    versteeg_example_6_2_case,
):
    solver = versteeg_example_6_2_case.step_solver
    context = CouplingIterationContext(solver)

    momentum_coeffs, momentum_system = context.assemble_momentum_system()

    assert context.geometry is solver.geometry
    assert context.field is solver.field
    assert context.pressure_relaxation == solver.pressure_relaxation
    assert context.velocity_relaxation == solver.velocity_relaxation
    np.testing.assert_allclose(momentum_system.diagonal, momentum_coeffs.a_p)


def test_strategy_can_run_multiple_internal_pressure_velocity_stages(
    versteeg_example_6_2_case,
    monkeypatch,
):
    class TwoStageStrategy(SIMPLECouplingStrategy):
        def __init__(self):
            self.contexts = []

        def run_iteration(self, context):
            self.contexts.append(context)
            context.run_pressure_velocity_stage()
            return context.run_pressure_velocity_stage()

    strategy = TwoStageStrategy()
    solver = SimpleStepSolver(
        versteeg_example_6_2_case.geometry,
        versteeg_example_6_2_case.field,
        versteeg_example_6_2_case.momentum_asm,
        versteeg_example_6_2_case.pressure_correction_asm,
        coupling_strategy=strategy,
    )
    stage_results = [
        np.full(solver.geometry.n_pressure, 1.0),
        np.full(solver.geometry.n_pressure, 2.0),
    ]
    calls = []

    def fake_stage():
        calls.append("stage")
        return stage_results[len(calls) - 1]

    monkeypatch.setattr(solver, "run_pressure_velocity_stage", fake_stage)

    p_prime = solver.run_single_iteration()

    assert calls == ["stage", "stage"]
    assert len(strategy.contexts) == 1
    assert strategy.contexts[0].step_solver is solver
    np.testing.assert_allclose(p_prime, stage_results[-1])


def test_simple_step_solver_assembles_momentum_system(versteeg_example_6_2_case):
    coeffs, system = versteeg_example_6_2_case.step_solver.assemble_momentum_system()

    assert system.diagonal.shape == (versteeg_example_6_2_case.geometry.n_velocity,)
    np.testing.assert_allclose(system.diagonal, coeffs.a_p)
    np.testing.assert_allclose(system.rhs, coeffs.source)
    np.testing.assert_allclose(system.lower[1:], -coeffs.a_w[1:])
    np.testing.assert_allclose(system.upper[:-1], -coeffs.a_e[:-1])


def test_simple_step_solver_solves_linear_system_with_tdma(versteeg_example_6_2_case):
    system = LinearSystem(
        lower=np.array([0.0, -1.0, -1.0]),
        diagonal=np.array([2.0, 2.0, 2.0]),
        upper=np.array([-1.0, -1.0, 0.0]),
        rhs=np.array([1.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(
        versteeg_example_6_2_case.step_solver.solve_linear_system(system),
        tdma(system),
    )


def test_simple_step_solver_relaxes_starred_velocity(versteeg_example_6_2_case):
    solver = SimpleStepSolver(
        versteeg_example_6_2_case.geometry,
        versteeg_example_6_2_case.field,
        versteeg_example_6_2_case.momentum_asm,
        versteeg_example_6_2_case.pressure_correction_asm,
        velocity_relaxation=0.25,
    )
    old_velocity = np.array([1.0, 2.0, 3.0, 4.0])
    starred_velocity = np.array([5.0, 6.0, 7.0, 8.0])

    np.testing.assert_allclose(
        solver.relax_velocity(old_velocity, starred_velocity),
        [2.0, 3.0, 4.0, 5.0],
    )


def test_simple_step_solver_exposes_reusable_iteration_stages(
    versteeg_example_6_2_case,
):
    solver = SimpleStepSolver(
        versteeg_example_6_2_case.geometry,
        versteeg_example_6_2_case.field,
        versteeg_example_6_2_case.momentum_asm,
        versteeg_example_6_2_case.pressure_correction_asm,
    )

    momentum_coeffs = solver.predict_momentum()
    p_prime = solver.solve_pressure_correction(momentum_coeffs)
    pressure_before = solver.field.p.copy()
    velocity_before = solver.field.u.copy()

    solver.correct_pressure(p_prime)
    solver.correct_velocity(momentum_coeffs, p_prime)

    np.testing.assert_allclose(
        solver.field.p,
        pressure_before + solver.pressure_relaxation * p_prime,
    )
    expected_velocity = velocity_before.copy()
    for i in range(solver.geometry.n_velocity):
        expected_velocity[i] += momentum_coeffs.d[i] * (p_prime[i] - p_prime[i + 1])
    np.testing.assert_allclose(solver.field.u, expected_velocity)


def test_simple_step_solver_uses_simple_coupling_strategy_by_default(
    versteeg_example_6_2_case,
):
    assert isinstance(
        versteeg_example_6_2_case.step_solver.coupling_strategy,
        SIMPLECouplingStrategy,
    )


def test_pressure_velocity_step_solver_keeps_simple_step_solver_alias():
    assert SimpleStepSolver is PressureVelocityStepSolver


def test_simple_step_solver_applies_simple_correction(versteeg_example_6_2_case):
    step_solver = versteeg_example_6_2_case.step_solver
    momentum_coeffs, _ = step_solver.assemble_momentum_system()
    pressure_before = versteeg_example_6_2_case.field.p.copy()
    velocity_before = versteeg_example_6_2_case.field.u.copy()
    p_prime = np.array([0.0, 0.1, 0.2, 0.3, 0.0])

    step_solver.apply_simple_correction(momentum_coeffs, p_prime)

    np.testing.assert_allclose(
        versteeg_example_6_2_case.field.p,
        pressure_before + step_solver.pressure_relaxation * p_prime,
    )
    np.testing.assert_allclose(versteeg_example_6_2_case.field.p_prime, p_prime)
    expected_velocity = velocity_before.copy()
    for i in range(versteeg_example_6_2_case.geometry.n_velocity):
        expected_velocity[i] += momentum_coeffs.d[i] * (p_prime[i] - p_prime[i + 1])
    np.testing.assert_allclose(versteeg_example_6_2_case.field.u, expected_velocity)


def test_simple_step_solver_generic_correction_hook_matches_compatibility_alias():
    p_prime = np.array([0.0, 0.1, 0.2, 0.3, 0.0])
    generic_case = build_versteeg_example_6_2_case()
    alias_case = build_versteeg_example_6_2_case()

    generic_momentum, _ = generic_case.step_solver.assemble_momentum_system()
    alias_momentum, _ = alias_case.step_solver.assemble_momentum_system()
    generic_case.step_solver.apply_pressure_velocity_correction(
        generic_momentum,
        p_prime,
    )
    alias_case.step_solver.apply_simple_correction(alias_momentum, p_prime)

    np.testing.assert_allclose(generic_case.field.p, alias_case.field.p)
    np.testing.assert_allclose(generic_case.field.u, alias_case.field.u)
    np.testing.assert_allclose(generic_case.field.p_prime, alias_case.field.p_prime)


def test_simplec_strategy_builds_pressure_correction_momentum_coefficients(
    versteeg_example_6_2_case,
):
    strategy = SIMPLECCouplingStrategy()
    momentum_coeffs, _ = versteeg_example_6_2_case.step_solver.assemble_momentum_system()

    simplec_momentum = strategy.pressure_correction_momentum_coefficients(
        versteeg_example_6_2_case.geometry,
        momentum_coeffs,
        versteeg_example_6_2_case.step_solver.velocity_relaxation,
    )

    expected_d = momentum_coeffs.d.copy()
    denominator = (
        momentum_coeffs.a_p / versteeg_example_6_2_case.step_solver.velocity_relaxation
        - momentum_coeffs.a_w
        - momentum_coeffs.a_e
    )
    for i in range(versteeg_example_6_2_case.geometry.n_velocity):
        if denominator[i] != 0.0:
            expected_d[i] = (
                versteeg_example_6_2_case.geometry.velocity_area(i) / denominator[i]
            )

    np.testing.assert_allclose(simplec_momentum.d, expected_d)
    np.testing.assert_allclose(simplec_momentum.a_p, momentum_coeffs.a_p)
    assert simplec_momentum is not momentum_coeffs


def test_simplec_correction_coefficients_use_reference_1d_denominator(
    versteeg_example_6_2_case,
):
    strategy = SIMPLECCouplingStrategy()
    geometry = versteeg_example_6_2_case.geometry
    momentum_coeffs, _ = versteeg_example_6_2_case.step_solver.assemble_momentum_system()
    velocity_relaxation = versteeg_example_6_2_case.step_solver.velocity_relaxation

    d = strategy.correction_coefficients(
        geometry,
        momentum_coeffs,
        velocity_relaxation,
    )

    np.testing.assert_allclose(
        d,
        geometry.velocity_areas
        / (
            momentum_coeffs.a_p / velocity_relaxation
            - momentum_coeffs.a_w
            - momentum_coeffs.a_e
        ),
    )
    assert not np.allclose(d, momentum_coeffs.d)


def test_simplec_step_solver_assembles_matrix_underrelaxed_momentum_system(
    versteeg_example_6_2_case,
):
    solver = SimpleStepSolver(
        versteeg_example_6_2_case.geometry,
        versteeg_example_6_2_case.field,
        versteeg_example_6_2_case.momentum_asm,
        versteeg_example_6_2_case.pressure_correction_asm,
        velocity_relaxation=0.55,
        coupling_strategy=SIMPLECCouplingStrategy(),
    )
    old_velocity = solver.field.u.copy()

    momentum_coeffs, system = solver.assemble_momentum_system()

    expected_diagonal = momentum_coeffs.a_p / solver.velocity_relaxation
    expected_rhs = momentum_coeffs.source + (
        (1.0 - solver.velocity_relaxation) / solver.velocity_relaxation
    ) * momentum_coeffs.a_p * old_velocity
    np.testing.assert_allclose(system.diagonal, expected_diagonal)
    np.testing.assert_allclose(system.rhs, expected_rhs)
    np.testing.assert_allclose(system.lower[1:], -momentum_coeffs.a_w[1:])
    np.testing.assert_allclose(system.upper[:-1], -momentum_coeffs.a_e[:-1])


def test_simplec_velocity_prediction_is_not_relaxed_twice(versteeg_example_6_2_case):
    strategy = SIMPLECCouplingStrategy()
    old_velocity = np.array([1.0, 2.0, 3.0, 4.0])
    starred_velocity = np.array([5.0, 6.0, 7.0, 8.0])

    np.testing.assert_allclose(
        strategy.relax_velocity(old_velocity, starred_velocity, 0.55),
        starred_velocity,
    )


def test_simplec_relaxed_denominator_regularizes_constant_area_case():
    from simplecfd.cases import build_case_by_name

    case = build_case_by_name("constant_area_1d", coupling="simplec")
    momentum_coeffs, _ = case.step_solver.assemble_momentum_system()

    system = case.step_solver.assemble_pressure_correction_system(momentum_coeffs)

    assert np.all(np.isfinite(system.diagonal))
    assert np.all(system.diagonal > 0.0)


def test_simplec_step_solver_assembles_pressure_correction_with_simplec_d(
    versteeg_example_6_2_case,
):
    solver = SimpleStepSolver(
        versteeg_example_6_2_case.geometry,
        versteeg_example_6_2_case.field,
        versteeg_example_6_2_case.momentum_asm,
        versteeg_example_6_2_case.pressure_correction_asm,
        coupling_strategy=SIMPLECCouplingStrategy(),
    )
    momentum_coeffs, _ = solver.assemble_momentum_system()
    simple_system = versteeg_example_6_2_case.step_solver.assemble_pressure_correction_system(
        momentum_coeffs,
    )
    simplec_system = solver.assemble_pressure_correction_system(momentum_coeffs)
    simplec_momentum = solver.pressure_correction_momentum_coefficients(momentum_coeffs)

    expected_east_coefficient = (
        versteeg_example_6_2_case.geometry.velocity_area(1) * simplec_momentum.d[1]
    )

    np.testing.assert_allclose(simplec_system.upper[1], -expected_east_coefficient)
    assert not np.allclose(simplec_system.diagonal[1:4], simple_system.diagonal[1:4])
    np.testing.assert_allclose(simplec_system.rhs, simple_system.rhs)


def test_simplec_strategy_corrects_velocity_with_simplec_d(versteeg_example_6_2_case):
    strategy = SIMPLECCouplingStrategy()
    momentum_coeffs, _ = versteeg_example_6_2_case.step_solver.assemble_momentum_system()
    simplec_momentum = strategy.pressure_correction_momentum_coefficients(
        versteeg_example_6_2_case.geometry,
        momentum_coeffs,
        versteeg_example_6_2_case.step_solver.velocity_relaxation,
    )
    field = versteeg_example_6_2_case.field
    velocity_before = field.u.copy()
    p_prime = np.array([0.0, 0.1, 0.2, 0.3, 0.0])

    strategy.correct_velocity(
        versteeg_example_6_2_case.geometry,
        field,
        simplec_momentum,
        p_prime,
    )

    expected_velocity = velocity_before.copy()
    for i in range(versteeg_example_6_2_case.geometry.n_velocity):
        expected_velocity[i] += simplec_momentum.d[i] * (p_prime[i] - p_prime[i + 1])
    np.testing.assert_allclose(field.u, expected_velocity)


def test_simplec_case_runs_one_iteration_with_finite_fields():
    case = build_versteeg_example_6_2_case(coupling="simplec")

    p_prime = case.step_solver.run_single_iteration()

    assert np.all(np.isfinite(p_prime))
    assert np.all(np.isfinite(case.field.p))
    assert np.all(np.isfinite(case.field.u))


def test_simpler_strategy_runs_absolute_pressure_then_momentum_then_correction(
    versteeg_example_6_2_case,
    monkeypatch,
):
    solver = SimpleStepSolver(
        versteeg_example_6_2_case.geometry,
        versteeg_example_6_2_case.field,
        versteeg_example_6_2_case.momentum_asm,
        versteeg_example_6_2_case.pressure_correction_asm,
        pressure_relaxation=0.25,
        velocity_relaxation=1.0,
        coupling_strategy=SIMPLERCouplingStrategy(),
    )
    original_momentum, _ = solver.assemble_momentum_system()
    updated_momentum = original_momentum
    momentum_system = LinearSystem(
        lower=np.zeros(solver.geometry.n_velocity),
        diagonal=np.ones(solver.geometry.n_velocity),
        upper=np.zeros(solver.geometry.n_velocity),
        rhs=np.arange(10.0, 14.0),
    )
    pressure_correction_system = LinearSystem(
        lower=np.zeros(solver.geometry.n_pressure),
        diagonal=np.ones(solver.geometry.n_pressure),
        upper=np.zeros(solver.geometry.n_pressure),
        rhs=np.arange(20.0, 25.0),
    )
    absolute_pressure = np.array([10.0, 8.0, 6.0, 3.0, 0.0])
    momentum_velocity = np.array([2.0, 3.0, 4.0, 5.0])
    p_prime = np.array([0.0, 100.0, 200.0, 300.0, 0.0])
    calls = []

    def fake_assemble_momentum_system():
        calls.append(("assemble_momentum", solver.field.p.copy()))
        if len([call for call in calls if call[0] == "assemble_momentum"]) == 1:
            return original_momentum, momentum_system
        return updated_momentum, momentum_system

    def fake_pseudo_velocities(fields, momentum):
        calls.append(("pseudo_velocity", fields.p.copy(), momentum))
        return np.ones(solver.geometry.n_velocity)

    def fake_assemble_pressure_correction_system(momentum):
        calls.append(("assemble_pressure_correction", solver.field.p.copy(), momentum))
        return pressure_correction_system

    def fake_solve_linear_system(system):
        calls.append(("solve", system))
        solve_count = len([call for call in calls if call[0] == "solve"])
        if solve_count == 1:
            return absolute_pressure.copy()
        if solve_count == 2:
            return momentum_velocity.copy()
        return p_prime.copy()

    monkeypatch.setattr(solver, "assemble_momentum_system", fake_assemble_momentum_system)
    monkeypatch.setattr(solver.momentum_asm, "pseudo_velocities", fake_pseudo_velocities)
    monkeypatch.setattr(
        solver,
        "assemble_pressure_correction_system",
        fake_assemble_pressure_correction_system,
    )
    monkeypatch.setattr(solver, "solve_linear_system", fake_solve_linear_system)

    returned_p_prime = solver.run_single_iteration()

    assert [call[0] for call in calls] == [
        "assemble_momentum",
        "pseudo_velocity",
        "solve",
        "assemble_momentum",
        "solve",
        "assemble_pressure_correction",
        "solve",
    ]
    np.testing.assert_allclose(calls[3][1], absolute_pressure)
    np.testing.assert_allclose(calls[5][1], absolute_pressure)
    assert calls[5][2] is updated_momentum
    np.testing.assert_allclose(returned_p_prime, p_prime)
    np.testing.assert_allclose(solver.field.p_prime, p_prime)
    np.testing.assert_allclose(solver.field.p, absolute_pressure)

    expected_velocity = momentum_velocity.copy()
    for i in range(solver.geometry.n_velocity):
        expected_velocity[i] += updated_momentum.d[i] * (p_prime[i] - p_prime[i + 1])
    np.testing.assert_allclose(solver.field.u, expected_velocity)


def test_simpler_strategy_does_not_apply_simple_pressure_correction(
    versteeg_example_6_2_case,
    monkeypatch,
):
    solver = SimpleStepSolver(
        versteeg_example_6_2_case.geometry,
        versteeg_example_6_2_case.field,
        versteeg_example_6_2_case.momentum_asm,
        versteeg_example_6_2_case.pressure_correction_asm,
        pressure_relaxation=0.5,
        velocity_relaxation=1.0,
        coupling_strategy=SIMPLERCouplingStrategy(),
    )
    absolute_pressure = np.array([10.0, 7.0, 4.0, 1.0, 0.0])
    p_prime = np.array([0.0, 50.0, 60.0, 70.0, 0.0])
    solve_results = [
        absolute_pressure.copy(),
        np.ones(solver.geometry.n_velocity),
        p_prime.copy(),
    ]

    def fake_solve_linear_system(system):
        return solve_results.pop(0)

    monkeypatch.setattr(solver, "solve_linear_system", fake_solve_linear_system)

    solver.run_single_iteration()

    np.testing.assert_allclose(solver.field.p, absolute_pressure)
    assert not np.allclose(
        solver.field.p,
        absolute_pressure + solver.pressure_relaxation * p_prime,
    )


def test_simpler_case_runs_one_iteration_with_finite_fields():
    case = build_versteeg_example_6_2_case()
    solver = SimpleStepSolver(
        case.geometry,
        case.field,
        case.momentum_asm,
        case.pressure_correction_asm,
        coupling_strategy=SIMPLERCouplingStrategy(),
    )

    p_prime = solver.run_single_iteration()

    assert np.all(np.isfinite(p_prime))
    assert np.all(np.isfinite(case.field.p_prime))
    assert np.all(np.isfinite(case.field.p))
    assert np.all(np.isfinite(case.field.u))


@pytest.mark.parametrize(
    ("coupling", "expected"),
    [
        (
            "simple",
            {
                "p_prime": [
                    0.0,
                    1.1475492003781151,
                    2.922230103919454,
                    4.345640619474677,
                    0.0,
                ],
                "p": [
                    10.0,
                    8.303284440264681,
                    7.0455610727436175,
                    5.541948433632274,
                    0.0,
                ],
                "u": [
                    1.8427765942685261,
                    2.3692841926309622,
                    3.3169978696833473,
                    5.528329782805578,
                ],
            },
        ),
        (
            "simplec",
            {
                "p_prime": [
                    0.0,
                    1.1954554000256903,
                    1.820139884119075,
                    2.069958429254852,
                    0.0,
                ],
                "p": [
                    10.0,
                    8.336818780017984,
                    6.274097918883353,
                    3.948970900478396,
                    0.0,
                ],
                "u": [
                    1.9411869994208437,
                    2.4958118563982272,
                    3.4941365989575184,
                    5.823560998262532,
                ],
            },
        ),
        (
            "simpler",
            {
                "p_prime": [
                    0.0,
                    1.186808676597557,
                    1.6778637783925847,
                    1.5703849388156776,
                    0.0,
                ],
                "p": [
                    10.0,
                    8.871749561233107,
                    8.653156528900707,
                    6.853751375036391,
                    0.0,
                ],
                "u": [
                    1.5262332641730987,
                    1.9622999110796986,
                    2.7472198755115778,
                    4.578699792519297,
                ],
            },
        ),
    ],
)
def test_pressure_velocity_couplings_keep_one_iteration_regression(
    coupling,
    expected,
):
    case = build_versteeg_example_6_2_case(coupling=coupling)

    returned_p_prime = case.step_solver.run_single_iteration()

    np.testing.assert_allclose(returned_p_prime, expected["p_prime"], rtol=1e-14)
    np.testing.assert_allclose(case.field.p_prime, expected["p_prime"], rtol=1e-14)
    np.testing.assert_allclose(case.field.p, expected["p"], rtol=1e-14)
    np.testing.assert_allclose(case.field.u, expected["u"], rtol=1e-14)


@pytest.mark.parametrize(
    ("coupling", "expected_by_kind"),
    [
        ("simple", {"momentum": 1, "pressure_correction": 1}),
        ("simplec", {"momentum": 1, "pressure_correction": 1}),
        (
            "simpler",
            {"absolute_pressure": 1, "momentum": 1, "pressure_correction": 1},
        ),
    ],
)
def test_pressure_velocity_couplings_count_linear_solves_by_iteration(
    coupling,
    expected_by_kind,
):
    case = build_versteeg_example_6_2_case(coupling=coupling)

    case.step_solver.run_single_iteration()

    expected_total = sum(expected_by_kind.values())
    assert case.step_solver.linear_solve_history == [
        {"total": expected_total, "by_kind": expected_by_kind}
    ]
    assert case.step_solver.linear_solve_totals() == {
        "total": expected_total,
        "by_kind": expected_by_kind,
    }


def test_simple_step_solver_delegates_coupling_to_strategy(
    versteeg_example_6_2_case,
    monkeypatch,
):
    class RecordingCouplingStrategy:
        def __init__(self):
            self.relaxation_calls = []
            self.correction_calls = []

        def relax_velocity(
            self,
            old_velocity,
            starred_velocity,
            velocity_relaxation,
        ):
            self.relaxation_calls.append(
                (old_velocity.copy(), starred_velocity.copy(), velocity_relaxation)
            )
            return starred_velocity.copy()

        def apply_correction(
            self,
            geometry,
            field,
            momentum_coeffs,
            p_prime,
            pressure_relaxation,
        ):
            self.correction_calls.append(
                (geometry, field, momentum_coeffs, p_prime.copy(), pressure_relaxation)
            )
            field.p_prime = p_prime.copy()

        def correct_pressure(self, field, p_prime, pressure_relaxation):
            field.p += pressure_relaxation * p_prime

        def correct_velocity(self, geometry, field, momentum_coeffs, p_prime):
            for i in range(geometry.n_velocity):
                field.u[i] += momentum_coeffs.d[i] * (p_prime[i] - p_prime[i + 1])

    strategy = RecordingCouplingStrategy()
    solver = SimpleStepSolver(
        versteeg_example_6_2_case.geometry,
        versteeg_example_6_2_case.field,
        versteeg_example_6_2_case.momentum_asm,
        versteeg_example_6_2_case.pressure_correction_asm,
        pressure_relaxation=0.35,
        velocity_relaxation=0.55,
        coupling_strategy=strategy,
    )

    calls = []

    def fake_tdma(system):
        calls.append(system)
        if len(calls) == 1:
            return np.arange(1.0, 5.0)
        return np.arange(5.0)

    monkeypatch.setattr("simplecfd.simple_loop.tdma", fake_tdma)

    p_prime = solver.run_single_iteration()

    assert len(strategy.relaxation_calls) == 1
    assert len(strategy.correction_calls) == 1
    assert strategy.relaxation_calls[0][2] == 0.55
    assert strategy.correction_calls[0][4] == 0.35
    np.testing.assert_allclose(p_prime, np.arange(5.0))
    np.testing.assert_allclose(versteeg_example_6_2_case.field.p_prime, np.arange(5.0))


def test_simple_step_solver_accepts_configurable_pressure_relaxation(versteeg_example_6_2_case):
    solver = SimpleStepSolver(
        versteeg_example_6_2_case.geometry,
        versteeg_example_6_2_case.field,
        versteeg_example_6_2_case.momentum_asm,
        versteeg_example_6_2_case.pressure_correction_asm,
        pressure_relaxation=0.35,
    )

    assert solver.pressure_relaxation == 0.35


def test_simple_step_solver_accepts_configurable_velocity_relaxation(versteeg_example_6_2_case):
    solver = SimpleStepSolver(
        versteeg_example_6_2_case.geometry,
        versteeg_example_6_2_case.field,
        versteeg_example_6_2_case.momentum_asm,
        versteeg_example_6_2_case.pressure_correction_asm,
        velocity_relaxation=0.55,
    )

    assert solver.velocity_relaxation == 0.55


def test_versteeg_case_builder_configures_relaxation_factors():
    case = build_versteeg_example_6_2_case(
        pressure_relaxation=0.42,
        velocity_relaxation=0.62,
    )

    assert case.step_solver.pressure_relaxation == 0.42
    assert case.step_solver.velocity_relaxation == 0.62


@pytest.mark.parametrize("pressure_relaxation", [0.0, -0.1, 1.1])
def test_simple_step_solver_rejects_invalid_pressure_relaxation(
    versteeg_example_6_2_case,
    pressure_relaxation,
):
    with pytest.raises(ValueError, match="pressure_relaxation"):
        SimpleStepSolver(
            versteeg_example_6_2_case.geometry,
            versteeg_example_6_2_case.field,
            versteeg_example_6_2_case.momentum_asm,
            versteeg_example_6_2_case.pressure_correction_asm,
            pressure_relaxation=pressure_relaxation,
        )


@pytest.mark.parametrize("velocity_relaxation", [0.0, -0.1, 1.1])
def test_simple_step_solver_rejects_invalid_velocity_relaxation(
    versteeg_example_6_2_case,
    velocity_relaxation,
):
    with pytest.raises(ValueError, match="velocity_relaxation"):
        SimpleStepSolver(
            versteeg_example_6_2_case.geometry,
            versteeg_example_6_2_case.field,
            versteeg_example_6_2_case.momentum_asm,
            versteeg_example_6_2_case.pressure_correction_asm,
            velocity_relaxation=velocity_relaxation,
        )


def test_simple_step_solver_iteration_uses_assembly_solve_and_correction(
    versteeg_example_6_2_case,
    monkeypatch,
):
    solver = SimpleStepSolver(
        versteeg_example_6_2_case.geometry,
        versteeg_example_6_2_case.field,
        versteeg_example_6_2_case.momentum_asm,
        versteeg_example_6_2_case.pressure_correction_asm,
    )

    calls = []

    def fake_tdma(system):
        assert isinstance(system, LinearSystem)
        calls.append(system)
        if len(calls) == 1:
            return np.zeros(versteeg_example_6_2_case.geometry.n_velocity)
        return np.zeros(versteeg_example_6_2_case.geometry.n_pressure)

    monkeypatch.setattr("simplecfd.simple_loop.tdma", fake_tdma)

    p_prime = solver.run_single_iteration()

    assert len(calls) == 2
    np.testing.assert_allclose(
        p_prime,
        np.zeros(versteeg_example_6_2_case.geometry.n_pressure),
    )
