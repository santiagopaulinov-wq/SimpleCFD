import numpy as np

from simplecfd.solver import PressureVelocitySolver, SIMPLESolver


def test_pressure_velocity_solver_keeps_simple_solver_alias():
    assert SIMPLESolver is PressureVelocitySolver


def test_simple_solver_converges_versteeg_example_6_2(versteeg_example_6_2_case):
    result = versteeg_example_6_2_case.solver.solve()

    assert result["converged"] is True, f"El solver no convergio. Residual final: {result['residual']}"
    assert result["residual"] < 1e-5
    assert result["continuity_residual"] < 1e-5
    assert result["momentum_residual"] < 1e-5
    assert result["continuity_residual_vector"].shape == (versteeg_example_6_2_case.geometry.n_pressure,)
    assert result["momentum_residual_vector"].shape == (versteeg_example_6_2_case.geometry.n_velocity,)
    np.testing.assert_allclose(
        result["momentum_residual"],
        np.linalg.norm(result["momentum_residual_vector"], ord=np.inf),
    )
    assert len(versteeg_example_6_2_case.field.p) == 5
    assert len(versteeg_example_6_2_case.field.u) == 4
    assert len(result["residual_history"]) == result["iterations"]
    assert result["residual_history"][-1] == result["residual"]


def test_simple_solver_result_reports_linear_solve_counts(versteeg_example_6_2_case):
    result = versteeg_example_6_2_case.solver.solve()

    assert len(result["linear_solve_history"]) == result["iterations"]
    assert all(
        entry == {"total": 2, "by_kind": {"momentum": 1, "pressure_correction": 1}}
        for entry in result["linear_solve_history"]
    )
    assert result["linear_solve_counts"] == {
        "total": 2 * result["iterations"],
        "by_kind": {
            "momentum": result["iterations"],
            "pressure_correction": result["iterations"],
        },
    }


def test_simple_solver_mass_residual_is_pressure_correction_rhs_norm(versteeg_example_6_2_case):
    momentum = versteeg_example_6_2_case.momentum_asm.assemble(versteeg_example_6_2_case.field)
    expected_rhs = versteeg_example_6_2_case.pressure_correction_asm.assemble(
        versteeg_example_6_2_case.field,
        momentum,
    ).rhs

    solver = versteeg_example_6_2_case.solver

    np.testing.assert_allclose(solver.calculate_continuity_residual_vector(), expected_rhs)
    np.testing.assert_allclose(solver.calculate_mass_residual_vector(), expected_rhs)
    np.testing.assert_allclose(
        solver.calculate_mass_residual_norm(),
        np.linalg.norm(expected_rhs, ord=np.inf),
    )


def test_simple_solver_reports_continuity_residual_vector_by_pressure_node(
    versteeg_example_6_2_case,
):
    solver = versteeg_example_6_2_case.solver
    continuity_vector = solver.calculate_continuity_residual_vector()

    assert continuity_vector.shape == (versteeg_example_6_2_case.geometry.n_pressure,)
    np.testing.assert_allclose(
        solver.calculate_mass_residual_norm(),
        np.linalg.norm(continuity_vector, ord=np.inf),
    )


def test_simple_solver_convergence_state_combines_momentum_and_continuity(
    versteeg_example_6_2_case,
    monkeypatch,
):
    solver: SIMPLESolver = versteeg_example_6_2_case.solver

    continuity_vector = np.array([0.0, 1.0e-6, 0.0])
    momentum_vector = np.array([1.0e-6, -2.0e-5])
    monkeypatch.setattr(solver, "calculate_continuity_residual_vector", lambda: continuity_vector)
    monkeypatch.setattr(solver, "calculate_momentum_residual_vector", lambda: momentum_vector)

    state = solver.convergence_state()

    assert state["continuity_residual"] == 1.0e-6
    assert state["momentum_residual"] == 2.0e-5
    assert state["residual"] == 2.0e-5
    assert state["converged"] is False
    np.testing.assert_allclose(state["continuity_residual_vector"], continuity_vector)
    np.testing.assert_allclose(state["momentum_residual_vector"], momentum_vector)


def test_simple_solver_builds_result_from_convergence_state(versteeg_example_6_2_case):
    state = {
        "residual": 9.0e-6,
        "continuity_residual": 1.0e-6,
        "continuity_residual_vector": np.array([0.0, 1.0e-6, 0.0]),
        "momentum_residual": 9.0e-6,
        "momentum_residual_vector": np.array([1.0e-6, -9.0e-6]),
    }

    result = versteeg_example_6_2_case.solver.build_result(
        converged=True,
        iterations=7,
        residual_history=[1.0, 9.0e-6],
        state=state,
    )

    assert result["converged"] is True
    assert result["iterations"] == 7
    assert result["residual"] == 9.0e-6
    assert result["residual_history"] == [1.0, 9.0e-6]
    assert result["continuity_residual"] == 1.0e-6
    assert result["momentum_residual"] == 9.0e-6
    assert result["linear_solve_history"] == []
    assert result["linear_solve_counts"] == {"total": 0, "by_kind": {}}
    np.testing.assert_allclose(result["continuity_residual_vector"], state["continuity_residual_vector"])
    np.testing.assert_allclose(result["momentum_residual_vector"], state["momentum_residual_vector"])


def test_simple_solver_reports_momentum_residual_vector_by_node(versteeg_example_6_2_case):
    solver = versteeg_example_6_2_case.solver
    momentum_vector = solver.calculate_momentum_residual_vector()

    assert momentum_vector.shape == (versteeg_example_6_2_case.geometry.n_velocity,)
    np.testing.assert_allclose(
        solver.calculate_momentum_residual_norm(),
        np.linalg.norm(momentum_vector, ord=np.inf),
    )
