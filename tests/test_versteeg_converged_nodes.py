import numpy as np
import pytest


PRESSURE_NODE_NAMES = ["A", "B", "C", "D", "E"]
VELOCITY_NODE_NAMES = ["1", "2", "3", "4"]

EXPECTED_PRESSURE = np.array(
    [10.0, 9.004194245994, 8.250611297971, 6.194307405519, 0.0],
)
EXPECTED_VELOCITY = np.array(
    [1.382679709419, 1.777731054967, 2.488823476954, 4.148039128257],
)
EXPECTED_MASS_FLOW = 0.62220586923855


@pytest.fixture
def converged_versteeg_solution(versteeg_example_6_2_case):
    result = versteeg_example_6_2_case.solver.solve()
    assert result["converged"] is True
    return versteeg_example_6_2_case, result


@pytest.mark.parametrize("node", range(5), ids=PRESSURE_NODE_NAMES)
def test_all_pressure_nodes_match_versteeg_converged_solution(converged_versteeg_solution, node):
    case, _ = converged_versteeg_solution

    np.testing.assert_allclose(case.field.p[node], EXPECTED_PRESSURE[node], rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("node", range(4), ids=VELOCITY_NODE_NAMES)
def test_all_velocity_nodes_match_versteeg_converged_solution(converged_versteeg_solution, node):
    case, _ = converged_versteeg_solution

    np.testing.assert_allclose(case.field.u[node], EXPECTED_VELOCITY[node], rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("node", range(4), ids=VELOCITY_NODE_NAMES)
def test_all_velocity_nodes_have_same_converged_mass_flow(converged_versteeg_solution, node):
    case, _ = converged_versteeg_solution
    mass_flow = case.field.u[node] * case.geometry.velocity_area(node)

    np.testing.assert_allclose(mass_flow, EXPECTED_MASS_FLOW, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("node", range(1, 4), ids=PRESSURE_NODE_NAMES[1:-1])
def test_all_internal_pressure_nodes_satisfy_continuity_balance(converged_versteeg_solution, node):
    case, result = converged_versteeg_solution
    continuity_residual = case.solver.calculate_mass_residual_vector()

    assert abs(continuity_residual[node]) < result["continuity_residual"] + 1e-12
    assert abs(continuity_residual[node]) < 1e-5


@pytest.mark.parametrize("node", range(4), ids=VELOCITY_NODE_NAMES)
def test_all_velocity_nodes_satisfy_momentum_balance(converged_versteeg_solution, node):
    case, result = converged_versteeg_solution
    momentum_residual = case.solver.calculate_momentum_residual_vector()

    assert abs(momentum_residual[node]) < result["momentum_residual"] + 1e-12
    assert abs(momentum_residual[node]) < 1e-5
