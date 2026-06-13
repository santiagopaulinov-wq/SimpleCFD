from pathlib import Path

import numpy as np

from plot_convergence import create_convergence_plots
from simplecfd.telemetry import (
    collect_case_convergence,
    collect_simple_history,
    collect_versteeg_convergence,
)


def test_versteeg_telemetry_tracks_all_nozzle_nodes_until_convergence():
    history = collect_versteeg_convergence(tolerance=1e-5, max_iterations=100)

    assert history.pressure.shape[0] == history.iterations.size
    assert history.velocity.shape[0] == history.iterations.size
    assert history.momentum_residual.shape[0] == history.iterations.size
    assert history.continuity_residual.shape[0] == history.iterations.size
    assert history.pressure.shape[1] == 5
    assert history.velocity.shape[1] == 4
    assert history.momentum_residual.shape[1] == 4
    assert history.continuity_residual.shape[1] == 5
    assert history.iterations[0] == 0
    assert history.iterations[-1] > 1
    assert history.momentum_norm[-1] < 1e-5
    assert history.continuity_norm[-1] < 1e-5
    np.testing.assert_allclose(
        history.residual_norm,
        np.maximum(history.momentum_norm, history.continuity_norm),
    )


def test_versteeg_telemetry_reconstructs_initial_and_final_snapshots():
    history = collect_versteeg_convergence(tolerance=1e-5, max_iterations=100)

    initial = history.snapshot(0)
    final = history.final_snapshot()

    assert initial["iteration"] == 0
    np.testing.assert_allclose(initial["pressure"], [10.0, 7.5, 5.0, 2.5, 0.0])
    np.testing.assert_allclose(
        initial["velocity"],
        [2.2222222222, 2.8571428571, 4.0, 6.6666666667],
        rtol=1e-10,
    )
    assert final["iteration"] == history.final_iteration
    np.testing.assert_allclose(final["pressure"], history.pressure[-1])
    np.testing.assert_allclose(final["velocity"], history.velocity[-1])
    np.testing.assert_allclose(final["momentum_residual"], history.momentum_residual[-1])
    np.testing.assert_allclose(final["continuity_residual"], history.continuity_residual[-1])


def test_generic_simple_history_uses_supplied_case(versteeg_example_6_2_case):
    history = collect_simple_history(versteeg_example_6_2_case)

    assert history.final_iteration > 0
    np.testing.assert_allclose(history.pressure[-1], versteeg_example_6_2_case.field.p)
    np.testing.assert_allclose(history.velocity[-1], versteeg_example_6_2_case.field.u)
    assert history.residual_norm[-1] < versteeg_example_6_2_case.solver.tolerance


def test_case_telemetry_collects_registered_nozzle_and_method_history():
    history = collect_case_convergence(
        "linear_nozzle_1d",
        method="simplec",
        tolerance=1e-4,
        max_iterations=20,
        pressure_relaxation=0.45,
        velocity_relaxation=0.45,
        inlet_area=1.0,
        outlet_area=0.55,
        n_pressure=6,
        dx=0.2,
        mass_flow_guess=0.6,
        inlet_pressure=5.0,
        outlet_pressure=1.0,
    )

    assert history.iterations[0] == 0
    assert history.final_iteration <= 20
    assert history.pressure.shape == (history.iterations.size, 6)
    assert history.velocity.shape == (history.iterations.size, 5)
    assert history.momentum_residual.shape == (history.iterations.size, 5)
    assert history.continuity_residual.shape == (history.iterations.size, 6)
    assert np.all(np.isfinite(history.residual_norm))


def test_case_telemetry_uses_same_final_state_as_solver_result():
    from simplecfd.cases import build_case_by_name

    case = build_case_by_name(
        "smooth_linear_nozzle_1d",
        coupling="simpler",
        tolerance=1e-4,
        max_iterations=6,
    )

    history = collect_simple_history(case)
    final = history.final_snapshot()

    np.testing.assert_allclose(final["pressure"], case.field.p)
    np.testing.assert_allclose(final["velocity"], case.field.u)
    assert final["residual_norm"] == history.residual_norm[-1]


def test_versteeg_telemetry_mass_flow_converges_at_every_velocity_node():
    history = collect_versteeg_convergence(tolerance=1e-5, max_iterations=100)
    velocity_areas = np.array([0.45, 0.35, 0.25, 0.15])
    final_mass_flow = history.velocity[-1] * velocity_areas

    np.testing.assert_allclose(
        final_mass_flow,
        np.full(4, final_mass_flow[0]),
        rtol=1e-10,
        atol=1e-10,
    )


def test_convergence_plot_generation_creates_four_png_files():
    history = collect_versteeg_convergence(tolerance=1e-5, max_iterations=100)
    paths = create_convergence_plots(
        output_dir=Path("outputs/test_convergence_plots"),
        history=history,
        tolerance=1e-5,
    )

    assert {path.name for path in paths} == {
        "velocity_convergence.png",
        "pressure_convergence.png",
        "momentum_balance_convergence.png",
        "continuity_balance_convergence.png",
    }
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0
