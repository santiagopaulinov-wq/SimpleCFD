import numpy as np
import pytest

from simplecfd.mesh_refinement import run_mesh_refinement_study


def test_mesh_refinement_study_runs_family_and_interpolates_profiles():
    study = run_mesh_refinement_study(
        "linear_nozzle_1d",
        node_counts=(5, 7, 9, 11),
        length=1.0,
        sample_count=25,
        inlet_area=1.0,
        outlet_area=0.6,
        mass_flow_guess=0.6,
        inlet_pressure=5.0,
        outlet_pressure=1.0,
        max_iterations=300,
        pressure_relaxation=0.45,
        velocity_relaxation=0.45,
    )

    assert study["case_name"] == "linear_nozzle_1d"
    assert study["node_counts"] == (5, 7, 9, 11)
    assert len(study["runs"]) == 4
    assert study["reference"]["n_pressure"] == 11
    np.testing.assert_allclose(study["sample_positions"]["pressure"][0], 0.0)
    np.testing.assert_allclose(study["sample_positions"]["pressure"][-1], 1.0)

    for run, node_count in zip(study["runs"], study["node_counts"]):
        assert run["n_pressure"] == node_count
        assert run["n_velocity"] == node_count - 1
        assert run["h"] == pytest.approx(1.0 / (node_count - 1))
        assert run["converged"] is True
        for field in ("pressure", "velocity", "mass_flow"):
            assert run["interpolated_profiles"][field].shape == (25,)
            assert np.all(np.isfinite(run["interpolated_profiles"][field]))
            assert set(run["reference_errors"][field]) == {"linf", "l2"}


def test_mesh_refinement_study_reports_residuals_mass_flow_and_spatial_indicators():
    study = run_mesh_refinement_study(
        node_counts=(5, 7, 9, 11),
        sample_count=25,
        max_iterations=300,
        pressure_relaxation=0.45,
        velocity_relaxation=0.45,
    )

    residuals = study["residuals"]
    mass_flow = study["mass_flow"]
    spatial = study["spatial_convergence"]

    assert residuals["all_converged"] is True
    assert residuals["final"].shape == (4,)
    assert residuals["max_final_residual"] < 1e-4
    assert mass_flow["final_means"].shape == (4,)
    assert mass_flow["successive_difference"].shape == (3,)
    assert mass_flow["max_difference_to_reference"] > 0.0
    np.testing.assert_allclose(mass_flow["difference_to_reference"][-1], 0.0)

    for field in ("pressure", "velocity", "mass_flow"):
        assert spatial[field]["reference_linf"].shape == (4,)
        assert spatial[field]["reference_l2"].shape == (4,)
        assert spatial[field]["successive_linf"].shape == (3,)
        assert spatial[field]["error_reduction"].shape == (3,)
        assert spatial[field]["observed_order"].shape == (2,)
        assert spatial[field]["monotone_to_reference"] is True
        np.testing.assert_allclose(spatial[field]["reference_linf"][-1], 0.0)

    assert np.all(np.isfinite(spatial["velocity"]["observed_order"]))
    assert np.all(spatial["velocity"]["observed_order"] > 0.0)


def test_mesh_refinement_study_sorts_node_counts_and_rejects_invalid_inputs():
    study = run_mesh_refinement_study(
        node_counts=(9, 5, 7),
        sample_count=9,
        max_iterations=150,
    )

    assert study["node_counts"] == (5, 7, 9)

    with pytest.raises(ValueError, match="at least two"):
        run_mesh_refinement_study(node_counts=(5,))
    with pytest.raises(ValueError, match="duplicates"):
        run_mesh_refinement_study(node_counts=(5, 5, 7))
    with pytest.raises(ValueError, match="integers >= 3"):
        run_mesh_refinement_study(node_counts=(2, 5))
    with pytest.raises(ValueError, match="length"):
        run_mesh_refinement_study(node_counts=(5, 7), length=0.0)
    with pytest.raises(ValueError, match="sample_count"):
        run_mesh_refinement_study(node_counts=(5, 7), sample_count=1)
    with pytest.raises(ValueError, match="n_pressure"):
        run_mesh_refinement_study(node_counts=(5, 7), n_pressure=5)
