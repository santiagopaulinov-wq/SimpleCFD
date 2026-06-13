import numpy as np
import pytest

from simplecfd.cases import build_versteeg_example_6_2_case
from simplecfd.schemes import CentralDifference, Upwind


@pytest.mark.parametrize(
    ("scheme", "mass_flux", "expected"),
    [
        (Upwind(), 1.0, 2.0),
        (Upwind(), -1.0, 8.0),
        (CentralDifference(), 1.0, 5.0),
        (CentralDifference(), -1.0, 5.0),
    ],
    ids=[
        "upwind_positive_flux",
        "upwind_negative_flux",
        "central_positive_flux",
        "central_negative_flux",
    ],
)
def test_convection_scheme_interpolation_variants(scheme, mass_flux, expected):
    assert scheme.interpolate(west_value=2.0, east_value=8.0, mass_flux=mass_flux) == expected


@pytest.mark.parametrize(
    ("scheme", "mass_flux", "expected_west", "expected_east"),
    [
        (Upwind(), 4.0, 4.0, 0.0),
        (Upwind(), -4.0, 0.0, 4.0),
        (CentralDifference(), 4.0, 2.0, -2.0),
        (CentralDifference(), -4.0, -2.0, 2.0),
    ],
    ids=[
        "upwind_positive_flux",
        "upwind_negative_flux",
        "central_positive_flux",
        "central_negative_flux",
    ],
)
def test_convection_scheme_coefficient_variants(
    scheme,
    mass_flux,
    expected_west,
    expected_east,
):
    assert scheme.west_coefficient(mass_flux) == expected_west
    assert scheme.east_coefficient(mass_flux) == expected_east


def test_versteeg_case_with_upwind_still_converges():
    case = build_versteeg_example_6_2_case(scheme=Upwind(), max_iterations=100)
    result = case.solver.solve()

    assert result["converged"] is True
    assert result["residual"] < case.solver.tolerance
    assert result["iterations"] <= case.solver.max_iterations
    assert case.field.p.shape == (case.geometry.n_pressure,)
    assert case.field.u.shape == (case.geometry.n_velocity,)
    assert result["continuity_residual_vector"].shape == (case.geometry.n_pressure,)
    assert result["momentum_residual_vector"].shape == (case.geometry.n_velocity,)
    assert np.all(np.isfinite(case.field.p))
    assert np.all(np.isfinite(case.field.u))
    assert np.all(np.isfinite(result["continuity_residual_vector"]))
    assert np.all(np.isfinite(result["momentum_residual_vector"]))


def test_versteeg_case_with_central_difference_runs_solver_with_finite_state():
    case = build_versteeg_example_6_2_case(scheme=CentralDifference(), max_iterations=100)
    result = case.solver.solve()

    assert result["iterations"] == case.solver.max_iterations
    assert case.field.p.shape == (case.geometry.n_pressure,)
    assert case.field.u.shape == (case.geometry.n_velocity,)
    assert result["continuity_residual_vector"].shape == (case.geometry.n_pressure,)
    assert result["momentum_residual_vector"].shape == (case.geometry.n_velocity,)
    assert np.isfinite(result["residual"])
    assert np.isfinite(result["continuity_residual"])
    assert np.isfinite(result["momentum_residual"])
    assert np.all(np.isfinite(case.field.p))
    assert np.all(np.isfinite(case.field.u))
    assert np.all(np.isfinite(result["continuity_residual_vector"]))
    assert np.all(np.isfinite(result["momentum_residual_vector"]))


def test_versteeg_case_builder_preserves_selected_scheme_instance():
    scheme = CentralDifference()
    case = build_versteeg_example_6_2_case(scheme=scheme)

    assert case.momentum_asm.scheme is scheme
