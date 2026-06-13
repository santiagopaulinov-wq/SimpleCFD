import json
from pathlib import Path

import numpy as np
import pytest

from simplecfd.cases import build_case_by_name, list_available_cases
from simplecfd.schemes import Upwind


REGRESSION_DIR = Path(__file__).parent / "regression_expected"
SCHEMES = {
    "upwind": Upwind,
}


def load_regression_records():
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(REGRESSION_DIR.glob("*.json"))
    ]


@pytest.mark.parametrize(
    "record",
    load_regression_records(),
    ids=lambda record: f"{record['case_name']}[{record['variant']}]",
)
def test_registered_case_matches_numerical_regression(record):
    case_name = record["case_name"]
    configuration = record["configuration"]
    expected = record["expected"]
    tolerances = record["tolerances"]

    assert case_name in list_available_cases()

    scheme = SCHEMES[configuration["scheme"]]()
    case = build_case_by_name(
        case_name,
        coupling=configuration["coupling"],
        scheme=scheme,
        tolerance=configuration["tolerance"],
        max_iterations=configuration["max_iterations"],
        pressure_relaxation=configuration["pressure_relaxation"],
        velocity_relaxation=configuration["velocity_relaxation"],
    )

    result = case.solver.solve()

    assert result["converged"] is expected["converged"]
    assert abs(result["iterations"] - expected["iterations"]) <= tolerances["iterations_abs"]
    np.testing.assert_allclose(
        result["residual"],
        expected["residual"],
        rtol=tolerances["residual_rtol"],
        atol=tolerances["residual_atol"],
    )
    np.testing.assert_allclose(
        result["continuity_residual"],
        expected["continuity_residual"],
        rtol=tolerances["residual_rtol"],
        atol=tolerances["residual_atol"],
    )
    np.testing.assert_allclose(
        result["momentum_residual"],
        expected["momentum_residual"],
        rtol=tolerances["residual_rtol"],
        atol=tolerances["residual_atol"],
    )

    np.testing.assert_allclose(
        case.field.p,
        expected["final_pressure"],
        rtol=tolerances["field_rtol"],
        atol=tolerances["field_atol"],
    )
    np.testing.assert_allclose(
        case.field.u,
        expected["final_velocity"],
        rtol=tolerances["field_rtol"],
        atol=tolerances["field_atol"],
    )
    np.testing.assert_allclose(
        case.field.p_prime,
        expected["final_pressure_correction"],
        rtol=tolerances["field_rtol"],
        atol=tolerances["field_atol"],
    )
    np.testing.assert_allclose(
        result["continuity_residual_vector"],
        expected["continuity_residual_vector"],
        rtol=tolerances["residual_vector_rtol"],
        atol=tolerances["residual_vector_atol"],
    )
    np.testing.assert_allclose(
        result["momentum_residual_vector"],
        expected["momentum_residual_vector"],
        rtol=tolerances["residual_vector_rtol"],
        atol=tolerances["residual_vector_atol"],
    )


def test_simpler_versteeg_6_2_upwind_converges_to_finite_reference_profile():
    case = build_case_by_name(
        "versteeg_6_2",
        coupling="simpler",
        scheme=Upwind(),
        tolerance=1e-5,
        max_iterations=100,
        pressure_relaxation=0.7,
        velocity_relaxation=0.7,
    )

    result = case.solver.solve()
    mass_flow = case.field.u * case.geometry.velocity_areas

    assert result["converged"] is True
    assert result["iterations"] == 5
    np.testing.assert_allclose(result["residual"], 2.777772396544975e-7, rtol=1e-8)
    np.testing.assert_allclose(result["continuity_residual"], 0.0, atol=1e-14)
    np.testing.assert_allclose(result["momentum_residual"], result["residual"])
    np.testing.assert_allclose(
        case.field.p,
        [10.0, 9.004187390938, 8.25061016982, 6.19427887696, 0.0],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        case.field.u,
        [1.382683639283, 1.777736107649, 2.488830550709, 4.148050917848],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        mass_flow,
        np.full(case.geometry.n_velocity, 0.622207637677),
        rtol=1e-10,
        atol=1e-12,
    )
