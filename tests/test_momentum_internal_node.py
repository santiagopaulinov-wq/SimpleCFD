import numpy as np
import pytest

from simplecfd.assembly.momentum import MomentumAssembler
from simplecfd.fields import Field
from simplecfd.geometry import Geometry
from simplecfd.schemes import CentralDifference, Upwind


@pytest.fixture
def initial_versteeg_momentum_coefficients():
    geometry = Geometry(
        pressure_areas=np.array([0.5, 0.4, 0.3, 0.2, 0.1]),
        velocity_areas=np.array([0.45, 0.35, 0.25, 0.15]),
        dx=0.5,
    )
    fields = Field(
        p=np.array([10.0, 7.5, 5.0, 2.5, 0.0]),
        u=np.array([2.22222, 2.85714, 4.0, 6.66666]),
        p_prime=np.zeros(5),
    )

    return MomentumAssembler(geometry, density=1.0, scheme=Upwind()).assemble(fields)


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (
            1,
            {
                "f_w": 1.015872,
                "f_e": 1.028571,
                "a_w": 1.015872,
                "a_e": 0.0,
                "a_p": 1.028571,
                "source": 0.875,
                "d": 0.34027,
            },
        ),
        (
            2,
            {
                "f_w": 1.028571,
                "f_e": 1.066666,
                "a_w": 1.028571,
                "a_e": 0.0,
                "a_p": 1.066666,
                "source": 0.625,
                "d": 0.23437,
            },
        ),
    ],
    ids=["velocity_node_2", "velocity_node_3"],
)
def test_each_internal_momentum_node_matches_versteeg_example_6_2(
    initial_versteeg_momentum_coefficients,
    node,
    expected,
):
    coeffs = initial_versteeg_momentum_coefficients

    np.testing.assert_allclose(coeffs.f_w[node], expected["f_w"], rtol=1e-5)
    np.testing.assert_allclose(coeffs.f_e[node], expected["f_e"], rtol=1e-5)
    np.testing.assert_allclose(coeffs.a_w[node], expected["a_w"], rtol=1e-5)
    np.testing.assert_allclose(coeffs.a_e[node], expected["a_e"], rtol=1e-5)
    np.testing.assert_allclose(coeffs.a_p[node], expected["a_p"], rtol=1e-5)
    np.testing.assert_allclose(coeffs.source[node], expected["source"], rtol=1e-5)
    np.testing.assert_allclose(coeffs.d[node], expected["d"], rtol=1e-4)


def test_upwind_internal_momentum_assembly_preserves_golden_coefficients():
    geometry = Geometry.versteeg_example_6_2()
    fields = Field.versteeg_example_6_2_initial_guess()

    coeffs = MomentumAssembler(geometry, density=1.0, scheme=Upwind()).assemble(fields)

    np.testing.assert_allclose(coeffs.a_w[1:3], [1.015872, 1.028571], rtol=1e-5)
    np.testing.assert_allclose(coeffs.a_e[1:3], [0.0, 0.0], rtol=1e-5)
    np.testing.assert_allclose(coeffs.a_p[1:3], [1.028571, 1.066666], rtol=1e-5)


def test_central_difference_changes_internal_momentum_coefficients_controlled():
    geometry = Geometry.versteeg_example_6_2()
    fields = Field.versteeg_example_6_2_initial_guess()

    upwind = MomentumAssembler(geometry, density=1.0, scheme=Upwind()).assemble(fields)
    central = MomentumAssembler(
        geometry,
        density=1.0,
        scheme=CentralDifference(),
    ).assemble(fields)

    np.testing.assert_allclose(central.f_w[1:3], upwind.f_w[1:3])
    np.testing.assert_allclose(central.f_e[1:3], upwind.f_e[1:3])
    np.testing.assert_allclose(central.source[1:3], upwind.source[1:3])
    np.testing.assert_allclose(central.a_w[1:3], 0.5 * central.f_w[1:3])
    np.testing.assert_allclose(central.a_e[1:3], -0.5 * central.f_e[1:3])
    np.testing.assert_allclose(
        central.a_p[1:3],
        central.a_w[1:3] + central.a_e[1:3] + (central.f_e[1:3] - central.f_w[1:3]),
    )
    assert not np.allclose(central.a_w[1:3], upwind.a_w[1:3])
    assert not np.allclose(central.a_p[1:3], upwind.a_p[1:3])
    assert np.all(np.isfinite(central.d[1:3]))
