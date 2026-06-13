import numpy as np
import pytest

from simplecfd.assembly.momentum import MomentumAssembler
from simplecfd.boundary import InletStagnationPressure, OutletFixedPressure
from simplecfd.fields import Field
from simplecfd.geometry import Geometry
from simplecfd.schemes import Upwind


@pytest.mark.parametrize(
    ("node", "assembler_kwargs", "expected"),
    [
        (
            0,
            {"inlet": InletStagnationPressure(stagnation_pressure=10.0)},
            {
                "f_w": 1.0,
                "f_e": 1.01587,
                "a_w": 0.0,
                "a_e": 0.0,
                "a_p": 1.42087,
                "source": 3.125,
                "d": 0.31670,
            },
        ),
        (
            3,
            {"outlet": OutletFixedPressure(pressure=0.0)},
            {
                "f_w": 1.06666,
                "f_e": 1.0,
                "a_w": 1.06666,
                "a_e": 0.0,
                "a_p": 1.0,
                "source": 0.375,
                "d": 0.15,
            },
        ),
    ],
    ids=["inlet_velocity_node_1", "outlet_velocity_node_4"],
)
def test_each_momentum_boundary_matches_versteeg_example_6_2(
    node,
    assembler_kwargs,
    expected,
):
    geometry = Geometry.versteeg_example_6_2()
    fields = Field.versteeg_example_6_2_initial_guess()
    coeffs = MomentumAssembler(
        geometry=geometry,
        density=1.0,
        scheme=Upwind(),
        **assembler_kwargs,
    ).assemble(fields)

    np.testing.assert_allclose(coeffs.f_w[node], expected["f_w"], rtol=1e-5)
    np.testing.assert_allclose(coeffs.f_e[node], expected["f_e"], rtol=1e-5)
    np.testing.assert_allclose(coeffs.a_w[node], expected["a_w"], rtol=1e-5)
    np.testing.assert_allclose(coeffs.a_e[node], expected["a_e"], rtol=1e-5)
    np.testing.assert_allclose(coeffs.a_p[node], expected["a_p"], rtol=1e-5)
    np.testing.assert_allclose(coeffs.source[node], expected["source"], rtol=1e-5)
    np.testing.assert_allclose(coeffs.d[node], expected["d"], rtol=1e-4)


def test_full_momentum_assembly_matches_example_6_2_summary():
    geometry = Geometry.versteeg_example_6_2()
    fields = Field.versteeg_example_6_2_initial_guess()
    coeffs = MomentumAssembler(
        geometry=geometry,
        density=1.0,
        scheme=Upwind(),
        inlet=InletStagnationPressure(stagnation_pressure=10.0),
        outlet=OutletFixedPressure(pressure=0.0),
    ).assemble(fields)

    np.testing.assert_allclose(coeffs.a_p, [1.42087, 1.02857, 1.06666, 1.0], rtol=1e-5)
    np.testing.assert_allclose(coeffs.source, [3.125, 0.875, 0.625, 0.375], rtol=1e-5)
    np.testing.assert_allclose(coeffs.d, [0.31670, 0.34027, 0.23437, 0.15], rtol=1e-4)
