import numpy as np
import pytest

from simplecfd.assembly import MomentumAssembler, PressureCorrectionAssembler
from simplecfd.boundary import InletStagnationPressure, OutletFixedPressure
from simplecfd.coefficients import LinearSystem
from simplecfd.fields import Field
from simplecfd.geometry import Geometry
from simplecfd.linalg import tdma
from simplecfd.schemes import Upwind


def test_tdma_solves_known_tridiagonal_system():
    system = LinearSystem(
        lower=np.array([0.0, -1.0, -1.0]),
        diagonal=np.array([2.0, 2.0, 2.0]),
        upper=np.array([-1.0, -1.0, 0.0]),
        rhs=np.array([1.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(tdma(system), [1.0, 1.0, 1.0])


def test_tdma_does_not_mutate_input_system():
    system = LinearSystem(
        lower=np.array([0.0, -1.0, -1.0]),
        diagonal=np.array([2.0, 2.0, 2.0]),
        upper=np.array([-1.0, -1.0, 0.0]),
        rhs=np.array([1.0, 0.0, 1.0]),
    )
    original = tuple(array.copy() for array in (system.lower, system.diagonal, system.upper, system.rhs))

    tdma(system)

    for actual, expected in zip(
        (system.lower, system.diagonal, system.upper, system.rhs),
        original,
    ):
        np.testing.assert_allclose(actual, expected)


def test_tdma_rejects_inconsistent_shapes():
    system = LinearSystem(
        lower=np.array([0.0, -1.0]),
        diagonal=np.array([2.0, 2.0, 2.0]),
        upper=np.array([-1.0, -1.0, 0.0]),
        rhs=np.array([1.0, 0.0, 1.0]),
    )

    with pytest.raises(ValueError, match="same size"):
        tdma(system)


def test_tdma_solves_versteeg_example_6_2_pressure_correction():
    geometry = Geometry.versteeg_example_6_2()
    initial_fields = Field.versteeg_example_6_2_initial_guess()
    momentum = MomentumAssembler(
        geometry=geometry,
        density=1.0,
        scheme=Upwind(),
        inlet=InletStagnationPressure(stagnation_pressure=10.0),
        outlet=OutletFixedPressure(pressure=0.0),
    ).assemble(initial_fields)
    fields = Field(
        p=initial_fields.p,
        u=np.array([2.19935, 3.02289, 3.50087, 4.10926]),
        p_prime=np.zeros(geometry.n_pressure),
    )
    system = PressureCorrectionAssembler(geometry, density=1.0).assemble(fields, momentum)

    pressure_correction = tdma(system)

    np.testing.assert_allclose(
        pressure_correction,
        [0.0, 1.63935, 4.17461, 6.20805, 0.0],
        rtol=1e-4,
        atol=1e-5,
    )
