import numpy as np
import pytest

from simplecfd.fields import Field
from simplecfd.geometry import Geometry


def test_geometry_from_pressure_areas_averages_velocity_areas():
    geometry = Geometry.from_pressure_areas(
        pressure_areas=np.array([0.5, 0.4, 0.3]),
        dx=0.25,
    )

    np.testing.assert_allclose(geometry.velocity_areas, np.array([0.45, 0.35]))
    assert geometry.n_pressure == 3
    assert geometry.n_velocity == 2
    np.testing.assert_allclose(geometry.dx_values, [0.25, 0.25])
    np.testing.assert_allclose(geometry.pressure_positions, [0.0, 0.25, 0.5])
    np.testing.assert_allclose(geometry.velocity_positions, [0.125, 0.375])
    assert geometry.length == 0.5


def test_geometry_accepts_nonuniform_pressure_node_spacing():
    geometry = Geometry.from_pressure_areas(
        pressure_areas=np.array([1.0, 0.8, 0.6, 0.4]),
        dx=np.array([0.1, 0.2, 0.4]),
    )

    np.testing.assert_allclose(geometry.dx_values, [0.1, 0.2, 0.4])
    np.testing.assert_allclose(geometry.pressure_positions, [0.0, 0.1, 0.3, 0.7])
    np.testing.assert_allclose(geometry.velocity_positions, [0.05, 0.2, 0.5])
    np.testing.assert_allclose(geometry.west_velocity_spacing(1), 0.15)
    np.testing.assert_allclose(geometry.east_velocity_spacing(1), 0.3)
    np.testing.assert_allclose(geometry.velocity_control_volume_width(2), 0.4)
    np.testing.assert_allclose(geometry.length, 0.7)


def test_geometry_rejects_non_staggered_sizes():
    with pytest.raises(ValueError, match="n_pressure - 1"):
        Geometry(
            pressure_areas=np.array([0.5, 0.4, 0.3]),
            velocity_areas=np.array([0.45, 0.35, 0.25]),
            dx=0.25,
        )


def test_geometry_rejects_invalid_nonuniform_spacing():
    with pytest.raises(ValueError, match="n_pressure - 1"):
        Geometry.from_pressure_areas(
            pressure_areas=np.array([1.0, 0.8, 0.6]),
            dx=np.array([0.1, 0.2, 0.3]),
        )

    with pytest.raises(ValueError, match="dx must be positive"):
        Geometry.from_pressure_areas(
            pressure_areas=np.array([1.0, 0.8, 0.6]),
            dx=np.array([0.1, 0.0]),
        )


def test_geometry_has_versteeg_example_6_2_data():
    geometry = Geometry.versteeg_example_6_2()

    np.testing.assert_allclose(geometry.pressure_areas, [0.5, 0.4, 0.3, 0.2, 0.1])
    np.testing.assert_allclose(geometry.velocity_areas, [0.45, 0.35, 0.25, 0.15])
    assert geometry.dx == 0.5
    np.testing.assert_allclose(geometry.dx_values, [0.5, 0.5, 0.5, 0.5])
    assert geometry.length == 2.0


def test_field_from_geometry_has_consistent_shapes():
    geometry = Geometry.from_pressure_areas(
        pressure_areas=np.array([0.5, 0.4, 0.3]),
        dx=0.25,
    )
    field = Field.from_geometry(geometry, pressure=10.0, velocity=2.0)

    np.testing.assert_allclose(field.p, np.array([10.0, 10.0, 10.0]))
    np.testing.assert_allclose(field.u, np.array([2.0, 2.0]))
    np.testing.assert_allclose(field.p_prime, np.zeros(3))
    field.validate_against(geometry)


def test_field_rejects_inconsistent_staggered_velocity_size():
    with pytest.raises(ValueError, match="n_pressure - 1"):
        Field(
            p=np.array([1.0, 2.0, 3.0]),
            u=np.array([1.0, 2.0, 3.0]),
            p_prime=np.zeros(3),
        )


def test_field_has_versteeg_example_6_2_initial_guess():
    geometry = Geometry.versteeg_example_6_2()
    field = Field.versteeg_example_6_2_initial_guess()

    field.validate_against(geometry)
    np.testing.assert_allclose(field.p, [10.0, 7.5, 5.0, 2.5, 0.0])
    np.testing.assert_allclose(
        field.u,
        [2.2222222222, 2.8571428571, 4.0, 6.6666666667],
        rtol=1e-10,
    )
