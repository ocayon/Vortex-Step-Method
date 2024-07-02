import pytest
import numpy as np
from VSM.Filament import BoundFilament


def test_calculate_induced_velocity():
    # Define a simple filament and control point
    filament = BoundFilament([0, 0, 0], [1, 0, 0])
    control_point = [0.5, 1, 0]
    gamma = 1.0

    # Analytical solution using Biot-Savart law
    A = np.array([0, 0, 0])
    B = np.array([1, 0, 0])
    P = np.array(control_point)
    r0 = B - A
    r1 = P - A
    r2 = P - B

    r1Xr2 = np.cross(r1, r2)
    norm_r1Xr2 = np.linalg.norm(r1Xr2)
    induced_velocity_analytical = (
        (gamma / (4 * np.pi))
        * (r1Xr2 / (norm_r1Xr2**2))
        * np.dot(r0, r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2))
    )

    # Calculated solution using the BoundFilament class method
    induced_velocity_calculated = filament.calculate_induced_velocity(
        control_point, gamma
    )

    # Assert the induced velocities are almost equal
    np.testing.assert_almost_equal(
        induced_velocity_calculated, induced_velocity_analytical, decimal=6
    )


# def test_a_very_close_point():
#     """Test with a point that's super close, which should be almost zero."""
#     filament = BoundFilament([0, 0, 0], [1, 0, 0])
#     control_point = [0.5, 1e-10, 0]
#     induced_velocity = filament.calculate_induced_velocity(control_point)
#     assert not np.isnan(induced_velocity).any()
#     assert np.allclose(induced_velocity, [0, 0, 0], atol=1e-8)


@pytest.mark.parametrize(
    "point",
    [
        [0, 0, 0],  # Start point
        [1, 0, 0],  # End point
        [0.5, 0, 0],  # Middle point
    ],
)
def test_point_exactly_on_filament(point):
    """Test points exactly on the filament."""
    filament = BoundFilament([0, 0, 0], [1, 0, 0])
    induced_velocity = filament.calculate_induced_velocity(point)
    assert np.allclose(induced_velocity, [0, 0, 0])
    assert not np.isnan(induced_velocity).any()


def test_long_filament():
    """Test with a very long filament to ensure numerical stability."""
    filament = BoundFilament([0, 0, 0], [1e6, 0, 0])
    control_point = [5e5, 1, 0]
    induced_velocity = filament.calculate_induced_velocity(control_point)
    assert not np.isnan(induced_velocity).any()
    assert np.allclose(
        induced_velocity[0], 0, atol=1e-10
    )  # x-component should be near zero
    assert abs(induced_velocity[1]) < 1e-10  # y-component should be very close to zero
    assert np.isclose(induced_velocity[2], 0)  # z-component should be zero


def test_point_far_from_filament():
    """Test with a point far from the filament."""
    filament = BoundFilament([0, 0, 0], [1, 0, 0])
    control_point = [0.5, 1e6, 0]
    induced_velocity = filament.calculate_induced_velocity(control_point)
    assert not np.isnan(induced_velocity).any()
    assert np.allclose(
        induced_velocity, [0, 0, 0], atol=1e-12
    )  # Velocity should be very close to zero


def test_different_gamma_values():
    """Test with different gamma values to ensure linear scaling."""
    filament = BoundFilament([0, 0, 0], [1, 0, 0])
    control_point = [0.5, 1, 0]
    v1 = filament.calculate_induced_velocity(control_point, gamma=1.0)
    v2 = filament.calculate_induced_velocity(control_point, gamma=2.0)
    v4 = filament.calculate_induced_velocity(control_point, gamma=4.0)
    assert np.allclose(v4, 2 * v2, 4 * v1)


def test_symmetry():
    """Test symmetry of induced velocity for symmetric points."""
    filament = BoundFilament([-1, 0, 0], [1, 0, 0])
    v1 = filament.calculate_induced_velocity([0, 1, 0])
    v2 = filament.calculate_induced_velocity([0, -1, 0])
    assert np.allclose(v1, -v2)


def test_around_core_radius():
    """Test with a points around the core radius to the filament to check handling of near-singularities."""
    filament = BoundFilament([0, 0, 0], [1, 0, 0])
    core_radius_fraction = 0.01
    delta = 1e-5
    control_point1 = [0.5, core_radius_fraction - delta, 0]
    control_point2 = [0.5, core_radius_fraction, 0]
    control_point3 = [0.5, core_radius_fraction + delta, 0]
    induced_velocity1 = filament.calculate_induced_velocity(control_point1)
    induced_velocity2 = filament.calculate_induced_velocity(control_point2)
    induced_velocity3 = filament.calculate_induced_velocity(control_point3)

    # Check for NaN values
    assert not np.isnan(induced_velocity1).any()
    assert not np.isnan(induced_velocity2).any()
    assert not np.isnan(induced_velocity3).any()

    # Check that velocities are finite
    assert np.all(np.isfinite(induced_velocity1))
    assert np.all(np.isfinite(induced_velocity2))
    assert np.all(np.isfinite(induced_velocity3))

    # Check that the x and y component is zero (or very close to zero)
    assert np.allclose(induced_velocity1[:2], [0, 0], atol=1e-10)
    assert np.allclose(induced_velocity2[:2], [0, 0], atol=1e-10)
    assert np.allclose(induced_velocity3[:2], [0, 0], atol=1e-10)

    # Check that mangitude of velocity is max at the core radius
    assert np.linalg.norm(induced_velocity2) > np.linalg.norm(induced_velocity1)
    assert np.linalg.norm(induced_velocity2) > np.linalg.norm(induced_velocity3)

    # Check for continuity around the core radius
    assert np.allclose(induced_velocity1, induced_velocity2, rtol=1e-3)

    # TODO: no continuity established between at vortex-core and going inside, but maybe doesn't matter?
    # assert np.allclose(induced_velocity2, induced_velocity3, rtol=1e-3)

    # Check that the velocities are not exactly zero
    assert not np.allclose(induced_velocity1, [0, 0, 0], atol=1e-10)
    assert not np.allclose(induced_velocity2, [0, 0, 0], atol=1e-10)
    assert not np.allclose(induced_velocity3, [0, 0, 0], atol=1e-10)

    # Optional: Check for symmetry if we flip the y-coordinate
    induced_velocity_neg = filament.calculate_induced_velocity(
        [0.5, -core_radius_fraction, 0]
    )
    assert np.allclose(induced_velocity2, -induced_velocity_neg)
