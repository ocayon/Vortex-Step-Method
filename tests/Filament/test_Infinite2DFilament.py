import pytest
import numpy as np
from VSM.Filament import Infinite2DFilament


def test_calculate_induced_velocity():

    A = np.array([0, 0, 0])
    B = np.array([1, 0, 0])
    point = np.array([0.5, 1, 0])
    r0 = B - A
    r3 = point - A

    # From OlD CODES: def velocity_induced_bound_2D(ringvec):
    # Analytically calculate the induced velocity
    cross = [
        r0[1] * r3[2] - r0[2] * r3[1],
        r0[2] * r3[0] - r0[0] * r3[2],
        r0[0] * r3[1] - r0[1] * r3[0],
    ]

    ind_vel = (
        cross
        / (cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
        / 2
        / np.pi
        * np.linalg.norm(r0)
    )

    # Use the function to calculate the induced velocity
    filament = Infinite2DFilament(A, B)
    induced_velocity_calculated = filament.calculate_induced_velocity(point)

    np.testing.assert_almost_equal(induced_velocity_calculated, ind_vel, decimal=6)


@pytest.mark.parametrize(
    "point",
    [
        [0, 0, 0],  # Start point
        [1, 0, 0],  # End point
        [0.5, 0, 0],  # Middle point
        [2, 0, 0],  # Point on the line beyond end point
        [-1, 0, 0],  # Point on the line before start point
    ],
)
def test_point_on_filament_line(point):
    """Test points on the filament line."""
    filament = Infinite2DFilament([0, 0, 0], [1, 0, 0])
    induced_velocity = filament.calculate_induced_velocity(point)
    assert np.allclose(induced_velocity, [0, 0, 0], atol=1e-10)
    assert not np.isnan(induced_velocity).any()


def test_point_far_from_filament():
    """Test with a point far from the filament."""
    filament = Infinite2DFilament([0, 0, 0], [1, 0, 0])
    control_point = [0.5, 1e10, 0]
    induced_velocity = filament.calculate_induced_velocity(control_point)
    assert not np.isnan(induced_velocity).any()
    assert np.allclose(
        induced_velocity, [0, 0, 0], atol=1e-8
    )  # Velocity should be very small but not zero


def test_different_gamma_values():
    """Test with different gamma values to ensure linear scaling."""
    filament = Infinite2DFilament([0, 0, 0], [1, 0, 0])
    control_point = [0.5, 1, 0]
    v1 = filament.calculate_induced_velocity(control_point, gamma=1.0)
    v2 = filament.calculate_induced_velocity(control_point, gamma=2.0)
    v4 = filament.calculate_induced_velocity(control_point, gamma=4.0)
    assert np.allclose(v4, 2 * v2)
    assert np.allclose(v4, 4 * v1)


def test_symmetry():
    """Test symmetry of induced velocity for symmetric points."""
    filament = Infinite2DFilament([-1, 0, 0], [1, 0, 0])
    v1 = filament.calculate_induced_velocity([0, 1, 0])
    v2 = filament.calculate_induced_velocity([0, -1, 0])
    assert np.allclose(v1, -v2)


def test_velocity_decay():
    """Test that velocity magnitude decays with distance from the filament."""
    filament = Infinite2DFilament([0, 0, 0], [1, 0, 0])
    v1 = filament.calculate_induced_velocity([0.5, 1, 0])
    v2 = filament.calculate_induced_velocity([0.5, 2, 0])
    v3 = filament.calculate_induced_velocity([0.5, 4, 0])
    assert np.linalg.norm(v1) > np.linalg.norm(v2) > np.linalg.norm(v3)
    assert np.allclose(np.linalg.norm(v1) / np.linalg.norm(v2), 2)
    assert np.allclose(np.linalg.norm(v1) / np.linalg.norm(v3), 4)


def test_invariance_to_filament_length():
    """Test that the induced velocity is independent of the filament's length."""
    filament1 = Infinite2DFilament([0, 0, 0], [1, 0, 0])
    filament2 = Infinite2DFilament([0, 0, 0], [10, 0, 0])
    control_point = [0.5, 1, 0]
    v1 = filament1.calculate_induced_velocity(control_point)
    v2 = filament2.calculate_induced_velocity(control_point)
    assert np.allclose(v1, v2)


def test_perpendicular_velocity():
    """Test that the induced velocity is perpendicular to the vector from the filament to the point."""
    filament = Infinite2DFilament([0, 0, 0], [1, 0, 0])
    control_point = [0.5, 1, 0]
    induced_velocity = filament.calculate_induced_velocity(control_point)
    vector_to_point = np.array([0, 1, 0])  # Perpendicular vector from filament to point
    assert np.isclose(np.dot(induced_velocity, vector_to_point), 0, atol=1e-10)


def test_around_core_radius():
    """Test with points around the core radius to the filament to check handling of near-singularities."""
    filament = Infinite2DFilament([0, 0, 0], [1, 0, 0])
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

    # Check that the y component is zero (or very close to zero)
    assert np.isclose(induced_velocity1[1], 0, atol=1e-10)
    assert np.isclose(induced_velocity2[1], 0, atol=1e-10)
    assert np.isclose(induced_velocity3[1], 0, atol=1e-10)

    # Check that the velocities are not exactly zero
    assert not np.allclose(induced_velocity1, [0, 0, 0], atol=1e-10)
    assert not np.allclose(induced_velocity2, [0, 0, 0], atol=1e-10)
    assert not np.allclose(induced_velocity3, [0, 0, 0], atol=1e-10)

    # Check that the magnitude of velocity is max at the core radius
    assert np.linalg.norm(induced_velocity2) > np.linalg.norm(induced_velocity1)
    assert np.linalg.norm(induced_velocity2) > np.linalg.norm(induced_velocity3)

    # Check for continuity around the core radius
    assert np.allclose(induced_velocity1, induced_velocity2, rtol=1e-3)
    assert np.allclose(induced_velocity2, induced_velocity3, rtol=1e-3)

    # Optional: Check for symmetry if we flip the y-coordinate
    induced_velocity_neg = filament.calculate_induced_velocity(
        [0.5, -core_radius_fraction, 0]
    )
    assert np.allclose(induced_velocity2, -induced_velocity_neg)
