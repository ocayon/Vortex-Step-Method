import pytest
import numpy as np
from VSM.Filament import SemiInfiniteFilament


def test_calculate_induced_velocity_semi_infinite():
    # Define a simple filament and control point
    x1 = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    filament_direction = 1
    Uinf = 1.0
    semi_infinite_filament = SemiInfiniteFilament(
        x1, direction, Uinf, filament_direction
    )

    control_point = np.array([0.5, 0.5, 2])
    gamma = 5
    induced_velocity_calc = semi_infinite_filament.calculate_induced_velocity(
        control_point, gamma
    )
    print(f"induced_velocity_calc: {induced_velocity_calc}")

    # Analytical solution using Biot-Savart law
    r1 = control_point - x1
    r1_cross_direction = np.cross(r1, direction)
    r_perp = np.dot(r1, direction) * direction
    alpha0 = 1.25643
    nu = 1.48e-5
    epsilon = np.sqrt(4 * alpha0 * nu * np.linalg.norm(r_perp) / Uinf)

    if np.linalg.norm(r1_cross_direction) > epsilon:
        K = (
            gamma
            / (4 * np.pi * np.linalg.norm(r1_cross_direction) ** 2)
            * (1 + np.dot(r1, direction) / np.linalg.norm(r1))
        )
        induced_velocity_analytical = K * r1_cross_direction * filament_direction
    else:
        r1_proj = np.dot(r1, direction) * direction + epsilon * (
            r1 / np.linalg.norm(r1) - direction
        ) / np.linalg.norm(r1 / np.linalg.norm(r1) - direction)
        r1_cross_direction_proj = np.cross(r1_proj, direction)
        K_proj = (
            gamma
            / (4 * np.pi * np.linalg.norm(r1_cross_direction_proj) ** 2)
            * (1 + np.dot(r1_proj, direction) / np.linalg.norm(r1_proj))
        )
        induced_velocity_analytical = (
            K_proj * r1_cross_direction_proj * filament_direction
        )

    # Assert the induced velocities are almost equal
    np.testing.assert_almost_equal(
        induced_velocity_calc, induced_velocity_analytical, decimal=6
    )


def test_a_very_close_point():
    """Test with a point that's super close, which should be almost zero."""
    x1 = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    filament_direction = 1
    Uinf = 1.0
    semi_infinite_filament = SemiInfiniteFilament(
        x1, direction, Uinf, filament_direction
    )
    point = [0.5, 1e-10, 0]
    induced_velocity = semi_infinite_filament.calculate_induced_velocity(point)
    assert not np.isnan(induced_velocity).any()
    assert not np.isinf(induced_velocity).any()


@pytest.mark.parametrize(
    "point",
    [
        [0, 0, 0],  # Start point
        [0.5, 0, 0],  # Along filament
        [5, 0, 0],  # Further along filament
    ],
)
def test_point_exactly_on_filament(point):
    x1 = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    filament_direction = 1
    Uinf = 10
    semi_infinite_filament = SemiInfiniteFilament(
        x1, direction, Uinf, filament_direction
    )

    point = np.array([0.5, 0, 0])  # Point on the filament
    gamma = 1.0

    velocity = semi_infinite_filament.calculate_induced_velocity(point, gamma)
    assert np.allclose(velocity, np.zeros(3), atol=1e-6)


def test_point_far_from_filament():
    x1 = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    filament_direction = 1
    Uinf = 1.0
    semi_infinite_filament = SemiInfiniteFilament(
        x1, direction, Uinf, filament_direction
    )

    far_point = np.array([0, 1e6, 0])
    gamma = 1.0

    induced_velocity = semi_infinite_filament.calculate_induced_velocity(
        far_point, gamma
    )

    # Velocity should decrease with distance^2, due to Biot-Savart law
    assert not np.isnan(induced_velocity).any()


def test_different_gamma_values():
    """Test with different gamma values to ensure linear scaling."""
    x1 = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    filament_direction = 1
    Uinf = 1.0
    semi_infinite_filament = SemiInfiniteFilament(
        x1, direction, Uinf, filament_direction
    )
    control_point = [0.5, 1, 0]
    v1 = semi_infinite_filament.calculate_induced_velocity(control_point, gamma=1.0)
    v2 = semi_infinite_filament.calculate_induced_velocity(control_point, gamma=2.0)
    v4 = semi_infinite_filament.calculate_induced_velocity(control_point, gamma=4.0)
    assert np.allclose(v4, 2 * v2, 4 * v1)


def test_symmetry():
    """Test symmetry of induced velocity for symmetric points."""
    x1 = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    filament_direction = 1
    Uinf = 1.0
    semi_infinite_filament = SemiInfiniteFilament(
        x1, direction, Uinf, filament_direction
    )
    vel_point_pos_y = semi_infinite_filament.calculate_induced_velocity([0, 1, 0])
    vel_point_neg_y = semi_infinite_filament.calculate_induced_velocity([0, -1, 0])

    semi_infinite_filament = SemiInfiniteFilament(
        x1, direction, Uinf, -filament_direction
    )
    vel_point_pos_y_neg_dir = semi_infinite_filament.calculate_induced_velocity(
        [0, 1, 0]
    )
    assert np.allclose(vel_point_pos_y, -vel_point_neg_y)
    assert np.allclose(vel_point_pos_y, -vel_point_pos_y_neg_dir)


def test_uinf_effect():
    x1 = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    filament_direction = 1

    control_point = np.array([0, 2, 0])  # Point away from filament
    gamma = 1.0
    Uinf1 = 1.0
    Uinf2 = 10.0

    semi_infinite_filament1 = SemiInfiniteFilament(
        x1, direction, Uinf1, filament_direction
    )
    semi_infinite_filament2 = SemiInfiniteFilament(
        x1, direction, Uinf2, filament_direction
    )

    velocity1 = semi_infinite_filament1.calculate_induced_velocity(control_point, gamma)
    velocity2 = semi_infinite_filament2.calculate_induced_velocity(control_point, gamma)

    # Velocity should be equal for the same gamma
    assert np.allclose(np.linalg.norm(velocity1), np.linalg.norm(velocity2))


def test_around_core_radius():
    """Test with a points around the core radius to the filament to check handling of near-singularities."""
    x1 = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    filament_direction = 1
    Uinf = 1.0
    semi_infinite_filament = SemiInfiniteFilament(
        x1, direction, Uinf, filament_direction
    )
    core_radius_fraction = 5
    delta = 1e-5
    control_point1 = [0.5, core_radius_fraction - delta, 0]
    control_point2 = [0.5, core_radius_fraction, 0]
    control_point3 = [0.5, core_radius_fraction + delta, 0]
    induced_velocity1 = semi_infinite_filament.calculate_induced_velocity(
        control_point1
    )
    induced_velocity2 = semi_infinite_filament.calculate_induced_velocity(
        control_point2
    )
    induced_velocity3 = semi_infinite_filament.calculate_induced_velocity(
        control_point3
    )

    print(f"induced_velocity1: {induced_velocity1}")
    print(f"induced_velocity2: {induced_velocity2}")
    print(f"induced_velocity3: {induced_velocity3}")

    # # Check for NaN values
    # assert not np.isnan(induced_velocity1).any()
    # assert not np.isnan(induced_velocity2).any()
    # assert not np.isnan(induced_velocity3).any()

    # # Check that velocities are finite
    # assert np.all(np.isfinite(induced_velocity1))
    # assert np.all(np.isfinite(induced_velocity2))
    # assert np.all(np.isfinite(induced_velocity3))

    # # Check that the x and y component is zero (or very close to zero)
    # assert np.allclose(induced_velocity1[:2], [0, 0], atol=1e-10)
    # assert np.allclose(induced_velocity2[:2], [0, 0], atol=1e-10)
    # assert np.allclose(induced_velocity3[:2], [0, 0], atol=1e-10)

    # # Check that mangitude of velocity is max at the core radius
    # assert np.linalg.norm(induced_velocity2) > np.linalg.norm(induced_velocity1)
