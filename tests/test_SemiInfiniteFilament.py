import numpy as np
from VSM.HorshoeVortex import SemiInfiniteFilament


def test_calculate_induced_velocity_semi_infinite():
    # Define a simple filament and control point
    x1 = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    filament_direction = np.array([1, 0, 0])
    semi_infinite_filament = SemiInfiniteFilament(x1, direction, filament_direction)

    control_point = np.array([1, 1, 0])
    gamma = 1.0
    Uinf = 1.0

    # Analytical solution using Biot-Savart law
    r1 = control_point - x1
    r1_cross_direction = np.cross(r1, direction)
    r_perp = np.dot(r1, direction) * direction
    epsilon = np.sqrt(
        4
        * semi_infinite_filament.alpha0
        * semi_infinite_filament.nu
        * np.linalg.norm(r_perp)
        / Uinf
    )

    if np.linalg.norm(r1_cross_direction) / np.linalg.norm(direction) > epsilon:
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

    # Calculated solution using the SemiInfiniteFilament class method
    induced_velocity_calculated = semi_infinite_filament.calculate_induced_velocity(
        control_point, gamma, Uinf
    )

    # Assert the induced velocities are almost equal
    np.testing.assert_almost_equal(
        induced_velocity_calculated, induced_velocity_analytical, decimal=6
    )
