import numpy as np
from VSM.HorshoeVortex import BoundFilament, SemiInfiniteFilament


def test_combined_filaments():
    # Setup: Define bound filaments and semi-infinite filaments
    bound_filaments = [
        BoundFilament(x1=np.array([3, 0, 0]), x2=np.array([1, 0, 0])),  # Gamma = 5
        BoundFilament(x1=np.array([-1, 0, 0]), x2=np.array([1, 0, 0])),  # Gamma = 10
        BoundFilament(x1=np.array([-3, 0, 0]), x2=np.array([-1, 0, 0])),  # Gamma = 2
    ]

    semi_infinite_filaments = [
        SemiInfiniteFilament(
            np.array([1, 0, 0]), np.array([1, 0, 0]), filament_direction=1
        ),  # Gamma = 5
        SemiInfiniteFilament(
            np.array([1, 0, 0]), np.array([1, 0, 0]), filament_direction=-1
        ),  # Gamma = 10
        SemiInfiniteFilament(
            np.array([-1, 0, 0]), np.array([1, 0, 0]), filament_direction=1
        ),  # Gamma = 2
    ]

    # Gamma values for the filaments
    gammas = [5, 10, 2]

    # Control point
    control_point = np.array([80, -1, 0])

    # Analytical solutions for the combined induced velocities
    analytical_solutions = {
        5: np.array([0.2653, 0, 0]),
        10: np.array([-1.5915, 0, 0]),
        2: np.array([0.1061, 0, 0]),
    }

    # Calculate total induced velocity at the control point
    total_velocity = np.zeros(3)

    for i, bound_filament in enumerate(bound_filaments):
        gamma = gammas[i]
        total_velocity += bound_filament.calculate_induced_velocity(
            control_point, gamma
        )

    for i, semi_infinite_filament in enumerate(semi_infinite_filaments):
        gamma = gammas[i]
        total_velocity += semi_infinite_filament.calculate_induced_velocity(
            control_point, gamma
        )

    # Compare calculated velocity with analytical solutions
    for gamma_value in gammas:
        np.testing.assert_almost_equal(
            total_velocity,
            analytical_solutions[gamma_value],
            decimal=4,
            err_msg=f"Failed for gamma = {gamma_value}",
        )
