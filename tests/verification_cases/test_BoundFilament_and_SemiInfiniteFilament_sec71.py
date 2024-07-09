import numpy as np
from VSM.Filament import SemiInfiniteFilament


def test_combined_filaments():
    # Gamma values for the filaments
    gammas = [2, 10, 5]
    y = [-3, -1, 1, 3]
    Uinf = 1.0

    horseshoe1_filaments = [
        # BoundFilament(x1=np.array([0, y[0], 0]), x2=np.array([0, y[1], 0])),
        SemiInfiniteFilament(
            np.array([0, y[0], 0]),
            np.array([1, 0, 0]),
            Uinf,
            filament_direction=1,
        ),
        SemiInfiniteFilament(
            np.array([0, y[1], 0]),
            np.array([1, 0, 0]),
            Uinf,
            filament_direction=-1,
        ),
    ]
    horseshoe2_filaments = [
        # BoundFilament(x1=np.array([0, y[1], 0]), x2=np.array([0, y[2], 0])),
        SemiInfiniteFilament(
            np.array([0, y[1], 0]),
            np.array([1, 0, 0]),
            Uinf,
            filament_direction=1,
        ),
        SemiInfiniteFilament(
            np.array([0, y[2], 0]),
            np.array([1, 0, 0]),
            Uinf,
            filament_direction=-1,
        ),
    ]
    horseshoe3_filaments = [
        # BoundFilament(x1=np.array([0, y[2], 0]), x2=np.array([0, y[3], 0])),
        SemiInfiniteFilament(
            np.array([0, y[2], 0]),
            np.array([1, 0, 0]),
            Uinf,
            filament_direction=1,
        ),
        SemiInfiniteFilament(
            np.array([0, y[3], 0]),
            np.array([1, 0, 0]),
            Uinf,
            filament_direction=-1,
        ),
    ]

    horseshoes = [horseshoe1_filaments, horseshoe2_filaments, horseshoe3_filaments]
    # Control point
    control_point = np.array([0, 0, 0])

    # Analytical solutions for the combined induced velocities
    # analytical_solutions = {
    #     5: np.array([0, 0, 0.2653]),
    #     10: np.array([0, 0, -1.5915]),
    #     2: np.array([0, 0, 0.1061]),
    # }

    solution = [0, 0, 0]
    for i in range(3):
        velocity_induced = 0
        for filament in horseshoes[i]:
            velocity_induced += filament.calculate_induced_velocity(
                control_point, gammas[i], core_radius_fraction=0.01
            )[2]

        solution[i] = velocity_induced

    # Assert
    assert np.allclose(solution, [0.1061, -1.5915, 0.2653], atol=1e-4)
