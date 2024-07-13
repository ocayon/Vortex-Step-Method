import numpy as np
import logging
from VSM.Filament import SemiInfiniteFilament, BoundFilament


def test_combined_filaments():
    # Gamma values for the filaments
    Uinf = np.array([1, 0, 0])
    va_norm = np.linalg.norm(Uinf)
    va_unit = Uinf / va_norm

    left_gamma2 = np.array([0, -1, 0])
    right_gamma2 = np.array([0, -3, 0])
    horseshoe1_filaments = [
        BoundFilament(x1=left_gamma2, x2=right_gamma2),
        SemiInfiniteFilament(
            np.array(left_gamma2),
            va_unit,
            va_norm,
            filament_direction=-1,
        ),
        SemiInfiniteFilament(
            np.array(right_gamma2),
            va_unit,
            va_norm,
            filament_direction=1,
        ),
    ]
    left_gamma10 = np.array([0, 1, 0])
    right_gamma10 = np.array([0, -1, 0])
    horseshoe2_filaments = [
        BoundFilament(x1=left_gamma10, x2=right_gamma10),
        SemiInfiniteFilament(
            np.array(left_gamma10),
            va_unit,
            va_norm,
            filament_direction=-1,
        ),
        SemiInfiniteFilament(
            np.array(right_gamma10),
            va_unit,
            va_norm,
            filament_direction=1,
        ),
    ]
    left_gamma5 = np.array([0, 3, 0])
    right_gamma5 = np.array([0, 1, 0])
    horseshoe3_filaments = [
        BoundFilament(x1=left_gamma5, x2=right_gamma5),
        SemiInfiniteFilament(
            np.array(left_gamma5),
            va_unit,
            va_norm,
            filament_direction=-1,
        ),
        SemiInfiniteFilament(
            np.array(right_gamma5),
            va_unit,
            va_norm,
            filament_direction=1,
        ),
    ]

    horseshoes = [horseshoe1_filaments, horseshoe2_filaments, horseshoe3_filaments]

    # The Test Case
    gammas = [2, 10, 5]
    evaluation_point = np.array([82, -1.1, 0])
    evaluation_point = np.array([0, 0, 0])

    # Analytical solutions for the combined induced velocities
    analytical_solutions = {
        2: np.array([0, 0, 0.1061]),
        10: np.array([0, 0, -1.5915]),
        5: np.array([0, 0, 0.2653]),
    }

    solution = [0, 0, 0]
    for i in range(3):
        velocity_induced = [0, 0, 0]
        gamma = gammas[i]
        for j, filament in enumerate(horseshoes[i]):
            logging.debug(f"j: {j},gamma: {gamma}, filament: {filament}")
            if j == 0:
                if evaluation_point[0] == filament.x1[0]:
                    tempvel = [0, 0, 0]
                else:
                    tempvel = filament.velocity_3D_bound_vortex(
                        evaluation_point, gamma, core_radius_fraction=1e-5
                    )
                logging.debug(f"BOUND tempvel: {tempvel}")
            if j == 1:
                tempvel = filament.velocity_3D_trailing_vortex_semiinfinite(
                    va_unit, evaluation_point, gamma, va_norm
                )
                logging.debug(f"SEMIINFINITE 1 tempvel: {tempvel}")
            elif j == 2:
                tempvel = filament.velocity_3D_trailing_vortex_semiinfinite(
                    va_unit, evaluation_point, gamma, va_norm
                )
                logging.debug(f"SEMIINFINITE 2 tempvel: {tempvel}")
            velocity_induced[0] += tempvel[0]
            velocity_induced[1] += tempvel[1]
            velocity_induced[2] += tempvel[2]

        solution[i] = velocity_induced
        logging.debug(f"i: {i}")
        logging.debug(f"Analytical Solution: {analytical_solutions[gammas[i]]}")
        logging.debug(f"Numerical Solution: {solution[i]}")

        # Assert
        assert np.allclose(solution[i], analytical_solutions[gammas[i]], atol=1e-4)
