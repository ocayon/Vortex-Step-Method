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
