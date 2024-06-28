import pytest
import numpy as np
from VSM.Filament import HorshoeVortex


def test_initialization():
    LE_point_1 = np.array([0, 0, 0])
    TE_point_1 = np.array([1, 0, 0])
    LE_point_2 = np.array([0, 1, 0])
    TE_point_2 = np.array([1, 1, 0])
    aerodynamic_center_location = 0.25
    control_point_location = np.array([0.5, 0.5, 0.1])

    hv = HorshoeVortex(
        LE_point_1,
        TE_point_1,
        LE_point_2,
        TE_point_2,
        aerodynamic_center_location,
        control_point_location,
    )

    assert len(hv.filaments) == 3
    assert np.allclose(hv.bound_point_1, np.array([0.25, 0, 0]))
    assert np.allclose(hv.bound_point_2, np.array([0.25, 1, 0]))


def test_gamma_setter():

    LE_point_1 = np.array([0, 0, 0])
    TE_point_1 = np.array([1, 0, 0])
    LE_point_2 = np.array([0, 1, 0])
    TE_point_2 = np.array([1, 1, 0])
    aerodynamic_center_location = 0.25
    control_point_location = 0.75
    horshoe_vortex = HorshoeVortex(
        LE_point_1,
        TE_point_1,
        LE_point_2,
        TE_point_2,
        aerodynamic_center_location,
        control_point_location,
    )
    # Test setting a valid gamma value
    horshoe_vortex.gamma = 1.0
    assert horshoe_vortex.gamma == 1.0, "Gamma should be set to 1.0"

    # Test setting gamma to zero
    horshoe_vortex.gamma = 0.0
    assert horshoe_vortex.gamma == 0.0, "Gamma should be set to 0.0"

    # Test setting an invalid gamma value (negative)
    try:
        horshoe_vortex.gamma = -1.0
    except ValueError as e:
        assert (
            str(e) == "Gamma must be a non-negative value"
        ), "Should raise ValueError for negative gamma"
    else:
        assert False, "Expected ValueError for negative gamma"

    # Test updating gamma value
    horshoe_vortex.gamma = 2.5
    assert horshoe_vortex.gamma == 2.5, "Gamma should be updated to 2.5"


###########################################


# def test_calculate_velocity_induced_bound_2D_single_vortex():
#     # Test case for a single horseshoe vortex
#     LE_point = np.array([0, 0, 0])
#     TE_point = np.array([1, 0, 0])
#     hv = HorshoeVortex(
#         LE_point,
#         TE_point,
#         np.array([0, 1, 0]),
#         np.array([1, 1, 0]),
#         0.25,
#         np.array([0.5, 0.5, 0]),
#     )
#     hv.gamma = 2.0  # As shown in Figure 7.1
#     control_point = np.array([80, -1, 0])  # Cp location from Figure 7.1

#     velocity = hv.calculate_velocity_induced_bound_2D(control_point)

#     # The expected velocity should be close to the analytical solution in Table 7.1
#     expected_velocity = np.array([0, 0.1061, 0])
#     assert np.allclose(velocity, expected_velocity, atol=1e-4)


# def test_calculate_velocity_induced_bound_2D_multiple_vortices():
#     # Test case for multiple horseshoe vortices
#     LE_point = np.array([0, 0, 0])
#     TE_point = np.array([1, 0, 0])
#     control_point = np.array([80, -1, 0])  # Cp location from Figure 7.1

#     # Create three horseshoe vortices with different strengths
#     hv1 = HorshoeVortex(
#         LE_point,
#         TE_point,
#         np.array([0, 1, 0]),
#         np.array([1, 1, 0]),
#         0.25,
#         control_point,
#     )
#     hv1.gamma = 2.0

#     hv2 = HorshoeVortex(
#         LE_point,
#         TE_point,
#         np.array([0, 2, 0]),
#         np.array([1, 2, 0]),
#         0.25,
#         control_point,
#     )
#     hv2.gamma = 10.0

#     hv3 = HorshoeVortex(
#         LE_point,
#         TE_point,
#         np.array([0, 3, 0]),
#         np.array([1, 3, 0]),
#         0.25,
#         control_point,
#     )
#     hv3.gamma = 5.0

#     # Calculate induced velocities
#     v1 = hv1.calculate_velocity_induced_bound_2D(control_point)
#     v2 = hv2.calculate_velocity_induced_bound_2D(control_point)
#     v3 = hv3.calculate_velocity_induced_bound_2D(control_point)

#     total_velocity = v1 + v2 + v3

#     # Compare with analytical results from Table 7.1
#     expected_total_y = 0.1061 - 1.5915 + 0.2653
#     assert np.isclose(total_velocity[1], expected_total_y, atol=1e-4)


# def test_calculate_velocity_induced_bound_2D_core_correction():
#     LE_point = np.array([0, 0, 0])
#     TE_point = np.array([1, 0, 0])
#     hv = HorshoeVortex(
#         LE_point,
#         TE_point,
#         np.array([0, 1, 0]),
#         np.array([1, 1, 0]),
#         0.25,
#         np.array([0.5, 0.5, 0]),
#     )
#     hv.gamma = 1.0

#     # Test point very close to the vortex filament
#     close_point = np.array([0.5, 0.001, 0])

#     velocity = hv.calculate_velocity_induced_bound_2D(close_point)

#     # The velocity should be finite due to core correction
#     assert np.all(np.isfinite(velocity))
#     assert np.linalg.norm(velocity) > 0


# def test_calculate_velocity_induced_bound_2D_biot_savart():
#     # Test if the function follows the Biot-Savart law for a simple case
#     LE_point = np.array([0, 0, 0])
#     TE_point = np.array([1, 0, 0])
#     hv = HorshoeVortex(
#         LE_point,
#         TE_point,
#         np.array([0, 1, 0]),
#         np.array([1, 1, 0]),
#         0.25,
#         np.array([0.5, 0.5, 0]),
#     )
#     hv.gamma = 1.0

#     # Point above the middle of the vortex
#     test_point = np.array([0.5, 0, 1])

#     velocity = hv.calculate_velocity_induced_bound_2D(test_point)

#     # The velocity should be in the y-direction only
#     assert np.isclose(velocity[0], 0, atol=1e-6)
#     assert velocity[1] > 0
#     assert np.isclose(velocity[2], 0, atol=1e-6)

#     # Magnitude should follow the Biot-Savart law
#     r = 1.0  # Distance from the filament
#     expected_magnitude = hv.gamma / (2 * np.pi * r)
#     assert np.isclose(np.linalg.norm(velocity), expected_magnitude, rtol=1e-2)


###########################################


###########################################


###########################################


###########################################


###########################################


###########################################


# def test_calculate_velocity_induced_bound_2D():
#     hv = HorshoeVortex(
#         np.zeros(3),
#         np.ones(3),
#         np.array([0, 1, 0]),
#         np.array([1, 1, 0]),
#         0.25,
#         np.array([0.5, 0.5, 0.1]),
#     )
#     hv.gamma = 1.0
#     control_point = np.array([0.5, 0.5, 0.1])
#     velocity = hv.calculate_velocity_induced_bound_2D(control_point)
#     assert velocity.shape == (3,)
#     assert np.allclose(velocity, np.array([0, 0, 0]))


# def test_update_filaments_for_wake():
#     hv = HorshoeVortex(
#         np.zeros(3),
#         np.ones(3),
#         np.array([0, 1, 0]),
#         np.array([1, 1, 0]),
#         0.25,
#         np.array([0.5, 0.5, 0.1]),
#     )
#     point1 = np.array([0, 0, 0])
#     point2 = np.array([1, 1, 1])
#     dir = np.array([0, 0, 1])

#     hv.update_filaments_for_wake(point1, point2, dir)
#     assert len(hv.filaments) == 5


# def test_calculate_filaments_for_plotting():
#     hv = HorshoeVortex(
#         np.zeros(3),
#         np.ones(3),
#         np.array([0, 1, 0]),
#         np.array([1, 1, 0]),
#         0.25,
#         np.array([0.5, 0.5, 0.1]),
#     )
#     filaments = hv.calculate_filaments_for_plotting()
#     assert len(filaments) == 3
#     assert all(len(f) == 3 for f in filaments)


# def test_bound_filament():
#     bf = BoundFilament(np.array([0, 0, 0]), np.array([1, 0, 0]))
#     point = np.array([0.5, 1, 0])
#     gamma = 1.0

#     velocity = bf.calculate_induced_velocity(point, gamma)
#     assert velocity.shape == (3,)


# def test_semi_infinite_filament():
#     sif = SemiInfiniteFilament(np.array([0, 0, 0]), np.array([1, 0, 0]), 1)
#     point = np.array([0.5, 1, 0])
#     gamma = 1.0

#     velocity = sif.calculate_induced_velocity(point, gamma)
#     assert velocity.shape == (3,)
